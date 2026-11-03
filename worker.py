# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=1.0", "pyarrow>=15"]
# ///
"""Titan Learn - cloud media archiver (runs as an HF Job, not on a laptop, not on the Gateway).

Pipeline per asset, in order, with nothing skipped:

    PULL from VOA -> TEMP file -> VERIFY (magic MIME + declared size + SHA-256)
    -> UPLOAD to the dataset (one commit per batch) -> REMOTE VERIFY (lfs sha256 + size)
    -> WRITE STATE -> PURGE temp

State lives in the dataset, so any run resumes where the last one stopped and an asset
already marked verified_remote is never downloaded or uploaded again. Records are not
rewritten here: the join lives in `_worker/<kind>_refs.jsonl`, because replacing 100 MB of
shards to attach a few media refs would be wasteful and would churn the dataset.

    hf jobs uv worker.py --hardware cpu-basic --timeout 3600 --secret HF_TOKEN=... \
        -- --kind audio --batches 4 --batch-assets 400 --workers 8

    python worker.py --kind audio --batches 1 --dry-run     # same code, locally
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

REPO = os.environ.get("HF_LEARN_DATASET_REPO", "M-hv1/Titan-Learn")
WRITE_REVISION = os.environ.get("HF_LEARN_DATASET_REVISION", "staging")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
MAGIC = {
    "audio/mpeg": (b"ID3",),
    "video/mp4": (b"ftyp",),
    "application/pdf": (b"%PDF-",),
}
EXPECTED_FAMILY = {"audio": ("audio/mpeg",), "video": ("video/mp4",), "pdf": ("application/pdf",)}
MAX_ATTEMPTS = 3
BASE_VIDEO = re.compile(r"_(hq|mobile|\d+p)\.mp4$")
# Records are read at one immutable commit; the resume state is read from the branch,
# because state by definition moves and pinning it would make a resumed run stale.
RECORDS_REVISION = os.environ.get("HF_LEARN_DATASET_PIN", "staging")
STATE_REVISION = os.environ.get("HF_LEARN_STATE_REVISION", "staging")
# the live CDN first: the pre-2011 www.voanews.com paths are mostly dead and would only
# burn retry attempts if the queue were sorted by url
HOST_PRIORITY = {"voa-audio.voanews.eu": 0, "av.voanews.com": 1, "podcast.voanews.com": 2}


def log(message: str) -> None:
    # progress goes to stderr so stdout stays a single parseable JSON report
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def token() -> str:
    value = os.environ.get("HF_TOKEN") or ""
    if not value:
        cached = Path.home() / ".cache" / "huggingface" / "token"
        value = cached.read_text(encoding="utf-8").strip() if cached.exists() else ""
    if not value:
        raise SystemExit("no HF token: set HF_TOKEN or pass --secret HF_TOKEN")
    return value


def hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=token())


def read_file(api, path: str, required: bool = False, revision: str = "") -> bytes | None:
    """Read one file from the dataset, tolerating a first run where it does not exist yet."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

    try:
        local = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=path,
                                revision=revision or RECORDS_REVISION, token=api.token)
        return Path(local).read_bytes()
    except (EntryNotFoundError, HfHubHTTPError, FileNotFoundError):
        if required:
            raise SystemExit(f"required file missing from the dataset: {path}")
        return None


def load_state(api) -> dict:
    raw = read_file(api, "_worker/media_index.json", revision=STATE_REVISION)
    if not raw:
        return {"objects": {}, "assets": {}, "history": []}
    return json.loads(raw)


def load_refs(api) -> dict[str, list[dict]]:
    raw = read_file(api, "_worker/audio_refs.jsonl", revision=STATE_REVISION)
    refs: dict[str, list[dict]] = {}
    for line in (raw or b"").decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            refs[row["item_id"]] = row["media_refs"]
    return refs


def queue_from_shards(api, kind: str) -> list[tuple[str, str]]:
    """Build the work list from the dataset itself, so the corpus is the only source of truth."""
    manifest = json.loads(read_file(api, "manifest.json", required=True))
    wanted: dict[str, str] = {}
    for shard in manifest["shards"]:
        import pyarrow.parquet as pq

        table = pq.read_table(_local(api, shard["path"]), columns=["id", "content"])
        for row in table.to_pylist():
            content = row["content"]
            item_id = row["id"].split(":")[-1]
            if kind == "audio":
                url = (content.get("audio") or {}).get("url")
                if url:
                    wanted.setdefault(url, item_id)
            elif kind == "video":
                for url in content.get("video") or []:
                    if not BASE_VIDEO.search(url):
                        wanted.setdefault(url, item_id)
                        break
            elif kind == "pdf":
                for url in content.get("pdf") or []:
                    wanted.setdefault(url, item_id)
    def rank(url: str) -> tuple[int, str]:
        host = urllib.parse.urlsplit(url).netloc
        return (HOST_PRIORITY.get(host, 9), url)

    return sorted(((item_id, url) for url, item_id in wanted.items()), key=lambda row: rank(row[1]))


_CACHE_DIR: str | None = None


def _local(api, path: str) -> str:
    global _CACHE_DIR
    from huggingface_hub import hf_hub_download

    _CACHE_DIR = _CACHE_DIR or tempfile.mkdtemp(prefix="voa-shards-")
    return hf_hub_download(repo_id=REPO, repo_type="dataset", filename=path,
                           revision=RECORDS_REVISION, token=api.token, local_dir=_CACHE_DIR)


def sniff(head: bytes) -> str | None:
    for mime, signatures in MAGIC.items():
        for signature in signatures:
            if head.startswith(signature) or (signature == b"ftyp" and head[4:8] == signature):
                return mime
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    return None


def pull(url: str, directory: str) -> dict:
    """Download to temp and verify the bytes before anything is trusted."""
    name = hashlib.sha256(url.encode()).hexdigest()[:20] + (Path(urllib.parse.urlsplit(url).path).suffix or ".bin")
    target = Path(directory) / name
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as response:
            declared = int(response.headers.get("Content-Length") or 0)
            header_mime = (response.headers.get("Content-Type") or "").split(";")[0].strip()
            data = response.read()
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        return {"url": url, "status": "failed", "error": f"{type(error).__name__}: {error}"}
    if not data:
        return {"url": url, "status": "failed", "error": "empty body"}
    if declared and declared != len(data):
        return {"url": url, "status": "failed", "error": f"truncated {len(data)} of {declared}"}
    mime = sniff(data[:12])
    if not mime:
        return {"url": url, "status": "quarantined",
                "error": f"no known media signature (header {header_mime or 'none'})"}
    target.write_bytes(data)
    if target.stat().st_size != len(data):
        return {"url": url, "status": "failed", "error": "temp size mismatch"}
    return {"url": url, "local": str(target), "bytes": len(data), "mime": mime,
            "header_mime": header_mime, "sha256": hashlib.sha256(data).hexdigest(),
            "status": "verified_local"}


def object_path(digest: str, local: str) -> str:
    suffix = Path(local).suffix or ".bin"
    return f"media/{digest[0:2]}/{digest}{suffix}"


def remote_oids(api, paths: list[str]) -> dict[str, tuple[str, int]]:
    found: dict[str, tuple[str, int]] = {}
    for start in range(0, len(paths), 500):
        chunk = paths[start:start + 500]
        for entry in api.get_paths_info(REPO, chunk, expand=True, repo_type="dataset",
                                        revision=WRITE_REVISION):
            lfs = getattr(entry, "lfs", None)
            digest = (getattr(lfs, "sha256", "") or "").lower()
            if digest:
                found[entry.path] = (digest, int(getattr(lfs, "size", 0) or 0))
    return found


def done(key: str, index: dict) -> bool:
    row = index["assets"].get(key)
    if not row:
        return False
    if row["status"] == "verified_remote":
        return True
    return row["status"] in ("failed", "quarantined") and row.get("attempts", 1) >= MAX_ATTEMPTS


def video_gate_allowed(api) -> tuple[bool, str]:
    """Video waits for two things: a measured size sample, and audio finished + verified.

    Both are read from the dataset, so the ordering is enforced by evidence on the shared
    store rather than by a human remembering to wait.
    """
    raw = read_file(api, "reports/video_probe.json", revision=STATE_REVISION)
    if not raw:
        return False, "no reports/video_probe.json in the dataset"
    probe = json.loads(raw)
    ok_count = (probe.get("head_results") or {}).get("ok", 0)
    if ok_count < 30:
        return False, f"sample too small: {ok_count} successful HEADs"
    if probe.get("decision") != "video archiving may start":
        return False, f"sample says: {probe.get('decision')}"
    check = read_file(api, "reports/audio_verify.json", revision=STATE_REVISION)
    if not check:
        return False, "audio is not verified yet: no reports/audio_verify.json"
    verdict = json.loads(check)
    if not verdict.get("integrity_ok"):
        return False, ("audio verification found integrity problems: "
                       f"{verdict.get('indexed_missing_or_mismatch')} objects, "
                       f"{verdict.get('dangling_refs')} dangling refs")
    if not verdict.get("complete"):
        return False, (f"audio not finished: {verdict.get('audio_queue_remaining')} assets pending, "
                       f"{verdict.get('remote_orphans_not_indexed')} objects not yet accounted for")
    return True, (f"gate open on {ok_count} HEADs; audio complete and verified "
                  f"{verdict.get('indexed_ok')} objects / {verdict.get('bytes_gb')} GB")


class Stopped(Exception):
    """The runner is about to end this job; the checkpoint gets written before it does."""


def _on_signal(signum, frame):
    """A runner winding down still gets its checkpoint written before the process leaves."""
    print(f"signal {signum}: flushing checkpoint before exit", file=sys.stderr, flush=True)
    if STATE_HOOK and not STATE_HOOK.get("dry_run"):
        try:
            write_state(STATE_HOOK["api"], STATE_HOOK["index"], STATE_HOOK["refs"],
                        STATE_HOOK["kind"])
        except Exception as error:
            print(f"checkpoint flush failed: {error}", file=sys.stderr, flush=True)
    raise SystemExit(128 + signum)


STATE_HOOK: dict = {}


def run(kind: str, batches: int, batch_assets: int, workers: int, max_minutes: float,
        dry_run: bool, reserve_minutes: float = 10.0) -> dict:
    api = hf_api()
    if kind == "video":
        allowed, why = video_gate_allowed(api)
        if not allowed:
            return {"aborted": f"video gate closed: {why}"}
    index = load_state(api)
    refs = load_refs(api)
    queue = queue_from_shards(api, kind)
    STATE_HOOK.clear()
    STATE_HOOK.update({"api": api, "index": index, "refs": refs, "kind": kind, "dry_run": dry_run})
    log(f"{kind}: queue {len(queue)} assets, index has {len(index['objects'])} objects")
    started = time.time()
    history: list[dict] = []
    last_batch_seconds = 0.0
    for batch_number in range(1, batches + 1):
        # the margin is adaptive: a batch only starts if the measured length of the last
        # one still fits, plus a reserve, before this step's deadline
        if (time.time() - started) + last_batch_seconds + reserve_minutes * 60 > max_minutes * 60:
            history.append({"batch": batch_number, "skipped": "safety margin before the deadline",
                            "elapsed_minutes": round((time.time() - started) / 60, 1),
                            "last_batch_minutes": round(last_batch_seconds / 60, 1),
                            "deadline_minutes": max_minutes, "reserve_minutes": reserve_minutes})
            break
        pending = [(item_id, url) for item_id, url in queue
                   if not done(f"{item_id}:{kind}:{url}", index)]
        if not pending:
            history.append({"batch": batch_number, "drained": True, "assets": 0})
            break
        slice_ = pending[:batch_assets]
        batch_start = time.time()
        stage = tempfile.mkdtemp(prefix=f"voa-{kind}-")
        jobs: list[dict] = []
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda pair: pull(pair[1], stage), slice_))
        for (item_id, url), result in zip(slice_, results):
            key = f"{item_id}:{kind}:{url}"
            if result["status"] != "verified_local":
                previous = index["assets"].get(key, {})
                index["assets"][key] = {"url": url, **{k: v for k, v in result.items() if k != "local"},
                                        "attempts": int(previous.get("attempts", 0)) + 1,
                                        "attempted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                continue
            if result["mime"] not in EXPECTED_FAMILY[kind]:
                index["assets"][key] = {"url": url, "status": "quarantined",
                                        "error": f"{result['mime']} in a {kind} slot", "attempts": 1}
                continue
            jobs.append({**result, "key": key, "item_id": item_id})
        fresh = {row["sha256"]: row for row in jobs if row["sha256"] not in index["objects"]}
        outcome = {"batch": batch_number, "attempted": len(slice_), "verified_local": len(jobs),
                   "new_objects": len(fresh)}
        if dry_run:
            shutil.rmtree(stage, ignore_errors=True)
            history.append({**outcome, "dry_run": True})
            break
        if fresh:
            for digest, row in fresh.items():
                destination = Path(stage) / object_path(digest, row["local"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(row["local"], destination)
            api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=stage, path_in_repo=".",
                              revision=WRITE_REVISION,
                              commit_message=f"{kind} batch {batch_number}: {len(fresh)} objects")
        remote = remote_oids(api, sorted({object_path(r["sha256"], r["local"]) for r in jobs}))
        verified = 0
        for row in jobs:
            path = object_path(row["sha256"], row["local"])
            actual_sha, actual_bytes = remote.get(path, ("", 0))
            if actual_sha != row["sha256"] or (actual_bytes and actual_bytes != row["bytes"]):
                index["assets"][row["key"]] = {"url": row["url"], "status": "failed",
                                               "error": f"remote {actual_sha or 'missing'} / {actual_bytes}",
                                               "attempts": 3, "object": path}
                continue
            if row["sha256"] not in index["objects"]:
                index["objects"][row["sha256"]] = {"object": path, "bytes": row["bytes"], "mime": row["mime"],
                                                   "kind": kind, "sha256": row["sha256"], "url": row["url"],
                                                   "archived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            index["assets"][row["key"]] = {"url": row["url"], "status": "verified_remote", "object": path,
                                           "sha256": row["sha256"], "bytes": row["bytes"], "mime": row["mime"]}
            refs.setdefault(row["item_id"], []).append(
                {"kind": kind, "object": path, "sha256": row["sha256"], "bytes": row["bytes"],
                 "mime": row["mime"], "source_url": row["url"]})
            verified += 1
        shutil.rmtree(stage, ignore_errors=True)
        write_state(api, index, refs, kind)
        remaining = sum(1 for item_id, url in queue if not done(f"{item_id}:{kind}:{url}", index))
        outcome.update({"verified_remote": verified, "queue_remaining": remaining,
                        "bytes_archived_mb": round(sum(o["bytes"] for o in index["objects"].values()) / 1048576, 1),
                        "cache_dir_removed": True})
        history.append(outcome)
        last_batch_seconds = time.time() - batch_start
        log(f"batch {batch_number}: {verified}/{len(jobs)} verified remote, "
            f"{remaining} assets still pending, {round(last_batch_seconds / 60, 1)} min")
        if verified == 0 and not fresh:
            history.append({"stop": "no progress in this batch; check the failures"})
            break
    return {"kind": kind, "queue_assets": len(queue), "batches": history,
            "objects_total": len(index["objects"]),
            "asset_lines": len(index["assets"]),
            "by_status": dict(Counter(a["status"] for a in index["assets"].values())),
            "written_to": {"repo": REPO, "revision": WRITE_REVISION,
                           "records_revision": RECORDS_REVISION,
                           "state_revision": STATE_REVISION, "refs_items": len(refs)},
            "minutes": round((time.time() - started) / 60, 1)}


def write_state(api, index: dict, refs: dict[str, list[dict]], kind: str) -> None:
    """Commit the resume state with the data, so the next run starts from here."""
    stage = tempfile.mkdtemp(prefix="voa-state-")
    root = Path(stage)
    (root / "_worker").mkdir(parents=True, exist_ok=True)
    (root / "_worker" / "media_index.json").write_text(json.dumps(index, ensure_ascii=False),
                                                        encoding="utf-8")
    with (root / "_worker" / f"{kind}_refs.jsonl").open("w", encoding="utf-8") as handle:
        for item_id, rows in sorted(refs.items()):
            handle.write(json.dumps({"item_id": item_id, "media_refs": rows}, ensure_ascii=False) + "\n")
    (root / "_worker" / f"{kind}_checkpoint.json").write_text(json.dumps({
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "objects": len(index["objects"]),
        "by_status": dict(Counter(a["status"] for a in index["assets"].values())),
        "object_bytes_mb": round(sum(o["bytes"] for o in index["objects"].values()) / 1048576, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=stage, path_in_repo=".",
                      revision=WRITE_REVISION, commit_message=f"resume state after {kind} batch")
    shutil.rmtree(stage, ignore_errors=True)


class Stopped(Exception):
    """Kept as the named reason for an interrupted run; signals exit through _on_signal."""


def main() -> int:
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="audio", choices=("audio", "video", "pdf"))
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--batch-assets", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-minutes", type=float, default=55.0)
    parser.add_argument("--reserve-minutes", type=float, default=10.0,
                        help="time kept back so a batch finishes and checkpoints before the deadline")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run(args.kind, args.batches, args.batch_assets, args.workers,
                 args.max_minutes, args.dry_run, args.reserve_minutes)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
