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
from pathlib import Path, PurePosixPath

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


def load_refs(api, kind: str) -> dict[str, list[dict]]:
    """Per-kind reference rows, kept in their own file so a video run never rewrites audio."""
    raw = read_file(api, f"_worker/{kind}_refs.jsonl", revision=STATE_REVISION)
    refs: dict[str, list[dict]] = {}
    for line in (raw or b"").decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            refs[row["item_id"]] = [entry for entry in row["media_refs"] if entry.get("kind") == kind]
    return {key: value for key, value in refs.items() if value}


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


def misplaced(name: str) -> bool:
    """True when a staged path is not SHA-bucketed media or a state file one level down.

    The hub caps a directory at 10000 files, and the flat root is what filled it once, so
    media may only live at media/<sha[:2]>/<sha>.<ext> and state at _worker/ or reports/.
    """
    parts = PurePosixPath(name).parts
    if parts[0] == "media":
        return len(parts) != 3 or len(parts[1]) != 2
    return parts[0] not in ("_worker", "reports") or len(parts) != 2


def remote_oids(api, paths: list[str]) -> dict[str, tuple[str, int]]:
    found: dict[str, tuple[str, int]] = {}
    for index, start in enumerate(range(0, len(paths), 500)):
        if index:
            time.sleep(0.5)  # these reads share the same request window as commits
        chunk = paths[start:start + 500]
        for entry in api.get_paths_info(REPO, chunk, expand=True, repo_type="dataset",
                                        revision=WRITE_REVISION):
            lfs = getattr(entry, "lfs", None)
            digest = (getattr(lfs, "sha256", "") or "").lower()
            if digest:
                found[entry.path] = (digest, int(getattr(lfs, "size", 0) or 0))
    return found


class RateLimited(Exception):
    """The hub said no. The asset is not failed, and it is never retried blindly."""

    def __init__(self, retry_after: float, headers: dict) -> None:
        super().__init__(f"hub rate limited (retry after {retry_after:.0f}s)")
        self.retry_after = retry_after
        self.headers = headers


def api_headers(url: str | None = None) -> dict:
    """The hub's own accounting, read off a cheap response instead of guessed at."""
    wanted = ("ratelimit", "ratelimit-policy", "retry-after")
    try:
        request = urllib.request.Request(url or f"https://huggingface.co/api/datasets/{REPO}",
                                        headers={"Authorization": "Bearer " + token()})
        with urllib.request.urlopen(request, timeout=60) as response:
            return {key: value for key, value in response.headers.items()
                    if any(part in key.lower() for part in wanted)}
    except urllib.error.HTTPError as error:
        return {key: value for key, value in (error.headers or {}).items()
                if any(part in key.lower() for part in wanted)} or {"http_status": str(error.code)}
    except Exception as error:
        return {"probe_error": f"{type(error).__name__}: {str(error)[:80]}"}


def rate_state(headers: dict) -> dict:
    """Parse RateLimit `"api";r=988;t=277` plus the q/w window out of RateLimit-Policy."""
    state = {"remaining": None, "reset_in": None, "quota": None, "window": None}
    for part in (headers.get("RateLimit") or headers.get("ratelimit") or "").split(";"):
        part = part.strip().strip('"')
        if part.startswith("r="):
            state["remaining"] = int(part[2:])
        elif part.startswith("t="):
            state["reset_in"] = int(part[2:])
    policy = headers.get("RateLimit-Policy") or headers.get("ratelimit-policy") or ""
    quota = re.search(r"q=(\d+)", policy)
    window = re.search(r"w=(\d+)", policy)
    if quota:
        state["quota"] = int(quota.group(1))
    if window:
        state["window"] = int(window.group(1))
    retry = headers.get("Retry-After") or headers.get("retry-after")
    if retry:
        state["retry_after"] = int(retry)
    return state


class Budget:
    """Commit accounting measured from the dataset, never from an assumed HF number.

    The observed window is `q=1000; w=300` requests per 5 minutes, but these caps are the
    operator's precaution: they stop before a limit is reached rather than discovering it.
    """

    def __init__(self, api, max_per_run: int, min_interval: float,
                 min_remaining: int) -> None:
        self.api = api
        self.max_per_run = max_per_run
        self.min_interval = min_interval
        self.min_remaining = min_remaining
        self.used = 0
        self.headers: dict = {}
        self.state: dict = {}
        self.last_commit = 0.0
        self.files = 0
        self.bytes = 0
        self.last_checkpoint: dict = {}
        self.stop_reason = ""
        self.state_current_flushed = False

    def commits_last_24h(self) -> int:
        day_ago = time.time() - 86400
        count = 0
        for commit in self.api.list_repo_commits(REPO, repo_type="dataset", revision=WRITE_REVISION):
            stamp = getattr(commit, "created_at", None)
            if stamp is None:
                continue
            created = stamp.timestamp() if hasattr(stamp, "timestamp") else float(stamp)
            if created >= day_ago:
                count += 1
            else:
                break
        return count

    def allow(self) -> str:
        """Why this commit must wait, or "" when it may go out now."""
        if self.used >= self.max_per_run:
            return f"max_commits_per_run reached ({self.max_per_run})"
        if self.last_commit:
            wait = self.min_interval - (time.time() - self.last_commit)
            if wait > 0:
                time.sleep(wait)
        self.headers = api_headers()
        self.state = rate_state(self.headers)
        remaining = self.state.get("remaining")
        if remaining is not None and remaining < self.min_remaining:
            return (f"rate window low: remaining={remaining} "
                    f"reset_in={self.state.get('reset_in')}s of q={self.state.get('quota')}")
        # No assumed daily ceiling: the measured window and the hub's own refusals decide.
        # commits_last_24h() stays in the report as observation only.
        return ""

    def allow_ignoring_pacing(self) -> str:
        """Same caps and the same rate window, without waiting - used once at the end."""
        self.headers = api_headers()
        self.state = rate_state(self.headers)
        remaining = self.state.get("remaining")
        if remaining is not None and remaining < self.min_remaining:
            return (f"rate window low: remaining={remaining} "
                    f"reset_in={self.state.get('reset_in')}s of q={self.state.get('quota')}")
        return ""

    def record(self, files: int, uploaded_bytes: int, checkpoint: dict) -> None:
        self.used += 1
        self.files += files
        self.bytes += uploaded_bytes
        self.last_commit = time.time()
        self.last_checkpoint = checkpoint

    def report(self, kind: str) -> dict:
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kind": kind,
            "commits_this_run": self.used,
            "commits_last_24h": self.commits_last_24h(),
            "cap_commits_per_run": self.max_per_run,
            "min_commit_interval_seconds": self.min_interval,
            "files_uploaded_this_run": self.files,
            "bytes_uploaded_this_run": self.bytes,
            "bytes_uploaded_this_run_mb": round(self.bytes / 1048576, 1),
            "last_rate_limit_headers": self.headers,
            "rate_window": self.state,
            "last_verified_checkpoint": self.last_checkpoint,
            "stopped_because": self.stop_reason,
            "hub_window_observed": "RateLimit-Policy q=1000 w=300 (requests per 5 min); "
                                   "these caps are precautionary, not a claimed HF limit",
        }


def done(key: str, index: dict) -> bool:
    """True when this asset needs no fetching: archived, terminal-failed, or uploaded
    and awaiting remote confirmation (re-downloading that would be pure waste)."""
    row = index["assets"].get(key)
    if not row:
        return False
    if row["status"] in ("verified_remote", "uploaded_unverified"):
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
            publish_state(STATE_HOOK["api"], STATE_HOOK["index"], STATE_HOOK["refs"],
                          STATE_HOOK["kind"], STATE_HOOK["budget"])
        except Exception as error:
            print(f"checkpoint flush failed: {error}", file=sys.stderr, flush=True)
    raise SystemExit(128 + signum)


STATE_HOOK: dict = {}


def promote_pending_uploads(api, index: dict, refs: dict[str, list[dict]]) -> int:
    """A run that died after uploading but before confirming leaves assets unverified.

    Confirming them from the hub's own metadata costs one request per 500 and stops the
    next run from treating them as still to be done.
    """
    waiting = {key: row for key, row in index["assets"].items()
               if row.get("status") == "uploaded_unverified"}
    if not waiting:
        return 0
    remote = remote_oids(api, sorted({row["object"] for row in waiting.values()}))
    promoted = 0
    for key, row in waiting.items():
        item_id = key.split(":")[0]
        actual_sha, actual_bytes = remote.get(row["object"], ("", 0))
        if actual_sha.lower() == row["sha256"] and (not actual_bytes or actual_bytes == row["bytes"]):
            row["status"] = "verified_remote"
            promoted += 1
            continue
        # The bytes are not in the repo, so this is our own interrupted commit, not a dead
        # provider URL: keep it retryable and forget the object line, otherwise the next
        # batch would call missing bytes "reused" and the index would claim an object that
        # was never committed.
        row["status"] = "upload_incomplete"
        row["attempts"] = 1
        row["error"] = f"indexed object not confirmed remotely ({actual_sha or 'missing'})"
        index["objects"].pop(row["sha256"], None)
        kept = [entry for entry in refs.get(item_id, []) if entry["sha256"] != row["sha256"]]
        if kept:
            refs[item_id] = kept
        else:
            refs.pop(item_id, None)
    log(f"promoted {promoted}/{len(waiting)} assets left unconfirmed by an earlier run")
    return promoted


def snapshot(index: dict, refs: dict, fresh: dict, jobs: list[dict]) -> dict:
    """What the keys this batch is about to overwrite currently hold, so a refusal undoes it."""
    touched_items: dict[str, list[dict] | None] = {}
    for row in jobs:
        item_id = row["item_id"]
        if item_id not in touched_items:
            touched_items[item_id] = list(refs[item_id]) if item_id in refs else None
    return {"objects": {digest: index["objects"].get(digest) for digest in fresh},
            "assets": {row["key"]: index["assets"].get(row["key"]) for row in jobs},
            "refs": touched_items}


def rollback(index: dict, refs: dict, saved: dict) -> None:
    for digest, previous in saved["objects"].items():
        if previous is None:
            index["objects"].pop(digest, None)
        else:
            index["objects"][digest] = previous
    for key, previous in saved["assets"].items():
        if previous is None:
            index["assets"].pop(key, None)
        else:
            index["assets"][key] = previous
    for item_id, previous in saved["refs"].items():
        if previous is None:
            refs.pop(item_id, None)
        else:
            refs[item_id] = previous


def looks_like_quota(message: str) -> bool:
    """Only the hub's own refusal words may turn a stop into a wait."""
    lowered = message.lower()
    return any(marker in lowered for marker in
               ("429", "rate limit", "rate-limit", "retry-after", "too many requests", "quota"))


def run(kind: str, batches: int, batch_assets: int, workers: int, max_minutes: float,
        dry_run: bool, reserve_minutes: float = 10.0, commit_files: int = 90,
        max_commits_per_run: int = 30,
        min_commit_interval: float = 20.0, min_remaining: int = 150) -> dict:
    api = hf_api()
    until = backoff_until(api)
    if until and time.time() < until:
        return {"kind": kind, "waiting_for_backoff": True,
                "seconds_left": int(until - time.time()),
                "reason": "a previous run was rate limited; this run exits without touching the hub"}
    if kind == "video":
        allowed, why = video_gate_allowed(api)
        if not allowed:
            return {"aborted": f"video gate closed: {why}"}
    budget = Budget(api, max_commits_per_run, min_commit_interval, min_remaining)
    if reserve_minutes * 60 >= max_minutes * 60:
        # with a reserve that swallows the whole step, no batch can ever start: saying so is
        # the difference between a refused run and a green one that archived nothing
        return {"kind": kind, "refused": f"max_minutes {max_minutes} must exceed reserve_minutes "
                                         f"{reserve_minutes}, otherwise no batch can start",
                "batches": [], "queue_assets": 0, "objects_total": 0, "asset_lines": 0,
                "by_status": {}, "commit_budget": budget.report(kind), "minutes": 0.0}
    index = load_state(api)
    refs = load_refs(api, kind)
    promote_pending_uploads(api, index, refs)
    queue = queue_from_shards(api, kind)
    STATE_HOOK.clear()
    STATE_HOOK.update({"api": api, "index": index, "refs": refs, "kind": kind,
                       "dry_run": dry_run, "budget": budget})
    log(f"{kind}: queue {len(queue)} assets, index has {len(index['objects'])} objects, "
        f"{commit_files} files per commit")
    started = time.time()
    history: list[dict] = []
    last_batch_seconds = 0.0
    dirty = False
    refused = ""
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
        batch_start = time.time()
        stage = tempfile.mkdtemp(prefix=f"voa-{kind}-")
        # Downloads never touch the upload stage: a file sitting at the stage root is
        # committed to the repo root, and the hub refuses any push that takes one directory
        # past 10000 files. That refusal is what killed the audio run on 2026-11-03.
        scratch = tempfile.mkdtemp(prefix=f"voa-dl-{kind}-")

        def discard() -> None:
            for directory in (stage, scratch):
                shutil.rmtree(directory, ignore_errors=True)

        # dynamic sizing: keep pulling candidates until this many DISTINCT new objects are
        # staged, so media plus state stays one commit inside the safe operations band
        jobs: list[dict] = []
        new_digests: set[str] = set()
        wave = max(workers, min(commit_files, 30))
        cursor = 0
        scanned = 0
        while len(new_digests) < commit_files and cursor < min(len(pending), batch_assets):
            chunk = pending[cursor:cursor + wave]
            cursor += len(chunk)
            scanned += len(chunk)
            with futures.ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda pair: pull(pair[1], scratch), chunk))
            for (item_id, url), result in zip(chunk, results):
                key = f"{item_id}:{kind}:{url}"
                if result["status"] != "verified_local":
                    previous = index["assets"].get(key, {})
                    index["assets"][key] = {"url": url,
                                            **{k: v for k, v in result.items() if k != "local"},
                                            "attempts": int(previous.get("attempts", 0)) + 1,
                                            "attempted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                          time.gmtime())}
                    continue
                if result["mime"] not in EXPECTED_FAMILY[kind]:
                    index["assets"][key] = {"url": url, "status": "quarantined",
                                            "error": f"{result['mime']} in a {kind} slot", "attempts": 1}
                    continue
                jobs.append({**result, "key": key, "item_id": item_id})
                if result["sha256"] not in index["objects"]:
                    new_digests.add(result["sha256"])
        # trim to exactly commit_files distinct NEW digests: several items can share one
        # audio file, and without this a duplicate pushes the commit past the split point
        accepted: set[str] = set()
        kept: list[dict] = []
        for row in jobs:
            if row["sha256"] in index["objects"] or row["sha256"] in accepted:
                kept.append(row)
            elif len(accepted) < commit_files:
                accepted.add(row["sha256"])
                kept.append(row)
        jobs = kept
        fresh = {digest: next(row for row in jobs if row["sha256"] == digest)
                 for digest in accepted}
        outcome = {"batch": batch_number, "candidates_scanned": scanned,
                   "verified_local": len(jobs), "new_objects": len(fresh)}
        if dry_run:
            discard()
            history.append({**outcome, "dry_run": True})
            break
        block = budget.allow()
        if block:
            discard()
            budget.stop_reason = block
            history.append({**outcome, "commit_blocked": block})
            save_backoff(api, kind, budget)
            dirty = False  # the backoff commit already carried the state
            break
        # captured before anything is mutated: a refused commit must be able to put these
        # keys back exactly as the repository still holds them
        saved = snapshot(index, refs, fresh, jobs)
        for digest, row in fresh.items():
            destination = Path(stage) / object_path(digest, row["local"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row["local"], destination)
        # one pass over every job: two items can reference the same audio file, and each
        # of them needs its own asset line and ref row whether or not the bytes are new
        for row in jobs:
            digest = row["sha256"]
            path = object_path(digest, row["local"])
            reused = digest in index["objects"]
            if not reused:
                index["objects"][digest] = {"object": path, "bytes": row["bytes"],
                                            "mime": row["mime"], "kind": kind, "sha256": digest,
                                            "url": row["url"],
                                            "archived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                         time.gmtime())}
            entry = {"url": row["url"], "object": path, "sha256": digest,
                     "bytes": row["bytes"], "mime": row["mime"],
                     "status": "verified_remote" if reused else "uploaded_unverified"}
            if reused:
                entry["reused_object"] = True
            index["assets"][row["key"]] = entry
            refs.setdefault(row["item_id"], []).append({"kind": kind, **entry})
        staged = sorted(str(path.relative_to(stage)).replace("\\", "/")
                        for path in Path(stage).rglob("*") if path.is_file())
        offenders = [name for name in staged if misplaced(name)]
        if offenders:
            discard()
            raise RuntimeError(f"media must be SHA-bucketed, state one level down; "
                               f"offenders: {offenders[:5]}")
        stage_state(stage, index, refs, kind, budget, len(fresh))
        # one commit: the batch's media files AND the resume state AND the budget report
        try:
            commit = api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=stage,
                                       path_in_repo=".", revision=WRITE_REVISION,
                                       commit_message=f"{kind}: {len(fresh)} objects + resume "
                                                      f"state (batch {batch_number})")
        except Exception as error:
            # nothing landed, so the resume state must not remember this batch either:
            # an entry may only claim bytes the repository already holds
            rollback(index, refs, saved)
            discard()
            dirty = False
            waited = looks_like_quota(str(error))
            refused = f"commit refused by the hub ({'quota' if waited else type(error).__name__})"
            budget.stop_reason = f"{refused}: {str(error)[:300]}"
            history.append({"batch": batch_number, "refused": refused, "error": str(error)[:500],
                            "rate_window": budget.state, "rolled_back_objects": len(fresh),
                            "rolled_back_assets": len(jobs)})
            if waited:
                save_backoff(api, kind, budget)  # sleep until the window resets, then resume
            log(budget.stop_reason)
            break
        budget.files += len(fresh) + 4
        budget.bytes += sum(row["bytes"] for row in fresh.values())
        remote = remote_oids(api, sorted({object_path(r["sha256"], r["local"]) for r in jobs}))
        verified = mismatches = 0
        for row in jobs:
            path = object_path(row["sha256"], row["local"])
            actual_sha, actual_bytes = remote.get(path, ("", 0))
            if actual_sha.lower() != row["sha256"] or (actual_bytes and actual_bytes != row["bytes"]):
                mismatches += 1
                index["assets"][row["key"]] = {"url": row["url"], "status": "failed",
                                               "error": f"remote {actual_sha or 'missing'} / {actual_bytes}",
                                               "attempts": 3, "object": path}
                continue
            index["assets"][row["key"]]["status"] = "verified_remote"
            verified += 1
        discard()
        remaining = sum(1 for item_id, url in queue if not done(f"{item_id}:{kind}:{url}", index))
        checkpoint = {"commit": getattr(commit, "commit_url", "") or str(commit)[:60],
                      "objects": len(index["objects"]), "queue_remaining": remaining}
        budget.record(len(fresh) + 4, sum(row["bytes"] for row in fresh.values()), checkpoint)
        # promotions land after the commit, so the next batch or the final flush carries them
        dirty = bool(verified or mismatches)
        if mismatches:
            # only the rare correction costs a second commit; a clean batch costs exactly one
            correction = tempfile.mkdtemp(prefix="voa-fix-")
            stage_state(correction, index, refs, kind, budget, 0)
            api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=correction,
                              path_in_repo=".", revision=WRITE_REVISION,
                              commit_message=f"{kind}: state correction after {mismatches} mismatches")
            shutil.rmtree(correction, ignore_errors=True)
        outcome.update({"verified_remote": verified, "mismatches": mismatches,
                        "commits_used": budget.used, "commits_last_24h": budget.commits_last_24h(),
                        "rate_remaining": budget.state.get("remaining"),
                        "queue_remaining": remaining,
                        "bytes_archived_mb": round(sum(o["bytes"] for o in index["objects"].values())
                                                   / 1048576, 1),
                        "cache_dir_removed": True})
        history.append(outcome)
        last_batch_seconds = time.time() - batch_start
        log(f"batch {batch_number}: {verified} verified in 1 commit, "
            f"{remaining} pending, {round(last_batch_seconds / 60, 1)} min")
        if verified == 0 and not fresh:
            history.append({"stop": "no progress in this batch; check the failures"})
            break
    if not budget.stop_reason:
        budget.stop_reason = budget.allow_ignoring_pacing() or "queue or batch budget exhausted"
    # one tidy-up commit per RUN, never per batch, and none at all when a
    # rate-limited batch already wrote the state with its backoff marker
    if dirty and not dry_run:
        try:
            publish_state(api, index, refs, kind, budget)
            budget.state_current_flushed = True
        except Exception as error:
            log(f"final state publish failed: {error}")
    return {"kind": kind, "queue_assets": len(queue), "batches": history,
            "refused": refused,
            "objects_total": len(index["objects"]),
            "asset_lines": len(index["assets"]),
            "by_status": dict(Counter(a["status"] for a in index["assets"].values())),
            "commit_budget": budget.report(kind),
            "written_to": {"repo": REPO, "revision": WRITE_REVISION,
                           "records_revision": RECORDS_REVISION,
                           "state_revision": STATE_REVISION, "refs_items": len(refs)},
            "minutes": round((time.time() - started) / 60, 1)}


def stage_state(directory: str, index: dict, refs: dict[str, list[dict]], kind: str,
                budget: "Budget", new_objects: int) -> int:
    """Write media_index, refs, checkpoint and the budget report into a staging dir.

    These land in the same commit as the batch's media files, which is what removes the
    separate state commit that used to follow every batch.
    """
    root = Path(directory)
    (root / "_worker").mkdir(parents=True, exist_ok=True)
    (root / "_worker" / "media_index.json").write_text(json.dumps(index, ensure_ascii=False),
                                                        encoding="utf-8")
    with (root / "_worker" / f"{kind}_refs.jsonl").open("w", encoding="utf-8") as handle:
        for item_id, rows in sorted(refs.items()):
            handle.write(json.dumps({"item_id": item_id, "media_refs": rows}, ensure_ascii=False) + "\n")
    checkpoint = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "objects": len(index["objects"]),
        "by_status": dict(Counter(a["status"] for a in index["assets"].values())),
        "object_bytes_mb": round(sum(o["bytes"] for o in index["objects"].values()) / 1048576, 1),
        "new_objects_in_this_commit": new_objects,
    }
    (root / "_worker" / f"{kind}_checkpoint.json").write_text(json.dumps(checkpoint, ensure_ascii=False,
                                                                         indent=1), encoding="utf-8")
    (root / "reports" / "commit_budget.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "reports" / "commit_budget.json").write_text(json.dumps(budget.report(kind),
                                                                    ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    return len(checkpoint)


def publish_state(api, index: dict, refs: dict[str, list[dict]], kind: str,
                  budget: "Budget") -> None:
    """Standalone state commit, used for the final flush, a rate-limit backoff and signals."""
    stage = tempfile.mkdtemp(prefix="voa-state-")
    stage_state(stage, index, refs, kind, budget, 0)
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=stage, path_in_repo=".",
                      revision=WRITE_REVISION, commit_message=f"{kind}: resume state flush")
    shutil.rmtree(stage, ignore_errors=True)


def save_backoff(api, kind: str, budget: "Budget") -> None:
    """Record when it is safe to try again, so a waiting run costs zero requests."""
    stage = tempfile.mkdtemp(prefix="voa-backoff-")
    reset = budget.state.get("reset_in") or budget.state.get("retry_after") or 300
    (Path(stage) / "_worker").mkdir(parents=True, exist_ok=True)
    (Path(stage) / "_worker" / "backoff.json").write_text(json.dumps({
        "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "not_before_epoch": int(time.time() + reset + 60),
        "wait_seconds": reset + 60,
        "reason": budget.stop_reason,
        "last_rate_limit_headers": budget.headers,
        "rate_window": budget.state,
        "kind": kind,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    stage_state(stage, STATE_HOOK["index"], STATE_HOOK["refs"], kind, budget, 0)
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=stage, path_in_repo=".",
                      revision=WRITE_REVISION, commit_message=f"{kind}: backoff marker (rate limited)")
    shutil.rmtree(stage, ignore_errors=True)


def backoff_until(api) -> int:
    raw = read_file(api, "_worker/backoff.json", revision=STATE_REVISION)
    if not raw:
        return 0
    try:
        return int(json.loads(raw).get("not_before_epoch") or 0)
    except (json.JSONDecodeError, ValueError):
        return 0


class Stopped(Exception):
    """Named reason for an interrupted run; signals flush state through _on_signal."""


def main() -> int:
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="audio", choices=("audio", "video", "pdf"))
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-assets", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-minutes", type=float, default=55.0)
    parser.add_argument("--reserve-minutes", type=float, default=10.0,
                        help="time kept back so a batch finishes and checkpoints before the deadline")
    parser.add_argument("--commit-files", type=int, default=90,
                        help="distinct new objects per commit, state included in the same commit")
    parser.add_argument("--max-commits-per-run", type=int,
                        default=int(os.environ.get("MAX_COMMITS_PER_RUN", "30")))
    parser.add_argument("--min-commit-interval", type=float, default=20.0,
                        help="minimum seconds between commits (pacing)")
    parser.add_argument("--min-remaining", type=int, default=150,
                        help="stop committing when the hub rate window drops below this")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run(args.kind, args.batches, args.batch_assets, args.workers,
                 args.max_minutes, args.dry_run, args.reserve_minutes, args.commit_files,
                 args.max_commits_per_run, args.min_commit_interval, args.min_remaining)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    # a hub refusal is a failure, never a quiet green step
    return 1 if report.get("refused") else 0


if __name__ == "__main__":
    raise SystemExit(main())
