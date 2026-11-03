# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=1.31,<2", "pyarrow>=15"]
# ///
"""Titan Learn - independent media verification, run between archival phases.

This does not trust the archiver's own bookkeeping. It re-reads the dataset, asks the hub
for the real LFS sha256 and byte size of every object, compares that against the index,
then checks both directions: an indexed object that is not remote, a remote object nobody
indexed, and reference rows that point at something that is not actually there.

The output is what the video gate reads, so "audio finished and verified" is evidence on
the shared store rather than a claim in a chat log.

    python verify_media.py --kind audio            # check + publish the report
    python verify_media.py --kind audio --no-publish
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

try:
    from .worker import (REPO, STATE_REVISION, WRITE_REVISION, hf_api, queue_from_shards,
                        read_file, remote_oids, done)
except ImportError:  # run as a plain file from the worker repo
    from worker import (REPO, STATE_REVISION, WRITE_REVISION, hf_api, queue_from_shards,
                        read_file, remote_oids, done)


def all_remote_media_paths(api) -> set[str]:
    """Everything actually sitting under media/ in the dataset right now."""
    return {path for path in api.list_repo_files(REPO, repo_type="dataset", revision=WRITE_REVISION)
            if path.startswith("media/")}


def verify(kind: str, publish: bool, recent: int = 0) -> dict:
    """`recent` bounds a between-batches sweep; only a full sweep can call the phase complete."""
    api = hf_api()
    index = json.loads(read_file(api, "_worker/media_index.json", revision=STATE_REVISION).decode("utf-8"))
    refs_raw = read_file(api, f"_worker/{kind}_refs.jsonl", revision=STATE_REVISION) or b""
    refs = [json.loads(line) for line in refs_raw.decode("utf-8").splitlines() if line.strip()]
    objects = index["objects"]

    queue = queue_from_shards(api, kind)
    remaining = sum(1 for item_id, url in queue if not done(f"{item_id}:{kind}:{url}", index))
    terminal_failed = sum(1 for item_id, url in queue
                          if index["assets"].get(f"{item_id}:{kind}:{url}", {}).get("status")
                          in ("failed", "quarantined"))

    of_kind = {digest: obj for digest, obj in objects.items() if obj.get("kind") == kind}
    scope = "full"
    if recent and len(of_kind) > recent:
        # newest first, so a partial sweep still covers exactly what just went in
        scope = f"recent {recent}"
        of_kind = dict(sorted(of_kind.items(), key=lambda kv: kv[1].get("archived_at") or "",
                              reverse=True)[:recent])
    paths = sorted(obj["object"] for obj in of_kind.values())
    remote = remote_oids(api, paths)
    match = mismatch = 0
    offenders: list[str] = []
    for digest, obj in of_kind.items():
        actual_sha, actual_bytes = remote.get(obj["object"], ("", 0))
        if actual_sha.lower() == obj["sha256"] and (not actual_bytes or int(actual_bytes) == obj["bytes"]):
            match += 1
        else:
            mismatch += 1
            if len(offenders) < 10:
                offenders.append(f"{obj['object']}: remote sha {actual_sha or 'missing'} "
                                 f"bytes {actual_bytes} vs {obj['bytes']}")

    live = all_remote_media_paths(api)
    indexed_paths = {obj["object"] for obj in objects.values()}
    orphans = sorted(live - indexed_paths)
    missing_remotely = sorted(indexed_paths - live)

    ref_rows = sum(len(entry["media_refs"]) for entry in refs)
    ref_targets = {row["sha256"] for entry in refs for row in entry["media_refs"]}
    dangling_refs = sorted(ref_targets - set(objects))
    indexed_digests = {digest for digest, obj in objects.items() if obj.get("kind") == kind}
    items_without_ref = len(indexed_digests) - len(ref_targets & indexed_digests)

    bytes_for_kind = sum(obj["bytes"] for obj in objects.values() if obj.get("kind") == kind)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": kind,
        "scope": scope,
        "method": ("independent re-read of the dataset plus LFS metadata from the hub API; "
                   "the archiver's own counters are not trusted here"),
        "queue_total_assets": len(queue),
        "audio_queue_remaining" if kind == "audio" else "video_queue_remaining": remaining,
        "terminal_failed_assets": terminal_failed,
        "indexed_objects_for_kind": len(paths),
        "indexed_ok": match,
        "indexed_missing_or_mismatch": mismatch,
        "offenders": offenders,
        "remote_objects_present": len(live),
        "remote_orphans_not_indexed": len(orphans),
        "orphan_examples": orphans[:10],
        "indexed_paths_absent_remotely": missing_remotely[:10],
        "refs_rows": ref_rows,
        "refs_items": len(refs),
        "dangling_ref_targets": dangling_refs[:10],
        "dangling_refs": len(dangling_refs),
        "items_indexed_without_ref": max(0, items_without_ref),
        "bytes_gb": round(bytes_for_kind / 1073741824, 2),
        "bytes_mb": round(bytes_for_kind / 1048576, 1),
        # integrity answers "is what we claim to have archived actually there and identical";
        # completion additionally requires the queue drained with nothing unaccounted for
        "integrity_ok": mismatch == 0 and not missing_remotely and not dangling_refs,
        "all_match": mismatch == 0 and not missing_remotely and not dangling_refs,
        # only a full sweep may declare the phase complete; a between-batches sweep cannot
        "complete": (scope == "full" and remaining == 0 and mismatch == 0
                     and not missing_remotely and not dangling_refs and not orphans),
        "complete_note": ("orphans are expected while a batch is mid-commit; re-run this check "
                          "once the queue is drained before trusting `complete`"
                          if orphans and remaining == 0 else ""),
    }
    if publish:
        stage = Path(tempfile.mkdtemp(prefix="voa-verify-"))
        (stage / "reports").mkdir(parents=True, exist_ok=True)
        name = "audio_verify.json" if kind == "audio" else "video_verify.json"
        (stage / "reports" / name).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(stage), path_in_repo=".",
                          revision=WRITE_REVISION, commit_message=f"independent {kind} verification")
        report["published_as"] = f"reports/{name}"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="audio", choices=("audio", "video"))
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--recent", type=int, default=0,
                        help="check only the newest N objects (a cheap between-batch sweep)")
    args = parser.parse_args()
    report = verify(args.kind, not args.no_publish, args.recent)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if report["integrity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
