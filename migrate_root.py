"""VOA adapter - MIGRATE stage: move flat root media into SHA-bucketed paths.

The hub allows at most 10000 files in one directory. An earlier staging bug uploaded each
audio file twice - once straight into the repository root, named by the source URL hash,
and once at media/<sha[:2]>/<sha>.<ext> - and the root copy is what reached that ceiling
and made every commit a 400 Bad Request.

This module finishes the job properly: nothing is re-fetched from VOA, nothing is parked in
one legacy folder (that would only move the ceiling), and every root file is either proven
already present at its SHA-bucketed path or copied there from the dataset itself. A root
copy is deleted only after the branched copy is verified by SHA-256 and size and the index,
refs and checkpoint all point at it.

    python -m voa_factory.migrate_root                    # plan only, publishes nothing
    python -m voa_factory.migrate_root --apply --max-commits 25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath

from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from voa_factory import worker
from voa_factory.worker import REPO, WRITE_REVISION, log, object_path

MEDIA_SUFFIXES = (".mp3", ".m4a", ".wav", ".ogg", ".mp4", ".m4v", ".webm", ".pdf")
CHECKPOINT = "_worker/root_migration.json"


def is_media(path: str) -> bool:
    return "/" not in path and Path(path).suffix.lower() in MEDIA_SUFFIXES


def url_lookup(api) -> dict[str, tuple[str, str]]:
    """Root file name -> (item_id, source url), recovered from the pinned queue.

    The root names are sha256(url)[:20], so the provider URL of an unindexed copy is
    recoverable without touching VOA again.
    """
    lookup: dict[str, tuple[str, str]] = {}
    for item_id, url in worker.queue_from_shards(api, "audio") + worker.queue_from_shards(api, "video"):
        name = hashlib.sha256(url.encode()).hexdigest()[:20]
        lookup.setdefault(name, (item_id, url))
    return lookup


def classify(api, tree: set[str]) -> dict[str, dict]:
    """Decide, from hub metadata alone, what each root file needs. No bytes move here."""
    roots = sorted(path for path in tree if is_media(path))
    remote = worker.remote_oids(api, roots)
    wanted = sorted({object_path(digest, root) for root, (digest, _) in remote.items() if digest})
    branched = worker.remote_oids(api, wanted)
    decisions: dict[str, dict] = {}
    for root in roots:
        digest, size = remote.get(root, ("", 0))
        if not digest:
            decisions[root] = {"case": "unreadable", "target": "", "sha256": "", "bytes": int(size or 0)}
            continue
        digest = digest.lower()
        target = object_path(digest, root)
        actual_sha, actual_bytes = branched.get(target, ("", 0))
        if target in tree and actual_sha.lower() != digest:
            # the slot holds different bytes than the root copy: never merge, never delete
            decisions[root] = {"case": "conflict", "target": target, "sha256": digest,
                               "bytes": int(size or 0)}
        elif target in tree and actual_bytes and int(actual_bytes) != int(size or 0):
            decisions[root] = {"case": "conflict", "target": target, "sha256": digest,
                               "bytes": int(size or 0)}
        elif target in tree:
            decisions[root] = {"case": "duplicate", "target": target, "sha256": digest,
                               "bytes": int(size or 0)}
        else:
            decisions[root] = {"case": "copy", "target": target, "sha256": digest,
                               "bytes": int(size or 0)}
    return decisions


def record_object(index: dict, refs: dict, digest: str, target: str, size: int,
                  mime: str, url: str, item_id: str, kind: str) -> None:
    """Write the index, asset and ref lines a migrated object needs, with its provenance."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    existing = index["objects"].get(digest)
    if existing is None:
        index["objects"][digest] = {"object": target, "bytes": int(size), "mime": mime,
                                    "kind": kind, "sha256": digest, "url": url,
                                    "archived_at": stamp, "migrated_from_root": True}
    entry = {"url": url, "object": target, "sha256": digest, "bytes": int(size), "mime": mime,
             "status": "verified_remote", "migrated_from_root": True}
    index["assets"].setdefault(f"{item_id}:{kind}:{url}", entry)
    refs.setdefault(item_id, []).append({"kind": kind, **entry})


def state_ops(api, index: dict, refs: dict, kind: str, budget, report: dict,
              full: bool = True) -> list:
    """The files that describe progress.

    `full` rewrites media_index, refs and the archiver checkpoint through the archiver's own
    staging, so the two writers can never drift in shape. A plain delete batch only needs the
    migration checkpoint: the index and refs already point at the branched paths, and pushing
    an 8.7 MB index into every one of twenty batches would be pure weight.
    """
    stage = tempfile.mkdtemp(prefix="voa-migrate-")
    try:
        worker_dir = Path(stage) / "_worker"
        worker_dir.mkdir(parents=True, exist_ok=True)
        (worker_dir / Path(CHECKPOINT).name).write_text(
            json.dumps(progress(report), ensure_ascii=False, indent=1), encoding="utf-8")
        if full:
            worker.stage_state(stage, index, refs, kind, budget, 0)
            reports = Path(stage) / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            (reports / "root_migration.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        ops = []
        for path in sorted(Path(stage).rglob("*")):
            if path.is_file():
                ops.append(CommitOperationAdd(path_in_repo=str(path.relative_to(stage))
                                              .replace("\\", "/"),
                                             path_or_fileobj=path.read_bytes()))
        return ops
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def progress(report: dict) -> dict:
    """The resumable tail of a migration: what is left, and why."""
    return {key: report[key] for key in
            ("generated_at", "root_media_files", "cases", "head_commit_at_plan", "applied",
             "root_media_files_remaining", "deleted_total", "commits_used", "stopped_because")
            if key in report}


def head_sha(api) -> str:
    return api.list_repo_commits(REPO, repo_type="dataset", revision=WRITE_REVISION)[0].commit_id


def commit_sha(commit) -> str:
    return (getattr(commit, "commit_id", "") or "").strip() or \
        (getattr(commit, "commit_url", "") or "").rstrip("/").split("/")[-1]


def guard_single_writer(api, allowed: set[str]) -> None:
    """The archiver owns the same state files; refuse to interleave with it.

    `allowed` is the commit this run started from plus every commit the migration itself
    landed, so its own batches never look like a competing writer.
    """
    now = head_sha(api)
    if now not in allowed:
        raise RuntimeError(f"staging moved to {now[:12]}, which this migration did not write: "
                           f"another writer is active")


def bytes_from_dataset(api, root: str, expect_sha: str, expect_bytes: int) -> bytes:
    """Read one already-archived object back out of our own dataset (hub storage read).

    This is not a provider request: nothing here touches VOA. The bytes are re-proven by
    SHA-256 and size before they move to their SHA-bucketed path.
    """
    data = worker.read_file(api, root, required=True, revision=WRITE_REVISION)
    digest = hashlib.sha256(data).hexdigest()
    if digest != expect_sha or len(data) != expect_bytes:
        raise RuntimeError(f"root copy {root} reads {digest}/{len(data)}, expected "
                           f"{expect_sha}/{expect_bytes}")
    return data


def branch_copies(api, index: dict, refs: dict, decisions: dict, lookup: dict, kind: str,
                  budget, report: dict, allowed: set[str]) -> None:
    """One commit for every root file that has no branched object yet, then verify it."""
    staged: list[tuple[str, dict, bytes]] = []
    for root, row in sorted(decisions.items()):
        if row["case"] != "copy":
            continue
        name = Path(root).stem
        if name not in lookup:
            report["batches"].append({"kept": root, "reason": "no corpus url traces to it"})
            continue
        staged.append((root, row, bytes_from_dataset(api, root, row["sha256"], row["bytes"])))
    if not staged:
        return
    ops = [CommitOperationAdd(path_in_repo=row["target"], path_or_fileobj=data)
           for _, row, data in staged]
    for root, row, data in staged:
        item_id, url = lookup[Path(root).stem]
        mime = worker.sniff(data[:12]) or "application/octet-stream"
        record_object(index, refs, row["sha256"], row["target"], row["bytes"], mime,
                      url, item_id, kind)
    guard_single_writer(api, allowed)
    commit = api.create_commit(repo_id=REPO, repo_type="dataset", revision=WRITE_REVISION,
                               commit_message=f"migrate root media: branch {len(staged)} objects "
                                              f"+ resume state",
                               operations=ops + state_ops(api, index, refs, kind, budget, report))
    allowed.add(commit_sha(commit))
    budget.record(len(staged) + 4, 0, {"commit": getattr(commit, "commit_url", ""),
                                       "branched": len(staged)})
    report["batches"].append({"branched": len(staged),
                              "commit": getattr(commit, "commit_url", "")[:60]})
    live = worker.remote_oids(api, [row["target"] for _, row, _ in staged])
    for root, row, data in staged:
        sha, size = live.get(row["target"], ("", 0))
        if sha.lower() != row["sha256"] or int(size or 0) != len(data):
            raise RuntimeError(f"{root} did not verify at {row['target']} after branching")
        row["case"] = "duplicate"  # branched and proven, so phase 2 may now drop the root copy


def migrate(apply: bool, per_commit: int, max_commits: int, kind: str = "audio") -> dict:
    api = worker.hf_api()
    index = worker.load_state(api)
    refs = worker.load_refs(api, kind)
    started_at = head_sha(api)
    tree = set(api.list_repo_files(REPO, repo_type="dataset", revision=WRITE_REVISION))
    decisions = classify(api, tree)
    # the queue download is 110 MB of shards; only worth it when a row still needs provenance
    need_urls = any(row["case"] == "copy" or (row["case"] == "duplicate"
                                              and row["sha256"] not in index["objects"])
                    for row in decisions.values())
    lookup = url_lookup(api) if need_urls else {}
    summary = Counter(row["case"] for row in decisions.values())
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "root_media_files": len(decisions), "cases": dict(summary),
              "head_commit_at_plan": started_at, "applied": apply, "batches": []}
    if not apply:
        report["unexplained"] = sorted(root for root, row in decisions.items()
                                       if row["case"] in ("unreadable", "conflict"))[:20]
        report["copies_needed"] = sum(1 for row in decisions.values() if row["case"] == "copy")
        return report

    budget = worker.Budget(api, max_per_run=max_commits, min_interval=0.0, min_remaining=0)
    allowed = {started_at}
    guard_single_writer(api, allowed)

    # phase 1: branch the root copies that have no SHA-bucketed object yet
    branch_copies(api, index, refs, decisions, lookup, kind, budget, report, allowed)

    # phase 2: delete root copies whose branched object is present, matching and referenced
    deletable = sorted(root for root, row in decisions.items() if row["case"] == "duplicate")
    for start in range(0, len(deletable), per_commit):
        if budget.used >= max_commits:
            report["stopped_because"] = f"max_commits {max_commits} for this invocation"
            break
        batch = deletable[start:start + per_commit]
        guard_single_writer(api, allowed)
        recheck = worker.remote_oids(api, [decisions[root]["target"] for root in batch])
        cleared = []
        for root in batch:
            row = decisions[root]
            sha, size = recheck.get(row["target"], ("", 0))
            if sha.lower() != row["sha256"] or (size and int(size) != row["bytes"]):
                report["batches"].append({"kept": root, "reason": f"{row['target']} no longer matches"})
                continue
            if row["sha256"] not in index["objects"]:
                # branched bytes exist but the index never learned them: adopt before deleting
                name = Path(root).stem
                item_id, url = lookup.get(name, ("", ""))
                if not url:
                    report["batches"].append({"kept": root, "reason": "untraceable url"})
                    continue
                record_object(index, refs, row["sha256"], row["target"], row["bytes"],
                              "audio/mpeg", url, item_id, kind)
            cleared.append(root)
        if not cleared:
            continue
        report["deleted_total"] = report.get("deleted_total", 0) + len(cleared)
        report.update({"root_media_files_remaining": len(deletable) - (start + len(cleared)),
                       "deleted_in_this_batch": len(cleared), "commits_used": budget.used + 1})
        ops = [CommitOperationDelete(path_in_repo=root) for root in cleared] + state_ops(
            api, index, refs, kind, budget, report, full=False)
        commit = api.create_commit(repo_id=REPO, repo_type="dataset", revision=WRITE_REVISION,
                                   commit_message=f"migrate root media: delete {len(cleared)} "
                                                  f"flat copies + resume state",
                                   operations=ops)
        allowed.add(commit_sha(commit))
        budget.record(len(cleared) + 4, 0, {"commit": getattr(commit, "commit_url", ""),
                                            "deleted": len(cleared)})
        report["batches"].append({"deleted": len(cleared), "first": cleared[0],
                                  "commit": getattr(commit, "commit_url", "")[:60]})
        log(f"deleted {len(cleared)} root copies, {len(deletable) - start - len(cleared)} to go")
    left = len(deletable) - report.get("deleted_total", 0)
    if left == 0 and report.get("deleted_total") and budget.used < max_commits:
        # the migration's last commit carries the full state, so the dataset ends the phase
        # with index, refs, checkpoint and report all describing the branched layout
        guard_single_writer(api, allowed)
        report["root_media_files_remaining"] = 0
        commit = api.create_commit(repo_id=REPO, repo_type="dataset", revision=WRITE_REVISION,
                                   commit_message="migrate root media: full resume state",
                                   operations=state_ops(api, index, refs, kind, budget, report))
        allowed.add(commit_sha(commit))
        budget.record(4, 0, {"commit": getattr(commit, "commit_url", ""), "flush": "full state"})
        report["migrated_complete"] = True
    report["commits_this_invocation"] = budget.used
    return report


def audit(api, index: dict, refs: dict) -> dict:
    """The five numbers that decide whether the migration is finished."""
    tree = set(api.list_repo_files(REPO, repo_type="dataset", revision=WRITE_REVISION))
    root_media = sorted(path for path in tree if is_media(path))
    fanout = Counter(str(PurePosixPath(path).parent) if "/" in path else "/"
                     for path in tree)
    over = {name: count for name, count in fanout.items() if count >= 10000}
    missing = [digest for digest, obj in index["objects"].items() if obj["object"] not in tree]
    mismatch = 0
    paths = sorted(obj["object"] for obj in index["objects"].values())
    for start in range(0, len(paths), 500):
        chunk = paths[start:start + 500]
        remote = worker.remote_oids(api, chunk)
        for path in chunk:
            digest = Path(path).stem
            expect = index["objects"][digest]
            sha, size = remote.get(path, ("", 0))
            if sha.lower() != digest or int(size or 0) != int(expect["bytes"]):
                mismatch += 1
        time.sleep(0.5)
    ref_targets = {row["sha256"] for entry in refs.values() for row in entry}
    dangling = sorted(ref_targets - set(index["objects"]))
    unbucketed = [path for path in tree
                  if path.startswith("media/") and worker.misplaced(path)]
    return {"root_media_files": len(root_media),
            "folders_over_limit": len(over), "largest_folder": fanout.most_common(3),
            "missing": len(missing), "checksum_mismatch": mismatch,
            "dangling_refs": len(dangling), "media_paths_not_bucketed": len(unbucketed),
            "objects_indexed": len(index["objects"]),
            "clean": (not root_media and not over and not missing and not mismatch
                      and not dangling and not unbucketed)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--per-commit", type=int, default=500)
    parser.add_argument("--max-commits", type=int, default=25)
    parser.add_argument("--kind", default="audio")
    args = parser.parse_args()
    api = worker.hf_api()
    report = migrate(args.apply, args.per_commit, args.max_commits, args.kind)
    print(json.dumps({key: value for key, value in report.items() if key != "batches"},
                     ensure_ascii=False, indent=1))
    print(json.dumps(report["batches"][-6:], ensure_ascii=False, indent=1))
    if args.apply:
        index = worker.load_state(api)
        refs = worker.load_refs(api, args.kind)
        numbers = audit(api, index, refs)
        report["audit"] = numbers
        print(json.dumps(numbers, ensure_ascii=False, indent=1))
        return 0 if numbers["clean"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
