# titan-learn-worker

Cloud archiver for the Titan Learn VOA corpus. It moves the media queue off a laptop and
off the Titan Gateway: GitHub's runners pull from VOA, verify, upload to the dataset,
re-verify remotely, then purge.

## What one run does

```
PULL from VOA -> temp file -> VERIFY (magic-byte MIME + declared size + SHA-256)
  -> UPLOAD to M-hv1/Titan-Learn (one commit per batch)
  -> REMOTE VERIFY (lfs sha256 + byte size, scoped to just this batch)
  -> WRITE resume state -> PURGE temp
```

Two rules are enforced in code, not by convention:

- **Nothing is fetched twice.** `_worker/media_index.json` records each asset's status; an
  asset at `verified_remote` is skipped, and a dead one stops after 3 attempts. So the
  already-archived objects cost no bandwidth on the next run.
- **Video cannot start without a measured size sample.** `--kind video` reads
  `reports/video_probe.json` and aborts unless it holds at least 30 successful HEADs and
  the decision `video archiving may start`. That gate is what produced the 134.1 GB
  estimate before any video was pulled.

## Setup (one time)

Create this as a **public** repository. Public repos get unlimited free minutes on
`ubuntu-24.04`; a private repo would burn its 2,000 min/month on the first day of runs.
The repository contains code only - no VOA content - so publishing it exposes nothing.

```bash
# from this directory
git init -b main
git add .
git commit -m "VOA media archiver for the Titan Learn dataset"
gh repo create M-hv1/titan-learn-worker --public --source=. --push
gh secret set HF_TOKEN < ~/.cache/huggingface/token
```

Then trigger it once from the Actions tab (or `gh workflow run archive.yml`) and watch the
log: the first run should report the pinned manifest, then start committing batches.

Optional repository variables (defaults are already in the workflow):

| variable | default | meaning |
| --- | --- | --- |
| `HF_LEARN_DATASET_REPO` | `M-hv1/Titan-Learn` | where media and state are written |
| `HF_LEARN_STATE_REVISION` | `staging` | branch holding `_worker/` state |
| `HF_LEARN_RECORDS_PIN` | `c1680e48da39…` | immutable commit records are read from |

## Cadence and expected volume

Scheduled every 6 hours, up to ~145 minutes per media type, 500 audio or 300 video assets
per batch. Measured averages: audio 2.87 MB/asset over 32,344 assets (~93 GB), video
median 7.3 / mean 12.8 MB over 10,703 assets (~134 GB). Audio finishes first; video follows
in the same job once its gate is open.

## Where the truth lives

The dataset is the source of truth, not this repo and not any machine:

- records: `data/part-*.parquet` at a pinned commit
- media: `media/<sha256[0:2]>/<sha256>.<ext>`, content-addressed
- join: `_worker/<kind>_refs.jsonl` (item -> object), so a record never needs a live VOA url
- resume: `_worker/media_index.json` + `_worker/<kind>_checkpoint.json`

`worker.py` is a copy of `gw-src/voa_factory/worker.py` (the canonical source). If you edit
it here, sync it back; the local archiver and the cloud job are the same code path, which
is what makes the checkpoint portable between them.
