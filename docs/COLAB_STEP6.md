# Step 6 — the one Colab cell

Paste this into a single Colab cell. **Re-run it after any disconnect, at any
point.** It reports where things stand, skips what is done, resumes what is
partial, and continues. Running it twice in a row is safe; running it after a
completed run does nothing.

```python
# ── Step 6: resilient runner. Re-run this exact cell after any disconnect. ──
import os, subprocess, sys, importlib

REPO_DIR = "/content/indic-fake-voice-detection"
REPO_URL = "https://github.com/Shruti-sivaramakrishnan/indic-fake-voice-detection.git"
BRANCH   = "claude/ml-repo-structure-review-fpox8u"

# 1. Repo: clone if absent, fast-forward if present.
if os.path.exists(os.path.join(REPO_DIR, "model_pipeline.py")):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", BRANCH],
                   check=False)
else:
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR],
                   check=True)

# 2. Import path. Drop any stale module first — a Colab kernel that imported
#    `model_pipeline` before the file existed caches a namespace package and
#    every later import fails with "(unknown location)".
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
sys.modules.pop("model_pipeline", None)

import model_pipeline
importlib.reload(model_pipeline)

# 3. One call: mount Drive, report status, build/resume cache, train/resume,
#    calibrate, write docs/STEP6_RESULTS.md.
summary = model_pipeline.run_step6_resilient()
```

## What it does, in order

1. **Mounts Google Drive**, retrying once with `force_remount`. Refuses to
   continue without it — Colab's local disk is wiped on disconnect, so a cache
   or checkpoint under `/content` is gone exactly when resume matters. Pass
   `require_drive=False` to override.
2. **Clones or fast-forwards the repo.**
3. **Prints a status report** — cache percentage per split, epochs completed,
   best EER so far, checkpoint timestamps.
4. **Builds or resumes the feature cache** (skips clips already cached).
5. **Trains or resumes training** from `last_model.pt`.
6. **Calibrates** temperature on validation logits.
7. **Writes `docs/STEP6_RESULTS.md`** and prints the git commands to commit it.

All state lives at
`/content/drive/MyDrive/indic-fake-voice-detection/{feature_cache,checkpoints}`.

## Just check status, don't run anything

```python
model_pipeline.run_step6_resilient(status_only=True)
```

## Useful variations

```python
# Cheap end-to-end rehearsal (minutes, not hours). Not reportable.
model_pipeline.run_step6_resilient(max_clips_per_split=200, shard_size=50)

# Bigger batches if the timing breakdown shows extraction dominating on GPU.
model_pipeline.run_step6_resilient(extract_batch_size=16)

# Train longer: raise num_epochs and re-run — it continues, it does not restart.
from model_pipeline import TrainConfig
model_pipeline.run_step6_resilient(config=TrainConfig(num_epochs=25))
```

## Reading the status report

```
FEATURE CACHE
  train      : 1,688 cached / 24,881 expected  ->    6.8% partial
  validation : 212 cached / 3,110 expected     ->    6.8% partial

TRAINING
  no checkpoint — training has not started (0 / 15)
```

`COMPLETE` on a split means every clip is accounted for — cached **or** logged
as a decode failure. The manifest asserts
`num_clips + num_failed == expected_clips`, and training refuses to start on a
cache that does not add up or that lost more than 1% of clips.

## If something looks wrong

| Symptom | What it means |
|---|---|
| `COULD NOT MOUNT GOOGLE DRIVE` | Authorize the popup, re-run the cell. |
| `LOCAL DISK, WIPED ON DISCONNECT` | Running with `require_drive=False`. State will not survive. |
| `REHEARSAL cap=200` | The cache is from a capped rehearsal. Delete the cache dir to build the real one. |
| `Checkpoint ... was trained with hidden_dim=X` | The config changed since the checkpoint. Delete `checkpoints/` or restore the old config. |
| `Training already complete` | Nothing to do. Raise `num_epochs` to continue training. |

## Caveat on resume

RNG state is **not** restored across a resume, so a resumed run is not
bit-identical to an uninterrupted one. Per-epoch shuffling stays deterministic
(the sampler is seeded by epoch number), and model/optimizer/loss-history/best-EER
state all carry over correctly — but if you need an exactly reproducible run for
the report, run it start-to-finish without interruption.
