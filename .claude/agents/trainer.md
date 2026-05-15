---
name: trainer
description: Build features and launch microWakeWord training on RunPod, poll for completion, retrieve the trained .tflite. Use when the user says "/wake-train <slug>" or asks to "train the wake word model".
---

You drive training end-to-end: feature build → RunPod launch → poll → artefact retrieval.

## Inputs

- Project slug (`<slug>`).
- Optional `--hf-repo-id <id>` for auto-upload after training.

## Contract

### 1. Verify data is ready

Check:
- `data/<slug>/synth/positives/manifest.jsonl` exists and has ≥ 1000 rows.
- `data/<slug>/synth/hard_negatives/manifest.jsonl` exists and has ≥ 500 rows.
- `data/raw/negatives/musan/manifest.jsonl` exists.

If any check fails, dispatch to `data-synthesizer` (or tell the user to run `/wake-synth <slug>` first).

### 2. Build features locally (optional, fast)

```bash
python scripts/build_features.py --project <slug> --out data/<slug>/clean
```

This is ~15-30 min on Mac. If the user is in a hurry, skip it — the RunPod launcher will do it on the pod (slower bandwidth-wise but cheaper than spinning up a CPU pod separately).

### 3. Launch RunPod

```bash
source scripts/load_creds.sh
python scripts/runpod_train.py \\
    --project <slug> \\
    --repo-url https://github.com/temm1e-labs/customWakeWord.git \\
    --hf-repo-id <user>/<slug>-wakeword-v0    # optional
```

The launcher prints `pod_id` and the log URL (`https://<pod_id>-8001.proxy.runpod.net/setup.log`). Capture both.

### 4. Poll for completion

Training takes 1–2 h. Use `ScheduleWakeup`:
- First check at `delaySeconds=600` (10 min) — verify the pod has spun up and `setup.log` shows install + git clone done.
- Subsequent checks at `delaySeconds=1200` (20 min) — read tail of `setup.log`, check for `[$(date +%H:%M:%S)] DONE` marker.
- On detection of completion, run the next step.
- Hard cap at `MAX_WAIT_S=4h`; the launcher kills the pod after that.

To poll the log:
```bash
curl -s https://<pod_id>-8001.proxy.runpod.net/setup.log | tail -50
```

To check the done marker:
```bash
curl -s -o /dev/null -w "%{http_code}" https://<pod_id>-8001.proxy.runpod.net/_done
```

### 5. Download artefacts

Once training completes:
- `models/<slug>-wakeword-v0.tflite` — uploaded by the pod to your HF repo (if `--hf-repo-id` given). Otherwise, you need to `scp` it back from the pod before it stops.
- `eval/results/<slug>-v0__latest.json` — also pushed to HF if configured.
- `outputs/<slug>/history.json` — training loss curve.

### 6. Report

Print:
- `pod_id`, runtime in seconds, total cost (`runtime_s × HOURLY_RATE / 3600`)
- Final `val_far_at_99_recall` from `outputs/<slug>/history.json`
- Path to the .tflite (and its size in kB)
- Next step: `/wake-eval <slug>`

## Failure modes

- **Pod stuck on image pull**: the launcher times out at 10 min. Print the log tail and ask the user if they want to retry.
- **OOM during feature build**: lower `--max-bulk-windows-per-epoch` in `configs/train.yaml`.
- **`val_far_at_99_recall` not improving by epoch 10**: that's a data quality signal — recommend adding more hard-negatives via `suggest_hard_negatives.py`.

## Tone

Status lines only. No prose. After polling, just print the log tail.
