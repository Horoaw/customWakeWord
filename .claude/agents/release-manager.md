---
name: release-manager
description: Publish an evaluated microWakeWord model and ESPHome v2 manifest.
---

Release only a trained and held-out-evaluated model.

## Inputs

- Project slug.
- Hugging Face repo id.
- Optional `--private` flag.

## Contract

### 1. Pre-flight

Require all of the following:

- `models/<slug>-wakeword-v0.tflite`
- `eval/results/<slug>-v0__latest.json`
- `configs/examples/<slug>/wake_phrases.yaml`
- Non-zero positive and bulk-negative task counts in the eval summary

Do not substitute a default threshold when evaluation is missing.

### 2. Emit the ESPHome manifest

```bash
python scripts/emit_manifest.py \
    --project <slug> \
    --eval-json eval/results/<slug>-v0__latest.json
```

This writes `configs/examples/<slug>/manifest.json` and verifies that the model
exists.

### 3. Upload

```bash
source scripts/load_creds.sh
python scripts/upload_to_hf.py \
    --project <slug> \
    --model models/<slug>-wakeword-v0.tflite \
    --repo-id <user>/<slug>-wakeword-v0 \
    --eval-json eval/results/<slug>-v0__latest.json \
    --esphome configs/examples/<slug>/manifest.json
```

Confirm that both the model and manifest URLs return successfully.

### 4. Persist release metadata

Write `configs/examples/<slug>/release.yaml` with the repo id, measured
threshold, version, timestamp, and eval result path.

### 5. Optional GitHub release

Ask before creating a public release. Include the `.tflite`, manifest, and eval
JSON. Generate release notes from the eval summary and disclose training-data
license constraints.

### 6. Report

Return the Hugging Face URL, manifest path, FRR, FAR/hour, threshold, and any
remaining device-validation caveat.
