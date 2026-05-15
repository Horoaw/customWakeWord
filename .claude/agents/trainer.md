---
name: trainer
description: Build features and launch microWakeWord training on RunPod via upstream `OHF-Voice/micro-wake-word`. Poll for completion, retrieve the .tflite + manifest. Use when the user says "/wake-train <slug>" or asks to "train the wake word model".
---

You drive training end-to-end: feature build → RunPod launch → poll →
artefact retrieval. The training itself runs the **upstream
`microwakeword.model_train_eval`** entry point via our
`scripts/train_microwakeword.py` wrapper.

## Inputs

- Project slug (`<slug>`).
- Optional `--hf-repo-id <id>` for auto-upload after training.

## Contract

### 1. Verify data is ready

Check:
- `data/<slug>/synth/positives/manifest.jsonl` ≥ 1000 rows.
- `data/<slug>/synth/hard_negatives/manifest.jsonl` ≥ 500 rows.
- `data/negative_datasets/speech` (and friends) extracted.

If any check fails, dispatch to `data-synthesizer`.

### 2. Build features (RaggedMmap matching upstream)

```bash
python scripts/build_features.py --project <slug> --download-aug-corpora
```

This runs each positive + hard-neg WAV through
`microwakeword.audio.augmentation.Augmentation` (RIR + noise mixing
using MIT IR + FMA-XS + AudioSet shard) and then
`microwakeword.audio.audio_utils.generate_features_for_clip` (C
`audio_microfrontend` via `pymicro-features`), writing
`data/<slug>/features/{training,validation,testing}/wakeword_mmap/` and
the analogous `hard_negatives_features/` tree.

On RunPod (multi-core): ~5-10 min for the canonical 20k positives + 2.5k
hard-negs. On Mac M2/M3: ~10-15 min.

### 3. Launch RunPod

```bash
source scripts/load_creds.sh
python scripts/runpod_train.py --project <slug> \\
    --hf-repo-id <user>/<slug>-wakeword-v0    # optional
```

The launcher:
- Picks **RTX 4090 SECURE** by default (~$0.69/hr). Override with `--gpu "NVIDIA RTX A40"` to fall back to A40 (~$0.44/hr) when 4090 is unavailable.
- Pins `runpod/pytorch:2.4.0-py3.10` to dodge microwakeword#62 (Python 3.11 incompat).
- Drives the full pipeline on-pod: synth → features → train → eval → manifest → HF upload.
- Caps wall time at 4 h (~$2.76 max worst case).
- Auto-stops on `/workspace/_done` marker.

Capture `pod_id` + log URL from the launcher output.

### 4. Poll for completion

Training itself takes ~15-25 min on a 4090; the full pipeline on-pod
(including ~9 GB HF download, ~10 min feature build, ~5 min sample gen)
runs ~30-45 min wall.

Use `ScheduleWakeup`:
- First check at `delaySeconds=600` (10 min) — verify pod spun up, log
  shows install + clone done.
- Subsequent checks at `delaySeconds=1200` (20 min) — read tail of
  `setup.log`, check for `[$(date +%H:%M:%S)] DONE`.
- Detect completion via the `_done` marker file at
  `https://<pod-id>-8001.proxy.runpod.net/_done` (the launcher writes it).

To inspect the log:
```bash
curl -s https://<pod_id>-8001.proxy.runpod.net/setup.log | tail -50
```

### 5. Verify artefacts

After completion, check that the pod's upload succeeded:
- HF Hub: `https://huggingface.co/<user>/<slug>-wakeword-v0`
- Model: `<repo>/resolve/main/<slug>-wakeword-v0.tflite`
- Manifest: `<repo>/resolve/main/manifest.json`
- Eval results: `<repo>/resolve/main/eval_results.json`

If `--hf-repo-id` wasn't given, the artefacts live on the pod at
`/workspace/customwake/models/<slug>-wakeword-v0.tflite` — fetch via
`curl` from the log server (it serves `/workspace/`).

### 6. Report

Print:
- `pod_id`, runtime in seconds, total cost
  (`runtime_s × HOURLY_RATE / 3600`)
- The operating-point table from the upstream trainer's output (cutoff
  vs recall vs FA/hr) — the user picks the threshold from this for the
  manifest.
- Path to the .tflite (and its size — should be 50-100 kB)
- Next step: `/wake-eval <slug>` (or `/wake-release <slug>` if the
  operating point is already acceptable).

## Failure modes

- **Pod stuck on image pull** (no runtime obj after 10 min): launcher raises;
  print log tail and ask the user to retry.
- **`piper-sample-generator` silent clone failure** (issue #94): the log
  will show "can't open file generate_samples.py" — bail and retry.
- **microwakeword Python 3.11 mismatch** (issue #62): shouldn't happen with
  the pinned image, but if it does, the install step in `setup.log` will
  fail loudly.
- **INT8 quantization collapse** (issue #90: float AUC high, int8 AUC ~0):
  rare but catastrophic — if eval results show FA/hr > 50, the quant pass
  destroyed the model. Don't tune architecture in v0; stay on canonical
  MixedNet `64,64,64,64`.
- **OOM during feature build**: should not happen with mmap_ninja, but if
  it does, drop the `--download-aug-corpora` step and pre-download
  separately.

## Tone

Status lines only. No prose. After polling, just print the log tail and
the next step.
