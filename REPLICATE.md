# REPLICATE — End-to-end reproduction

Step-by-step replication for the Tofu wake-word. For any other slug, substitute `--project <slug>` and `configs/examples/<slug>/` paths.

**Total cost**: ~$0.55–3 in RunPod credit. **Wall time**: ~3 h total, ~45 min active.

---

## 0. Prerequisites

- macOS or Linux, **Python 3.10** (microwakeword pins it)
- Homebrew (Mac): `brew install python@3.10 git git-lfs ffmpeg sox`
- RunPod account with **≥ $5 credit** and an API key
- HuggingFace account + **write** token
- GitHub PAT (for release; or omit and ship to HF only)

---

## 1. Credentials

```bash
mkdir -p ~/.config/tofu-wake && chmod 700 ~/.config/tofu-wake
cat > ~/.config/tofu-wake/hf.env << 'EOF'
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
EOF
cat > ~/.config/tofu-wake/runpod.env << 'EOF'
export RUNPOD_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
cat > ~/.config/tofu-wake/gh.env << 'EOF'
export GH_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
cat > ~/.config/tofu-wake/together.env << 'EOF'
export TOGETHER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx  # optional, for suggest_hard_negatives.py
EOF
chmod 600 ~/.config/tofu-wake/*.env
```

Test:
```bash
source scripts/load_creds.sh
```

---

## 2. Local Python env

```bash
git clone https://github.com/temm1e-labs/customWakeWord && cd customWakeWord
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs `microwakeword` from upstream (`OHF-Voice/micro-wake-word`), TensorFlow, `pymicro-features`, `mmap-ninja`, `audiomentations`, `piper-phonemize-cross`, and friends.

Validate:
```bash
python -c "import microwakeword; print(microwakeword.__file__)"
```

---

## 3. Confirm Tofu project is initialized

```bash
ls configs/examples/tofu/
# wake_phrases.yaml  hard_negatives.yaml  training_parameters.yaml  README.md
```

To re-bootstrap from scratch:
```bash
python scripts/init_wake.py --name tofu \
    --phrases "hey tofu,hi tofu,hello tofu,okay tofu" --force
```

Optional LLM-driven hard-neg extension:
```bash
source scripts/load_creds.sh
python scripts/suggest_hard_negatives.py --name tofu  # ~$0.001
```

---

## 4. One-shot RunPod pipeline (recommended)

This runs every stage on the pod: synth → bulk download → features → train → eval → manifest → HF upload.

```bash
source scripts/load_creds.sh
python scripts/runpod_train.py --project tofu \
    --hf-repo-id <your_username>/tofu-wakeword-v0
```

The launcher prints `pod_id` + log URL on `:8001/setup.log`. Wall time: 30–45 min. Cost: ~$0.50 on 4090 SECURE.

Tail the log mid-run:
```bash
curl -s https://<pod_id>-8001.proxy.runpod.net/setup.log | tail -50
```

Wait for `/_done` marker:
```bash
curl -s -o /dev/null -w "%{http_code}\n" https://<pod_id>-8001.proxy.runpod.net/_done
# 200 = done, 404 = still running
```

---

## 4-alt. Run stages manually (debugging or offline)

If you want to inspect each step or you don't have a RunPod account yet:

### 4a. Synth positives (Mac MPS ~30-60 min; pod 4090 ~5 min)

```bash
python scripts/synth_positives.py --project tofu
```

Watch `data/tofu/synth/positives/manifest.jsonl` grow. Resumable: re-run picks up where it left off.

### 4b. Synth hard-negs

```bash
python scripts/synth_hard_negatives.py --project tofu
```

### 4c. Download upstream HF negatives (~9 GB, ~3 min on a fast pipe)

```bash
python scripts/download_hf_negatives.py --out data/negative_datasets
```

### 4d. Feature extraction

```bash
python scripts/build_features.py --project tofu --download-aug-corpora
```

### 4e. Train + INT8 export

```bash
python scripts/train_microwakeword.py --project tofu \
    --training-config configs/examples/tofu/training_parameters.yaml
```

Output:
- `trained_models/tofu/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`
- `models/tofu-wakeword-v0.tflite` (copied)
- operating-point table printed to stdout

---

## 5. Eval against held-out tasks

```bash
python scripts/seed_eval_tasks.py --project tofu  # optional; samples from training manifest
python -m eval.runner --project tofu \
    --model models/tofu-wakeword-v0.tflite \
    --threshold 0.85 \
    --out eval/results/tofu-v0__$(date +%s).json
```

Output:
- FRR on positives
- FAR/hr on bulk
- per-hard-neg-bucket FAR
- ROC across detection thresholds

If FRR > 5% or FAR/hr > 1.0, see iteration playbook in [`TRAINING_PLAN.md`](TRAINING_PLAN.md) §7.

---

## 6. Emit ESPHome manifest

```bash
python scripts/emit_manifest.py --project tofu --threshold 0.85
# writes configs/examples/tofu/manifest.json
```

If you let `--threshold` default and have an eval JSON, it auto-picks the operating point.

---

## 7. Publish to HuggingFace

```bash
python scripts/upload_to_hf.py --project tofu \
    --model models/tofu-wakeword-v0.tflite \
    --repo-id <your_username>/tofu-wakeword-v0 \
    --eval-json eval/results/tofu-v0__<latest>.json \
    --esphome configs/examples/tofu/manifest.json
```

Pushes: `tofu-wakeword-v0.tflite`, `manifest.json`, `eval_results.json`, `README.md` (auto-generated model card).

---

## 8. GitHub release

```bash
gh release create v0.1.0 \
    models/tofu-wakeword-v0.tflite \
    configs/examples/tofu/manifest.json \
    eval/results/tofu-v0__<latest>.json \
    --title "tofuWakeWord v0.1.0" \
    --notes "FRR=X.X% FAR=Y.Y/hr. See model card on HF."
```

---

## 9. Flash to ESP32-S3

Minimal ESPHome config for an ESP32-S3-DevKitC-1 + INMP441 I2S mic:

```yaml
esphome:
  name: tofu
esp32:
  board: esp32-s3-devkitc-1
  framework: { type: esp-idf }

i2s_audio:
  i2s_lrclk_pin: GPIO5
  i2s_bclk_pin: GPIO6

microphone:
  - platform: i2s_audio
    id: tofu_mic
    i2s_din_pin: GPIO7
    pdm: false

micro_wake_word:
  microphone: tofu_mic
  vad:
  models:
    - model: https://huggingface.co/<your_username>/tofu-wakeword-v0/resolve/main/manifest.json
  on_wake_word_detected:
    - logger.log:
        format: "wake word: %s"
        args: ['wake_word.c_str()']
```

```bash
esphome run tofu.yaml
```

Say "hey tofu" near the mic. The ESPHome log fires. If it doesn't, lower `probability_cutoff` and re-flash. If it triggers too often, raise it.

---

## 10. With Claude Code

Each step above is also a slash command:

```
/wake-status                                # see pipeline state
/wake-new <slug> "<phrases>"                # bootstrap a new wake word
/wake-synth <slug>                          # steps 4a-c
/wake-train <slug> --hf-repo-id <user/repo> # steps 4d-7 (auto-uploads if --hf-repo-id given)
/wake-eval <slug>                           # step 5
/wake-release <slug>                        # steps 6-8
```

See [`CLAUDE.md`](CLAUDE.md) for the agent contract.

---

## Pre-flight checklist

Before running step 4, walk [`READY_TO_TRAIN.md`](READY_TO_TRAIN.md). Most failures (Python 3.11 mismatch, missing tokens, RunPod credit absent) are caught there.
