# REPLICATE — End-to-end reproduction

Step-by-step. Mac M-series + a RunPod account. ~$1 of compute, ~4 days of wall time, mostly idle while CPU TTS and downloads run.

---

## 0. Prerequisites

- macOS or Linux, Python 3.11
- Homebrew (Mac) — `brew install python@3.11 git git-lfs ffmpeg sox`
- RunPod account with $5+ credit and an API key
- HuggingFace account + write token
- GitHub PAT (only needed if cloning the repo onto a RunPod pod)

---

## 1. Credentials

Create `~/.config/tofu-wake/` and drop four env files (chmod 600 each):

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
export TOGETHER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx  # optional, only if using Together TTS
EOF
chmod 600 ~/.config/tofu-wake/*.env
```

Then any script can pull them with:

```bash
source scripts/load_creds.sh
```

---

## 2. Local Python env

```bash
git clone https://github.com/<you>/tofuWakeWord && cd tofuWakeWord
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(microWakeWord, audiomentations, Piper, etc.)

---

## 3. Synthesize positives (~30 min on Mac M-series CPU)

```bash
python scripts/synth_positives.py \
    --phrases configs/wake_phrases.yaml \
    --out data/synth/positives \
    --count 10000
```

Watch `data/synth/positives/manifest.jsonl` grow. Each row: `{voice_id, phrase, speed, wav_path, duration_s}`. Stop early any time — the next step picks up wherever this leaves off.

Sanity check:

```bash
wc -l data/synth/positives/manifest.jsonl   # should hit ~10000
ls data/synth/positives/*.wav | head -5     # eyeball a few
afplay data/synth/positives/$(ls data/synth/positives | grep wav | head -1)
```

---

## 4. Synthesize hard negatives (~5 min)

```bash
python scripts/synth_hard_negatives.py \
    --phrases configs/hard_negatives.yaml \
    --out data/synth/hard_negatives \
    --count 2500
```

---

## 5. Download bulk corpora (~30 GB, ~1 h)

```bash
python scripts/collect_negatives.py \
    --out data/raw/negatives \
    --corpora musan,demand,commonvoice,audioset_subset,librispeech \
    --hours 300
```

Outputs:
- `data/raw/negatives/musan/`
- `data/raw/negatives/demand/`
- `data/raw/negatives/commonvoice/`
- `data/raw/negatives/audioset/`
- `data/raw/negatives/librispeech/`
- `data/raw/negatives/rirs/`  (also pulls OpenSLR-28)
- `data/raw/negatives/manifest.jsonl`

If you're on slow internet, run this on a RunPod CPU-only pod and `rsync` the result back.

---

## 6. Feature extraction + splits (~30 min on Mac)

```bash
python scripts/build_features.py \
    --positives data/synth/positives \
    --hard-negatives data/synth/hard_negatives \
    --bulk-negatives data/raw/negatives \
    --rir data/raw/negatives/rirs \
    --noise "data/raw/negatives/musan/noise,data/raw/negatives/demand" \
    --out data/clean \
    --train-reps 5 \
    --positive-oversample 3 \
    --seed 42
```

Produces:
- `data/clean/train.tfrecord`
- `data/clean/val.tfrecord`
- `data/clean/test.tfrecord`
- `data/clean/meta.json`

The split is deterministic on `--seed`; the test set is identified by `meta.json["test_ids"]` so you can audit any held-out sample by id.

---

## 7. Hand-curate the eval set

Inspect 50 random positives + 50 random hard-negs from the test split. Move any obvious failures (silent files, completely wrong transcripts) to `eval/tasks/`:

```bash
python scripts/seed_eval_tasks.py \
    --from data/clean/test.tfrecord \
    --positives 50 \
    --hard-negatives 50 \
    --bulk-stream-minutes 60 \
    --out eval/tasks
```

This is the **only manual step** in the pipeline. Plan for 30 min of listening.

---

## 8. Train on RunPod (~$0.40–0.80, ~1–2 h)

```bash
source scripts/load_creds.sh
python scripts/runpod_train.py
```

The launcher prints:
- `pod_id`
- API proxy URL on `:8000` (used by eval if you want to score against the float model before quant)
- Log server URL on `:8001/setup.log` — tail this anytime

It auto-detects when training + INT8 export finishes and stops the pod. If anything goes wrong, the pod stays alive long enough for you to inspect `setup.log` (capped at MAX_WAIT_S = 4 h to avoid runaway costs).

The merged `.tflite` is uploaded back to your HuggingFace repo (configured in `configs/train.yaml:hub_model_id`).

---

## 9. Eval

```bash
python -m eval.runner \
    --model models/tofu-wakeword-v0.tflite \
    --tasks eval/tasks \
    --out eval/results/tofu-v0__$(date +%s).json
```

Outputs per-task results + summary metrics: FRR, FAR/hour, per-hard-neg-bucket FAR, ROC across detection thresholds.

If the metrics miss the targets in [`PLAN.md`](PLAN.md) §6, the per-bucket breakdown points to the failure mode:
- High FRR + low FAR → more positive diversity (add accents, emotions, speeds)
- Low FRR + high FAR on a specific hard-neg bucket → add more of that bucket
- Both bad → augmentation is too aggressive, or representative dataset for INT8 quant was bad → check `data/clean/meta.json` for the representative samples

---

## 10. Publish

```bash
python scripts/upload_to_hf.py \
    --model models/tofu-wakeword-v0.tflite \
    --repo-id <your_username>/tofu-wakeword-v0 \
    --eval-json eval/results/tofu-v0__<latest>.json \
    --esphome configs/esphome_tofu.yaml
```

Then on GitHub:

```bash
gh release create v0.1.0 \
    models/tofu-wakeword-v0.tflite \
    configs/esphome_tofu.yaml \
    eval/results/tofu-v0__<latest>.json \
    --title "tofuWakeWord v0.1.0" \
    --notes "First trained Tofu wake-word. FRR=X.X% FAR=Y.Y/hr. See model card."
```

---

## 11. Flash to ESP32-S3

(Outside this repo — but here's the minimal ESPHome config you need.)

`tofu.yaml`:
```yaml
esphome:
  name: tofu
esp32:
  board: esp32-s3-devkitc-1
  framework:
    type: esp-idf

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
    - model: hey_tofu
      probability_cutoff: 0.85
      sliding_window_size: 5
      url: https://huggingface.co/<your_username>/tofu-wakeword-v0/resolve/main/tofu-wakeword-v0.tflite
  on_wake_word_detected:
    - logger.log:
        format: "wake word: %s"
        args: ['wake_word.c_str()']
```

```bash
esphome run tofu.yaml
```

Say "hey tofu" near the mic. The log should fire. If it doesn't, lower `probability_cutoff` and re-flash.

---

## 12. With Claude Code

Each step above is also a slash command in `.claude/commands/`. The agentic flow:

```
/tofu-status        # see where the pipeline is
/tofu-synth         # steps 3–5 (synthesis + bulk download)
/tofu-train         # steps 6–8 (feature build + RunPod train)
/tofu-eval          # step 9
/tofu-release       # step 10
```

The agents handle credential sourcing, polling RunPod via ScheduleWakeup, generating the model card from the eval JSON, etc. See [`CLAUDE.md`](CLAUDE.md) for the agent contract.
