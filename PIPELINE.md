# PIPELINE — customWakeWord Technical Execution

The exact technical recipe. Every command, every config, every file path. Designed so Claude Code executes autonomously and a future maintainer can reproduce.

This doc walks the **Tofu worked example** end-to-end. For *any* wake word, substitute the slug — every command below works with `--project <slug>` or `configs/examples/<slug>/...` interchangeably. See [`README.md`](README.md) for the generic-toolkit framing and [`EXAMPLES.md`](EXAMPLES.md) for other worked examples (Sunny, Jarvis).

---

## Stage 0 — Local Mac Bootstrap

```bash
# Tools
brew install python@3.11 git git-lfs ffmpeg sox

# Python env
cd ~/Documents/Github/tofuWakeWord
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Piper binary + voices (used by synth_positives.py)
# Mac: https://github.com/rhasspy/piper/releases — download piper_macos_aarch64.tar.gz
# Linux: pip install piper-tts is fine; on Mac, the precompiled binary is faster.
mkdir -p ~/.local/share/piper && cd ~/.local/share/piper
# (download + extract the appropriate release here)
```

Local Mac runs: TTS synthesis, augmentation, eval, ESPHome YAML emit, GGUF-style export sanity checks.
Local Mac does NOT run: full training (use RunPod), AudioSet/Common Voice downloads (use a RunPod CPU-only pod for bandwidth).

---

## Stage 1 — Data Pipeline

### 1a. Positive synthesis

Script: `scripts/synth_positives.py`
- Reads `configs/wake_phrases.yaml` — defines the wake phrases + per-phrase weight + augmentation knobs.
- Iterates over registered TTS backends (Piper, Kokoro, MeloTTS, Parler-TTS) and their voices.
- For each (voice × phrase × speed × emotion) tuple, synthesizes one 16 kHz mono WAV.
- Output: `data/synth/positives/<voice>__<phrase_slug>__<speed>__<seed>.wav` + `manifest.jsonl` row per file.

```bash
python scripts/synth_positives.py \
    --phrases configs/wake_phrases.yaml \
    --out data/synth/positives \
    --count 10000 \
    --engines piper,kokoro,melotts,parler
```

Target counts (see `configs/wake_phrases.yaml`):
- "hey tofu" — 5000
- "hi tofu" — 2500
- "hello tofu" — 1500
- "okay tofu" — 1000

### 1b. Hard-negative synthesis

Script: `scripts/synth_hard_negatives.py`
- Reads `configs/hard_negatives.yaml` — defines collision phrases.
- Same TTS pool, same speeds, same voices.
- Output: `data/synth/hard_negatives/<voice>__<phrase_slug>__<speed>__<seed>.wav`.

Adversarial phrases (target ~2500 total):
- "hey doofus", "hi doofus" — `oo` rhyme
- "to-do list", "to-do", "to-go" — disyllabic /tu-/
- "two pugs", "two pups", "tutu"
- "hey [name]" with common 2-syllable names (tofu confused with proper nouns)
- "tofu" alone (no greeting) — must NOT fire
- Cooking sentences: "I love tofu", "this tofu is good"
- "toad", "ufo", "tofu burger"
- "phew", "boo who", "go to"

### 1c. Bulk-negative collection

Script: `scripts/collect_negatives.py`
- Downloads / mirrors:
  - **MUSAN** (~11 GB) — music + speech + noise
  - **DEMAND** (~10 GB, 48 kHz) — real-room noise, 18 environments
  - **Common Voice** English subset (~50 GB) — sampled to ~50 h
  - **AudioSet** unbalanced segments matching `Speech`, `Music`, `Household sounds`, `Vehicle`
  - **LibriSpeech** `test-clean` (~360 MB) — held-out for eval FAR
- Output: `data/raw/negatives/<corpus>/...` + per-corpus `manifest.jsonl`.

```bash
python scripts/collect_negatives.py --out data/raw/negatives \
    --corpora musan,demand,commonvoice,audioset_subset,librispeech
```

### 1d. RIR + noise corpora

Script: included in `collect_negatives.py` with `--rirs`.
- **OpenSLR-28 RIRS_NOISES** (~1 GB) — real + simulated room impulse responses.
- **MIT McDermott IR Survey** (~50 MB) — ~270 real IRs covering churches, halls, bathrooms.

### 1e. Augment + feature extraction

Script: `scripts/build_features.py`
- For each WAV (positive or negative):
  1. Random crop to 1.5 s window
  2. Apply `audiomentations` chain:
     - `RoomSimulator` OR `ApplyImpulseResponse` (50% each, p=0.7)
     - `AddBackgroundNoise` from MUSAN/DEMAND at SNR ∈ [-5, +25] dB
     - `PitchShift` ±2 semitones (p=0.3)
     - `TimeStretch` 0.85–1.15× (p=0.3)
     - `Mp3Compression` 32–96 kbps OR `Opus` 8–24 kbps (p=0.4)
     - `Gain` ±10 dB
     - `BandPassFilter` 200–7800 Hz (cheap mic sim, p=0.5)
  3. Compute 40-bin log-mel features at 25 ms hop, 60 ms window
  4. Quantize to INT8 (microWakeWord's expected input format)
- Splits:
  - **train**: 80% positives + 80% hard-negs + 80% bulk-neg samples × 5 augmentation reps
  - **val**: 10% positives + 10% hard-negs + 10% bulk × 1 rep (no augmentation)
  - **test**: 10% positives + 10% hard-negs + 10% bulk × 1 rep (no augmentation) — **HELD OUT, never touched during training**
- Output: `data/clean/{train,val,test}.tfrecord` + `data/clean/meta.json`.

```bash
python scripts/build_features.py \
    --positives data/synth/positives \
    --hard-negatives data/synth/hard_negatives \
    --bulk-negatives data/raw/negatives \
    --rir data/raw/negatives/rirs \
    --noise data/raw/negatives/musan/noise,data/raw/negatives/demand \
    --out data/clean \
    --train-reps 5 \
    --positive-oversample 3
```

---

## Stage 2 — Training

### 2a. Local sanity (optional, Mac M-series)

For pipeline validation only — Mac CPU can train ~100 steps in a few minutes to confirm the data path is wired correctly. Do **not** expect a usable model from this.

```bash
python scripts/train_microwakeword.py \
    --data data/clean \
    --config configs/train.yaml \
    --out models/tofu-wakeword-smoke.tflite \
    --max-steps 100 \
    --skip-quant
```

### 2b. RunPod training

Script: `scripts/runpod_train.py`

GPU: NVIDIA A40 48GB SECURE @ ~$0.39/hr (or 4090 @ ~$0.34/hr — A40 is more reliable for TF). Wall time: 1–2 h.

```bash
source scripts/load_creds.sh
python scripts/runpod_train.py
```

The launcher:
1. Clones this repo onto the pod
2. Starts a `python3 -m http.server 8001` on `/workspace/` so logs are inspectable mid-run via `https://<pod-id>-8001.proxy.runpod.net/setup.log`
3. Runs `scripts/train_microwakeword.py` with `configs/train.yaml`
4. On completion, runs `scripts/export_tflite.py` to INT8-quantize for ESP32-S3 deployment
5. Uploads `tofu-wakeword-v0.tflite` to HuggingFace Hub
6. Stops the pod

Training config (`configs/train.yaml`):
```yaml
model:
  architecture: micro_wake_word_inception
  features:
    n_mels: 40
    win_ms: 60
    hop_ms: 25
    n_features: 194              # ~5s context at 25ms hop, microWakeWord default
  inception:
    n_blocks: 3
    filters: [16, 32, 32]
    kernel_strides: [3, 3, 3]

training:
  positive_weight: 3.0            # positives are minority class
  hard_negative_weight: 2.0       # hard-negs more painful than bulk-negs
  batch_size: 256
  epochs: 50
  optimizer: adam
  learning_rate: 1e-3
  lr_schedule: cosine
  early_stop_patience: 5
  early_stop_metric: val_far_at_99_recall

eval:
  target_recall: 0.99             # accept 1% false-reject rate
  target_far_per_hour: 0.5        # ≤ 0.5 false fires per hour of background audio
  detection_threshold_grid: [0.5, 0.7, 0.8, 0.9, 0.95]

quantization:
  scheme: int8
  representative_n: 1000          # representative dataset for INT8 calibration
  target_runtime: tflite_micro_esp32s3
```

### 2c. Polling

Either:
- Use Claude Code's `/tofu-train` (which polls automatically via the trainer agent and uses `ScheduleWakeup` for long runs), or
- `curl -s https://<pod-id>-8001.proxy.runpod.net/setup.log | tail -30` periodically.

---

## Stage 3 — Eval

### 3a. Held-out test scoring

Script: `eval/runner.py`

```bash
python -m eval.runner \
    --model models/tofu-wakeword-v0.tflite \
    --tasks eval/tasks \
    --out eval/results/tofu-v0__$(date +%s).json
```

Metrics:
- **FRR** (False Reject Rate) on `eval/tasks/positives/*.wav` — should be ≤ 5%
- **FAR/hour** (False Accept Rate per hour) measured against `eval/tasks/bulk_negative_stream.wav` (a 1-hour concatenation of held-out LibriSpeech `test-clean` + AudioSet TV/music) — should be ≤ 1.0 per hour
- **Hard-neg confusion matrix** on `eval/tasks/hard_negatives/*.wav` — per-phrase FAR
- **ROC curve** — FAR vs FRR across detection-threshold grid

Each `eval/tasks/<bucket>/*.json` task file points at one WAV and declares its expected outcome:

```json
{
  "id": "pos_001",
  "audio_path": "eval/tasks/positives/hey_tofu_quan_001.wav",
  "phrase_uttered": "hey tofu",
  "expected": "fire",
  "metadata": {"speaker": "quan", "snr_db": 12, "distance_m": 1.5}
}
```

### 3b. Adversarial set

`eval/tasks/hard_negatives/` — at minimum:
- "hey doofus" × 20 voices
- "hi tofu" said at distance/SNR boundary
- "tofu" alone × 20
- "I love tofu" × 20
- Common food/recipe podcast 5-minute clips (real-world high-stakes false-fire test)

### 3c. Aggregator

Eval JSON output includes per-bucket pass/fail + global metrics + ROC. The `release-manager` agent picks the **operating point** (threshold) that maximizes `min(FRR ≤ 5%, FAR/hour ≤ 1.0)` and writes it into the model card.

---

## Stage 4 — Export + Deploy

### 4a. Quantization

INT8 post-training quantization (already done at end of Stage 2 by `export_tflite.py`):
- Representative dataset = 1000 random training-set windows
- Target: <100 kB `.tflite`, <10 ms inference on ESP32-S3 @ 240 MHz

### 4b. ESPHome integration

`scripts/export_tflite.py --emit-esphome configs/esphome_tofu.yaml` produces an ESPHome wake-word YAML stanza:

```yaml
micro_wake_word:
  vad:
  models:
    - model: hey_tofu
      probability_cutoff: 0.85
      sliding_window_size: 5
      url: https://huggingface.co/<user>/tofu-wakeword-v0/resolve/main/tofu-wakeword-v0.tflite
```

Drop this into any ESPHome config for an ESP32-S3 device and flash.

### 4c. Bare-metal C++ integration

For non-ESPHome firmwares, the artefact ships with a minimal `inference.cpp` example using `tflite::MicroInterpreter`. ~20 lines of glue.

---

## Stage 5 — Distribution

### 5a. HuggingFace release

Script: `scripts/upload_to_hf.py`

- Repo: `<user>/tofu-wakeword-v0`
- Files: `tofu-wakeword-v0.tflite`, `manifest.json`, `esphome_tofu.yaml`, `model_card.md`
- Model card: trigger phrases, FRR/FAR numbers, ESP32-S3 footprint, recipe link, citation

### 5b. GitHub release

```bash
gh release create v0.1.0 \
    models/tofu-wakeword-v0.tflite \
    configs/esphome_tofu.yaml \
    eval/results/tofu-v0__<latest>.json \
    --title "tofuWakeWord v0.1.0" \
    --notes-file RELEASE_v0_1_0.md
```

---

## Tool Reference

| Stage | Tool | Why |
|---|---|---|
| TTS | Piper, Kokoro, MeloTTS, Parler-TTS | Permissive + diverse |
| Bulk audio | MUSAN, DEMAND, AudioSet, Common Voice | Standard KWS negatives |
| Augmentation | audiomentations + torch-audiomentations | Battle-tested, CPU+GPU |
| Features | librosa-style 40-mel | microWakeWord's expected input |
| Architecture | microWakeWord (streaming Inception) | INT8 TFLite Micro on ESP32-S3 |
| Training | TensorFlow 2.x + Keras | microWakeWord's stack |
| Quantization | TFLite INT8 PTQ | <100 kB target |
| Runtime | TFLite Micro / ESPHome | ESP32-S3 native |
| Provisioning | RunPod (A40 or 4090) | Cheapest reliable GPU |
| Storage | HuggingFace Hub | Free + public |
| Logging | RunPod-served `setup.log` on :8001 | Survives silent pod death |

---

## Sources

- [microWakeWord](https://github.com/kahrendt/microWakeWord) + [docs](https://microwakeword.com/) + [Kevin's writeup](https://www.kevinahrendt.com/micro-wake-word)
- [openWakeWord](https://github.com/dscripka/openWakeWord) + [synthetic data doc](https://github.com/dscripka/openWakeWord/blob/main/docs/synthetic_data_generation.md) + [Hey Jarvis recipe](https://github.com/dscripka/openWakeWord/blob/main/docs/models/hey_jarvis.md)
- [Piper](https://github.com/rhasspy/piper) + [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- [MeloTTS](https://github.com/myshell-ai/MeloTTS)
- [Parler-TTS](https://github.com/huggingface/parler-tts)
- [audiomentations](https://github.com/iver56/audiomentations) + [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
- [ESPHome micro_wake_word component](https://esphome.io/components/micro_wake_word/)
- [MUSAN](https://www.openslr.org/17/) · [DEMAND](https://zenodo.org/records/1227121) · [OpenSLR-28 RIRs](https://www.openslr.org/28/) · [Speech Commands v2](https://huggingface.co/datasets/google/speech_commands) · [Common Voice](https://commonvoice.mozilla.org/) · [AudioSet](https://research.google.com/audioset/)
