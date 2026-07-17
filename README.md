# customWakeWord

Training toolkit for producing custom INT8 microWakeWord models for
**ESP32-S3 devices running ESPHome**.

The repository implements project configuration, Piper sample generation,
MicroFrontend-compatible feature extraction, upstream microWakeWord training,
held-out streaming evaluation, ESPHome manifest generation, and Hugging Face
upload helpers.

For a current comparison of ready-made models, turnkey trainers, this training
pipeline, and Espressif WakeNet, see
[`ESP32_S3_OPTIONS_ZH.md`](ESP32_S3_OPTIONS_ZH.md) (Chinese).

It is a training toolkit, not a collection of ready-made models. This checkout
does not contain trained `.tflite` artifacts or a quality guarantee for an
arbitrary phrase.

> **Firmware compatibility:** output is an INT8 `.tflite` for ESPHome's
> `micro_wake_word`. It does not load on XiaoZhi, ESP-S3-BOX stock firmware, or
> firmware based on Espressif ESP-SR/WakeNet. Those systems expect a proprietary
> `wn9_*.bin` model.

## Quick start: personal GPU server

The recommended path is the Docker launcher on a Linux server with an NVIDIA
GPU. The host only needs Docker, an NVIDIA driver, and NVIDIA Container
Toolkit; Python 3.10, TensorFlow, microWakeWord, Piper, and `ffmpeg` are provided
by the training image.

```bash
git clone https://github.com/Horoaw/customWakeWord.git
cd customWakeWord

# Default project: xingxing; default wake phrase: 星星
bash scripts/train_local_server.sh
```

The wake phrase is configured near the top of
[`scripts/train_local_server.sh`](scripts/train_local_server.sh):

```bash
PROJECT_NAME="${PROJECT_NAME:-xingxing}"
WAKE_PHRASES="${WAKE_PHRASES:-星星}"
LANGUAGE="${LANGUAGE:-zh}"
```

You can also override the settings without editing the file:

```bash
PROJECT_NAME=nihao_xingxing \
WAKE_PHRASES="你好星星" \
LANGUAGE=zh \
bash scripts/train_local_server.sh
```

Use a new `PROJECT_NAME` whenever the wake phrase changes. The launcher checks
existing YAML before training and refuses to mix old audio/features with a new
phrase. Completed synthesis and feature stages are reused on subsequent runs.

The full run performs:

1. GPU/container and TensorFlow preflight checks.
2. Wake-word project initialization.
3. Positive and hard-negative Piper synthesis.
4. Public Hugging Face negative-feature download.
5. MicroFrontend feature generation and MixedNet training.
6. INT8 streaming TFLite export.
7. Optional held-out evaluation and ESPHome manifest generation.

Primary outputs:

```text
models/<project>-wakeword-v0.tflite
trained_models/<project>-v0/
configs/examples/<project>/manifest.json   # only after threshold/evaluation
```

Public model/data downloads do not normally require `HF_TOKEN`. If the
pre-built image is unavailable or you prefer to build it on the server:

```bash
docker build -t customwake-deps:local .
TRAIN_IMAGE=customwake-deps:local bash scripts/train_local_server.sh
```

## Implemented

- Unicode-safe project initialization for English and Mandarin projects.
- Baseline `training_parameters.yaml` generation for every new project.
- Piper synthesis:
  - English LibriTTS-R generation through Piper Sample Generator v3, with up
    to 904 speaker embeddings.
  - Configurable standard Piper ONNX voices through lightweight `piper-tts`,
    including Mandarin defaults.
- LLM-assisted and hand-editable hard-negative phrase lists.
- Upstream microWakeWord augmentation, MicroFrontend features, RaggedMmap
  datasets, MixedNet training, and streaming INT8 export.
- Held-out evaluation using the same `pymicro_features` frontend and streaming
  input shape used by the deployed model.
- FRR, hard-negative bucket FAR, and FAR/hour from measured bulk-audio duration.
- ESPHome v2 manifest and Hugging Face model-card generation.
- Personal-server Docker launcher plus RunPod and Lambda launch helpers.

## Not automatic

- A generated model is not necessarily usable. Reliable wake words normally
  require real recordings, deployment-specific negatives, threshold tuning,
  and repeated device tests.
- Kokoro, MeloTTS, and Parler-TTS are research options only; the executable
  synthesis path currently uses Piper.
- The repository does not produce ESP-SR/WakeNet `.bin` models.
- Raspberry Pi/openWakeWord training is not implemented.
- RunPod training cannot create a trustworthy release unless held-out eval
  audio is made available separately. Without eval tasks it exports the model
  and intentionally skips the manifest and upload.

## Manual environment requirements

- Python 3.10 for the pinned training environment.
- Linux with an NVIDIA GPU for full training. CPU can run initialization,
  small synthesis checks, and evaluation.
- `git`, `ffmpeg`, and enough disk for generated audio and negative datasets.
- Approximately 10 GB for the default precomputed negative feature datasets,
  plus generated samples and augmentation corpora.

These requirements apply when running the Python stages manually instead of
using `scripts/train_local_server.sh`. The Dockerfile contains the reproducible
GPU dependency environment.
For a local Mandarin synthesis smoke test, the full GPU stack is unnecessary:

```bash
python -m pip install "piper-tts>=1.3,<2" pyyaml
```

## Mandarin example: 星星

`星星` is a short everyday phrase with high-risk collisions such as `猩猩`,
`醒醒`, `行星`, `新星`, `小星星`, and ordinary sentences containing the word.
Review the generated negatives before spending GPU time. An exact homophone
cannot be reliably separated using acoustics alone; a longer phrase such as
`你好星星` is normally easier to deploy.

```bash
# 1. A reviewed baseline already exists at configs/examples/xingxing.
# To create a separate variant instead:
python scripts/init_wake.py --name my_xingxing --phrases "星星" --language zh

# 2. Review the included collision list, then synthesize audio.
python -m pip install "piper-tts>=1.3,<2" pyyaml
python scripts/synth_positives.py --project xingxing
python scripts/synth_hard_negatives.py --project xingxing

# 3. Download shared negative features and build project features.
python scripts/download_hf_negatives.py --out data/negative_datasets
python scripts/build_features.py --project xingxing --download-aug-corpora

# 4. Train on a configured GPU host.
python scripts/train_microwakeword.py --project xingxing
```

The Mandarin config downloads two standard Piper ONNX voices into
`data/tts_models` on first use. It does not need the English LibriTTS-R `.pt`
generator or PyTorch. Add more
properly licensed Mandarin voices and real recordings for speaker diversity.
The default bulk speech negatives are English-heavy, so Mandarin deployment
also requires held-out Mandarin conversation, television, music, and room
audio.

## English example

```bash
python scripts/init_wake.py \
  --name sunny \
  --phrases "hey sunny,hi sunny,hello sunny,okay sunny" \
  --language en

python scripts/synth_positives.py --project sunny
python scripts/synth_hard_negatives.py --project sunny
python scripts/download_hf_negatives.py --out data/negative_datasets
python scripts/build_features.py --project sunny --download-aug-corpora
python scripts/train_microwakeword.py --project sunny
```

Existing worked configurations are under `configs/examples/xingxing`,
`configs/examples/tofu`, and `configs/examples/greet`.

## Held-out evaluation

Do not evaluate only on TTS samples used to build training features. Add raw,
held-out negative audio under `data/raw/negatives` or pass another directory.
Real recordings from the target microphones should be added as task JSON files
under `eval/tasks/<project>/positives`.

```bash
python scripts/seed_eval_tasks.py \
  --project xingxing \
  --positives 100 \
  --hard-negatives 200 \
  --bulk-audio-dir data/raw/negatives \
  --bulk-stream-minutes 60

python -m eval.runner \
  --project xingxing \
  --model models/xingxing-wakeword-v0.tflite \
  --threshold 0.85 \
  --out eval/results/xingxing-v0__latest.json
```

Run evaluation at several thresholds and select an operating point appropriate
for the device. Initial targets are typically FRR <= 5% and FAR <= 1/hour, but
short common phrases may not reach both targets.

The evaluator:

1. Resamples input to mono 16 kHz.
2. Runs the C MicroFrontend from `pymicro_features`.
3. Feeds the model's streaming input slices in order.
4. Resets model state between tasks.
5. Applies the configured sliding probability window.
6. Uses actual bulk-audio duration for FAR/hour.

## ESPHome manifest

A manifest now requires both an existing model and an explicit or evaluated
threshold. It will not silently publish the previous placeholder cutoff.

```bash
python scripts/emit_manifest.py \
  --project xingxing \
  --eval-json eval/results/xingxing-v0__latest.json
```

The generated JSON can be referenced by ESPHome:

```yaml
micro_wake_word:
  microphone: wake_mic
  vad:
  models:
    - model: https://example.com/xingxing/manifest.json
```

ESPHome's model JSON contains the model path, language, probability cutoff,
sliding-window size, feature step, and tensor arena requirement.

## Cloud training

RunPod and Lambda helpers provision a host and execute synthesis, feature
generation, and training:

```bash
python scripts/runpod_train.py --project xingxing
python scripts/lambda_train.py --project xingxing
```

Read `RUNPOD_RECIPE.md` or `LAMBDA_SETUP.md` before use. These scripts incur
cloud cost and require credentials. Generated data is gitignored, so held-out
evaluation data must be mounted, uploaded, or evaluated after retrieving the
model.

## Repository layout

```text
configs/examples/<project>/
  wake_phrases.yaml
  hard_negatives.yaml
  training_parameters.yaml

scripts/
  train_local_server.sh
  init_wake.py
  synth_positives.py
  synth_hard_negatives.py
  suggest_hard_negatives.py
  download_hf_negatives.py
  collect_negatives.py
  build_features.py
  train_microwakeword.py
  seed_eval_tasks.py
  emit_manifest.py
  upload_to_hf.py
  runpod_train.py
  lambda_train.py

eval/
  runner.py
  schema.py
  tasks/<project>/
  results/

webui/
  app.py
```

## Data and license

Repository code is Apache 2.0.

Training data and resulting model redistribution terms depend on the selected
voices and corpora. In particular, the default precomputed microWakeWord
negative datasets are marked CC-BY-NC-4.0 and are not suitable for a commercial
model. Commercial projects must replace them with appropriately licensed
speech, noise, music, and evaluation data and record the provenance in the
model card.

## Status

The pipeline code is implemented, but this repository does not include a
trained or independently validated model. A release should only be considered
complete after device-specific held-out evaluation and ESP32-S3 testing.
