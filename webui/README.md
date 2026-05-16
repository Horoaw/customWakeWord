---
title: customWakeWord live tester
emoji: 🎤
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# customWakeWord — live tester

Browser-side test harness for any wake-word model trained by this repo's
pipeline. Records audio in the browser (or upload a WAV), runs the same
INT8 streaming inference path the ESP32-S3 firmware uses, and reports
fire/no-fire + a per-window probability trace.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r webui/requirements.txt
python webui/app.py
# → open http://localhost:7860
```

## Deploy to a Hugging Face Space (no infra)

```bash
# One-time
huggingface-cli login

# Create + push the Space
huggingface-cli repo create customwakeword-tester --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<you>/customwakeword-tester
cd customwakeword-tester
cp /path/to/customWakeWord/webui/{app.py,requirements.txt,README.md} .
git add . && git commit -m "init" && git push
```

The Space builds in ~3 min, then anyone with the URL can test the
default model (`nagisanzeninz/tofu-wakeword-v0`) or any other published
customWakeWord build by editing the **Hugging Face repo id** box.

## How the inference matches the ESP32-S3 firmware

This UI was specifically wired to use **the same audio pipeline the
on-device firmware uses**, so what you hear here is what your ESP32
will hear:

1. **Resample** browser audio (typically 44.1 / 48 kHz float32) → 16 kHz
   int16 PCM via `scipy.signal.resample_poly`.
2. **Mel features** via the C `MicroFrontend` (rhasspy/pymicro-features),
   identical to the `audio_microfrontend` op the tflite-micro runtime
   compiles into the ESP32 binary.
3. **Streaming INT8 inference** via `ai_edge_litert` — same TFLite kernel
   set, same int8 quantization parameters baked into the `.tflite` file.
4. **Threshold** is read from `manifest.json` (`micro.probability_cutoff`),
   the same JSON ESPHome loads next to the `.tflite`.

There are intentionally no smoothing, debounce, or VAD layers — those
live in ESPHome's `micro_wake_word` component on the device. The raw
per-window probability trace lets you see what the model thinks
moment-to-moment.

## Adding your own model

Train via the upstream pipeline, push to HF Hub:

```bash
python scripts/upload_to_hf.py --project myproject \
    --model models/myproject-wakeword-v0.tflite \
    --repo-id you/myproject-wakeword-v0 \
    --eval-json eval/results/myproject-v0__latest.json \
    --esphome configs/examples/myproject/manifest.json
```

Then paste the repo id into the Hugging Face repo id box. The model
card, manifest, and `.tflite` get fetched lazily on first run.
