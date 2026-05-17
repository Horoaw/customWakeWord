# Release protocol

Pre-flight checks before publishing a wake-word model to HuggingFace + downstream
firmwares. This is the canonical answer to "where can my `.tflite` actually
run?" — the surprising answer is **only ESPHome**, and shipping to non-ESPHome
ESP32 firmwares (XiaoZhi, ESP-S3-BOX-3, anything using `esp-sr`) requires a
**separate** model in a **different** format that **we currently cannot
self-produce**.

Confirmed 2026-05-17 from research into the XiaoZhi firmware stack
([78/xiaozhi-esp32](https://github.com/78/xiaozhi-esp32)) and Espressif's
[ESP-SR](https://github.com/espressif/esp-sr) v2.0 (released 2025-02-14).

---

## Firmware target matrix

| Target firmware | Wake-word backend | Model format | Our pipeline produces it? |
|---|---|---|---|
| **ESPHome** (Home Assistant) | `micro_wake_word` component (kahrendt) | INT8 `.tflite` (TFLM, ~80 kB) | **Yes** — current pipeline output |
| **XiaoZhi** (78/xiaozhi-esp32) | ESP-SR `WakeNet` | `wn9_*.bin` (proprietary) | **No** — see below |
| **ESP-S3-BOX-3** stock fw | ESP-SR `WakeNet` | `wn9_*.bin` | **No** |
| **esp-skainet** examples | ESP-SR `WakeNet` | `wn9_*.bin` | **No** |
| **Custom firmware** (DIY ESP32-S3 + TFLM glue) | TensorFlow Lite Micro | INT8 `.tflite` | **Yes** — but you write the glue |

**Bottom line:** our pipeline currently targets exactly one ecosystem
(ESPHome). For tofuchan.io toys that ship XiaoZhi, our `.tflite` is not
loadable as-is.

---

## Why microWakeWord `.tflite` ≠ ESP-SR `.bin`

These are two different model families:

- **microWakeWord** (kahrendt, used by ESPHome): MixedNet streaming
  convolutional architecture, INT8 quantized via TFLite-Micro tooling.
  Inference runs in the standard `tflite-micro` interpreter (`AllOpsResolver`
  with a curated subset). Public training code, public format.

- **ESP-SR WakeNet**: dilated-convolution architecture (WakeNet9 / 9l /
  9s, plus newer `_tts2` variants from Espressif's TTS Pipeline v2.0).
  Inference runs through Espressif's closed-source `esp-sr` C library which
  expects a custom Espressif binary layout — **not** a TFLite model.
  **Training code is closed-source**. The public esp-sr repo contains only
  inference + pre-built `.bin` model files.

There is **no public TFLite → ESP-SR converter** and there is **no
self-service WakeNet trainer**. We confirmed this by reading the esp-sr
repo, the v2 release notes, and Espressif's customization documentation.

---

## Paths to an ESP-SR `.bin` (when we need one)

In order of cost / control:

1. **Use a pre-baked XiaoZhi wake word.** XiaoZhi ships 26+ official wake
   words (你好小智, Hi ESP, Alexa, etc.). Zero training. Zero cost. But:
   our brand words (`tofu`, `hey/hi/hello`) are not on the list.

2. **Espressif's free customization process.** Submit corpus → Espressif
   trains internally → 2–3 weeks turnaround → free. Requires a corpus of
   500+ real speakers (men + women + 100+ children) at 1 m and 3 m, 16 kHz
   mono WAV. Our TTS-synthesized corpus does **not** meet the speaker-count
   criterion (it's voice-cloned from a few hundred LibriTTS speakers, not
   genuine human samples). Could still be worth trying.

3. **Espressif's TTS Pipeline v2.0 (free).** New as of 2025. Trains a
   `wn9_<word>_tts2-*.bin` from TTS-only samples. The pipeline itself does
   not appear to be public — submission flow still goes through Espressif
   sales.

4. **[CustomESP-SR.com](https://custom-espsr.com/) ($1000 / word / ~10
   business days).** Third-party paid service. Production-ready `wn9_*.bin`
   royalty-free. They handle dataset design + audio collection + training
   + browser-based pre-delivery testing.

5. **Fork XiaoZhi to add a TFLite-Micro backend.** Replace XiaoZhi's
   `EspWakeWord` class with a `TfliteWakeWord` class that wraps the
   `tflite-micro` interpreter and our `.tflite`. Keeps our pipeline; adds
   firmware engineering work. The DEV.to writeup
   [ESP32-S3 + TFLite Micro](https://dev.to/zediot/esp32-s3-tensorflow-lite-micro-a-practical-guide-to-local-wake-word-edge-ai-inference-5540)
   demonstrates the inference path works at 15–20 FPS on a 240 MHz core.

---

## Pre-release checklist

Before tagging an HF release as "production-ready" for a given firmware target:

- [ ] **State the target.** Pick one row from the matrix. The model card
      MUST declare which firmware ecosystem the artifacts work in.
- [ ] **Test on real hardware** in that ecosystem. A `.tflite` that passes
      our held-out FRR/FAR eval still has to load + run on the device.
      For ESPHome: flash an ESP32-S3 + check the model card's YAML stanza
      ingests cleanly + the device wakes on the phrase.
- [ ] **Document the unsupported firmwares.** If we shipped only the
      `.tflite`, the model card and README MUST explicitly say "this model
      does not load on XiaoZhi / ESP-S3-BOX-3 / esp-skainet — those need a
      separate WakeNet `.bin`." Users will otherwise assume it works
      everywhere.
- [ ] **Plan the ESP-SR companion** (when relevant). For tofuchan.io toys
      using XiaoZhi: file a parallel issue tracking the ESP-SR `.bin`
      production path (one of options 1–5 above). The `.tflite` and the
      `.bin` should release together under one HF repo (with two artifacts)
      so users have one place to look.

---

## HuggingFace repo layout (target)

```
nagisanzeninz/<project>-wakeword-v<N>/
├── README.md                                  ← model card
├── <project>-wakeword-v<N>.tflite             ← microWakeWord / ESPHome
├── <project>-wakeword-v<N>.tflite.json        ← microWakeWord manifest
├── esphome.yaml                               ← drop-in ESPHome stanza
│
├── wn9_<project>_tts2-*.bin                   ← ESP-SR / XiaoZhi (when avail)
├── wakenet_partition.csv                      ← XiaoZhi assets.h hints
│
└── eval/
    ├── frr_far.json                           ← held-out scores
    └── audio_samples/                         ← failure-mode WAVs
```

The `wn9_*.bin` row is aspirational — present once we have an ESP-SR
production path established for the project. Until then, the model card
must say so.

---

## Memory pointers

- Provider economics: [PROVIDERS.md](PROVIDERS.md)
- RunPod operational gotchas: [RUNPOD_RECIPE.md](RUNPOD_RECIPE.md)
- Failure modes from v0 build: [LESSONS_v0.md](LESSONS_v0.md)
- This doc covers firmware-format compatibility — strictly orthogonal to
  the above three.
