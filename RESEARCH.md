# RESEARCH — Wake Word for AI Toys (2025–26)

A distilled survey of open-source wake-word / keyword-spotting (KWS) work as of 2026-05. Informs every choice in [`PIPELINE.md`](PIPELINE.md) and [`PLAN.md`](PLAN.md).

---

## 1. Model landscape

### microWakeWord — chosen for tofuWakeWord
- Kevin Ahrendt's INT8 TFLite Micro models built explicitly for **ESP32-S3**.
- Streaming Inception architecture; <10 ms inference on the LX7 core at 240 MHz.
- Ships as TFLite, ~50–100 kB on disk. Integrated into Home Assistant's Voice PE and S3-BOX, and exposed via ESPHome's `micro_wake_word` component.
- Training stack: TensorFlow / Keras + tflite-micro for export.
- Repo: <https://github.com/kahrendt/microWakeWord>
- Docs: <https://microwakeword.com/>
- ESPHome: <https://esphome.io/components/micro_wake_word/>
- ESP32-S3 working example: <https://github.com/kahrendt/esphome-on-device-wake-word>
- Writeup: <https://www.kevinahrendt.com/micro-wake-word>

### openWakeWord — sibling path for SBCs (deferred)
- David Scripka's 3-stage pipeline (mel → frozen Google audio embedding → small classifier head).
- Apache 2.0 framework code; pretrained weights are CC BY-NC-SA 4.0. For a hobby/personal Tofu, the NC license is fine.
- ONNX + TFLite runtime; runs comfortably on RPi Zero 2W / RPi 3+/4/5.
- Repo: <https://github.com/dscripka/openWakeWord>
- Synthetic data doc: <https://github.com/dscripka/openWakeWord/blob/main/docs/synthetic_data_generation.md>
- Hey Jarvis recipe (≈200k synthetic positives + 31k h negatives): <https://github.com/dscripka/openWakeWord/blob/main/docs/models/hey_jarvis.md>
- Custom verifier models (per-user fine-tune): <https://github.com/dscripka/openWakeWord/blob/main/docs/custom_verifier_models.md>

### Other options (evaluated, not chosen)
| Option | Note |
|---|---|
| Porcupine (Picovoice) | Best accuracy, closed weights, "type-to-train" — but free tier caps at 3 monthly active users. Commercial not viable for a toy at scale. <https://picovoice.ai/pricing/> |
| Mycroft Precise | Deprecated upstream; only the [OVOS fork](https://github.com/OpenVoiceOS/precise-lite) is maintained. RNN-based, outdated. |
| Snowboy | Defunct since 2020. |
| EfficientWord-Net | Few-shot (4–6 samples), ~95% reported, OK for prototyping. Ceiling lower than a retrained microWakeWord. <https://github.com/Ant-Brain/EfficientWord-Net> |
| WeKWS (WeNet) | Production-ready KWS toolkit, heavier; good if we want full architectural control. <https://github.com/wenet-e2e/wekws> |
| Howl (Castorini) | Research reference codebase. <https://github.com/castorini/howl> |
| Whisper-tiny / Distil-Whisper KWS | Heavy for MCUs; only relevant if Tofu also needs ASR. <https://huggingface.co/blog/fine-tune-whisper> |

General KWS index: <https://github.com/zycv/awesome-keyword-spotting>

---

## 2. Data requirements

### Positives
- microWakeWord's notebook gets a working model from ~3,000–10,000 synthetic positives.
- openWakeWord's reference recipes:
  - Tutorial models: ~3,400 synthetic positives.
  - "Hey Jarvis" production model: ~200,000 synthetic positives.
- **For Tofu v0**: target ~10,000 synthetic positives across four wake phrases, weighted toward "hey tofu" which we expect to be the most common.

### Hard negatives
- The single highest-leverage data category for wake words. Open-source models almost always fail on phonetic neighbors.
- For "tofu" specifically: `doofus`, `to-do`, `to-go`, `to-you`, `tutu`, `two pugs`, `toad`, `tofu` (no greeting), `I love tofu`, `tofu burger`, `phew`, `boo who`.
- **For Tofu v0**: ~2,500 synthetic hard negatives covering these collisions.
- Iterate: after v0 ships, log every real-world false-fire and feed it back as v1 negatives.

### Bulk negatives
- "Hey Jarvis" used ~31,000 hours of negative audio. For a hobby model, ~200–500 hours is enough.
- Sources:
  - **MUSAN** (~109 h music+speech+noise): <https://arxiv.org/abs/1510.08484> · mirror <https://www.openslr.org/17/>
  - **DEMAND** (48 kHz real-room noise, 18 environments): <https://zenodo.org/records/1227121>
  - **Common Voice** (Mozilla, multi-lingual, thousands of voices): <https://commonvoice.mozilla.org/>
  - **AudioSet** (2M YT clips, 632 classes): <https://research.google.com/audioset/>
  - **LibriSpeech** `test-clean` for held-out eval FAR: <https://www.openslr.org/12/>
  - **Google Speech Commands v2** as easy negatives: <https://huggingface.co/datasets/google/speech_commands>

---

## 3. TTS for synthetic positives

The key insight from both openWakeWord and microWakeWord: **voice/accent/prosody diversity at the data layer is more important than model size**.

### Open-source — what we'll use
| Engine | Voices | License | Notes |
|---|---|---|---|
| **Piper** | 100+ across ~30 languages | MIT | Local ONNX, CPU-fast. The default workhorse. Repo: <https://github.com/rhasspy/piper>. Voices: <https://huggingface.co/rhasspy/piper-voices>. Samples: <https://rhasspy.github.io/piper-samples/>. Sample generator built for openWakeWord: <https://github.com/rhasspy/piper-sample-generator>. GPL'd current fork: <https://github.com/OHF-Voice/piper1-gpl>. |
| **Kokoro-82M** | ~60+ | Apache 2.0 | 82M params, MOS ~4.2, multilingual (en/ja/zh/fr/es/hi/it/pt). Currently #1 on TTS Arena. <https://huggingface.co/hexgrad/Kokoro-82M>. |
| **MeloTTS** | 4 English accents (US/UK/AU/IN) + more languages | MIT | CPU real-time, broad accent variety. <https://github.com/myshell-ai/MeloTTS>. |
| **Parler-TTS** | natural-language voice prompts | Apache 2.0 | Steer the voice with text ("a young woman, slightly fast, slightly excited"). Best for prosody diversity. <https://github.com/huggingface/parler-tts>. |
| **OpenVoice v2** | cloning + accent transfer | MIT | <https://github.com/myshell-ai/OpenVoice>. |
| **XTTS-v2 (Coqui)** | clone from 6s | NC | OK for hobby Tofu but not for shipped commercial data. <https://huggingface.co/coqui/XTTS-v2> · active fork: <https://github.com/idiap/coqui-ai-TTS>. |
| **F5-TTS** | expressive | Apache-ish | Diffusion, slower. <https://github.com/SWivid/F5-TTS>. |
| **StyleTTS 2** | diffusion style | MIT | <https://github.com/yl4579/StyleTTS2>. |
| **ChatTTS** | en/zh prosody | various | <https://github.com/2noise/ChatTTS>. |
| **Sesame CSM-1B** | conditioned conversational TTS | Apache 2.0 | <https://huggingface.co/sesame/csm-1b>. |

### Commercial APIs (rough costs)
| Service | $/1M chars | Voices | Notes |
|---|---|---|---|
| ElevenLabs | $150–300 | 5000+ via Voice Library | Best timbral diversity. <https://elevenlabs.io/pricing> |
| Azure Neural TTS | $15 | 400+ across 140 locales | Broadest accent library. <https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/> |
| Google Cloud TTS | $16 (Neural2), $30 (Chirp 3 HD) | 380+ | <https://cloud.google.com/text-to-speech/pricing> |
| OpenAI TTS | $15 (mini), $30 (hd) | ~10 | Limited voice count. <https://platform.openai.com/docs/guides/text-to-speech> |
| PlayHT | subscription | 900+ | <https://play.ht/pricing/> |
| Cartesia Sonic-3 | ~$100/1M | 100+ | Very low latency. <https://cartesia.ai/pricing> · <https://cartesia.ai/voices> |

### Best mix for Tofu v0 (hobby, free)
Piper + Kokoro + MeloTTS + Parler-TTS together yield 200+ distinct voices for $0. Add an optional ~$30 of Azure Neural TTS later for the broadest accent outliers if v0 misses on specific regional pronunciations.

---

## 4. Augmentation chain

Standard openWakeWord-style chain, applied per-sample at training time:

1. **RIR convolution** for room acoustics
   - Corpora: **OpenSLR-28 RIRS_NOISES** (<https://www.openslr.org/28/>) + **MIT McDermott IR Survey** (<https://mcdermottlab.mit.edu/Reverb/IR_Survey.html>).
   - Or generate on the fly with `pyroomacoustics` (<https://github.com/LCAV/pyroomacoustics>).

2. **Additive noise** at random SNR (-5 to +25 dB) from MUSAN, DEMAND, AudioSet segments.

3. **Pitch shift** ±2 semitones + **time-stretch** 0.85–1.15× (gentle prosody jitter).

4. **Codec degradation** — round-trip through Opus 8–24 kbps, AMR-NB, MP3 32–96 kbps, μ-law. Cheap toy mics are the actual deployment target; codec artefacts are critical.

5. **Level jitter / random gain** ±10 dB; random DC offset; occasional clipping.

6. **Spec augmentation** (frequency/time masks at the spectrogram stage).

7. **Bandlimiting** (200 Hz HPF, 7.8 kHz LPF) — simulates cheap MEMS mics on the toy.

Libraries:
- **audiomentations** (CPU, NumPy): <https://github.com/iver56/audiomentations> — has `ApplyImpulseResponse`, `AddBackgroundNoise`, `Mp3Compression`, `RoomSimulator`.
- **torch-audiomentations** (GPU batched): <https://github.com/asteroid-team/torch-audiomentations>.
- **WavAugment** (Meta, fast SoX-based): <https://github.com/facebookresearch/WavAugment>.

---

## 5. Deployment

| Target | Runtime | Model | Footprint |
|---|---|---|---|
| **ESP32-S3** (Tofu's target) | TFLite Micro via ESPHome | microWakeWord INT8 | <100 kB, <10 ms inference |
| RPi Zero 2W → Pi 5 | ONNX Runtime or TFLite Runtime | openWakeWord | ~hundreds of KB, real-time |
| Linux x86 / macOS dev | ONNX Runtime | openWakeWord | sanity tests, eval loop |
| Android | ONNX Mobile | openWakeWord | Atlas Voice / HAwake examples |

References:
- ESPHome micro_wake_word component: <https://esphome.io/components/micro_wake_word/>
- ESP32-S3 working example: <https://github.com/kahrendt/esphome-on-device-wake-word>
- openWakeWord runtime: <https://github.com/dscripka/openWakeWord>

---

## 6. The recipe — concrete recommendation

For tofuWakeWord v0:

1. **Model**: microWakeWord (ESP32-S3 target). Sibling openWakeWord training kept in `scripts/` for future RPi-class deployment but not run for v0.
2. **Positives**: 10,000 synthetic across "hey tofu" / "hi tofu" / "hello tofu" / "okay tofu" via Piper + Kokoro + MeloTTS + Parler-TTS.
3. **Hard negatives**: 2,500 synthetic across the collision set in §2.
4. **Bulk negatives**: ~300 h sampled from MUSAN + DEMAND + AudioSet + Common Voice.
5. **Augmentation**: audiomentations chain above (RIR + noise + codec + EQ + gain + spec).
6. **Splits**: 80/10/10 train/val/test, test set held out from the very first script run.
7. **Training**: microWakeWord notebook ported into `scripts/train_microwakeword.py`, run on RunPod A40 ~1–2 h.
8. **Eval target**: FRR ≤ 5% on positives, FAR ≤ 1.0/h on bulk-negative stream, FAR ≤ 5% on each hard-negative bucket.
9. **Iterate**: ship v0, log false-fires from real Tofu owners (or family beta testers), fold into v1 negatives. Optional per-user custom verifier (3–5 personal samples) for households that need lower FAR.

License hygiene (hobby tier, so loose): we still default to Apache/MIT TTS (Piper/Kokoro/MeloTTS/Parler-TTS) for the v0 dataset so the artefact is freely shareable. XTTS-v2 and ElevenLabs available as fallbacks for diversity if v0 underperforms on specific accents.
