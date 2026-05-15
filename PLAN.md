# PLAN — Tofu Wake Word v0

The concrete numbers behind [`PIPELINE.md`](PIPELINE.md). Updated 2026-05-15.

---

## 1. Wake phrases

| Phrase | Target positives | Weight in training |
|---|---|---|
| "hey tofu" | 5,000 | 1.0 |
| "hi tofu" | 2,500 | 1.0 |
| "hello tofu" | 1,500 | 0.8 |
| "okay tofu" | 1,000 | 0.7 |
| **Total** | **10,000** | — |

All four are wired through `configs/wake_phrases.yaml`. The model is trained as a **single-class detector** (any of the four → "wake"); the architecture doesn't distinguish phrases, only the trigger event. If we later want phrase-specific behavior ("okay tofu" → command mode vs "hey tofu" → greeting), we can either (a) ship a tiny secondary classifier or (b) train per-phrase heads. v0 keeps it simple.

Variations baked into TTS prompts:
- Speed: 0.85×, 0.95×, 1.05×, 1.15× (4 settings × ~2500 voices/phrase combos)
- Emotion (Parler-TTS): neutral, excited, calm, child-like (when voice supports it)
- Distance simulation deferred to augmentation stage (RIR + level)

---

## 2. Hard negatives

The collision set, ranked by how likely we are to misfire on each:

| Bucket | Examples | Target |
|---|---|---|
| Phonetic /tu/+/fu/ collisions | "to-do list", "to-do", "to-go", "to you", "two-fold" | 600 |
| Rhyming insults | "hey doofus", "hi doofus", "you doofus" | 400 |
| Common phrases starting with /he/ or /hai/ | "hey there", "hey buddy", "hi there", "hello world" | 300 |
| "tofu" alone (no greeting → must NOT fire) | "tofu", "tofu salad", "I love tofu", "this tofu is amazing" | 500 |
| Food / cooking contexts | "tofu burger", "fried tofu", "silken tofu", "tofu scramble", "marinated tofu" | 300 |
| Disyllabic /tu/ words | "tutu", "two pugs", "two pups", "to-do", "Tutu" (name) | 200 |
| Misc near-matches | "toad", "phew", "ufo", "boo who", "ho-fu" | 200 |
| **Total** | — | **2,500** |

Configured in `configs/hard_negatives.yaml`. After v0 ships, we add a `live_misfires.yaml` populated from real-world false-fires.

---

## 3. Bulk negatives

Target: ~300 h of audio with no "tofu"-adjacent content.

| Source | Hours sampled | Purpose |
|---|---|---|
| **LibriSpeech train-clean-100 + 360** | 100 h | Clean read speech |
| **Common Voice English** | 50 h | Crowdsourced everyday speech, accent diversity |
| **MUSAN speech** | 30 h | Mixed speech segments |
| **MUSAN music** | 20 h | Background music |
| **MUSAN noise** | 10 h | Pure noise (machinery, ambient) |
| **DEMAND** | 10 h | Real-room noise (kitchen, café, bus, park…) |
| **AudioSet subset** | 80 h | TV, radio, podcast, household sounds |
| **Total** | **~300 h** | |

We deliberately **exclude** LibriSpeech `test-clean` from training so we can use it as a clean eval-FAR substrate.

---

## 4. Augmentation chain

Applied per-sample at TFRecord-build time. Each positive is replicated 5× with different augmentation realizations; each negative 1×.

```
input WAV (16 kHz mono)
  → random crop to 1.5 s window
  → RoomSimulator OR ApplyImpulseResponse  (p=0.7, 50/50 mix)
  → AddBackgroundNoise (SNR ∈ [-5, +25] dB)  (p=0.9 for positives, 0.6 for negs)
  → PitchShift ±2 semitones                  (p=0.3)
  → TimeStretch 0.85–1.15×                   (p=0.3)
  → BandPassFilter 200–7800 Hz               (p=0.5, cheap-mic sim)
  → Mp3Compression 32–96 kbps                (p=0.2)
  → OpusCompression 8–24 kbps                (p=0.2)
  → Gain ±10 dB                              (p=1.0)
  → 40-mel spectrogram (60 ms win, 25 ms hop)
  → 1.5 s window @ 10 ms hop → 150 frames × 40 mel features
  → uint16 spectrogram (matches on-device audio_microfrontend output)
```

Per-sample replication: each positive is rendered into ~10 spectrogram windows via `SpectrogramGeneration(slide_frames=10, repetition=2)` for training; 1 window for val/test. Total effective training set: **~400k positive features** + **~25k hard-neg features** + ~9 GB of pre-computed bulk-negative mmap from upstream.

---

## 5. Training config (upstream microWakeWord)

Lives at [`configs/examples/tofu/training_parameters.yaml`](configs/examples/tofu/training_parameters.yaml). Schema matches `microwakeword.model_train_eval.load_config()`.

Key choices for v0:
- **Architecture**: MixedNet `pointwise_filters="64,64,64,64"`, `mixconv_kernel_sizes="[5],[7,11],[9,15],[23]"`, `stride=3` (same as the shipped Okay Nabu / Hey Jarvis models).
- **Window**: `clip_duration_ms: 1500`, `window_step_ms: 10` (matches on-device ESPHome runtime).
- **Features**: 40 mel-like channels via `pymicro-features.MicroFrontend` (C `audio_microfrontend` op) — bit-exact to what the device computes.
- **Training**: `training_steps: [10000]`, `batch_size: 128`, `learning_rates: [0.001]`, `negative_class_weight: [20]`.
- **Hard-neg penalty**: `penalty_weight: 5.0` on the Tofu hard-negatives mmap (vs default 1.0) — actively suppresses phonetic neighbors.
- **Dinner-party boost**: `sampling_weight: 15.0`, `penalty_weight: 3.0` on CHiME6 dinner-party (vs default 10.0/1.0) — catches the "triggers on a podcast" failure mode.
- **Best-checkpoint pick**: `maximization_metric: average_viable_recall`, `target_minimization: 0.5` FA/hr.

Hardware budget on RunPod:
- RTX 4090 SECURE ~$0.69/hr (primary)
- A40 SECURE ~$0.44/hr (fallback when 4090 unavailable)
- Estimated training wall: 15–25 min for 10k steps. Full on-pod pipeline (synth + HF download + features + train + eval + manifest): 30–45 min.
- **Estimated cost per training run: $0.40–0.55** on 4090 SECURE; **$0.25–0.35** on 4090 COMMUNITY (spot).

---

## 6. Eval targets

A v0 ship is gated on:

| Metric | Target | Stretch |
|---|---|---|
| FRR (positives) | ≤ 5% | ≤ 2% |
| FAR (bulk negatives) | ≤ 1.0 / hour | ≤ 0.5 / hour |
| FAR per hard-negative bucket | ≤ 5% | ≤ 2% |
| `.tflite` size | ≤ 100 kB | ≤ 60 kB |
| ESP32-S3 inference latency | ≤ 10 ms | ≤ 6 ms |

If the model misses targets, the eval JSON's per-bucket breakdown tells us where to invest more positives or negatives.

---

## 7. Timeline (best case)

| Day | Step | Compute |
|---|---|---|
| D0 (today) | Scaffold repo (done) | $0 |
| D1 | TTS synthesis (~10k positives + 2.5k hard-negs) on Mac CPU | $0 |
| D1 | Download bulk corpora (~30 GB) | $0 |
| D2 | Feature extraction + TFRecord build (Mac) | $0 |
| D2 | Eval-task hand curation (~50 held-out positives, 50 hard-negs) | $0 |
| D3 | First RunPod training run | $0.40–0.80 |
| D3 | Eval v0 against held-out tasks | $0 |
| D3 | Publish to HuggingFace + write model card | $0 |
| D4+ | Iterate: log misfires → v1 negatives → retrain | $0.40 per iteration |

**Total D0 → v0 launch: ~$1, ~4 days wall time.**

---

## 8. Open questions

- **Multi-language**: do we need "tofu" wake in Vietnamese / Chinese / Japanese? Defer to v1.
- **Speaker adaptation**: should each toy fine-tune a per-owner custom verifier head? openWakeWord supports this; microWakeWord doesn't natively. Defer to v2.
- **Wake + ASR pipeline**: after wake, does the toy stream to a remote LLM (cloud) or do on-device ASR? Out of scope for this repo — this only owns the wake gate.
- **Quantization quality**: INT8 PTQ may drop more than 1 pp recall vs the float model. If it does, add quant-aware training (QAT) in `train_microwakeword.py` for v1.
- **VAD pre-gating**: ESPHome's `micro_wake_word` component has an optional VAD gate. Cuts power on near-silence. Recommended on, but adds latency. Test both in v0 eval.

---

## 9. v1+ ideas

- **Per-user custom verifier**: 3–5 recordings from the toy owner → train a tiny 2-layer head on top of v0 embeddings. ~95% reduction in FAR for that user.
- **Multi-phrase classifier**: replace the binary head with a 5-way softmax (4 wake phrases + "not-wake") so the toy can react differently to "hey tofu" vs "okay tofu".
- **Personality-conditioned wakes**: detect the prosody of the wake (excited vs calm) and pass that label to the downstream LLM as a context flag.
- **Emotion synth**: use Parler-TTS / ChatTTS to add explicit emotion to positives → better generalization across user moods.
