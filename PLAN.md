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
  → 194-frame context window (microWakeWord default ≈ 5 s look-back)
  → INT8 quantize (matching tflite-micro runtime expectations)
```

Total effective training-set size: 10k × 5 = **50k positive features** + 2.5k × 5 = **12.5k hard-negative features** + ~300h × hop_rate ≈ **40M bulk-negative windows** (subsampled to ~500k per epoch to keep the positive:negative ratio in the 1:10 range).

---

## 5. Training config (microWakeWord)

```yaml
model:
  architecture: micro_wake_word_inception
  inception:
    n_blocks: 3
    filters: [16, 32, 32]
    kernel_strides: [3, 3, 3]
  features:
    n_mels: 40
    win_ms: 60
    hop_ms: 25
    n_frames: 194

class_weights:
  positive: 3.0
  hard_negative: 2.0
  bulk_negative: 1.0

optimizer:
  name: adam
  learning_rate: 1e-3
  schedule: cosine
  warmup_steps: 500

training:
  batch_size: 256
  epochs: 50
  early_stop_metric: val_far_at_99_recall
  early_stop_patience: 5

quantization:
  scheme: int8_ptq
  representative_n: 1000
  target: tflite_micro_esp32s3
```

Hardware budget on RunPod:
- A40 48 GB SECURE ~$0.39/hr
- Estimated wall: 1–2 h for 50 epochs (typically converges by epoch 20–25)
- **Estimated cost per training run: $0.40–0.80**

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
