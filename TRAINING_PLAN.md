# TRAINING_PLAN — Tofu wake-word v0

**Target deployment**: ESP32-S3 via ESPHome's `micro_wake_word` component.
**Wake phrases**: `hey tofu`, `hi tofu`, `hello tofu`, `okay tofu`.
**License posture**: hobby — CC-BY-NC negative corpora allowed; commercial-redeploy requires substitution.

| Header | Value |
|---|---|
| Total compute budget | **$2–6** (target: $3) |
| Wall time, first attempt | **~3 hours active** (mostly waiting on TTS + bulk download) |
| Wall time, iteration | **~30–45 min/cycle** |
| Confidence v0 hits **hobby-grade** targets¹ | **~85%** |
| Confidence v0 hits **production-grade** targets² | **~55%** without iteration; **~80%** after 2–3 iterations |

¹ hobby-grade ≡ recall ≥ 90%, FA/hr ≤ 2 on bulk speech/music, no triggers on the word "tofu" alone in normal speech.
² production-grade ≡ recall ≥ 97%, FA/hr ≤ 0.5 on DipCo, matches the shipped Okay Nabu / Hey Jarvis quality bar.

---

## 1. Architecture decision

### Picked: **MixedNet via upstream `OHF-Voice/micro-wake-word`** (formerly `kahrendt/microWakeWord`)

Rationale:
- It's the architecture ESPHome's `micro_wake_word` v2 component is designed for. Hey Jarvis, Okay Nabu, Alexa, and the VAD model all ship as MixedNet `.tflite`s with INT8 quantization and `feature_step_size: 10`, `sliding_window_size: 5`.
- The repo includes streaming-Conv wrappers (`microwakeword/layers/stream.py`) that produce stateful TFLite suitable for `tflite-micro` without architectural surgery. My from-scratch Keras reimpl in the current scaffold would NOT produce a deployable model — confirmed by reading upstream's `convert_saved_model_to_tflite` which uses `_experimental_variable_quantization=True` to keep streaming state variables quantized.
- Active community: third-party "Hey Frank" build by malonestar hit **0.103 FA/hr with 97.58% recall** on a 50k-positive variant of the canonical recipe.

### Rejected: from-scratch Keras Inception (current scaffold)

The script I wrote in `scripts/train_microwakeword.py` produces a model that:
- Has wrong tensor shapes for ESPHome runtime (no streaming-state I/O).
- Skips the `audio_microfrontend` PCAN/noise-reduction preprocessing the on-device frontend uses (it's a custom TFLite op via `pymicro-features`).
- Quantizes to vanilla INT8 instead of `inference_input_type=int8, output_type=uint8` with quantized variables.

→ **Will refactor to thin wrapper around upstream before training.**

---

## 2. Data plan

### 2.1 Positives — Piper TTS

| Item | Value | Source |
|---|---|---|
| Engine | piper-sample-generator (`kahrendt/piper-sample-generator` `mps-support` branch for Mac, or `rhasspy/piper-sample-generator` for Linux) | <https://github.com/kahrendt/piper-sample-generator/tree/mps-support> |
| Voice model | `en_US-libritts_r-medium.pt` (PyTorch generator) — **904 distinct speakers** from LibriTTS-R | <https://github.com/rhasspy/piper-sample-generator/releases/tag/v2.0.0> |
| Sample count per phrase | 5,000 — `hey tofu` is primary | |
| Phrases | 4 (hey/hi/hello/okay tofu) | configs/examples/tofu/wake_phrases.yaml |
| **Total positives** | **20,000 WAV @ 16 kHz mono** | |
| Variation grid | 3 length-scales × 904² speaker pairs × stochastic noise | upstream default |
| Generation time, 4090 | ~3–5 minutes for 20k @ batch 100 | extrapolated from "100 samples/sec on 2080 Ti" |
| Generation time, M2/M3 Pro MPS | ~30–60 minutes | extrapolated, no published benchmark |

**20k is 2× the canonical notebook's 10k and 0.4× malonestar's 50k**. Sweet spot for a one-shot v0 — enough diversity, fast enough to iterate.

### 2.2 Hard negatives — Piper TTS, adversarial

Same engine, targeting the collision set in `configs/examples/tofu/hard_negatives.yaml` plus LLM-generated additions via `scripts/suggest_hard_negatives.py`:

| Bucket | Examples | Count |
|---|---|---|
| Phonetic /tu/+/fu/ collisions | "to-do list", "to-go", "to you", "two-fold" | 600 |
| Rhyme insults | "hey doofus", "you doofus" | 400 |
| Greetings, no tofu | "hey there", "hi there", "hello world" | 300 |
| "tofu" alone — **critical** | "I love tofu", "where's the tofu", "tofu salad" | 500 |
| Food context | "fried tofu", "silken tofu", "tofu burger" | 300 |
| Disyllabic /tu/ | "tutu", "two pugs", "to-do" | 200 |
| Near-match nouns | "toad", "ufo", "boo who" | 200 |
| **Total** | | **2,500** |

Augmented with `penalty_weight: 5.0` per malonestar's successful pattern, much higher than upstream's default `1.0` — this teaches the model to actively suppress these phrases, not just ignore them.

### 2.3 Bulk negatives — pre-built HF mmap dataset

| File | Size | Content | Sampling weight |
|---|---|---|---|
| `speech.zip` | 2.96 GB | LibriSpeech `train-other-500` + VOiCES, ≈300 h | 10.0 |
| `dinner_party.zip` | 0.41 GB | CHiME6 train | 15.0 (raised from default 10) |
| `no_speech.zip` | 1.86 GB | FMA-medium + FSD50K + WHAM (music + ambient) | 5.0 |
| `dinner_party_eval.zip` | 0.08 GB | CHiME6 dev/eval (validation/test only) | 0.0 (val/test split) |
| `speech_background.zip` | 2.65 GB | validation ambient | (auto-routed) |
| `dinner_party_background.zip` | 0.14 GB | validation ambient | (auto-routed) |
| `no_speech_background.zip` | 0.94 GB | validation ambient | (auto-routed) |
| **Total** | **~9 GB** | already-computed spectrograms in RaggedMmap format | |

License: **CC-BY-NC-4.0**. Hobby use only as currently specified. If Tofu ships commercially, substitute with MUSAN + AudioSet + Common Voice (which my `scripts/collect_negatives.py` already supports).

Source: <https://huggingface.co/datasets/kahrendt/microwakeword>

### 2.4 Augmentation — applied at feature-generation time

Per `microwakeword/audio/augmentation.py` defaults from the canonical notebook:

```python
augmentation_probabilities = {
    "SevenBandParametricEQ": 0.1, "TanhDistortion": 0.1,
    "PitchShift": 0.1, "BandStopFilter": 0.1,
    "AddColorNoise": 0.1, "AddBackgroundNoise": 0.75,
    "Gain": 1.0, "RIR": 0.5
}
background_min_snr_db = -5, background_max_snr_db = 10
augmentation_duration_s = 3.2
slide_frames = 10  # repetition per positive
```

Augmentation corpora (auto-downloaded into the pod):
- RIR: `davidscripka/MIT_environmental_impulse_responses` (HF) → ~270 real IRs
- Music: `mchl914/fma_xsmall` → ~8 GB of FMA music
- Noise: AudioSet `bal_train09.tar` shard from `agkphysics/AudioSet`

Each of the 20k positives is **augmented into ~10 distinct windows** via `slide_frames=10`, yielding **~200k positive training spectrograms**.

---

## 3. Pipeline stages

### Stage 0 — Refactor (~2 hours of code work, no compute $)

Replace `scripts/train_microwakeword.py` with a wrapper that:

```python
# pseudo-code
import argparse
import microwakeword.data as input_data
import microwakeword.mixednet as mixednet
from microwakeword.model_train_eval import load_config, train_model, evaluate_model

flags = build_argparse_namespace(
    training_config="training_parameters.yaml",
    train=1, restore_checkpoint=0, test_tflite_streaming_quantized=1,
    use_weights="best_weights",
    model_name="mixednet",
    pointwise_filters="64,64,64,64",
    repeat_in_block="1,1,1,1",
    mixconv_kernel_sizes="[5],[7,11],[9,15],[23]",
    residual_connection="0,0,0,0",
    first_conv_filters=32,
    first_conv_kernel_size=5,
    stride=3,
)
config = load_config(flags, mixednet)
data_processor = input_data.FeatureHandler(config)
model = mixednet.model(flags, config["training_input_shape"], config["batch_size"])
train_model(config, model, data_processor, restore_checkpoint=False)
evaluate_model(config, model, data_processor, ...)  # emits .tflite
```

Replace `scripts/build_features.py` with one that:
1. Reads positive/hard-negative WAV manifests from our existing `synth_*.py` output.
2. For each WAV, calls `microwakeword.audio.audio_utils.generate_features_for_clip(audio, step_ms=10)` to get `(n_frames, 40) uint16` spectrograms.
3. Wraps the generator with `mmap_ninja.ragged.RaggedMmap.from_generator(out_dir, sample_generator, batch_size=100)`.
4. Writes mmaps at `data/tofu/features/{training,validation,testing}/wakeword_mmap/` matching upstream's expected layout.

Replace `scripts/export_tflite.py` with a no-op (upstream's `evaluate_model(..., test_tflite_streaming_quantized=1)` produces the deployable artefact at `trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`).

Add `scripts/emit_manifest.py` that writes the ESPHome v2 manifest JSON:

```json
{
  "type": "micro",
  "wake_word": "Tofu",
  "author": "<user>",
  "model": "tofu_wakeword_v0.tflite",
  "trained_languages": ["en"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.85,
    "feature_step_size": 10,
    "sliding_window_size": 5,
    "tensor_arena_size": 30000,
    "minimum_esphome_version": "2024.7.0"
  }
}
```

The `probability_cutoff` value will be filled in from the operating-point table the training script prints.

### Stage 1 — Bootstrap & data (~3 h wall, ~1 h active)

On RunPod 4090 SECURE pod (or A40 if 4090 unavailable):

```bash
# A. Repo clone + deps
git clone https://github.com/temm1e-labs/customWakeWord && cd customWakeWord
pip install -e .  # installs our package + microwakeword as dep

# B. Piper sample gen — 20k positives across 4 phrases (~10 min)
for phrase in "hey tofu" "hi tofu" "hello tofu" "okay tofu"; do
    python3 piper-sample-generator/generate_samples.py "$phrase" \
        --max-samples 5000 --batch-size 200 \
        --output-dir data/tofu/synth/positives/$(slug $phrase)
done

# C. Piper sample gen — 2.5k hard negatives (~3 min)
python scripts/synth_hard_negatives.py --phrases configs/examples/tofu/hard_negatives.yaml \
    --out data/tofu/synth/hard_negatives --count 2500

# D. Download HF negative datasets (~9 GB, ~3 min on RunPod's egress)
python scripts/download_hf_negatives.py --out data/negative_datasets

# E. Feature extraction (~10 min, 16 cores)
python scripts/build_features.py --project tofu \
    --positives data/tofu/synth/positives \
    --hard-negatives data/tofu/synth/hard_negatives \
    --augmentation-corpora-from-hf \
    --out data/tofu/features
```

### Stage 2 — Train (~15–25 min wall)

```bash
python scripts/train_microwakeword.py --project tofu \
    --training-config configs/examples/tofu/training_parameters.yaml
```

Training config (10k steps single-phase for v0; can extend to two-phase if metrics demand):

```yaml
window_step_ms: 10
train_dir: trained_models/tofu
features:
  - features_dir: data/tofu/features
    sampling_weight: 2.0
    penalty_weight: 1.0
    truth: true
    truncation_strategy: truncate_start
    type: mmap
  - features_dir: data/tofu/hard_negatives_features
    sampling_weight: 3.0
    penalty_weight: 5.0       # high penalty — actively suppress
    truth: false
    truncation_strategy: random
    type: mmap
  - features_dir: data/negative_datasets/speech
    sampling_weight: 10.0
    penalty_weight: 1.0
    truth: false
    truncation_strategy: random
    type: mmap
  - features_dir: data/negative_datasets/dinner_party
    sampling_weight: 15.0
    penalty_weight: 3.0
    truth: false
    truncation_strategy: random
    type: mmap
  - features_dir: data/negative_datasets/no_speech
    sampling_weight: 5.0
    penalty_weight: 1.0
    truth: false
    truncation_strategy: random
    type: mmap
  - features_dir: data/negative_datasets/dinner_party_eval
    sampling_weight: 0.0
    penalty_weight: 1.0
    truth: false
    truncation_strategy: split
    type: mmap
training_steps: [10000]
positive_class_weight: [1]
negative_class_weight: [20]
learning_rates: [0.001]
batch_size: 128
clip_duration_ms: 1500
eval_step_interval: 500
target_minimization: 0.5
minimization_metric: average_viable_false_positives_per_hour
maximization_metric: average_viable_recall
```

### Stage 3 — Eval + threshold tuning (~5–10 min wall)

The trainer prints a cutoff table at end-of-training:
```
cutoff | recall | far/hr
0.50   | 0.99   | 1.2
0.70   | 0.98   | 0.6
0.85   | 0.96   | 0.3
0.95   | 0.91   | 0.1
```

Pick the cutoff where `recall ≥ 0.95` AND `far/hr ≤ 0.5`. Write it into `manifest.json:probability_cutoff`.

Run `eval/runner.py` against our held-out positives to double-check:
```bash
python -m eval.runner --project tofu \
    --model trained_models/tofu/stream_state_internal_quant.tflite \
    --threshold <picked-cutoff>
```

### Stage 4 — Manifest + release (~5 min wall)

```bash
python scripts/emit_manifest.py --project tofu --threshold <picked-cutoff>
python scripts/upload_to_hf.py --project tofu --repo-id <user>/tofu-wakeword-v0
gh release create v0.1.0 trained_models/tofu/stream_state_internal_quant.tflite \
    configs/examples/tofu/manifest.json \
    configs/examples/tofu/esphome.yaml \
    --title "tofuWakeWord v0.1.0"
```

### Stage 5 — On-device test

Flash an ESP32-S3-DevKitC-1 + INMP441 I2S mic with the ESPHome YAML and exercise:
- 20 utterances of each phrase from 1m / 3m / through-wall
- 1 hour of background podcast/TV/music — count false fires
- 10 utterances of "I love tofu" / "tofu burger" — must NOT fire

Gate on the v0 acceptance criteria in §5.

---

## 4. Budget

### Compute

| Item | Hardware | Wall | $/hr | Cost |
|---|---|---|---|---|
| First training run | RunPod 4090 SECURE | 1.0 h (incl. setup) | $0.69 | **$0.69** |
| Bulk download cache (network volume) | — | persistent | $0.05/GB/mo × 50 GB | **$0.25/mo** |
| Iteration runs (×3 expected) | RunPod 4090 SECURE | 0.5 h each | $0.69 | **$1.04** |
| Hard-neg LLM suggestions (Together) | — | one-shot | — | **$0.001** |
| HuggingFace Hub | free | — | — | $0 |
| GitHub | free (private repo, public release) | — | — | $0 |
| **Subtotal** | | | | **$2.00** |
| Contingency (spot interruption, retry, debugging) | | | | **$1.00–4.00** |
| **TOTAL EXPECTED** | | | | **$2–6** |

If using 4090 COMMUNITY (spot, $0.34/hr), halve all GPU costs. ~$1–3 total.

### Time

| Item | Hours |
|---|---|
| Stage 0 — refactor scripts | ~2 active |
| Stage 1 — data (mostly downloads) | ~1 wall, ~10 min active |
| Stage 2 — train | ~0.3 wall, fully unattended |
| Stage 3 — eval | ~0.2 active |
| Stage 4 — release | ~0.2 active |
| Stage 5 — on-device test | ~1 active |
| **TOTAL FIRST CYCLE** | **~5 hours total** (~3 h active) |
| **Per iteration** | **~45 min** (~20 min active) |

---

## 5. Acceptance criteria

### v0 ship gate (hobby)

| Metric | Target |
|---|---|
| Recall on held-out positives | ≥ 90% |
| FA/hr on `dinner_party_eval` (CHiME6 ambient speech) | ≤ 2.0 |
| FA on each hard-negative bucket | ≤ 10% |
| FA on "I love tofu" / "tofu salad" / "tofu burger" | ≤ 5% |
| `.tflite` size | ≤ 50 kB |
| `tensor_arena_size` | ≤ 35 kB |
| On-device inference latency | ≤ 10 ms |

### Stretch (production-grade — matches Okay Nabu / Hey Jarvis)

| Metric | Target |
|---|---|
| Recall | ≥ 97% |
| FA/hr | ≤ 0.3 |
| FA on contextual "tofu" uses | ≤ 1% |

---

## 6. Risk register & confidence

### Confidence assessment per stage

| Stage | Confidence works first try | Why |
|---|---|---|
| Stage 0 (refactor) | 95% | upstream API documented in research; wrapper is mechanical |
| Stage 1 (data) | 85% | HF dataset is mature; Piper has well-documented quirks; main risk is bandwidth on bulk download |
| Stage 2 (train) | 90% | canonical recipe widely used; main risks are Python version drift (issue #62) and Jupyter kernel deadlocks (#74) |
| Stage 3 (eval + threshold) | 90% | upstream prints the cutoff table; we just pick |
| Stage 4 (release) | 99% | HF / GitHub uploads are trivial |
| Stage 5 (device test) | 70% | the real risk — see below |

### Top 5 risks, ranked

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **Model triggers on "I love tofu" / "tofu salad" in normal speech** | Medium-High | Critical for hobby UX | Bucket has 500 hard-negs @ `penalty_weight: 5.0`. If still failing → 2nd iteration with 2x bucket size. Probably needs real-mic recordings to definitively close. |
| 2 | **High FA/hr on TV/music/dinner-party scenes** | Medium | Critical | Raise `negative_class_weight` to 50; bump `dinner_party.sampling_weight` to 20. malonestar hit 0.103 FA/hr with these knobs. |
| 3 | **INT8 quantization collapse** (issue #90 — float AUC 0.999, int8 AUC 0.021) | Low-Medium | Catastrophic | Stay on the canonical `pointwise_filters="64,64,64,64"`; don't tune architecture in v0. Add `--use_weights="best_weights"` to evaluate float model first. |
| 4 | **Children's voices in deployment** — Piper LibriTTS-R is adult-speaker-skewed | Medium | Moderate (toy use case) | Defer to v1. Workaround: when Piper voices are exhausted, add a small batch from Kokoro's `af_river` / `am_puck` (child-leaning timbres). Or: have a kid sit down with the toy and record 50 wakes; weight at 5x. |
| 5 | **CC-BY-NC license on HF negatives** prevents commercial deployment | High if scope changes | Moderate | Already noted — for hobby OK. For commercial: substitute MUSAN+CV+AudioSet (my `collect_negatives.py` already does this; just plumb it into the mmap build). |

### Other notable failure modes from upstream issues

- **Python 3.11 incompatibility** (issue #62) → pin pod to `python:3.10` Docker image.
- **`piper-sample-generator` silent clone failure** (issue #94) → assert files exist before training.
- **Kernel deadlock on `Clips` import** (issue #74) → restart Jupyter kernel after `pip install -e .` if running interactively.
- **Streaming-state mismatch** (issue #69) → strictly use upstream's `convert_model_saved(..., mode=Modes.STREAM_INTERNAL_STATE_INFERENCE)` path; don't hand-roll.
- **`tensor_arena_size` allocation failure** on device (esphome/issues/7242) → start at 30000, bump if needed; ensure PSRAM is enabled in ESPHome.

---

## 7. Iteration playbook

If v0 misses targets, the loop is **~45 min per iteration**:

### "FA/hr too high"
1. Raise `negative_class_weight` from 20 → 30 → 50.
2. Raise `dinner_party.sampling_weight` from 15 → 20 → 25.
3. Raise `penalty_weight` on the worst-misfiring bucket.
4. Re-train (15 min) → re-eval (5 min).

### "Recall too low"
1. More positives: regenerate Piper with `--max-samples 10000` per phrase (10 min).
2. Lower `probability_cutoff` from 0.85 → 0.7.
3. If still failing: switch to **two-phase training** (`training_steps: [25000, 20000]`) per malonestar pattern.

### "Triggers on 'I love tofu'"
1. Add 500 more hand-curated synthetic clips of contextual `tofu` use.
2. Boost that bucket's `penalty_weight` from 5 → 10.
3. Re-train.

### "Triggers on TV podcast"
1. Sample a real 1-hour podcast that triggers it.
2. Add it to `no_speech` mmap.
3. Re-train.

---

## 8. Go / no-go

We are **GO** subject to user confirmation. Before running, I need approval to:

1. **Refactor 3 scripts** to use upstream microWakeWord (Stage 0). ~2 hours of code work, no compute.
2. **Spin a RunPod 4090 SECURE pod** for ~1 hour first run (~$0.70).
3. **Use the CC-BY-NC HF negative datasets** for v0 (hobby use OK; production swap-in is documented).
4. **Push intermediate artefacts to the GitHub repo and HuggingFace Hub** (model card, eval JSON, manifest).
5. Operate within a **$10 hard budget cap** — bail and re-evaluate before exceeding.

### What "ready to train" requires

Once you say go:
- [ ] Stage 0 refactor merged to `main`
- [ ] `~/.config/tofu-wake/{hf,runpod}.env` populated
- [ ] `pip install -r requirements.txt` clean on Mac
- [ ] Smoke test: `python scripts/init_wake.py --name tofu --force` reproduces existing config
- [ ] Smoke test: a single Piper sample generates locally on Mac (validates the MPS path)
- [ ] Pod size confirmed: 4090 SECURE @ $0.69/hr or A40 SECURE @ $0.44/hr as fallback

After that, `/wake-train tofu` is one command and we're off.

---

## Sources

- **microWakeWord upstream**: <https://github.com/OHF-Voice/micro-wake-word> (formerly `kahrendt/microWakeWord`)
- **Canonical training notebook**: <https://github.com/OHF-Voice/micro-wake-word/blob/main/notebooks/basic_training_notebook.ipynb>
- **piper-sample-generator**: <https://github.com/rhasspy/piper-sample-generator>, MPS fork <https://github.com/kahrendt/piper-sample-generator/tree/mps-support>
- **HF negative datasets**: <https://huggingface.co/datasets/kahrendt/microwakeword>
- **Reference "Hey Frank" build** (0.103 FA/hr, 97.58% recall): <https://github.com/malonestar/custom-micro-wake-word-model>
- **ESPHome v2 manifests** (Okay Nabu, Hey Jarvis, Alexa): <https://github.com/esphome/micro-wake-word-models/tree/main/models/v2>
- **ESPHome micro_wake_word component**: <https://esphome.io/components/micro_wake_word/>
- **HA community thread on custom wake words**: <https://community.home-assistant.io/t/home-assistant-voice-pe-custom-wake-words-please/845139>
- **RunPod 4090 pricing**: <https://www.runpod.io/gpu-models/rtx-4090>
- **Known issues**: <https://github.com/OHF-Voice/micro-wake-word/issues> (#62, #69, #74, #79, #90, #94)
