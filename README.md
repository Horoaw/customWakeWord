# customWakeWord

> A repeatable, agent-driven pipeline for training **any custom wake word** end-to-end — from a single user-supplied phrase to a deployable INT8 TFLite Micro model on **ESP32-S3**. Synthesize the data, train the model, eval against adversarial collisions, and ship to HuggingFace + ESPHome — with one slash command per stage.

Give it a phrase. It gives you a wake-word model.

> ⚠ **Firmware compatibility.** Our output is an INT8 `.tflite` for
> ESPHome's `micro_wake_word`. It does **not** load on XiaoZhi,
> ESP-S3-BOX-3 stock firmware, or anything else built on Espressif's
> `esp-sr` — those expect a `wn9_*.bin` WakeNet model in a proprietary
> format with no public converter. See
> [`RELEASE_PROTOCOL.md`](RELEASE_PROTOCOL.md) for the full target matrix
> and the five paths to producing an ESP-SR companion model.

```
python scripts/init_wake.py --name sunny --phrases "hey sunny,hi sunny,okay sunny"
/wake-synth sunny       # synthesize ~12k positives + ~2.5k hard-negatives via TTS
/wake-train sunny       # train microWakeWord on RunPod
/wake-eval sunny        # FRR + FAR/hour against held-out tasks
/wake-release sunny     # push to HuggingFace Hub + emit ESPHome YAML
```

That's the whole flow. Each slash command is a Claude Code agent that orchestrates the underlying Python scripts; the scripts run independently as a plain CLI if you don't use Claude Code.

---

## What it is

A toolkit that bundles:

1. **A TTS data-synthesis pipeline** that pulls from four permissive open-source engines (Piper, Kokoro, MeloTTS, Parler-TTS) for ~200 distinct voices, accents, and prosody settings — all free, all local-CPU.

2. **An LLM-assisted hard-negative generator** that takes your wake phrase and proposes likely false-fire collisions (`scripts/suggest_hard_negatives.py`). You can also hand-author the collision list.

3. **A standard audiomentations chain** (RIR convolution + additive noise from MUSAN/DEMAND + codec degradation + pitch/speed jitter) so the model survives cheap MEMS mics and noisy rooms.

4. **A [microWakeWord](https://github.com/kahrendt/microWakeWord)-based training loop** that produces an INT8 `.tflite` <100 kB, <10 ms inference on ESP32-S3, runnable via [ESPHome](https://esphome.io/components/micro_wake_word/) with a single YAML stanza.

5. **A held-out eval harness** with concrete operating-point metrics (FRR ≤ 5%, FAR ≤ 1/hour) so you can ship vs. iterate with a clear signal.

6. **A `.claude/` agent layer** (CLAUDE.md + agents + slash commands) so the whole pipeline runs from natural language inside Claude Code — RunPod provisioning, polling, model card generation, HF Hub upload all driven by agents.

---

## Pipeline at a glance

```
  $ python scripts/init_wake.py --name <slug> --phrases "<phrase1>,<phrase2>,…"
  configs/examples/<slug>/{wake_phrases,hard_negatives}.yaml
        │
        ▼   Piper + Kokoro + MeloTTS + Parler-TTS
  data/<slug>/synth/positives/         (~10k WAV)
        │
        ▼   adversarial: from LLM (scripts/suggest_hard_negatives.py) + hand
  data/<slug>/synth/hard_negatives/    (~2.5k WAV)
        │
        ▼   bulk: MUSAN + DEMAND + Common Voice + AudioSet
  data/raw/negatives/                  (~300 h, shared across projects)
        │
        ▼   audiomentations: RIR + noise + codec + EQ + pitch/speed
  data/<slug>/clean/{train,val,test}.tfrecord
        │
        ▼   microWakeWord (TF/Keras, INT8 QAT/PTQ)
        │   on RunPod A40, ~$0.40, target ~1 h
  models/<slug>-wakeword-v0.tflite  (<100 kB INT8 for ESP32-S3)
        │
        ▼   eval/runner.py — FRR on positives, FAR/hour on bulk audio
  eval/results/<slug>-v0__<ts>.json
        │
        ▼   HuggingFace Hub + ESPHome YAML snippet + GitHub Release
```

See [`PIPELINE.md`](PIPELINE.md) for the exact commands and [`RESEARCH.md`](RESEARCH.md) for the literature/model survey that informs every choice.

---

## Quick start — the worked example

**[Tofu](configs/examples/tofu/)** is the example wake word that ships with this repo: a robotic toy that wakes on "hey tofu" / "hi tofu" / "hello tofu" / "okay tofu". See [`PLAN.md`](PLAN.md) for the v0 numbers (10k positives, 2.5k hard-negs, 300 h bulk, FRR ≤ 5% / FAR ≤ 1/hr targets).

```bash
git clone https://github.com/temm1e-labs/customWakeWord && cd customWakeWord
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# credentials (place in ~/.config/tofu-wake/{hf,gh,together,runpod}.env, chmod 600)
source scripts/load_creds.sh

# 1. Synthesize Tofu positives (~30 min on Mac CPU)
python scripts/synth_positives.py \
    --phrases configs/examples/tofu/wake_phrases.yaml \
    --out data/tofu/synth/positives \
    --count 10000

# 2. Synthesize hard negatives (~5 min)
python scripts/synth_hard_negatives.py \
    --phrases configs/examples/tofu/hard_negatives.yaml \
    --out data/tofu/synth/hard_negatives \
    --count 2500

# 3. Bulk negatives (~30 GB, ~1 h)
python scripts/collect_negatives.py --out data/raw/negatives

# 4. Augment + build train/val/test splits
python scripts/build_features.py --project tofu --out data/tofu/clean

# 5. Train on RunPod (~$0.40, ~1 h)
python scripts/runpod_train.py --project tofu

# 6. Eval
python -m eval.runner --project tofu \
    --model models/tofu-wakeword-v0.tflite \
    --out eval/results/tofu-v0__$(date +%s).json

# 7. Publish
python scripts/upload_to_hf.py --project tofu \
    --repo-id <you>/tofu-wakeword-v0
```

---

## Quick start — your own wake word

```bash
# 1. Bootstrap a fresh project. Generates configs/examples/<name>/{wake_phrases,hard_negatives}.yaml
python scripts/init_wake.py --name sunny \
    --phrases "hey sunny,hi sunny,hello sunny,okay sunny"

# 2. (Recommended) ask an LLM for harder, less-obvious collisions:
python scripts/suggest_hard_negatives.py --name sunny

# 3-7. Follow the Tofu example above with `--project sunny` everywhere.
```

With Claude Code, the whole flow is:

```
/wake-new sunny "hey sunny,hi sunny,hello sunny,okay sunny"
/wake-synth sunny
/wake-train sunny
/wake-eval sunny
/wake-release sunny
```

The agents in `.claude/agents/` handle credentials, RunPod polling, model card generation, and ESPHome YAML emission. See [`CLAUDE.md`](CLAUDE.md) for the agent contract.

---

## Repository contents

```
customWakeWord/
├── README.md                   ← this file
├── CLAUDE.md                   ← entry point for Claude Code agents
├── PIPELINE.md                 ← exact technical recipe (generic)
├── RESEARCH.md                 ← wake-word literature + model survey
├── PLAN.md                     ← v0 plan numbers for the Tofu example
├── EXAMPLES.md                 ← Tofu, sunny, jarvis worked examples
├── REPLICATE.md                ← step-by-step end-to-end replication
├── BUDGET_LOG.md               ← line-by-line compute spend tracker
├── LICENSE                     ← Apache 2.0
│
├── configs/
│   ├── templates/              ← wake_template.yaml + hard_negatives_template.yaml
│   ├── examples/
│   │   └── tofu/               ← worked example: phrases.yaml + hard_negatives.yaml + README
│   ├── data.yaml               ← shared augmentation chain config
│   ├── train.yaml              ← shared microWakeWord training hyperparams
│   └── templates/esphome_template.yaml
│
├── data/                       ← (gitignored) per-project + shared corpora
│   ├── <project>/synth/        ← project-specific TTS outputs
│   ├── <project>/clean/        ← project-specific TFRecord splits
│   └── raw/negatives/          ← MUSAN/DEMAND/CV/AudioSet (shared across projects)
│
├── eval/
│   ├── tasks/<project>/        ← per-project held-out tasks
│   ├── runner.py               ← FRR + FAR/hour scorer for .tflite
│   ├── schema.py               ← EvalTask + EvalResult dataclasses
│   └── results/                ← per-project JSON eval outputs
│
├── scripts/
│   ├── load_creds.sh           ← source HF/GH/Together/RunPod tokens
│   ├── init_wake.py            ← bootstrap a new wake-word project from a phrase
│   ├── suggest_hard_negatives.py ← LLM-driven adversarial generator
│   ├── synth_positives.py      ← Piper + Kokoro + MeloTTS + Parler-TTS
│   ├── synth_hard_negatives.py ← same engines, adversarial phrases
│   ├── collect_negatives.py    ← MUSAN/CV/DEMAND/AudioSet downloader
│   ├── augment.py              ← shared augmentation chain (module, not CLI)
│   ├── build_features.py       ← WAV → 40-mel → TFRecord splits
│   ├── train_microwakeword.py  ← TF/Keras training loop wrapper
│   ├── runpod_train.py         ← RunPod A40 launcher with log server
│   ├── export_tflite.py        ← INT8 PTQ + TFLite Micro export for ESP32-S3
│   ├── upload_to_hf.py         ← model + ESPHome YAML → HuggingFace Hub
│   └── seed_eval_tasks.py      ← hand-curate held-out test set from TFRecord
│
├── models/                     ← trained .tflite checkpoints (gitignored)
└── .claude/
    ├── agents/                 ← data-synthesizer, trainer, evaluator, release-manager
    └── commands/               ← /wake-new, /wake-synth, /wake-train, /wake-eval, /wake-release, /wake-status
```

---

## Why microWakeWord (and when to switch)

For **MCU-class deployments** (ESP32-S3, ESP-IDF, ESPHome), microWakeWord is the default and the only mature option in 2025–26. INT8 TFLite Micro, <10 ms streaming inference, <100 kB on disk, ships in Home Assistant Voice PE.

If your wake-word needs to run on a **Raspberry Pi-class SBC**, swap in [openWakeWord](https://github.com/dscripka/openWakeWord) — the data pipeline above feeds it identically; only `scripts/train_microwakeword.py` and `scripts/export_tflite.py` change. We've left the openWakeWord head untrained in v0 to keep the surface focused; see [`RESEARCH.md`](RESEARCH.md) for the comparison.

If your wake-word needs to run on a **microcontroller smaller than ESP32-S3** (Cortex-M4, etc.), microWakeWord's published model size may still fit but Picovoice Porcupine has stronger production support — note its [free-tier limits](https://picovoice.ai/pricing/) before shipping commercially.

---

## Status

- **2026-05-15** — v0 scaffold landed. Generic toolkit + Tofu worked example. No models trained yet. See [`PLAN.md`](PLAN.md) for the next steps.

---

## License

Apache License 2.0 for the code in this repo. See [LICENSE](LICENSE).

Training data — RIRs, bulk negatives — are downloaded from third parties at their own licenses (mostly CC-BY / CC0); see [`RESEARCH.md`](RESEARCH.md) for the per-corpus terms.

Published `.tflite` artefacts default to Apache 2.0 — synthesized from permissively-licensed Piper / Kokoro / MeloTTS / Parler-TTS voices specifically so the model can ship without license entanglement.
