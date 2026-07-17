# CLAUDE.md — customWakeWord agent contract

This file is the entry point for Claude Code working inside this repo. Everything below tells you (the agent) the shape of the project, what each slash command does, and which subagent to dispatch for a given user intent.

---

## What this repo does

A pipeline for training custom wake-word models from a single phrase:

```
user phrase  →  init_wake.py  →  TTS synth  →  augment  →  microWakeWord train  →  INT8 .tflite  →  HF Hub + ESPHome
```

The phrase initializes a baseline project, but a release also requires held-out
audio, an evaluated threshold, and target-device validation.

---

## Layout you must know

| Path | Purpose |
|---|---|
| `scripts/init_wake.py` | Bootstrap a new wake-word project from a CLI flag |
| `scripts/synth_positives.py` | Piper TTS synthesis (English generator or configured ONNX voices) |
| `scripts/synth_hard_negatives.py` | TTS adversarials |
| `scripts/suggest_hard_negatives.py` | LLM-driven collision generator (Together/OpenAI/Ollama) |
| `scripts/collect_negatives.py` | Bulk negative corpora download |
| `scripts/build_features.py` | WAV → RaggedMmap MicroFrontend features with augmentation |
| `scripts/train_microwakeword.py` | Upstream MixedNet training + streaming INT8 export |
| `scripts/runpod_train.py` | RunPod A40 launcher with log server on :8001 |
| `scripts/emit_manifest.py` | Validated ESPHome v2 model manifest |
| `scripts/upload_to_hf.py` | HF Hub model + model card emit |
| `eval/runner.py` | FRR + FAR/hour scorer |
| `configs/examples/<slug>/` | Phrase, hard-negative, and training YAML |
| `data/<slug>/` | Per-project audio + features (gitignored) |
| `models/` | Trained .tflite (gitignored) |

---

## Slash commands you respond to

| Command | What you do |
|---|---|
| `/wake-new <slug> "<phrases>"` | Dispatch to the **wake-bootstrapper** subagent. It runs `init_wake.py`, then optionally `suggest_hard_negatives.py`. |
| `/wake-synth <slug>` | Dispatch to **data-synthesizer**. It runs positive + hard-neg TTS synthesis and collects bulk negatives. |
| `/wake-train <slug>` | Dispatch to **trainer**. It runs `build_features.py`, then `runpod_train.py`, polls via ScheduleWakeup, downloads the .tflite. |
| `/wake-eval <slug>` | Dispatch to **evaluator**. It runs `seed_eval_tasks.py` (if needed) and `eval/runner.py`, reports metrics. |
| `/wake-release <slug>` | Dispatch to **release-manager**. It runs `upload_to_hf.py`, generates the ESPHome YAML, opens a GitHub Release PR. |
| `/wake-status [<slug>]` | Read project state from disk. Report which steps have run, what artefacts exist, where to resume. |

Each slash command is a thin trigger; the real logic lives in the subagent file in `.claude/agents/`. Read the agent file before dispatching to understand its contract.

---

## Behavior rules

1. **Credentials**: every script that touches RunPod / HF / Together expects `source scripts/load_creds.sh` to have been run. The agents source it on your behalf via Bash. If a token is missing, ask the user to populate `~/.config/tofu-wake/<service>.env`.

2. **Long-running jobs**: training runs take 1–2 h on RunPod. Use `ScheduleWakeup` with `delaySeconds=270` for active polling once the pod is up; switch to `delaySeconds=1200` for the long-tail finish wait. Do NOT busy-poll the pod's log server in a tight loop.

3. **Destructive actions** (delete data dirs, force-push, drop HF repos): confirm with the user before running.

4. **No invented file paths**: when a script's CLI flag wants a path, infer it from the slug (`data/<slug>/...`, `configs/examples/<slug>/...`). Don't make up new locations.

5. **Match the user's wake-word target**:
   - Default is `microWakeWord` for ESP32-S3 (INT8 TFLite Micro).
   - If the user asks for Raspberry Pi/openWakeWord or ESP-SR/WakeNet, state that
     the requested training/export path is not shipped in this repository.

6. **When unsure**: read [`PIPELINE.md`](PIPELINE.md), [`PLAN.md`](PLAN.md), [`EXAMPLES.md`](EXAMPLES.md). They are the authoritative docs.

---

## Defaults

- Project slug: Unicode-safe with no spaces; ASCII ids remain recommended for cloud tooling.
- Default positives count: 10,000 across 4 phrases (5k + 2.5k + 1.5k + 1k).
- Default hard-negs count: 2,500.
- Default training negatives: upstream precomputed speech and dinner-party features.
- Default GPU: A40 SECURE on RunPod (~$0.39/hr).
- Default eval thresholds: target FRR ≤ 5%, FAR/hour ≤ 1.0, .tflite ≤ 100 kB.

---

## Memory hooks

If the user mentions a project slug in conversation, prefer their slug over the default `tofu`. If they reference an HF repo id, persist it to `configs/examples/<slug>/release.yaml` so subsequent `/wake-release` calls find it without re-asking.
