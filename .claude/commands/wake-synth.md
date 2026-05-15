---
description: Synthesize TTS positives + hard-negatives and collect bulk negatives for a wake-word project. Usage `/wake-synth <slug>`.
---

The user just invoked `/wake-synth`. The first positional arg is the project slug. If missing, ask.

Verify `configs/examples/<slug>/wake_phrases.yaml` exists. If not, suggest `/wake-new <slug> "<phrases>"` first and stop.

Dispatch to the **data-synthesizer** subagent with the slug. Pass through any extra flags the user gave (`--count`, `--negatives-only`, `--skip-bulk`).

After it returns, surface the counts (N positives, M hard-negs, X h bulk) and suggest the next step (`/wake-train <slug>`).
