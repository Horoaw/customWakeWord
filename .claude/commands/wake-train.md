---
description: Build features and launch microWakeWord training on RunPod for a wake-word project. Usage `/wake-train <slug>` (optionally `--hf-repo-id <id>`).
---

The user just invoked `/wake-train`. First positional arg = project slug.

Pre-flight: verify the data is ready (see `.claude/agents/trainer.md` for the exact checks). If not, suggest `/wake-synth <slug>` and stop.

Dispatch to the **trainer** subagent. It will:
1. Build features locally (if not already done).
2. Launch `scripts/runpod_train.py`.
3. Capture `pod_id` + log URL.
4. Poll via ScheduleWakeup until the `_done` marker appears or MAX_WAIT_S hits.
5. Report runtime, cost, and final metrics.

If the user gave `--hf-repo-id`, pass it through so the pod auto-uploads on completion.

After the subagent returns, suggest `/wake-eval <slug>`.
