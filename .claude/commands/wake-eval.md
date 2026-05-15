---
description: Run held-out eval on a trained wake-word .tflite, report FRR + FAR + per-bucket metrics. Usage `/wake-eval <slug>` (optionally `--threshold 0.9`).
---

The user just invoked `/wake-eval`. First positional arg = project slug.

Dispatch to the **evaluator** subagent. It will:
1. Verify `models/<slug>-wakeword-v0.tflite` exists.
2. Seed `eval/tasks/<slug>/` if empty (via `scripts/seed_eval_tasks.py`).
3. Run `python -m eval.runner`.
4. Report a metrics table + per-bucket breakdown.
5. Give a verdict (ship-ready, needs more X) and suggest the next action.

If the user gave `--threshold`, pass it through. If they asked for a sweep, run the threshold loop.
