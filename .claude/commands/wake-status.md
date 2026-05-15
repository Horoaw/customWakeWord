---
description: Show the state of all wake-word projects in this repo (or a specific one). Usage `/wake-status` (all) or `/wake-status <slug>`.
---

Inspect the file system to report the state of one or all wake-word projects.

For each project found under `configs/examples/*/` (or the single `<slug>` the user named), report:

| Stage | Marker file | Status |
|---|---|---|
| Bootstrapped | `configs/examples/<slug>/wake_phrases.yaml` | exists / missing |
| Positives | `data/<slug>/synth/positives/manifest.jsonl` | row count (target ~10k) |
| Hard-negs | `data/<slug>/synth/hard_negatives/manifest.jsonl` | row count (target ~2.5k) |
| Bulk | `data/raw/negatives/musan/manifest.jsonl` | exists / missing (shared) |
| Features | `data/<slug>/clean/meta.json` | n_train / n_val / n_test |
| Trained | `outputs/<slug>/model.keras` | exists / missing + mtime |
| TFLite | `models/<slug>-wakeword-v0.tflite` | exists / missing + size |
| Eval | `eval/results/<slug>-v0__latest.json` (or most recent) | FRR / FAR if found |
| Released | `configs/examples/<slug>/release.yaml` | HF URL if set |

For each, recommend the next step. Print as a table with one row per project. Use rich formatting if the user has the `rich` library; otherwise plain markdown.

Do NOT modify any files in status mode — this is read-only.
