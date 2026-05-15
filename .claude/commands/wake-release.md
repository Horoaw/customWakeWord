---
description: Publish a trained wake-word model to HuggingFace Hub + emit ESPHome YAML. Usage `/wake-release <slug> --hf-repo-id <user>/<slug>-wakeword-v0`.
---

The user just invoked `/wake-release`. First positional arg = project slug.

Read `configs/examples/<slug>/release.yaml` if it exists — that may already have the HF repo id from a prior release.

If no repo id is known (neither from args nor release.yaml), ASK the user for one. Suggest the canonical form `<user>/<slug>-wakeword-v0`.

Dispatch to the **release-manager** subagent. It will:
1. Render ESPHome YAML.
2. Upload model + model card + eval JSON to HF Hub.
3. Persist `configs/examples/<slug>/release.yaml`.
4. Optionally open a GitHub Release (ASK before running `gh release create` — visible to others).
5. Append a row to `BUDGET_LOG.md`.

After the subagent returns, surface the HF URL and the ESPHome flash command.
