---
description: Bootstrap a new custom wake-word project. Usage `/wake-new <slug> "<phrase1>,<phrase2>,..."`.
---

The user just invoked `/wake-new`. Parse the arguments:
- First positional arg: the project slug (lowercase, no spaces).
- Remaining args (joined): a comma-separated list of trigger phrases.

If the user supplied no args, ask them for both. Show a one-line example.

Dispatch to the **wake-bootstrapper** subagent with the parsed slug + phrases. The subagent will run `scripts/init_wake.py` and optionally `scripts/suggest_hard_negatives.py`.

After it returns, summarize: which files were created and what to do next (`/wake-synth <slug>`).
