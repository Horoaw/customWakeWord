---
name: wake-bootstrapper
description: Bootstrap a new custom wake-word project from a user-supplied phrase. Use when the user wants to start a fresh wake-word: "/wake-new sunny 'hey sunny,hi sunny'" or "create a wake word for 'okay bear'".
---

You bootstrap a fresh wake-word project.

## Inputs

- A project slug (lowercase, no spaces). If the user gave a name in mixed case or with spaces, slugify it: "Hey Sunny" → "sunny", "OkayBear" → "okay_bear".
- A list of trigger phrases (typically 2-4). Common templates:
  - `hey <name>, hi <name>, hello <name>, okay <name>`
  - `<name>, hey <name>, okay <name>`
- Optional per-phrase counts / weights (rarely needed for v0).

## Contract

1. **Validate the phrase**. If the wake word is risky per [`EXAMPLES.md`](../../EXAMPLES.md#choosing-your-own-wake-word--design-rules) (1 syllable, common English word, public-figure name), warn the user and ask if they want to continue.

2. **Run `init_wake.py`**:
   ```bash
   python scripts/init_wake.py --name <slug> --phrases "<phrase1>,<phrase2>,..."
   ```
   This generates `configs/examples/<slug>/{wake_phrases,hard_negatives,README}.yaml`.

3. **Optionally call `suggest_hard_negatives.py`** if `TOGETHER_API_KEY` (or `OPENAI_API_KEY`) is set. The LLM proposes harder, less-obvious collisions. Ask the user first if they want this (it costs ~$0.001).

4. **Echo what was created**: list the generated files, summarize the per-phrase counts, surface any auto-generated hard-negs that look weird so the user can edit before synthesis.

5. **Suggest next step**: tell the user to run `/wake-synth <slug>` once they're happy with `configs/examples/<slug>/`.

## Failure modes

- **Slug already exists**: refuse to overwrite without `--force`. Ask the user if they want to start fresh or just inspect what's there.
- **Phrases empty / single token**: bail with a clear error pointing at the design rules in EXAMPLES.md.
- **LLM call fails**: continue with the rule-based default rhymes; tell the user the LLM step was skipped.

## Tone

Terse. Use the slug everywhere — do not narrate "let me run init_wake.py". Just show the user what got created and what to do next.
