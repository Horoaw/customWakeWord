#!/usr/bin/env python3
"""Bootstrap a new custom wake-word project from a single command.

    python scripts/init_wake.py --name sunny \\
        --phrases "hey sunny,hi sunny,hello sunny,okay sunny"

Writes:
    configs/examples/sunny/wake_phrases.yaml      ← positive phrases
    configs/examples/sunny/hard_negatives.yaml    ← auto-generated collisions
    configs/examples/sunny/README.md              ← per-project notes

The hard-negatives are generated with a simple phonetic-similarity
heuristic over the primary phrase. For better coverage, run
`scripts/suggest_hard_negatives.py --name sunny` afterward to invoke an
LLM that proposes additional non-obvious collisions.

Optional flags:
    --counts "5000,2500,1500,1000"   per-phrase synthetic sample target
    --weights "1.0,1.0,0.8,0.7"      per-phrase loss weight
    --seed 42                         random seed for the phrase config
    --force                           overwrite existing project directory
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def parse_csv(s: str, cast=str):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def split_wake_word(phrases: list[str]) -> str:
    """Extract the noun that all phrases share (e.g. ["hey tofu", "hi tofu"] → "tofu")."""
    tokens_per_phrase = [p.lower().split() for p in phrases]
    # The wake word is typically the last token shared across all phrases.
    last_tokens = set(t[-1] for t in tokens_per_phrase if t)
    if len(last_tokens) == 1:
        return last_tokens.pop()
    # Fall back: take the longest shared trailing word
    common = tokens_per_phrase[0][-1]
    for toks in tokens_per_phrase[1:]:
        if toks and toks[-1] == common:
            continue
        common = ""
        break
    return common or phrases[0].split()[-1]


# Phonetic-similarity heuristic for generating rhyme collisions.
# Replace each consonant with phonetically-near alternatives, and each
# vowel with near vowels. Crude but covers obvious cases.
_CONSONANT_NEIGHBORS = {
    "p": "b f", "b": "p v", "t": "d", "d": "t",
    "k": "g", "g": "k",
    "f": "v p", "v": "f b",
    "s": "z sh", "z": "s",
    "m": "n", "n": "m",
    "r": "l w", "l": "r",
    "w": "r", "y": "",
    "h": "", "j": "y zh",
}
_VOWEL_NEIGHBORS = {
    "a": "ai au o e",
    "e": "i a",
    "i": "e y",
    "o": "u oa au",
    "u": "o oo",
}


def generate_rhymes(word: str, limit: int = 12) -> list[str]:
    """Return a small list of phonetic neighbors for `word`."""
    out: set[str] = set()
    if not word:
        return []
    chars = list(word.lower())
    # Swap last consonant cluster
    for i in range(len(chars) - 1, -1, -1):
        c = chars[i]
        if c in _CONSONANT_NEIGHBORS:
            for alt in _CONSONANT_NEIGHBORS[c].split():
                if not alt:
                    continue
                new = chars.copy()
                new[i] = alt
                out.add("".join(new))
            break
    # Swap vowels
    for i, c in enumerate(chars):
        if c in _VOWEL_NEIGHBORS:
            for alt in _VOWEL_NEIGHBORS[c].split():
                new = chars.copy()
                new[i] = alt
                out.add("".join(new))
    # Initial-syllable variants
    out.add(f"two-{word[1:]}" if len(word) > 1 else word)
    out.add(f"do-{word[1:]}" if len(word) > 1 else word)
    out.discard(word)
    return sorted(out)[:limit]


def build_phrases_yaml(name: str, phrases: list[str], counts: list[int],
                       weights: list[float], seed: int) -> dict:
    cfg = {
        "wake_name": name,
        "phrases": [],
        "variation": {
            "speeds": [0.85, 0.95, 1.05, 1.15],
            "emotions": {
                "parler": [
                    "speaking calmly, neutral",
                    "speaking excitedly, slightly fast",
                    "speaking softly, almost whispering",
                    "speaking with a friendly child-like tone",
                ],
            },
            "random_seed": seed,
        },
        "engines": [
            {"name": "piper", "weight": 0.5, "voices": "ALL"},
            {"name": "kokoro", "weight": 0.25, "voices": "ALL_EN"},
            {"name": "melotts", "weight": 0.15, "voices": ["EN-US", "EN-BR", "EN-AU", "EN-IN"]},
            {"name": "parler", "weight": 0.10, "voices": ["random_seed_grid_20"]},
        ],
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "format": "wav",
            "trim_silence": True,
            "target_duration_s": 1.5,
            "pad_with_silence_if_short": True,
        },
    }
    for phrase, count, weight in zip(phrases, counts, weights):
        cfg["phrases"].append({
            "text": phrase,
            "count": count,
            "weight": weight,
        })
    return cfg


def build_hard_negatives_yaml(name: str, wake_word: str, primary_phrase: str,
                              seed: int) -> dict:
    rhymes = generate_rhymes(wake_word, limit=10)
    # Build buckets with reasonable defaults; users will edit.
    rhyme_phrases = [f"to-{wake_word[1:]}" if len(wake_word) > 1 else ""]
    rhyme_phrases.extend(rhymes)
    rhyme_phrases = [p for p in rhyme_phrases if p and p != wake_word][:12]

    return {
        "wake_name": name,
        "buckets": [
            {
                "id": "rhyme_collisions",
                "target_count": 600,
                "notes": f"Phonetic neighbors of '{wake_word}'. Edit freely.",
                "phrases": rhyme_phrases,
            },
            {
                "id": "greeting_no_wake",
                "target_count": 300,
                "notes": "Greetings that don't address the wake word.",
                "phrases": [
                    "hey there", "hi there", "hello world",
                    "hey buddy", "hi friend", "hello everyone",
                ],
            },
            {
                "id": "wake_no_greeting",
                "target_count": 500,
                "notes": f"'{wake_word}' alone — must NOT fire.",
                "phrases": [
                    wake_word,
                    f"the {wake_word}",
                    f"that {wake_word}",
                    f"my {wake_word}",
                    f"a {wake_word}",
                ],
            },
            {
                "id": "contextual_use",
                "target_count": 300,
                "notes": f"'{wake_word}' used in normal speech.",
                "phrases": [
                    f"I like {wake_word}",
                    f"do we have any {wake_word}",
                    f"where is the {wake_word}",
                    f"{wake_word} is great",
                ],
            },
            {
                "id": "near_match_words",
                "target_count": 200,
                "notes": "Open bucket — extend with real misfires after v0.",
                "phrases": [],
            },
        ],
        "engines": [
            {"name": "piper", "weight": 0.5, "voices": "ALL"},
            {"name": "kokoro", "weight": 0.25, "voices": "ALL_EN"},
            {"name": "melotts", "weight": 0.15, "voices": ["EN-US", "EN-BR", "EN-AU", "EN-IN"]},
            {"name": "parler", "weight": 0.10, "voices": ["random_seed_grid_20"]},
        ],
        "variation": {
            "speeds": [0.85, 0.95, 1.05, 1.15],
            "random_seed": seed + 4200,
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "format": "wav",
            "trim_silence": True,
            "target_duration_s": 1.5,
            "pad_with_silence_if_short": True,
        },
    }


def write_project_readme(out_dir: Path, name: str, phrases: list[str], wake_word: str) -> None:
    body = f"""# {name} wake-word project

Bootstrapped by `scripts/init_wake.py`.

## Trigger phrases

{chr(10).join(f"- `{p}`" for p in phrases)}

## Wake word: `{wake_word}`

## Next steps

```bash
# 1. (Optional) ask an LLM for harder collisions:
python scripts/suggest_hard_negatives.py --name {name}

# 2. Synthesize positives:
python scripts/synth_positives.py \\
    --phrases configs/examples/{name}/wake_phrases.yaml \\
    --out data/{name}/synth/positives \\
    --count 10000

# 3. Synthesize hard-negatives:
python scripts/synth_hard_negatives.py \\
    --phrases configs/examples/{name}/hard_negatives.yaml \\
    --out data/{name}/synth/hard_negatives \\
    --count 2500

# 4. Build features + train (see PIPELINE.md):
python scripts/build_features.py --project {name} --out data/{name}/clean
python scripts/runpod_train.py --project {name}

# 5. Eval:
python -m eval.runner --model models/{name}-wakeword-v0.tflite

# 6. Release:
python scripts/upload_to_hf.py --project {name}
```

Or with Claude Code slash commands:

```
/wake-synth {name}
/wake-train {name}
/wake-eval {name}
/wake-release {name}
```
"""
    out_dir.joinpath("README.md").write_text(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    help="Project slug, e.g. 'sunny' or 'jarvis'.")
    ap.add_argument("--phrases", required=True,
                    help='Comma-separated trigger phrases, e.g. "hey sunny,hi sunny".')
    ap.add_argument("--counts", default=None,
                    help='Comma-separated per-phrase sample counts (defaults to a decaying ramp).')
    ap.add_argument("--weights", default=None,
                    help='Comma-separated per-phrase weights.')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing project directory.")
    ap.add_argument("--out-root", default="configs/examples",
                    help="Where to write the project directory.")
    args = ap.parse_args()

    name = slugify(args.name)
    if not name:
        print("ERROR: --name must contain at least one alnum char", file=sys.stderr)
        return 1

    phrases = [p.strip() for p in args.phrases.split(",") if p.strip()]
    if not phrases:
        print("ERROR: --phrases produced no entries", file=sys.stderr)
        return 1

    # Default per-phrase counts: decay 5000 → 2500 → 1500 → 1000 → 1000 → …
    defaults = [5000, 2500, 1500, 1000]
    if args.counts:
        counts = parse_csv(args.counts, int)
    else:
        counts = [defaults[min(i, len(defaults) - 1)] for i in range(len(phrases))]

    if args.weights:
        weights = parse_csv(args.weights, float)
    else:
        weights = [1.0, 1.0, 0.8, 0.7] + [0.6] * max(0, len(phrases) - 4)
        weights = weights[: len(phrases)]

    if len(counts) != len(phrases) or len(weights) != len(phrases):
        print(f"ERROR: phrases ({len(phrases)}), counts ({len(counts)}), weights "
              f"({len(weights)}) must all be the same length", file=sys.stderr)
        return 1

    out_dir = Path(args.out_root) / name
    if out_dir.exists() and not args.force:
        print(f"ERROR: {out_dir} already exists. Pass --force to overwrite.",
              file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    wake_word = split_wake_word(phrases)

    phrases_yaml = build_phrases_yaml(name, phrases, counts, weights, args.seed)
    hard_yaml = build_hard_negatives_yaml(name, wake_word, phrases[0], args.seed)

    (out_dir / "wake_phrases.yaml").write_text(yaml.safe_dump(phrases_yaml, sort_keys=False))
    (out_dir / "hard_negatives.yaml").write_text(yaml.safe_dump(hard_yaml, sort_keys=False))
    write_project_readme(out_dir, name, phrases, wake_word)

    print(f"=== Initialized wake-word project '{name}' ===")
    print(f"  output:        {out_dir}/")
    print(f"  wake word:     {wake_word}")
    print(f"  phrases:       {len(phrases)} → {sum(counts)} synthetic samples total")
    print(f"  rhyme rhymes:  {len(hard_yaml['buckets'][0]['phrases'])} auto-generated")
    print()
    print(f"  next: python scripts/suggest_hard_negatives.py --name {name}")
    print(f"        (optional — uses an LLM to add non-obvious collisions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
