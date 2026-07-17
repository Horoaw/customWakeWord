#!/usr/bin/env python3
"""Bootstrap a new custom wake-word project from a single command.

    python scripts/init_wake.py --name sunny \\
        --phrases "hey sunny,hi sunny,hello sunny,okay sunny"

Writes:
    configs/examples/sunny/wake_phrases.yaml      ← positive phrases
    configs/examples/sunny/hard_negatives.yaml    ← auto-generated collisions
    configs/examples/sunny/training_parameters.yaml ← training config
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
import unicodedata
from pathlib import Path

import yaml


def slugify(s: str) -> str:
    """Return a filesystem-safe, Unicode-preserving project id."""
    normalized = unicodedata.normalize("NFKC", s).lower().strip()
    return re.sub(r"[^\w-]+", "_", normalized, flags=re.UNICODE).strip("_")


def detect_language(phrases: list[str]) -> str:
    """Infer the primary language for sensible project defaults."""
    text = "".join(phrases)
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "zh"
    return "en"


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


def build_engines(language: str) -> list[dict]:
    if language == "zh":
        return [{
            "name": "piper",
            "weight": 1.0,
            "voices": [
                {
                    "id": "zh_CN-xiao_ya-medium",
                    "model_url": (
                        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                        "zh/zh_CN/xiao_ya/medium/zh_CN-xiao_ya-medium.onnx"
                    ),
                    "config_url": (
                        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                        "zh/zh_CN/xiao_ya/medium/zh_CN-xiao_ya-medium.onnx.json"
                    ),
                },
                {
                    "id": "zh_CN-huayan-x_low",
                    "model_url": (
                        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                        "zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx"
                    ),
                    "config_url": (
                        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                        "zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx.json"
                    ),
                },
            ],
        }]
    return [{"name": "piper", "weight": 1.0, "voices": "ALL"}]


def build_phrases_yaml(name: str, phrases: list[str], counts: list[int],
                       weights: list[float], seed: int, language: str) -> dict:
    wake_word = split_wake_word(phrases)

    cfg = {
        "project_name": name,
        "wake_name": wake_word,
        "language": language,
        "phrases": [],
        "variation": {
            "speeds": [0.85, 0.95, 1.05, 1.15],
            "random_seed": seed,
        },
        "engines": build_engines(language),
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


def build_hard_negatives_yaml(name: str, wake_word: str, seed: int, language: str,
                              positive_phrases: list[str]) -> dict:
    rhymes = generate_rhymes(wake_word, limit=10) if language == "en" else []
    # Build buckets with reasonable defaults; language-specific phonetic
    # neighbors still need human/LLM review.
    rhyme_phrases = ([f"to-{wake_word[1:]}" if len(wake_word) > 1 else ""]
                      if language == "en" else [])
    rhyme_phrases.extend(rhymes)
    rhyme_phrases = [p for p in rhyme_phrases if p and p != wake_word][:12]

    if language == "zh":
        greeting_phrases = ["你好", "嗨", "您好", "早上好", "晚上好"]
        standalone_context = [
            f"我刚才说了{wake_word}",
            f"你听见{wake_word}了吗",
            f"电视里提到了{wake_word}",
        ]
        contextual = [
            f"我喜欢{wake_word}",
            f"我们来聊聊{wake_word}",
            f"那个{wake_word}在哪里",
            f"{wake_word}真的很好",
        ]
    else:
        greeting_phrases = [
            "hey there", "hi there", "hello world",
            "hey buddy", "hi friend", "hello everyone",
        ]
        standalone_context = [
            f"the {wake_word}", f"that {wake_word}",
            f"my {wake_word}", f"a {wake_word}",
        ]
        contextual = [
            f"I like {wake_word}", f"do we have any {wake_word}",
            f"where is the {wake_word}", f"{wake_word} is great",
        ]

    positive_set = {p.strip().casefold() for p in positive_phrases}

    def without_positives(items: list[str]) -> list[str]:
        return [p for p in items if p.strip().casefold() not in positive_set]

    return {
        "project_name": name,
        "wake_name": wake_word,
        "language": language,
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
                "phrases": greeting_phrases,
            },
            {
                "id": "wake_no_greeting",
                "target_count": 500,
                "notes": f"Context containing '{wake_word}' that should not fire.",
                "phrases": without_positives(standalone_context),
            },
            {
                "id": "contextual_use",
                "target_count": 300,
                "notes": f"'{wake_word}' used in normal speech.",
                "phrases": without_positives(contextual),
            },
            {
                "id": "near_match_words",
                "target_count": 200,
                "notes": "Open bucket — extend with real misfires after v0.",
                "phrases": [],
            },
        ],
        "engines": build_engines(language),
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


def build_training_parameters(name: str) -> dict:
    """Build an upstream-compatible baseline training configuration."""
    return {
        "train_dir": f"trained_models/{name}",
        "window_step_ms": 10,
        "clip_duration_ms": 1500,
        "features": [
            {
                "features_dir": f"data/{name}/features",
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": f"data/{name}/hard_negatives_features",
                "sampling_weight": 3.0,
                "penalty_weight": 5.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "data/negative_datasets/speech",
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "data/negative_datasets/dinner_party",
                "sampling_weight": 15.0,
                "penalty_weight": 3.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "data/negative_datasets/dinner_party_eval",
                "sampling_weight": 0.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "split",
                "type": "mmap",
            },
        ],
        "training_steps": [10000],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": 128,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "target_minimization": 0.5,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }


def write_project_readme(out_dir: Path, name: str, phrases: list[str], wake_word: str) -> None:
    body = f"""# {name} wake-word project

Bootstrapped by `scripts/init_wake.py`.

## Trigger phrases

{chr(10).join(f"- `{p}`" for p in phrases)}

## Wake word: `{wake_word}`

## Next steps

```bash
# 1. Optionally ask an LLM for harder collisions.
python scripts/suggest_hard_negatives.py --name {name}

# 2. Synthesize project audio.
python scripts/synth_positives.py --project {name}
python scripts/synth_hard_negatives.py --project {name}

# 3. Download shared negatives and build device-compatible features.
python scripts/download_hf_negatives.py --out data/negative_datasets
python scripts/build_features.py --project {name} --download-aug-corpora

# 4. Train on an existing GPU host (or use runpod_train.py).
python scripts/train_microwakeword.py --project {name}

# 5. Seed held-out tasks and evaluate.
python scripts/seed_eval_tasks.py --project {name} --bulk-audio-dir data/raw/negatives
python -m eval.runner --project {name} --model models/{name}-wakeword-v0.tflite

# 6. Emit an ESPHome manifest from a measured operating point.
python scripts/emit_manifest.py --project {name} --eval-json eval/results/{name}-v0__latest.json
```
"""
    out_dir.joinpath("README.md").write_text(body, encoding="utf-8")


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
    ap.add_argument("--language", choices=["en", "zh"], default=None,
                    help="Primary sample language (auto-detected when omitted).")
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
    language = args.language or detect_language(phrases)

    phrases_yaml = build_phrases_yaml(name, phrases, counts, weights, args.seed, language)
    hard_yaml = build_hard_negatives_yaml(
        name, wake_word, args.seed, language, phrases
    )
    training_yaml = build_training_parameters(name)

    (out_dir / "wake_phrases.yaml").write_text(
        yaml.safe_dump(phrases_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (out_dir / "hard_negatives.yaml").write_text(
        yaml.safe_dump(hard_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (out_dir / "training_parameters.yaml").write_text(
        yaml.safe_dump(training_yaml, sort_keys=False), encoding="utf-8"
    )
    write_project_readme(out_dir, name, phrases, wake_word)

    print(f"=== Initialized wake-word project '{name}' ===")
    print(f"  output:        {out_dir}/")
    print(f"  wake word:     {wake_word}")
    print(f"  language:      {language}")
    print(f"  phrases:       {len(phrases)} → {sum(counts)} synthetic samples total")
    print(f"  rhyme rhymes:  {len(hard_yaml['buckets'][0]['phrases'])} auto-generated")
    print()
    print(f"  next: python scripts/suggest_hard_negatives.py --name {name}")
    print("        (optional — uses an LLM to add non-obvious collisions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
