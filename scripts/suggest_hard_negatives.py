#!/usr/bin/env python3
"""Use an LLM to suggest non-obvious adversarial phrases for a wake word.

Reads `configs/examples/<name>/wake_phrases.yaml`, asks an LLM (Together AI
by default; OpenAI or local Ollama via flags) for likely false-fire
collisions, and merges the result into
`configs/examples/<name>/hard_negatives.yaml`.

The LLM is asked specifically for:
- Phonetic neighbors that share 2+ phonemes with the wake word.
- Common phrases used in everyday speech that contain the wake word.
- Greetings to other entities that share the same prosody.
- Words/phrases in the same semantic field (food, animals, etc).

Cost: ~$0.001 per wake-word project (Together's small open models).

Usage:
    python scripts/suggest_hard_negatives.py --name sunny
    python scripts/suggest_hard_negatives.py --name sunny --backend openai --model gpt-4o-mini
    python scripts/suggest_hard_negatives.py --name sunny --backend ollama --model llama3.1:8b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


PROMPT_TEMPLATE = """You are an audio-ML data engineer. I'm training a custom wake-word \
detector that should fire on these phrases:

{phrases_block}

The wake word itself is: "{wake_word}"

Generate a JSON object with the following fields, each holding 15-30 short phrases that:

1. "rhyme_collisions": Words or short phrases that rhyme with or share the final \
syllable of "{wake_word}" — these are the most common false-fires.
2. "greeting_no_wake": Greetings to other people/entities that share the same \
prosody and rhythm as "hey {wake_word}" but don't include "{wake_word}".
3. "wake_no_greeting": Casual/everyday phrases that contain the word "{wake_word}" \
WITHOUT a greeting (e.g. "I have a {wake_word}", "I love {wake_word}").
4. "contextual_use": "{wake_word}" used as a regular noun/verb in 5-10 word sentences \
that adults and children would actually say at home.
5. "near_match_words": Words sharing 2+ phonemes with "{wake_word}" that someone \
might say nearby.

Constraints:
- All phrases ≤ 6 words, conversational tone.
- No proper names of celebrities (privacy).
- No profanity or slurs.
- Return ONLY a JSON object, no markdown fences, no commentary."""


def load_phrases(project: Path) -> tuple[str, list[str]]:
    pp = project / "wake_phrases.yaml"
    if not pp.exists():
        raise FileNotFoundError(f"{pp} not found — run init_wake.py first")
    cfg = yaml.safe_load(pp.read_text())
    phrases = [p["text"] for p in cfg["phrases"]]
    wake_word = cfg.get("wake_name") or phrases[0].split()[-1]
    return wake_word, phrases


def call_together(model: str, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return r.choices[0].message.content


def call_openai(model: str, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return r.choices[0].message.content


def call_ollama(model: str, prompt: str) -> str:
    import requests
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "format": "json", "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]


BACKENDS = {
    "together": (call_together, "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    "openai": (call_openai, "gpt-4o-mini"),
    "ollama": (call_ollama, "llama3.1:8b"),
}


def merge_into_negatives(project: Path, suggestions: dict) -> None:
    hn_path = project / "hard_negatives.yaml"
    cfg = yaml.safe_load(hn_path.read_text()) if hn_path.exists() else {"buckets": []}
    by_id = {b["id"]: b for b in cfg.get("buckets", [])}
    for bucket_id, new_phrases in suggestions.items():
        if not isinstance(new_phrases, list):
            continue
        if bucket_id not in by_id:
            cfg.setdefault("buckets", []).append({
                "id": bucket_id,
                "target_count": 200,
                "phrases": [],
            })
            by_id[bucket_id] = cfg["buckets"][-1]
        existing = set(by_id[bucket_id].get("phrases", []))
        for p in new_phrases:
            if isinstance(p, str) and p.strip():
                existing.add(p.strip())
        by_id[bucket_id]["phrases"] = sorted(existing)
    hn_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Project slug under configs/examples/")
    ap.add_argument("--backend", default="together", choices=list(BACKENDS.keys()))
    ap.add_argument("--model", default=None, help="Override backend's default model id")
    ap.add_argument("--out-root", default="configs/examples")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    project = Path(args.out_root) / args.name
    if not project.exists():
        print(f"ERROR: {project} not found. Run init_wake.py first.", file=sys.stderr)
        return 1

    wake_word, phrases = load_phrases(project)
    phrases_block = "\n".join(f"  - {p}" for p in phrases)
    prompt = PROMPT_TEMPLATE.format(
        phrases_block=phrases_block,
        wake_word=wake_word,
    )

    backend_fn, default_model = BACKENDS[args.backend]
    model = args.model or default_model
    print(f"=== suggesting hard negatives for '{args.name}' ===")
    print(f"  backend: {args.backend}")
    print(f"  model:   {model}")
    print(f"  wake:    {wake_word}")
    print()

    if args.dry_run:
        print(prompt)
        return 0

    raw = backend_fn(model, prompt)
    try:
        suggestions = json.loads(raw)
    except json.JSONDecodeError:
        print("ERROR: LLM returned invalid JSON:", file=sys.stderr)
        print(raw[:500], file=sys.stderr)
        return 1

    total_new = sum(len(v) for v in suggestions.values() if isinstance(v, list))
    print(f"  got {total_new} new phrases across {len(suggestions)} buckets")
    for bid, plist in suggestions.items():
        if isinstance(plist, list):
            print(f"    {bid}: {len(plist)} phrases")

    merge_into_negatives(project, suggestions)
    print(f"\nmerged into {project}/hard_negatives.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
