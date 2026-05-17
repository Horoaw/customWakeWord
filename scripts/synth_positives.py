#!/usr/bin/env python3
"""Synthesize positives for a wake-word project using piper-sample-generator.

This is the canonical recipe path used by OHF-Voice/micro-wake-word's
basic_training_notebook.ipynb. It shells out to piper-sample-generator
(a separate repo at https://github.com/rhasspy/piper-sample-generator)
which uses the `en_US-libritts_r-medium.pt` VITS generator model with
**904 distinct speakers** + tempo/noise jitter for per-sample variation.

Reads `configs/examples/<project>/wake_phrases.yaml`. For each phrase,
runs `piper-sample-generator/generate_samples.py "<phrase>" --max-samples N`
into `data/<project>/synth/positives/<phrase_slug>/`. Writes a unified
`manifest.jsonl` at the end so downstream `build_features.py` knows which
WAVs belong to which phrase.

Resumable: if a phrase's directory already has the requested sample count,
that phrase is skipped.

Usage:
    python scripts/synth_positives.py --project tofu
    python scripts/synth_positives.py --project tofu --count 20000  # override total
    python scripts/synth_positives.py --project tofu --psg-dir ./piper-sample-generator
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_URL_LINUX = "https://github.com/rhasspy/piper-sample-generator"
REPO_URL_MPS = "https://github.com/kahrendt/piper-sample-generator"
MPS_BRANCH = "mps-support"
GEN_MODEL_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/"
    "v2.0.0/en_US-libritts_r-medium.pt"
)
GEN_MODEL_NAME = "en_US-libritts_r-medium.pt"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def ensure_psg(psg_dir: Path) -> Path:
    """Clone piper-sample-generator + download the generator model if needed.

    Auto-picks the MPS-support fork on Darwin.
    """
    if not psg_dir.exists():
        import platform
        if platform.system() == "Darwin":
            print(f"  cloning {REPO_URL_MPS} (branch {MPS_BRANCH}) → {psg_dir}", flush=True)
            subprocess.check_call([
                "git", "clone", "-b", MPS_BRANCH, "--depth", "1",
                REPO_URL_MPS, str(psg_dir),
            ])
        else:
            print(f"  cloning {REPO_URL_LINUX} → {psg_dir}", flush=True)
            subprocess.check_call([
                "git", "clone", "--depth", "1", REPO_URL_LINUX, str(psg_dir),
            ])

    model_path = psg_dir / "models" / GEN_MODEL_NAME
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading generator model → {model_path}", flush=True)
        subprocess.check_call([
            "wget", "-q", "--show-progress", "-O", str(model_path), GEN_MODEL_URL,
        ])
    return psg_dir


def generate_for_phrase(psg_dir: Path, phrase: str, out_dir: Path,
                        count: int, batch_size: int = 100,
                        max_speakers: int | None = 904,
                        *, max_retries: int = 3,
                        min_ratio: float = 0.95,
                        retry_sleep_s: float = 10.0,
                        ) -> tuple[int, bool]:
    """Run piper-sample-generator for one phrase.

    Returns (actual_wav_count, ok). ok=False if after `max_retries` attempts
    the on-disk count is still below `min_ratio * count` — typically a
    transient RunPod disk Errno 5 or piper-internal failure that didn't
    recover. The caller is expected to treat ok=False as fatal (don't write
    the manifest; exit non-zero so the wrapper script's `set -e` aborts).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.wav"))
    if len(existing) >= count:
        print(f"  ✓ {phrase}: already have {len(existing)} ≥ {count}, skipping", flush=True)
        return len(existing), True

    needed = count - len(existing)
    print(f"  → {phrase}: generating {needed} more ({len(existing)} already on disk)", flush=True)

    cmd = [
        sys.executable, str(psg_dir / "generate_samples.py"),
        phrase,
        "--max-samples", str(needed),
        "--batch-size", str(batch_size),
        "--output-dir", str(out_dir),
    ]
    if max_speakers is not None:
        cmd.extend(["--max-speakers", str(max_speakers)])

    for attempt in range(1, max_retries + 1):
        try:
            subprocess.check_call(cmd)
            break
        except subprocess.CalledProcessError as e:
            actual = len(list(out_dir.glob("*.wav")))
            if attempt >= max_retries:
                print(
                    f"  ✗ {phrase}: failed after {max_retries} attempts ({e}); "
                    f"{actual}/{count} on disk",
                    file=sys.stderr, flush=True,
                )
                break
            print(
                f"  ⚠ {phrase}: attempt {attempt}/{max_retries} failed ({e}); "
                f"{actual}/{count} on disk, retrying in {retry_sleep_s:.0f}s",
                file=sys.stderr, flush=True,
            )
            time.sleep(retry_sleep_s)
            new_needed = count - actual
            if new_needed <= 0:
                break
            for i, tok in enumerate(cmd):
                if tok == "--max-samples":
                    cmd[i + 1] = str(new_needed)
                    break

    actual = len(list(out_dir.glob("*.wav")))
    ok = actual >= int(count * min_ratio)
    return actual, ok


def write_manifest(positives_root: Path, by_phrase: dict[str, Path]) -> None:
    """Write a single manifest.jsonl for downstream feature extraction."""
    mfp = positives_root / "manifest.jsonl"
    n = 0
    with mfp.open("w") as f:
        for phrase, phrase_dir in by_phrase.items():
            for wav in sorted(phrase_dir.glob("*.wav")):
                f.write(json.dumps({
                    "file_id": wav.stem,
                    "wav_path": str(wav),
                    "phrase": phrase,
                    "label": "positive",
                    "engine": "piper_sample_generator",
                    "voice_model": GEN_MODEL_NAME,
                }) + "\n")
                n += 1
    print(f"  wrote {mfp} ({n} rows)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True,
                    help="Project slug under configs/examples/")
    ap.add_argument("--config", default=None,
                    help="Path to wake_phrases.yaml (default: configs/examples/<project>/wake_phrases.yaml)")
    ap.add_argument("--out", default=None,
                    help="Output root (default: data/<project>/synth/positives)")
    ap.add_argument("--count", type=int, default=None,
                    help="Override total positive count (default: sum of per-phrase counts in YAML)")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--max-speakers", type=int, default=904,
                    help="Cap on Piper voice speaker count (default 904 = LibriTTS-R full).")
    ap.add_argument("--psg-dir", default="piper-sample-generator",
                    help="Where to clone piper-sample-generator (default ./piper-sample-generator)")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else Path(
        f"configs/examples/{args.project}/wake_phrases.yaml"
    )
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found. Run scripts/init_wake.py first.", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(cfg_path.read_text())
    out_root = Path(args.out) if args.out else Path(f"data/{args.project}/synth/positives")
    out_root.mkdir(parents=True, exist_ok=True)

    phrases = cfg["phrases"]
    raw_total = sum(p["count"] for p in phrases)
    if args.count is not None:
        scale = args.count / raw_total
    else:
        scale = 1.0
    per_phrase_count = {p["text"]: max(1, int(round(p["count"] * scale))) for p in phrases}

    print(f"=== piper-sample-generator: positives for '{args.project}' ===", flush=True)
    print(f"  config:   {cfg_path}")
    print(f"  out:      {out_root}")
    print(f"  total:    {sum(per_phrase_count.values())} ({len(phrases)} phrases)")
    print(f"  speakers: {args.max_speakers}")
    print(flush=True)

    psg_dir = ensure_psg(Path(args.psg_dir))

    by_phrase: dict[str, Path] = {}
    shortfalls: list[tuple[str, int, int]] = []
    for phrase, count in per_phrase_count.items():
        phrase_dir = out_root / slug(phrase)
        by_phrase[phrase] = phrase_dir
        actual, ok = generate_for_phrase(psg_dir, phrase, phrase_dir, count,
                                         batch_size=args.batch_size,
                                         max_speakers=args.max_speakers)
        if not ok:
            shortfalls.append((phrase, actual, count))

    if shortfalls:
        print("\n=== FAILED: some phrases short of target ===", file=sys.stderr)
        for phrase, actual, count in shortfalls:
            pct = 100 * actual / count if count else 0
            print(f"  ✗ {phrase}: {actual}/{count} ({pct:.0f}%)",
                  file=sys.stderr)
        print("Not writing manifest. Wipe data/<project>/synth/positives "
              "and rerun; the pod disk likely had a transient I/O error "
              "(see RUNPOD_RECIPE.md).", file=sys.stderr)
        return 1

    write_manifest(out_root, by_phrase)

    total = sum(len(list(d.glob("*.wav"))) for d in by_phrase.values())
    print(f"\n=== done: {total} positive WAVs across {len(by_phrase)} phrases ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
