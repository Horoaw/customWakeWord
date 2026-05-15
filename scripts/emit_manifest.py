#!/usr/bin/env python3
"""Emit the ESPHome v2 manifest JSON for a trained wake-word .tflite.

ESPHome's `micro_wake_word` component reads a JSON sidecar with the
`.tflite` to know the probability cutoff, sliding window, tensor arena
size, and minimum ESPHome version. This script writes that JSON from
the project's eval results (or a hand-supplied threshold).

Reference v2 schema (from `okay_nabu.json` in esphome/micro-wake-word-models):
    {
      "type": "micro",
      "wake_word": "Okay Nabu",
      "author": "Kevin Ahrendt",
      "website": "...",
      "model": "okay_nabu.tflite",
      "trained_languages": ["en"],
      "version": 2,
      "micro": {
        "probability_cutoff": 0.97,
        "feature_step_size": 10,
        "sliding_window_size": 5,
        "tensor_arena_size": 26080,
        "minimum_esphome_version": "2024.7.0"
      }
    }

Usage:
    python scripts/emit_manifest.py --project tofu --threshold 0.85
    python scripts/emit_manifest.py --project tofu --eval-json eval/results/tofu-v0__latest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def pick_threshold_from_eval(eval_json: Path,
                             target_recall: float = 0.95,
                             max_far_per_hour: float = 0.5) -> float:
    """Read the ROC table from an eval JSON and return the best operating point."""
    data = json.loads(eval_json.read_text())
    roc = data.get("summary", {}).get("roc") or data.get("roc", [])
    if not roc:
        # eval/runner.py at v0 emits a single operating point per --threshold.
        # Default to 0.85 if we have no curve to read from.
        print(f"WARN: {eval_json} has no ROC; defaulting threshold=0.85", file=sys.stderr)
        return 0.85
    candidates = [r for r in roc if r.get("recall", 0) >= target_recall
                  and r.get("far_per_hour", 1e9) <= max_far_per_hour]
    if not candidates:
        # Pick the threshold that maximizes recall subject to FA/hr cap.
        under_cap = [r for r in roc if r.get("far_per_hour", 1e9) <= max_far_per_hour]
        candidates = under_cap or roc
    # Among acceptable points, pick the one with the lowest threshold
    # (i.e., highest recall margin).
    best = sorted(candidates, key=lambda r: r.get("threshold", 1.0))[0]
    return float(best["threshold"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="probability_cutoff to embed. Overrides --eval-json.")
    ap.add_argument("--eval-json", default=None,
                    help="Default: eval/results/<project>-v0__latest.json")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--max-far-per-hour", type=float, default=0.5)
    ap.add_argument("--tflite", default=None,
                    help="Default: models/<project>-wakeword-v0.tflite")
    ap.add_argument("--tensor-arena-size", type=int, default=30000,
                    help="Bytes. Bump if ESPHome reports allocation failure.")
    ap.add_argument("--feature-step-size", type=int, default=10)
    ap.add_argument("--sliding-window-size", type=int, default=5)
    ap.add_argument("--minimum-esphome-version", default="2024.7.0")
    ap.add_argument("--display-name", default=None,
                    help="Human-readable wake-word name (default: project.title()).")
    ap.add_argument("--author", default=None,
                    help="Author for the manifest (default: $USER or 'tofuWakeWord').")
    ap.add_argument("--website", default="https://github.com/temm1e-labs/customWakeWord")
    ap.add_argument("--languages", default="en",
                    help="Comma-separated language codes (ISO 639-1).")
    ap.add_argument("--out", default=None,
                    help="Default: configs/examples/<project>/manifest.json")
    args = ap.parse_args()

    project = args.project
    eval_path = Path(args.eval_json) if args.eval_json else Path(
        f"eval/results/{project}-v0__latest.json")
    tflite = Path(args.tflite) if args.tflite else Path(
        f"models/{project}-wakeword-v0.tflite")
    out = Path(args.out) if args.out else Path(
        f"configs/examples/{project}/manifest.json")

    if args.threshold is not None:
        threshold = args.threshold
    elif eval_path.exists():
        threshold = pick_threshold_from_eval(eval_path, args.target_recall,
                                             args.max_far_per_hour)
    else:
        print(f"WARN: --threshold not given and {eval_path} missing; "
              f"defaulting to 0.85", file=sys.stderr)
        threshold = 0.85

    import os
    author = args.author or os.environ.get("USER") or "tofuWakeWord"
    display = args.display_name or project.replace("_", " ").title()

    manifest = {
        "type": "micro",
        "wake_word": display,
        "author": author,
        "website": args.website,
        "model": tflite.name,
        "trained_languages": [s.strip() for s in args.languages.split(",") if s.strip()],
        "version": 2,
        "micro": {
            "probability_cutoff": float(threshold),
            "feature_step_size": args.feature_step_size,
            "sliding_window_size": args.sliding_window_size,
            "tensor_arena_size": args.tensor_arena_size,
            "minimum_esphome_version": args.minimum_esphome_version,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"=== wrote {out} ===")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
