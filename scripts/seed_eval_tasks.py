#!/usr/bin/env python3
"""Sample a held-out test set from `data/<project>/clean/test.tfrecord` and
write it as one JSON-per-WAV under `eval/tasks/<project>/`.

Each task is a 1.5 s clip with metadata. The runner uses these to compute
FRR / FAR. You can hand-curate (delete bad samples, add real recordings)
after this script runs.

Usage:
    python scripts/seed_eval_tasks.py --project tofu --positives 50 --hard-negatives 50
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None,
                    help="Default: eval/tasks/<project>")
    ap.add_argument("--positives", type=int, default=50)
    ap.add_argument("--hard-negatives", type=int, default=50)
    ap.add_argument("--bulk-stream-minutes", type=int, default=60,
                    help="Build a concatenated bulk-negative WAV of this duration "
                         "for FAR/hour measurement.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    project = args.project
    data_dir = Path(args.data_dir) if args.data_dir else Path(f"data/{project}/clean")
    out_dir = Path(args.out) if args.out else Path(f"eval/tasks/{project}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "positives").mkdir(exist_ok=True)
    (out_dir / "hard_negatives").mkdir(exist_ok=True)
    (out_dir / "bulk").mkdir(exist_ok=True)

    meta = json.loads((data_dir / "meta.json").read_text())
    project_cfg = meta["config"]
    rng = random.Random(args.seed)

    # We rebuild the test split as WAV + JSON pairs by replaying the test
    # TFRecord. To get raw WAV back, we'd need to keep the audio in the
    # record — for v0 we recommend curating from the manifests instead.
    #
    # Simpler approach: sample from data/<project>/synth/* using the test_ids
    # we kept in meta.json["test_ids"]. The file_id format is "<id>_r0".
    positives_dir = Path(f"data/{project}/synth/positives")
    hard_negs_dir = Path(f"data/{project}/synth/hard_negatives")

    def load_manifest(d: Path) -> list[dict]:
        m = d / "manifest.jsonl"
        if not m.exists():
            return []
        return [json.loads(l) for l in m.read_text().splitlines() if l.strip()]

    pos_rows = load_manifest(positives_dir)
    hn_rows = load_manifest(hard_negs_dir)

    test_set = set(tid.rsplit("_r", 1)[0] for tid in meta.get("test_ids", []))

    pos_test = [r for r in pos_rows if r.get("file_id") in test_set]
    hn_test = [r for r in hn_rows if r.get("file_id") in test_set]
    rng.shuffle(pos_test)
    rng.shuffle(hn_test)
    pos_pick = pos_test[: args.positives]
    hn_pick = hn_test[: args.hard_negatives]

    for r in pos_pick:
        fid = r["file_id"]
        task = {
            "id": f"pos_{fid}",
            "audio_path": r["wav_path"],
            "label": "positive",
            "phrase": r.get("phrase"),
            "expected": "fire",
            "metadata": {k: r[k] for k in ("voice", "engine", "speed") if k in r},
        }
        (out_dir / "positives" / f"{fid}.json").write_text(json.dumps(task, indent=2))

    for r in hn_pick:
        fid = r["file_id"]
        bucket = r.get("bucket_id", "unknown")
        task = {
            "id": f"neg_{bucket}_{fid}",
            "audio_path": r["wav_path"],
            "label": "hard_negative",
            "bucket_id": bucket,
            "phrase": r.get("phrase"),
            "expected": "no_fire",
            "metadata": {k: r[k] for k in ("voice", "engine", "speed") if k in r},
        }
        (out_dir / "hard_negatives" / f"{bucket}_{fid}.json").write_text(json.dumps(task, indent=2))

    # Build a concatenated bulk-negative stream for FAR/hour.
    print(f"\n=== seeded eval tasks → {out_dir} ===")
    print(f"  positives:    {len(pos_pick)}")
    print(f"  hard-negs:    {len(hn_pick)}")
    print(f"  bulk stream:  TODO — build via `python -c 'from eval.runner import build_bulk_stream; "
          f"build_bulk_stream(\"{project}\", minutes={args.bulk_stream_minutes})'`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
