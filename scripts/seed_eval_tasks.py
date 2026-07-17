#!/usr/bin/env python3
"""Create held-out evaluation tasks from manifests and raw negative audio.

Usage:
    python scripts/seed_eval_tasks.py --project tofu \
        --positives 50 --hard-negatives 50 \
        --bulk-audio-dir data/raw/negatives --bulk-stream-minutes 60
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

try:
    from scripts.build_features import assign_split
except ImportError:  # direct `python scripts/seed_eval_tasks.py` execution
    from build_features import assign_split


def load_manifest(directory: Path) -> list[dict]:
    manifest = directory / "manifest.jsonl"
    if not manifest.exists():
        return []
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_task(path: Path, task: dict) -> None:
    path.write_text(
        json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default=None, help="Default: eval/tasks/<project>")
    ap.add_argument("--positives", type=int, default=50)
    ap.add_argument("--hard-negatives", type=int, default=50)
    ap.add_argument("--bulk-stream-minutes", type=int, default=60,
                    help="Select up to this many minutes of raw negative audio.")
    ap.add_argument("--bulk-audio-dir", default="data/raw/negatives",
                    help="Directory recursively containing held-out negative audio.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    project = args.project
    out_dir = Path(args.out) if args.out else Path(f"eval/tasks/{project}")
    for bucket in ("positives", "hard_negatives", "bulk"):
        (out_dir / bucket).mkdir(parents=True, exist_ok=True)
    for pattern, bucket in (("pos_*.json", "positives"),
                            ("neg_*.json", "hard_negatives"),
                            ("bulk_*.json", "bulk")):
        for stale in (out_dir / bucket).glob(pattern):
            stale.unlink()

    rng = random.Random(args.seed)
    pos_rows = load_manifest(Path(f"data/{project}/synth/positives"))
    hn_rows = load_manifest(Path(f"data/{project}/synth/hard_negatives"))

    pos_test = [r for r in pos_rows
                if assign_split(str(r.get("file_id", ""))) == "testing"]
    hn_test = [r for r in hn_rows
               if assign_split(str(r.get("file_id", ""))) == "testing"]
    rng.shuffle(pos_test)
    rng.shuffle(hn_test)
    pos_pick = pos_test[:args.positives]
    hn_pick = hn_test[:args.hard_negatives]

    for index, row in enumerate(pos_pick):
        fid = str(row["file_id"])
        task = {
            "id": f"pos_{index:05d}_{fid}",
            "audio_path": row["wav_path"],
            "label": "positive",
            "phrase": row.get("phrase"),
            "expected": "fire",
            "metadata": {k: row[k] for k in ("voice", "engine", "speed") if k in row},
        }
        write_task(out_dir / "positives" / f"pos_{index:05d}.json", task)

    for index, row in enumerate(hn_pick):
        fid = str(row["file_id"])
        bucket = row.get("bucket_id", "unknown")
        task = {
            "id": f"neg_{index:05d}_{fid}",
            "audio_path": row["wav_path"],
            "label": "hard_negative",
            "bucket_id": bucket,
            "phrase": row.get("phrase"),
            "expected": "no_fire",
            "metadata": {k: row[k] for k in ("voice", "engine", "speed") if k in row},
        }
        write_task(out_dir / "hard_negatives" / f"neg_{index:05d}.json", task)

    bulk_dir = Path(args.bulk_audio_dir)
    bulk_count = 0
    bulk_seconds = 0.0
    if bulk_dir.exists():
        import soundfile as sf
        candidates = [
            path for path in bulk_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"}
        ]
        rng.shuffle(candidates)
        target_seconds = args.bulk_stream_minutes * 60
        for path in candidates:
            try:
                duration = sf.info(str(path)).duration
            except Exception:
                continue
            task_id = f"bulk_{bulk_count:06d}"
            write_task(out_dir / "bulk" / f"{task_id}.json", {
                "id": task_id,
                "audio_path": str(path),
                "label": "bulk_negative",
                "expected": "no_fire",
                "metadata": {"duration_s": duration},
            })
            bulk_count += 1
            bulk_seconds += duration
            if bulk_seconds >= target_seconds:
                break

    print(f"\n=== seeded eval tasks -> {out_dir} ===")
    print(f"  positives:    {len(pos_pick)}")
    print(f"  hard-negs:    {len(hn_pick)}")
    print(f"  bulk audio:   {bulk_count} files / {bulk_seconds / 60:.1f} min")
    if not bulk_count:
        print(f"  note: add held-out audio under {bulk_dir} and rerun for FAR/hour")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
