#!/usr/bin/env python3
"""Synthesize TTS hard-negatives for Tofu.

Mirrors `synth_positives.py` but reads `configs/hard_negatives.yaml`, fans
out across buckets of collision phrases, and writes to
`data/synth/hard_negatives/`. Each manifest row has a `bucket_id` field so
eval can compute per-bucket FAR.

Usage:
    python scripts/synth_hard_negatives.py \\
        --phrases configs/hard_negatives.yaml \\
        --out data/synth/hard_negatives \\
        --count 2500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import yaml

from synth_positives import (  # noqa: E402  — sibling module
    ENGINES_REGISTRY,
    load_engines,
    file_id,
    load_manifest,
    append_manifest,
)


def plan_jobs(cfg: dict, engines: dict, total_count: int) -> list[dict]:
    rng = random.Random(cfg["variation"]["random_seed"])
    buckets = cfg["buckets"]

    raw_total = sum(b["target_count"] for b in buckets)
    scale = total_count / raw_total if raw_total else 1.0

    eng_weights = [(e["name"], float(e["weight"])) for e in cfg["engines"] if e["name"] in engines]
    if not eng_weights:
        return []
    w_total = sum(w for _, w in eng_weights)
    eng_cdf, cum = [], 0.0
    for name, w in eng_weights:
        cum += w / w_total
        eng_cdf.append((name, cum))

    speeds = cfg["variation"]["speeds"]

    jobs = []
    for bucket in buckets:
        n = max(1, int(round(bucket["target_count"] * scale)))
        phrases = bucket["phrases"]
        for _ in range(n):
            r = rng.random()
            engine_name = next(name for name, c in eng_cdf if r <= c)
            engine = engines[engine_name]
            voice_hint = next((e.get("voices") for e in cfg["engines"] if e["name"] == engine_name), "ALL")
            voices = engine.list_voices(voice_hint)
            if not voices:
                continue
            jobs.append({
                "engine": engine_name,
                "voice": rng.choice(voices),
                "phrase": rng.choice(phrases),
                "speed": rng.choice(speeds),
                "emotion": None,
                "seed": rng.randint(0, 2**31 - 1),
                "bucket_id": bucket["id"],
            })
    rng.shuffle(jobs)
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", default="configs/hard_negatives.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=2500)
    ap.add_argument("--engines", default="piper,kokoro,melotts,parler")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.phrases).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Tofu hard-negative synthesis ===")
    print(f"  out:    {out_dir}")
    print(f"  count:  {args.count}")
    print()

    engines = load_engines(args.engines.split(","))
    if not engines:
        print("ERROR: no engines available", file=sys.stderr)
        return 1

    jobs = plan_jobs(cfg, engines, args.count)
    print(f"\nplanned {len(jobs)} jobs across {len(cfg['buckets'])} buckets")
    if args.dry_run:
        return 0

    seen = load_manifest(out_dir)
    sr = cfg["audio"]["sample_rate"]
    t0 = time.time()
    done = 0
    for job in jobs:
        fid = file_id(job["voice"], job["phrase"], job["speed"], job["emotion"], job["seed"])
        if fid in seen:
            continue
        wav_path = out_dir / f"{job['bucket_id']}__{fid}.wav"
        try:
            ok = engines[job["engine"]].synth(
                job["phrase"], job["voice"], job["speed"],
                job["emotion"], wav_path, sr,
            )
        except Exception as e:
            print(f"  ✗ {fid}: {e}", file=sys.stderr)
            ok = False
        if not ok or not wav_path.exists():
            continue
        append_manifest(out_dir, {
            "file_id": fid,
            "wav_path": str(wav_path.relative_to(out_dir.parent.parent)),
            "engine": job["engine"],
            "voice": job["voice"],
            "phrase": job["phrase"],
            "speed": job["speed"],
            "emotion": job["emotion"],
            "seed": job["seed"],
            "label": "hard_negative",
            "bucket_id": job["bucket_id"],
        })
        done += 1
        if done % 50 == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            print(f"  [{done}/{len(jobs) - len(seen)}] {rate:.1f} samples/s")

    print(f"\ndone. {done} new samples; manifest at {out_dir}/manifest.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
