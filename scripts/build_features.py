#!/usr/bin/env python3
"""Convert synthetic + bulk audio into the TFRecord splits microWakeWord trains on.

Generic over wake-word projects: pass `--project <slug>` and this script reads
from `data/<slug>/synth/{positives,hard_negatives}/` and writes to
`data/<slug>/clean/{train,val,test}.tfrecord`.

Pipeline per WAV:
  1. Load, resample to 16 kHz mono.
  2. Random crop to 1.5 s window.
  3. Apply augmentation chain (scripts/augment.py:build_aug_chain).
  4. Compute 40-bin log-mel features at 25 ms hop.
  5. Quantize to INT8.
  6. Write to TFRecord with label + project metadata.

Splits: 80/10/10 train/val/test. The test split is identifiable via
`meta.json["test_ids"]` so you can audit any held-out sample by id.

Augmentation reps: each positive is replicated 5× with different realizations
in train; 1× in val/test (no augmentation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from augment import build_aug_chain, compute_log_mel  # noqa: E402


def load_manifests(positives_dir: Path, hard_negs_dir: Path,
                   bulk_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for mfp in [positives_dir / "manifest.jsonl", hard_negs_dir / "manifest.jsonl"]:
        if not mfp.exists():
            continue
        for line in mfp.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    for bd in bulk_dirs:
        # Bulk corpora aren't strictly manifested per-WAV; use their corpus manifest.
        for mfp in bd.rglob("manifest.jsonl"):
            for line in mfp.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row["label"] = "bulk_negative"
                row["wav_path"] = str(mfp.parent / row["path"])
                rows.append(row)
    return rows


def assign_split(file_id: str, train: float, val: float) -> str:
    """Deterministic hash split — same file_id always lands in same split."""
    h = int(hashlib.md5(file_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < train:
        return "train"
    if h < train + val:
        return "val"
    return "test"


def load_audio(path: str, sample_rate: int):
    import soundfile as sf
    import librosa
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio


def pad_or_crop(audio: np.ndarray, target_samples: int, rng: random.Random) -> np.ndarray:
    if len(audio) > target_samples:
        start = rng.randint(0, len(audio) - target_samples)
        return audio[start:start + target_samples]
    if len(audio) < target_samples:
        pad = target_samples - len(audio)
        left = rng.randint(0, pad)
        return np.pad(audio, (left, pad - left), mode="constant")
    return audio


def make_writer(out_path: Path):
    import tensorflow as tf
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return tf.io.TFRecordWriter(str(out_path))


def write_example(writer, features: np.ndarray, label: int, project: str, file_id: str):
    import tensorflow as tf
    feat_bytes = features.tobytes()
    ex = tf.train.Example(features=tf.train.Features(feature={
        "features": tf.train.Feature(bytes_list=tf.train.BytesList(value=[feat_bytes])),
        "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=list(features.shape))),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        "project": tf.train.Feature(bytes_list=tf.train.BytesList(value=[project.encode()])),
        "file_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_id.encode()])),
    }))
    writer.write(ex.SerializeToString())


LABEL_MAP = {"positive": 1, "hard_negative": 0, "bulk_negative": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Wake-word project slug.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--rir-dir", default="data/raw/negatives/RIRS_NOISES")
    ap.add_argument("--noise-dirs", default="data/raw/negatives/musan/noise,data/raw/negatives/demand")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: data/<project>/clean)")
    ap.add_argument("--bulk-dirs", default=None,
                    help="Comma-separated bulk negative dirs (default: data/raw/negatives/{musan/speech,musan/music,librispeech,commonvoice,audioset})")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-bulk-windows-per-epoch", type=int, default=500000)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data_config).read_text())
    project = args.project
    out_dir = Path(args.out) if args.out else Path(f"data/{project}/clean")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    positives_dir = Path(f"data/{project}/synth/positives")
    hard_negs_dir = Path(f"data/{project}/synth/hard_negatives")
    bulk_default = ",".join([
        "data/raw/negatives/musan/speech",
        "data/raw/negatives/musan/music",
        "data/raw/negatives/librispeech",
        "data/raw/negatives/commonvoice",
        "data/raw/negatives/audioset",
    ])
    bulk_dirs = [Path(d) for d in (args.bulk_dirs or bulk_default).split(",") if Path(d).exists()]
    rir_dir = Path(args.rir_dir)
    rir_paths = list(rir_dir.rglob("*.wav")) if rir_dir.exists() else []
    noise_paths = []
    for d in args.noise_dirs.split(","):
        p = Path(d.strip())
        if p.exists():
            noise_paths.extend(p.rglob("*.wav"))

    print(f"=== build features for project '{project}' ===")
    print(f"  positives:    {positives_dir}")
    print(f"  hard-negs:    {hard_negs_dir}")
    print(f"  bulk-negs:    {[str(d) for d in bulk_dirs]}")
    print(f"  rirs:         {len(rir_paths)} files")
    print(f"  noise:        {len(noise_paths)} files")
    print(f"  out:          {out_dir}")
    print()

    sr = cfg["audio"]["sample_rate"]
    target_samples = int(cfg["audio"]["window_s"] * sr)
    feat_cfg = cfg["audio"]["features"]
    split_cfg = cfg["split"]
    reps_cfg = cfg["reps"]

    rows = load_manifests(positives_dir, hard_negs_dir, bulk_dirs)
    if not rows:
        print(f"ERROR: no manifest rows found for project '{project}'", file=sys.stderr)
        return 1

    # Build label-conditioned aug chains.
    aug_chain_by_label = {
        label: build_aug_chain(cfg["augment"], rir_paths, noise_paths, sr, label)
        for label in ("positive", "hard_negative", "bulk_negative")
    }

    writers = {split: make_writer(out_dir / f"{split}.tfrecord")
               for split in ("train", "val", "test")}
    counts = {split: 0 for split in ("train", "val", "test")}
    test_ids = []

    for i, row in enumerate(rows):
        if i % 500 == 0:
            print(f"  [{i}/{len(rows)}] processed; counts={counts}")
        label_str = row.get("label", "bulk_negative")
        label = LABEL_MAP[label_str]
        wav_path = row.get("wav_path") or row.get("path")
        if not wav_path:
            continue
        # Project-aware paths: synth manifests use relative paths; bulk uses absolute.
        if not Path(wav_path).is_absolute():
            wav_path = str((positives_dir.parent.parent.parent / wav_path).resolve())
        try:
            audio = load_audio(wav_path, sr)
        except Exception as e:
            print(f"  ✗ {wav_path}: {e}", file=sys.stderr)
            continue
        file_id = row.get("file_id") or hashlib.md5(wav_path.encode()).hexdigest()[:12]
        split = assign_split(file_id, split_cfg["train"], split_cfg["val"])
        n_reps = reps_cfg.get(label_str, 1) if split == "train" else reps_cfg["val"]

        if split == "test":
            test_ids.append(file_id)

        for rep in range(n_reps):
            clip = pad_or_crop(audio, target_samples, rng)
            if split == "train":
                clip = aug_chain_by_label[label_str](samples=clip, sample_rate=sr)
            mel = compute_log_mel(
                clip, sr,
                n_mels=feat_cfg["n_mels"],
                win_ms=feat_cfg["win_ms"],
                hop_ms=feat_cfg["hop_ms"],
            )
            # Pad/crop time axis to n_frames
            t = mel.shape[1]
            target_t = feat_cfg["n_frames"]
            if t > target_t:
                start = rng.randint(0, t - target_t)
                mel = mel[:, start:start + target_t]
            elif t < target_t:
                mel = np.pad(mel, ((0, 0), (0, target_t - t)), mode="constant")
            write_example(writers[split], mel.astype(np.float32),
                          label, project, f"{file_id}_r{rep}")
            counts[split] += 1

    for w in writers.values():
        w.close()

    meta = {
        "project": project,
        "n_train": counts["train"],
        "n_val": counts["val"],
        "n_test": counts["test"],
        "test_ids": test_ids,
        "config": cfg,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\ndone. {counts}")
    print(f"meta: {out_dir}/meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
