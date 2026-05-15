#!/usr/bin/env python3
"""Download bulk negative-audio corpora.

Pulls:
- MUSAN                 (music + speech + noise)
- DEMAND                (real-room noise)
- LibriSpeech           (clean read speech)
- Common Voice English  (crowdsourced, accent-diverse)
- AudioSet subset       (TV/podcast/household)
- OpenSLR-28 RIRs       (room impulse responses for augmentation)

Each corpus lands under `data/raw/negatives/<corpus>/` along with a
`manifest.jsonl` listing every WAV / FLAC + duration.

Designed to be re-runnable: if a tarball is already extracted, the script
skips it. The Common Voice and AudioSet portions are gated behind explicit
flags because they require either a HF token (CV) or the
`audioset-downloader` extra (AudioSet).

Usage:
    python scripts/collect_negatives.py --out data/raw/negatives \\
        --corpora musan,demand,librispeech,rirs,commonvoice,audioset_subset \\
        --hours 300
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm


CORPUS_URLS = {
    "musan": "https://www.openslr.org/resources/17/musan.tar.gz",
    "rirs": "https://www.openslr.org/resources/28/rirs_noises.zip",
    "librispeech_test_clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "librispeech_train_clean_100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
}


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ already downloaded: {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} → {dest}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pb:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pb.update(len(chunk))
    return True


def _extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} → {out_dir}")
    if archive.suffix in (".gz", ".tgz") or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir)
    elif archive.suffix == ".zip":
        subprocess.check_call(["unzip", "-q", "-n", str(archive), "-d", str(out_dir)])


def _walk_wavs(root: Path, exts={".wav", ".flac", ".mp3", ".ogg", ".opus"}) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]


def _write_manifest(corpus: str, files: list[Path], out_root: Path) -> None:
    import soundfile as sf
    mfp = out_root / corpus / "manifest.jsonl"
    mfp.parent.mkdir(parents=True, exist_ok=True)
    total_s = 0.0
    with mfp.open("w") as f:
        for p in tqdm(files, desc=f"{corpus} manifest"):
            try:
                info = sf.info(str(p))
                dur = info.duration
            except Exception:
                continue
            total_s += dur
            f.write(json.dumps({
                "path": str(p.relative_to(out_root)),
                "duration_s": dur,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "corpus": corpus,
            }) + "\n")
    print(f"  {corpus}: {len(files)} files, {total_s/3600:.1f} h")


def fetch_musan(out_root: Path) -> None:
    arc = out_root / "_archives" / "musan.tar.gz"
    _download(CORPUS_URLS["musan"], arc)
    _extract(arc, out_root)
    files = _walk_wavs(out_root / "musan")
    _write_manifest("musan", files, out_root)


def fetch_rirs(out_root: Path) -> None:
    arc = out_root / "_archives" / "rirs_noises.zip"
    _download(CORPUS_URLS["rirs"], arc)
    _extract(arc, out_root)
    # Extracted as RIRS_NOISES/{simulated_rirs,real_rirs,...}
    files = _walk_wavs(out_root / "RIRS_NOISES")
    _write_manifest("rirs", files, out_root)


def fetch_librispeech(out_root: Path) -> None:
    for split in ("test_clean", "train_clean_100"):
        arc = out_root / "_archives" / f"librispeech_{split}.tar.gz"
        _download(CORPUS_URLS[f"librispeech_{split}"], arc)
        _extract(arc, out_root / "librispeech")
    files = _walk_wavs(out_root / "librispeech")
    _write_manifest("librispeech", files, out_root)


def fetch_demand(out_root: Path) -> None:
    # DEMAND is 18 zip files on Zenodo. Pull the ones likely to be in a
    # toy's deployment environment: KITCHEN, LIVING, OFFICE, PARK, BUS, STREET.
    base = "https://zenodo.org/records/1227121/files"
    envs = ["DKITCHEN", "DLIVING", "DOFFICE", "PRESTO", "STREET", "TRAFFIC"]
    out_demand = out_root / "demand"
    out_demand.mkdir(parents=True, exist_ok=True)
    for env in envs:
        arc = out_root / "_archives" / f"{env}_16k.zip"
        try:
            _download(f"{base}/{env}_16k.zip", arc)
            _extract(arc, out_demand)
        except Exception as e:
            print(f"  ✗ DEMAND {env}: {e}", file=sys.stderr)
    files = _walk_wavs(out_demand)
    _write_manifest("demand", files, out_root)


def fetch_commonvoice(out_root: Path) -> None:
    """Requires HF_TOKEN. Streams Mozilla Common Voice English."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"ERROR: pip install datasets — {e}", file=sys.stderr)
        return
    if not os.environ.get("HF_TOKEN"):
        print("WARN: HF_TOKEN not set, skipping commonvoice", file=sys.stderr)
        return
    out_cv = out_root / "commonvoice"
    out_cv.mkdir(parents=True, exist_ok=True)
    print("  streaming commonvoice/en validated split…")
    ds = load_dataset("mozilla-foundation/common_voice_17_0", "en",
                      split="validated", streaming=True, token=os.environ["HF_TOKEN"])
    import soundfile as sf
    n = 0
    for ex in ds:
        if n >= 30000:  # cap at ~50 h
            break
        try:
            audio = ex["audio"]
            wav_path = out_cv / f"cv_{n:06d}.wav"
            sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
            n += 1
        except Exception:
            continue
    files = _walk_wavs(out_cv)
    _write_manifest("commonvoice", files, out_root)


def fetch_audioset_subset(out_root: Path) -> None:
    """Best-effort: AudioSet itself is YouTube-hosted and rate-limited.
    We use the HF mirror `agkphysics/AudioSet` for a small sampled subset."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"ERROR: pip install datasets — {e}", file=sys.stderr)
        return
    out_as = out_root / "audioset"
    out_as.mkdir(parents=True, exist_ok=True)
    print("  streaming AudioSet subset (TV/Music/Household)…")
    try:
        ds = load_dataset("agkphysics/AudioSet", split="train", streaming=True)
    except Exception as e:
        print(f"  ✗ AudioSet load failed: {e}", file=sys.stderr)
        return
    import soundfile as sf
    target_labels = {"Speech", "Music", "Television", "Household sounds", "Inside, small room"}
    n = 0
    for ex in ds:
        if n >= 5000:
            break
        labels = set(ex.get("labels", []))
        if not (labels & target_labels):
            continue
        try:
            audio = ex["audio"]
            wav = out_as / f"as_{n:06d}.wav"
            sf.write(str(wav), audio["array"], audio["sampling_rate"])
            n += 1
        except Exception:
            continue
    files = _walk_wavs(out_as)
    _write_manifest("audioset", files, out_root)


FETCHERS = {
    "musan": fetch_musan,
    "rirs": fetch_rirs,
    "demand": fetch_demand,
    "librispeech": fetch_librispeech,
    "commonvoice": fetch_commonvoice,
    "audioset_subset": fetch_audioset_subset,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpora", default="musan,rirs,librispeech,demand",
                    help="Comma-separated subset of: " + ",".join(FETCHERS.keys()))
    ap.add_argument("--hours", type=int, default=300,
                    help="Soft cap on total bulk-negative hours (advisory)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== bulk negative collection → {out} ===")
    t0 = time.time()
    for corpus in args.corpora.split(","):
        fn = FETCHERS.get(corpus.strip())
        if not fn:
            print(f"WARN: unknown corpus {corpus}", file=sys.stderr)
            continue
        print(f"\n--- {corpus} ---")
        try:
            fn(out)
        except Exception as e:
            print(f"  ✗ {corpus} failed: {e}", file=sys.stderr)

    print(f"\nelapsed: {(time.time() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
