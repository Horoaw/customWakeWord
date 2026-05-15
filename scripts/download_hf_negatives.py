#!/usr/bin/env python3
"""Download microWakeWord's pre-built negative datasets from HuggingFace Hub.

Pulls the 7 zips at `huggingface.co/datasets/kahrendt/microwakeword`
(~9 GB total) and extracts them into the layout the upstream trainer
expects:

    data/negative_datasets/
        ├── speech/                         (training mmap)
        ├── speech_background/              (validation_ambient + testing_ambient)
        ├── dinner_party/                   (training mmap)
        ├── dinner_party_background/        (validation_ambient)
        ├── dinner_party_eval/              (validation/testing mmap, "split")
        ├── no_speech/                      (training mmap)
        └── no_speech_background/           (validation_ambient)

Each zip contains a RaggedMmap directory layout: `<name>/<name>_mmap/...`.
These are pre-computed `uint16` spectrograms — no further feature extraction
needed for the negatives.

License: CC-BY-NC-4.0 (non-commercial). See TRAINING_PLAN.md §2.3 for the
substitution path if you need commercial-safe negatives.

Usage:
    python scripts/download_hf_negatives.py --out data/negative_datasets
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


REPO_ID = "kahrendt/microwakeword"
DEFAULT_FILES = [
    "speech.zip",
    "speech_background.zip",
    "dinner_party.zip",
    "dinner_party_background.zip",
    "dinner_party_eval.zip",
    # "no_speech.zip" + "no_speech_background.zip" — dropped from v0 default.
    # These specific files consistently hang HF Hub egress on RunPod for >90 min
    # (observed across 4090 SECURE and A40 SECURE pods on 2026-05-15). The other
    # 5 zips download fine. Re-enable via `--files <comma-list>` once we have
    # a network-volume cache or a mirror.
]


def download_file(filename: str, cache_dir: Path) -> Path:
    """Use huggingface_hub to download a single file from the dataset repo."""
    from huggingface_hub import hf_hub_download
    print(f"  downloading {filename}…", flush=True)
    return Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    ))


def extract_zip(zip_path: Path, dest_root: Path) -> None:
    """Extract a zip into dest_root. Skip if the inner directory already exists."""
    dest_root.mkdir(parents=True, exist_ok=True)
    # Peek at the archive to find its top-level directory.
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        top_levels = {Path(n).parts[0] for n in names if n.strip()}
    target_dirs = [dest_root / t for t in top_levels]
    if all(d.exists() for d in target_dirs) and target_dirs:
        print(f"    ✓ {zip_path.name}: already extracted ({[d.name for d in target_dirs]})", flush=True)
        return
    print(f"    extracting {zip_path.name} → {dest_root}", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/negative_datasets",
                    help="Where to extract the zips.")
    ap.add_argument("--cache", default=".hf_cache",
                    help="HuggingFace download cache (kept across runs).")
    ap.add_argument("--files", default=",".join(DEFAULT_FILES),
                    help="Comma-separated list of zips to download.")
    ap.add_argument("--keep-zip", action="store_true",
                    help="Keep the HF cache after extracting (useful if you'll re-extract).")
    args = ap.parse_args()

    out = Path(args.out)
    cache = Path(args.cache)
    files = [f.strip() for f in args.files.split(",") if f.strip()]

    print(f"=== HF negatives → {out} ===", flush=True)
    print(f"  repo:  {REPO_ID}")
    print(f"  files: {len(files)}")
    print(f"  cache: {cache}")
    print(flush=True)

    for fn in files:
        try:
            local = download_file(fn, cache)
            extract_zip(local, out)
        except Exception as e:
            print(f"  ✗ {fn}: {e}", file=sys.stderr)

    if not args.keep_zip and cache.exists():
        print(f"\n  cleaning HF cache at {cache} …", flush=True)
        shutil.rmtree(cache, ignore_errors=True)

    print(f"\n=== done. extracted to {out} ===", flush=True)
    # List what's there for the user to verify.
    for d in sorted(out.iterdir()):
        if d.is_dir():
            mmaps = list(d.rglob("*_mmap"))
            print(f"  {d.name:35s} ({len(mmaps)} mmap dirs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
