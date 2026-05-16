#!/usr/bin/env python3
"""Convert positive + hard-negative WAVs into upstream-compatible RaggedMmap features.

Replaces my earlier from-scratch TFRecord builder. This version delegates
to `microwakeword.audio.audio_utils.generate_features_for_clip` (the C
`audio_microfrontend` op via pymicro-features) so the features are
bit-exact with what the on-device ESPHome runtime expects.

For positives (and hard-negatives), each WAV goes through:
    1. `microwakeword.audio.augmentation.Augmentation` — RIR + background-noise
       mixing using HF corpora (auto-downloaded if --download-aug-corpora).
    2. `microwakeword.audio.spectrograms.SpectrogramGeneration` — produces
       sliding spectrogram windows (slide_frames=10 for train, =1 for val/test).
    3. `mmap_ninja.ragged.RaggedMmap.from_generator` — writes the spectrograms.

Output layout (matches upstream's FeatureHandler expectations):
    data/<project>/features/
        ├── training/wakeword_mmap/
        ├── validation/wakeword_mmap/
        └── testing/wakeword_mmap/
    data/<project>/hard_negatives_features/
        ├── training/wakeword_mmap/
        ├── validation/wakeword_mmap/
        └── testing/wakeword_mmap/

Bulk negatives are already mmap'd by `download_hf_negatives.py` — this
script does NOT process them.

Usage:
    python scripts/build_features.py --project tofu
    python scripts/build_features.py --project tofu --download-aug-corpora
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def assign_split(file_id: str, train_p: float = 0.8, val_p: float = 0.1) -> str:
    """Deterministic hash split — same file_id always lands in the same split."""
    h = int(hashlib.md5(file_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < train_p:
        return "training"
    if h < train_p + val_p:
        return "validation"
    return "testing"


def load_manifest_paths(manifest_jsonl: Path) -> list[tuple[str, str]]:
    """Return list of (file_id, wav_path) for each row in the manifest."""
    out = []
    if not manifest_jsonl.exists():
        return out
    for line in manifest_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        wav = d.get("wav_path")
        fid = d.get("file_id") or Path(wav).stem
        if wav and Path(wav).exists():
            out.append((fid, wav))
    return out


def maybe_download_aug_corpora(work_dir: Path) -> tuple[list[str], list[str]]:
    """Download the RIR + background-noise corpora the canonical notebook uses.

    Returns (impulse_paths, background_paths).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    rir_dir = work_dir / "mit_rirs"
    fma_dir = work_dir / "fma_16k"
    as_dir = work_dir / "audioset_16k"

    if not rir_dir.exists():
        print(f"  downloading MIT IR survey → {rir_dir}", flush=True)
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "davidscripka/MIT_environmental_impulse_responses",
                split="train", streaming=True,
            )
            import soundfile as sf
            rir_dir.mkdir(parents=True, exist_ok=True)
            for i, ex in enumerate(ds):
                if i >= 300:
                    break
                a = ex["audio"]
                sf.write(str(rir_dir / f"rir_{i:05d}.wav"),
                         a["array"], a["sampling_rate"])
        except Exception as e:
            print(f"  ⚠ RIR download failed: {e}", file=sys.stderr)

    # FMA-XS music
    if not fma_dir.exists():
        print(f"  downloading FMA-xsmall → {fma_dir}", flush=True)
        try:
            import urllib.request
            arc = work_dir / "fma_xs.zip"
            if not arc.exists():
                urllib.request.urlretrieve(
                    "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip",
                    arc,
                )
            import zipfile
            with zipfile.ZipFile(arc) as z:
                z.extractall(fma_dir)
        except Exception as e:
            print(f"  ⚠ FMA-XS download failed: {e}", file=sys.stderr)

    # AudioSet shard
    if not as_dir.exists():
        print(f"  downloading AudioSet bal_train09 → {as_dir}", flush=True)
        try:
            from huggingface_hub import hf_hub_download
            import tarfile
            tar_path = Path(hf_hub_download(
                repo_id="agkphysics/AudioSet",
                filename="bal_train09.tar",
                repo_type="dataset",
            ))
            with tarfile.open(tar_path) as tf:
                tf.extractall(as_dir)
        except Exception as e:
            print(f"  ⚠ AudioSet download failed: {e}", file=sys.stderr)

    impulse_paths = [str(rir_dir)] if rir_dir.exists() else []
    background_paths = [str(p) for p in (fma_dir, as_dir) if p.exists()]
    return impulse_paths, background_paths


def build_features_for_corpus(wavs: list[tuple[str, str]], out_root: Path,
                              impulse_paths: list[str], background_paths: list[str],
                              train_p: float = 0.8, val_p: float = 0.1) -> None:
    """For each WAV, generate augmented spectrograms and write to RaggedMmap dirs.

    Splits per-WAV deterministically by file_id hash; positive replication
    (`slide_frames=10` for training, 1 for val/test) is handled by upstream's
    SpectrogramGeneration.
    """
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    out_root.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = {"training": [], "validation": [], "testing": []}
    for fid, path in wavs:
        by_split[assign_split(fid, train_p, val_p)].append(path)

    aug_probs = {
        "SevenBandParametricEQ": 0.1, "TanhDistortion": 0.1,
        "PitchShift": 0.1, "BandStopFilter": 0.1,
        "AddColorNoise": 0.1, "AddBackgroundNoise": 0.75,
        "Gain": 1.0, "RIR": 0.5,
    }
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities=aug_probs,
        impulse_paths=impulse_paths,
        background_paths=background_paths,
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    for split, paths in by_split.items():
        if not paths:
            continue
        # RunPod pod disks intermittently fail mid-write with OSError
        # [Errno 5], leaving WAVs in three broken states: missing,
        # zero-byte, or size-claims-OK-but-data-never-flushed (sparse
        # or pre-allocated). A `Path.stat().st_size` check misses the
        # third case — the file appears non-empty but `wave.open()`
        # errors or the datasets audio decoder hits FileNotFoundError
        # on the symlink path. Open each WAV header to validate.
        import wave as _wave
        def _readable_wav(p: str) -> bool:
            try:
                if not Path(p).is_file() or Path(p).stat().st_size < 4096:
                    return False
                with _wave.open(p, "rb") as w:
                    return w.getnframes() > 0
            except Exception:
                return False
        before = len(paths)
        paths = [p for p in paths if _readable_wav(p)]
        if len(paths) < before:
            print(f"  ⚠ {split}: dropped {before - len(paths)} dead/unreadable WAVs "
                  f"(probably RunPod disk Errno 5 from synth stage)",
                  file=sys.stderr, flush=True)
        if not paths:
            print(f"  ⚠ {split}: no usable WAVs after filter, skipping split",
                  file=sys.stderr, flush=True)
            continue
        mmap_dir = out_root / split / "wakeword_mmap"
        if mmap_dir.exists():
            print(f"  ✓ {split}: {mmap_dir} already exists, skipping", flush=True)
            continue
        mmap_dir.parent.mkdir(parents=True, exist_ok=True)

        # Stage WAVs into a flat directory of symlinks so Clips can find
        # them via glob. Upstream microwakeword dropped the
        # `filepath_text_files=[...]` kwarg in favour of
        # `input_directory + file_pattern` only — symlinks let us keep
        # the original storage layout (per-phrase subdirs) while
        # presenting a flat view to Clips without copying WAV bytes.
        #
        # CRITICAL: symlinks store their target string verbatim, and the
        # kernel resolves relative targets from the SYMLINK'S directory,
        # not from the process CWD. The manifest's wav_path is relative
        # ("data/tofu/synth/positives/hey_tofu/0.wav") which would point
        # to a non-existent location when followed from
        # data/tofu/features/training/_clips/. Resolve to absolute first.
        symlink_dir = out_root / split / "_clips"
        symlink_dir.mkdir(parents=True, exist_ok=True)
        for old in symlink_dir.glob("*.wav"):
            old.unlink()
        for i, p in enumerate(paths):
            link = symlink_dir / f"{i:07d}.wav"
            link.symlink_to(Path(p).resolve())

        clips = Clips(input_directory=str(symlink_dir), file_pattern="*.wav")

        slide_frames = 10 if split == "training" else 1
        repetition = 2 if split == "training" else 1
        specgen = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            step_ms=10,
            slide_frames=slide_frames,
        )

        def _gen():
            for _ in range(repetition):
                yield from specgen.spectrogram_generator()

        print(f"  building {split} → {mmap_dir} ({len(paths)} WAVs, x{repetition})",
              flush=True)
        RaggedMmap.from_generator(
            out_dir=str(mmap_dir),
            sample_generator=_gen(),
            batch_size=100,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--positives-manifest", default=None,
                    help="Default: data/<project>/synth/positives/manifest.jsonl")
    ap.add_argument("--hard-negs-manifest", default=None,
                    help="Default: data/<project>/synth/hard_negatives/manifest.jsonl")
    ap.add_argument("--out-features", default=None,
                    help="Default: data/<project>/features")
    ap.add_argument("--out-hard-negs", default=None,
                    help="Default: data/<project>/hard_negatives_features")
    ap.add_argument("--aug-work-dir", default="data/aug_corpora",
                    help="Where to cache RIR + noise + music corpora.")
    ap.add_argument("--download-aug-corpora", action="store_true",
                    help="Download MIT IR + FMA-XS + AudioSet shard if missing.")
    ap.add_argument("--no-aug", action="store_true",
                    help="Skip augmentation (faster for sanity tests; do NOT use for v0).")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    args = ap.parse_args()

    project = args.project
    pos_mf = Path(args.positives_manifest) if args.positives_manifest else Path(
        f"data/{project}/synth/positives/manifest.jsonl")
    hn_mf = Path(args.hard_negs_manifest) if args.hard_negs_manifest else Path(
        f"data/{project}/synth/hard_negatives/manifest.jsonl")
    out_features = Path(args.out_features) if args.out_features else Path(
        f"data/{project}/features")
    out_hn = Path(args.out_hard_negs) if args.out_hard_negs else Path(
        f"data/{project}/hard_negatives_features")

    pos_wavs = load_manifest_paths(pos_mf)
    hn_wavs = load_manifest_paths(hn_mf)
    print(f"=== build features for '{project}' ===", flush=True)
    print(f"  positives:    {len(pos_wavs)} WAVs from {pos_mf}")
    print(f"  hard-negs:    {len(hn_wavs)} WAVs from {hn_mf}")
    print(f"  out features: {out_features}")
    print(f"  out hard-negs: {out_hn}")
    print(flush=True)

    if not pos_wavs:
        print(f"ERROR: no positive WAVs at {pos_mf}. Run synth_positives.py first.",
              file=sys.stderr)
        return 1

    impulse_paths: list[str] = []
    background_paths: list[str] = []
    if not args.no_aug:
        if args.download_aug_corpora:
            impulse_paths, background_paths = maybe_download_aug_corpora(Path(args.aug_work_dir))
        else:
            work = Path(args.aug_work_dir)
            if (work / "mit_rirs").exists():
                impulse_paths = [str(work / "mit_rirs")]
            for d in ("fma_16k", "audioset_16k"):
                if (work / d).exists():
                    background_paths.append(str(work / d))
            if not impulse_paths or not background_paths:
                print("WARN: augmentation corpora missing; pass --download-aug-corpora "
                      "to fetch them. Proceeding with whatever's present.", file=sys.stderr)

    build_features_for_corpus(pos_wavs, out_features, impulse_paths, background_paths,
                              train_p=args.train, val_p=args.val)
    if hn_wavs:
        build_features_for_corpus(hn_wavs, out_hn, impulse_paths, background_paths,
                                  train_p=args.train, val_p=args.val)

    print(f"\n=== done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
