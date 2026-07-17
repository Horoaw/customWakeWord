#!/usr/bin/env python3
"""Synthesize positive examples for a wake-word project with Piper.

This follows OHF-Voice/micro-wake-word's Piper recipe. English projects use
the `en_US-libritts_r-medium.pt` generator with up to 904 speaker embeddings;
multilingual projects use configured standard Piper ONNX voices directly via
`piper-tts`, avoiding the PyTorch dependency of piper-sample-generator.

Reads `configs/examples/<project>/wake_phrases.yaml`. For each phrase,
generates audio into `data/<project>/synth/positives/<phrase_slug>/`. Writes
a unified `manifest.jsonl` at the end so downstream `build_features.py` knows
which WAVs belong to which phrase.

Resumable: if a phrase's directory already has the requested sample count,
that phrase is skipped.

Usage:
    python scripts/synth_positives.py --project tofu
    python scripts/synth_positives.py --project tofu --count 20000  # override total
    python scripts/synth_positives.py --project tofu --psg-dir ./piper-sample-generator
"""
from __future__ import annotations

import argparse
import itertools as it
import json
import math
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.request
import uuid
import wave
from pathlib import Path

import yaml


REPO_URL = "https://github.com/rhasspy/piper-sample-generator"
PSG_VERSION = "v3.0.0"
GEN_MODEL_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/"
    "v2.0.0/en_US-libritts_r-medium.pt"
)
GEN_MODEL_NAME = "en_US-libritts_r-medium.pt"
PIPER_INSTALL_HINT = (
    "Standard Piper ONNX voices require piper-tts. Install it with: "
    'python -m pip install "piper-tts>=1.3,<2"'
)


def slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower().strip()
    value = re.sub(r"[^\w-]+", "_", normalized, flags=re.UNICODE).strip("_")
    if value:
        return value
    import hashlib
    return f"phrase_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:10]}"


def ensure_psg(psg_dir: Path) -> Path:
    """Clone piper-sample-generator for the English `.pt` generator path."""
    if not psg_dir.exists():
        print(f"  cloning {REPO_URL} ({PSG_VERSION}) → {psg_dir}", flush=True)
        subprocess.check_call([
            "git", "clone", "--branch", PSG_VERSION, "--depth", "1",
            REPO_URL, str(psg_dir),
        ])

    return psg_dir


def ensure_standard_piper() -> None:
    """Fail before downloading ONNX voices when the lightweight runtime is absent."""
    try:
        import piper  # noqa: F401
    except ImportError as e:
        raise RuntimeError(PIPER_INSTALL_HINT) from e


def resolve_models(cfg: dict, psg_dir: Path | None,
                   model_cache: Path = Path("data/tts_models")) -> list[Path]:
    """Resolve configured Piper voices to local model paths.

    Existing English configs use ``voices: ALL`` and fall back to the
    LibriTTS-R generator. New multilingual configs use a list of model/config
    URLs consumed directly by piper-tts.
    """
    piper = next((e for e in cfg.get("engines", []) if e.get("name") == "piper"), None)
    if not piper:
        raise ValueError("No implemented TTS engine found. Configure an engine named 'piper'.")

    voices = piper.get("voices", "ALL")
    if voices == "ALL":
        if psg_dir is None:
            raise ValueError("English generator configuration requires piper-sample-generator")
        model_path = psg_dir / "models" / GEN_MODEL_NAME
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  downloading English generator model → {model_path}", flush=True)
            urllib.request.urlretrieve(GEN_MODEL_URL, model_path)
        return [model_path]
    if not isinstance(voices, list) or not voices:
        raise ValueError("piper voices must be 'ALL' or a non-empty list")
    models: list[Path] = []
    model_cache.mkdir(parents=True, exist_ok=True)
    for voice in voices:
        if not isinstance(voice, dict):
            raise ValueError("standard Piper voices must provide id, model_url and config_url")
        voice_id = slug(str(voice.get("id", "voice")))
        model_url = voice.get("model_url")
        config_url = voice.get("config_url")
        if not model_url or not config_url:
            raise ValueError(f"Piper voice {voice_id!r} is missing model_url/config_url")
        suffix = Path(str(model_url).split("?", 1)[0]).suffix or ".onnx"
        if suffix.lower() != ".onnx":
            raise ValueError(f"Piper voice {voice_id!r} model_url must point to an .onnx file")
        model_path = model_cache / f"{voice_id}{suffix}"
        config_path = Path(f"{model_path}.json")
        for url, dest in ((model_url, model_path), (config_url, config_path)):
            if not dest.exists():
                print(f"  downloading {voice_id} → {dest}", flush=True)
                urllib.request.urlretrieve(url, dest)
        models.append(model_path)
    return models


def generate_standard_piper_samples(phrase: str, output_dir: Path, count: int,
                                    models: list[Path],
                                    length_scales: list[float] | None,
                                    max_speakers: int | None) -> None:
    """Generate ONNX Piper samples without importing PyTorch."""
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError as e:
        raise RuntimeError(PIPER_INSTALL_HINT) from e

    voices = [PiperVoice.load(str(model), use_cuda=False) for model in models]
    scales = length_scales or [0.85, 1.0, 1.15]
    settings = it.cycle(it.product(
        voices, scales, [0.55, 0.667, 0.8], [0.65, 0.8, 0.95]
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        voice, length_scale, noise_scale, noise_w_scale = next(settings)
        num_speakers = voice.config.num_speakers
        if max_speakers is not None:
            num_speakers = min(num_speakers, max_speakers)
        speaker_id = index % max(1, num_speakers)
        with wave.open(str(output_dir / f"{index:07d}.wav"), "wb") as wav_file:
            voice.synthesize_wav(
                phrase,
                wav_file=wav_file,
                syn_config=SynthesisConfig(
                    speaker_id=speaker_id,
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_w_scale=noise_w_scale,
                ),
            )


def build_generator_command(psg_dir: Path, phrase: str, out_dir: Path,
                            count: int, batch_size: int, models: list[Path],
                            max_speakers: int | None,
                            length_scales: list[float] | None) -> tuple[list[str], str | None]:
    """Build the piper-sample-generator v3 command for an English `.pt` model."""
    if any(m.suffix != ".pt" for m in models):
        raise RuntimeError("build_generator_command is only used for `.pt` generator models")
    cmd = [
        sys.executable, str((psg_dir / "generate_samples.py").resolve()), phrase,
        "--max-samples", str(count), "--batch-size", str(batch_size),
        "--output-dir", str(out_dir.resolve()),
        "--model", str(models[0].resolve()),
    ]
    if max_speakers is not None:
        cmd.extend(["--max-speakers", str(max_speakers)])
    if length_scales:
        cmd.append("--length-scales")
        cmd.extend(str(scale) for scale in length_scales)
    return cmd, None


def generate_for_phrase(psg_dir: Path, phrase: str, out_dir: Path,
                        count: int, batch_size: int = 100,
                        max_speakers: int | None = 904,
                        models: list[Path] | None = None,
                        length_scales: list[float] | None = None,
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

    models = models or [psg_dir / "models" / GEN_MODEL_NAME]
    standard_onnx = all(model.suffix == ".onnx" for model in models)

    for attempt in range(1, max_retries + 1):
        actual_before = len(list(out_dir.glob("*.wav")))
        remaining = count - actual_before
        if remaining <= 0:
            break
        error: Exception | None = None
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=out_dir) as pending:
            pending_dir = Path(pending)
            try:
                if standard_onnx:
                    generate_standard_piper_samples(
                        phrase, pending_dir, remaining, models,
                        length_scales, max_speakers,
                    )
                else:
                    cmd, cwd = build_generator_command(
                        psg_dir, phrase, pending_dir, remaining, batch_size,
                        models, max_speakers, length_scales,
                    )
                    subprocess.check_call(cmd, cwd=cwd)
            except Exception as e:  # preserve any WAVs completed before failure
                error = e
            for wav in pending_dir.glob("*.wav"):
                wav.replace(out_dir / f"{uuid.uuid4().hex}.wav")

        if error is None:
            break

        e = error
        if isinstance(e, RuntimeError) and "require piper-tts" in str(e):
            print(f"  ✗ {phrase}: {e}", file=sys.stderr, flush=True)
            break
        else:
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

    actual = len(list(out_dir.glob("*.wav")))
    ok = actual >= max(1, math.ceil(count * min_ratio))
    return actual, ok


def write_manifest(positives_root: Path, by_phrase: dict[str, Path],
                   models: list[Path]) -> None:
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
                    "engine": (
                        "piper" if all(model.suffix == ".onnx" for model in models)
                        else "piper_sample_generator"
                    ),
                    "voice_model": ",".join(model.name for model in models),
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
                    help="English generator checkout (not used for ONNX voices).")
    ap.add_argument("--model-cache", default="data/tts_models",
                    help="Cache directory for standard Piper ONNX voices.")
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

    piper_cfg = next((e for e in cfg.get("engines", []) if e.get("name") == "piper"), {})
    uses_english_generator = piper_cfg.get("voices", "ALL") == "ALL"
    try:
        if not uses_english_generator:
            ensure_standard_piper()
        psg_dir = (
            ensure_psg(Path(args.psg_dir))
            if uses_english_generator else Path(args.psg_dir)
        )
        models = resolve_models(
            cfg, psg_dir if uses_english_generator else None, Path(args.model_cache)
        )
    except (OSError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"  models:   {', '.join(m.name for m in models)}", flush=True)
    length_scales = [float(v) for v in cfg.get("variation", {}).get("speeds", [])]

    by_phrase: dict[str, Path] = {}
    shortfalls: list[tuple[str, int, int]] = []
    for phrase, count in per_phrase_count.items():
        phrase_dir = out_root / slug(phrase)
        by_phrase[phrase] = phrase_dir
        actual, ok = generate_for_phrase(psg_dir, phrase, phrase_dir, count,
                                         batch_size=args.batch_size,
                                         max_speakers=args.max_speakers,
                                         models=models,
                                         length_scales=length_scales)
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

    write_manifest(out_root, by_phrase, models)

    total = sum(len(list(d.glob("*.wav"))) for d in by_phrase.values())
    print(f"\n=== done: {total} positive WAVs across {len(by_phrase)} phrases ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
