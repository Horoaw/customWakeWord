#!/usr/bin/env python3
"""Synthesize TTS positives for the Tofu wake-word.

Reads `configs/wake_phrases.yaml`, fans out across registered TTS engines
(Piper, Kokoro, MeloTTS, Parler-TTS) and their voices, and writes 16 kHz
mono WAVs to `data/synth/positives/` along with a `manifest.jsonl` line
per file.

Resumable: if `manifest.jsonl` already exists, this script picks up where
it left off — useful because Mac CPU TTS for 10k samples can take ~30 min
and you may want to interrupt.

Usage:
    python scripts/synth_positives.py \\
        --phrases configs/wake_phrases.yaml \\
        --out data/synth/positives \\
        --count 10000

The four engines are loaded lazily so a missing one (e.g. you didn't
install MeloTTS) is degraded gracefully — that engine is just skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pip install pyyaml", file=sys.stderr); sys.exit(1)


# -----------------------------------------------------------------------------
# Engine adapters
# -----------------------------------------------------------------------------

class _EngineBase:
    name: str = "base"

    def list_voices(self, hint) -> list[str]:
        raise NotImplementedError

    def synth(self, text: str, voice: str, speed: float, emotion: str | None,
              out_path: Path, sample_rate: int) -> bool:
        """Render `text` with `voice` at `speed` to `out_path`. Return True on success."""
        raise NotImplementedError


class PiperEngine(_EngineBase):
    """Wraps `piper-tts` (pip) or the precompiled `piper` binary on Mac."""
    name = "piper"

    def __init__(self):
        try:
            from piper import PiperVoice  # noqa: F401
            self._mode = "py"
        except Exception:
            import shutil
            if shutil.which("piper"):
                self._mode = "bin"
            else:
                raise RuntimeError(
                    "piper not available (neither `pip install piper-tts` nor `piper` on PATH)"
                )
        self._voice_cache: dict[str, object] = {}

    def list_voices(self, hint) -> list[str]:
        # Conventional layout: ~/.local/share/piper/voices/<lang>/<voice>/*.onnx
        # If the user only has a couple of voices installed, we use those.
        # If hint == "ALL" we try every directory; if a list, we use only those.
        voices_dir = Path.home() / ".local/share/piper/voices"
        if not voices_dir.exists():
            return []
        all_voices: list[str] = []
        for p in voices_dir.rglob("*.onnx"):
            all_voices.append(p.stem)
        if hint == "ALL":
            return sorted(set(all_voices))
        if isinstance(hint, list):
            return [v for v in all_voices if v in hint]
        return sorted(set(all_voices))

    def synth(self, text, voice, speed, emotion, out_path, sample_rate):
        # Piper exposes a length_scale knob: <1 = faster, >1 = slower.
        # speed=1.05 → length_scale = 1/1.05 ≈ 0.952.
        length_scale = 1.0 / max(speed, 0.1)
        if self._mode == "py":
            return self._synth_py(text, voice, length_scale, out_path, sample_rate)
        return self._synth_bin(text, voice, length_scale, out_path, sample_rate)

    def _synth_py(self, text, voice, length_scale, out_path, sample_rate):
        from piper import PiperVoice
        import wave
        if voice not in self._voice_cache:
            model_path = next(
                (Path.home() / ".local/share/piper/voices").rglob(f"{voice}.onnx"),
                None,
            )
            if model_path is None:
                return False
            self._voice_cache[voice] = PiperVoice.load(str(model_path))
        v = self._voice_cache[voice]
        with wave.open(str(out_path), "wb") as wf:
            v.synthesize(text, wf, length_scale=length_scale)
        return True

    def _synth_bin(self, text, voice, length_scale, out_path, sample_rate):
        import subprocess
        model_path = next(
            (Path.home() / ".local/share/piper/voices").rglob(f"{voice}.onnx"),
            None,
        )
        if model_path is None:
            return False
        try:
            subprocess.run(
                ["piper", "--model", str(model_path),
                 "--output_file", str(out_path),
                 "--length_scale", str(length_scale)],
                input=text.encode(), check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return False


class KokoroEngine(_EngineBase):
    name = "kokoro"

    def __init__(self):
        try:
            from kokoro import KPipeline
        except Exception as e:
            raise RuntimeError(f"kokoro not installed: {e}")
        # Lazy: only instantiate the pipeline for the languages we use.
        self._pipelines: dict[str, "KPipeline"] = {}
        self._KPipeline = KPipeline

    def _get_pipeline(self, lang: str):
        if lang not in self._pipelines:
            self._pipelines[lang] = self._KPipeline(lang_code=lang)
        return self._pipelines[lang]

    def list_voices(self, hint) -> list[str]:
        en_voices = [
            "af_alloy", "af_aoede", "af_bella", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
            "am_onyx", "am_puck", "am_santa",
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        ]
        if hint == "ALL_EN" or hint == "ALL":
            return en_voices
        if isinstance(hint, list):
            return [v for v in en_voices if v in hint]
        return en_voices

    def synth(self, text, voice, speed, emotion, out_path, sample_rate):
        import numpy as np
        import soundfile as sf
        lang = "a" if voice.startswith(("af_", "am_")) else "b"
        pipe = self._get_pipeline(lang)
        audio_chunks: list[np.ndarray] = []
        for _, _, audio in pipe(text, voice=voice, speed=speed):
            audio_chunks.append(audio)
        if not audio_chunks:
            return False
        full = np.concatenate(audio_chunks)
        sf.write(str(out_path), full, sample_rate)
        return True


class MeloTTSEngine(_EngineBase):
    name = "melotts"

    def __init__(self):
        try:
            from melo.api import TTS
        except Exception as e:
            raise RuntimeError(f"MeloTTS not installed: {e}")
        self._TTS = TTS
        self._models: dict[str, object] = {}

    def _get(self, voice):
        if voice not in self._models:
            self._models[voice] = self._TTS(language=voice, device="auto")
        return self._models[voice]

    def list_voices(self, hint) -> list[str]:
        defaults = ["EN-US", "EN-BR", "EN-AU", "EN-IN"]
        if isinstance(hint, list):
            return [v for v in defaults if v in hint]
        return defaults

    def synth(self, text, voice, speed, emotion, out_path, sample_rate):
        m = self._get(voice)
        speaker_ids = m.hps.data.spk2id
        speaker_id = next(iter(speaker_ids.values()))
        m.tts_to_file(text, speaker_id, str(out_path), speed=speed)
        return True


class ParlerTTSEngine(_EngineBase):
    name = "parler"

    def __init__(self):
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            import torch
        except Exception as e:
            raise RuntimeError(f"parler-tts not installed: {e}")
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self._device = device
        self._tok = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-mini-v1"
        ).to(device)
        self._torch = torch

    def list_voices(self, hint) -> list[str]:
        # Parler doesn't have discrete voices — "voice" is a free-text style prompt.
        # We synthesize a fixed grid of 20 random style prompts at init.
        rng = random.Random(2026)
        bases = [
            "a young woman", "a middle-aged man", "an elderly woman",
            "a teenage boy", "a small child", "a young man with an Australian accent",
            "a woman with a British accent", "a man with an Indian accent",
            "a soft-spoken woman", "an excited man",
        ]
        moods = ["calmly", "excitedly", "warmly", "happily", "playfully"]
        prompts = []
        for _ in range(20):
            prompts.append(f"{rng.choice(bases)} speaking {rng.choice(moods)}, clearly")
        return prompts

    def synth(self, text, voice, speed, emotion, out_path, sample_rate):
        import soundfile as sf
        full_prompt = voice if not emotion else f"{voice}; {emotion}"
        desc = self._tok(full_prompt, return_tensors="pt").input_ids.to(self._device)
        ids = self._tok(text, return_tensors="pt").input_ids.to(self._device)
        with self._torch.no_grad():
            gen = self._model.generate(input_ids=desc, prompt_input_ids=ids)
        audio = gen.cpu().numpy().squeeze()
        sf.write(str(out_path), audio, self._model.config.sampling_rate)
        return True


ENGINES_REGISTRY = {
    "piper": PiperEngine,
    "kokoro": KokoroEngine,
    "melotts": MeloTTSEngine,
    "parler": ParlerTTSEngine,
}


def load_engines(names: list[str]) -> dict[str, _EngineBase]:
    out: dict[str, _EngineBase] = {}
    for name in names:
        cls = ENGINES_REGISTRY.get(name)
        if not cls:
            print(f"WARN: unknown engine {name}, skipping", file=sys.stderr)
            continue
        try:
            out[name] = cls()
            print(f"  ✓ engine loaded: {name}")
        except Exception as e:
            print(f"  ✗ engine {name} unavailable: {e}", file=sys.stderr)
    return out


# -----------------------------------------------------------------------------
# Manifest + dispatch
# -----------------------------------------------------------------------------

def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def file_id(voice: str, phrase: str, speed: float, emotion: str | None, seed: int) -> str:
    h = hashlib.md5(f"{voice}|{phrase}|{speed}|{emotion}|{seed}".encode()).hexdigest()[:8]
    return f"{slug(voice)[:24]}__{slug(phrase)[:32]}__{speed:.2f}__{h}"


def load_manifest(out_dir: Path) -> set[str]:
    """Return set of file_ids already synthesized."""
    mfp = out_dir / "manifest.jsonl"
    if not mfp.exists():
        return set()
    seen = set()
    for line in mfp.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        seen.add(d["file_id"])
    return seen


def append_manifest(out_dir: Path, row: dict) -> None:
    mfp = out_dir / "manifest.jsonl"
    with mfp.open("a") as f:
        f.write(json.dumps(row) + "\n")


def plan_jobs(cfg: dict, engines: dict[str, _EngineBase], total_count: int) -> list[dict]:
    """Build a flat list of (engine, voice, phrase, speed, emotion) jobs sampled
    according to per-phrase counts and per-engine weights."""
    rng = random.Random(cfg["variation"]["random_seed"])
    phrases = cfg["phrases"]

    # Per-phrase target counts, scaled to total_count if needed.
    raw_total = sum(p["count"] for p in phrases)
    scale = total_count / raw_total if raw_total else 1.0
    per_phrase_count = {p["text"]: max(1, int(round(p["count"] * scale))) for p in phrases}

    # Engine weights → cumulative distribution.
    eng_weights = [(e["name"], float(e["weight"])) for e in cfg["engines"] if e["name"] in engines]
    if not eng_weights:
        return []
    w_total = sum(w for _, w in eng_weights)
    eng_cdf: list[tuple[str, float]] = []
    cum = 0.0
    for name, w in eng_weights:
        cum += w / w_total
        eng_cdf.append((name, cum))

    speeds = cfg["variation"]["speeds"]
    emotions_for: dict[str, list[str]] = cfg["variation"].get("emotions", {})

    jobs: list[dict] = []
    for phrase_cfg in phrases:
        phrase = phrase_cfg["text"]
        n = per_phrase_count[phrase]
        for _ in range(n):
            r = rng.random()
            engine_name = next(name for name, cum in eng_cdf if r <= cum)
            engine = engines[engine_name]
            voice_hint = next((e.get("voices") for e in cfg["engines"] if e["name"] == engine_name), "ALL")
            voice_list = engine.list_voices(voice_hint)
            if not voice_list:
                continue
            voice = rng.choice(voice_list)
            speed = rng.choice(speeds)
            emotion = None
            if engine_name in emotions_for:
                emotion = rng.choice(emotions_for[engine_name])
            seed = rng.randint(0, 2**31 - 1)
            jobs.append({
                "engine": engine_name,
                "voice": voice,
                "phrase": phrase,
                "speed": speed,
                "emotion": emotion,
                "seed": seed,
            })
    rng.shuffle(jobs)
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", default="configs/wake_phrases.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--engines", default="piper,kokoro,melotts,parler",
                    help="Comma-separated subset of available engines")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan jobs, print summary, exit without synthesis")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.phrases).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Tofu wake-word positive synthesis ===")
    print(f"  config:  {args.phrases}")
    print(f"  out:     {out_dir}")
    print(f"  count:   {args.count}")
    print(f"  engines: {args.engines}")
    print()

    engines = load_engines(args.engines.split(","))
    if not engines:
        print("ERROR: no engines available; install at least one TTS engine", file=sys.stderr)
        return 1

    jobs = plan_jobs(cfg, engines, args.count)
    print(f"\nplanned {len(jobs)} jobs across {len(engines)} engines")

    if args.dry_run:
        print(json.dumps(jobs[:5], indent=2))
        return 0

    seen = load_manifest(out_dir)
    sr = cfg["audio"]["sample_rate"]
    t0 = time.time()
    done = 0
    for job in jobs:
        fid = file_id(job["voice"], job["phrase"], job["speed"], job["emotion"], job["seed"])
        if fid in seen:
            continue
        wav_path = out_dir / f"{fid}.wav"
        engine = engines[job["engine"]]
        ok = False
        try:
            ok = engine.synth(job["phrase"], job["voice"], job["speed"],
                              job["emotion"], wav_path, sr)
        except Exception as e:
            print(f"  ✗ {fid}: {e}", file=sys.stderr)
        if not ok or not wav_path.exists():
            continue
        row = {
            "file_id": fid,
            "wav_path": str(wav_path.relative_to(out_dir.parent.parent)),
            "engine": job["engine"],
            "voice": job["voice"],
            "phrase": job["phrase"],
            "speed": job["speed"],
            "emotion": job["emotion"],
            "seed": job["seed"],
            "label": "positive",
        }
        append_manifest(out_dir, row)
        done += 1
        if done % 100 == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            remaining = (len(jobs) - len(seen) - done) / max(rate, 1e-6)
            print(f"  [{done}/{len(jobs) - len(seen)}] {rate:.1f} samples/s, ~{remaining/60:.1f} min left")

    print(f"\ndone. {done} new samples, manifest at {out_dir}/manifest.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
