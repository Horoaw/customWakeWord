"""Run a trained .tflite against the held-out task set, score FRR + FAR/hour.

Usage:
    python -m eval.runner --project tofu \\
        --model models/tofu-wakeword-v0.tflite \\
        --tasks eval/tasks/tofu \\
        --out eval/results/tofu-v0__$(date +%s).json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eval.schema import EvalResult, EvalSummary, EvalTask


def load_tasks(tasks_dir: Path) -> list[EvalTask]:
    out = []
    for bucket in ("positives", "hard_negatives", "bulk"):
        d = tasks_dir / bucket
        if not d.exists():
            continue
        for jp in sorted(d.glob("*.json")):
            data = json.loads(jp.read_text())
            out.append(EvalTask(**data))
    return out


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    import soundfile as sf
    import librosa
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def compute_log_mel(audio, sr, n_mels=40, win_ms=60, hop_ms=25,
                    fmin=20, fmax=7600) -> np.ndarray:
    import librosa
    win_length = int(sr * win_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    n_fft = 1 << (win_length - 1).bit_length()
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


class TFLiteRunner:
    def __init__(self, model_path: str):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter  # type: ignore
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.in_details = self.interp.get_input_details()[0]
        self.out_details = self.interp.get_output_details()[0]
        self.in_shape = self.in_details["shape"]   # e.g. [1, 40, 194]
        # INT8 input quant params
        self.in_scale, self.in_zero = self.in_details.get("quantization", (1.0, 0))
        self.out_scale, self.out_zero = self.out_details.get("quantization", (1.0, 0))

    def predict(self, features: np.ndarray) -> float:
        """Run a single window through the model. Returns float probability in [0, 1]."""
        if self.in_details["dtype"] == np.int8:
            q = np.round(features / max(self.in_scale, 1e-9) + self.in_zero).astype(np.int32)
            q = np.clip(q, -128, 127).astype(np.int8)
            inp = q.reshape(self.in_shape)
        else:
            inp = features.astype(np.float32).reshape(self.in_shape)
        self.interp.set_tensor(self.in_details["index"], inp)
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_details["index"])
        if self.out_details["dtype"] == np.int8:
            return float((int(out.flat[0]) - self.out_zero) * self.out_scale)
        return float(out.flat[0])


def stream_features(audio: np.ndarray, sr: int, n_frames: int,
                    hop_features: int = 20) -> list[np.ndarray]:
    """Slide an `n_frames`-long mel window over the clip with `hop_features` of stride."""
    mel = compute_log_mel(audio, sr)
    t = mel.shape[1]
    if t <= n_frames:
        # Pad single window
        if t < n_frames:
            mel = np.pad(mel, ((0, 0), (0, n_frames - t)), mode="constant")
        return [mel]
    out = []
    start = 0
    while start + n_frames <= t:
        out.append(mel[:, start:start + n_frames])
        start += hop_features
    return out


def run_one(runner: TFLiteRunner, task: EvalTask, threshold: float,
            n_frames: int = 194) -> EvalResult:
    t0 = time.time()
    try:
        audio = load_audio(task.audio_path)
        windows = stream_features(audio, 16000, n_frames)
        probs = [runner.predict(w) for w in windows]
    except Exception as e:
        return EvalResult(
            task_id=task.id, label=task.label, bucket_id=task.bucket_id,
            fired=False, expected_fire=(task.expected == "fire"), passed=False,
            max_probability=0.0, fire_count=0, duration_s=time.time() - t0,
            error_msg=str(e),
        )
    max_prob = max(probs) if probs else 0.0
    fire_count = sum(1 for p in probs if p >= threshold)
    fired = fire_count > 0
    expected_fire = task.expected == "fire"
    passed = (fired == expected_fire)
    return EvalResult(
        task_id=task.id, label=task.label, bucket_id=task.bucket_id,
        fired=fired, expected_fire=expected_fire, passed=passed,
        max_probability=max_prob, fire_count=fire_count,
        duration_s=time.time() - t0,
    )


def summarize(results: list[EvalResult], threshold: float, project: str,
              model_path: str, bulk_minutes: float) -> EvalSummary:
    pos = [r for r in results if r.label == "positive"]
    hn = [r for r in results if r.label == "hard_negative"]
    bulk = [r for r in results if r.label == "bulk_negative"]

    frr = 0.0
    if pos:
        n_missed = sum(1 for r in pos if not r.fired)
        frr = n_missed / len(pos)

    # FAR/hour from bulk: each bulk task is treated as a small slice; sum fire counts
    # and divide by total bulk audio duration in hours.
    bulk_fires = sum(r.fire_count for r in bulk)
    far_per_hour = bulk_fires / max(bulk_minutes / 60.0, 1e-6) if bulk_minutes > 0 else 0.0

    per_bucket = defaultdict(list)
    for r in hn:
        if r.bucket_id:
            per_bucket[r.bucket_id].append(r)
    per_bucket_far = {
        bid: sum(1 for r in rs if r.fired) / len(rs)
        for bid, rs in per_bucket.items() if rs
    }

    return EvalSummary(
        project=project,
        model_path=model_path,
        threshold=threshold,
        n_positives=len(pos),
        n_hard_negatives=len(hn),
        bulk_stream_minutes=bulk_minutes,
        frr=frr,
        far_per_hour=far_per_hour,
        per_bucket_far=per_bucket_far,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--model", required=True, help="Path to .tflite")
    ap.add_argument("--tasks", default=None,
                    help="Default: eval/tasks/<project>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="Detection threshold; FAR/FRR computed at this op-point.")
    ap.add_argument("--n-frames", type=int, default=194)
    ap.add_argument("--bulk-minutes", type=float, default=60.0,
                    help="Total minutes of bulk audio in the eval set (for FAR/hour denom).")
    args = ap.parse_args()

    tasks_dir = Path(args.tasks) if args.tasks else Path(f"eval/tasks/{args.project}")
    if not tasks_dir.exists():
        print(f"ERROR: tasks dir {tasks_dir} not found", file=sys.stderr)
        return 1
    tasks = load_tasks(tasks_dir)
    if not tasks:
        print(f"ERROR: no tasks under {tasks_dir}", file=sys.stderr)
        return 1
    print(f"Running {len(tasks)} tasks on {args.model} @ threshold {args.threshold}")

    runner = TFLiteRunner(args.model)
    results: list[EvalResult] = []
    for i, task in enumerate(tasks):
        if i % 25 == 0:
            print(f"  [{i}/{len(tasks)}]")
        r = run_one(runner, task, args.threshold, args.n_frames)
        results.append(r)

    summary = summarize(results, args.threshold, args.project,
                        args.model, args.bulk_minutes)
    print()
    print(f"  FRR:          {summary.frr:.2%}")
    print(f"  FAR/hour:     {summary.far_per_hour:.2f}")
    print(f"  per-bucket FAR:")
    for bid, val in sorted(summary.per_bucket_far.items()):
        print(f"    {bid}: {val:.2%}")

    out_path = Path(args.out) if args.out else (
        Path("eval/results") / f"{args.project}-v0__{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }, indent=2))
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
