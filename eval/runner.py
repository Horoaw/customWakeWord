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
import math
import time
from collections import deque
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
            data = json.loads(jp.read_text(encoding="utf-8"))
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


def compute_micro_features(audio: np.ndarray) -> np.ndarray:
    """Generate the same 40-channel MicroFrontend features used in training."""
    from pymicro_features import MicroFrontend

    pcm16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    audio_bytes = pcm16.tobytes()
    frontend = MicroFrontend()
    features = []
    idx = 0
    while idx + 160 * 2 < len(audio_bytes):
        result = frontend.process_samples(audio_bytes[idx: idx + 160 * 2])
        if result.samples_read <= 0:
            break
        idx += result.samples_read * 2
        if result.features:
            features.append(result.features)
    if not features:
        return np.empty((0, 40), dtype=np.float32)
    return np.asarray(features, dtype=np.float32)


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
        self.in_shape = tuple(int(v) for v in self.in_details["shape"])
        if self.in_shape[-1] != 40:
            raise ValueError(f"expected a 40-feature streaming input, got {self.in_shape}")
        self.n_slices = math.prod(self.in_shape) // 40
        # INT8 input quant params
        self.in_scale, self.in_zero = self.in_details.get("quantization", (1.0, 0))
        self.out_scale, self.out_zero = self.out_details.get("quantization", (1.0, 0))

    def reset(self) -> None:
        reset = getattr(self.interp, "reset_all_variables", None)
        if reset:
            reset()

    def predict(self, features: np.ndarray) -> float:
        """Run a single window through the model. Returns float probability in [0, 1]."""
        in_dtype = self.in_details["dtype"]
        if np.issubdtype(in_dtype, np.integer):
            q = np.round(features / max(self.in_scale, 1e-9) + self.in_zero).astype(np.int32)
            limits = np.iinfo(in_dtype)
            q = np.clip(q, limits.min, limits.max).astype(in_dtype)
            inp = q.reshape(self.in_shape)
        else:
            inp = features.astype(np.float32).reshape(self.in_shape)
        self.interp.set_tensor(self.in_details["index"], inp)
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_details["index"])
        if np.issubdtype(self.out_details["dtype"], np.integer):
            return float((int(out.flat[0]) - self.out_zero) * self.out_scale)
        return float(out.flat[0])


def stream_features(audio: np.ndarray, n_slices: int) -> list[np.ndarray]:
    """Split MicroFrontend output into the chunks expected by the streaming model."""
    features = compute_micro_features(audio)
    if not len(features):
        return []
    usable = len(features) - (len(features) % n_slices)
    return [features[i:i + n_slices] for i in range(0, usable, n_slices)]


def detection_events(probabilities: list[float], threshold: float,
                     sliding_window_size: int, step_ms: int) -> tuple[list[float], int]:
    """Apply ESPHome-style probability smoothing and count distinct detections."""
    window: deque[float] = deque(maxlen=max(1, sliding_window_size))
    smoothed: list[float] = []
    events = 0
    active = False
    for probability in probabilities:
        window.append(probability)
        average = sum(window) / len(window)
        smoothed.append(average)
        above = len(window) == window.maxlen and average >= threshold
        if above and not active:
            events += 1
        active = above
    return smoothed, events


def run_one(runner: TFLiteRunner, task: EvalTask, threshold: float,
            sliding_window_size: int = 5, feature_step_ms: int = 10) -> EvalResult:
    t0 = time.time()
    try:
        audio = load_audio(task.audio_path)
        runner.reset()
        windows = stream_features(audio, runner.n_slices)
        probs = [runner.predict(w) for w in windows]
        smoothed, fire_count = detection_events(
            probs, threshold, sliding_window_size,
            feature_step_ms * runner.n_slices,
        )
    except Exception as e:
        return EvalResult(
            task_id=task.id, label=task.label, bucket_id=task.bucket_id,
            fired=False, expected_fire=(task.expected == "fire"), passed=False,
            max_probability=0.0, fire_count=0, duration_s=0.0,
            processing_time_s=time.time() - t0,
            error_msg=str(e),
        )
    max_prob = max(smoothed) if smoothed else 0.0
    fired = fire_count > 0
    expected_fire = task.expected == "fire"
    passed = (fired == expected_fire)
    return EvalResult(
        task_id=task.id, label=task.label, bucket_id=task.bucket_id,
        fired=fired, expected_fire=expected_fire, passed=passed,
        max_probability=max_prob, fire_count=fire_count,
        duration_s=len(audio) / 16000.0,
        processing_time_s=time.time() - t0,
    )


def summarize(results: list[EvalResult], threshold: float, project: str,
              model_path: str, bulk_minutes: float | None = None) -> EvalSummary:
    pos = [r for r in results if r.label == "positive"]
    hn = [r for r in results if r.label == "hard_negative"]
    bulk = [r for r in results if r.label == "bulk_negative"]

    frr = 0.0
    if pos:
        n_missed = sum(1 for r in pos if not r.fired)
        frr = n_missed / len(pos)

    # FAR/hour from bulk: each bulk task is treated as a small slice; sum fire counts
    # and divide by total bulk audio duration in hours.
    bulk_fires = sum(r.fire_count for r in bulk if not r.error_msg)
    if bulk_minutes is None:
        bulk_minutes = sum(r.duration_s for r in bulk if not r.error_msg) / 60.0
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
    ap.add_argument("--sliding-window-size", type=int, default=5)
    ap.add_argument("--feature-step-ms", type=int, default=10)
    ap.add_argument("--bulk-minutes", type=float, default=None,
                    help="Override measured bulk audio duration (normally auto-calculated).")
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
        r = run_one(runner, task, args.threshold, args.sliding_window_size,
                    args.feature_step_ms)
        results.append(r)

    summary = summarize(results, args.threshold, args.project,
                        args.model, args.bulk_minutes)
    print()
    print(f"  FRR:          {summary.frr:.2%}")
    print(f"  FAR/hour:     {summary.far_per_hour:.2f}")
    print("  per-bucket FAR:")
    for bid, val in sorted(summary.per_bucket_far.items()):
        print(f"    {bid}: {val:.2%}")

    out_path = Path(args.out) if args.out else (
        Path("eval/results") / f"{args.project}-v0__{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
