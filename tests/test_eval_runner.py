import pytest

from eval.runner import detection_events, summarize
from eval.schema import EvalResult


def result(label: str, *, fired: bool, fire_count: int, duration_s: float) -> EvalResult:
    return EvalResult(
        task_id=f"{label}-{duration_s}",
        label=label,
        bucket_id="collision" if label == "hard_negative" else None,
        fired=fired,
        expected_fire=label == "positive",
        passed=(fired == (label == "positive")),
        max_probability=0.9 if fired else 0.1,
        fire_count=fire_count,
        duration_s=duration_s,
    )


def test_detection_uses_full_sliding_window_and_counts_transitions():
    smoothed, events = detection_events(
        [0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9],
        threshold=0.8,
        sliding_window_size=3,
        step_ms=30,
    )

    assert smoothed[1] == pytest.approx(0.9)
    assert events == 2


def test_far_uses_measured_bulk_audio_duration():
    results = [
        result("positive", fired=True, fire_count=1, duration_s=1.5),
        result("bulk_negative", fired=True, fire_count=1, duration_s=1800),
        result("bulk_negative", fired=False, fire_count=0, duration_s=1800),
    ]

    summary = summarize(results, 0.85, "xingxing", "model.tflite")

    assert summary.frr == 0
    assert summary.bulk_stream_minutes == pytest.approx(60)
    assert summary.far_per_hour == pytest.approx(1)
