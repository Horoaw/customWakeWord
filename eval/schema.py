"""Eval schema: task definition + result rows.

Tasks are stored as one JSON file per WAV clip under
`eval/tasks/<project>/{positives,hard_negatives,bulk}/`. Results are
aggregated into a single `eval/results/<project>-v0__<ts>.json` per run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalTask:
    id: str
    audio_path: str
    label: str                                 # "positive" | "hard_negative" | "bulk_negative"
    expected: str                              # "fire" | "no_fire"
    phrase: Optional[str] = None
    bucket_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    task_id: str
    label: str
    bucket_id: Optional[str]
    fired: bool
    expected_fire: bool
    passed: bool
    max_probability: float
    fire_count: int                            # number of windows above threshold
    duration_s: float
    error_msg: Optional[str] = None


@dataclass
class EvalSummary:
    project: str
    model_path: str
    threshold: float
    n_positives: int
    n_hard_negatives: int
    bulk_stream_minutes: float
    frr: float                                 # false reject rate on positives
    far_per_hour: float                        # bulk false-accept rate per hour
    per_bucket_far: dict[str, float] = field(default_factory=dict)
    roc: list[dict] = field(default_factory=list)  # [{threshold, frr, far_per_hour}]
