import json

from scripts.emit_manifest import pick_threshold_from_eval


def test_single_point_eval_threshold_is_reused(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"summary": {"threshold": 0.91}}), encoding="utf-8")

    assert pick_threshold_from_eval(path) == 0.91


def test_roc_threshold_respects_recall_and_far(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({
        "summary": {
            "roc": [
                {"threshold": 0.7, "recall": 0.99, "far_per_hour": 2.0},
                {"threshold": 0.8, "recall": 0.97, "far_per_hour": 0.4},
                {"threshold": 0.9, "recall": 0.94, "far_per_hour": 0.1},
            ]
        }
    }), encoding="utf-8")

    assert pick_threshold_from_eval(path) == 0.8
