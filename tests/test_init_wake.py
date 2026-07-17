import sys

import yaml

from scripts.init_wake import (
    build_hard_negatives_yaml,
    build_phrases_yaml,
    build_training_parameters,
    detect_language,
    slugify,
    split_wake_word,
    main,
)
from scripts.synth_positives import slug


def test_mandarin_project_is_unicode_safe():
    assert slugify("星星") == "星星"
    assert slug("星星") == "星星"
    assert detect_language(["星星"]) == "zh"
    assert split_wake_word(["星星"]) == "星星"


def test_mandarin_config_uses_standard_piper_voices():
    config = build_phrases_yaml("xingxing", ["星星"], [5000], [1.0], 42, "zh")

    assert config["wake_name"] == "星星"
    assert config["language"] == "zh"
    voices = config["engines"][0]["voices"]
    assert len(voices) >= 2
    assert all(voice["model_url"].endswith(".onnx") for voice in voices)
    assert all(voice["config_url"].endswith(".onnx.json") for voice in voices)


def test_bare_trigger_is_never_generated_as_a_hard_negative():
    config = build_hard_negatives_yaml(
        "xingxing", "星星", 42, "zh", ["星星"]
    )

    negatives = {
        phrase
        for bucket in config["buckets"]
        for phrase in bucket.get("phrases", [])
    }
    assert "星星" not in negatives


def test_training_config_points_at_new_project():
    config = build_training_parameters("xingxing")

    assert config["train_dir"] == "trained_models/xingxing"
    assert config["features"][0]["features_dir"] == "data/xingxing/features"
    assert config["features"][1]["features_dir"] == "data/xingxing/hard_negatives_features"


def test_cli_writes_complete_mandarin_project(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "init_wake.py", "--name", "xingxing", "--phrases", "星星",
        "--language", "zh", "--out-root", str(tmp_path),
    ])

    assert main() == 0
    project = tmp_path / "xingxing"
    assert {path.name for path in project.iterdir()} == {
        "README.md", "hard_negatives.yaml", "training_parameters.yaml",
        "wake_phrases.yaml",
    }
    positives = yaml.safe_load(
        (project / "wake_phrases.yaml").read_text(encoding="utf-8")
    )
    assert positives["phrases"][0]["text"] == "星星"
