from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_esphome_template_references_manifest_not_nonexistent_url_option():
    template = (ROOT / "configs/templates/esphome_template.yaml").read_text(
        encoding="utf-8"
    )

    assert 'model: "{{manifest_url}}"' in template
    assert "      url:" not in template
    assert "ota:\n  - platform: esphome" in template


def test_training_stack_is_pinned_and_shared_with_lambda():
    installer = (ROOT / "scripts/install_training_stack.sh").read_text(
        encoding="utf-8"
    )
    lambda_setup = (ROOT / "scripts/_lambda_setup.sh").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "4665173cd35f1cff9a61e06fc427f124766c488e" in installer
    assert "PIPER_SAMPLE_GENERATOR_REF:-v3.0.0" in installer
    assert "bash scripts/install_training_stack.sh" in lambda_setup
    assert "bash /tmp/install_training_stack.sh" in dockerfile


def test_lambda_does_not_publish_without_held_out_eval_tasks():
    setup = (ROOT / "scripts/_lambda_setup.sh").read_text(encoding="utf-8")

    eval_guard = setup.index("if find \"eval/tasks/${PROJECT}\"")
    manifest = setup.index('write_stage "emit_manifest"', eval_guard)
    no_eval = setup.index("manifest and upload skipped", manifest)
    assert eval_guard < manifest < no_eval


def test_requirements_match_pinned_microwakeword_numpy_floor():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "tensorflow>=2.18,<2.22" in requirements
    assert "numpy>=2.0,<2.1" in requirements
