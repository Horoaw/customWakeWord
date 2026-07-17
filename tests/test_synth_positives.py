import sys
import types
import wave

from scripts.synth_positives import (
    build_generator_command,
    generate_standard_piper_samples,
)


def test_v3_command_uses_english_generator_model(tmp_path):
    psg_dir = tmp_path / "piper-sample-generator"
    psg_dir.mkdir()
    model = tmp_path / "generator.pt"

    command, cwd = build_generator_command(
        psg_dir, "hey sunny", tmp_path / "audio", 100, 10,
        [model], 904, [0.9, 1.1],
    )

    assert command[1] == str((psg_dir / "generate_samples.py").resolve())
    assert "hey sunny" in command
    assert command[command.index("--model") + 1] == str(model.resolve())
    assert command[-3:] == ["--length-scales", "0.9", "1.1"]
    assert command[command.index("--max-speakers") + 1] == "904"
    assert cwd is None


def test_standard_piper_generates_unicode_samples_without_torch(tmp_path, monkeypatch):
    calls = []

    class FakeSynthesisConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeVoice:
        config = types.SimpleNamespace(num_speakers=2)

        @classmethod
        def load(cls, model, use_cuda):
            assert model.endswith(".onnx")
            assert use_cuda is False
            return cls()

        def synthesize_wav(self, text, wav_file, syn_config):
            calls.append((text, syn_config))
            wav_file.setframerate(16_000)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(b"\x00\x00" * 160)

    fake_piper = types.ModuleType("piper")
    fake_piper.PiperVoice = FakeVoice
    fake_piper.SynthesisConfig = FakeSynthesisConfig
    monkeypatch.setitem(sys.modules, "piper", fake_piper)

    model_a = tmp_path / "voice-a.onnx"
    model_b = tmp_path / "voice-b.onnx"
    output_dir = tmp_path / "audio"
    generate_standard_piper_samples(
        "星星", output_dir, 5, [model_a, model_b], [0.9, 1.1], 1
    )

    wavs = sorted(output_dir.glob("*.wav"))
    assert len(wavs) == 5
    assert [text for text, _ in calls] == ["星星"] * 5
    assert all(config.speaker_id == 0 for _, config in calls)
    with wave.open(str(wavs[0]), "rb") as wav_file:
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnframes() == 160
