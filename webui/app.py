"""Browser-side test harness for a customWakeWord TFLite Micro model.

Gradio app that loads a project's `.tflite` from the HF Hub, takes audio
either from the mic or as a WAV upload, runs the same streaming
microwakeword inference path the on-device ESP32-S3 uses, and reports
fire/no-fire + a per-chunk probability trace.

Designed to drop into a Hugging Face Space — `app.py` + `requirements.txt`
in the Space repo is all it needs. Locally: `python webui/app.py`.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly
from scipy.io import wavfile


# Default project / HF repo. Override in a Space by setting the env vars
# or editing the dropdown below.
DEFAULT_HF_REPO = os.environ.get("WAKE_HF_REPO", "nagisanzeninz/tofu-wakeword-v0")
DEFAULT_PROJECT = os.environ.get("WAKE_PROJECT", "tofu")
DEFAULT_THRESHOLD = float(os.environ.get("WAKE_THRESHOLD", "0.85"))

# microwakeword feature constants — these track the upstream defaults used
# at training time (`microwakeword.audio.audio_utils.generate_features_for_clip`).
TARGET_SR = 16000
STEP_MS = 10


_MODEL_CACHE: dict[str, tuple[Interpreter, dict, float]] = {}


def load_model(hf_repo: str) -> tuple[Interpreter, dict, float]:
    """Return (interpreter, manifest, probability_cutoff) cached per-repo."""
    if hf_repo in _MODEL_CACHE:
        return _MODEL_CACHE[hf_repo]

    # The training pipeline uploads the .tflite under a project-specific
    # name (e.g. `tofu-wakeword-v0.tflite`), so auto-discover by listing the
    # repo and grabbing the first .tflite found. ESPHome v2 manifest may
    # land as `manifest.json` (preferred) or be embedded in `esphome.yaml`;
    # if neither is present we fall back to DEFAULT_THRESHOLD.
    from huggingface_hub import HfApi
    api = HfApi()
    siblings = api.list_repo_files(repo_id=hf_repo, repo_type="model")
    tflite_name = next((f for f in siblings if f.endswith(".tflite")), None)
    if not tflite_name:
        raise RuntimeError(
            f"No .tflite in repo {hf_repo}. "
            f"Files seen: {siblings}"
        )
    tflite_path = hf_hub_download(repo_id=hf_repo, filename=tflite_name)

    manifest, cutoff = {}, DEFAULT_THRESHOLD
    if "manifest.json" in siblings:
        mf_path = hf_hub_download(repo_id=hf_repo, filename="manifest.json")
        manifest = json.loads(Path(mf_path).read_text())
        cutoff = float(manifest.get("micro", {}).get("probability_cutoff",
                                                     DEFAULT_THRESHOLD))

    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    _MODEL_CACHE[hf_repo] = (interp, manifest, cutoff)
    return interp, manifest, cutoff


def to_mono_16k(sample_rate: int, audio: np.ndarray) -> np.ndarray:
    """Mono 16 kHz int16 from any input shape (mic gives float32 [-1, 1])."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SR:
        # resample_poly handles non-integer ratios cleanly via up/down
        from math import gcd
        g = gcd(sample_rate, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sample_rate // g)
    if audio.dtype != np.int16:
        # mic returns float32 in [-1, 1]; convert to int16 PCM that
        # generate_features_for_clip expects.
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)
    return audio


def spectrogram_from_pcm(pcm16: np.ndarray) -> np.ndarray:
    """Mel spectrogram matching microwakeword's training-time features.

    Vendored copy of `microwakeword.audio.audio_utils.generate_features_for_clip`
    (use_c=True branch only) so the HF Space requirements stay light —
    we don't need TensorFlow client-side, just the C MicroFrontend
    (`pymicro-features` wheel, ~100 kB) that ships with the same audio
    frontend the tflite-micro runtime uses on-device. Apache 2.0,
    Kevin Ahrendt.
    """
    from pymicro_features import MicroFrontend

    audio = pcm16.tobytes()
    frontend = MicroFrontend()
    features = []
    idx = 0
    n = len(audio)
    while idx + 160 * 2 < n:
        out = frontend.process_samples(audio[idx: idx + 160 * 2])
        idx += out.samples_read * 2
        if out.features:
            features.append(out.features)
    return np.array(features).astype(np.float32)


def run_model(interpreter: Interpreter, spectrogram: np.ndarray) -> list[float]:
    """Slide the model's input window across the spectrogram, return per-step probs."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_dtype = input_details[0]["dtype"]
    is_quant = in_dtype == np.int8
    n_slices = int(input_details[0]["shape"][1])
    in_scale, in_zp = (input_details[0]["quantization_parameters"]["scales"][0],
                       input_details[0]["quantization_parameters"]["zero_points"][0]) if is_quant else (1.0, 0)
    out_scale, out_zp = (output_details[0]["quantization_parameters"]["scales"][0],
                         output_details[0]["quantization_parameters"]["zero_points"][0]) if is_quant else (1.0, 0)

    # Scale uint16-encoded spectrograms to float32 like upstream does.
    spec = spectrogram
    if np.issubdtype(spec.dtype, np.uint16):
        spec = spec.astype(np.float32) * 0.0390625

    probs = []
    stride = n_slices  # non-overlapping windows for a quick UI; tweak if needed
    for end in range(n_slices, len(spec) + 1, stride):
        chunk = spec[end - n_slices: end]
        if is_quant:
            q = np.clip(chunk / in_scale + in_zp, -128, 127).astype(np.int8)
            interpreter.set_tensor(input_details[0]["index"],
                                   q.reshape(input_details[0]["shape"]))
        else:
            interpreter.set_tensor(input_details[0]["index"],
                                   chunk.astype(np.float32).reshape(input_details[0]["shape"]))
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details[0]["index"])[0][0]
        prob = float((raw - out_zp) * out_scale) if is_quant else float(raw)
        probs.append(prob)

    return probs


def predict(hf_repo: str, audio) -> tuple[str, float, "plt.Figure | None"]:
    """Gradio handler. `audio` is (sample_rate, np.ndarray) from gr.Audio."""
    if audio is None:
        return "↑ Record or upload some audio first.", 0.0, None
    sample_rate, raw = audio
    pcm = to_mono_16k(int(sample_rate), raw)
    if len(pcm) < TARGET_SR // 2:
        return "Clip is too short — need at least ~0.5 s.", 0.0, None

    interp, manifest, cutoff = load_model(hf_repo)
    spec = spectrogram_from_pcm(pcm)
    probs = run_model(interp, spec)
    if not probs:
        return "Clip is shorter than the model's window. Speak for at least 1 s.", 0.0, None

    max_p = max(probs)
    fired = max_p >= cutoff
    label = (f"🎯 **WAKE WORD DETECTED** (peak prob {max_p:.3f} ≥ cutoff {cutoff:.2f})"
             if fired else
             f"😴 No wake word (peak prob {max_p:.3f} < cutoff {cutoff:.2f})")

    fig, ax = plt.subplots(figsize=(6, 2.2))
    t = np.arange(len(probs)) * (STEP_MS * spec.shape[0] / max(len(probs), 1)) / 1000.0
    ax.plot(t, probs, marker="o", markersize=3)
    ax.axhline(cutoff, color="red", linestyle="--", linewidth=1, label=f"cutoff = {cutoff:.2f}")
    ax.set_xlabel("approx. window-end (s)")
    ax.set_ylabel("p(wake)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return label, max_p, fig


with gr.Blocks(title="customWakeWord — live tester") as demo:
    gr.Markdown(f"""
# customWakeWord — live tester

Records audio in your browser (or upload a WAV) and runs the **same
streaming INT8 inference path the ESP32-S3 firmware uses**, returning
fire/no-fire + a per-chunk probability trace.

**Default model:** `{DEFAULT_HF_REPO}` — change to test any
HF-published customWakeWord build.
""")
    with gr.Row():
        repo = gr.Textbox(value=DEFAULT_HF_REPO, label="Hugging Face repo id",
                          info="anyone/yourwake-v0 — must have model.tflite + manifest.json")
    with gr.Tab("🎤 Mic"):
        mic = gr.Audio(sources=["microphone"], type="numpy",
                       label="Say the wake phrase (1–3 s)")
        mic_btn = gr.Button("Run", variant="primary")
    with gr.Tab("📁 Upload WAV"):
        wav = gr.Audio(sources=["upload"], type="numpy",
                       label="Upload a 16 kHz mono WAV (other formats are resampled)")
        wav_btn = gr.Button("Run", variant="primary")

    verdict = gr.Markdown()
    prob = gr.Number(label="Peak probability (0–1)", precision=4)
    trace = gr.Plot(label="Per-window probability")

    mic_btn.click(predict, inputs=[repo, mic], outputs=[verdict, prob, trace])
    wav_btn.click(predict, inputs=[repo, wav], outputs=[verdict, prob, trace])

    gr.Markdown("""
---
### Notes
- **Threshold** comes from the model's `manifest.json` (`micro.probability_cutoff`). The published default for `tofu-wakeword-v0` is tuned for ESP32 — increase it if you get false fires.
- **Mic capture** in HF Spaces gets routed through HTTPS; first run will ask for browser mic permission.
- **The model is INT8**: input is quantized to int8, output is dequantized to float per the TFLite quantization parameters. Same arithmetic the on-device runtime uses.
- The full training pipeline is at <https://github.com/temm1e-labs/customWakeWord>.
""")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
