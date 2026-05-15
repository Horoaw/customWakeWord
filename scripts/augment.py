"""Audio augmentation chain used by build_features.py.

Centralised so the same `Compose` is shared between training and (optionally)
eval-time stress-tests.

Built on top of `audiomentations` (CPU). For GPU-batched augmentation,
swap in `torch-audiomentations` — same API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def build_aug_chain(
    aug_cfg: dict,
    rir_paths: Sequence[Path] | None = None,
    noise_paths: Sequence[Path] | None = None,
    sample_rate: int = 16000,
    label: str = "positive",
):
    """Return an `audiomentations.Compose` configured per `aug_cfg`.

    `label` is "positive" / "hard_negative" / "bulk_negative" — only used to
    pick the appropriate `add_background_noise` p value.
    """
    from audiomentations import (
        Compose, AddBackgroundNoise, ApplyImpulseResponse, RoomSimulator,
        PitchShift, TimeStretch, BandPassFilter, Mp3Compression, Gain,
    )

    transforms = []

    rir_p = aug_cfg["apply_impulse_response"]["p"]
    room_p = aug_cfg["room_simulator"]["p"]
    if rir_paths and rir_p > 0:
        transforms.append(ApplyImpulseResponse(ir_path=[str(p) for p in rir_paths], p=rir_p))
    if room_p > 0:
        transforms.append(RoomSimulator(p=room_p))

    bg = aug_cfg["add_background_noise"]
    p_bg = bg["p_positives"] if label == "positive" else bg["p_negatives"]
    if noise_paths and p_bg > 0:
        transforms.append(AddBackgroundNoise(
            sounds_path=[str(p) for p in noise_paths],
            min_snr_db=bg["snr_db_min"],
            max_snr_db=bg["snr_db_max"],
            p=p_bg,
        ))

    ps = aug_cfg["pitch_shift"]
    if ps["p"] > 0:
        transforms.append(PitchShift(min_semitones=ps["semitones_min"],
                                     max_semitones=ps["semitones_max"], p=ps["p"]))

    ts = aug_cfg["time_stretch"]
    if ts["p"] > 0:
        transforms.append(TimeStretch(min_rate=ts["rate_min"],
                                      max_rate=ts["rate_max"],
                                      p=ts["p"]))

    bp = aug_cfg["bandpass_filter"]
    if bp["p"] > 0:
        transforms.append(BandPassFilter(min_center_freq=bp["low_hz"],
                                         max_center_freq=bp["high_hz"],
                                         p=bp["p"]))

    mp3 = aug_cfg["mp3_compression"]
    if mp3["p"] > 0:
        transforms.append(Mp3Compression(min_bitrate=mp3["bitrate_min"],
                                         max_bitrate=mp3["bitrate_max"],
                                         p=mp3["p"]))

    gain = aug_cfg["gain"]
    if gain["p"] > 0:
        transforms.append(Gain(min_gain_db=gain["gain_db_min"],
                               max_gain_db=gain["gain_db_max"],
                               p=gain["p"]))
    return Compose(transforms=transforms, shuffle=False)


def compute_log_mel(
    audio: np.ndarray, sr: int, n_mels: int = 40,
    win_ms: int = 60, hop_ms: int = 25, fmin: int = 20, fmax: int = 7600,
) -> np.ndarray:
    """40-bin log-mel spectrogram, matching microWakeWord's expected input shape."""
    import librosa
    win_length = int(sr * win_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    n_fft = 1 << (win_length - 1).bit_length()  # next pow of 2
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def quantize_int8(features: np.ndarray, scale: float = 1.0, zero_point: int = 0) -> np.ndarray:
    """Map float32 features → int8 to match tflite-micro runtime expectations."""
    q = np.round(features / scale + zero_point).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)
