"""
Robot Savo — Audio utilities for STT gateway.

This module provides helper functions to:

- Decode arbitrary audio bytes (WAV/OGG/etc.) using soundfile.
- Convert audio to mono.
- Resample to 16 kHz, which is what faster-whisper expects.

We keep this module small and dependency-light: only numpy + soundfile.
"""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import soundfile as sf


TARGET_SR = 16_000


def _resample_linear(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Simple linear-resampling implementation using NumPy only.

    This is not the fanciest resampler in the world, but it is:
      - deterministic
      - good enough for STT on short utterances
      - avoids pulling in heavy dependencies (librosa, resampy, etc.)
    """
    if orig_sr == target_sr:
        return samples

    if samples.size == 0:
        return samples

    # Compute new length with rounding.
    duration_sec = samples.shape[0] / float(orig_sr)
    new_length = max(1, int(round(duration_sec * target_sr)))

    # Original and target sample positions.
    orig_positions = np.linspace(0.0, 1.0, num=samples.shape[0], endpoint=False)
    new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)

    # Linear interpolation.
    resampled = np.interp(new_positions, orig_positions, samples).astype(np.float32)
    return resampled


def decode_audio_to_mono_16k(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode raw audio bytes to mono, 16 kHz float32 waveform.

    Returns:
      samples: np.ndarray of shape (N,), dtype=float32
      sample_rate: int (always 16_000)

    Raises:
      RuntimeError on decode failures or empty output.
    """
    if not audio_bytes:
        raise RuntimeError("Empty audio payload; cannot decode")

    try:
        with io.BytesIO(audio_bytes) as buf:
            # soundfile returns:
            #   - data: np.ndarray, shape (n_samples,) or (n_samples, n_channels)
            #   - sr: sample rate (int)
            data, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to decode audio: {exc}") from exc

    # Ensure we have at least 1D array.
    samples = np.asarray(data, dtype=np.float32)

    if samples.ndim == 2:
        # (n_samples, n_channels) → mono by averaging channels
        samples = samples.mean(axis=1)

    if samples.size == 0:
        raise RuntimeError("Decoded audio is empty")

    # Resample to TARGET_SR if needed.
    if sr != TARGET_SR:
        samples = _resample_linear(samples, sr, TARGET_SR)
        sr = TARGET_SR

    return samples, sr
