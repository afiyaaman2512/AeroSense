"""
Time-frequency representation: waveform -> log-mel spectrogram.

Kept as a single reusable function so the exact same transform is
guaranteed at training time, validation time, and ICBHI external-test
time — a mismatch here is a classic silent cause of "generalizes badly."
"""
from dataclasses import dataclass

import numpy as np
import librosa

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS


@dataclass
class SpectrogramConfig:
    sample_rate: int = SAMPLE_RATE
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    n_mels: int = N_MELS
    fmin: float = 20.0
    fmax: float | None = None      # None -> sr/2 (Nyquist)


def audio_to_melspectrogram(y: np.ndarray, cfg: SpectrogramConfig = None) -> np.ndarray:
    """Waveform -> log-scaled mel spectrogram, shape (n_mels, n_frames)."""
    cfg = cfg or SpectrogramConfig()
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def normalize_spectrogram(spec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-sample standardization (zero mean, unit variance). Computed per
    clip rather than with a global dataset statistic so the function has
    no dependency on train-set-only values leaking scale info."""
    mean, std = spec.mean(), spec.std()
    return (spec - mean) / (std + eps)


def spectrogram_shape(cfg: SpectrogramConfig, clip_seconds: float) -> tuple[int, int]:
    """Expected (n_mels, n_frames) for a fixed clip length — used to set
    the CNN's input shape without guessing."""
    n_samples = int(round(cfg.sample_rate * clip_seconds))
    n_frames = 1 + n_samples // cfg.hop_length
    return cfg.n_mels, n_frames


def waveform_to_model_input(y: np.ndarray) -> np.ndarray:
    """Convenience wrapper: raw waveform -> normalized log-mel spectrogram,
    ready to be turned into a tensor."""
    spec = audio_to_melspectrogram(y)
    return normalize_spectrogram(spec)
