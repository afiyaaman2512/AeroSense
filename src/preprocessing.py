"""
Audio cleaning & standardization.

Everything here is dataset-agnostic: it works on a path to a .wav/.ogg/.webm
file and returns a clean numpy waveform. Dataset-specific label handling
lives elsewhere (built during Phase 3, once exploration finalizes labels).

Design rule: this module must never look at test-set (ICBHI) data to decide
its own parameters — every parameter is fixed in config.py up front.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

from config import SAMPLE_RATE, CLIP_SECONDS


@dataclass
class PreprocessConfig:
    sample_rate: int = SAMPLE_RATE
    clip_seconds: float = CLIP_SECONDS
    top_db_trim: float = 30.0          # silence-trim threshold
    normalize: bool = True             # peak-normalize amplitude


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio, force mono, resample to `sr`. Returns float32 waveform."""
    y, orig_sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:                      # stereo -> mono
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y


def trim_silence(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Trim leading/trailing near-silence. Falls back to original signal
    if trimming would remove everything (e.g. very quiet recordings)."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed if trimmed.size > 0 else y


def peak_normalize(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Scale so the max absolute amplitude is 1.0. Silent clips pass through
    unchanged rather than dividing by zero."""
    peak = np.max(np.abs(y))
    return y / (peak + eps) if peak > eps else y


def fix_length(y: np.ndarray, sr: int, clip_seconds: float) -> np.ndarray:
    """Force a waveform to an exact length in seconds: center-pad short
    clips with zeros, center-crop long clips. Guarantees every input to
    the spectrogram stage has an identical shape."""
    target_len = int(round(sr * clip_seconds))
    if len(y) == target_len:
        return y
    if len(y) < target_len:
        pad_total = target_len - len(y)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(y, (pad_left, pad_right), mode="constant")
    # longer than target: center crop
    start = (len(y) - target_len) // 2
    return y[start : start + target_len]


def segment_by_annotation(
    y: np.ndarray, sr: int, start_s: float, end_s: float
) -> np.ndarray:
    """Cut one respiratory cycle out of a longer recording using ICBHI-style
    (start_time, end_time) annotations, in seconds."""
    start_idx = max(0, int(round(start_s * sr)))
    end_idx = min(len(y), int(round(end_s * sr)))
    return y[start_idx:end_idx]


def preprocess_file(
    path: str | Path,
    cfg: Optional[PreprocessConfig] = None,
    segment: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """
    Full standardization pipeline for one file:
    load -> mono/resample -> (optional annotation-based segment)
    -> trim silence -> normalize -> fix length.

    `segment=(start_s, end_s)` is used for ICBHI cycle-level cuts;
    leave it None for whole-clip datasets like COUGHVID.
    """
    cfg = cfg or PreprocessConfig()
    y = load_audio(path, sr=cfg.sample_rate)

    if segment is not None:
        y = segment_by_annotation(y, cfg.sample_rate, *segment)

    y = trim_silence(y, top_db=cfg.top_db_trim)

    if cfg.normalize:
        y = peak_normalize(y)

    y = fix_length(y, cfg.sample_rate, cfg.clip_seconds)
    return y


def is_usable(y: np.ndarray, min_rms: float = 1e-4) -> bool:
    """Flags corrupted/empty/near-silent clips so they can be excluded
    before training rather than silently poisoning a batch."""
    if y is None or len(y) == 0 or not np.isfinite(y).all():
        return False
    rms = np.sqrt(np.mean(y**2))
    return rms >= min_rms
