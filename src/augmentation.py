"""
Augmentation — applied ONLY to the training split, never to validation
or the frozen ICBHI external test set (see Phase completion checks in
the project roadmap).

Two families:
  - waveform-level: time shift, gain, additive noise (mild, realistic)
  - spectrogram-level: SpecAugment-style time/frequency masking

Deliberately excluded: pitch shifting, aggressive time-stretch, or
mixup-style blending of respiratory sounds — these can produce acoustic
patterns a pulmonologist would never actually hear, which defeats the
point of a clinically-grounded pipeline. SMOTE is not used on raw audio
for the same reason (interpolating waveforms/pixels of a spectrogram
does not correspond to any real physical signal).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class AugmentConfig:
    p_time_shift: float = 0.5
    max_shift_frac: float = 0.15     # fraction of clip length

    p_gain: float = 0.5
    gain_db_range: tuple[float, float] = (-6.0, 6.0)

    p_noise: float = 0.3
    noise_snr_db_range: tuple[float, float] = (15.0, 30.0)

    p_time_mask: float = 0.5
    max_time_mask_frac: float = 0.1  # fraction of frames masked
    n_time_masks: int = 1

    p_freq_mask: float = 0.5
    max_freq_mask_frac: float = 0.1  # fraction of mel bins masked
    n_freq_masks: int = 1


# ------------------------------------------------------------ waveform ----
def time_shift(y: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    if rng.random() > cfg.p_time_shift:
        return y
    max_shift = int(len(y) * cfg.max_shift_frac)
    shift = rng.integers(-max_shift, max_shift + 1)
    return np.roll(y, shift)


def random_gain(y: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    if rng.random() > cfg.p_gain:
        return y
    gain_db = rng.uniform(*cfg.gain_db_range)
    factor = 10.0 ** (gain_db / 20.0)
    return np.clip(y * factor, -1.0, 1.0)


def add_noise(y: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    if rng.random() > cfg.p_noise:
        return y
    snr_db = rng.uniform(*cfg.noise_snr_db_range)
    signal_power = np.mean(y**2) + 1e-12
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(noise_power), size=y.shape).astype(np.float32)
    return y + noise


def augment_waveform(y: np.ndarray, cfg: AugmentConfig, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = time_shift(y, cfg, rng)
    y = random_gain(y, cfg, rng)
    y = add_noise(y, cfg, rng)
    return y


# ---------------------------------------------------------- spectrogram ---
def time_mask(spec: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    if rng.random() > cfg.p_time_mask:
        return spec
    spec = spec.copy()
    n_frames = spec.shape[1]
    max_width = max(1, int(n_frames * cfg.max_time_mask_frac))
    for _ in range(cfg.n_time_masks):
        width = rng.integers(1, max_width + 1)
        start = rng.integers(0, max(1, n_frames - width))
        spec[:, start : start + width] = spec.mean()
    return spec


def freq_mask(spec: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    if rng.random() > cfg.p_freq_mask:
        return spec
    spec = spec.copy()
    n_mels = spec.shape[0]
    max_width = max(1, int(n_mels * cfg.max_freq_mask_frac))
    for _ in range(cfg.n_freq_masks):
        width = rng.integers(1, max_width + 1)
        start = rng.integers(0, max(1, n_mels - width))
        spec[start : start + width, :] = spec.mean()
    return spec


def augment_spectrogram(spec: np.ndarray, cfg: AugmentConfig, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spec = time_mask(spec, cfg, rng)
    spec = freq_mask(spec, cfg, rng)
    return spec


# -------------------------------------------------------- class balance ---
def class_weights_from_counts(counts: dict) -> dict:
    """Inverse-frequency class weights, for use in the loss function as an
    alternative/complement to augmentation when imbalance is severe."""
    total = sum(counts.values())
    n_classes = len(counts)
    return {cls: total / (n_classes * n) for cls, n in counts.items()}
