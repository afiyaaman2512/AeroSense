"""
PyTorch Dataset that ties the pipeline together:
  file path (+ optional cycle segment) -> preprocess_file -> spectrogram
  -> (train-only) augmentation -> tensor

Expects a manifest as a pandas DataFrame/CSV with columns:
    filepath   : str, path to the audio file
    label      : str or int, class label (finalized in Phase 3 exploration)
    start_s    : float, optional — cycle start time (ICBHI-style); NaN/absent
                 for whole-clip datasets like COUGHVID
    end_s      : float, optional — cycle end time
    split      : one of "train" / "val" / "test"

Building the actual manifest (reading COUGHVID CSV metadata / ICBHI
annotation .txt files and mapping them to a common label set) happens in
Phase 3, once the label schemes have been inspected side by side —
that decision should not be baked into this generic loader.
"""
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing import PreprocessConfig, preprocess_file, is_usable
from spectrogram import SpectrogramConfig, audio_to_melspectrogram, normalize_spectrogram
from augmentation import AugmentConfig, augment_waveform, augment_spectrogram


class AeroSenseDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        label_to_idx: dict,
        split: str,
        preprocess_cfg: Optional[PreprocessConfig] = None,
        spec_cfg: Optional[SpectrogramConfig] = None,
        augment_cfg: Optional[AugmentConfig] = None,
        augment: Optional[bool] = None,
    ):
        assert split in ("train", "val", "test")
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.split = split
        self.preprocess_cfg = preprocess_cfg or PreprocessConfig()
        self.spec_cfg = spec_cfg or SpectrogramConfig()
        self.augment_cfg = augment_cfg or AugmentConfig()
        # Augmentation defaults to train-only (Phase 6 rule): val/test stay
        # representative of real-world data, never artificially augmented.
        self.augment = augment if augment is not None else (split == "train")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        segment = None
        if "start_s" in row and "end_s" in row and pd.notna(row.get("start_s")):
            segment = (float(row["start_s"]), float(row["end_s"]))

        y = preprocess_file(row["filepath"], self.preprocess_cfg, segment=segment)

        if not is_usable(y):
            # Corrupted/near-silent clip slipped through exploration's
            # quality filter — return silence rather than crashing a
            # training run; these should be rare and are logged upstream.
            y = np.zeros(int(self.preprocess_cfg.sample_rate * self.preprocess_cfg.clip_seconds),
                          dtype=np.float32)

        if self.augment:
            y = augment_waveform(y, self.augment_cfg)

        spec = audio_to_melspectrogram(y, self.spec_cfg)
        spec = normalize_spectrogram(spec)

        if self.augment:
            spec = augment_spectrogram(spec, self.augment_cfg)

        spec_tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, n_mels, n_frames) — 1 channel for CNN
        label_idx = self.label_to_idx[row["label"]]
        return spec_tensor, torch.tensor(label_idx, dtype=torch.long)


def build_label_map(labels: Sequence[str]) -> dict:
    """Deterministic label -> index mapping, sorted so it's reproducible
    across runs/machines rather than depending on dict/set ordering."""
    return {label: i for i, label in enumerate(sorted(set(labels)))}
