"""
Baseline CNN for log-mel spectrogram classification.

Deliberately simple (Phase completion rule: understand this before
reaching for transfer learning in a later phase). Input shape is derived
from spectrogram.spectrogram_shape() so it always matches whatever
SAMPLE_RATE / N_MELS / CLIP_SECONDS are set in config.py — never hardcode
the shape in two places.
"""
import torch
import torch.nn as nn

from config import N_MELS, CLIP_SECONDS
from spectrogram import SpectrogramConfig, spectrogram_shape


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    4 conv blocks -> global average pool -> dropout -> linear classifier.
    Global average pooling (instead of flatten+big-linear) keeps the model
    small and makes Grad-CAM straightforward in Phase 6 (the last conv
    layer's feature maps map directly onto the spectrogram's time/freq axes).
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128, pool=False),  # keep this layer's spatial size
                                              # larger — it's the Grad-CAM target
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """The conv layer Grad-CAM (Phase 6) should hook into."""
        return self.features[-1].block[0]


def build_model(n_classes: int = 2) -> BaselineCNN:
    return BaselineCNN(n_classes=n_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick shape sanity check — run with: python src/model.py
    n_mels, n_frames = spectrogram_shape(SpectrogramConfig(), CLIP_SECONDS)
    model = build_model(n_classes=2)
    dummy = torch.randn(4, 1, n_mels, n_frames)
    out = model(dummy)
    print(f"input shape:  {tuple(dummy.shape)}")
    print(f"output shape: {tuple(out.shape)}  (expected: (4, 2))")
    print(f"trainable parameters: {count_parameters(model):,}")
