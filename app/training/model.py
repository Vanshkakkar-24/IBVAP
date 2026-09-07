"""PyTorch deep-learning neural network for custom face detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyFaceDetector(nn.Module):
    """Lightweight CNN for face classification, bounding box regression, and facial landmarks.

    Input: (B, 3, 128, 128) image tensor normalized to [0, 1].
    Outputs:
      - logits: (B, 1) raw logit for face classification (apply sigmoid for probability).
      - bbox: (B, 4) normalized bounding box [x, y, w, h] in [0, 1].
      - landmarks: (B, 10) normalized coordinates for 5 landmarks (left eye, right eye, nose, mouth left, mouth right).
    """

    def __init__(self, input_size: int = 128) -> None:
        super().__init__()
        self.input_size = input_size
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 128 -> 64
            ConvBlock(32, 64),   # 64 -> 32
            ConvBlock(64, 128),  # 32 -> 16
            ConvBlock(128, 256), # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_shared = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Output heads
        self.cls_head = nn.Linear(128, 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )
        self.landmark_head = nn.Sequential(
            nn.Linear(128, 10),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)
        shared = self.fc_shared(feat)

        logits = self.cls_head(shared)
        bbox = self.bbox_head(shared)
        landmarks = self.landmark_head(shared)

        return logits, bbox, landmarks
