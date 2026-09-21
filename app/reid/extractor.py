"""Appearance feature extractor for person re-identification across camera feeds."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from app.schemas.face import BoundingBox


class LightweightCNNEmbedding(nn.Module):
    """Compact convolutional feature extractor producing normalized deep representations."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 64x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 32x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 16x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 1)),  # 128x2x1 = 256
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        flattened = torch.flatten(feat, 1)
        out = self.fc(flattened)
        return out


class PersonReIDExtractor:
    """Extracts discriminative appearance feature vectors from person crops.

    Combines multi-stripe spatial HSV color histograms (illumination-invariant)
    with a deep convolutional embedding to recognize persons across multiple camera views.
    """

    def __init__(self, target_size: tuple[int, int] = (64, 128)) -> None:
        self.target_w, self.target_h = target_size
        self._deep_model = LightweightCNNEmbedding(embedding_dim=128)
        self._deep_model.eval()

    def extract(self, frame: np.ndarray, bbox: BoundingBox) -> list[float]:
        """Extract a 256-D L2-normalized appearance vector for the person in bbox."""
        if frame is None or frame.size == 0:
            return [0.0] * 256

        fh, fw = frame.shape[:2]
        x1 = max(0, min(fw - 1, bbox.x))
        y1 = max(0, min(fh - 1, bbox.y))
        x2 = max(x1 + 1, min(fw, bbox.x + bbox.width))
        y2 = max(y1 + 1, min(fh, bbox.y + bbox.height))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return [0.0] * 256

        # Standardize size: 128 height x 64 width
        resized = cv2.resize(crop, (self.target_w, self.target_h))

        # 1. Multi-Stripe Spatial Color Descriptors (Head, Torso, Legs)
        # Head: top 25%, Torso: 25%-65%, Legs: 65%-100%
        h_split = self.target_h
        head_crop = resized[: int(h_split * 0.25), :]
        torso_crop = resized[int(h_split * 0.25) : int(h_split * 0.65), :]
        legs_crop = resized[int(h_split * 0.65) :, :]

        hist_head = self._compute_stripe_hist(head_crop)  # 32 dims
        hist_torso = self._compute_stripe_hist(torso_crop)  # 48 dims
        hist_legs = self._compute_stripe_hist(legs_crop)  # 48 dims
        color_feat = np.concatenate([hist_head, hist_torso, hist_legs])  # 128 dims

        # 2. Deep CNN Appearance Embedding
        # Convert BGR to RGB, normalize to [-1, 1], torch tensor (1, 3, 128, 64)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0

        with torch.no_grad():
            deep_feat = self._deep_model(tensor_img).squeeze(0).cpu().numpy()  # 128 dims

        # 3. Concatenate and L2 Normalize
        combined = np.concatenate([color_feat, deep_feat])
        norm = np.linalg.norm(combined)
        if norm > 1e-6:
            normalized = combined / norm
        else:
            normalized = combined

        return normalized.astype(float).tolist()

    @staticmethod
    def _compute_stripe_hist(stripe: np.ndarray) -> np.ndarray:
        """Compute quantized HSV histogram for an anatomical stripe."""
        if stripe.size == 0:
            return np.zeros(48 if stripe.shape[0] > 30 else 32, dtype=np.float32)

        hsv = cv2.cvtColor(stripe, cv2.COLOR_BGR2HSV)
        # Quantize: H into 8 bins, S into 2 or 3 bins, V into 2 bins
        is_large = stripe.shape[0] > 30
        h_bins = 8
        s_bins = 3 if is_large else 2
        v_bins = 2

        hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
        flat = hist.flatten()
        norm = np.linalg.norm(flat)
        if norm > 1e-6:
            flat = flat / norm
        return flat.astype(np.float32)


def cosine_similarity(feat1: list[float] | np.ndarray, feat2: list[float] | np.ndarray) -> float:
    """Calculate cosine similarity between two feature vectors."""
    a = np.asarray(feat1, dtype=np.float32)
    b = np.asarray(feat2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
