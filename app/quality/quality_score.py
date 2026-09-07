"""Heuristic face usability scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.quality.blur import laplacian_variance
from app.quality.brightness import average_brightness
from app.schemas.face import BoundingBox, FaceQuality


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class QualityConfig:
    blur_threshold: float
    brightness_min: float
    brightness_max: float
    size_reference_area: int
    size_weight: float
    blur_weight: float
    brightness_weight: float

    @property
    def total_weight(self) -> float:
        total = self.size_weight + self.blur_weight + self.brightness_weight
        return total if total > 0 else 1.0


class FaceQualityAssessor:
    """Computes explainable, heuristic face usability metrics."""

    def __init__(self, config: QualityConfig) -> None:
        self.config = config

    def assess(self, frame: np.ndarray, bbox: BoundingBox) -> FaceQuality:
        x1, y1, x2, y2 = bbox.as_xyxy()
        crop = frame[y1:y2, x1:x2] if frame is not None and frame.size else np.empty((0, 0, 3))

        blur = laplacian_variance(crop)
        brightness = average_brightness(crop)
        area = bbox.area

        size_score = clamp01(area / self.config.size_reference_area)
        blur_score = clamp01(blur / self.config.blur_threshold)
        brightness_score = self._brightness_score(brightness)

        weighted = (
            size_score * self.config.size_weight
            + blur_score * self.config.blur_weight
            + brightness_score * self.config.brightness_weight
        ) / self.config.total_weight

        return FaceQuality(
            score=round(clamp01(weighted), 4),
            blur=round(blur, 4),
            brightness=round(brightness, 4),
            width=bbox.width,
            height=bbox.height,
            area=area,
            size_score=round(size_score, 4),
            blur_score=round(blur_score, 4),
            brightness_score=round(brightness_score, 4),
        )

    def _brightness_score(self, brightness: float) -> float:
        low = self.config.brightness_min
        high = self.config.brightness_max
        midpoint = (low + high) / 2
        half_range = max((high - low) / 2, 1.0)
        distance = abs(brightness - midpoint)
        return clamp01(1.0 - distance / half_range)

