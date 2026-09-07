"""Detector abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RawFaceDetection:
    """Detector-native face candidate before quality and tracking."""

    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: list[list[float]] | None = None


class FaceDetector(ABC):
    """Interface implemented by all face detectors."""

    name: str = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[RawFaceDetection]:
        """Return face candidates for a BGR image frame."""


class DetectorError(RuntimeError):
    """Raised when detection cannot be performed safely."""


class ModelLoadError(DetectorError):
    """Raised when a model cannot be loaded."""

