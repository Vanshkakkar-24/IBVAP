"""Temporary tracking interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.face import FaceDetection


class FaceTracker(ABC):
    """Associates detections across nearby frames without identifying people."""

    @abstractmethod
    def update(self, detections: list[FaceDetection]) -> list[FaceDetection]:
        """Attach temporary track IDs to detections."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all active temporary tracks."""

