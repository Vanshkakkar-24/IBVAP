from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.detection.base import FaceDetector, ModelLoadError, RawFaceDetection
from app.detection.postprocess import confidence_filter
from app.detection.yunet import YuNetDetector
from app.services.face_service import FaceDetectionService


class FakeDetector(FaceDetector):
    name = "fake"

    def __init__(self, detections: list[RawFaceDetection]) -> None:
        self.detections = detections

    def detect(self, frame: np.ndarray) -> list[RawFaceDetection]:
        if frame is None or frame.size == 0:
            return []
        return self.detections


def test_yunet_missing_model_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError):
        YuNetDetector(tmp_path / "missing.onnx")


def test_valid_image_produces_detections() -> None:
    settings = Settings(MODEL_PATH="models/fake.onnx")
    detector = FakeDetector([RawFaceDetection(bbox=(5, 5, 40, 40), confidence=0.95)])
    service = FaceDetectionService(settings, detector=detector)
    frame = np.full((100, 100, 3), 180, dtype=np.uint8)
    metadata = service.process_frame(frame, frame_id=1)
    assert len(metadata.faces) == 1
    assert metadata.faces[0].confidence == 0.95


def test_empty_scene_produces_empty_list() -> None:
    settings = Settings(MODEL_PATH="models/fake.onnx")
    service = FaceDetectionService(settings, detector=FakeDetector([]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    metadata = service.process_frame(frame, frame_id=1)
    assert metadata.faces == []


def test_confidence_filtering() -> None:
    detections = [
        RawFaceDetection(bbox=(0, 0, 40, 40), confidence=0.90),
        RawFaceDetection(bbox=(0, 0, 40, 40), confidence=0.40),
    ]
    assert len(confidence_filter(detections, 0.65)) == 1


def test_invalid_frame_is_handled_safely() -> None:
    settings = Settings(MODEL_PATH="models/fake.onnx")
    service = FaceDetectionService(settings, detector=FakeDetector([]))
    metadata = service.process_frame(np.array([], dtype=np.uint8), frame_id=2)
    assert metadata.faces == []

