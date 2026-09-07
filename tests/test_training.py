from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from app.detection.custom_detector import CustomTinyFaceDetector
from app.training.dataset import FaceDataset, generate_synthetic_dataset
from app.training.model import TinyFaceDetector


def test_tiny_face_detector_forward_shapes() -> None:
    model = TinyFaceDetector(input_size=128)
    dummy = torch.randn(2, 3, 128, 128)
    logits, bboxes, landmarks = model(dummy)

    assert logits.shape == (2, 1)
    assert bboxes.shape == (2, 4)
    assert landmarks.shape == (2, 10)


def test_synthetic_dataset_generation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        samples = generate_synthetic_dataset(tmpdir, num_samples=10)
        assert len(samples) == 10
        manifest = Path(tmpdir) / "annotations.json"
        assert manifest.exists()

        dataset = FaceDataset(samples, image_size=128)
        assert len(dataset) == 10
        img, label, bbox, lm = dataset[0]
        assert img.shape == (3, 128, 128)
        assert label.shape == (1,)
        assert bbox.shape == (4,)
        assert lm.shape == (10,)


def test_custom_face_detector_detect() -> None:
    model_path = Path("models/custom_face_detector.onnx")
    if not model_path.exists():
        return

    detector = CustomTinyFaceDetector(model_path=model_path, confidence_threshold=0.01)
    dummy_frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    results = detector.detect(dummy_frame)
    assert isinstance(results, list)
    if results:
        det = results[0]
        assert len(det.bbox) == 4
        assert 0.0 <= det.confidence <= 1.0
        assert det.landmarks is not None
        assert len(det.landmarks) == 5
