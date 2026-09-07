from __future__ import annotations

import cv2
import numpy as np

from app.quality.blur import laplacian_variance
from app.quality.brightness import average_brightness
from app.quality.quality_score import FaceQualityAssessor, QualityConfig
from app.quality.size import meets_minimum_size
from app.schemas.face import BoundingBox


def test_blur_calculation_responds_to_sharpness() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (60, 60), (255, 255, 255), -1)
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    assert laplacian_variance(image) > laplacian_variance(blurred)


def test_brightness_calculation() -> None:
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    assert average_brightness(image) == 128


def test_minimum_size_check() -> None:
    assert meets_minimum_size(BoundingBox(x=0, y=0, width=30, height=30), 30, 30)
    assert not meets_minimum_size(BoundingBox(x=0, y=0, width=29, height=30), 30, 30)


def test_quality_score_stays_in_range() -> None:
    assessor = FaceQualityAssessor(
        QualityConfig(
            blur_threshold=100,
            brightness_min=50,
            brightness_max=210,
            size_reference_area=1000,
            size_weight=0.4,
            blur_weight=0.35,
            brightness_weight=0.25,
        )
    )
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    quality = assessor.assess(frame, BoundingBox(x=10, y=10, width=40, height=40))
    assert 0.0 <= quality.score <= 1.0

