"""Common post-processing filters for face detections."""

from __future__ import annotations

from app.detection.base import RawFaceDetection


def confidence_filter(detections: list[RawFaceDetection], threshold: float) -> list[RawFaceDetection]:
    return [face for face in detections if face.confidence >= threshold]


def minimum_size_filter(
    detections: list[RawFaceDetection],
    min_width: int,
    min_height: int,
) -> list[RawFaceDetection]:
    output: list[RawFaceDetection] = []
    for face in detections:
        _, _, width, height = face.bbox
        if width >= min_width and height >= min_height:
            output.append(face)
    return output


def clamp_detection_to_frame(
    detection: RawFaceDetection,
    frame_width: int,
    frame_height: int,
) -> RawFaceDetection | None:
    x, y, width, height = detection.bbox
    x1 = max(0, min(int(x), frame_width - 1))
    y1 = max(0, min(int(y), frame_height - 1))
    x2 = max(0, min(int(x + width), frame_width))
    y2 = max(0, min(int(y + height), frame_height))
    clamped_width = x2 - x1
    clamped_height = y2 - y1
    if clamped_width <= 0 or clamped_height <= 0:
        return None
    return RawFaceDetection(
        bbox=(x1, y1, clamped_width, clamped_height),
        confidence=float(max(0.0, min(1.0, detection.confidence))),
        landmarks=detection.landmarks,
    )

