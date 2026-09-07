"""YuNet detector implementation using OpenCV's FaceDetectorYN API."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.detection.base import FaceDetector, ModelLoadError, RawFaceDetection


class YuNetDetector(FaceDetector):
    """OpenCV YuNet detector wrapped behind the FaceDetector interface."""

    name = "YuNet"

    def __init__(
        self,
        model_path: Path | str,
        confidence_threshold: float = 0.65,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
        execution_provider: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.execution_provider = execution_provider

        if not self.model_path.exists():
            raise ModelLoadError(f"YuNet model not found at {self.model_path}")
        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise ModelLoadError(
                "OpenCV was installed without FaceDetectorYN. Install opencv-contrib-python-headless."
            )

        backend_id = cv2.dnn.DNN_BACKEND_OPENCV
        target_id = cv2.dnn.DNN_TARGET_CPU
        if execution_provider.lower() == "gpu":
            backend_id = cv2.dnn.DNN_BACKEND_CUDA
            target_id = cv2.dnn.DNN_TARGET_CUDA

        try:
            self._detector = cv2.FaceDetectorYN_create(
                str(self.model_path),
                "",
                (320, 320),
                self.confidence_threshold,
                self.nms_threshold,
                self.top_k,
                backend_id,
                target_id,
            )
        except Exception as exc:  # pragma: no cover - OpenCV error type varies by build
            raise ModelLoadError(f"Failed to initialize YuNet: {exc}") from exc

    def detect(self, frame: np.ndarray) -> list[RawFaceDetection]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []
        if frame.ndim != 3 or frame.shape[2] != 3:
            return []

        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []

        detections: list[RawFaceDetection] = []
        for row in faces:
            x, y, w, h = row[:4]
            landmarks = row[4:14].reshape((5, 2)).astype(float).tolist()
            score = float(row[14])
            detections.append(
                RawFaceDetection(
                    bbox=(int(round(x)), int(round(y)), int(round(w)), int(round(h))),
                    confidence=max(0.0, min(1.0, score)),
                    landmarks=landmarks,
                )
            )
        return detections

