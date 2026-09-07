"""Inference engine for the trained custom TinyFaceDetector ONNX model."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.detection.base import FaceDetector, ModelLoadError, RawFaceDetection


class CustomTinyFaceDetector(FaceDetector):
    """Trained custom CNN face detector running via ONNX Runtime."""

    name = "TinyFaceNet"

    def __init__(
        self,
        model_path: Path | str = "models/custom_face_detector.onnx",
        confidence_threshold: float = 0.50,
        input_size: int = 128,
        execution_provider: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.input_size = input_size

        if not self.model_path.exists():
            raise ModelLoadError(
                f"Custom face model not found at {self.model_path}. "
                "Train it first by running: python -m app.training.train"
            )

        providers = ["CPUExecutionProvider"]
        if execution_provider.lower() == "gpu":
            providers.insert(0, "CUDAExecutionProvider")

        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load ONNX custom detector: {exc}") from exc

    def detect(self, frame: np.ndarray) -> list[RawFaceDetection]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []
        if frame.ndim != 3 or frame.shape[2] != 3:
            return []

        h, w = frame.shape[:2]
        # Preprocess: resize to model input size, convert BGR to RGB
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)  # (1, 3, 128, 128)

        # Run ONNX inference
        try:
            outs = self.session.run(None, {"input": tensor})
            logits, bboxes, landmarks = outs[0], outs[1], outs[2]
        except Exception:
            return []

        logit = float(logits[0][0])
        # Sigmoid activation for confidence
        conf = 1.0 / (1.0 + math.exp(-logit))

        if conf < self.confidence_threshold:
            return []

        # Denormalize bbox to original frame dimensions
        nx, ny, nw, nh = bboxes[0]
        bx = int(round(float(nx) * w))
        by = int(round(float(ny) * h))
        bw = int(round(float(nw) * w))
        bh = int(round(float(nh) * h))

        # Denormalize landmarks: 5 points (x, y)
        raw_lm = landmarks[0]
        pts = []
        for i in range(0, 10, 2):
            lx = float(raw_lm[i]) * w
            ly = float(raw_lm[i + 1]) * h
            pts.append([lx, ly])

        return [
            RawFaceDetection(
                bbox=(max(0, bx), max(0, by), max(1, bw), max(1, bh)),
                confidence=round(conf, 4),
                landmarks=pts,
            )
        ]
