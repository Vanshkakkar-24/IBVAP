"""Person detector implementations supporting YOLOv8 ONNX and OpenCV HOG fallback."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.detection.base import DetectorError, ModelLoadError
from app.detection.weights import ensure_person_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RawPersonDetection:
    """Detector-native person detection before tracking and association."""

    bbox: tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    class_name: str = "person"


class PersonDetector(ABC):
    """Abstract base class for person detectors."""

    name: str = "person_base"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[RawPersonDetection]:
        """Detect person bounding boxes from a BGR image frame."""


class YoloPersonDetector(PersonDetector):
    """YOLOv8 ONNX person detector leveraging ONNX Runtime or OpenCV DNN."""

    name = "YOLOv8"

    def __init__(
        self,
        model_path: Path | str = Path("models/yolov8n.onnx"),
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
        execution_provider: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = input_size
        self.execution_provider = execution_provider.lower()
        self._session = None
        self._dnn_net = None

        if not self.model_path.exists():
            # Attempt to download
            if not ensure_person_model(self.model_path):
                raise ModelLoadError(f"YOLOv8 model file not found at {self.model_path}")

        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self.execution_provider == "gpu" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(str(self.model_path), session_options, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            logger.info("Initialized YoloPersonDetector using ONNX Runtime with providers=%s", providers)
        except Exception as exc:
            logger.warning("ONNX Runtime initialization failed (%s); falling back to OpenCV DNN", exc)
            try:
                self._dnn_net = cv2.dnn.readNetFromONNX(str(self.model_path))
                if self.execution_provider == "gpu":
                    self._dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self._dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                else:
                    self._dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self._dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("Initialized YoloPersonDetector using OpenCV DNN")
            except Exception as dnn_exc:
                raise ModelLoadError(f"Failed to load YOLO model via both ONNX Runtime and OpenCV DNN: {dnn_exc}") from dnn_exc

    def detect(self, frame: np.ndarray) -> list[RawPersonDetection]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]
        in_w, in_h = self.input_size

        # Preprocessing: resize to 640x640, BGR to RGB, normalize to [0, 1]
        resized = cv2.resize(frame, (in_w, in_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        # Transpose from (H, W, C) to (1, C, H, W)
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        # Inference
        if self._session is not None:
            outputs = self._session.run(None, {self._input_name: blob})
            raw_out = outputs[0]  # shape: (1, 84, 8400)
        else:
            self._dnn_net.setInput(blob)
            raw_out = self._dnn_net.forward()

        # Output shape is (1, 84, 8400) -> transpose to (8400, 84)
        if raw_out.ndim == 3:
            predictions = np.transpose(raw_out[0], (1, 0))
        else:
            predictions = raw_out

        # In YOLOv8 COCO models:
        # columns 0..3: cx, cy, w, h
        # column 4: person class probability (class 0)
        person_scores = predictions[:, 4]
        mask = person_scores >= self.confidence_threshold
        valid_preds = predictions[mask]
        if len(valid_preds) == 0:
            return []

        scores = valid_preds[:, 4]
        cx = valid_preds[:, 0]
        cy = valid_preds[:, 1]
        w = valid_preds[:, 2]
        h = valid_preds[:, 3]

        scale_x = orig_w / in_w
        scale_y = orig_h / in_h

        # Convert cx, cy, w, h to x1, y1, w, h in original frame coordinates
        x1 = np.maximum(0, (cx - w / 2.0) * scale_x).astype(int)
        y1 = np.maximum(0, (cy - h / 2.0) * scale_y).astype(int)
        box_w = np.minimum(orig_w - x1, w * scale_x).astype(int)
        box_h = np.minimum(orig_h - y1, h * scale_y).astype(int)

        boxes = [
            [int(x1[i]), int(y1[i]), int(box_w[i]), int(box_h[i])]
            for i in range(len(valid_preds))
            if box_w[i] > 10 and box_h[i] > 10
        ]
        confidences = [float(scores[i]) for i in range(len(boxes))]

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        if len(indices) == 0:
            return []

        # indices can be a flat 1D array or 2D array depending on OpenCV version
        flat_indices = np.array(indices).flatten()

        detections: list[RawPersonDetection] = []
        for idx in flat_indices:
            bx, by, bw, bh = boxes[idx]
            detections.append(
                RawPersonDetection(
                    bbox=(bx, by, bw, bh),
                    confidence=round(confidences[idx], 4),
                    class_name="person",
                )
            )
        return detections


class HOGPersonDetector(PersonDetector):
    """OpenCV HOG + Linear SVM person detector for zero-download offline fallback."""

    name = "OpenCV_HOG"

    def __init__(self, confidence_threshold: float = 0.20) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[RawPersonDetection]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]
        # Downscale large frames for faster HOG processing
        target_w = min(orig_w, 640)
        scale = orig_w / target_w
        if scale > 1.0:
            proc_frame = cv2.resize(frame, (target_w, int(orig_h / scale)))
        else:
            proc_frame = frame

        rects, weights = self.hog.detectMultiScale(
            proc_frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        detections: list[RawPersonDetection] = []
        for (rx, ry, rw, rh), weight in zip(rects, weights):
            conf = float(weight[0]) if isinstance(weight, (list, np.ndarray)) else float(weight)
            # Map back to original coordinate system
            bx = int(round(rx * scale))
            by = int(round(ry * scale))
            bw = int(round(rw * scale))
            bh = int(round(rh * scale))

            # Clamp
            bx = max(0, min(orig_w - 1, bx))
            by = max(0, min(orig_h - 1, by))
            bw = max(1, min(orig_w - bx, bw))
            bh = max(1, min(orig_h - by, bh))

            # Normalize HOG weight to a 0..1 confidence heuristic
            normalized_conf = min(1.0, max(0.1, 0.5 + conf * 0.25))
            if normalized_conf >= self.confidence_threshold:
                detections.append(
                    RawPersonDetection(
                        bbox=(bx, by, bw, bh),
                        confidence=round(normalized_conf, 4),
                        class_name="person",
                    )
                )
        return detections


def create_person_detector(
    backend: str = "auto",
    model_path: Path | str = Path("models/yolov8n.onnx"),
    confidence_threshold: float = 0.45,
    execution_provider: str = "cpu",
) -> PersonDetector:
    """Factory creating the appropriate person detector according to settings."""
    choice = str(backend).lower()
    if choice == "hog":
        logger.info("Using HOGPersonDetector (configured)")
        return HOGPersonDetector(confidence_threshold=confidence_threshold)

    if choice == "yolo":
        return YoloPersonDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            execution_provider=execution_provider,
        )

    # auto mode: try YOLO first, fallback to HOG
    try:
        detector = YoloPersonDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            execution_provider=execution_provider,
        )
        logger.info("Using YoloPersonDetector (auto detected)")
        return detector
    except Exception as exc:
        logger.warning("Could not initialize YoloPersonDetector (%s); falling back to HOGPersonDetector", exc)
        return HOGPersonDetector(confidence_threshold=confidence_threshold)
