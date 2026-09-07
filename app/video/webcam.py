"""Webcam ingestion via OpenCV."""

from __future__ import annotations

import logging
from typing import Iterator

import cv2
import numpy as np

from app.video.video_file import VideoOpenError

import os
import sys

logger = logging.getLogger(__name__)


def list_available_cameras(max_to_test: int = 4) -> list[int]:
    """Scan and return indices of accessible cameras on the system."""
    available = []
    # Temporarily set log level to suppress harmless backend probing warnings
    prev_log = os.environ.get("OPENCV_LOG_LEVEL", "")
    os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
    try:
        for idx in range(max_to_test):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available.append(idx)
                cap.release()
    finally:
        if prev_log:
            os.environ["OPENCV_LOG_LEVEL"] = prev_log
        else:
            os.environ.pop("OPENCV_LOG_LEVEL", None)
    return available


class WebcamReader:
    """Reads frames from a local camera index."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        # On Windows, cv2.CAP_DSHOW is often significantly faster and more reliable
        if sys.platform.startswith("win"):
            self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                self.capture = cv2.VideoCapture(camera_index)
        else:
            self.capture = cv2.VideoCapture(camera_index)

        if not self.capture.isOpened():
            available = list_available_cameras()
            avail_msg = f"Available camera indices detected: {available}" if available else "No working cameras detected."
            raise VideoOpenError(
                f"Could not open webcam index {camera_index}. {avail_msg} "
                "Ensure camera permissions are enabled and no other application (Zoom/Teams/Browser) is using it."
            )
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.fps <= 0.0 or self.fps > 120.0:
            self.fps = 30.0
        logger.info("Camera connected index=%s (fps=%.1f)", camera_index, self.fps)

    def frames(self, should_stop) -> Iterator[tuple[int, np.ndarray]]:
        frame_id = 0
        consecutive_failures = 0
        try:
            while not should_stop():
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 50:
                        logger.error("Camera stopped producing frames after 50 consecutive attempts.")
                        break
                    continue
                consecutive_failures = 0
                yield frame_id, frame
                frame_id += 1
        finally:
            self.release()

    def release(self) -> None:
        if self.capture:
            self.capture.release()


