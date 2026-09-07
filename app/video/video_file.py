"""Uploaded/local video file ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoOpenError(RuntimeError):
    """Raised when OpenCV cannot open a video source."""


class VideoFileReader:
    """Streams video frames without loading the whole file into memory."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise VideoOpenError(f"Could not open video file: {self.path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        frame_id = 0
        try:
            while True:
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    break
                yield frame_id, frame
                frame_id += 1
        finally:
            self.release()

    def release(self) -> None:
        if self.capture:
            self.capture.release()


