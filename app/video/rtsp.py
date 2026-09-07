"""RTSP stream ingestion with reconnect behavior."""

from __future__ import annotations

import logging
import time
from typing import Iterator

import cv2
import numpy as np

from app.logging_config import mask_url_secret

logger = logging.getLogger(__name__)


class RtspReader:
    """Reads frames from an RTSP stream and reconnects when needed."""

    def __init__(self, url: str, reconnect_seconds: float = 5.0) -> None:
        self.url = url
        self.reconnect_seconds = reconnect_seconds
        self.capture: cv2.VideoCapture | None = None
        self.fps = 0.0

    def _connect(self) -> bool:
        self.release()
        safe_url = mask_url_secret(self.url)
        logger.info("Connecting RTSP stream url=%s", safe_url)
        self.capture = cv2.VideoCapture(self.url)
        if not self.capture.isOpened():
            logger.warning("RTSP connection failed url=%s", safe_url)
            self.release()
            return False
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        logger.info("RTSP connected url=%s", safe_url)
        return True

    def frames(self, should_stop) -> Iterator[tuple[int, np.ndarray]]:
        frame_id = 0
        while not should_stop():
            if self.capture is None and not self._connect():
                time.sleep(self.reconnect_seconds)
                continue

            ok, frame = self.capture.read() if self.capture else (False, None)
            if not ok or frame is None:
                logger.warning("RTSP disconnected; reconnecting")
                self.release()
                time.sleep(self.reconnect_seconds)
                continue

            yield frame_id, frame
            frame_id += 1
        self.release()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

