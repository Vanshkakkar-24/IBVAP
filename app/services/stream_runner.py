"""Background webcam/RTSP stream processing."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

from app.api.websocket import WebSocketManager
from app.services.face_service import FaceDetectionService
from app.video.rtsp import RtspReader
from app.video.sampling import FrameSampler
from app.video.video_file import VideoOpenError
from app.video.webcam import WebcamReader

logger = logging.getLogger(__name__)


class StreamController:
    """Owns one active camera-like stream."""

    def __init__(self, service: FaceDetectionService, websocket_manager: WebSocketManager) -> None:
        self.service = service
        self.websocket_manager = websocket_manager
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.running_source: str | None = None
        self.last_error: str | None = None

    def start_webcam(self, camera_index: int, camera_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self._start(
            source_name=f"webcam:{camera_index}",
            camera_id=camera_id,
            loop=loop,
            reader_factory=lambda: WebcamReader(camera_index),
            source_type="webcam",
        )

    def start_rtsp(self, rtsp_url: str, camera_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self._start(
            source_name="rtsp",
            camera_id=camera_id,
            loop=loop,
            reader_factory=lambda: RtspReader(rtsp_url, self.service.settings.rtsp_reconnect_seconds),
            source_type="rtsp",
        )

    def stop(self) -> bool:
        if self._thread is None:
            return False
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None
        self.running_source = None
        self.service.metrics.processing_ended_at = __import__("time").perf_counter()
        return True

    def _start(
        self,
        source_name: str,
        camera_id: str,
        loop: asyncio.AbstractEventLoop,
        reader_factory: Callable,
        source_type: str,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"A stream is already running: {self.running_source}")
        self.service.ensure_ready()
        self.service.tracker.reset()
        self.service.metrics.reset()
        self.service.metrics.processing_started_at = __import__("time").perf_counter()
        self._stop_event.clear()
        self.last_error = None
        self.running_source = source_name
        self._thread = threading.Thread(
            target=self._run,
            args=(reader_factory, camera_id, source_type, loop),
            daemon=True,
        )
        self._thread.start()

    def _run(self, reader_factory: Callable, camera_id: str, source_type: str, loop: asyncio.AbstractEventLoop) -> None:
        try:
            reader = reader_factory()
            self.service.metrics.input_fps = getattr(reader, "fps", 0.0)
            sampler = FrameSampler(getattr(reader, "fps", 0.0), self.service.settings.process_fps)
            logger.info("Video processing started source=%s", source_type)
            for frame_id, frame in reader.frames(self._stop_event.is_set):
                if not sampler.should_process(frame_id):
                    continue
                metadata = self.service.process_frame(
                    frame,
                    frame_id=frame_id,
                    camera_id=camera_id,
                    input_fps=getattr(reader, "fps", 0.0),
                    source_type=source_type,
                )
                payload = metadata.model_dump(mode="json")
                asyncio.run_coroutine_threadsafe(self.websocket_manager.broadcast_json(payload), loop)
        except VideoOpenError as exc:
            self.last_error = str(exc)
            logger.error("Stream open failed: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive background guard
            self.last_error = str(exc)
            logger.exception("Stream processing failed")
        finally:
            self.running_source = None
            self.service.metrics.processing_ended_at = __import__("time").perf_counter()

