"""Scalable Multi-Camera Stream Manager coordinating multiple concurrent video sources."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.api.websocket import WebSocketManager
from app.config import Settings
from app.reid.registry import GlobalPersonRegistry
from app.schemas.face import CameraStreamConfig, CrossCameraTransition, FaceDetectionFrame, GlobalPersonInfo
from app.services.face_service import FaceDetectionService
from app.video.rtsp import RtspReader
from app.video.sampling import FrameSampler
from app.video.video_file import VideoFileReader, VideoOpenError
from app.video.webcam import WebcamReader

logger = logging.getLogger(__name__)


@dataclass
class CameraWorker:
    """Represents a running background stream worker for a single camera."""

    camera_id: str
    source_type: str
    source_value: str
    service: FaceDetectionService
    stop_event: threading.Event
    thread: threading.Thread | None = None
    is_active: bool = False
    last_error: str | None = None
    latest_frame_metadata: FaceDetectionFrame | None = None


class MultiCameraManager:
    """Orchestrates multiple camera streams, synchronizing their Re-ID with a shared GlobalPersonRegistry."""

    def __init__(
        self,
        settings: Settings,
        websocket_manager: WebSocketManager,
        global_registry: GlobalPersonRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.websocket_manager = websocket_manager
        self.global_registry = (
            global_registry
            if global_registry is not None
            else GlobalPersonRegistry(similarity_threshold=settings.reid_similarity_threshold)
        )
        self.workers: dict[str, CameraWorker] = {}
        self._lock = threading.Lock()

    def add_camera(
        self,
        camera_id: str,
        source_type: str,
        source_value: str,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Register and start a new camera stream."""
        with self._lock:
            if camera_id in self.workers and self.workers[camera_id].is_active:
                raise RuntimeError(f"Camera '{camera_id}' is already active.")

            # Create a dedicated FaceDetectionService for this camera, sharing the central GlobalPersonRegistry
            service = FaceDetectionService(
                settings=self.settings,
                global_registry=self.global_registry,
            )
            service.ensure_ready()

            stop_event = threading.Event()
            worker = CameraWorker(
                camera_id=camera_id,
                source_type=source_type.lower(),
                source_value=source_value,
                service=service,
                stop_event=stop_event,
                is_active=False,
            )
            self.workers[camera_id] = worker

            # Launch runner thread
            thread = threading.Thread(
                target=self._run_camera_loop,
                args=(worker, loop),
                daemon=True,
                name=f"Worker-{camera_id}",
            )
            worker.thread = thread
            worker.is_active = True
            thread.start()
            logger.info("Started camera worker camera_id=%s source=%s:%s", camera_id, source_type, source_value)

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera stream."""
        with self._lock:
            worker = self.workers.get(camera_id)
            if worker is None:
                return False

            worker.stop_event.set()
            if worker.thread and worker.thread.is_alive():
                worker.thread.join(timeout=3.0)
            worker.is_active = False
            del self.workers[camera_id]
            logger.info("Stopped and removed camera worker camera_id=%s", camera_id)
            return True

    def stop_all(self) -> None:
        """Stop all active camera workers."""
        with self._lock:
            for cid, worker in list(self.workers.items()):
                worker.stop_event.set()
                if worker.thread and worker.thread.is_alive():
                    worker.thread.join(timeout=2.0)
                worker.is_active = False
            self.workers.clear()

    def get_active_cameras(self) -> list[dict[str, str | bool | None]]:
        """List all active cameras and their current statuses."""
        with self._lock:
            return [
                {
                    "camera_id": w.camera_id,
                    "source_type": w.source_type,
                    "source_value": w.source_value,
                    "is_active": w.is_active,
                    "last_error": w.last_error,
                    "person_count": len(w.latest_frame_metadata.persons) if w.latest_frame_metadata else 0,
                    "face_count": len(w.latest_frame_metadata.faces) if w.latest_frame_metadata else 0,
                }
                for w in self.workers.values()
            ]

    def get_global_persons(self) -> list[GlobalPersonInfo]:
        """Return all globally tracked identities across all cameras."""
        return self.global_registry.get_active_persons()

    def get_transitions(self) -> list[CrossCameraTransition]:
        """Return all cross-camera handoff events."""
        return self.global_registry.get_transitions()

    def _run_camera_loop(self, worker: CameraWorker, loop: asyncio.AbstractEventLoop | None) -> None:
        reader = None
        try:
            # Instantiate reader based on source type
            if worker.source_type == "webcam":
                idx = int(worker.source_value) if worker.source_value.isdigit() else 0
                reader = WebcamReader(idx)
            elif worker.source_type == "rtsp":
                reader = RtspReader(worker.source_value, self.settings.rtsp_reconnect_seconds)
            elif worker.source_type in {"video", "file"}:
                reader = VideoFileReader(worker.source_value)
            else:
                raise ValueError(f"Unsupported source type: {worker.source_type}")

            fps = getattr(reader, "fps", 0.0) or 10.0
            sampler = FrameSampler(fps, self.settings.process_fps)
            worker.service.metrics.input_fps = fps

            for frame_id, frame in reader.frames(worker.stop_event.is_set):
                if not sampler.should_process(frame_id):
                    continue

                metadata = worker.service.process_frame(
                    frame,
                    frame_id=frame_id,
                    camera_id=worker.camera_id,
                    input_fps=fps,
                    source_type=worker.source_type,
                )
                worker.latest_frame_metadata = metadata

                # Broadcast to WebSockets if event loop is running
                if loop and not loop.is_closed():
                    payload = metadata.model_dump(mode="json")
                    asyncio.run_coroutine_threadsafe(self.websocket_manager.broadcast_json(payload), loop)
        except Exception as exc:
            worker.last_error = str(exc)
            logger.error("Camera loop error in %s: %s", worker.camera_id, exc)
        finally:
            if reader:
                try:
                    reader.release()
                except Exception:
                    pass
            worker.is_active = False
