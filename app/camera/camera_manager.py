"""Camera Manager orchestrating dynamic camera registration, lifecycles, and workers."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from app.api.websocket import WebSocketManager
from app.camera.camera_worker import CameraWorker
from app.config import Settings, get_settings
from app.db.database import get_db_session
from app.db.repository import IBVAPRepository
from app.reid.global_identity_manager import GlobalIdentityManager

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages the full lifecycle of multiple concurrent camera streams."""

    def __init__(
        self,
        settings: Settings,
        global_identity_manager: GlobalIdentityManager,
        websocket_manager: WebSocketManager,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.settings = settings
        self.global_identity_manager = global_identity_manager
        self.websocket_manager = websocket_manager
        self.event_loop = event_loop

        self._lock = threading.RLock()
        self.workers: dict[str, CameraWorker] = {}

    def initialize_from_db(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Load cameras from persistent storage and start enabled workers."""
        if loop is not None:
            self.event_loop = loop

        with self._lock:
            try:
                with get_db_session() as session:
                    repo = IBVAPRepository(session)
                    cameras = repo.get_all_cameras()

                logger.info("Found %d registered cameras in database", len(cameras))
                for cam in cameras:
                    worker = self._create_worker_instance(
                        camera_id=cam.id,
                        name=cam.name,
                        camera_type=cam.type,
                        stream_url=cam.stream_url,
                        location=cam.location,
                        enabled=cam.enabled,
                    )
                    self.workers[cam.id] = worker
                    if cam.enabled:
                        worker.start()
            except Exception as exc:
                logger.warning("Database camera initialization error: %s", exc)

    def register_camera(
        self,
        camera_id: str,
        name: str,
        stream_url: str,
        camera_type: str = "webcam",
        location: str = "Default",
        enabled: bool = True,
        start_immediately: bool = True,
    ) -> dict[str, Any]:
        """Add or update a camera configuration and optionally launch worker."""
        with self._lock:
            # 1. Persist to DB
            try:
                with get_db_session() as session:
                    repo = IBVAPRepository(session)
                    cam = repo.create_or_update_camera(
                        camera_id=camera_id,
                        name=name,
                        stream_url=stream_url,
                        camera_type=camera_type,
                        location=location,
                        status="online" if (enabled and start_immediately) else "offline",
                        enabled=enabled,
                    )
            except Exception as exc:
                logger.warning("DB sync error for camera %s: %s", camera_id, exc)

            # 2. Stop existing worker if running
            if camera_id in self.workers:
                self.workers[camera_id].stop()

            # 3. Create new worker
            worker = self._create_worker_instance(
                camera_id=camera_id,
                name=name,
                camera_type=camera_type,
                stream_url=stream_url,
                location=location,
                enabled=enabled,
            )
            self.workers[camera_id] = worker

            # 4. Start worker if enabled
            if enabled and start_immediately:
                worker.start()

            return self.get_camera(camera_id) or {
                "camera_id": camera_id,
                "name": name,
                "type": camera_type,
                "stream_url": stream_url,
                "location": location,
                "status": worker.status,
                "enabled": enabled,
            }

    def start_camera(self, camera_id: str) -> bool:
        """Start a specific camera stream worker."""
        with self._lock:
            worker = self.workers.get(camera_id)
            if not worker:
                # Attempt to load from DB
                with get_db_session() as session:
                    repo = IBVAPRepository(session)
                    cam = repo.get_camera(camera_id)
                    if cam:
                        worker = self._create_worker_instance(
                            camera_id=cam.id,
                            name=cam.name,
                            camera_type=cam.type,
                            stream_url=cam.stream_url,
                            location=cam.location,
                            enabled=True,
                        )
                        self.workers[camera_id] = worker

            if worker:
                worker.enabled = True
                worker.start()
                return True
            return False

    def stop_camera(self, camera_id: str) -> bool:
        """Stop a specific camera worker."""
        with self._lock:
            worker = self.workers.get(camera_id)
            if worker:
                worker.stop()
                return True
            return False

    def update_camera(
        self,
        camera_id: str,
        name: str | None = None,
        stream_url: str | None = None,
        camera_type: str | None = None,
        location: str | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any] | None:
        """Update existing camera parameters."""
        with self._lock:
            worker = self.workers.get(camera_id)
            if not worker:
                return None

            new_name = name if name is not None else worker.name
            new_url = stream_url if stream_url is not None else worker.stream_url
            new_type = camera_type if camera_type is not None else worker.camera_type
            new_loc = location if location is not None else worker.location
            new_enabled = enabled if enabled is not None else worker.enabled

            restart_needed = (new_url != worker.stream_url) or (new_type != worker.camera_type)
            was_running = worker.status in {"online", "reconnecting"}

            if restart_needed and was_running:
                worker.stop()

            # Update DB
            try:
                with get_db_session() as session:
                    repo = IBVAPRepository(session)
                    repo.create_or_update_camera(
                        camera_id=camera_id,
                        name=new_name,
                        stream_url=new_url,
                        camera_type=new_type,
                        location=new_loc,
                        status=worker.status,
                        enabled=new_enabled,
                    )
            except Exception as exc:
                logger.warning("DB update error for camera %s: %s", camera_id, exc)

            # Recreate worker if settings changed
            if restart_needed:
                worker = self._create_worker_instance(
                    camera_id=camera_id,
                    name=new_name,
                    camera_type=new_type,
                    stream_url=new_url,
                    location=new_loc,
                    enabled=new_enabled,
                )
                self.workers[camera_id] = worker
                if new_enabled:
                    worker.start()
            else:
                worker.name = new_name
                worker.location = new_loc
                worker.enabled = new_enabled
                if not new_enabled and was_running:
                    worker.stop()
                elif new_enabled and not was_running:
                    worker.start()

            return self.get_camera(camera_id)

    def delete_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera from management and persistent storage."""
        with self._lock:
            worker = self.workers.get(camera_id)
            if worker:
                worker.stop()
                del self.workers[camera_id]

            try:
                with get_db_session() as session:
                    repo = IBVAPRepository(session)
                    return repo.delete_camera(camera_id)
            except Exception as exc:
                logger.warning("DB delete error for camera %s: %s", camera_id, exc)
                return worker is not None

    def get_all_cameras(self) -> list[dict[str, Any]]:
        """List all configured cameras and their current runtime statuses."""
        with self._lock:
            results = []
            seen_ids = set()
            try:
                with get_db_session() as session:
                    repo = IBVAPRepository(session)
                    for cam in repo.get_all_cameras():
                        seen_ids.add(cam.id)
                        worker = self.workers.get(cam.id)
                        status = worker.status if worker else cam.status
                        fps = worker.current_fps if worker else 0.0
                        detection_count = worker.detection_count if worker else 0

                        results.append(
                            {
                                "camera_id": cam.id,
                                "name": cam.name,
                                "type": cam.type,
                                "stream_url": cam.stream_url,
                                "location": cam.location,
                                "status": status,
                                "enabled": cam.enabled,
                                "fps": fps,
                                "detection_count": detection_count,
                            }
                        )
            except Exception as exc:
                logger.debug("Error listing cameras from DB: %s", exc)

            # In case worker was created in-memory without DB sync
            for cid, worker in self.workers.items():
                if cid not in seen_ids:
                    results.append(
                        {
                            "camera_id": worker.camera_id,
                            "name": worker.name,
                            "type": worker.camera_type,
                            "stream_url": worker.stream_url,
                            "location": worker.location,
                            "status": worker.status,
                            "enabled": worker.enabled,
                            "fps": worker.current_fps,
                            "detection_count": worker.detection_count,
                        }
                    )
            return results

    def get_camera(self, camera_id: str) -> dict[str, Any] | None:
        """Get details for a single camera."""
        with self._lock:
            worker = self.workers.get(camera_id)
            try:
                with get_db_session() as session:
                    repo = IBVAPRepository(session)
                    cam = repo.get_camera(camera_id)
                    if cam:
                        return {
                            "camera_id": cam.id,
                            "name": cam.name,
                            "type": cam.type,
                            "stream_url": cam.stream_url,
                            "location": cam.location,
                            "status": worker.status if worker else cam.status,
                            "enabled": cam.enabled,
                            "fps": worker.current_fps if worker else 0.0,
                            "detection_count": worker.detection_count if worker else 0,
                            "last_error": worker.last_error if worker else None,
                        }
            except Exception:
                pass

            if worker:
                return {
                    "camera_id": worker.camera_id,
                    "name": worker.name,
                    "type": worker.camera_type,
                    "stream_url": worker.stream_url,
                    "location": worker.location,
                    "status": worker.status,
                    "enabled": worker.enabled,
                    "fps": worker.current_fps,
                    "detection_count": worker.detection_count,
                    "last_error": worker.last_error,
                }
            return None

    def stop_all(self) -> None:
        """Stop all running camera workers."""
        with self._lock:
            for worker in self.workers.values():
                worker.stop()
            self.workers.clear()

    def _create_worker_instance(
        self,
        camera_id: str,
        name: str,
        camera_type: str,
        stream_url: str,
        location: str,
        enabled: bool,
    ) -> CameraWorker:
        return CameraWorker(
            camera_id=camera_id,
            name=name,
            camera_type=camera_type,
            stream_url=stream_url,
            location=location,
            settings=self.settings,
            global_identity_manager=self.global_identity_manager,
            websocket_manager=self.websocket_manager,
            enabled=enabled,
            event_loop=self.event_loop,
        )


_camera_manager = None


def get_camera_manager(
    settings: Settings | None = None,
    global_identity_manager: GlobalIdentityManager | None = None,
    websocket_manager: WebSocketManager | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
) -> CameraManager:
    global _camera_manager
    if _camera_manager is None:
        if settings is None or global_identity_manager is None or websocket_manager is None:
            raise RuntimeError("CameraManager not initialized; pass required dependencies.")
        _camera_manager = CameraManager(
            settings=settings,
            global_identity_manager=global_identity_manager,
            websocket_manager=websocket_manager,
            event_loop=event_loop,
        )
    return _camera_manager
