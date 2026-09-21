"""Isolated background worker process/thread for a single camera feed.

Performs:
1. Video ingestion (Webcam, RTSP via MediaMTX, Mobile camera)
2. Decoupled FPS sampling
3. YOLO Person Detection
4. Independent BoT-SORT local tracking
5. Whole-body appearance Re-ID feature extraction
6. Global identity association (PERSON-xxxxx)
7. Independent YuNet Face Detection (STRICTLY NO FACE RECOGNITION)
8. Quality assessment & cooldown-gated Face Snapshot capture
9. PostgreSQL persistent logging & Redis high-frequency caching
10. WebSocket real-time broadcast
11. Camera failure handling with auto-reconnection
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from app.api.websocket import WebSocketManager
from app.association.person import PersonFaceAssociator
from app.cache.redis_client import RedisStateManager, get_redis_manager
from app.config import Settings
from app.db.database import get_db_session
from app.db.repository import IBVAPRepository
from app.detection.person_detector import PersonDetector, create_person_detector
from app.detection.postprocess import clamp_detection_to_frame, confidence_filter, minimum_size_filter
from app.detection.yunet import YuNetDetector
from app.quality.blur import laplacian_variance
from app.quality.quality_score import FaceQualityAssessor, QualityConfig
from app.reid.extractor import PersonReIDExtractor
from app.reid.global_identity_manager import GlobalIdentityManager
from app.schemas.face import BoundingBox, FaceDetection, FaceDetectionFrame, FaceQuality, PersonDetection
from app.storage.object_storage import ObjectStorageService, get_storage_service
from app.tracking.bot_sort import BoTSORTTrack, BoTSORTTracker
from app.video.rtsp import RtspReader
from app.video.sampling import FrameSampler
from app.video.video_file import VideoFileReader
from app.video.webcam import WebcamReader

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CameraWorker:
    """Independent worker thread dedicated to a single camera stream."""

    def __init__(
        self,
        camera_id: str,
        name: str,
        camera_type: str,
        stream_url: str,
        location: str,
        settings: Settings,
        global_identity_manager: GlobalIdentityManager,
        websocket_manager: WebSocketManager,
        enabled: bool = True,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.name = name
        self.camera_type = camera_type.lower()
        self.stream_url = stream_url
        self.location = location
        self.settings = settings
        self.global_identity_manager = global_identity_manager
        self.websocket_manager = websocket_manager
        self.enabled = enabled
        self.event_loop = event_loop

        # Lifecycle state
        self.status = "offline"  # online, offline, reconnecting, stopped, error
        self.last_error: str | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        # Dedicated BoT-SORT local tracker (NEVER shared across cameras)
        self.bot_sort = BoTSORTTracker(
            camera_id=self.camera_id,
            high_thresh=settings.bot_sort_high_thresh,
            low_thresh=settings.bot_sort_low_thresh,
            match_thresh=settings.bot_sort_match_thresh,
            track_buffer=settings.bot_sort_track_buffer,
        )

        # AI Models (Lazy-loaded on worker thread start)
        self.person_detector: PersonDetector | None = None
        self.face_detector: YuNetDetector | None = None
        self.reid_extractor: PersonReIDExtractor | None = None
        self.associator = PersonFaceAssociator()
        self.quality = FaceQualityAssessor(
            QualityConfig(
                blur_threshold=settings.blur_threshold,
                brightness_min=settings.brightness_min,
                brightness_max=settings.brightness_max,
                size_reference_area=settings.size_score_reference_area,
                size_weight=settings.quality_size_weight,
                blur_weight=settings.quality_blur_weight,
                brightness_weight=settings.quality_brightness_weight,
            )
        )

        self.storage_service: ObjectStorageService = get_storage_service()
        self.redis_manager: RedisStateManager = get_redis_manager()

        # Cooldown tracking: global_person_id -> timestamp of last captured snapshot
        self.last_snapshot_time: dict[str, float] = {}
        # Movement log throttle: global_person_id -> timestamp of last MOVING event
        self.last_movement_time: dict[str, float] = {}
        # Active track IDs set
        self.known_track_ids: set[int] = set()

        # Live performance stats
        self.current_fps: float = 0.0
        self.detection_count: int = 0
        self.latest_frame_metadata: dict[str, Any] | None = None

        # Live frame buffer for MJPEG / image feed
        self.latest_raw_frame: np.ndarray | None = None
        self.latest_annotated_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

    def get_latest_jpeg(self, annotated: bool = True) -> bytes:
        """Return latest JPEG bytes of camera feed, or a status placeholder if unready."""
        with self._frame_lock:
            frame = self.latest_annotated_frame if annotated else self.latest_raw_frame
            if frame is not None and frame.size > 0:
                success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    return buf.tobytes()

        # Generate live placeholder frame if no video frames yet
        placeholder = np.full((480, 640, 3), (15, 20, 35), dtype=np.uint8)
        status_text = f"CAMERA {self.camera_id}: {self.status.upper()}"
        cv2.putText(placeholder, status_text, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(placeholder, f"Source: {self.stream_url}", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(placeholder, f"Location: {self.location}", (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        _, buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            logger.warning("Worker for camera %s is already running.", self.camera_id)
            return

        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run_loop,
            name=f"CameraWorker-{self.camera_id}",
            daemon=True,
        )
        self.thread.start()
        logger.info("Launched CameraWorker thread for %s (%s)", self.camera_id, self.stream_url)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.status = "stopped"
        self._sync_camera_status("stopped")
        logger.info("Stopped CameraWorker thread for %s", self.camera_id)

    def _create_reader(self):
        """Instantiate appropriate video reader based on camera type."""
        source = self.stream_url.strip()
        if self.camera_type == "webcam":
            idx = int(source) if source.isdigit() else 0
            return WebcamReader(idx)
        elif self.camera_type in {"rtsp", "phone", "mobile"} or source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://"):
            return RtspReader(source, reconnect_seconds=self.settings.rtsp_reconnect_seconds)
        elif self.camera_type in {"video", "file"}:
            return VideoFileReader(source)
        else:
            # Fallback to index if digits, else rtsp
            if source.isdigit():
                return WebcamReader(int(source))
            return RtspReader(source, reconnect_seconds=self.settings.rtsp_reconnect_seconds)


    def _ensure_models(self) -> None:
        """Initialize detection and Re-ID models on demand when worker runs."""
        if self.person_detector is None:
            self.person_detector = create_person_detector(
                backend=self.settings.person_detector_backend,
                model_path=self.settings.person_model_path,
                confidence_threshold=self.settings.person_confidence_threshold,
                execution_provider=self.settings.execution_provider,
            )
        if self.face_detector is None:
            self.face_detector = YuNetDetector(
                model_path=self.settings.model_path,
                confidence_threshold=self.settings.confidence_threshold,
                nms_threshold=self.settings.nms_threshold,
                execution_provider=self.settings.execution_provider,
            )
        if self.reid_extractor is None:
            self.reid_extractor = PersonReIDExtractor()

    def _run_loop(self) -> None:
        """Main worker loop with automatic failure recovery and reconnection."""
        logger.info("Starting processing loop for camera %s", self.camera_id)
        try:
            self._ensure_models()
        except Exception as exc:
            logger.warning("Error initializing models for camera %s: %s", self.camera_id, exc)

        while not self.stop_event.is_set():
            reader = None
            try:
                self.status = "reconnecting"
                self._sync_camera_status("reconnecting")

                reader = self._create_reader()
                fps = getattr(reader, "fps", 0.0) or 10.0
                sampler = FrameSampler(fps, self.settings.process_fps)

                self.status = "online"
                self.last_error = None
                self._sync_camera_status("online")
                self._log_camera_event("CAMERA_CONNECTED", severity="info")

                logger.info(
                    "[%s] %s EVENT=CAMERA_CONNECTED stream=%s",
                    utc_now().strftime("%H:%M:%S"),
                    self.camera_id,
                    self.stream_url,
                )

                frame_counter = 0
                fps_timer = time.perf_counter()

                for frame_id, frame in reader.frames(self.stop_event.is_set):
                    if self.stop_event.is_set():
                        break

                    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                        continue

                    # Rate limit AI processing
                    if not sampler.should_process(frame_id):
                        continue

                    # Process frame
                    self._process_single_frame(frame, frame_id)

                    frame_counter += 1
                    elapsed = time.perf_counter() - fps_timer
                    if elapsed >= 1.0:
                        self.current_fps = round(frame_counter / elapsed, 1)
                        frame_counter = 0
                        fps_timer = time.perf_counter()

            except Exception as exc:
                self.last_error = str(exc)
                self.status = "offline"
                self._sync_camera_status("offline")
                self._log_camera_event("CAMERA_DISCONNECTED", severity="warning", meta={"error": str(exc)})
                logger.error(
                    "[%s] %s EVENT=CAMERA_DISCONNECTED error=%s",
                    utc_now().strftime("%H:%M:%S"),
                    self.camera_id,
                    exc,
                )
            finally:
                if reader is not None:
                    try:
                        reader.release()
                    except Exception:
                        pass

            # If stopped explicitly, exit cleanly
            if self.stop_event.is_set():
                break

            # Backoff sleep before reconnecting
            logger.info("Camera %s offline. Retrying in %.1fs...", self.camera_id, self.settings.rtsp_reconnect_seconds)
            for _ in range(int(self.settings.rtsp_reconnect_seconds * 10)):
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)

        self.status = "stopped"
        self._sync_camera_status("stopped")

    def _process_single_frame(self, frame: np.ndarray, frame_id: int) -> None:
        """Executes the complete detection, tracking, Re-ID, and face snapshot pipeline."""
        now = utc_now()
        now_ts = time.time()
        height, width = frame.shape[:2]

        # 1. Person Detection (YOLO)
        raw_persons = []
        try:
            raw_persons = self.person_detector.detect(frame)
        except Exception as exc:
            logger.warning("YOLO detection error in %s: %s", self.camera_id, exc)

        # 2. Local Person Tracking (BoT-SORT)
        tracked_persons: list[BoTSORTTrack] = self.bot_sort.update(raw_persons)
        self.detection_count = len(tracked_persons)

        # 3. Person Re-ID & Global Identity Association (CRITICAL: whole-body appearance only)
        current_frame_track_ids = set()
        person_detections_payload = []

        for track in tracked_persons:
            current_frame_track_ids.add(track.local_track_id)

            # Extract 256-D body crop embedding
            feat = self.reid_extractor.extract(frame, track.bbox)
            track.appearance_embedding = feat

            # Match or register via GlobalIdentityManager
            gid, is_new, sim = self.global_identity_manager.match_or_register(
                camera_id=self.camera_id,
                local_track_id=track.local_track_id,
                embedding=feat,
                has_face=track.has_face,
                movement_direction=track.movement_direction,
                confidence=track.confidence,
            )
            track.global_person_id = gid

            # Handle track lifecycle movement logging
            is_first_appearance = track.local_track_id not in self.known_track_ids
            if is_first_appearance:
                self.known_track_ids.add(track.local_track_id)
                self._log_person_movement(gid, track.local_track_id, "ENTER", track.bbox, track.confidence)

            # Periodic MOVING log (every 3 seconds)
            last_mov = self.last_movement_time.get(gid, 0.0)
            if now_ts - last_mov >= 3.0:
                self._log_person_movement(gid, track.local_track_id, "MOVING", track.bbox, track.confidence)
                self.last_movement_time[gid] = now_ts

            person_detections_payload.append(
                {
                    "person_id": f"person_{track.local_track_id:03d}",
                    "track_id": track.local_track_id,
                    "global_person_id": track.global_person_id,
                    "bbox": [track.bbox.x, track.bbox.y, track.bbox.width, track.bbox.height],
                    "confidence": round(track.confidence, 4),
                    "movement_direction": track.movement_direction,
                    "has_face": track.has_face,
                    "timestamp": now.isoformat(),
                }
            )

        # 4. Face Detection (YuNet - STRICTLY INDEPENDENT, NO FACE RECOGNITION)
        raw_faces = self.face_detector.detect(frame)
        clamped_faces = [
            f for f in (clamp_detection_to_frame(face, width, height) for face in raw_faces)
            if f is not None
        ]
        filtered_faces = confidence_filter(clamped_faces, self.settings.confidence_threshold)
        filtered_faces = minimum_size_filter(filtered_faces, self.settings.min_face_width, self.settings.min_face_height)

        face_detections_payload = []
        for idx, raw_face in enumerate(filtered_faces, start=1):
            face_box = BoundingBox(x=raw_face.bbox[0], y=raw_face.bbox[1], width=raw_face.bbox[2], height=raw_face.bbox[3])
            face_quality = self.quality.assess(frame, face_box)

            # Check anatomical association: Does face belong to any tracked person?
            matched_track: BoTSORTTrack | None = None
            for p_track in tracked_persons:
                if self.associator._is_face_in_upper_person(face_box, p_track.bbox):
                    matched_track = p_track
                    break

            if matched_track is not None:
                matched_track.face = FaceDetection(
                    face_id=f"face_{idx:03d}",
                    bbox=face_box,
                    confidence=round(raw_face.confidence, 4),
                    quality=face_quality,
                )

                # 5. Face Snapshot Capture & Cooldown Verification
                gid = matched_track.global_person_id
                if gid and not gid.startswith("PROV-"):
                    last_snap = self.last_snapshot_time.get(gid, 0.0)
                    cooldown_expired = (now_ts - last_snap) >= self.settings.face_snapshot_cooldown

                    # Strict quality criteria
                    is_sharp = face_quality.blur >= self.settings.face_min_sharpness
                    is_large = face_box.width >= self.settings.face_min_size and face_box.height >= self.settings.face_min_size
                    is_confident = raw_face.confidence >= self.settings.face_min_confidence

                    if cooldown_expired and is_sharp and is_large and is_confident:
                        # Capture and store face snapshot without facial recognition
                        snapshot_url = self.storage_service.save_face_snapshot(
                            frame=frame,
                            face_bbox=face_box,
                            person_id=gid,
                            camera_id=self.camera_id,
                        )
                        if snapshot_url:
                            self.last_snapshot_time[gid] = now_ts
                            self._record_face_snapshot(
                                person_id=gid,
                                local_track_id=matched_track.local_track_id,
                                image_url=snapshot_url,
                                confidence=raw_face.confidence,
                            )
                            logger.info(
                                "[%s] %s Track=%d GlobalPerson=%s EVENT=FACE_SNAPSHOT_CAPTURED url=%s (sharpness=%.1f)",
                                now.strftime("%H:%M:%S"),
                                self.camera_id,
                                matched_track.local_track_id,
                                gid,
                                snapshot_url,
                                face_quality.blur,
                            )

            face_detections_payload.append(
                {
                    "face_id": f"face_{idx:03d}",
                    "bbox": [face_box.x, face_box.y, face_box.width, face_box.height],
                    "confidence": round(raw_face.confidence, 4),
                    "quality_score": face_quality.score,
                    "sharpness": face_quality.blur,
                    "associated_track_id": matched_track.local_track_id if matched_track else None,
                    "associated_global_id": matched_track.global_person_id if matched_track else None,
                }
            )

        # 6. Check for EXITED tracks
        lost_track_ids = self.known_track_ids - current_frame_track_ids
        for lost_id in list(lost_track_ids):
            # Check if tracker permanently lost this track
            is_still_in_lost = any(t.local_track_id == lost_id for t in self.bot_sort.lost_tracks)
            if not is_still_in_lost:
                # Track has exited
                self.known_track_ids.discard(lost_id)
                gid = self.global_identity_manager.local_to_global.get((self.camera_id, lost_id), f"TRACK-{lost_id}")
                self._log_person_movement(gid, lost_id, "EXIT", confidence=1.0)
                logger.info(
                    "[%s] %s Track=%d GlobalPerson=%s EVENT=EXIT",
                    now.strftime("%H:%M:%S"),
                    self.camera_id,
                    lost_id,
                    gid,
                )

        # 7. Render Visual Overlays onto Frame Buffer
        annotated_frame = frame.copy()

        # Draw Person Bounding Boxes (Emerald Green)
        for track in tracked_persons:
            bx, by, bw, bh = track.bbox.x, track.bbox.y, track.bbox.width, track.bbox.height
            cv2.rectangle(annotated_frame, (bx, by), (bx + bw, by + bh), (0, 220, 100), 2)

            label = f"{track.global_person_id or ('Trk ' + str(track.local_track_id))} [{track.movement_direction}]"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            banner_y = max(0, by - th - 6)
            cv2.rectangle(annotated_frame, (bx, banner_y), (bx + tw + 6, banner_y + th + 6), (0, 220, 100), -1)
            cv2.putText(annotated_frame, label, (bx + 3, banner_y + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 15, 25), 1, cv2.LINE_AA)

        # Draw Face Bounding Boxes (Cyan)
        for raw_face in filtered_faces:
            fx, fy, fw, fh = raw_face.bbox
            cv2.rectangle(annotated_frame, (fx, fy), (fx + fw, fy + fh), (240, 200, 0), 2)
            flabel = f"Face {int(raw_face.confidence * 100)}%"
            (ftw, fth), _ = cv2.getTextSize(flabel, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            fbanner_y = max(0, fy - fth - 4)
            cv2.rectangle(annotated_frame, (fx, fbanner_y), (fx + ftw + 4, fbanner_y + fth + 4), (240, 200, 0), -1)
            cv2.putText(annotated_frame, flabel, (fx + 2, fbanner_y + fth + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (10, 15, 25), 1, cv2.LINE_AA)

        # Draw Top Status Banner
        header_text = f"[{self.camera_id}] {self.name} | FPS: {self.current_fps:.1f} | Persons: {len(tracked_persons)}"
        cv2.putText(annotated_frame, header_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, header_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 1, cv2.LINE_AA)

        # Store in thread-safe frame buffer
        with self._frame_lock:
            self.latest_raw_frame = frame
            self.latest_annotated_frame = annotated_frame

        # 8. Update Redis High-Frequency State
        self.latest_frame_metadata = {
            "camera_id": self.camera_id,
            "name": self.name,
            "location": self.location,
            "timestamp": now.isoformat(),
            "frame_id": frame_id,
            "status": self.status,
            "fps": self.current_fps,
            "detection_count": self.detection_count,
            "persons": person_detections_payload,
            "faces": face_detections_payload,
        }
        self.redis_manager.set_active_tracks(self.camera_id, person_detections_payload)
        self.redis_manager.set_camera_state(
            self.camera_id,
            {
                "camera_id": self.camera_id,
                "status": self.status,
                "fps": self.current_fps,
                "detection_count": self.detection_count,
                "last_seen": now.isoformat(),
            },
        )

        # 9. Broadcast to WebSockets
        if self.event_loop and not self.event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.websocket_manager.broadcast_json(self.latest_frame_metadata),
                self.event_loop,
            )


    # ==================== Database Helpers ====================

    def _sync_camera_status(self, status: str) -> None:
        try:
            with get_db_session() as session:
                repo = IBVAPRepository(session)
                repo.update_camera_status(self.camera_id, status)
        except Exception:
            pass

    def _log_camera_event(self, event_type: str, severity: str = "info", meta: dict | None = None) -> None:
        now = utc_now()
        payload = meta or {}
        payload["camera_id"] = self.camera_id
        try:
            with get_db_session() as session:
                repo = IBVAPRepository(session)
                repo.log_event(
                    event_type=event_type,
                    severity=severity,
                    camera_id=self.camera_id,
                    metadata=payload,
                    timestamp=now,
                )
        except Exception:
            pass

        try:
            self.redis_manager.push_realtime_event(
                {
                    "event_type": event_type,
                    "severity": severity,
                    "camera_id": self.camera_id,
                    "timestamp": now.isoformat(),
                    "metadata": payload,
                }
            )
        except Exception:
            pass

    def _log_person_movement(
        self,
        person_id: str,
        track_id: int,
        event_type: str,
        bbox: BoundingBox | None = None,
        confidence: float = 1.0,
    ) -> None:
        now = utc_now()
        bbox_coords = [bbox.x, bbox.y, bbox.width, bbox.height] if bbox else None
        try:
            with get_db_session() as session:
                repo = IBVAPRepository(session)
                repo.log_movement(
                    person_id=person_id,
                    camera_id=self.camera_id,
                    track_id=track_id,
                    event_type=event_type,
                    bbox=bbox_coords,
                    confidence=confidence,
                    timestamp=now,
                )
        except Exception:
            pass

        try:
            self.redis_manager.push_realtime_event(
                {
                    "event_type": f"PERSON_{event_type}",
                    "severity": "info",
                    "person_id": person_id,
                    "camera_id": self.camera_id,
                    "track_id": track_id,
                    "timestamp": now.isoformat(),
                    "bbox": bbox_coords,
                }
            )
        except Exception:
            pass

    def _record_face_snapshot(
        self,
        person_id: str,
        local_track_id: int,
        image_url: str,
        confidence: float,
    ) -> None:
        now = utc_now()
        try:
            with get_db_session() as session:
                repo = IBVAPRepository(session)
                repo.save_face_snapshot(
                    person_id=person_id,
                    camera_id=self.camera_id,
                    track_id=local_track_id,
                    image_url=image_url,
                    confidence=confidence,
                    timestamp=now,
                )
                repo.log_movement(
                    person_id=person_id,
                    camera_id=self.camera_id,
                    track_id=local_track_id,
                    event_type="FACE_DETECTED",
                    confidence=confidence,
                    timestamp=now,
                )
        except Exception:
            pass

        try:
            self.redis_manager.push_realtime_event(
                {
                    "event_type": "FACE_SNAPSHOT_CAPTURED",
                    "severity": "info",
                    "person_id": person_id,
                    "camera_id": self.camera_id,
                    "track_id": local_track_id,
                    "image_url": image_url,
                    "timestamp": now.isoformat(),
                }
            )
        except Exception:
            pass
