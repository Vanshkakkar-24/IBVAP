"""Main face detection pipeline service."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable
from uuid import uuid4

import cv2
import numpy as np

from app.association.person import PersonFaceAssociator
from app.config import Settings
from app.detection.base import FaceDetector, ModelLoadError
from app.detection.person_detector import PersonDetector, create_person_detector
from app.detection.postprocess import clamp_detection_to_frame, confidence_filter, minimum_size_filter
from app.detection.yunet import YuNetDetector
from app.quality.quality_score import FaceQualityAssessor, QualityConfig
from app.reid.extractor import PersonReIDExtractor
from app.reid.registry import GlobalPersonRegistry
from app.schemas.face import BoundingBox, FaceDetection, FaceDetectionFrame, MetricsSnapshot, PersonDetection
from app.services.metrics import MetricsAccumulator
from app.tracking.base import FaceTracker
from app.tracking.person_tracker import PersonTracker, PersonTrackState
from app.tracking.tracker import IoUFaceTracker
from app.video.sampling import FrameSampler
from app.video.video_file import VideoFileReader
from app.visualization.drawer import FaceDrawer

logger = logging.getLogger(__name__)


class FaceDetectionService:
    """Coordinates preprocessing, detection, filtering, quality, tracking, and metadata."""

    def __init__(
        self,
        settings: Settings,
        detector: FaceDetector | None = None,
        tracker: FaceTracker | None = None,
        person_detector: PersonDetector | None = None,
        global_registry: GlobalPersonRegistry | None = None,
    ) -> None:
        self.settings = settings
        self._detector = detector
        self._person_detector = person_detector
        self.tracker = tracker if tracker is not None else IoUFaceTracker()
        self.person_tracker = PersonTracker(
            iou_threshold=0.30,
            max_missed=settings.reid_max_missed_frames,
        )
        self.associator = PersonFaceAssociator()
        self.reid_extractor = PersonReIDExtractor()
        self.global_registry = (
            global_registry
            if global_registry is not None
            else GlobalPersonRegistry(similarity_threshold=settings.reid_similarity_threshold)
        )
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
        self.drawer = FaceDrawer(enabled=settings.enable_visualization)
        self.metrics = MetricsAccumulator(
            execution_mode=settings.execution_provider,
            detector="YuNet",
        )

    @property
    def person_detector(self) -> PersonDetector:
        if self._person_detector is None:
            self._person_detector = create_person_detector(
                backend=self.settings.person_detector_backend,
                model_path=self.settings.person_model_path,
                confidence_threshold=self.settings.person_confidence_threshold,
                execution_provider=self.settings.execution_provider,
            )
        return self._person_detector

    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            backend = str(self.settings.detector_backend).lower()
            if backend == "custom":
                from app.detection.custom_detector import CustomTinyFaceDetector

                custom_path = (
                    self.settings.model_path
                    if str(self.settings.model_path).endswith("custom_face_detector.onnx")
                    else Path("models/custom_face_detector.onnx")
                )
                self._detector = CustomTinyFaceDetector(
                    model_path=custom_path,
                    confidence_threshold=self.settings.confidence_threshold,
                    execution_provider=self.settings.execution_provider,
                )
                self.metrics.detector = "TinyFaceNet"
                logger.info("Face detector initialized detector=TinyFaceNet model=%s", custom_path)
            else:
                self._detector = YuNetDetector(
                    model_path=self.settings.model_path,
                    confidence_threshold=self.settings.confidence_threshold,
                    nms_threshold=self.settings.nms_threshold,
                    execution_provider=self.settings.execution_provider,
                )
                self.metrics.detector = "YuNet"
                logger.info("Face detector initialized detector=YuNet model=%s", self.settings.model_path)
        return self._detector

    def ensure_ready(self) -> None:
        _ = self.detector
        if self.settings.enable_person_detection:
            _ = self.person_detector

    def reset_trackers(self) -> None:
        self.tracker.reset()
        self.person_tracker.reset()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        camera_id: str | None = None,
        input_fps: float | None = None,
        source_type: str | None = None,
    ) -> FaceDetectionFrame:
        start = time.perf_counter()
        cam_id = camera_id or self.settings.camera_id
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            latency_ms = (time.perf_counter() - start) * 1000
            self.metrics.update(0, 0.0, latency_ms, [], [], person_count=0)
            return FaceDetectionFrame(
                camera_id=cam_id,
                frame_id=frame_id,
                faces=[],
                persons=[],
                input_fps=input_fps,
                processing_fps=self.metrics.snapshot().processing_fps,
                source_type=source_type,
            )

        height, width = frame.shape[:2]

        # 1. Person Detection & Tracking (continuous tracking even if face is not visible)
        tracked_persons: list[PersonTrackState] = []
        if self.settings.enable_person_detection:
            try:
                raw_persons = self.person_detector.detect(frame)
                if self.settings.enable_person_tracking:
                    tracked_persons = self.person_tracker.update(raw_persons)
                else:
                    tracked_persons = [
                        PersonTrackState(
                            track_id=idx + 1,
                            bbox=BoundingBox(x=p.bbox[0], y=p.bbox[1], width=p.bbox[2], height=p.bbox[3]),
                            confidence=p.confidence,
                        )
                        for idx, p in enumerate(raw_persons)
                    ]
            except Exception as exc:
                logger.warning("Person detection error: %s", exc)

        # 2. Face Detection & Postprocessing
        inference_start = time.perf_counter()
        raw_faces = self.detector.detect(frame)
        inference_ms = (time.perf_counter() - inference_start) * 1000

        clamped = [
            item
            for item in (clamp_detection_to_frame(face, width, height) for face in raw_faces)
            if item is not None
        ]
        filtered = confidence_filter(clamped, self.settings.confidence_threshold)
        filtered = minimum_size_filter(filtered, self.settings.min_face_width, self.settings.min_face_height)

        faces: list[FaceDetection] = []
        for index, raw in enumerate(filtered, start=1):
            bbox = BoundingBox(x=raw.bbox[0], y=raw.bbox[1], width=raw.bbox[2], height=raw.bbox[3])
            quality = self.quality.assess(frame, bbox)
            faces.append(
                FaceDetection(
                    face_id=f"face_{index:03d}",
                    bbox=bbox,
                    confidence=round(raw.confidence, 4),
                    quality=quality,
                    landmarks=raw.landmarks,
                )
            )

        if self.settings.enable_tracking:
            faces = self.tracker.update(faces)

        # 3. Person-Face Association & Cross-Camera Re-ID
        persons: list[PersonDetection] = []
        if self.settings.enable_person_detection and tracked_persons:
            # Associate visible faces to person tracks (if face is hidden, track persists with face=None)
            self.associator.attach_faces_to_persons(tracked_persons, faces)

            for track in tracked_persons:
                if self.settings.enable_reid:
                    feat = self.reid_extractor.extract(frame, track.bbox)
                    track.reid_embedding = feat
                    gid, _, _ = self.global_registry.match_or_register(
                        cam_id, track.track_id, feat, has_face=track.has_face
                    )
                    track.global_person_id = gid

                persons.append(
                    PersonDetection(
                        person_id=f"person_{track.track_id:03d}",
                        track_id=track.track_id,
                        global_person_id=track.global_person_id,
                        bbox=track.bbox,
                        confidence=round(track.confidence, 4),
                        has_face=track.has_face,
                        face=track.face,
                        reid_embedding=track.reid_embedding,
                    )
                )

        latency_ms = (time.perf_counter() - start) * 1000
        self.metrics.update(
            face_count=len(faces),
            inference_ms=inference_ms,
            latency_ms=latency_ms,
            confidences=[face.confidence for face in faces],
            qualities=[face.quality.score for face in faces],
            person_count=len(persons),
        )
        metadata = FaceDetectionFrame(
            camera_id=cam_id,
            frame_id=frame_id,
            faces=faces,
            persons=persons,
            input_fps=input_fps,
            processing_fps=self.metrics.snapshot().processing_fps,
            source_type=source_type,
        )
        return metadata

    def process_image_path(self, image_path: Path | str, camera_id: str | None = None) -> FaceDetectionFrame:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Invalid image file: {image_path}")
        self._start_metrics(0.0)
        metadata = self.process_frame(frame, 0, camera_id=camera_id, source_type="image")
        self._end_metrics()
        return metadata

    def process_video_path(
        self,
        video_path: Path | str,
        camera_id: str | None = None,
        output_annotated: bool | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[FaceDetectionFrame], MetricsSnapshot, Path | None]:
        reader = VideoFileReader(video_path)
        output_path: Path | None = None
        writer: cv2.VideoWriter | None = None
        output_annotated = self.settings.save_annotated_output if output_annotated is None else output_annotated
        self._start_metrics(reader.fps)
        self.reset_trackers()

        try:
            sampler = FrameSampler(reader.fps, self.settings.process_fps)
            metadata_frames: list[FaceDetectionFrame] = []
            for frame_id, frame in reader.frames():
                if progress_callback and reader.total_frames > 0:
                    progress_callback(frame_id + 1, reader.total_frames)

                if not sampler.should_process(frame_id):
                    continue
                metadata = self.process_frame(
                    frame,
                    frame_id=frame_id,
                    camera_id=camera_id,
                    input_fps=reader.fps,
                    source_type="video",
                )
                metadata_frames.append(metadata)

                if output_annotated and self.settings.enable_visualization:
                    if writer is None:
                        self.settings.ensure_storage_dirs()
                        output_path = self.settings.output_dir / f"annotated_{uuid4().hex}.mp4"
                        h, w = frame.shape[:2]
                        fps_out = reader.fps or max(self.settings.process_fps, 1.0)
                        
                        # Try browser-friendly avc1 first, fallback to mp4v
                        writer = None
                        for fourcc_code in ("avc1", "H264", "mp4v"):
                            try:
                                candidate = cv2.VideoWriter(
                                    str(output_path),
                                    cv2.VideoWriter_fourcc(*fourcc_code),
                                    fps_out,
                                    (w, h),
                                )
                                if candidate.isOpened():
                                    writer = candidate
                                    break
                            except Exception:
                                continue
                        if writer is None:
                            writer = cv2.VideoWriter(
                                str(output_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps_out,
                                (w, h),
                            )
                    writer.write(self.drawer.draw(frame, metadata))
        finally:
            if writer is not None:
                writer.release()
            self._end_metrics()

        logger.info(
            "Video processing completed path=%s frames=%s faces=%s",
            video_path,
            self.metrics.processed_frames,
            self.metrics.detected_faces,
        )
        return metadata_frames, self.metrics.snapshot(), output_path

    def draw(self, frame: np.ndarray, metadata: FaceDetectionFrame) -> np.ndarray:
        return self.drawer.draw(frame, metadata)

    def current_metrics(self) -> MetricsSnapshot:
        return self.metrics.snapshot()

    def _start_metrics(self, input_fps: float) -> None:
        self.metrics.reset()
        self.metrics.input_fps = input_fps
        self.metrics.processing_started_at = time.perf_counter()

    def _end_metrics(self) -> None:
        self.metrics.processing_ended_at = time.perf_counter()

