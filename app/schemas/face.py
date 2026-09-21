"""Pydantic schemas used by API responses and WebSocket payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height


class FaceQuality(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    blur: float = Field(ge=0.0)
    brightness: float = Field(ge=0.0, le=255.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    area: int = Field(gt=0)
    size_score: float = Field(ge=0.0, le=1.0)
    blur_score: float = Field(ge=0.0, le=1.0)
    brightness_score: float = Field(ge=0.0, le=1.0)


class FaceDetection(BaseModel):
    face_id: str
    track_id: int | None = None
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    quality: FaceQuality
    landmarks: list[list[float]] | None = None


class PersonDetection(BaseModel):
    person_id: str
    track_id: int
    global_person_id: str | None = None
    bbox: BoundingBox
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    has_face: bool = False
    face: FaceDetection | None = None
    reid_embedding: list[float] | None = None


class FaceDetectionFrame(BaseModel):
    camera_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frame_id: int = Field(ge=0)
    faces: list[FaceDetection] = Field(default_factory=list)
    persons: list[PersonDetection] = Field(default_factory=list)
    input_fps: float | None = None
    processing_fps: float | None = None
    source_type: str | None = None


class CrossCameraTransition(BaseModel):
    global_person_id: str
    from_camera: str
    to_camera: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    similarity_score: float = 0.0


class GlobalPersonInfo(BaseModel):
    global_person_id: str
    current_camera: str | None = None
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    seen_count: int = 1
    has_face: bool = False
    camera_history: list[str] = Field(default_factory=list)


class CameraStreamConfig(BaseModel):
    camera_id: str
    source_type: str = "webcam"
    source_url_or_index: str = "0"
    enabled: bool = True


class MetricsSnapshot(BaseModel):
    input_fps: float = 0.0
    processing_fps: float = 0.0
    average_inference_ms: float = 0.0
    average_latency_ms: float = 0.0
    processed_frames: int = 0
    detected_faces: int = 0
    detected_persons: int = 0
    average_confidence: float = 0.0
    average_quality_score: float = 0.0
    execution_mode: str = "cpu"
    detector: str = "YuNet"


class StatusResponse(BaseModel):
    service: str = "face-detection"
    running_source: str | None = None
    model_path: str
    model_present: bool
    tracking_enabled: bool
    visualization_enabled: bool
    details: dict[str, Any] = Field(default_factory=dict)


class StartCameraRequest(BaseModel):
    camera_index: int | None = Field(default=None, ge=0)
    camera_id: str | None = None


class StartRtspRequest(BaseModel):
    rtsp_url: str | None = None
    camera_id: str | None = None


class CommandResponse(BaseModel):
    ok: bool
    message: str


class UploadResponse(BaseModel):
    ok: bool
    message: str
    metadata: list[FaceDetectionFrame] = Field(default_factory=list)
    metrics: MetricsSnapshot
    output_video: str | None = None
    video_url: str | None = None
    download_url: str | None = None


class DetectFrameRequest(BaseModel):
    image_base64: str
    camera_id: str | None = None
    return_annotated: bool = True


class DetectFrameResponse(BaseModel):
    ok: bool
    metadata: FaceDetectionFrame
    annotated_image_base64: str | None = None


class CameraCreateRequest(BaseModel):
    camera_id: str
    name: str
    type: str = "webcam"  # webcam, rtsp, phone, video
    stream_url: str = "0"
    location: str = "Default"
    enabled: bool = True
    start_immediately: bool = True


class CameraUpdateRequest(BaseModel):
    name: str | None = None
    type: str | None = None
    stream_url: str | None = None
    location: str | None = None
    enabled: bool | None = None


class CameraResponse(BaseModel):
    camera_id: str
    name: str
    type: str
    stream_url: str
    location: str
    status: str
    enabled: bool
    fps: float = 0.0
    detection_count: int = 0


class PersonResponse(BaseModel):
    id: int | None = None
    global_person_id: str
    first_seen: str | None = None
    last_seen: str | None = None
    status: str = "active"


class PersonHistoryResponse(BaseModel):
    person: dict[str, Any]
    tracks: list[dict[str, Any]] = Field(default_factory=list)
    movements: list[dict[str, Any]] = Field(default_factory=list)
    snapshots: list[dict[str, Any]] = Field(default_factory=list)


class EventResponse(BaseModel):
    id: int
    person_id: str | None = None
    camera_id: str | None = None
    event_type: str
    severity: str = "info"
    timestamp: str | None = None
    metadata: dict[str, Any] | None = None


class FaceSnapshotResponse(BaseModel):
    id: int
    person_id: str
    camera_id: str
    track_id: int
    image_url: str
    timestamp: str | None = None
    confidence: float

