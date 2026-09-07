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


class FaceDetectionFrame(BaseModel):
    camera_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frame_id: int = Field(ge=0)
    faces: list[FaceDetection] = Field(default_factory=list)
    input_fps: float | None = None
    processing_fps: float | None = None
    source_type: str | None = None


class MetricsSnapshot(BaseModel):
    input_fps: float = 0.0
    processing_fps: float = 0.0
    average_inference_ms: float = 0.0
    average_latency_ms: float = 0.0
    processed_frames: int = 0
    detected_faces: int = 0
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

