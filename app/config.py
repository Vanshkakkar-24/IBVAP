"""Environment-backed application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_path: Path = Field(default=Path("models/face_detection_yunet.onnx"), alias="MODEL_PATH")
    confidence_threshold: float = Field(default=0.65, alias="CONFIDENCE_THRESHOLD", ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.30, alias="NMS_THRESHOLD", ge=0.0, le=1.0)
    min_face_width: int = Field(default=30, alias="MIN_FACE_WIDTH", ge=1)
    min_face_height: int = Field(default=30, alias="MIN_FACE_HEIGHT", ge=1)

    process_fps: float = Field(default=10.0, alias="PROCESS_FPS", ge=0.0)
    camera_index: int = Field(default=0, alias="CAMERA_INDEX", ge=0)
    rtsp_url: str | None = Field(default=None, alias="RTSP_URL")
    camera_id: str = Field(default="CAM_01", alias="CAMERA_ID")

    blur_threshold: float = Field(default=100.0, alias="BLUR_THRESHOLD", gt=0.0)
    brightness_min: float = Field(default=50.0, alias="BRIGHTNESS_MIN", ge=0.0, le=255.0)
    brightness_max: float = Field(default=210.0, alias="BRIGHTNESS_MAX", ge=0.0, le=255.0)
    size_score_reference_area: int = Field(default=120 * 120, alias="SIZE_SCORE_REFERENCE_AREA", ge=1)
    quality_size_weight: float = Field(default=0.40, alias="QUALITY_SIZE_WEIGHT", ge=0.0)
    quality_blur_weight: float = Field(default=0.35, alias="QUALITY_BLUR_WEIGHT", ge=0.0)
    quality_brightness_weight: float = Field(default=0.25, alias="QUALITY_BRIGHTNESS_WEIGHT", ge=0.0)

    enable_tracking: bool = Field(default=True, alias="ENABLE_TRACKING")
    enable_visualization: bool = Field(default=True, alias="ENABLE_VISUALIZATION")
    detector_backend: Literal["opencv", "custom", "yunet"] = Field(default="opencv", alias="DETECTOR_BACKEND")
    execution_provider: Literal["cpu", "gpu"] = Field(default="cpu", alias="EXECUTION_PROVIDER")

    upload_dir: Path = Field(default=Path("uploads"), alias="UPLOAD_DIR")
    output_dir: Path = Field(default=Path("outputs"), alias="OUTPUT_DIR")
    max_upload_mb: int = Field(default=500, alias="MAX_UPLOAD_MB", ge=1)
    allowed_cors_origins: list[str] = Field(default_factory=list, alias="ALLOWED_CORS_ORIGINS")
    rtsp_reconnect_seconds: float = Field(default=5.0, alias="RTSP_RECONNECT_SECONDS", ge=0.5)

    @field_validator("allowed_cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return value
        raise TypeError("ALLOWED_CORS_ORIGINS must be a comma-separated string or list")

    @field_validator("brightness_max")
    @classmethod
    def validate_brightness_range(cls, value: float, info):
        min_value = info.data.get("brightness_min", 0)
        if value <= min_value:
            raise ValueError("BRIGHTNESS_MAX must be greater than BRIGHTNESS_MIN")
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_storage_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

