"""Cloudflare R2 / S3-compatible object storage service with local disk fallback."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.schemas.face import BoundingBox

logger = logging.getLogger(__name__)


class ObjectStorageService:
    """Manages snapshot uploads to Cloudflare R2 / S3 or local disk storage."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_storage_dirs()
        self._s3_client = None
        self._init_client()

    def _init_client(self) -> None:
        if (
            self.settings.r2_account_id
            and self.settings.r2_access_key_id
            and self.settings.r2_secret_access_key
        ):
            try:
                import boto3

                endpoint_url = self.settings.r2_endpoint_url or f"https://{self.settings.r2_account_id}.r2.cloudflarestorage.com"
                self._s3_client = boto3.client(
                    "s3",
                    endpoint_url=endpoint_url,
                    aws_access_key_id=self.settings.r2_access_key_id,
                    aws_secret_access_key=self.settings.r2_secret_access_key,
                    region_name="auto",
                )
                logger.info("Initialized Cloudflare R2 client (bucket=%s)", self.settings.r2_bucket_name)
            except Exception as exc:
                logger.warning("Failed to initialize Cloudflare R2 client: %s; falling back to local storage", exc)
                self._s3_client = None
        else:
            logger.info("R2 credentials not provided; using local disk storage at %s", self.settings.snapshots_dir)

    def upload_bytes(self, data: bytes, object_key: str, content_type: str = "image/jpeg") -> str:
        """Upload binary bytes to R2 or local disk and return the public/accessible URL."""
        if self._s3_client is not None:
            try:
                self._s3_client.put_object(
                    Bucket=self.settings.r2_bucket_name,
                    Key=object_key,
                    Body=data,
                    ContentType=content_type,
                )
                if self.settings.r2_public_url:
                    return f"{self.settings.r2_public_url.rstrip('/')}/{object_key.lstrip('/')}"
                return f"https://{self.settings.r2_bucket_name}.r2.cloudflarestorage.com/{object_key}"
            except Exception as exc:
                logger.warning("R2 upload failed: %s; falling back to local disk", exc)

        # Local storage fallback
        filename = Path(object_key).name
        local_path = self.settings.snapshots_dir / filename
        local_path.write_bytes(data)
        return f"/api/snapshots/{filename}"

    def save_face_snapshot(
        self,
        frame: np.ndarray,
        face_bbox: BoundingBox,
        person_id: str,
        camera_id: str,
    ) -> str:
        """Crop face from frame, encode as JPEG, upload, and return URL."""
        if frame is None or frame.size == 0:
            return ""

        fh, fw = frame.shape[:2]
        x1 = max(0, min(fw - 1, face_bbox.x))
        y1 = max(0, min(fh - 1, face_bbox.y))
        x2 = max(x1 + 1, min(fw, face_bbox.x + face_bbox.width))
        y2 = max(y1 + 1, min(fh, face_bbox.y + face_bbox.height))

        # Add modest padding (15%) for better visual context
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(fw, x2 + pad_x)
        crop_y2 = min(fh, y2 + pad_y)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return ""

        success, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return ""

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_person_id = person_id.replace(" ", "_")
        safe_cam_id = camera_id.replace(" ", "_")
        filename = f"{safe_person_id}_{safe_cam_id}_{timestamp_str}_{uuid4().hex[:6]}.jpg"
        object_key = f"snapshots/{filename}"

        return self.upload_bytes(buffer.tobytes(), object_key)


_storage_service = None


def get_storage_service() -> ObjectStorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = ObjectStorageService()
    return _storage_service
