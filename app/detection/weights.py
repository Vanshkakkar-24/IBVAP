"""Utilities for downloading and managing detection model weights."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

YOLO_V8N_ONNX_URLS = [
    "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx",
]


def ensure_person_model(model_path: Path | str = Path("models/yolov8n.onnx")) -> bool:
    """Ensure that the person detection ONNX model is present.

    Attempts downloading if not present. Returns True if file exists or was
    successfully downloaded, False otherwise.
    """
    target = Path(model_path)
    if target.exists() and target.stat().st_size > 1_000_000:
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(".tmp")

    for url in YOLO_V8N_ONNX_URLS:
        try:
            logger.info("Downloading person detection model from %s ...", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=30) as response, open(temp_target, "wb") as out_file:
                chunk_size = 64 * 1024
                while chunk := response.read(chunk_size):
                    out_file.write(chunk)

            if temp_target.stat().st_size > 1_000_000:
                temp_target.replace(target)
                logger.info("Successfully downloaded person detection model to %s", target)
                return True
            else:
                temp_target.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to download model from %s: %s", url, exc)
            temp_target.unlink(missing_ok=True)

    return False
