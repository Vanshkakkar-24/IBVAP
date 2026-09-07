"""Face detection dataset management, synthetic generator, and webcam data capture."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    """PyTorch Dataset loading face detection samples."""

    def __init__(self, samples: list[dict[str, Any]], image_size: int = 128) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        image_path = item["image_path"]
        img = cv2.imread(str(image_path))
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and CHW format
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0

        label = torch.tensor([float(item.get("label", 1.0))], dtype=torch.float32)
        bbox = torch.tensor(item.get("bbox", [0.0, 0.0, 0.0, 0.0]), dtype=torch.float32)
        landmarks = torch.tensor(item.get("landmarks", [0.0] * 10), dtype=torch.float32)

        return img_tensor, label, bbox, landmarks


def generate_synthetic_dataset(output_dir: Path | str, num_samples: int = 120) -> list[dict[str, Any]]:
    """Generate a clean synthetic face dataset with realistic facial geometries and negative samples."""
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    width, height = 256, 256

    skin_tones = [
        (180, 200, 240),  # Light skin BGR
        (140, 175, 225),  # Medium skin BGR
        (100, 130, 190),  # Olive skin BGR
        (60, 90, 140),    # Brown skin BGR
        (40, 60, 100),    # Dark skin BGR
    ]

    for i in range(num_samples):
        img_filename = f"synth_{i:04d}.jpg"
        img_filepath = images_dir / img_filename

        # 80% positive face samples, 20% negative non-face background samples
        is_face = (i % 5) != 0

        # Background with subtle gradient/color
        bg_color = [random.randint(20, 180) for _ in range(3)]
        canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Add background noise
        noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if is_face:
            # Random face center and dimensions
            cx = random.randint(80, width - 80)
            cy = random.randint(80, height - 80)
            rx = random.randint(30, 50)
            ry = int(rx * random.uniform(1.2, 1.4))

            skin = random.choice(skin_tones)
            # Draw head ellipse
            cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, skin, -1, cv2.LINE_AA)

            # Left & Right eyes
            eye_y = cy - int(ry * 0.2)
            left_eye_x = cx - int(rx * 0.38)
            right_eye_x = cx + int(rx * 0.38)
            eye_r = max(2, int(rx * 0.12))
            cv2.circle(canvas, (left_eye_x, eye_y), eye_r, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.circle(canvas, (right_eye_x, eye_y), eye_r, (40, 40, 40), -1, cv2.LINE_AA)

            # Nose
            nose_x = cx
            nose_y = cy + int(ry * 0.1)
            cv2.line(canvas, (nose_x, nose_y - 4), (nose_x, nose_y + 4), (skin[0]//2, skin[1]//2, skin[2]//2), 2)

            # Mouth
            mouth_y = cy + int(ry * 0.45)
            mouth_w = int(rx * 0.45)
            left_mouth = (cx - mouth_w, mouth_y)
            right_mouth = (cx + mouth_w, mouth_y)
            cv2.line(canvas, left_mouth, right_mouth, (50, 50, 180), 3, cv2.LINE_AA)

            # Calculate bounding box [x, y, w, h] normalized to [0, 1]
            bx = max(0, cx - rx)
            by = max(0, cy - ry)
            bw = min(width - bx, rx * 2)
            bh = min(height - by, ry * 2)

            norm_bbox = [bx / width, by / height, bw / width, bh / height]
            norm_landmarks = [
                left_eye_x / width, eye_y / height,
                right_eye_x / width, eye_y / height,
                nose_x / width, nose_y / height,
                left_mouth[0] / width, left_mouth[1] / height,
                right_mouth[0] / width, right_mouth[1] / height,
            ]
            label = 1.0
        else:
            # Negative sample (random geometric shapes without a face)
            for _ in range(random.randint(2, 5)):
                pt1 = (random.randint(0, width), random.randint(0, height))
                pt2 = (random.randint(0, width), random.randint(0, height))
                cv2.rectangle(canvas, pt1, pt2, (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)), -1)
            norm_bbox = [0.0, 0.0, 0.0, 0.0]
            norm_landmarks = [0.0] * 10
            label = 0.0

        cv2.imwrite(str(img_filepath), canvas)
        samples.append({
            "image_path": str(img_filepath),
            "label": label,
            "bbox": norm_bbox,
            "landmarks": norm_landmarks,
        })

    # Save manifest
    manifest_path = out_path / "annotations.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    return samples


def capture_webcam_dataset(
    output_dir: Path | str,
    num_samples: int = 50,
    camera_index: int = 0,
) -> list[dict[str, Any]]:
    """Capture real training face samples directly from the physical webcam."""
    from app.config import get_settings
    from app.services.face_service import FaceDetectionService
    from app.video.webcam import WebcamReader

    out_path = Path(output_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    service = FaceDetectionService(settings)
    reader = WebcamReader(camera_index)

    samples: list[dict[str, Any]] = []
    print(f"\n[CAPTURE] Starting webcam data capture from camera index {camera_index}...")
    print(f"[CAPTURE] Target: {num_samples} face samples. Look at the camera and move slightly...")

    collected = 0
    try:
        for frame_id, frame in reader.frames(lambda: False):
            if frame is None:
                continue

            metadata = service.process_frame(frame, frame_id, source_type="webcam")
            if metadata.faces:
                face = metadata.faces[0]
                h, w = frame.shape[:2]
                bx, by, bw, bh = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height

                # Ensure valid dimensions
                if bw > 30 and bh > 30:
                    img_filename = f"webcam_face_{collected:04d}.jpg"
                    img_filepath = images_dir / img_filename
                    cv2.imwrite(str(img_filepath), frame)

                    landmarks = [0.0] * 10
                    if face.landmarks and len(face.landmarks) == 5:
                        flat_lm = []
                        for pt in face.landmarks:
                            flat_lm.extend([pt[0] / w, pt[1] / h])
                        landmarks = flat_lm

                    samples.append({
                        "image_path": str(img_filepath),
                        "label": 1.0,
                        "bbox": [bx / w, by / h, bw / w, bh / h],
                        "landmarks": landmarks,
                    })
                    collected += 1
                    print(f"\r[CAPTURE] Captured {collected}/{num_samples} frames...", end="", flush=True)

                    if collected >= num_samples:
                        break
            cv2.waitKey(100)
    finally:
        reader.release()

    manifest_path = out_path / "annotations.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"\n[CAPTURE] Completed! Saved {len(samples)} annotated samples to {out_path}")
    return samples
