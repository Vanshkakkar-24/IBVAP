"""CLI: run face detection on a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.config import get_settings
from app.services.face_service import FaceDetectionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect faces in an image.")
    parser.add_argument("--input", required=True, help="Path to an input image.")
    parser.add_argument("--output", help="Optional annotated output image path.")
    parser.add_argument("--camera-id", default=None, help="Camera/source identifier for metadata.")
    args = parser.parse_args()

    settings = get_settings()
    service = FaceDetectionService(settings)
    metadata = service.process_image_path(args.input, camera_id=args.camera_id)

    print(f"Faces detected: {len(metadata.faces)}")
    print(f"Average confidence: {_average([face.confidence for face in metadata.faces]):.3f}")
    print(f"Average quality: {_average([face.quality.score for face in metadata.faces]):.3f}")
    print(metadata.model_dump_json(indent=2))

    if args.output:
        frame = cv2.imread(args.input)
        annotated = service.draw(frame, metadata)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Annotated image: {output_path}")


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()

