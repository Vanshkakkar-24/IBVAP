"""CLI: benchmark the face detection pipeline on an MP4."""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.services.face_service import FaceDetectionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark face detection on a video.")
    parser.add_argument("--input", required=True, help="Path to an MP4 file.")
    parser.add_argument("--no-output", action="store_true", help="Disable annotated output during benchmark.")
    args = parser.parse_args()

    settings = get_settings()
    service = FaceDetectionService(settings)
    _, metrics, _ = service.process_video_path(args.input, output_annotated=not args.no_output)

    print("Benchmark results")
    print(f"Input FPS: {metrics.input_fps:.2f}")
    print(f"Processed frames: {metrics.processed_frames}")
    print(f"Detected faces: {metrics.detected_faces}")
    print(f"Average FPS: {metrics.processing_fps:.2f}")
    print(f"Average inference time: {metrics.average_inference_ms:.2f} ms")
    print(f"Average latency: {metrics.average_latency_ms:.2f} ms")
    print(f"Average confidence: {metrics.average_confidence:.3f}")
    print(f"Average quality: {metrics.average_quality_score:.3f}")
    print(f"Execution mode: {metrics.execution_mode}")
    print(f"Detector: {metrics.detector}")


if __name__ == "__main__":
    main()

