"""CLI: run face detection on an MP4 video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from app.config import get_settings
from app.services.face_service import FaceDetectionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect faces in an MP4 video.")
    parser.add_argument("--input", required=True, help="Path to an MP4 file.")
    parser.add_argument("--camera-id", default=None, help="Camera/source identifier for metadata.")
    parser.add_argument("--no-output", action="store_true", help="Disable annotated MP4 output.")
    parser.add_argument("--show", action="store_true", help="Preview processed frames in a window.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input video not found: {input_path}")
        sys.exit(1)

    service = FaceDetectionService(get_settings())
    print(f"Processing video: {input_path} ...")

    def on_progress(current: int, total: int) -> None:
        pct = (current / total) * 100 if total > 0 else 0.0
        bar_len = 30
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {current}/{total} frames ({pct:.1f}%)")
        sys.stdout.flush()

    try:
        _, metrics, output_video = service.process_video_path(
            input_path,
            camera_id=args.camera_id,
            output_annotated=not args.no_output,
            progress_callback=on_progress,
        )
    except Exception as exc:
        print(f"\n[ERROR] Video processing failed: {exc}")
        sys.exit(1)

    print("\n\n" + "=" * 45)
    print("  VIDEO PROCESSING COMPLETE")
    print("=" * 45)
    print(f"Frames processed:       {metrics.processed_frames}")
    print(f"Faces detected:         {metrics.detected_faces}")
    print(f"Average confidence:     {metrics.average_confidence:.3f}")
    print(f"Average quality score:  {metrics.average_quality_score:.3f}")
    print(f"Processing speed (FPS): {metrics.processing_fps:.2f}")
    if output_video:
        print(f"Annotated video saved:  {output_video}")
    print("=" * 45)

    if args.show and output_video and Path(output_video).exists():
        print("\nPlaying back annotated output (press [q] to stop)...")
        cap = cv2.VideoCapture(str(output_video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        delay = max(1, int(1000 / fps))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Annotated Video Playback", frame)
            if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

