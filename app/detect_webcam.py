"""CLI: run live webcam face detection."""

from __future__ import annotations

import argparse

import cv2

from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.services.face_service import FaceDetectionService
from app.video.sampling import FrameSampler
from app.video.video_file import VideoOpenError
from app.video.webcam import WebcamReader, list_available_cameras


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect faces from a live webcam.")
    parser.add_argument("--camera-index", type=int, default=None, help="OpenCV webcam index (default: 0).")
    parser.add_argument("--camera-id", default=None, help="Camera/source identifier for metadata.")
    parser.add_argument("--no-window", action="store_true", help="Run without cv2.imshow (headless / terminal only).")
    parser.add_argument("--list-cameras", action="store_true", help="Scan and list all detected webcam indices.")
    args = parser.parse_args()

    if args.list_cameras:
        print("Scanning for connected cameras...")
        cams = list_available_cameras()
        if cams:
            print(f"Detected working cameras at indices: {cams}")
        else:
            print("No working webcams found. Check camera connections and Windows privacy permissions.")
        return

    settings = get_settings()
    index = args.camera_index if args.camera_index is not None else settings.camera_index
    camera_id = args.camera_id or settings.camera_id
    service = FaceDetectionService(settings)

    print(f"\nConnecting to webcam index {index}...")
    try:
        reader = WebcamReader(index)
    except VideoOpenError as exc:
        print(f"\n[ERROR] Failed to open webcam: {exc}")
        print("\nTroubleshooting tips:")
        print("  1. Run `python -m app.detect_webcam --list-cameras` to check available indices.")
        print("  2. Ensure no other application (Zoom, MS Teams, Chrome, Camera app) is using the webcam.")
        print("  3. Check Windows Settings -> Privacy & Security -> Camera -> allow apps to access your camera.")
        print("  4. If using an external USB webcam, unplug and plug it back in.")
        return

    sampler = FrameSampler(reader.fps, settings.process_fps)
    enable_vis = settings.enable_visualization
    enable_tracking = settings.enable_tracking
    settings.ensure_storage_dirs()

    print("\n" + "=" * 55)
    print("  LIVE WEBCAM FACE DETECTION RUNNING")
    print("=" * 55)
    print("  Controls in video window:")
    print("    [q] or [ESC] - Quit")
    print("    [s]         - Save snapshot to outputs/")
    print("    [t]         - Toggle Tracking (IoU Tracker)")
    print("    [v]         - Toggle Visualization")
    print("=" * 55 + "\n")

    try:
        for frame_id, frame in reader.frames(lambda: False):
            if not sampler.should_process(frame_id):
                continue
            service.settings.enable_tracking = enable_tracking
            metadata = service.process_frame(
                frame,
                frame_id,
                camera_id=camera_id,
                input_fps=reader.fps,
                source_type="webcam",
            )
            face_count = len(metadata.faces)
            avg_conf = _average([face.confidence for face in metadata.faces])
            avg_qual = _average([face.quality.score for face in metadata.faces])
            fps = metadata.processing_fps or 0.0

            status_line = (
                f"\rFrame {frame_id:05d} | Faces: {face_count:2d} | "
                f"Avg Conf: {avg_conf:.2f} | Avg Quality: {avg_qual:.2f} | FPS: {fps:4.1f}"
            )
            print(status_line, end="", flush=True)

            if not args.no_window and enable_vis:
                annotated = service.draw(frame.copy(), metadata)
                # Add on-screen HUD help
                h, w = annotated.shape[:2]
                cv2.putText(
                    annotated,
                    f"Keys: [q] Quit  [s] Snapshot  [t] Track:{'ON' if enable_tracking else 'OFF'}",
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                try:
                    cv2.imshow("Live Face Detection (YuNet)", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # 'q' or ESC
                        print("\nStopping webcam...")
                        break
                    elif key == ord("s"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        snap_path = settings.output_dir / f"webcam_snapshot_{timestamp}.jpg"
                        cv2.imwrite(str(snap_path), annotated)
                        print(f"\n[SNAPSHOT] Saved to {snap_path}")
                    elif key == ord("t"):
                        enable_tracking = not enable_tracking
                        print(f"\n[TOGGLE] Tracking set to {enable_tracking}")
                    elif key == ord("v"):
                        service.drawer.enabled = not service.drawer.enabled
                        print(f"\n[TOGGLE] Visualization set to {service.drawer.enabled}")
                except cv2.error as cv_err:
                    print(f"\n[WARN] OpenCV GUI error: {cv_err}. Falling back to headless mode.")
                    args.no_window = True
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        reader.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("\nWebcam closed successfully.")


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()

