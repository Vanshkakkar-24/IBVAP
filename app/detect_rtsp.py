"""CLI: run live RTSP face detection."""

from __future__ import annotations

import argparse

import cv2

from app.config import get_settings
from app.services.face_service import FaceDetectionService
from app.video.rtsp import RtspReader
from app.video.sampling import FrameSampler


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect faces from an RTSP stream.")
    parser.add_argument("--url", default=None, help="RTSP URL. Defaults to RTSP_URL from .env.")
    parser.add_argument("--camera-id", default=None, help="Camera/source identifier for metadata.")
    parser.add_argument("--no-window", action="store_true", help="Run without cv2.imshow.")
    args = parser.parse_args()

    settings = get_settings()
    url = args.url or settings.rtsp_url
    if not url:
        raise SystemExit("RTSP URL required. Pass --url or set RTSP_URL.")

    service = FaceDetectionService(settings)
    reader = RtspReader(url, settings.rtsp_reconnect_seconds)
    sampler = FrameSampler(reader.fps, settings.process_fps)
    camera_id = args.camera_id or settings.camera_id

    print("Press q in the video window or Ctrl+C in the terminal to stop.")
    try:
        for frame_id, frame in reader.frames(lambda: False):
            if not sampler.should_process(frame_id):
                continue
            metadata = service.process_frame(frame, frame_id, camera_id=camera_id, input_fps=reader.fps, source_type="rtsp")
            print(f"frame={frame_id} faces={len(metadata.faces)} fps={(metadata.processing_fps or 0):.2f}")
            if not args.no_window and settings.enable_visualization:
                cv2.imshow("RTSP Face Detection", service.draw(frame, metadata))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

