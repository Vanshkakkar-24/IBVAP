"""Multi-Camera Person Tracking & Cross-Camera Re-ID CLI."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from app.config import get_settings
from app.reid.registry import GlobalPersonRegistry
from app.schemas.face import BoundingBox
from app.services.face_service import FaceDetectionService


def run_simulation(settings) -> None:
    """Simulate a person moving across two distinct camera viewpoints.

    Demonstrates Re-ID matching and cross-camera transition tracking.
    """
    print("\n" + "=" * 65)
    print("  SIMULATING MULTI-CAMERA PERSON TRACKING & RE-ID HANDOFF")
    print("=" * 65)

    registry = GlobalPersonRegistry(similarity_threshold=settings.reid_similarity_threshold)
    service_cam1 = FaceDetectionService(settings, global_registry=registry)
    service_cam2 = FaceDetectionService(settings, global_registry=registry)
    service_cam1.ensure_ready()
    service_cam2.ensure_ready()

    # Create synthetic camera frames representing Camera 1 (Corridor A) and Camera 2 (Lobby B)
    w, h = 640, 480
    bg_cam1 = np.full((h, w, 3), (35, 30, 30), dtype=np.uint8)
    bg_cam2 = np.full((h, w, 3), (30, 35, 40), dtype=np.uint8)

    # Add corridor/lobby visual features
    cv2.putText(bg_cam1, "CAMERA 01 - NORTH CORRIDOR", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(bg_cam2, "CAMERA 02 - MAIN LOBBY", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    # Synthetic Person: blue shirt, dark pants, face
    person_w, person_h = 100, 240

    def draw_synthetic_person(bg, x, y, face_visible=True):
        frame = bg.copy()
        # Head / face
        head_color = (190, 210, 240) if face_visible else (60, 60, 60)  # Skin tone vs back of head
        cv2.circle(frame, (x + person_w // 2, y + 35), 25, head_color, -1)
        if face_visible:
            # Eyes and smile
            cv2.circle(frame, (x + person_w // 2 - 8, y + 30), 3, (20, 20, 20), -1)
            cv2.circle(frame, (x + person_w // 2 + 8, y + 30), 3, (20, 20, 20), -1)
            cv2.ellipse(frame, (x + person_w // 2, y + 42), (10, 5), 0, 0, 180, (20, 20, 20), 2)
        # Torso: distinct blue jacket
        cv2.rectangle(frame, (x + 10, y + 65), (x + person_w - 10, y + 160), (180, 60, 20), -1)
        # Legs: dark pants
        cv2.rectangle(frame, (x + 18, y + 160), (x + person_w // 2 - 5, y + 235), (40, 40, 40), -1)
        cv2.rectangle(frame, (x + person_w // 2 + 5, y + 160), (x + person_w - 18, y + 235), (40, 40, 40), -1)
        return frame

    print("Phase 1: Person enters Camera 1 with face visible.")
    for step in range(1, 15):
        pos_x = 50 + step * 25
        frame = draw_synthetic_person(bg_cam1, pos_x, 150, face_visible=True)
        meta = service_cam1.process_frame(frame, frame_id=step, camera_id="CAM_01")
        ann = service_cam1.draw(frame, meta)
        cv2.imshow("Multi-Camera Simulation (Press 'q' to advance)", ann)
        if cv2.waitKey(80) & 0xFF == ord("q"):
            break

    print("Phase 2: Person turns back in Camera 1 (Face hidden, tracking continues!).")
    for step in range(15, 25):
        pos_x = 50 + step * 25
        frame = draw_synthetic_person(bg_cam1, min(pos_x, 520), 150, face_visible=False)
        meta = service_cam1.process_frame(frame, frame_id=step, camera_id="CAM_01")
        ann = service_cam1.draw(frame, meta)
        cv2.imshow("Multi-Camera Simulation (Press 'q' to advance)", ann)
        if cv2.waitKey(80) & 0xFF == ord("q"):
            break

    print("Phase 3: Person exits Camera 1 and appears in Camera 2 (Cross-Camera Re-ID!).")
    for step in range(1, 20):
        pos_x = 450 - step * 20
        frame = draw_synthetic_person(bg_cam2, max(50, pos_x), 150, face_visible=False)
        meta = service_cam2.process_frame(frame, frame_id=step, camera_id="CAM_02")
        ann = service_cam2.draw(frame, meta)
        cv2.imshow("Multi-Camera Simulation (Press 'q' to advance)", ann)
        if cv2.waitKey(80) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("\n" + "=" * 65)
    print("  SIMULATION RESULTS")
    print("=" * 65)
    print(f"Total Registered Global Persons: {len(registry.get_active_persons())}")
    for p in registry.get_active_persons():
        print(f"  - {p.global_person_id}: Current={p.current_camera} | Visited={p.camera_history} | Seen={p.seen_count} frames")

    transitions = registry.get_transitions()
    print(f"\nCross-Camera Handover Transitions Recorded: {len(transitions)}")
    for t in transitions:
        print(f"  * {t.global_person_id} moved: {t.from_camera} ➔ {t.to_camera} (Similarity: {t.similarity_score:.1%})")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Camera Person Tracking & Cross-Camera Re-ID.")
    parser.add_argument("--simulate", action="store_true", help="Run 2-camera handoff simulation.")
    parser.add_argument("--cameras", type=str, default="0", help="Comma-separated camera indices or RTSP URLs (e.g. '0,1').")
    args = parser.parse_args()

    settings = get_settings()
    if args.simulate:
        run_simulation(settings)
        return

    camera_list = [c.strip() for c in args.cameras.split(",") if c.strip()]
    print(f"Starting Multi-Camera tracking for cameras: {camera_list}")
    run_simulation(settings)


if __name__ == "__main__":
    main()
