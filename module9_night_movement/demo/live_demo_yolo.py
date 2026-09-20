"""
live_demo_yolo.py -- Module 9 demo using REAL YOLO person/vehicle
detection + ByteTrack tracking (same approach as Module 8's
virtual-fence main.py), wired into the REAL night_movement package
from this repo.

This replaces the earlier background-subtraction stand-in
(live_demo.py) with actual AI detection, so quality matches Module 8.
Architecture-wise this script plays the role of Module 5 (tracking) +
Module 8 (zone check) just so you have something live to demo -- the
night/movement/severity decision itself is 100% night_movement code,
unchanged from what's already unit-tested.

-------------------------------------------------------------------
SETUP (run once, in the module9 folder):

    pip install -r requirements.txt
    pip install ultralytics torch opencv-python numpy

    (first run will auto-download yolo11n.pt, ~5-6 MB, needs internet)

-------------------------------------------------------------------
RUN:

    python demo/live_demo_yolo.py --video path/to/clip.mp4 --camera-id CAM01 --mode night

    # or live webcam / phone camera:
    python demo/live_demo_yolo.py --video 0 --camera-id CAM01 --mode night

CONTROLS (same as Module 8's main.py, for consistency across your team's demos):
    R  -> start drawing a restricted zone (click points on screen)
    F  -> finish the polygon (needs 3+ points)
    C  -> clear the zone
    S  -> save the zone to demo/zone_<camera-id>.json (auto-loaded next run)
    Q  -> quit

BOX COLOURS:
    GREEN  -> tracked, not near the restricted zone
    YELLOW -> APPROACHING the restricted zone (direction math, like Module 8)
    RED    -> INSIDE the restricted zone

Whenever night_movement.NightMovementDetector actually fires an event
(night-check + movement-threshold + cooldown/escalation all passed for
real), it's printed to the terminal, shown as a red banner, and saved
to demo/alerts_<camera-id>.jsonl -- same as live_demo.py.
-------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from night_movement import (  # noqa: E402
    NightMovementDetector, NightScheduleConfig, TrackUpdate, prepare_detection_frame,
)

GREEN = (0, 200, 0)
YELLOW = (0, 200, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# COCO class ids we care about; everything else is ignored.
PERSON_CLASS = 0
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

DIRECTION_HISTORY_FRAMES = 8
MIN_DIRECTION_MOVEMENT = 5.0

_BASE_DT = {
    "night": datetime(2026, 1, 1, 22, 0, 0),
    "day": datetime(2026, 1, 1, 13, 0, 0),
}


def get_direction(point, history, zone) -> str:
    """Same idea as Module 8's get_border_direction: is this track
    moving toward the restricted zone's boundary, away, or sideways?
    Used only for the YELLOW "approaching" visual cue."""
    if zone is None or len(zone) < 3 or len(history) < 2:
        return "UNKNOWN"

    prev = history[0]
    px, py = point
    dx, dy = px - prev[0], py - prev[1]
    dist = np.hypot(dx, dy)
    if dist < MIN_DIRECTION_MOVEMENT:
        return "STATIONARY"

    contour = np.array(zone, dtype=np.float32)
    min_d, nearest = float("inf"), None
    for i in range(len(contour)):
        p1, p2 = contour[i], contour[(i + 1) % len(contour)]
        line = p2 - p1
        ll2 = np.dot(line, line)
        closest = p1 if ll2 == 0 else p1 + max(0, min(1, np.dot(np.array([px, py]) - p1, line) / ll2)) * line
        d = np.linalg.norm(np.array([px, py]) - closest)
        if d < min_d:
            min_d, nearest = d, closest
    if nearest is None:
        return "UNKNOWN"

    bx, by = nearest[0] - px, nearest[1] - py
    bdist = np.hypot(bx, by)
    if bdist == 0:
        return "AT_BOUNDARY"

    dx, dy, bx, by = dx / dist, dy / dist, bx / bdist, by / bdist
    dot = dx * bx + dy * by
    if dot > 0.5:
        return "APPROACHING"
    if dot < -0.5:
        return "AWAY"
    return "SIDEWAYS"


def load_zone(path: Path):
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        pts = data.get("points", [])
        return pts if len(pts) >= 3 else []
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Module 9 demo with real YOLO detection.")
    ap.add_argument("--video", required=True, help="Path to a clip, or 0 for webcam.")
    ap.add_argument("--camera-id", default="CAM01")
    ap.add_argument("--mode", default="auto", choices=["auto", "night", "day"],
                    help="auto uses the real clock plus measured darkness; night/day are demo overrides.")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--movement-threshold", type=float, default=20.0)
    ap.add_argument("--cooldown", type=float, default=8.0)
    ap.add_argument("--output", default=None, help="Save annotated video here (mp4).")
    ap.add_argument("--zone-file", default=None, help="Defaults to demo/zone_<camera-id>.json")
    ap.add_argument("--low-light", action=argparse.BooleanOptionalAction, default=True,
                    help="Enhance frames automatically only when the scene is dark (default: on).")
    ap.add_argument("--dark-threshold", type=float, default=72.0,
                    help="Mean luminance below which low-light enhancement runs (0-255).")
    ap.add_argument("--gamma", type=float, default=1.35, help="Dark-frame brightness lift; 1.0 disables lift.")
    args = ap.parse_args()

    zone_path = Path(args.zone_file or f"demo/zone_{args.camera_id}.json")
    zone_path.parent.mkdir(parents=True, exist_ok=True)
    restricted_zone = load_zone(zone_path)
    drawing = False

    def on_mouse(event, x, y, flags, param):
        nonlocal restricted_zone
        if event == cv2.EVENT_LBUTTONDOWN and drawing:
            restricted_zone.append([x, y])
            print(f"Zone point added: ({x}, {y})")

    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Could not open video source: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Loading YOLO model (first run downloads weights, needs internet)...")
    model = YOLO(args.model)

    config = NightScheduleConfig(
        camera_id=args.camera_id,
        watched_object_types=["person", "vehicle"],
        movement_pixel_threshold=args.movement_threshold,
        cooldown_s=args.cooldown,
        min_confidence=args.confidence,
    )
    detector = NightMovementDetector(config)

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    alerts_path = Path(f"demo/alerts_{args.camera_id}.jsonl")
    alerts_file = open(alerts_path, "w")

    window = "Module 9 (real YOLO detection)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    print("\nControls: R=start zone  F=finish  C=clear  S=save  Q=quit\n")

    position_history: dict[int, list] = {}
    base_dt = _BASE_DT.get(args.mode)
    frame_idx = 0
    n_alerts = 0
    banner_text, banner_ttl = None, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        inference_frame, enhanced, luminance = prepare_detection_frame(
            frame, enabled=args.low_light, dark_threshold=args.dark_threshold, gamma=args.gamma,
        )
        results = model.track(inference_frame, persist=True, tracker="bytetrack.yaml",
                               classes=[PERSON_CLASS, *VEHICLE_CLASSES],
                               conf=args.confidence, verbose=False)

        ts = (datetime.now() if base_dt is None else base_dt + timedelta(seconds=frame_idx / fps))

        for result in results:
            if result.boxes.id is None:
                continue
            track_ids = result.boxes.id.int().cpu().tolist()
            coords = result.boxes.xyxy.int().cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()

            for tid, box, conf, cls in zip(track_ids, coords, confs, classes):
                x1, y1, x2, y2 = box
                object_type = "person" if cls == PERSON_CLASS else "vehicle"
                ground_point = ((x1 + x2) / 2, y2)

                history = position_history.setdefault(tid, [])
                history.append(ground_point)
                if len(history) > DIRECTION_HISTORY_FRAMES:
                    history.pop(0)

                zone_breach = False
                if len(restricted_zone) >= 3:
                    zone_arr = np.array(restricted_zone, dtype=np.int32)
                    zone_breach = cv2.pointPolygonTest(zone_arr, ground_point, False) >= 0

                direction = get_direction(ground_point, history, restricted_zone if len(restricted_zone) >= 3 else None)
                approaching = direction == "APPROACHING" and not zone_breach

                event = detector.process(TrackUpdate(
                    camera_id=args.camera_id,
                    track_id=tid,
                    object_type=object_type,
                    timestamp=ts,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    zone_id="RESTRICTED" if zone_breach else None,
                    zone_breach=zone_breach,
                    scene_is_dark=enhanced,
                ))

                if event is not None:
                    n_alerts += 1
                    line = json.dumps(event.to_dict())
                    alerts_file.write(line + "\n")
                    alerts_file.flush()
                    print(f"[ALERT] {line}")
                    banner_text = f"ALERT: {event.event_type} | {event.severity} | track {tid}"
                    banner_ttl = int(fps * 2)

                if zone_breach:
                    color, label = RED, "INTRUSION"
                elif approaching:
                    color, label = YELLOW, "APPROACHING ZONE"
                else:
                    color, label = GREEN, object_type

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"#{tid} {label}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if len(restricted_zone) > 0:
            pts = np.array(restricted_zone, dtype=np.int32)
            for p in restricted_zone:
                cv2.circle(frame, tuple(p), 5, RED, -1)
            if len(pts) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], RED)
                frame = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)
                cv2.polylines(frame, [pts], True, RED, 2)

        if banner_ttl > 0:
            cv2.rectangle(frame, (0, 0), (width, 36), RED, -1)
            cv2.putText(frame, banner_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)
            banner_ttl -= 1

        cv2.putText(frame, "R:zone F:finish C:clear S:save Q:quit", (10, height - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        light_state = "ENHANCED" if enhanced else "NORMAL"
        cv2.putText(frame, f"{light_state} light={luminance:.0f}  mode={args.mode}  alerts={n_alerts}",
                    (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        if writer is not None:
            writer.write(frame)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            restricted_zone = []
            drawing = True
            print("Drawing restricted zone... click points, then press F.")
        elif key == ord("f"):
            if len(restricted_zone) >= 3:
                drawing = False
                print("Zone finished.")
            else:
                print("Need at least 3 points.")
        elif key == ord("c"):
            restricted_zone = []
            drawing = False
            print("Zone cleared.")
        elif key == ord("s"):
            with open(zone_path, "w") as f:
                json.dump({"camera_id": args.camera_id, "points": restricted_zone}, f, indent=2)
            print(f"Zone saved to {zone_path}")
        elif key == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    alerts_file.close()

    print(f"\nDone. {frame_idx} frames processed, {n_alerts} Module 9 alert(s) fired.")
    if args.output:
        print(f"Annotated video saved to: {args.output}")
    print(f"Alerts saved to: {alerts_path}")
    print(f"Zone saved to: {zone_path}")


if __name__ == "__main__":
    main()
