"""
live_demo.py -- Module 9 visual demo.

Takes an uploaded night-camera clip (or a live webcam feed), detects
moving objects, checks them against a restricted zone you mark with
mark_zone.py, and feeds every detection through the REAL
night_movement.NightMovementDetector from this package -- the same
code that is already unit-tested and will plug into Module 5/8/12.

What this script adds on top of night_movement (since Module 3/4/5
-- real AI detection + tracking -- aren't built by other teammates
yet) is a lightweight stand-in "detector + tracker" using OpenCV
background subtraction, purely so you have something to feed
TrackUpdate objects with for a live/video demo. Swap this stand-in for
the real Module 5 output later; night_movement itself does not change.

-------------------------------------------------------------------
BOX COLOURS (visual only, computed from geometry -- independent of
night_movement's own thresholds, just to make "approaching" visible):
    GREEN  -> object moving, but not near the restricted zone
    YELLOW -> object is APPROACHING the restricted zone
    RED    -> object is INSIDE the restricted zone (intrusion)

ALERT BANNER (top of frame, red) -> shown only when
night_movement.NightMovementDetector.process() actually returns a
NightMovementEvent (i.e. it passed night-check + movement-threshold +
cooldown/escalation logic for real). These are also printed to the
terminal and saved to demo/alerts_<camera_id>.jsonl.
-------------------------------------------------------------------

USAGE (run from inside the `module9` folder):

    # 1. install deps once
    pip install -r requirements.txt
    pip install opencv-python

    # 2. (optional) mark a restricted zone on your clip
    python demo/mark_zone.py --video path/to/clip.mp4 --camera-id CAM01 --out demo/zone_CAM01.json

    # 3. run the live demo on an uploaded clip, treating it as night footage
    python demo/live_demo.py --video path/to/clip.mp4 --camera-id CAM01 \
        --zone demo/zone_CAM01.json --mode night --live --output demo/out_CAM01.mp4

    # or straight off your webcam:
    python demo/live_demo.py --video 0 --camera-id CAM01 --zone demo/zone_CAM01.json --mode night --live

Press 'q' in the preview window to stop early.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from night_movement import NightMovementDetector, NightScheduleConfig  # noqa: E402

GREEN = (0, 200, 0)
YELLOW = (0, 200, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)

_BASE_DT = {
    "night": datetime(2026, 1, 1, 22, 0, 0),   # 10 PM -- inside default 18-06 night window
    "day": datetime(2026, 1, 1, 13, 0, 0),     # 1 PM -- outside it, for a "no false alerts" contrast demo
}


class SimpleTrack:
    """One tracked blob: just a centroid history, an ID, and an age."""

    def __init__(self, track_id: int, centroid, bbox):
        self.id = track_id
        self.centroid = centroid
        self.bbox = bbox
        self.age_frames = 0
        self.missed_frames = 0
        self.dist_history = deque(maxlen=15)  # signed distance to zone, for "approaching" logic


class SimpleTracker:
    """Greedy nearest-centroid tracker -- good enough to give stable IDs
    across frames for a demo. Not a substitute for Module 5's real tracker."""

    def __init__(self, max_distance: float = 80.0, max_missed: int = 10):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks: dict[int, SimpleTrack] = {}
        self._next_id = 1

    def update(self, detections: list[tuple[int, int, int, int]]) -> dict[int, SimpleTrack]:
        centroids = [((x1 + x2) // 2, y2) for (x1, y1, x2, y2) in detections]  # bottom-center

        unmatched_dets = set(range(len(detections)))
        for tid, tr in list(self.tracks.items()):
            best_j, best_d = None, self.max_distance
            for j in unmatched_dets:
                d = np.hypot(tr.centroid[0] - centroids[j][0], tr.centroid[1] - centroids[j][1])
                if d < best_d:
                    best_j, best_d = j, d
            if best_j is not None:
                tr.centroid = centroids[best_j]
                tr.bbox = detections[best_j]
                tr.age_frames += 1
                tr.missed_frames = 0
                unmatched_dets.discard(best_j)
            else:
                tr.missed_frames += 1

        for j in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = SimpleTrack(tid, centroids[j], detections[j])

        self.tracks = {tid: tr for tid, tr in self.tracks.items() if tr.missed_frames <= self.max_missed}
        return self.tracks


def load_zone(path: str | None):
    if not path:
        return None
    with open(path) as f:
        data = json.load(f)
    return np.array(data["points"], dtype=np.int32)


def draw_alert_banner(frame, text):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 36), RED, -1)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Module 9 live/video demo.")
    ap.add_argument("--video", required=True, help="Path to uploaded clip, or 0 for webcam.")
    ap.add_argument("--camera-id", default="CAM01")
    ap.add_argument("--zone", default=None, help="Path to zone JSON from mark_zone.py (optional).")
    ap.add_argument("--object-type", default="person", choices=["person", "vehicle"])
    ap.add_argument("--mode", default="night", choices=["night", "day"],
                     help="Simulated time-of-day fed to Module 9's night-schedule check "
                          "(so day-recorded clips can still be tested as if at night).")
    ap.add_argument("--min-area", type=int, default=600, help="Ignore motion blobs smaller than this (pixels).")
    ap.add_argument("--movement-threshold", type=float, default=20.0,
                     help="Passed straight to NightScheduleConfig.movement_pixel_threshold.")
    ap.add_argument("--cooldown", type=float, default=8.0,
                     help="Passed straight to NightScheduleConfig.cooldown_s (shorter than the 30s "
                          "production default so a short demo clip actually shows repeat alerts).")
    ap.add_argument("--output", default=None, help="Path to save the annotated output video (mp4).")
    ap.add_argument("--live", action="store_true", help="Show a live preview window (needs a display).")
    ap.add_argument("--alerts-out", default=None, help="Where to save fired events as JSON lines.")
    args = ap.parse_args()

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

    zone = load_zone(args.zone)
    zone_id = "ZONE1" if zone is not None else None

    config = NightScheduleConfig(
        camera_id=args.camera_id,
        watched_object_types=[args.object_type],
        movement_pixel_threshold=args.movement_threshold,
        cooldown_s=args.cooldown,
    )
    detector = NightMovementDetector(config)
    tracker = SimpleTracker()
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    alerts_path = Path(args.alerts_out or f"demo/alerts_{args.camera_id}.jsonl")
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    alerts_file = open(alerts_path, "w")

    base_dt = _BASE_DT[args.mode]
    frame_idx = 0
    n_alerts = 0
    banner_text = None
    banner_ttl = 0

    print(f"Running Module 9 demo | mode={args.mode} | zone={'yes' if zone is not None else 'no'} "
          f"| camera_id={args.camera_id}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fg_mask = bg_sub.apply(frame)
        # detectShadows marks shadow pixels as 127 (grey) vs 255 for real
        # foreground -- thresholding them out here is the same idea as
        # movement.py's jitter filtering: don't let shadows count as objects.
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) < args.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append((x, y, x + w, y + h))

        tracks = tracker.update(detections)
        ts = base_dt + timedelta(seconds=frame_idx / fps)

        overlay = frame.copy()
        if zone is not None:
            cv2.fillPoly(overlay, [zone], CYAN)
            frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
            cv2.polylines(frame, [zone], isClosed=True, color=CYAN, thickness=2)

        if banner_ttl > 0:
            draw_alert_banner(frame, banner_text)
            banner_ttl -= 1

        for tid, tr in tracks.items():
            x1, y1, x2, y2 = tr.bbox
            ground_point = ((x1 + x2) / 2.0, float(y2))

            zone_breach = False
            dist = -1e9
            if zone is not None:
                dist = cv2.pointPolygonTest(zone, ground_point, True)
                zone_breach = dist >= 0
            tr.dist_history.append(dist)

            approaching = False
            if zone is not None and not zone_breach and len(tr.dist_history) >= 5:
                if tr.dist_history[-1] - tr.dist_history[0] > 8:  # getting closer to the zone edge
                    approaching = True

            update = {
                "camera_id": args.camera_id,
                "track_id": tid,
                "object_type": args.object_type,
                "timestamp": ts,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "confidence": 0.9,
                "zone_id": zone_id if zone_breach else None,
                "zone_breach": zone_breach,
            }
            from night_movement import TrackUpdate
            event = detector.process(TrackUpdate(**update))

            if event is not None:
                n_alerts += 1
                line = json.dumps(event.to_dict())
                alerts_file.write(line + "\n")
                alerts_file.flush()
                print(f"[ALERT] {line}")
                banner_text = f"ALERT: {event.event_type} | {event.severity} | track {tid}"
                banner_ttl = int(fps * 2)  # keep banner up ~2s

            if zone_breach:
                color, label = RED, "INTRUSION"
            elif approaching:
                color, label = YELLOW, "APPROACHING ZONE"
            else:
                color, label = GREEN, args.object_type

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{tid} {label}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"mode={args.mode}  frame={frame_idx}  alerts={n_alerts}",
                    (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        if writer is not None:
            writer.write(frame)
        if args.live:
            cv2.imshow("Module 9 -- Night Movement Detection Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.live:
        cv2.destroyAllWindows()
    alerts_file.close()

    print(f"\nDone. {frame_idx} frames processed, {n_alerts} Module 9 alert(s) fired.")
    if args.output:
        print(f"Annotated video saved to: {args.output}")
    print(f"Alerts saved to: {alerts_path}")


if __name__ == "__main__":
    main()
