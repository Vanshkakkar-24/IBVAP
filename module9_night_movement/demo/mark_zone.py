"""
mark_zone.py -- Draw a restricted zone on a video's first frame and save it
as JSON that live_demo.py can load.

Run this from inside the `module9` folder, in a normal terminal with a
display (VS Code's integrated terminal on your own laptop works fine --
this will NOT work over SSH/headless servers since it opens a window).

USAGE:
    python demo/mark_zone.py --video path/to/clip.mp4 --camera-id CAM01 --out demo/zone_CAM01.json

CONTROLS (in the image window that pops up):
    Left click   -> add a corner point of the restricted zone (polygon)
    'u'          -> undo the last point
    's'          -> save the zone and exit (needs at least 3 points)
    'q' / Esc    -> quit WITHOUT saving

You can mark a different zone for every camera/video by giving each a
different --out path, e.g. demo/zone_CAM01.json, demo/zone_gate2.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_points: list[list[int]] = []


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append([x, y])


def main() -> None:
    ap = argparse.ArgumentParser(description="Mark a restricted zone on a video frame.")
    ap.add_argument("--video", required=True, help="Path to the night-camera clip (or 0 for webcam).")
    ap.add_argument("--camera-id", required=True, help="Camera ID, e.g. CAM01 (must match live_demo.py).")
    ap.add_argument("--out", required=True, help="Where to save the zone JSON.")
    ap.add_argument("--frame-index", type=int, default=0, help="Which frame to use for marking (default: first).")
    args = ap.parse_args()

    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Could not open video source: {args.video}")
        sys.exit(1)

    frame = None
    for _ in range(args.frame_index + 1):
        ok, frame = cap.read()
        if not ok:
            break
    cap.release()

    if frame is None:
        print("Could not read a frame from that video.")
        sys.exit(1)

    window = "Mark restricted zone  |  click corners, 'u' undo, 's' save, 'q' quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)

    print("Click the corners of the restricted zone in order, then press 's' to save.")

    while True:
        disp = frame.copy()
        for p in _points:
            cv2.circle(disp, tuple(p), 5, (0, 255, 255), -1)
        if len(_points) > 1:
            pts = np.array(_points, dtype=np.int32)
            cv2.polylines(disp, [pts], isClosed=len(_points) > 2, color=(0, 255, 255), thickness=2)
        cv2.putText(disp, f"points: {len(_points)}  (u=undo, s=save, q=quit)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u") and _points:
            _points.pop()
        elif key == ord("s"):
            if len(_points) >= 3:
                break
            print("Need at least 3 points to close a zone -- keep clicking.")
        elif key in (ord("q"), 27):
            print("Cancelled -- no zone saved.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"camera_id": args.camera_id, "points": _points}, f, indent=2)
    print(f"Saved zone with {len(_points)} points to {out_path}")


if __name__ == "__main__":
    main()
