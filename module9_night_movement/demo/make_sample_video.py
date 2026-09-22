"""
make_sample_video.py -- generates a small synthetic "night-like" test
clip (dark background, a moving bright blob that walks toward and into
a fixed rectangle) so you can test mark_zone.py + live_demo.py before
using a real uploaded night clip. Purely a practice/testing aid.

    python demo/make_sample_video.py --out demo/sample_night.mp4
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo/sample_night.mp4")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    w, h = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))

    for i in range(args.frames):
        frame = np.full((h, w, 3), 12, dtype=np.uint8)  # dark "night" background
        noise = (np.random.randn(h, w, 3) * 3).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # a "shadow"/jitter blob that barely moves - should NOT alert
        jx = 80 + int(3 * np.sin(i * 0.9))
        cv2.circle(frame, (jx, 300), 12, (60, 60, 60), -1)

        # a "person" walking left -> right, ending inside the zone (right side)
        px = int(40 + i * 3.5)
        cv2.rectangle(frame, (px, 150), (px + 25, 230), (200, 200, 200), -1)

        writer.write(frame)

    writer.release()
    print(f"Sample clip written to {args.out} ({args.frames} frames, {w}x{h} @ {args.fps}fps)")


if __name__ == "__main__":
    main()
