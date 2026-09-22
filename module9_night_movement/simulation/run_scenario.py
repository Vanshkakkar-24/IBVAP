#!/usr/bin/env python3
"""
Module 9 scenario runner --- implements the platform's "Simulation Mode"
(design doc Section 33) at the track level:

    Scenario tracks.json + expected.json
        v
    (stand-in for VideoFileSource + Module 2/3/5, which publish
     TrackUpdate objects in production)
        v
    Same NightMovementDetector used in production
        v
    Actual events
        v
    Compare with expected
        v
    PASS / FAIL

Why "tracks.json" instead of "video.mp4" for this module specifically:
Module 9 consumes Module 5's track-update output, not raw video. Until
Modules 1/2/3/5 are integrated, this harness lets Module 9 be developed
and regression-tested completely standalone against the *contract*
those modules will publish. Once the upstream modules land, the same
scenario folders can be re-pointed at real recorded night videos - only
the "load updates" step below changes; the detector and the PASS/FAIL
comparison do not.

Usage:
    python simulation/run_scenario.py                     # run all scenarios
    python simulation/run_scenario.py SCN004_night_intrusion   # run one
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from night_movement.config import NightScheduleConfig
from night_movement.detector import NightMovementDetector
from night_movement.models import TrackUpdate

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


def load_scenario(folder: Path):
    with open(folder / "tracks.json") as f:
        tracks_data = json.load(f)
    with open(folder / "expected.json") as f:
        expected = json.load(f)
    return tracks_data, expected


def build_updates(tracks_data: dict):
    camera_id = tracks_data["camera_id"]
    updates = []
    for u in tracks_data["updates"]:
        updates.append(
            TrackUpdate(
                camera_id=camera_id,
                track_id=u["track_id"],
                object_type=u["object_type"],
                timestamp=datetime.fromisoformat(u["timestamp"]),
                bbox=tuple(u["bbox"]),
                confidence=u["confidence"],
                zone_breach=u.get("zone_breach", False),
                zone_id=u.get("zone_id"),
            )
        )
    return updates


def run_one(folder: Path, config: NightScheduleConfig) -> bool:
    tracks_data, expected = load_scenario(folder)
    updates = build_updates(tracks_data)

    detector = NightMovementDetector(config)
    events = []
    for u in updates:
        result = detector.process(u)
        if result is not None:
            events.append(result)

    got_event = len(events) > 0
    expect_event = expected.get("expect_event", False)

    ok = got_event == expect_event
    if ok and got_event:
        last = events[-1]
        if "event_type" in expected:
            ok = ok and last.event_type == expected["event_type"]
        if "severity" in expected:
            ok = ok and last.severity == expected["severity"]
        if "reason_codes_include" in expected:
            ok = ok and all(rc in last.reason_codes for rc in expected["reason_codes_include"])

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {folder.name}: {tracks_data.get('description', '')}")
    if events:
        for e in events:
            print(f"        -> {json.dumps(e.to_dict())}")
    else:
        print("        -> (no event emitted)")
    return ok


def main():
    config = NightScheduleConfig(camera_id="CAM-001")  # defaults: 18:00-06:00 window

    if len(sys.argv) > 1:
        targets = [SCENARIOS_DIR / sys.argv[1]]
    else:
        targets = sorted(p for p in SCENARIOS_DIR.iterdir() if p.is_dir())

    results = [run_one(t, config) for t in targets]

    total, passed = len(results), sum(results)
    print(f"\n{passed}/{total} scenarios passed.")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
