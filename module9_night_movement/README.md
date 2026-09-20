# Module 9 --- Night-Time Movement Detection

Part of: **AI-Based Intelligent Video Analytics Platform for Border Surveillance**
(Problem Statement 26187). This module is owned per the team's task split
as **Module 9**.

## 1. What this module does (plain-language)

At night, an ordinary CCTV feed cannot tell the difference between
"nothing is happening" and "someone is walking near the fence." A
human has to keep watching. This module lets the *software* watch
instead: it looks at the objects Module 5 (tracking) is already
following, checks whether it is currently "night" at that camera,
checks if that object is actually moving in a meaningful way (not a
shadow or camera noise), and if so, raises a night-movement event that
the rest of the platform turns into an alert.

It does **not** do its own person/vehicle detection (that's Modules
3/4), and it does **not** decide whether a track is inside a
restricted zone (that's Module 8) --- it *consumes* that information
and adds the "was this at night, and was it real movement" judgement
on top, exactly as the system design doc's architecture intends.

## 2. Where it sits in the pipeline

```
Module 2 (frame pipeline) -> Module 3/4 (detection) -> Module 5 (tracking)
                                                              |
                                                  TrackUpdate (per frame)
                                                              v
                                            +----------------------------+
                                            |   MODULE 9 (this code)     |
                                            |  1. Night schedule check   |
                                            |  2. Object / zone filter   |
                                            |  3. Movement threshold     |
                                            |  4. Cooldown               |
                                            |  5. Severity / event_type  |
                                            +----------------------------+
                                                              |
                                                  NightMovementEvent
                                                              v
                                            Module 12 (Event Engine) -> Module 13 (Alerts)
```

Module 8 (Virtual Fence) is a *sibling*, not upstream: it sets
`zone_breach=True` / `zone_id=...` on the same TrackUpdate object
before it reaches this module, so a night-time zone crossing is
automatically escalated to `NIGHT_INTRUSION` / `CRITICAL` instead of
the default `NIGHT_MOVEMENT`.

## 3. Repository layout

```
module9/
  night_movement/
    __init__.py       Public API
    models.py          TrackUpdate (input) and NightMovementEvent (output) --
                        both shaped to match the platform's existing data
                        contracts (design doc Sections 8 and 31.4), so no
                        translation layer is needed to plug into Module 5 or Module 12.
    config.py          NightScheduleConfig + PostgreSQL row loader
    schedule.py         "Is it night?" -- fixed-hours (default) or
                        sunrise/sunset (optional, needs `astral`)
    movement.py         Net-displacement math that tells real walking/
                        driving apart from shadows/sensor noise
    low_light.py         Optional CLAHE preprocessing hook (design doc:
                        "Optional low-light enhancement later")
    detector.py          NightMovementDetector -- orchestrates the full
                        flow end to end
  tests/                26 unit + integration tests (pytest)
  simulation/
    run_scenario.py     Standalone scenario runner (platform "Simulation
                        Mode", Section 33) -- works at the track level so
                        Module 9 can be tested before Modules 1/2/3/5 are
                        wired up
    scenarios/
      SCN001_day_normal/       matches design doc Section 32.1 SCN-001
      SCN004_night_intrusion/  matches design doc Section 32.1 SCN-004
  schema.sql            PostgreSQL DDL for per-camera config
  requirements.txt
```

## 4. How to run it (for the demo / to prove it works)

```bash
cd module9
pip install -r requirements.txt

# unit + integration tests
python -m pytest tests/ -v

# scenario simulation (mirrors the platform's PASS/FAIL scenario format)
python simulation/run_scenario.py
```

Current status: **26/26 tests passing, 2/2 scenarios passing**
(SCN-001 daytime-no-alert, SCN-004 night-restricted-zone-entry).

## 4.1 Live dark-room / CCTV operation

The completed live runner is `demo/live_demo_yolo.py`. It uses YOLO +
ByteTrack to identify and track real **people and vehicles** from a USB
webcam, RTSP CCTV URL, or video file. Before each AI inference it measures
scene luminance. A dark frame is denoised, gamma-lifted, and contrast
enhanced (CLAHE); normal frames are passed through unchanged. The original
camera frame is still used for the recorded output, so the saved evidence is
not artificially brightened.

Darkness is also a first-class trigger: in `--mode auto` (the default), a
moving person/vehicle in a genuinely dark room is handled as night movement
even if the computer clock says daytime. Events include `DARK_SCENE` in their
reason codes. This is useful for an indoor CCTV camera with lights switched
off.

```bash
cd module9
pip install -r requirements.txt

# Webcam (use 1 instead of 0 if your CCTV/phone camera is the second camera)
python demo/live_demo_yolo.py --video 0 --camera-id CAM01 --mode auto

# Network CCTV stream; quote the RTSP URL
python demo/live_demo_yolo.py --video "rtsp://user:password@camera-ip:554/stream" --camera-id CAM01 --mode auto

# Dark video file, saving an annotated evidence video
python demo/live_demo_yolo.py --video demo/sample_night.mp4 --camera-id CAM01 \
  --mode auto --output demo/night_result.mp4
```

The on-screen status says `ENHANCED` while low-light processing is active.
Use `--dark-threshold 90` to treat a dimmer scene as dark sooner, or
`--no-low-light` to diagnose the raw-camera baseline. During the preview,
draw a restricted zone with **R**, click its points, press **F** to finish,
and **S** to save it. A moving person/vehicle inside that zone emits
`NIGHT_INTRUSION` at `CRITICAL` severity.

## 5. Design decisions worth explaining in the meeting

- **Net displacement, not total path length, for the movement check.**
  A shadow or sensor noise jitters back and forth; a person walking
  moves steadily in one direction. Measuring displacement between the
  start and end of a short time window (not the sum of every tiny
  step) is what naturally filters out jitter -- this directly answers
  the design doc's prototype instruction to "test shadows/noise false
  positives."

- **Escalation overrides cooldown.** A per-track cooldown stops the
  same person from spamming the Event Engine with an event every
  single frame. But if that same person then crosses into a
  restricted zone, that is a *more severe* situation and must not be
  silently swallowed just because a lower-severity event fired 10
  seconds earlier. The detector tracks the last-emitted severity and
  always lets a genuine escalation (movement -> intrusion) through
  immediately. This was caught and fixed via the test suite before
  being called "done" -- `test_scn004_night_restricted_zone_entry_is_critical`
  originally failed for exactly this reason.

- **Config lives in Postgres, not code.** Night hours differ by
  season and by BOP location; thresholds need tuning per camera once
  deployed. `NightScheduleConfig` + `schema.sql` let an operator change
  these from the dashboard (Module 21) without redeploying the AI
  pipeline.

- **Fixed-hours by default, sunrise/sunset optional.** The doc's
  prototype instruction says "use configurable night hours," so fixed
  start/end hour is the default, dependency-free path. A sunrise/sunset
  mode is included for later, real-deployment accuracy, but degrades
  gracefully to fixed-hours if the optional `astral` package isn't
  installed -- it will never crash the pipeline.

- **Track-level simulation harness, not video-level, for now.** The
  platform-wide Simulation Mode (Section 33) is defined at the
  video.mp4 + expected.json level, which assumes Modules 1/2/3/5 are
  already producing tracks. Since this module can be developed and
  fully tested in parallel with those, the harness here runs directly
  against a `tracks.json` (a recorded/mocked Module 5 output) so
  Module 9 doesn't have to wait on the rest of the team. Once the
  upstream modules are ready, the same scenario folders can be
  re-pointed at real night-time recordings with zero changes to the
  detector itself.

## 6. What's intentionally NOT in this module

- Actual person/vehicle detection -- Modules 3/4.
- Deciding if a point is inside a drawn zone -- Module 8 (this module
  only reads the `zone_breach` flag Module 8 already computed).
- Deduplicating events platform-wide, persistence, WebSocket push --
  Module 12 (Event Engine) and Module 13 (Alert Engine).
- Actually sending an SMS/siren/dashboard popup -- Module 13/15.

Keeping these boundaries intact is exactly what Section 3.1 of the
design doc ("Responsibility boundaries") asks for, and it's also why
this module can be developed, tested, and demoed without needing
Module 3, 4, 5, 8, or 12 to be finished first.

## 7. Next steps to take this from prototype to "wired in"

1. Replace `simulation/run_scenario.py`'s JSON loader with a real
   subscriber to whatever queue/interface Module 5 exposes (in-process
   call, Redis stream, or FastAPI internal event bus -- to be agreed
   with Module 1/2's owner).
2. Add a small FastAPI router (`GET/PUT /config/night-schedule/{camera_id}`)
   over `config.py` + `schema.sql` so the dashboard owner (Module 21) can
   expose the on/off toggle and threshold sliders in the operator UI.
3. Hand `NightMovementEvent.to_dict()` output directly to whatever
   function Module 12's owner exposes for "ingest an event" -- the
   field names already match Section 31.4's Event contract on purpose.
4. Once a real night-time test video is available, record its Module 5
   track output once and drop it in as a new `scenarios/` folder --
   the detector code does not change.
