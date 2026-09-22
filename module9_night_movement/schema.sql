-- Module 9 --- Night-Time Movement Detection
-- Per-camera configuration, editable from the dashboard (Module 21)
-- without restarting the AI pipeline process.

CREATE TABLE IF NOT EXISTS night_schedule_config (
    camera_id                  TEXT PRIMARY KEY,

    -- schedule
    mode                        TEXT NOT NULL DEFAULT 'fixed'
                                    CHECK (mode IN ('fixed', 'astral')),
    start_hour                  SMALLINT NOT NULL DEFAULT 18
                                    CHECK (start_hour BETWEEN 0 AND 23),
    end_hour                    SMALLINT NOT NULL DEFAULT 6
                                    CHECK (end_hour BETWEEN 0 AND 23),
    latitude                    DOUBLE PRECISION,
    longitude                   DOUBLE PRECISION,
    timezone                    TEXT NOT NULL DEFAULT 'Asia/Kolkata',

    -- object / zone filter
    watched_object_types        TEXT[] NOT NULL DEFAULT ARRAY['person', 'vehicle'],
    min_confidence               REAL NOT NULL DEFAULT 0.5,

    -- movement threshold (anti false-positive)
    movement_pixel_threshold    REAL NOT NULL DEFAULT 25.0,
    movement_time_window_s      REAL NOT NULL DEFAULT 3.0,
    min_track_age_s             REAL NOT NULL DEFAULT 1.0,

    -- alert-flood control
    cooldown_s                  REAL NOT NULL DEFAULT 30.0,

    -- severity mapping
    base_severity_person         TEXT NOT NULL DEFAULT 'HIGH',
    base_severity_vehicle        TEXT NOT NULL DEFAULT 'MEDIUM',

    enabled                      BOOLEAN NOT NULL DEFAULT TRUE,

    updated_at                   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Seed a sensible default row for the prototype camera used in demos.
INSERT INTO night_schedule_config (camera_id)
VALUES ('CAM-001')
ON CONFLICT (camera_id) DO NOTHING;
