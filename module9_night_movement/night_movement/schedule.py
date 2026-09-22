"""
Night-schedule check --- step 2 of the Module 9 flow:

    Detection + timestamp -> [Night schedule check] -> Object/zone filter -> ...

Kept as a standalone function/class so it can be unit-tested in
isolation and so a site can swap "fixed hours" for "sunrise/sunset"
without touching the detector logic.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Optional

from .config import NightScheduleConfig


def is_night_fixed(ts: datetime, start_hour: int, end_hour: int) -> bool:
    """
    True if `ts` falls inside the [start_hour, end_hour) window.
    Handles the overnight wraparound case, e.g. start=18, end=6
    means "night" is 18:00 -> 23:59 and 00:00 -> 05:59.
    """
    t = ts.time()
    start = time(hour=start_hour % 24)
    end = time(hour=end_hour % 24)

    if start == end:
        # 0-hour window configured -> treat as "always night" is almost
        # certainly a misconfiguration; fail safe to "never night".
        return False

    if start < end:
        # simple same-day window, e.g. 22 -> 23 (doesn't wrap midnight)
        return start <= t < end
    else:
        # wraps midnight, e.g. 18 -> 6
        return t >= start or t < end


def is_night_astral(ts: datetime, latitude: float, longitude: float, timezone: str) -> bool:
    """
    Sunrise/sunset based check. Requires the optional `astral` package.
    Falls back to a conservative fixed 18:00-06:00 window if `astral`
    is not installed, so the detector never hard-crashes in prototype
    environments that skip this optional dependency.
    """
    try:
        from astral import LocationInfo
        from astral.sun import sun
        import pytz
    except ImportError:
        return is_night_fixed(ts, 18, 6)

    loc = LocationInfo(latitude=latitude, longitude=longitude)
    tz = pytz.timezone(timezone)
    ts_local = ts.astimezone(tz) if ts.tzinfo else tz.localize(ts)
    s = sun(loc.observer, date=ts_local.date(), tzinfo=tz)
    return ts_local < s["sunrise"] or ts_local > s["sunset"]


def is_night(ts: datetime, cfg: NightScheduleConfig) -> bool:
    """Single entry point the detector calls - dispatches on cfg.mode."""
    if cfg.mode == "astral" and cfg.latitude is not None and cfg.longitude is not None:
        return is_night_astral(ts, cfg.latitude, cfg.longitude, cfg.timezone)
    return is_night_fixed(ts, cfg.start_hour, cfg.end_hour)
