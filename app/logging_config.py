"""Logging configuration for the face detection service."""

from __future__ import annotations

import logging


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def mask_url_secret(url: str | None) -> str | None:
    """Hide credentials in an RTSP URL before logging it."""

    if not url or "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    _, host = rest.rsplit("@", 1)
    return f"{scheme}://***:***@{host}"

