"""Face size quality helpers."""

from __future__ import annotations

from app.schemas.face import BoundingBox


def meets_minimum_size(bbox: BoundingBox, min_width: int, min_height: int) -> bool:
    return bbox.width >= min_width and bbox.height >= min_height


def face_area(bbox: BoundingBox) -> int:
    return bbox.width * bbox.height

