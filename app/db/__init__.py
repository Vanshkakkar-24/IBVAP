"""Database package for IBVAP multi-camera system."""

from app.db.database import get_db, init_db
from app.db.models import (
    Base,
    Camera,
    CameraTrack,
    Event,
    FaceSnapshot,
    Person,
    PersonEmbedding,
    PersonMovement,
)

__all__ = [
    "Base",
    "Camera",
    "Person",
    "PersonEmbedding",
    "CameraTrack",
    "PersonMovement",
    "FaceSnapshot",
    "Event",
    "init_db",
    "get_db",
]
