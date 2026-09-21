"""SQLAlchemy ORM models for IBVAP multi-camera tracking & logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Camera(Base):
    """Registered camera streams."""

    __tablename__ = "cameras"

    id = Column(String(64), primary_key=True, index=True)  # e.g. 'CAM-001'
    name = Column(String(128), nullable=False)
    type = Column(String(32), nullable=False, default="webcam")  # webcam, rtsp, phone
    stream_url = Column(String(512), nullable=False)
    location = Column(String(128), nullable=False, default="Default Location")
    status = Column(String(32), nullable=False, default="offline")  # online, offline, error
    enabled = Column(Boolean, nullable=False, default=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.id,
            "name": self.name,
            "type": self.type,
            "stream_url": self.stream_url,
            "location": self.location,
            "status": self.status,
            "enabled": self.enabled,
        }


class Person(Base):
    """Persistent globally tracked person identity."""

    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_person_id = Column(String(64), unique=True, index=True, nullable=False)  # 'PERSON-00017'
    first_seen = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    last_seen = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    status = Column(String(32), nullable=False, default="active")  # active, inactive

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "global_person_id": self.global_person_id,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "status": self.status,
        }


class PersonEmbedding(Base):
    """Historical Re-ID appearance embeddings for a tracked global person."""

    __tablename__ = "person_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(64), ForeignKey("persons.global_person_id"), nullable=False, index=True)
    embedding = Column(Text, nullable=False)  # JSON-encoded float array
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    def get_embedding(self) -> list[float]:
        try:
            return json.loads(self.embedding)
        except Exception:
            return []

    def set_embedding(self, emb: list[float]) -> None:
        self.embedding = json.dumps(emb)


class CameraTrack(Base):
    """Camera local track session mapping to a global person."""

    __tablename__ = "camera_tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), ForeignKey("cameras.id"), nullable=False, index=True)
    person_id = Column(String(64), ForeignKey("persons.global_person_id"), nullable=False, index=True)
    local_track_id = Column(Integer, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "person_id": self.person_id,
            "local_track_id": self.local_track_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }


class PersonMovement(Base):
    """Fine-grained tracking logs across cameras."""

    __tablename__ = "person_movements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(64), ForeignKey("persons.global_person_id"), nullable=False, index=True)
    camera_id = Column(String(64), ForeignKey("cameras.id"), nullable=False, index=True)
    track_id = Column(Integer, nullable=False)
    event_type = Column(String(32), nullable=False)  # ENTER, MOVING, FACE_DETECTED, EXIT, CAMERA_TRANSITION
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    bbox = Column(String(128), nullable=True)  # JSON-encoded [x, y, w, h] or "[x, y, w, h]"
    confidence = Column(Float, nullable=False, default=1.0)

    def to_dict(self) -> dict[str, Any]:
        bbox_val = None
        if self.bbox:
            try:
                bbox_val = json.loads(self.bbox)
            except Exception:
                bbox_val = self.bbox
        return {
            "id": self.id,
            "person_id": self.person_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "bbox": bbox_val,
            "confidence": self.confidence,
        }


class FaceSnapshot(Base):
    """Quality face snapshots captured without face recognition."""

    __tablename__ = "face_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(64), ForeignKey("persons.global_person_id"), nullable=False, index=True)
    camera_id = Column(String(64), ForeignKey("cameras.id"), nullable=False, index=True)
    track_id = Column(Integer, nullable=False)
    image_url = Column(String(512), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    confidence = Column(Float, nullable=False, default=1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "person_id": self.person_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "image_url": self.image_url,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": self.confidence,
        }


class Event(Base):
    """System-wide operational and tracking events."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(64), nullable=True, index=True)
    camera_id = Column(String(64), nullable=True, index=True)
    event_type = Column(String(64), nullable=False, index=True)  # CAMERA_TRANSITION, CAMERA_DISCONNECTED, etc.
    severity = Column(String(16), nullable=False, default="info")  # info, warning, critical
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    metadata_json = Column("metadata", Text, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        meta = None
        if self.metadata_json:
            try:
                meta = json.loads(self.metadata_json)
            except Exception:
                meta = {"raw": self.metadata_json}
        return {
            "id": self.id,
            "person_id": self.person_id,
            "camera_id": self.camera_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": meta,
        }
