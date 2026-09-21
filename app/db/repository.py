"""Database repository for IBVAP entities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import Camera, CameraTrack, Event, FaceSnapshot, Person, PersonEmbedding, PersonMovement

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IBVAPRepository:
    """Synchronous repository working with SQLAlchemy sessions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ==================== Cameras ====================

    def get_all_cameras(self) -> list[Camera]:
        return list(self.session.scalars(select(Camera).order_by(Camera.id)).all())

    def get_camera(self, camera_id: str) -> Camera | None:
        return self.session.scalar(select(Camera).where(Camera.id == camera_id))

    def create_or_update_camera(
        self,
        camera_id: str,
        name: str,
        stream_url: str,
        camera_type: str = "webcam",
        location: str = "Default",
        status: str = "offline",
        enabled: bool = True,
    ) -> Camera:
        cam = self.get_camera(camera_id)
        if cam is None:
            cam = Camera(
                id=camera_id,
                name=name,
                type=camera_type,
                stream_url=stream_url,
                location=location,
                status=status,
                enabled=enabled,
            )
            self.session.add(cam)
        else:
            cam.name = name
            cam.type = camera_type
            cam.stream_url = stream_url
            cam.location = location
            cam.status = status
            cam.enabled = enabled
        self.session.flush()
        return cam

    def update_camera_status(self, camera_id: str, status: str) -> None:
        cam = self.get_camera(camera_id)
        if cam:
            cam.status = status
            self.session.flush()

    def delete_camera(self, camera_id: str) -> bool:
        cam = self.get_camera(camera_id)
        if cam:
            self.session.delete(cam)
            self.session.flush()
            return True
        return False

    # ==================== Persons ====================

    def get_all_persons(self, limit: int = 100, status: str | None = None) -> list[Person]:
        stmt = select(Person).order_by(desc(Person.last_seen)).limit(limit)
        if status:
            stmt = stmt.where(Person.status == status)
        return list(self.session.scalars(stmt).all())

    def get_person_by_global_id(self, global_person_id: str) -> Person | None:
        return self.session.scalar(select(Person).where(Person.global_person_id == global_person_id))

    def create_or_touch_person(
        self,
        global_person_id: str,
        embedding: list[float] | None = None,
        status: str = "active",
    ) -> Person:
        person = self.get_person_by_global_id(global_person_id)
        now = utc_now()
        if person is None:
            person = Person(
                global_person_id=global_person_id,
                first_seen=now,
                last_seen=now,
                status=status,
            )
            self.session.add(person)
            self.session.flush()
        else:
            person.last_seen = now
            person.status = status
            self.session.flush()

        if embedding:
            emb_record = PersonEmbedding(
                person_id=global_person_id,
                embedding=json.dumps(embedding),
                created_at=now,
            )
            self.session.add(emb_record)
            self.session.flush()

        return person

    def get_person_history(self, global_person_id: str) -> dict[str, Any]:
        """Retrieve full chronological movement, snapshot, and event history for a person."""
        person = self.get_person_by_global_id(global_person_id)
        if not person:
            return {}

        movements = list(
            self.session.scalars(
                select(PersonMovement)
                .where(PersonMovement.person_id == global_person_id)
                .order_by(PersonMovement.timestamp)
            ).all()
        )
        snapshots = list(
            self.session.scalars(
                select(FaceSnapshot)
                .where(FaceSnapshot.person_id == global_person_id)
                .order_by(FaceSnapshot.timestamp)
            ).all()
        )
        tracks = list(
            self.session.scalars(
                select(CameraTrack)
                .where(CameraTrack.person_id == global_person_id)
                .order_by(CameraTrack.started_at)
            ).all()
        )

        return {
            "person": person.to_dict(),
            "tracks": [t.to_dict() for t in tracks],
            "movements": [m.to_dict() for m in movements],
            "snapshots": [s.to_dict() for s in snapshots],
        }

    # ==================== Movements & Tracks ====================

    def log_movement(
        self,
        person_id: str,
        camera_id: str,
        track_id: int,
        event_type: str,
        bbox: list[int] | tuple[int, int, int, int] | None = None,
        confidence: float = 1.0,
        timestamp: datetime | None = None,
    ) -> PersonMovement:
        now = timestamp or utc_now()
        mov = PersonMovement(
            person_id=person_id,
            camera_id=camera_id,
            track_id=track_id,
            event_type=event_type,
            timestamp=now,
            bbox=json.dumps(list(bbox)) if bbox else None,
            confidence=confidence,
        )
        self.session.add(mov)
        self.session.flush()
        return mov

    def record_track_start(
        self,
        camera_id: str,
        person_id: str,
        local_track_id: int,
        started_at: datetime | None = None,
    ) -> CameraTrack:
        now = started_at or utc_now()
        track = CameraTrack(
            camera_id=camera_id,
            person_id=person_id,
            local_track_id=local_track_id,
            started_at=now,
        )
        self.session.add(track)
        self.session.flush()
        return track

    def record_track_end(
        self,
        camera_id: str,
        local_track_id: int,
        ended_at: datetime | None = None,
    ) -> None:
        now = ended_at or utc_now()
        track = self.session.scalar(
            select(CameraTrack)
            .where(CameraTrack.camera_id == camera_id, CameraTrack.local_track_id == local_track_id)
            .order_by(desc(CameraTrack.started_at))
        )
        if track and track.ended_at is None:
            track.ended_at = now
            self.session.flush()

    # ==================== Face Snapshots ====================

    def save_face_snapshot(
        self,
        person_id: str,
        camera_id: str,
        track_id: int,
        image_url: str,
        confidence: float = 1.0,
        timestamp: datetime | None = None,
    ) -> FaceSnapshot:
        now = timestamp or utc_now()
        snapshot = FaceSnapshot(
            person_id=person_id,
            camera_id=camera_id,
            track_id=track_id,
            image_url=image_url,
            confidence=confidence,
            timestamp=now,
        )
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def get_face_snapshots(self, limit: int = 100, person_id: str | None = None) -> list[FaceSnapshot]:
        stmt = select(FaceSnapshot).order_by(desc(FaceSnapshot.timestamp)).limit(limit)
        if person_id:
            stmt = stmt.where(FaceSnapshot.person_id == person_id)
        return list(self.session.scalars(stmt).all())

    # ==================== Events ====================

    def log_event(
        self,
        event_type: str,
        severity: str = "info",
        person_id: str | None = None,
        camera_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> Event:
        now = timestamp or utc_now()
        ev = Event(
            person_id=person_id,
            camera_id=camera_id,
            event_type=event_type,
            severity=severity,
            timestamp=now,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.session.add(ev)
        self.session.flush()
        return ev

    def get_events(
        self,
        limit: int = 100,
        severity: str | None = None,
        camera_id: str | None = None,
        person_id: str | None = None,
    ) -> list[Event]:
        stmt = select(Event).order_by(desc(Event.timestamp)).limit(limit)
        if severity:
            stmt = stmt.where(Event.severity == severity)
        if camera_id:
            stmt = stmt.where(Event.camera_id == camera_id)
        if person_id:
            stmt = stmt.where(Event.person_id == person_id)
        return list(self.session.scalars(stmt).all())

    def get_event(self, event_id: int) -> Event | None:
        return self.session.scalar(select(Event).where(Event.id == event_id))
