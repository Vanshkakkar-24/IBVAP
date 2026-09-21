"""Database session and engine management supporting PostgreSQL and SQLite."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.database_url

        # Ensure parent directory exists if using SQLite
        if db_url.startswith("sqlite"):
            # Handle sqlite:///data/ibvap.db
            sqlite_path = db_url.replace("sqlite:///", "")
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
            )
            logger.info("Initialized SQLite database engine: %s", db_url)
        else:
            try:
                _engine = create_engine(db_url, pool_pre_ping=True)
                # Verify connection
                with _engine.connect() as conn:
                    pass
                logger.info("Initialized PostgreSQL database engine: %s", db_url)
            except Exception as exc:
                logger.warning(
                    "Failed to connect to PostgreSQL (%s); falling back to SQLite at data/ibvap.db", exc
                )
                Path("data").mkdir(parents=True, exist_ok=True)
                fallback_url = "sqlite:///data/ibvap.db"
                _engine = create_engine(
                    fallback_url,
                    connect_args={"check_same_thread": False},
                )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        engine = get_engine()
        _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionFactory


def init_db() -> None:
    """Create all database tables if they do not exist."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified/created successfully.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager yielding a database session for background threads/workers."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
