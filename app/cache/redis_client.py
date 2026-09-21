"""Redis real-time state manager with resilient in-memory fallback."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class InMemoryRedisFallback:
    """Thread-safe in-memory cache mimicking essential Redis operations."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._kv: dict[str, tuple[str, float | None]] = {}  # key -> (val, expire_time)
        self._hashes: dict[str, dict[str, str]] = {}
        self._lists: dict[str, list[str]] = {}

    def set(self, name: str, value: str, ex: int | None = None) -> bool:
        with self._lock:
            expire_at = time.time() + ex if ex else None
            self._kv[name] = (str(value), expire_at)
            return True

    def get(self, name: str) -> str | None:
        with self._lock:
            if name not in self._kv:
                return None
            val, expire_at = self._kv[name]
            if expire_at and time.time() > expire_at:
                del self._kv[name]
                return None
            return val

    def hset(self, name: str, key: str | None = None, value: str | None = None, mapping: dict[str, Any] | None = None) -> int:
        with self._lock:
            if name not in self._hashes:
                self._hashes[name] = {}
            count = 0
            if mapping:
                for k, v in mapping.items():
                    self._hashes[name][str(k)] = str(v)
                    count += 1
            if key is not None and value is not None:
                self._hashes[name][str(key)] = str(value)
                count += 1
            return count

    def hget(self, name: str, key: str) -> str | None:
        with self._lock:
            return self._hashes.get(name, {}).get(str(key))

    def hgetall(self, name: str) -> dict[str, str]:
        with self._lock:
            return dict(self._hashes.get(name, {}))

    def lpush(self, name: str, *values: str) -> int:
        with self._lock:
            if name not in self._lists:
                self._lists[name] = []
            for val in values:
                self._lists[name].insert(0, str(val))
            # Cap list size to 500 items to avoid unbounded memory
            if len(self._lists[name]) > 500:
                self._lists[name] = self._lists[name][:500]
            return len(self._lists[name])

    def lrange(self, name: str, start: int, end: int) -> list[str]:
        with self._lock:
            lst = self._lists.get(name, [])
            if end == -1:
                return lst[start:]
            return lst[start : end + 1]

    def delete(self, *names: str) -> int:
        with self._lock:
            count = 0
            for n in names:
                if n in self._kv:
                    del self._kv[n]
                    count += 1
                if n in self._hashes:
                    del self._hashes[n]
                    count += 1
                if n in self._lists:
                    del self._lists[n]
                    count += 1
            return count


class RedisStateManager:
    """High-frequency state management for active tracks, global identities, and live events."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any = None
        self.is_connected: bool = False
        self._init_redis()

    def _init_redis(self) -> None:
        try:
            import redis

            client = redis.Redis.from_url(
                self.settings.redis_url,
                socket_timeout=1.5,
                socket_connect_timeout=1.5,
                decode_responses=True,
            )
            client.ping()
            self._client = client
            self.is_connected = True
            logger.info("Connected to Redis server at %s", self.settings.redis_url)
        except Exception as exc:
            logger.info("Redis not reachable (%s); using resilient in-memory state manager", exc)
            self._client = InMemoryRedisFallback()
            self.is_connected = False

    # ==================== Active Tracks ====================

    def set_active_tracks(self, camera_id: str, tracks: list[dict[str, Any]]) -> None:
        key = f"camera:{camera_id}:tracks"
        self._client.set(key, json.dumps(tracks), ex=30)

    def get_active_tracks(self, camera_id: str) -> list[dict[str, Any]]:
        key = f"camera:{camera_id}:tracks"
        val = self._client.get(key)
        if val:
            try:
                return json.loads(val)
            except Exception:
                return []
        return []

    # ==================== Camera State ====================

    def set_camera_state(self, camera_id: str, state: dict[str, Any]) -> None:
        key = f"camera:{camera_id}:state"
        self._client.set(key, json.dumps(state), ex=60)

    def get_camera_state(self, camera_id: str) -> dict[str, Any] | None:
        key = f"camera:{camera_id}:state"
        val = self._client.get(key)
        if val:
            try:
                return json.loads(val)
            except Exception:
                return None
        return None

    # ==================== Global Person State ====================

    def set_global_person_state(self, global_person_id: str, state: dict[str, Any]) -> None:
        key = f"person:{global_person_id}:state"
        self._client.set(key, json.dumps(state), ex=120)

    def get_global_person_state(self, global_person_id: str) -> dict[str, Any] | None:
        key = f"person:{global_person_id}:state"
        val = self._client.get(key)
        if val:
            try:
                return json.loads(val)
            except Exception:
                return None
        return None

    # ==================== Real-time Events Queue ====================

    def push_realtime_event(self, event_data: dict[str, Any]) -> None:
        key = "system:realtime_events"
        self._client.lpush(key, json.dumps(event_data))

    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        key = "system:realtime_events"
        raw_items = self._client.lrange(key, 0, limit - 1)
        results = []
        for item in raw_items:
            try:
                results.append(json.loads(item))
            except Exception:
                pass
        return results


_redis_manager = None


def get_redis_manager() -> RedisStateManager:
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisStateManager()
    return _redis_manager
