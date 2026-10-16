"""In-memory registry of running registration and attendance sessions.

Keyed by a random id, so many phones can work at once. Finished sessions stay
readable for [api] session_ttl_s, then are swept. Sessions do not survive a
restart and are not shared between worker processes: run one worker.
"""
import threading
import time
import uuid

from app.core.config import config


class SessionLimitError(RuntimeError):
    pass


class SessionRegistry:
    def __init__(self, max_sessions=None, ttl_s=None):
        self.max_sessions = max_sessions or config.api.max_sessions
        self.ttl_s = config.api.session_ttl_s if ttl_s is None else ttl_s
        self._items = {}          # id -> session
        self._lock = threading.Lock()

    def add(self, session):
        with self._lock:
            self._sweep()
            if len(self._items) >= self.max_sessions:
                raise SessionLimitError("Too many open sessions - retry shortly")
            session.id = uuid.uuid4().hex
            session.lock = threading.Lock()
            self._items[session.id] = session
        return session

    def get(self, session_id):
        with self._lock:
            return self._items.get(session_id)

    def _sweep(self):
        now = time.time()
        for sid, s in list(self._items.items()):
            if s.final and now - s.finished_at > self.ttl_s:
                del self._items[sid]
            elif not s.final and now > s.deadline + self.ttl_s:
                del self._items[sid]    # abandoned mid-way
