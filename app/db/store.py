"""
SQLite store: people, their face embeddings and reference image, and
attendance records.

Every person has a name, an employee_id and an employee_number, each unique.
A version counter is bumped on every change to the face index, so the engine
can notice registrations and deletions and reload. WAL mode lets request
threads read while one writes.
"""
import sqlite3
import threading
from pathlib import Path

import numpy as np

from app.core.config import config

EMB_DIM = 512

_SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    employee_id     TEXT NOT NULL UNIQUE,
    employee_number TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS embeddings (
    id         INTEGER PRIMARY KEY,
    person_id  INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS face_images (
    person_id  INTEGER PRIMARY KEY REFERENCES people(id) ON DELETE CASCADE,
    image      BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- One row per successful attendance session. Every gate value is kept so a
-- disputed mark can be audited and thresholds re-tuned from real data.
CREATE TABLE IF NOT EXISTS attendance (
    id              INTEGER PRIMARY KEY,
    person_id       INTEGER REFERENCES people(id) ON DELETE SET NULL,
    name            TEXT NOT NULL,
    employee_id     TEXT NOT NULL,
    employee_number TEXT NOT NULL,
    similarity      REAL NOT NULL,
    liveness        REAL NOT NULL,   -- median over counted frames
    liveness_min    REAL NOT NULL,
    detection_score REAL NOT NULL,   -- minimum over counted frames
    face_ratio      REAL NOT NULL,   -- median face width / shorter image side
    frames          INTEGER NOT NULL,
    challenge       TEXT,            -- 'left' / 'right', NULL when disabled
    mode            TEXT NOT NULL DEFAULT 'single',   -- 'single' or 'group'
    image           BLOB,            -- face crop of the best frame
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS attendance_employee_time
    ON attendance (employee_id, created_at);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO meta (key, value) VALUES ('version', '0');
"""

_PERSON_COLS = ("id", "name", "employee_id", "employee_number", "created_at")
_ATTENDANCE_COLS = ("id", "name", "employee_id", "employee_number", "similarity",
                    "liveness", "liveness_min", "detection_score", "face_ratio",
                    "frames", "challenge", "mode", "created_at")


class FaceStore:
    def __init__(self, db_path=None):
        self.db_path = Path(config.paths.database if db_path is None else db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # shared by request threads; the lock serialises access
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        """Columns added after a database was first created. CREATE TABLE IF
        NOT EXISTS leaves an existing table as it was, so add them here."""
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(attendance)")}
        if "mode" not in cols:
            self.conn.execute(
                "ALTER TABLE attendance ADD COLUMN mode TEXT NOT NULL DEFAULT 'single'")

    # -- face index ----------------------------------------------------------

    def version(self):
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key='version'").fetchone()
        return int(row[0])

    def _bump_version(self):
        self.conn.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='version'")

    def find_conflict(self, name, employee_id, employee_number):
        """Name of another person already using this employee_id or
        employee_number, or None. Re-registering the same name is allowed."""
        with self._lock:
            row = self.conn.execute(
                "SELECT name FROM people WHERE (employee_id = ? OR employee_number = ?) "
                "AND name != ?", (employee_id, employee_number, name)).fetchone()
        return row[0] if row else None

    def save_person(self, name, employee_id, employee_number, vectors, image=None):
        """Store a registration in one transaction: the person row (created or
        updated), all embeddings, and the reference image."""
        with self._lock, self.conn:
            row = self.conn.execute("SELECT id FROM people WHERE name = ?", (name,)).fetchone()
            try:
                if row is None:
                    person_id = self.conn.execute(
                        "INSERT INTO people (name, employee_id, employee_number) "
                        "VALUES (?, ?, ?)", (name, employee_id, employee_number)).lastrowid
                else:
                    person_id = row[0]
                    self.conn.execute(
                        "UPDATE people SET employee_id = ?, employee_number = ? WHERE id = ?",
                        (employee_id, employee_number, person_id))
            except sqlite3.IntegrityError as e:
                raise ValueError("employee_id or employee_number already used "
                                 "by another person") from e
            # re-registering replaces the old samples rather than piling on
            self.conn.execute("DELETE FROM embeddings WHERE person_id = ?", (person_id,))
            self.conn.executemany(
                "INSERT INTO embeddings (person_id, vector) VALUES (?, ?)",
                [(person_id, np.asarray(v, dtype=np.float32).reshape(EMB_DIM).tobytes())
                 for v in vectors])
            if image is not None:
                self.conn.execute(
                    "INSERT INTO face_images (person_id, image) VALUES (?, ?) "
                    "ON CONFLICT(person_id) DO UPDATE SET image = excluded.image, "
                    "created_at = CURRENT_TIMESTAMP", (person_id, image))
            self._bump_version()
        return person_id

    def load_all(self):
        """(people, embeddings): a list of person dicts, one per embedding row,
        and a parallel (N, 512) float32 array."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT p.id, p.name, p.employee_id, p.employee_number, e.vector "
                "FROM embeddings e JOIN people p ON p.id = e.person_id "
                "ORDER BY e.id").fetchall()
        people = [{"person_id": r[0], "name": r[1], "employee_id": r[2],
                   "employee_number": r[3]} for r in rows]
        if not rows:
            return people, np.zeros((0, EMB_DIM), dtype=np.float32)
        vectors = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32)
        return people, vectors.reshape(len(rows), EMB_DIM).copy()

    # -- people ----------------------------------------------------------------

    def list_people(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT p.id, p.name, p.employee_id, p.employee_number, p.created_at, "
                "COUNT(e.id), EXISTS(SELECT 1 FROM face_images f WHERE f.person_id = p.id) "
                "FROM people p LEFT JOIN embeddings e ON e.person_id = p.id "
                "GROUP BY p.id ORDER BY p.name").fetchall()
        return [{**dict(zip(_PERSON_COLS, r[:5])), "samples": r[5],
                 "has_image": bool(r[6])} for r in rows]

    def get_person(self, employee_id):
        return next((p for p in self.list_people() if p["employee_id"] == employee_id), None)

    def get_face_image(self, employee_id):
        with self._lock:
            row = self.conn.execute(
                "SELECT f.image FROM face_images f JOIN people p ON p.id = f.person_id "
                "WHERE p.employee_id = ?", (employee_id,)).fetchone()
        return row[0] if row else None

    def delete_person(self, name):
        with self._lock, self.conn:
            cur = self.conn.execute("DELETE FROM people WHERE name = ?", (name,))
            if cur.rowcount:
                self._bump_version()
        return cur.rowcount > 0

    # -- attendance ------------------------------------------------------------

    def add_attendance(self, person, similarity, liveness, liveness_min,
                       detection_score, face_ratio, frames, challenge=None, image=None,
                       mode="single"):
        with self._lock, self.conn:
            attendance_id = self.conn.execute(
                "INSERT INTO attendance (person_id, name, employee_id, employee_number, "
                "similarity, liveness, liveness_min, detection_score, face_ratio, frames, "
                "challenge, image, mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (person["person_id"], person["name"], person["employee_id"],
                 person["employee_number"], float(similarity), float(liveness),
                 float(liveness_min), float(detection_score), float(face_ratio),
                 int(frames), challenge, image, mode)).lastrowid
        return self.get_attendance(attendance_id)

    def get_attendance(self, attendance_id):
        with self._lock:
            row = self.conn.execute(
                f"SELECT {', '.join(_ATTENDANCE_COLS)} FROM attendance WHERE id = ?",
                (attendance_id,)).fetchone()
        return dict(zip(_ATTENDANCE_COLS, row)) if row else None

    def last_attendance(self, employee_id, within_s):
        """The employee's newest record if younger than `within_s`, else None."""
        with self._lock:
            row = self.conn.execute(
                f"SELECT {', '.join(_ATTENDANCE_COLS)} FROM attendance "
                "WHERE employee_id = ? AND created_at >= datetime('now', ?) "
                "ORDER BY id DESC LIMIT 1",
                (employee_id, f"-{int(within_s)} seconds")).fetchone()
        return dict(zip(_ATTENDANCE_COLS, row)) if row else None

    def list_attendance(self, limit=100, day=None, employee_id=None):
        """Newest first. `day` is YYYY-MM-DD in the server's local time zone;
        created_at is stored in UTC."""
        where, args = [], []
        if day:
            where.append("date(created_at, 'localtime') = ?")
            args.append(day)
        if employee_id:
            where.append("employee_id = ?")
            args.append(employee_id)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {', '.join(_ATTENDANCE_COLS)} FROM attendance{clause} "
                "ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [dict(zip(_ATTENDANCE_COLS, r)) for r in rows]

    def get_attendance_image(self, attendance_id):
        with self._lock:
            row = self.conn.execute(
                "SELECT image FROM attendance WHERE id = ?", (attendance_id,)).fetchone()
        return row[0] if row and row[0] else None
