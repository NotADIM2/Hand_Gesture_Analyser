"""SQLite gesture history (architecture step: GESTURE HISTORY).

Writes go through a queue to one background thread that owns the write connection,
so saving a record never blocks the real-time camera loop. Reads open their own
short-lived connection; the database runs in WAL mode so reads and writes don't
block each other.
"""
import logging
import queue
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import config

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS gesture_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    gesture     TEXT NOT NULL,
    hand        TEXT,
    timestamp   TEXT NOT NULL,
    session_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_history_session ON gesture_history(session_id);
"""

_STOP = object()


@dataclass(frozen=True)
class GestureRecord:
    id: int
    gesture: str
    hand: str | None
    timestamp: str
    session_id: str | None


class GestureDatabase:
    def __init__(self, path: Path | str = config.DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
        conn.close()

        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._writer = threading.Thread(target=self._write_loop, name="gesture-db-writer", daemon=True)
        self._writer.start()

    # --- writes (queued, non-blocking) -----------------------------------------
    def insert(
        self,
        gesture: str,
        hand: str | None = None,
        session_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Queue a detection record. Returns immediately."""
        when = (timestamp or datetime.now()).isoformat(timespec="seconds")
        self._submit(
            "INSERT INTO gesture_history (gesture, hand, timestamp, session_id) VALUES (?, ?, ?, ?)",
            (gesture, hand, when, session_id),
        )

    def clear(self, session_id: str | None = None) -> None:
        """Delete all history, or only one session's. Waits until it is done."""
        if session_id is None:
            self._submit("DELETE FROM gesture_history", ())
        else:
            self._submit("DELETE FROM gesture_history WHERE session_id = ?", (session_id,))
        self.flush()

    def flush(self) -> None:
        """Block until every queued write has been saved."""
        self._queue.join()

    # --- reads -----------------------------------------------------------------
    def get_history(self, limit: int = 50, session_id: str | None = None) -> list[GestureRecord]:
        """Most recent records first."""
        sql = "SELECT id, gesture, hand, timestamp, session_id FROM gesture_history"
        params: tuple = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params = (session_id,)
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self._read(sql, params + (limit,))
        return [GestureRecord(*row) for row in rows]

    def count_gestures(self, session_id: str | None = None) -> dict[str, int]:
        """Gesture name -> number of detections, most frequent first."""
        sql = "SELECT gesture, COUNT(*) FROM gesture_history"
        params: tuple = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params = (session_id,)
        sql += " GROUP BY gesture ORDER BY COUNT(*) DESC, gesture"
        return dict(self._read(sql, params))

    def total(self, session_id: str | None = None) -> int:
        return sum(self.count_gestures(session_id).values())

    # --- lifecycle ---------------------------------------------------------------
    def close(self) -> None:
        """Save anything still queued, then stop the writer thread."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._writer.join(timeout=5)

    def __enter__(self) -> "GestureDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- internals ---------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def _read(self, sql: str, params: tuple) -> list[tuple]:
        conn = self._connect()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def _submit(self, sql: str, params: tuple) -> None:
        if self._closed:
            raise RuntimeError("GestureDatabase is closed")
        self._queue.put((sql, params))

    def _write_loop(self) -> None:
        conn = self._connect()
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    sql, params = item
                    with conn:  # commits, or rolls back on error
                        conn.execute(sql, params)
                except sqlite3.Error:
                    log.exception("Failed to write gesture history")
                finally:
                    self._queue.task_done()
        finally:
            conn.close()
