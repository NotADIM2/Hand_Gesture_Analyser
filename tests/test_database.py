import time
from datetime import datetime

import pytest

from database import GestureDatabase


@pytest.fixture
def db(tmp_path):
    database = GestureDatabase(tmp_path / "history.db")
    yield database
    database.close()


def test_insert_and_read_back(db):
    db.insert("FIST", "Right", "s1", timestamp=datetime(2026, 11, 7, 10, 30, 0))
    db.flush()
    [record] = db.get_history()
    assert (record.gesture, record.hand, record.session_id) == ("FIST", "Right", "s1")
    assert record.timestamp == "2026-11-07T10:30:00"


def test_history_newest_first_and_limited(db):
    for g in ["A", "B", "C", "D"]:
        db.insert(g)
    db.flush()
    assert [r.gesture for r in db.get_history(limit=3)] == ["D", "C", "B"]


def test_count_gestures(db):
    for g in ["FIST", "PEACE", "FIST", "FIST", "PEACE", "OK SIGN"]:
        db.insert(g, session_id="s1")
    db.insert("FIST", session_id="s2")
    db.flush()
    assert db.count_gestures() == {"FIST": 4, "PEACE": 2, "OK SIGN": 1}
    assert db.count_gestures("s1") == {"FIST": 3, "PEACE": 2, "OK SIGN": 1}
    assert db.total("s2") == 1


def test_filter_by_session(db):
    db.insert("FIST", session_id="s1")
    db.insert("PEACE", session_id="s2")
    db.flush()
    assert [r.gesture for r in db.get_history(session_id="s2")] == ["PEACE"]


def test_clear_all_and_by_session(db):
    for s in ["s1", "s1", "s2"]:
        db.insert("FIST", session_id=s)
    db.clear("s1")
    assert db.total() == 1
    db.clear()
    assert db.get_history() == []


def test_insert_does_not_block(db):
    start = time.perf_counter()
    for _ in range(500):
        db.insert("FIST")
    queued_in = time.perf_counter() - start
    db.flush()
    assert db.total() == 500
    assert queued_in < 0.1  # writes happen on the background thread


def test_close_saves_pending_writes(tmp_path):
    path = tmp_path / "history.db"
    database = GestureDatabase(path)
    for _ in range(50):
        database.insert("PEACE")
    database.close()
    reopened = GestureDatabase(path)
    try:
        assert reopened.total() == 50
    finally:
        reopened.close()


def test_insert_after_close_raises(tmp_path):
    database = GestureDatabase(tmp_path / "history.db")
    database.close()
    with pytest.raises(RuntimeError):
        database.insert("FIST")
