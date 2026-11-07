import sys
from datetime import datetime
from pathlib import Path

from database import GestureDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import session_report  # noqa: E402


def make_db(path):
    with GestureDatabase(path) as db:
        db.insert("FIST", "Right", "s1", datetime(2026, 11, 7, 10, 0, 0))
        db.insert("FIST", "Left", "s1", datetime(2026, 11, 7, 10, 2, 0))
        db.insert("SWIPE LEFT", "Right", "s1", datetime(2026, 11, 7, 10, 3, 0))
        db.insert("HEART", "Both", "s2", datetime(2026, 11, 7, 11, 0, 0))
        db.flush()
    return path


def test_report_from_history(tmp_path):
    out = tmp_path / "report.html"
    assert session_report.main(["--db", str(make_db(tmp_path / "h.db")), "--out", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    for text in ("FIST", "SWIPE LEFT", "HEART", "BOTH", ">4<", ">2<"):  # gestures, hands, totals, sessions
        assert text in page
    assert 'class="bar moving"' in page and 'class="bar two-hand"' in page


def test_latest_session_only(tmp_path):
    out = tmp_path / "latest.html"
    session_report.main(["--db", str(make_db(tmp_path / "h.db")), "--session", "latest", "--out", str(out)])
    page = out.read_text(encoding="utf-8")
    assert "HEART" in page and "SWIPE LEFT" not in page


def test_sessions_grouped_with_duration(tmp_path):
    sessions = session_report.group_sessions(session_report.load_records(make_db(tmp_path / "h.db")))
    assert [s.session_id for s in sessions] == ["s1", "s2"]
    assert sessions[0].minutes == 3.0
    assert sessions[0].top(1) == [("FIST", 2)]


def test_empty_history(tmp_path):
    page = session_report.build_report(session_report.load_records(tmp_path / "missing.db"))
    assert "Nothing recorded yet" in page and "no sessions yet" in page


def test_names_are_escaped():
    assert "<script>" not in session_report.bar_chart([("<script>", 1, "pose")])
