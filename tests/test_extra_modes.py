"""Gesture password, sign letters and the accuracy report."""
import sys
from pathlib import Path

import numpy as np
import pytest

import gesture_analyzer as ga
import motion_analyzer as ma
from controller import GesturePassword
from sign_language import SignSpeller
from synthetic_hand import make_hand

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate  # noqa: E402


def pose(gesture, newly_stable=True, hand="Right"):
    return ga.GestureResult(hand, gesture, gesture, ga.STABLE, 9, 10, newly_stable)


# --- Gesture password ----------------------------------------------------------

PASSWORD = ("FIST", "PEACE", "THUMBS UP")


def enter(password, gestures, start=0.0, step=1.0):
    unlocked_at = []
    for i, g in enumerate(gestures):
        if password.update([pose(g)], start + i * step):
            unlocked_at.append(i)
    return unlocked_at


def test_password_unlocks_in_order():
    pw = GesturePassword(PASSWORD)
    assert not pw.unlocked
    assert enter(pw, PASSWORD) == [2]
    assert pw.unlocked


def test_wrong_order_stays_locked():
    pw = GesturePassword(PASSWORD)
    enter(pw, ("PEACE", "FIST", "THUMBS UP"))
    assert not pw.unlocked


def test_wrong_pose_restarts_the_sequence():
    pw = GesturePassword(PASSWORD)
    enter(pw, ("FIST", "ROCK", "THUMBS UP"))
    assert not pw.unlocked and pw.progress == 0
    enter(pw, ("FIST", "PEACE", "FIST", "PEACE", "THUMBS UP"))
    assert pw.unlocked  # starting again with FIST works


def test_too_slow_restarts_the_sequence():
    pw = GesturePassword(PASSWORD, step_seconds=4.0)
    enter(pw, PASSWORD, step=5.0)
    assert not pw.unlocked


def test_flicker_of_the_same_pose_is_ignored():
    pw = GesturePassword(PASSWORD)
    enter(pw, ("FIST", "PEACE", "PEACE", "THUMBS UP"))
    assert pw.unlocked


def test_only_newly_stable_poses_count():
    pw = GesturePassword(PASSWORD)
    for i, g in enumerate(PASSWORD):
        pw.update([pose(g, newly_stable=False)], float(i))
    assert not pw.unlocked


def test_lock_and_no_password():
    pw = GesturePassword(PASSWORD)
    enter(pw, PASSWORD)
    pw.lock()
    assert not pw.unlocked
    assert GesturePassword(()).unlocked  # empty password = always unlocked


# --- Sign letters ------------------------------------------------------------------

def spell(speller, gestures):
    for g in gestures:
        speller.update([pose(g)], [])


def test_spells_letters():
    speller = SignSpeller()
    spell(speller, ["L SIGN", "OPEN PALM", "PEACE", "CALL ME"])
    assert speller.text == "L VY"


def test_unmapped_pose_adds_nothing():
    speller = SignSpeller()
    spell(speller, ["ROCK", "THUMBS UP"])
    assert speller.text == ""


def test_no_leading_or_double_spaces():
    speller = SignSpeller()
    spell(speller, ["OPEN PALM", "FIST", "OPEN PALM", "OPEN PALM"])
    assert speller.text == "S "


def test_swipe_left_deletes_and_wave_clears():
    speller = SignSpeller()
    spell(speller, ["FIST", "OK SIGN", "PEACE"])
    assert speller.update([], [ma.MotionEvent(ma.SWIPE_LEFT, "Right", 0)]) == "DELETE"
    assert speller.text == "SF"
    assert speller.update([], [ma.MotionEvent(ma.WAVE, "Right", 0)]) == "CLEAR"
    assert speller.text == ""


def test_text_keeps_the_latest_letters():
    speller = SignSpeller(max_length=3)
    spell(speller, ["FIST", "OK SIGN", "PEACE", "CALL ME"])
    assert speller.text == "FVY"


def test_sign_letters_use_known_gestures():
    speller = SignSpeller()
    assert set(speller.letters) <= set(ga.GESTURES)
    assert len(set(speller.letters.values())) == len(speller.letters)  # no letter twice


# --- Accuracy report -----------------------------------------------------------------

def sample(gesture, pts, person="alex"):
    return {"gesture": gesture, "world_landmarks": pts, "image_points": pts,
            "handedness": "Right", "source": "recorded", "person": person}


def test_evaluation_counts_and_confusion():
    samples = [
        sample("FIST", make_hand()),
        sample("ONE FINGER", make_hand(("index",))),
        sample("PEACE", make_hand(("index", "middle"), spread=0)),  # actually TWO FINGERS
    ]
    ev = evaluate.evaluate(samples)
    assert (ev.correct, ev.total) == (2, 3)
    assert ev.confusion[("PEACE", "TWO FINGERS")] == 1
    report = evaluate.format_report(ev)
    assert "**Overall: 2/3 correct (67%)**" in report
    assert "| PEACE | TWO FINGERS | 1 |" in report
    assert "| alex | 2 | 3 | 67% |" in report


def test_report_without_samples_explains_what_to_do():
    assert "record_samples.py" in evaluate.format_report(evaluate.evaluate([]))


def test_report_saved_as_markdown(tmp_path):
    out = tmp_path / "reports" / "accuracy.md"
    assert evaluate.main(["--save", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Gesture recognition accuracy")
