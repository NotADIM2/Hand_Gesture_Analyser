"""Hand exercise coach, sign-to-speech and the speaker."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import gesture_analyzer as ga
from database import GestureDatabase
from exercise_coach import CountToFive, ExerciseCoach, OpenClose, ThumbTouches
from feature_extractor import extract_features
from hand_detector import DetectedHand
from sign_language import SignSpeller
from speech import Speaker
from synthetic_hand import ALL_FINGERS, make_hand

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import session_report  # noqa: E402

FINGER_TIP = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}


def hand_and_features(pts):
    return DetectedHand("Right", 0.98, pts, pts, pts), extract_features(pts, pts, "Right")


OPEN = hand_and_features(make_hand(ALL_FINGERS, thumb="out"))
FIST = hand_and_features(make_hand())
HALF = hand_and_features(make_hand(curl=0.35))   # fingers only a little bent


def touching(finger):
    pts = make_hand(ALL_FINGERS)
    pts[4] = pts[FINGER_TIP[finger]] + [0.05, 0.0, 0.0]
    return hand_and_features(pts)


def feed(exercise, poses, frames_each=1):
    reps = 0
    for hand, f in poses:
        for _ in range(frames_each):
            reps += exercise.update(hand, f)
    return reps


# --- Open and close ----------------------------------------------------------------

def test_open_close_counts_each_fist_then_open():
    ex = OpenClose()
    assert feed(ex, [OPEN, FIST, OPEN, FIST, OPEN]) == 2
    assert "opens to 1" in ex.detail() and "closes to" in ex.detail()


def test_starting_with_a_fist_or_half_closing_doesnt_count():
    assert feed(OpenClose(), [FIST, OPEN]) == 0          # no open -> fist -> open yet
    assert feed(OpenClose(), [OPEN, HALF, OPEN, HALF, OPEN]) == 0


# --- Thumb touches -----------------------------------------------------------------------

def test_thumb_touches_count_a_full_round_in_order():
    ex = ThumbTouches()
    poses = []
    for finger in ("index", "middle", "ring", "pinky"):
        poses += [touching(finger), OPEN]
    assert feed(ex, poses) == 1
    assert ex.next_finger == "index"


def test_thumb_touches_wrong_order_does_not_advance():
    ex = ThumbTouches()
    feed(ex, [touching("middle"), OPEN, touching("pinky"), OPEN])
    assert ex.next_finger == "index"


# --- Count to five ------------------------------------------------------------------------

def counts(*sequence):
    shapes = {1: (("index",), "folded"), 2: (("index", "middle"), "folded"),
              3: (("index", "middle", "ring"), "folded"), 4: (ALL_FINGERS, "folded"), 5: (ALL_FINGERS, "out")}
    return [hand_and_features(make_hand(shapes[n][0], thumb=shapes[n][1])) for n in sequence]


def test_count_to_five_in_order():
    assert feed(CountToFive(), counts(1, 2, 3, 4, 5), frames_each=5) == 1


def test_count_to_five_needs_each_number_held_and_in_order():
    assert feed(CountToFive(), counts(1, 2, 3, 4, 5), frames_each=2) == 0   # too quick
    assert feed(CountToFive(), counts(1, 3, 5), frames_each=5) == 0         # skipped numbers


# --- The routine ----------------------------------------------------------------------------

def test_routine_moves_through_exercises_and_reports_results():
    coach = ExerciseCoach(routine=(("OPEN AND CLOSE", 2), ("COUNT TO FIVE", 1)))
    assert coach.start().startswith("OPEN AND CLOSE")
    messages, t = [], 0.0
    for hand, f in [OPEN, FIST, OPEN, FIST, OPEN] + [p for p in counts(1, 2, 3, 4, 5) for _ in range(5)]:
        t += 0.1
        messages += coach.update([hand], [f], t)
    assert messages == ["1/2", "2/2", "NEXT: COUNT TO FIVE", "1/1", "ROUTINE DONE"]
    results = coach.take_results()
    assert [r.exercise for r in results] == ["OPEN AND CLOSE", "COUNT TO FIVE"]
    assert results[0].reps == 2 and results[0].seconds > 0 and results[0].seconds_per_rep > 0
    assert coach.take_results() == []        # each result handed out once
    assert coach.status_lines()[0] == "EXERCISES DONE:"


def test_no_hand_no_progress_and_status_shows_instruction():
    coach = ExerciseCoach()
    coach.start()
    assert coach.update([], [], 1.0) == []
    assert coach.status_lines() == ["EXERCISE 1/3: OPEN AND CLOSE  0/10", "open your hand wide, then make a fist"]


def test_exercise_results_saved_and_in_report(tmp_path):
    db_path = tmp_path / "h.db"
    with GestureDatabase(db_path) as db:
        db.insert_exercise("OPEN AND CLOSE", 10, 15.0, "opens to 170 deg, closes to 60 deg", "s1",
                           datetime(2026, 11, 7, 10, 0))
        db.insert_exercise("OPEN AND CLOSE", 10, 9.0, "", "s2", datetime(2026, 11, 8, 10, 0))
        db.flush()
        saved = db.get_exercises()
    assert [(e.exercise, e.reps, e.seconds) for e in saved] == [("OPEN AND CLOSE", 10, 15.0), ("OPEN AND CLOSE", 10, 9.0)]
    out = tmp_path / "r.html"
    session_report.main(["--db", str(db_path), "--out", str(out)])
    page = out.read_text(encoding="utf-8")
    assert "Hand exercises" in page and "opens to 170 deg" in page and ">1.5 s<" in page and ">0.9 s<" in page


# --- Sign-to-speech ---------------------------------------------------------------------------

def pose(gesture):
    return ga.GestureResult("Right", gesture, gesture, ga.STABLE, 9, 10, True)


def test_space_finishes_a_word_and_thumbs_up_says_everything():
    speller = SignSpeller()
    for g in ("FOUR FINGERS", "TWO FINGERS", "CALL ME"):     # B U Y
        speller.update([pose(g)], [])
    assert speller.update([pose("OPEN PALM")], []) == "SPACE" and speller.last_word == "BUY"
    assert speller.update([pose("THUMBS UP")], []) == "SAY" and speller.text == "BUY "


def test_thumbs_up_with_nothing_spelled_says_nothing():
    assert SignSpeller().update([pose("THUMBS UP")], []) is None


def test_speaker_passes_text_on_stdin(tmp_path):
    out = tmp_path / "spoken.txt"
    script = f"import sys; open({str(out)!r}, 'a').write(sys.stdin.read() + '|')"
    speaker = Speaker(command=[sys.executable, "-c", script])
    speaker.say("hello")
    speaker.say("  ")          # blank: ignored
    speaker.say("buy; rm -rf")  # never part of the command, so nothing to inject into
    speaker.wait()
    assert out.read_text() == "hello|buy; rm -rf|"


def test_muted_speaker_does_nothing():
    speaker = Speaker(enabled=False)
    assert not speaker.available
    speaker.say("hello")   # no error, no thread
