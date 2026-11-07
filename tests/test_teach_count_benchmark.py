"""Teaching new poses, finger counting and the speed benchmark."""
import json

import numpy as np
import pytest

import gesture_analyzer as ga
import ui
from benchmark import StageTimer
from feature_extractor import extract_features
from hand_detector import DetectedHand
from synthetic_hand import ALL_FINGERS, gesture_hand, make_hand, rotate_3d
from teach import Teacher, TaughtGestures, WithTaughtGestures

rng = np.random.default_rng(7)

# The demo poses: none of them is a built-in gesture.
DEMO_POSES = {
    "PINKY PROMISE": lambda: make_hand(("pinky",)),
    "GERMAN THREE": lambda: make_hand(("index", "middle"), thumb="out"),
    "MIDDLE THREE": lambda: make_hand(("middle", "ring", "pinky")),
}


def features(pts, handedness="Right"):
    return extract_features(pts, pts, handedness)


def wobble(pts, tilt=10.0, noise=0.01):
    pts = rotate_3d(pts, rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt))
    return pts + rng.normal(0.0, noise, pts.shape)


def example(pts):
    return {"handedness": "Right", "world_landmarks": pts.tolist(), "image_points": pts.tolist()}


def detected(pts):
    return DetectedHand("Right", 0.98, pts, pts, pts)


def taught_demo(folder=None):
    taught = TaughtGestures(folder=folder)
    for name, build in DEMO_POSES.items():
        taught.add(name, [example(wobble(build())) for _ in range(30)], author="tester")
    return taught


# --- Taught gestures -----------------------------------------------------------------

@pytest.mark.parametrize("name", DEMO_POSES)
def test_taught_pose_is_recognised(name):
    taught = taught_demo()
    for _ in range(10):
        assert taught.match(features(wobble(DEMO_POSES[name](), tilt=20, noise=0.015))) == name


@pytest.mark.parametrize("gesture", ga.GESTURES)
def test_taught_poses_dont_take_over_built_in_gestures(gesture):
    classifier = WithTaughtGestures(ga.RuleBasedClassifier(), taught_demo())
    for curl in (0.7, 1.0):
        assert classifier.classify(features(gesture_hand(gesture, curl=curl))) == gesture


def test_rules_said_unknown_before_teaching():
    rules = ga.RuleBasedClassifier()
    for build in DEMO_POSES.values():
        assert rules.classify(features(build())) == ga.UNKNOWN


def test_one_file_per_pose_and_loaded_again(tmp_path):
    taught_demo(tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [
        "german_three.json", "middle_three.json", "pinky_promise.json"]
    data = json.loads((tmp_path / "pinky_promise.json").read_text(encoding="utf-8"))
    assert data["name"] == "PINKY PROMISE" and data["taught_by"] == ["tester"]
    assert len(data["examples"]) == 30 and np.array(data["examples"][0]["world_landmarks"]).shape == (21, 3)
    again = TaughtGestures(folder=tmp_path)   # e.g. a teammate after git pull
    assert again.names == sorted(DEMO_POSES)
    assert again.match(features(DEMO_POSES["PINKY PROMISE"]())) == "PINKY PROMISE"


def test_broken_pose_files_are_skipped_and_the_rest_still_work(tmp_path):
    taught_demo(tmp_path)
    (tmp_path / "conflict.json").write_text('{"name": "OOPS", <<<<<<< HEAD', encoding="utf-8")   # bad JSON
    (tmp_path / "no_examples.json").write_text('{"name": "EMPTY"}', encoding="utf-8")
    (tmp_path / "wrong_shape.json").write_text(
        '{"name": "BENT", "examples": [{"world_landmarks": [[0, 0]], "image_points": [[0, 0]]}]}', encoding="utf-8")
    (tmp_path / "no_name.json").write_text('{"name": "", "examples": []}', encoding="utf-8")
    taught = TaughtGestures(folder=tmp_path)
    assert taught.names == sorted(DEMO_POSES)
    assert sorted(s.split()[0] for s in taught.skipped) == [
        "conflict.json", "no_examples.json", "no_name.json", "wrong_shape.json"]
    assert taught.match(features(DEMO_POSES["PINKY PROMISE"]())) == "PINKY PROMISE"


def test_teaching_again_adds_examples_and_author(tmp_path):
    taught = TaughtGestures(folder=tmp_path)
    taught.add("PINKY PROMISE", [example(wobble(make_hand(("pinky",)))) for _ in range(5)], author="alex")
    taught.add("PINKY PROMISE", [example(wobble(make_hand(("pinky",)))) for _ in range(5)], author="sam")
    data = json.loads((tmp_path / "pinky_promise.json").read_text(encoding="utf-8"))
    assert data["taught_by"] == ["alex", "sam"] and len(data["examples"]) == 10
    assert TaughtGestures(folder=tmp_path).taught_by["PINKY PROMISE"] == ["alex", "sam"]


def test_nothing_taught_matches_nothing(tmp_path):
    assert TaughtGestures(folder=tmp_path / "empty").match(features(make_hand())) is None


# --- The teaching flow (t key) ---------------------------------------------------------

def type_name(teacher, text, now=0.0):
    for ch in text:
        teacher.handle_key(ord(ch), now)


def test_teach_flow_learns_and_saves_a_pose(tmp_path):
    taught = TaughtGestures(folder=tmp_path)
    teacher = Teacher(taught, author="alex", examples=10, countdown=3.0)
    teacher.start()
    assert teacher.naming
    type_name(teacher, "pinky promisx")
    teacher.handle_key(8, 0.0)          # backspace
    type_name(teacher, "e")
    assert teacher.name == "PINKY PROMISE"
    assert teacher.handle_key(13, 0.0) is None  # Enter: countdown starts
    assert "get ready" in teacher.status_line(1.0)

    hand = detected(DEMO_POSES["PINKY PROMISE"]())
    assert teacher.update([hand], 1.0) is None and not taught.names   # still counting down
    assert teacher.update([hand, hand], 3.5) is None                  # two hands: not recorded
    assert "recording 0/10" in teacher.status_line(3.5)
    messages = [teacher.update([hand], 3.5 + i / 30) for i in range(1, 25)]
    assert "LEARNED PINKY PROMISE" in messages
    assert taught.names == ["PINKY PROMISE"] and not teacher.active
    assert teacher.saved_to == tmp_path / "pinky_promise.json"
    assert json.loads(teacher.saved_to.read_text(encoding="utf-8"))["taught_by"] == ["alex"]


def test_teach_rejects_built_in_names_and_empty_names():
    teacher = Teacher(TaughtGestures(folder=None))
    teacher.start()
    assert teacher.handle_key(13, 0.0) is None and teacher.naming      # empty: ignored
    type_name(teacher, "fist")
    assert teacher.handle_key(13, 0.0) == "NAME TAKEN, PICK ANOTHER" and teacher.naming


def test_escape_cancels_and_keys_are_letters_while_naming():
    teacher = Teacher(TaughtGestures(folder=None))
    teacher.start()
    type_name(teacher, "q h")             # would be quit / history outside naming
    assert teacher.name == "Q H"
    assert teacher.handle_key(27, 0.0) == "TEACH CANCELLED" and not teacher.active


# --- Finger counting --------------------------------------------------------------------

def test_counts_fingers_on_one_hand():
    assert ui.finger_count_text([features(make_hand(("index", "middle")))]) == "Fingers up: 2"


def test_counts_fingers_on_two_hands():
    left = features(make_hand(ALL_FINGERS, thumb="out"), "Left")
    right = features(make_hand(("index", "middle")), "Right")
    assert ui.finger_count_text([left, right]) == "Fingers up: 7 (LEFT 5 + RIGHT 2)"


def test_finger_count_in_result_panel():
    f = features(make_hand(("index",)))
    result = ga.GestureResult("Right", ga.ONE_FINGER, ga.ONE_FINGER, ga.STABLE, 9, 10, False)
    assert "Fingers up: 1" in [t for t, _ in ui.result_lines(1, [result], None, [f])]


# --- Benchmark ----------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_stage_timer_averages_and_summary():
    clock = FakeClock()
    timer = StageTimer(clock=clock)
    for _ in range(4):
        timer.start()
        clock.t += 0.030
        timer.lap("camera")
        clock.t += 0.020
        timer.lap("hand detection")
        clock.t += 0.001
        timer.lap("drawing")
        timer.end_frame()
    averages = timer.averages_ms()
    assert averages["hand detection"] == pytest.approx(20.0)
    assert averages["camera"] == pytest.approx(30.0)
    summary = timer.summary()
    assert "4 frames" in summary and "| hand detection | 20.00 | 39% |" in summary
    assert "without waiting for the camera 21.0 ms" in summary
    assert timer.live_line() == "SPEED: 51.0 ms/frame, hand detection 20.0 ms"


def test_empty_timer():
    timer = StageTimer()
    assert timer.summary() == "No frames measured."
    assert timer.live_line() == "SPEED: measuring..."


# --- Taught poses vs each other, and taught letters --------------------------------------

def test_different_taught_poses_dont_clash():
    assert taught_demo().clashes() == []


def test_two_names_for_the_same_pose_clash():
    taught = taught_demo()
    taught.add("LITTLE FINGER", [example(wobble(make_hand(("pinky",)))) for _ in range(30)], author="sam")
    clashes = taught.clashes()
    assert any(c.startswith("LITTLE FINGER looks like PINKY PROMISE") for c in clashes)
    assert any(c.startswith("PINKY PROMISE looks like LITTLE FINGER") for c in clashes)


def test_taught_letter_is_spelled_in_sign_mode():
    from sign_language import SignSpeller

    speller = SignSpeller()
    for gesture in ("K", "FIST", "PINKY PROMISE", "I"):  # taught K, built-in S, a taught non-letter, taught I
        speller.update([ga.GestureResult("Right", gesture, gesture, ga.STABLE, 9, 10, True)], [])
    assert speller.text == "KSI"
