import pytest

import gesture_analyzer as ga
from conftest import load_real_samples
from feature_extractor import extract_features
from synthetic_hand import ALL_FINGERS, gesture_hand, make_hand, rotate_3d, rotate_in_image


def features(pts, handedness="Right"):
    return extract_features(pts, pts, handedness)


classifier = ga.RuleBasedClassifier()


def test_seventeen_gestures_defined():
    assert set(ga.GESTURES) == {
        ga.OPEN_PALM, ga.FIST, ga.THUMBS_UP, ga.THUMBS_DOWN, ga.ONE_FINGER,
        ga.TWO_FINGERS, ga.THREE_FINGERS, ga.PEACE, ga.OK_SIGN,
        ga.FOUR_FINGERS, ga.ROCK, ga.CALL_ME, ga.I_LOVE_YOU, ga.L_SIGN,
        ga.PINCH, ga.POINT_LEFT, ga.POINT_RIGHT,
    }
    assert len(ga.GESTURES) == 17


@pytest.mark.parametrize("gesture", ga.GESTURES)
def test_classifies_each_gesture(gesture):
    assert classifier.classify(features(gesture_hand(gesture))) == gesture


@pytest.mark.parametrize("gesture", ga.GESTURES)
@pytest.mark.parametrize("tilt", [(20, 0), (0, 30)])
def test_classification_survives_tilt(gesture, tilt):
    assert classifier.classify(features(rotate_3d(gesture_hand(gesture), *tilt))) == gesture


def test_thumb_sideways_is_not_thumbs_up():
    assert classifier.classify(features(make_hand(thumb="side"))) == ga.UNKNOWN


def test_unlisted_pattern_is_unknown():
    # Ring + pinky alone isn't one of the gestures.
    assert classifier.classify(features(make_hand(("ring", "pinky")))) == ga.UNKNOWN


def test_fist_with_thumb_near_index_is_not_pinch():
    # A fist's index finger is curled tight, so a nearby thumb doesn't make it a pinch.
    pts = make_hand()
    pts[4] = pts[8] + [-0.05, 0.0, 0.0]
    assert classifier.classify(features(pts)) == ga.FIST


@pytest.mark.parametrize("degrees", [0, 180])
def test_index_up_or_down_is_one_finger(degrees):
    assert classifier.classify(features(rotate_in_image(make_hand(("index",)), degrees))) == ga.ONE_FINGER


def test_upside_down_peace_still_peace():
    assert classifier.classify(features(rotate_in_image(gesture_hand(ga.PEACE), 180))) == ga.PEACE


def test_custom_rule_can_be_added():
    ring_pinky = ga.GestureRule("RING PINKY", lambda f: ga.only(f, "ring", "pinky"))
    custom = ga.RuleBasedClassifier(list(ga.DEFAULT_RULES) + [ring_pinky])
    assert custom.classify(features(make_hand(("ring", "pinky")))) == "RING PINKY"


@pytest.mark.parametrize(
    "sample",
    [s for s in load_real_samples() if s["gesture"]],
    ids=lambda s: f"{s['gesture']}:{s['source']}",
)
def test_real_samples(sample):
    f = extract_features(sample["world_landmarks"], sample["image_points"], sample["handedness"])
    assert classifier.classify(f) == sample["gesture"]


# --- GestureAnalyzer: smoothing and stability --------------------------------

class ScriptedClassifier:
    """Returns pre-set labels, one per call."""

    def __init__(self, labels):
        self.labels = iter(labels)

    def classify(self, features):
        return next(self.labels)


def right_hand():
    return features(make_hand(ALL_FINGERS, thumb="out"), "Right")


def run(analyzer, frames, hands_per_frame=None):
    hands = hands_per_frame or [right_hand()]
    return [analyzer.analyze(hands) for _ in range(frames)]


def test_becomes_stable_after_enough_agreeing_frames():
    analyzer = ga.GestureAnalyzer(ScriptedClassifier([ga.FIST] * 10), window=10, min_agreement=7)
    statuses = [r[0].status for r in run(analyzer, 10)]
    assert statuses[:6] == [ga.UNCERTAIN] * 6
    assert statuses[6:] == [ga.STABLE] * 4


def test_newly_stable_fires_once_per_gesture():
    labels = [ga.FIST] * 10 + [ga.OPEN_PALM] * 10
    analyzer = ga.GestureAnalyzer(ScriptedClassifier(labels), window=10, min_agreement=7)
    results = [r[0] for r in run(analyzer, 20)]
    logged = [(i, r.gesture) for i, r in enumerate(results) if r.newly_stable]
    assert [g for _, g in logged] == [ga.FIST, ga.OPEN_PALM]


def test_flicker_does_not_change_stable_gesture():
    labels = [ga.FIST] * 10 + [ga.OPEN_PALM, ga.FIST] * 3
    analyzer = ga.GestureAnalyzer(ScriptedClassifier(labels), window=10, min_agreement=7)
    results = [r[0] for r in run(analyzer, 16)]
    assert all(r.gesture == ga.FIST for r in results[10:])
    assert sum(r.newly_stable for r in results) == 1


def test_unknown_reported_as_unrecognised_and_rearms_logging():
    labels = [ga.FIST] * 7 + [ga.UNKNOWN] * 10 + [ga.FIST] * 10
    analyzer = ga.GestureAnalyzer(ScriptedClassifier(labels), window=7, min_agreement=5)
    results = [r[0] for r in run(analyzer, 27)]
    assert results[16].status == ga.UNRECOGNISED
    assert sum(r.newly_stable for r in results) == 2  # fist, then fist again after the pause


def test_two_hands_tracked_separately():
    left = features(gesture_hand(ga.FIST), "Left")
    right = features(gesture_hand(ga.OPEN_PALM), "Right")
    analyzer = ga.GestureAnalyzer(window=5, min_agreement=3)
    results = run(analyzer, 5, [left, right])[-1]
    assert [(r.hand, r.gesture, r.status) for r in results] == [
        ("Left", ga.FIST, ga.STABLE),
        ("Right", ga.OPEN_PALM, ga.STABLE),
    ]


def test_hand_leaving_frame_resets_its_history():
    analyzer = ga.GestureAnalyzer(window=5, min_agreement=3, grace_frames=2)
    run(analyzer, 5)
    for _ in range(3):  # gone longer than the grace period
        analyzer.analyze([])
    result = analyzer.analyze([right_hand()])[0]
    assert result.window == 1 and result.status == ga.UNCERTAIN


def test_brief_dropout_does_not_log_again():
    # Seen in a real run: a hand vanishing for a frame or two re-logged the same fist.
    analyzer = ga.GestureAnalyzer(window=5, min_agreement=3, grace_frames=15)
    logged = sum(r[0].newly_stable for r in run(analyzer, 5))
    for _ in range(3):
        analyzer.analyze([])
    logged += sum(r[0].newly_stable for r in run(analyzer, 5))
    assert logged == 1


def test_same_label_twice_kept_apart():
    # MediaPipe occasionally labels both hands "Right".
    a = features(gesture_hand(ga.FIST), "Right")
    b = features(gesture_hand(ga.OPEN_PALM), "Right")
    analyzer = ga.GestureAnalyzer(window=3, min_agreement=2)
    results = run(analyzer, 3, [a, b])[-1]
    assert [r.gesture for r in results] == [ga.FIST, ga.OPEN_PALM]


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        ga.GestureAnalyzer(window=5, min_agreement=6)
