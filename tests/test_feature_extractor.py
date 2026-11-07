import numpy as np
import pytest

from feature_extractor import extract_features, joint_angle
from synthetic_hand import ALL_FINGERS, make_hand, rotate_3d, rotate_in_image


def features(pts, handedness="Right"):
    return extract_features(pts, pts, handedness)


def test_joint_angle():
    a, b = np.array([1.0, 0, 0]), np.zeros(3)
    assert joint_angle(a, b, np.array([0, 1.0, 0])) == pytest.approx(90)
    assert joint_angle(a, b, np.array([-1.0, 0, 0])) == pytest.approx(180)
    assert joint_angle(b, b, a) == 0.0  # degenerate input doesn't crash


def test_open_hand_all_extended():
    f = features(make_hand(ALL_FINGERS, thumb="out"))
    assert f.extended_count == 5


def test_fist_nothing_extended():
    f = features(make_hand())
    assert f.extended_count == 0
    assert f.thumb_lateral < 0  # thumb folded across the palm


@pytest.mark.parametrize("fingers", [("index",), ("index", "middle"), ("index", "middle", "ring"), ("pinky",)])
def test_selected_fingers_extended(fingers):
    f = features(make_hand(fingers))
    assert f.extended_fingers == frozenset(fingers)


def test_scale_invariant():
    # World landmarks are in metres (palm ~0.09 m); results must not depend on hand size.
    pts = make_hand(("index", "middle"), spread=30)
    small, big = features(pts * 0.09), features(pts * 3)
    assert small.extended == big.extended
    assert small.index_middle_spread == pytest.approx(big.index_middle_spread)
    assert small.thumb_index_distance == pytest.approx(big.thumb_index_distance)


@pytest.mark.parametrize("degrees", [-60, -20, 25, 70, 180])
def test_finger_state_survives_rotation(degrees):
    pts = rotate_in_image(make_hand(("index", "middle")), degrees)
    assert features(pts).extended_fingers == {"index", "middle"}


@pytest.mark.parametrize("tilt", [(30, 0), (0, 40), (-25, 25)])
def test_finger_state_survives_tilt(tilt):
    pts = rotate_3d(make_hand(("index",)), *tilt)
    assert features(pts).extended_fingers == {"index"}


def test_peace_spread_measured():
    assert features(make_hand(("index", "middle"), spread=30)).index_middle_spread == pytest.approx(30, abs=1)
    assert features(make_hand(("index", "middle"), spread=0)).index_middle_spread == pytest.approx(0, abs=1)


def test_thumb_direction():
    side = make_hand(thumb="side")
    assert features(rotate_in_image(side, 90)).thumb_direction == "up"
    assert features(rotate_in_image(side, -90)).thumb_direction == "down"
    assert features(side).thumb_direction is None  # pointing sideways


def test_hand_direction():
    upright = make_hand(ALL_FINGERS)
    assert features(upright).hand_direction == "up"
    assert features(rotate_in_image(upright, 180)).hand_direction == "down"
    assert features(rotate_in_image(upright, 90)).hand_direction == "right"


def test_index_direction():
    pointing = make_hand(("index",))
    assert features(pointing).index_direction == "up"
    assert features(rotate_in_image(pointing, -90)).index_direction == "left"
    assert features(rotate_in_image(pointing, 90)).index_direction == "right"
    assert features(rotate_in_image(pointing, 180)).index_direction == "down"


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        extract_features(np.zeros((20, 3)), np.zeros((21, 3)))


def test_real_samples_finger_states(real_samples):
    # Real MediaPipe landmarks from photos (see tests/fixtures/sample_hands.json).
    expected = {
        "THUMBS UP": {"thumb"},
        "THUMBS DOWN": {"thumb"},
        "PEACE": {"index", "middle"},
        "ONE FINGER": {"index"},
    }
    checked = 0
    for s in real_samples:
        if s["gesture"] in expected:
            f = extract_features(s["world_landmarks"], s["image_points"], s["handedness"])
            assert f.extended_fingers == expected[s["gesture"]], s["source"]
            checked += 1
    assert checked >= 4
