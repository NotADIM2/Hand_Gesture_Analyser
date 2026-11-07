import numpy as np
import pytest

import motion_analyzer as ma
from hand_detector import DetectedHand
from synthetic_hand import make_hand

WIDTH = 640
FPS = 30
PALM = 80  # synthetic palm size in pixels


def hand_at(x, y, handedness="Right", pts=None):
    """A fist-shaped hand with its wrist at pixel (x, y)."""
    pts = make_hand() if pts is None else pts
    image_points = pts * PALM + [x, y, 0]
    return DetectedHand(handedness, 0.98, image_points / [WIDTH, 480, WIDTH], pts * 0.09, image_points)


def play(analyzer, frames, start=0.0):
    """Feed a list of per-frame hand lists at 30 FPS; returns every event in order."""
    events = []
    for i, hands in enumerate(frames):
        events += analyzer.update(hands, WIDTH, start + i / FPS)
    return events


def path(points, handedness="Right"):
    return [[hand_at(x, y, handedness)] for x, y in points]


def still(x, y, frames=10):
    return [(x, y)] * frames


def line(start, end, seconds):
    n = max(2, int(seconds * FPS))
    return [tuple(p) for p in np.linspace(start, end, n)]


def names(events):
    return [e.gesture for e in events]


@pytest.mark.parametrize(
    "start, end, gesture",
    [
        ((150, 300), (490, 300), ma.SWIPE_RIGHT),
        ((490, 300), (150, 300), ma.SWIPE_LEFT),
        ((320, 420), (320, 190), ma.SWIPE_UP),
        ((320, 190), (320, 420), ma.SWIPE_DOWN),
    ],
)
def test_fast_straight_movement_is_a_swipe(start, end, gesture):
    events = play(ma.MotionAnalyzer(), path(still(*start) + line(start, end, 0.4) + still(*end)))
    assert names(events) == [gesture]
    assert events[0].hand == "Right"


def test_slow_movement_is_not_a_swipe():
    assert play(ma.MotionAnalyzer(), path(line((150, 300), (490, 300), 3.0))) == []


def test_diagonal_movement_is_not_a_swipe():
    assert play(ma.MotionAnalyzer(), path(line((150, 420), (420, 150), 0.4))) == []


def test_jitter_is_ignored():
    rng = np.random.default_rng(0)
    points = [(320 + dx, 300 + dy) for dx, dy in rng.uniform(-4, 4, size=(90, 2))]
    assert play(ma.MotionAnalyzer(), path(points)) == []


def test_tracking_glitch_is_not_a_swipe():
    # MediaPipe swapping Left/Right makes a hand appear to jump across the frame in one frame.
    assert play(ma.MotionAnalyzer(), path(still(100, 300) + still(540, 300))) == []


def test_wave():
    t = np.arange(int(1.5 * FPS)) / FPS
    points = [(320 + 50 * np.sin(2 * np.pi * 2 * s), 300) for s in t]   # 2 waves a second
    assert names(play(ma.MotionAnalyzer(), path(points))) == [ma.WAVE]


def test_circle():
    angles = np.linspace(0, 2 * np.pi, int(1.2 * FPS))
    points = [(320 + 80 * np.cos(a), 280 + 80 * np.sin(a)) for a in angles]
    assert names(play(ma.MotionAnalyzer(), path(points))) == [ma.CIRCLE]


def test_one_movement_is_reported_once():
    # Keeps going in the same direction for a while: still a single swipe.
    events = play(ma.MotionAnalyzer(), path(line((60, 300), (600, 300), 0.7)))
    assert names(events) == [ma.SWIPE_RIGHT]


def test_each_hand_tracked_separately():
    left = line((490, 300), (150, 300), 0.4)
    right = still(560, 100, len(left))
    frames = [[hand_at(*a, "Left"), hand_at(*b, "Right")] for a, b in zip(left, right)]
    events = play(ma.MotionAnalyzer(), frames)
    assert [(e.gesture, e.hand) for e in events] == [(ma.SWIPE_LEFT, "Left")]


def test_trails_for_drawing():
    analyzer = ma.MotionAnalyzer()
    play(analyzer, path(line((100, 300), (140, 300), 0.3)))
    (trail,) = analyzer.trails(WIDTH)
    assert trail.shape[1] == 2 and trail[-1][0] > trail[0][0]


# --- Two hands ---------------------------------------------------------------

def two_hands(left_x, right_x, y=300):
    return [hand_at(left_x, y, "Left"), hand_at(right_x, y, "Right")]


def test_clap():
    gaps = np.linspace(260, 20, int(0.3 * FPS))
    frames = [two_hands(320 - g / 2, 320 + g / 2) for g in gaps] + [two_hands(310, 330)] * 10
    events = play(ma.MotionAnalyzer(), frames)
    assert [(e.gesture, e.hand) for e in events] == [(ma.CLAP, "Both")]


def test_hands_resting_together_is_not_a_clap():
    assert play(ma.MotionAnalyzer(), [two_hands(310, 330)] * 30) == []


def heart_pose(wrist_gap=120):
    a, b = make_hand() * PALM, make_hand() * PALM
    a[:, :2] += [320 - wrist_gap / 2, 300]
    b[:, :2] += [320 + wrist_gap / 2, 300]
    a[ma.INDEX_TIP, :2], b[ma.INDEX_TIP, :2] = (318, 150), (322, 150)   # index tips meet at the top
    a[ma.THUMB_TIP, :2], b[ma.THUMB_TIP, :2] = (318, 250), (322, 250)   # thumb tips meet below
    return [
        DetectedHand("Left", 0.98, a / [WIDTH, 480, WIDTH], a * 0.001, a),
        DetectedHand("Right", 0.98, b / [WIDTH, 480, WIDTH], b * 0.001, b),
    ]


def test_heart_reported_once_while_held():
    events = play(ma.MotionAnalyzer(), [heart_pose()] * 30)
    assert [(e.gesture, e.hand) for e in events] == [(ma.HEART, "Both")]


def test_heart_reported_again_after_letting_go():
    frames = [heart_pose()] * 10 + [two_hands(150, 490)] * 5 + [heart_pose()] * 10
    assert names(play(ma.MotionAnalyzer(), frames)) == [ma.HEART, ma.HEART]


def test_heart_needs_wrists_apart():
    assert not ma.is_heart(*heart_pose(wrist_gap=20))
    assert ma.is_heart(*heart_pose())


def test_count_swings():
    assert ma.count_swings(np.array([0, 1, 0, 1, 0]), 0.5) == 4
    assert ma.count_swings(np.array([0, 0.1, 0, 0.1]), 0.5) == 0
