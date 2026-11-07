from types import SimpleNamespace

import numpy as np

import gesture_analyzer as ga
import ui
from feature_extractor import extract_features
from hand_detector import DetectedHand
from synthetic_hand import gesture_hand


def fake_hand():
    pts = gesture_hand(ga.PEACE)
    image = pts[:, :2] * 150 + [320, 380]   # place it in the middle of a 640x480 frame
    image_points = np.column_stack([image, pts[:, 2] * 150])
    return DetectedHand("Right", 0.98, image_points / [640, 480, 640], pts * 0.09, image_points)


def result(status=ga.STABLE):
    return ga.GestureResult("Right", ga.PEACE, ga.PEACE, status, 9, 10, False)


def test_result_lines_match_spec():
    text = [t for t, _ in ui.result_lines(1, [result()])]
    assert "Detected Gesture: PEACE" in text
    assert "Hand: RIGHT" in text
    assert "Status: DETECTED" in text
    assert "Recognition: STABLE (9/10 frames agree)" in text
    assert not any("%" in t for t in text)  # no made-up confidence percentage


def test_no_hand_lines():
    text = [t for t, _ in ui.result_lines(0, [])]
    assert text == ["Hands detected: 0", "Status: NO HAND DETECTED"]


def test_history_lines():
    rec = SimpleNamespace(timestamp="2026-11-07T14:05:09", hand="Left", gesture="FIST")
    view = ui.HistoryView({"FIST": 3}, [rec], 12)
    text = [t for t, _ in ui.history_lines(view)]
    assert "FIST: 3" in text
    assert any(t.startswith("14:05:09 LEFT") for t in text)
    assert text[-1] == "All sessions: 12 records"


def test_draw_everything_on_a_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    hand = fake_hand()
    features = [extract_features(hand.world_landmarks, hand.image_points, "Right")]
    interface = ui.Interface.__new__(ui.Interface)  # skip creating a real window
    interface.show_history = interface.show_features = True
    interface.render(frame, [hand], [result()], features, 30.0, ui.HistoryView({"PEACE": 1}, [], 1))
    assert frame.any()


def test_panel_stays_inside_frame():
    frame = np.zeros((100, 100, 3), np.uint8)
    width, height = ui.draw_panel(frame, [("a long line of text " * 3, ui.WHITE)], (90, 90))
    assert width > 100  # wider than the frame: drawn clipped, not crashed


def test_fps_counter():
    counter = ui.FPSCounter()
    counter.tick()
    assert counter.tick() > 0
