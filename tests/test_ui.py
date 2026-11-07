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


def test_draw_moving_gesture_and_trail():
    from motion_analyzer import MotionEvent

    frame = np.zeros((480, 640, 3), np.uint8)
    interface = ui.Interface.__new__(ui.Interface)
    interface.show_history = interface.show_features = False
    moves = [MotionEvent("SWIPE LEFT", "Right", 0.0), MotionEvent("HEART", "Both", 0.1)]
    trail = np.array([[100.0, 200.0], [150.0, 210.0], [200.0, 220.0]])
    interface.render(frame, [], [], [], 30.0, moves=moves, trails=[trail], last_move=moves[-1])
    pink = (np.abs(frame.astype(int) - ui.TRAIL).sum(axis=2) < 30).sum()
    assert pink > 200  # the trail is clearly visible
    assert "Last move: HEART (BOTH HANDS)" in [t for t, _ in ui.result_lines(0, [], moves[-1])]
    assert ui.move_text(moves[-1]) == "HEART (BOTH HANDS)"
    assert ui.move_text(moves[0]) == "SWIPE LEFT (RIGHT HAND)"
    assert frame.any()


def test_heart_readout_in_feature_panel():
    from motion_analyzer import HeartMeasurements

    lines = ui.feature_lines([], HeartMeasurements(0.9, 0.2, 1.4, True))
    text, colour = lines[-1]
    assert "index tips 0.90 NO" in text and "thumb tips 0.20 ok" in text
    assert colour == ui.ORANGE


def test_control_mode_label_and_action_flash():
    frame = np.zeros((480, 640, 3), np.uint8)
    interface = ui.Interface.__new__(ui.Interface)
    interface.show_history = True
    interface.show_features = False
    interface.render(frame, [], [], [], 30.0, ui.HistoryView(), control="presentation", action="NEXT SLIDE")
    assert frame.any()
