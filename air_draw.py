"""Draw in the air with a pinch (extra mode: AIR DRAW).

Touch your thumb and index tips together to put the pen down, move to draw, and open
them to lift the pen. A fist clears the drawing. The pen sits between the two
fingertips, smoothed a little so the line isn't shaky.

Like the rest of the app, it works on landmarks and features, not on pixels, so it
can be tested without a webcam.
"""
import cv2
import numpy as np

import config
from feature_extractor import HandFeatures
from gesture_analyzer import FIST, GestureResult, track_keys
from hand_detector import DetectedHand

THUMB_TIP, INDEX_TIP = 4, 8


class AirCanvas:
    def __init__(self):
        self.canvas: np.ndarray | None = None
        self._pen: dict[str, np.ndarray] = {}   # hand key -> smoothed pen position while drawing

    def update(self, frame_shape: tuple, hands: list[DetectedHand],
               features: list[HandFeatures], results: list[GestureResult]) -> None:
        if self.canvas is None or self.canvas.shape != frame_shape:
            self.canvas = np.zeros(frame_shape, np.uint8)
        keys = track_keys(hands)
        for key, hand, f, result in zip(keys, hands, features, results):
            if result.gesture == FIST and result.newly_stable:
                self.clear()
                continue
            # Hysteresis: the pen goes down below one gap and only lifts above a
            # larger one, so a slightly shaky pinch doesn't break the line.
            gap_limit = config.DRAW_LIFT_RATIO if key in self._pen else config.DRAW_TOUCH_RATIO
            if f.thumb_index_distance >= gap_limit:
                self._pen.pop(key, None)
                continue
            tip = hand.image_points[[THUMB_TIP, INDEX_TIP], :2].mean(axis=0)
            previous = self._pen.get(key)
            if previous is None:
                self._pen[key] = tip
                continue
            pen = config.DRAW_SMOOTHING * previous + (1 - config.DRAW_SMOOTHING) * tip
            cv2.line(self.canvas, _px(previous), _px(pen), config.DRAW_COLOUR, config.DRAW_THICKNESS, cv2.LINE_AA)
            self._pen[key] = pen
        for key in set(self._pen) - set(keys):
            del self._pen[key]

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """Paint the drawing onto the frame."""
        if self.canvas is not None and self.canvas.shape == frame.shape:
            drawn = self.canvas.any(axis=2)
            frame[drawn] = self.canvas[drawn]
        return frame

    def clear(self) -> None:
        if self.canvas is not None:
            self.canvas[:] = 0
        self._pen.clear()

    @property
    def pen_down(self) -> bool:
        return bool(self._pen)


def _px(point: np.ndarray) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))
