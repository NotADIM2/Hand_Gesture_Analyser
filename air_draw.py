"""Draw shapes in the air with a pinch (mode 3: DRAW SHAPES).

Touch your thumb and index tips together to put the pen down, move to draw, and open
them to lift the pen. When the pen lifts, the stroke is handed to the shape
recogniser (shape_recognizer.py): a recognised circle, triangle, square, rectangle
or line replaces the shaky stroke with a clean outline; anything else stays as
drawn. A fist clears the drawing. The pen sits between the two fingertips, smoothed
a little so the line isn't shaky.

Like the rest of the app, it works on landmarks and features, not on pixels, so it
can be tested without a webcam.
"""
from dataclasses import dataclass

import cv2
import numpy as np

import config
from feature_extractor import HandFeatures
from gesture_analyzer import FIST, GestureResult, track_keys
from hand_detector import DetectedHand
from shape_recognizer import Shape, draw_shape, recognize

THUMB_TIP, INDEX_TIP = 4, 8


@dataclass(frozen=True)
class DrawnShape:
    hand: str      # "Left" / "Right"
    shape: Shape


class AirCanvas:
    def __init__(self, shapes: bool = True):
        self.shapes = shapes   # recognise shapes when the pen lifts
        self.canvas: np.ndarray | None = None
        self._pen: dict[str, np.ndarray] = {}      # hand key -> smoothed pen position while drawing
        self._stroke: dict[str, list] = {}         # hand key -> pen positions of the stroke being drawn
        self._before: np.ndarray | None = None     # the drawing before the current stroke

    def update(self, frame_shape: tuple, hands: list[DetectedHand],
               features: list[HandFeatures], results: list[GestureResult]) -> list[DrawnShape]:
        """Call once per frame; returns the shapes recognised this frame (when a pen lifts)."""
        if self.canvas is None or self.canvas.shape != frame_shape:
            self.canvas = np.zeros(frame_shape, np.uint8)
        keys = track_keys(hands)
        finished = []
        for key, hand, f, result in zip(keys, hands, features, results):
            if result.gesture == FIST and result.newly_stable:
                self.clear()
                return []
            # Hysteresis: the pen goes down below one gap and only lifts above a
            # larger one, so a slightly shaky pinch doesn't break the line.
            gap_limit = config.DRAW_LIFT_RATIO if key in self._pen else config.DRAW_TOUCH_RATIO
            if f.thumb_index_distance >= gap_limit:
                finished += self._lift(key)
                continue
            tip = hand.image_points[[THUMB_TIP, INDEX_TIP], :2].mean(axis=0)
            previous = self._pen.get(key)
            if previous is None:
                if not self._pen:
                    self._before = self.canvas.copy()
                self._pen[key] = tip
                self._stroke[key] = [tip]
                continue
            pen = config.DRAW_SMOOTHING * previous + (1 - config.DRAW_SMOOTHING) * tip
            cv2.line(self.canvas, _px(previous), _px(pen), config.DRAW_COLOUR, config.DRAW_THICKNESS, cv2.LINE_AA)
            self._pen[key] = pen
            self._stroke[key].append(pen)
        for key in set(self._pen) - set(keys):   # the hand left the picture: the pen lifts
            finished += self._lift(key)
        return finished

    def _lift(self, key: str) -> list[DrawnShape]:
        if self._pen.pop(key, None) is None:
            return []
        stroke = np.array(self._stroke.pop(key))
        shape = recognize(stroke) if self.shapes else None
        if shape is None:
            return []                       # not a shape: the stroke stays as drawn
        if not self._pen and self._before is not None:
            self.canvas[:] = self._before   # rub out the shaky stroke (unless another pen is drawing)
        draw_shape(self.canvas, shape)
        return [DrawnShape(key.split("#")[0], shape)]

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
        self._stroke.clear()
        self._before = None

    @property
    def pen_down(self) -> bool:
        return bool(self._pen)


def _px(point: np.ndarray) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))
