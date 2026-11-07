"""Draw shapes in the air with a pinch (mode 3: DRAW SHAPES).

Touch your thumb and index tips together to put the pen down, move to draw, and open
them to lift the pen. When the pen lifts, the stroke is handed to the shape
recogniser (shape_recognizer.py): a recognised circle, triangle, square, rectangle
or line replaces the shaky stroke with a clean outline; anything else stays as
drawn. A fist clears the drawing. The pen sits between the two fingertips, smoothed
a little so the line isn't shaky.

A pinch often loosens for a moment at a sharp corner. So a stroke is only recognised
once the pen has stayed up for DRAW_JOIN_FRAMES frames; if it goes down again near
where it lifted before that, the same stroke continues. A one-frame jump of the pen
across the picture is a tracking glitch and is skipped.

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
from shape_recognizer import Shape, analyse, draw_shape

THUMB_TIP, INDEX_TIP = 4, 8


@dataclass(frozen=True)
class DrawnShape:
    hand: str      # "Left" / "Right"
    shape: Shape


@dataclass
class _Lifted:
    points: list            # the stroke so far
    frame: int              # frame number when the pen lifted


class AirCanvas:
    def __init__(self, shapes: bool = True):
        self.shapes = shapes   # recognise shapes when the pen lifts
        self.canvas: np.ndarray | None = None
        self.last_stroke: str | None = None        # what the last stroke was, or why it isn't a shape
        self._pen: dict[str, np.ndarray] = {}      # hand key -> smoothed pen position while drawing
        self._stroke: dict[str, list] = {}         # hand key -> pen positions of the stroke being drawn
        self._lifted: dict[str, _Lifted] = {}      # hand key -> stroke waiting to be recognised
        self._before: np.ndarray | None = None     # the drawing before the current stroke
        self._jumps: dict[str, int] = {}           # hand key -> frames in a row the pen jumped
        self._frame = 0

    def update(self, frame_shape: tuple, hands: list[DetectedHand],
               features: list[HandFeatures], results: list[GestureResult]) -> list[DrawnShape]:
        """Call once per frame; returns the shapes recognised this frame."""
        if self.canvas is None or self.canvas.shape != frame_shape:
            self.canvas = np.zeros(frame_shape, np.uint8)
        self._frame += 1
        max_jump = config.DRAW_MAX_JUMP * frame_shape[1]
        keys = track_keys(hands)
        for key, hand, f, result in zip(keys, hands, features, results):
            if result.gesture == FIST and result.newly_stable:
                self.clear()
                return []
            # Hysteresis: the pen goes down below one gap and only lifts above a
            # larger one, so a slightly shaky pinch doesn't break the line.
            gap_limit = config.DRAW_LIFT_RATIO if key in self._pen else config.DRAW_TOUCH_RATIO
            if f.thumb_index_distance >= gap_limit:
                self._lift(key)
                continue
            tip = hand.image_points[[THUMB_TIP, INDEX_TIP], :2].mean(axis=0)
            if key not in self._pen:
                self._pen_down(key, tip)
            if np.linalg.norm(tip - self._pen[key]) > max_jump:
                self._jumps[key] = self._jumps.get(key, 0) + 1
                if self._jumps[key] <= config.DRAW_GLITCH_FRAMES:
                    continue   # tracking glitch, not a movement
                self._lift(key)   # it stayed there: the hand really moved, start a new stroke
                self._pen_down(key, tip)
            self._jumps.pop(key, None)
            previous = self._pen[key]
            pen = config.DRAW_SMOOTHING * previous + (1 - config.DRAW_SMOOTHING) * tip
            cv2.line(self.canvas, _px(previous), _px(pen), config.DRAW_COLOUR, config.DRAW_THICKNESS, cv2.LINE_AA)
            self._pen[key] = pen
            self._stroke[key].append(pen)
        for key in set(self._pen) - set(keys):   # the hand left the picture: the pen lifts
            self._lift(key)
        return self._finish_waiting()

    def _pen_down(self, key: str, tip: np.ndarray) -> None:
        waiting = self._lifted.pop(key, None)
        if waiting is not None and np.linalg.norm(tip - waiting.points[-1]) <= config.DRAW_JOIN_DISTANCE:
            # The pinch only loosened for a moment: carry on with the same stroke.
            self._pen[key] = waiting.points[-1]
            self._stroke[key] = waiting.points
            return
        if waiting is not None:
            self._lifted[key] = waiting   # a new stroke elsewhere; the old one still finishes
        if not self._pen and not self._lifted:
            self._before = self.canvas.copy()
        self._pen[key] = tip
        self._stroke[key] = [tip]

    def _lift(self, key: str) -> None:
        if self._pen.pop(key, None) is not None:
            self._lifted[key] = _Lifted(self._stroke.pop(key), self._frame)

    def _finish_waiting(self) -> list[DrawnShape]:
        finished = []
        for key, waiting in list(self._lifted.items()):
            if self._frame - waiting.frame < config.DRAW_JOIN_FRAMES:
                continue
            del self._lifted[key]
            if not self.shapes:
                continue
            result = analyse(np.array(waiting.points))
            self.last_stroke = result.reason
            if result.shape is None:
                continue                    # not a shape: the stroke stays as drawn
            if not self._pen and not self._lifted and self._before is not None:
                self.canvas[:] = self._before   # rub out the shaky stroke (unless another is in progress)
            draw_shape(self.canvas, result.shape)
            finished.append(DrawnShape(key.split("#")[0], result.shape))
        return finished

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
        self._lifted.clear()
        self._jumps.clear()
        self._before = None
        self.last_stroke = None

    @property
    def pen_down(self) -> bool:
        return bool(self._pen)


def _px(point: np.ndarray) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))
