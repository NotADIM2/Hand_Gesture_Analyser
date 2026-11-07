"""Draw shapes in the air with a pinch (mode 3: DRAW SHAPES).

Touch your thumb and index tips together to put the pen down, move to draw, and open
them to lift the pen. When the pen lifts, the stroke goes to the shape recogniser
(shape_recognizer.py): a circle, triangle, square, rectangle or line is replaced by a
clean outline; anything else stays as drawn.

Shapes can be drawn in one stroke or several: four separate sides that meet make a
square, three make a triangle, two arcs a circle. Strokes finished within
DRAW_GROUP_SECONDS of each other whose ends meet are tried together.

A pinch often loosens for a moment at a sharp corner, so a stroke only counts as
finished once the pen has stayed up for DRAW_JOIN_SECONDS; pinching again near where
it stopped carries on the same stroke. A one-frame jump of the pen across the picture
is a tracking glitch and is skipped. Wave (with the pen up) clears the drawing; z
undoes the last stroke or shape. The pen colour and size can be changed (the paint
bar of virtual buttons, buttons.py); each stroke keeps the colour it was drawn in. With
the default cyan pen a recognised shape turns green; other colours keep their colour.

The drawing is kept as a list of strokes and shapes, so nothing is ever rubbed out by
accident and undo is simple. Finished items are drawn once into a cached layer when
the drawing changes; each frame only copies that layer's drawn pixels and adds the
stroke in progress. Like the rest of the app, it works on landmarks, not on
pixels, so it can be tested without a webcam.
"""
from dataclasses import dataclass

import cv2
import numpy as np

import config
from feature_extractor import HandFeatures
from gesture_analyzer import GestureResult, track_keys
from hand_detector import DetectedHand
from shape_recognizer import LINE, Shape, analyse, draw_shape

THUMB_TIP, INDEX_TIP = 4, 8


@dataclass(frozen=True)
class DrawnShape:
    hand: str      # "Left" / "Right"
    shape: Shape


@dataclass
class Item:
    """One thing on the drawing: a stroke as drawn, or the clean shape it became."""
    points: np.ndarray            # the pen's path, (n, 2) pixels
    shape: Shape | None           # drawn instead of the points when recognised
    hand: str
    finished_at: float
    colour: tuple = config.DRAW_COLOUR
    thickness: int = config.DRAW_THICKNESS

    @property
    def open(self) -> bool:
        """Could be one side of a shape drawn in several strokes."""
        return self.shape is None or self.shape.name == LINE


@dataclass
class _Lifted:
    points: list
    at: float      # when the pen lifted


class AirCanvas:
    def __init__(self, shapes: bool = True):
        self.shapes = shapes                      # recognise shapes when the pen lifts
        self.colour = config.DRAW_COLOUR          # pen colour and thickness for new strokes
        self.thickness = config.DRAW_THICKNESS
        self.items: list[Item] = []
        self.last_stroke: str | None = None       # what the last stroke was, or why it isn't a shape
        self.pens: dict[str, tuple[np.ndarray, bool]] = {}   # hand key -> (pen position, down?) this frame
        self._pen: dict[str, np.ndarray] = {}     # hand key -> smoothed pen position while drawing
        self._stroke: dict[str, list] = {}        # hand key -> pen positions of the stroke being drawn
        self._lifted: dict[str, _Lifted] = {}     # hand key -> stroke waiting to be finished
        self._jumps: dict[str, int] = {}          # hand key -> frames in a row the pen jumped
        self._version = 0                         # bumped whenever self.items changes
        self._layer = None                        # (version, frame shape, pixel indices, colours)

    def update(self, frame_shape: tuple, hands: list[DetectedHand], features: list[HandFeatures],
               results: list[GestureResult], now: float) -> list[DrawnShape]:
        """Call once per frame; returns the shapes recognised this frame."""
        max_jump = config.DRAW_MAX_JUMP * frame_shape[1]
        keys = track_keys(hands)
        self.pens = {}
        for key, hand, f in zip(keys, hands, features):
            tip = hand.image_points[[THUMB_TIP, INDEX_TIP], :2].mean(axis=0)
            # Hysteresis: the pen goes down below one gap and only lifts above a
            # larger one, so a slightly shaky pinch doesn't break the line.
            gap_limit = config.DRAW_LIFT_RATIO if key in self._pen else config.DRAW_TOUCH_RATIO
            if f.thumb_index_distance >= gap_limit:
                self._lift(key, now)
                self.pens[key] = (tip, False)
                continue
            if key not in self._pen:
                self._pen_down(key, tip)
            if np.linalg.norm(tip - self._pen[key]) > max_jump:
                self._jumps[key] = self._jumps.get(key, 0) + 1
                if self._jumps[key] <= config.DRAW_GLITCH_FRAMES:
                    self.pens[key] = (self._pen[key], True)
                    continue   # tracking glitch, not a movement
                self._lift(key, now)   # it stayed there: the hand really moved, start a new stroke
                self._pen_down(key, tip)
            self._jumps.pop(key, None)
            pen = config.DRAW_SMOOTHING * self._pen[key] + (1 - config.DRAW_SMOOTHING) * tip
            self._pen[key] = pen
            self._stroke[key].append(pen)
            self.pens[key] = (pen, True)
        for key in set(self._pen) - set(keys):   # the hand left the picture: the pen lifts
            self._lift(key, now)
        return self._finish_waiting(now)

    def _pen_down(self, key: str, tip: np.ndarray) -> None:
        waiting = self._lifted.pop(key, None)
        if waiting is not None and np.linalg.norm(tip - waiting.points[-1]) <= config.DRAW_JOIN_DISTANCE:
            # The pinch only loosened for a moment: carry on with the same stroke.
            self._pen[key] = waiting.points[-1]
            self._stroke[key] = waiting.points
            return
        if waiting is not None:
            self._lifted[key] = waiting   # a new stroke elsewhere; the old one still finishes
        self._pen[key] = tip
        self._stroke[key] = [tip]

    def _lift(self, key: str, now: float) -> None:
        if self._pen.pop(key, None) is not None:
            self._lifted[key] = _Lifted(self._stroke.pop(key), now)

    def _finish_waiting(self, now: float) -> list[DrawnShape]:
        finished = []
        for key, waiting in list(self._lifted.items()):
            if now - waiting.at < config.DRAW_JOIN_SECONDS:
                continue
            del self._lifted[key]
            drawn = self._finish(np.array(waiting.points), key.split("#")[0], now)
            if drawn:
                finished.append(drawn)
        return finished

    def _finish(self, points: np.ndarray, hand: str, now: float) -> DrawnShape | None:
        if len(points) < 2:
            return None
        result = analyse(points) if self.shapes else None
        shape = result.shape if result else None
        self.items.append(Item(points, shape, hand, now, self.colour, self.thickness))
        self._version += 1
        if result is None:
            return None
        self.last_stroke = result.reason
        if shape is None or shape.name == LINE:
            # Maybe the last side of a shape drawn in several strokes.
            combined = self._combine_recent()
            if combined:
                shape = combined
        return DrawnShape(hand, shape) if shape else None

    def _combine_recent(self) -> Shape | None:
        """Try the latest strokes together (2 up to DRAW_GROUP_MAX): if their ends meet in a
        closed loop that is a shape, they become that one shape."""
        recent = []
        for item in reversed(self.items):
            if not item.open or (recent and recent[-1].finished_at - item.finished_at > config.DRAW_GROUP_SECONDS):
                break
            recent.append(item)
            if len(recent) == config.DRAW_GROUP_MAX:
                break
        recent.reverse()   # in drawing order
        for k in range(2, len(recent) + 1):
            group = recent[-k:]
            path = chain([item.points for item in group])
            # Several strokes make a shape only once the last one meets the first again.
            # The gap must be small next to the shortest stroke: three sides of a long
            # rectangle leave a gap as long as a whole side, and aren't a shape yet.
            shortest = min(_length(item.points) for item in group)
            if path is None or np.linalg.norm(path[-1] - path[0]) > max(
                    config.DRAW_MEET_MIN, config.DRAW_CLOSE_RATIO * shortest):
                continue
            result = analyse(path)
            if result.shape is not None and result.shape.name != LINE:
                del self.items[-k:]
                last = group[-1]
                self.items.append(Item(path, result.shape, last.hand, last.finished_at, last.colour, last.thickness))
                self._version += 1
                self.last_stroke = f"{result.reason} ({k} strokes)"
                return result.shape
        return None

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw the drawing, the strokes in progress and the pens onto the frame."""
        if self.items:
            _, _, index, colours = self._finished_layer(frame.shape)
            frame.reshape(-1, 3)[index] = colours
        _ink(frame, list(self._stroke.values()) + [w.points for w in self._lifted.values()],
             self.colour, self.thickness)
        for pen, down in self.pens.values():
            centre = tuple(int(v) for v in pen)
            if down:
                cv2.circle(frame, centre, max(self.thickness, 4), self.colour, -1, cv2.LINE_AA)
            else:   # a ring where the pen would touch down
                cv2.circle(frame, centre, max(self.thickness, 4) + 4, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _finished_layer(self, shape: tuple):
        """The finished strokes and shapes as (pixel indices, colours), redrawn only when they change."""
        if self._layer is None or self._layer[:2] != (self._version, shape):
            layer = np.zeros(shape, np.uint8)
            for item in self.items:   # in drawing order, so later strokes go on top
                if item.shape is None:
                    _ink(layer, [item.points], item.colour, item.thickness)
                else:
                    colour = config.SHAPE_COLOUR if item.colour == config.DRAW_COLOUR else item.colour
                    draw_shape(layer, item.shape, colour, item.thickness)
            flat = layer.reshape(-1, 3)
            index = np.flatnonzero(flat.any(axis=1))
            self._layer = (self._version, shape, index, flat[index])
        return self._layer

    def next_size(self) -> int:
        """Change the pen to the next size in PAINT_SIZES; returns the new thickness."""
        sizes = config.PAINT_SIZES
        self.thickness = sizes[(sizes.index(self.thickness) + 1) % len(sizes)] if self.thickness in sizes else sizes[0]
        return self.thickness

    def undo(self) -> bool:
        """Remove the last stroke or shape; False if there was nothing to undo."""
        if not self.items:
            return False
        self.items.pop()
        self._version += 1
        self.last_stroke = None
        return True

    def clear(self) -> None:
        self.items.clear()
        self._version += 1
        self._pen.clear()
        self._stroke.clear()
        self._lifted.clear()
        self._jumps.clear()
        self.last_stroke = None

    @property
    def pen_down(self) -> bool:
        return bool(self._pen)

    @property
    def empty(self) -> bool:
        return not self.items and not self._stroke and not self._lifted


def _ink(image: np.ndarray, strokes, colour=config.DRAW_COLOUR, thickness: int = config.DRAW_THICKNESS) -> None:
    """Draw strokes as they were drawn."""
    lines = [np.asarray(s).astype(np.int32).reshape(-1, 1, 2) for s in strokes if len(s) > 1]
    if lines:
        cv2.polylines(image, lines, False, colour, thickness, cv2.LINE_AA)


def chain(strokes: list[np.ndarray]) -> np.ndarray | None:
    """Join strokes end to end into one path, each one turned round if needed so that its
    nearer end meets the path so far. None if two of them don't meet."""
    path = strokes[0]
    second = strokes[1]
    # The first stroke may have been drawn either way round too: pick the closest pairing.
    options = [(a, b) for a in (path, path[::-1]) for b in (second, second[::-1])]
    path, second = min(options, key=lambda ab: np.linalg.norm(ab[0][-1] - ab[1][0]))
    previous = path
    for stroke in [second] + list(strokes[2:]):
        if np.linalg.norm(stroke[-1] - path[-1]) < np.linalg.norm(stroke[0] - path[-1]):
            stroke = stroke[::-1]
        if np.linalg.norm(stroke[0] - path[-1]) > _meet_distance(previous, stroke):
            return None
        path, previous = np.vstack([path, stroke]), stroke
    return path


def _meet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """How close two stroke ends must be to count as meeting: a share of the longer stroke
    (hand-drawn sides rarely meet exactly, and a short side often meets a long one)."""
    return max(config.DRAW_MEET_MIN, config.DRAW_MEET_RATIO * max(_length(a), _length(b)))


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
