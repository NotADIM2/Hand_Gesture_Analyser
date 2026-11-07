"""Recognise shapes drawn in the air (used by the draw mode).

When the pen lifts, the stroke (the pen's path in pixels) is classified:

    LINE                 an open stroke that is nearly straight
    CIRCLE               a closed stroke that is round
    TRIANGLE             a closed stroke with three corners
    SQUARE / RECTANGLE   a closed stroke with four corners (by side ratio)

A stroke is closed when it ends near where it started. Closed strokes are measured on
their convex hull (the outline a rubber band around the stroke would make), which
ignores wobble and the overshoot when people close a circle. The hull is compared
with the smallest rectangle and the smallest triangle that fit around it: a drawn
rectangle fills its box, a drawn triangle fills its triangle. A circle is checked
first, by roundness (circularity 4*pi*area / perimeter^2: 1 for a circle, about 0.85
for a hand-drawn square). A stroke much longer than its hull is a scribble.
"""
from dataclasses import dataclass

import cv2
import numpy as np

import config

LINE, CIRCLE, TRIANGLE, SQUARE, RECTANGLE = "LINE", "CIRCLE", "TRIANGLE", "SQUARE", "RECTANGLE"
SHAPES = (LINE, CIRCLE, TRIANGLE, SQUARE, RECTANGLE)


@dataclass(frozen=True)
class Shape:
    name: str
    points: np.ndarray   # the clean outline to draw: polygon corners, line ends, or circle as (cx, cy, r)


def recognize(stroke) -> Shape | None:
    pts = np.asarray(stroke, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 6:
        return None
    # Hand shake makes a stroke look longer than it is, so lengths are measured on a
    # lightly smoothed copy; corners and roundness use the stroke as drawn.
    path = float(np.linalg.norm(np.diff(_smooth(pts), axis=0), axis=1).sum())
    if path < config.SHAPE_MIN_SIZE:
        return None  # too small to be deliberate
    gap = float(np.linalg.norm(pts[-1] - pts[0]))

    if gap > config.SHAPE_CLOSED_RATIO * path:
        # Open stroke: only a straight line counts. Straightness is how far the stroke
        # strays from the line between its ends, so hand shake doesn't matter much.
        direction = (pts[-1] - pts[0]) / gap
        offsets = pts - pts[0]
        bend = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0]).max()
        if bend <= config.SHAPE_LINE_MAX_BEND * gap:
            return Shape(LINE, np.array([pts[0], pts[-1]]))
        return None

    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    perimeter = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)
    if not perimeter or not area or path > config.SHAPE_MAX_PATH_RATIO * perimeter:
        return None  # goes back and forth inside its outline: a scribble

    if 4 * np.pi * area / perimeter ** 2 >= config.SHAPE_CIRCLE_ROUNDNESS:
        return _circle(pts)
    # How well the outline fills the smallest rectangle / triangle around it. A drawn
    # rectangle fills its box almost completely, a circle only pi/4 (0.79) of it, a
    # triangle half; a triangle fills its own enclosing triangle almost completely.
    box = cv2.minAreaRect(hull)
    (_, _), (w, h), _ = box
    if area / (w * h) >= config.SHAPE_BOX_FILL:
        ratio = max(w, h) / min(w, h)
        name = SQUARE if ratio <= config.SHAPE_SQUARE_RATIO else RECTANGLE
        return Shape(name, cv2.boxPoints(box))
    triangle_area, triangle = cv2.minEnclosingTriangle(hull)
    if triangle_area and area / triangle_area >= config.SHAPE_TRIANGLE_FILL:
        return Shape(TRIANGLE, triangle.reshape(-1, 2))
    return None


def _smooth(pts: np.ndarray, window: int = config.SHAPE_SMOOTHING) -> np.ndarray:
    """Moving average along the stroke; the two ends stay where they were drawn."""
    if len(pts) <= window:
        return pts
    kernel = np.ones(window) / window
    padded = np.pad(pts, ((window // 2, window // 2), (0, 0)), mode="edge")
    return np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)]).astype(np.float32)


def _circle(pts: np.ndarray) -> Shape:
    (cx, cy), r = cv2.minEnclosingCircle(pts.reshape(-1, 1, 2))
    # The enclosing circle hugs the outermost wobble; the average distance is the drawn size.
    r = float(np.mean(np.linalg.norm(pts - [cx, cy], axis=1))) or r
    return Shape(CIRCLE, np.array([cx, cy, r]))


def draw_shape(canvas: np.ndarray, shape: Shape, colour=None, thickness: int | None = None) -> None:
    """Draw the clean version of a recognised shape."""
    colour = colour or config.SHAPE_COLOUR
    thickness = thickness or config.DRAW_THICKNESS
    if shape.name == CIRCLE:
        cx, cy, r = shape.points
        cv2.circle(canvas, (int(cx), int(cy)), int(r), colour, thickness, cv2.LINE_AA)
    elif shape.name == LINE:
        a, b = shape.points.astype(int)
        cv2.line(canvas, tuple(a), tuple(b), colour, thickness, cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [shape.points.astype(np.int32).reshape(-1, 1, 2)], True, colour, thickness, cv2.LINE_AA)
