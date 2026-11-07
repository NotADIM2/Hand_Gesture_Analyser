"""Shape recognition in the draw mode, and drawing shapes with a pinch."""
import numpy as np
import pytest

import config
import gesture_analyzer as ga
import shape_recognizer as sr
from air_draw import AirCanvas
from feature_extractor import extract_features
from hand_detector import DetectedHand
from synthetic_hand import make_hand

RNG_SEED = 7


def outline(corners, points=60):
    """Points along a closed polygon, back to the start, like one stroke of a pen."""
    corners = np.vstack([corners, corners[:1]]).astype(float)
    per_side = points // (len(corners) - 1)
    out = [a + (b - a) * t for a, b in zip(corners[:-1], corners[1:])
           for t in np.linspace(0, 1, per_side, endpoint=False)]
    return np.array(out + [corners[0]])


def circle(r=100.0, overshoot=0.1, squash=1.0):
    a = np.linspace(0, 2 * np.pi * (1 + overshoot), 60)
    return np.c_[320 + r * np.cos(a), 240 + r * squash * np.sin(a)]


SQUARE = [[220, 140], [400, 140], [400, 320], [220, 320]]
RECTANGLE = [[150, 180], [480, 180], [480, 300], [150, 300]]
TRIANGLE = [[320, 120], [440, 330], [200, 330]]


def shaky(points, rng, amount=4.0):
    return points + rng.normal(0, amount, points.shape)


@pytest.mark.parametrize("stroke, expected", [
    (circle(), sr.CIRCLE),
    (circle(60, overshoot=-0.03, squash=0.9), sr.CIRCLE),   # small, slightly open, a bit oval
    (outline(TRIANGLE), sr.TRIANGLE),
    (outline(SQUARE), sr.SQUARE),
    (outline(RECTANGLE), sr.RECTANGLE),
    (np.linspace([100, 200], [500, 260], 40), sr.LINE),
])
def test_clean_shapes(stroke, expected):
    assert sr.recognize(stroke).name == expected


@pytest.mark.parametrize("stroke, expected", [
    (circle(), sr.CIRCLE),
    (outline(TRIANGLE), sr.TRIANGLE),
    (outline(SQUARE), sr.SQUARE),
    (outline(RECTANGLE), sr.RECTANGLE),
    (np.linspace([100, 200], [500, 260], 40), sr.LINE),
])
def test_shaky_hand_drawn_shapes(stroke, expected):
    rng = np.random.default_rng(RNG_SEED)
    names = [getattr(sr.recognize(shaky(stroke, rng)), "name", None) for _ in range(50)]
    assert names.count(expected) >= 47, names


def test_tilted_square_is_still_a_square():
    t = np.radians(25)
    rotation = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    stroke = (outline(SQUARE) - [310, 230]) @ rotation.T + [310, 230]
    assert sr.recognize(stroke).name == sr.SQUARE


def test_scribbles_tiny_strokes_and_curves_are_not_shapes():
    rng = np.random.default_rng(RNG_SEED)
    scribbles = [sr.recognize(np.cumsum(rng.normal(0, 15, (60, 2)), axis=0) + 300) for _ in range(50)]
    assert sum(s is None for s in scribbles) >= 44
    assert sr.recognize(circle(10)) is None                     # too small to be deliberate
    assert sr.recognize(np.array([[0, 0], [1, 1]])) is None     # too few points
    arc = circle(100, overshoot=-0.5)                            # half a circle: open and curved
    assert sr.recognize(arc) is None


def test_clean_outlines_fit_what_was_drawn():
    cx, cy, r = sr.recognize(circle(100)).points
    assert abs(cx - 320) < 5 and abs(cy - 240) < 5 and abs(r - 100) < 5
    corners = sr.recognize(outline(SQUARE)).points
    assert len(corners) == 4
    assert np.allclose(sorted(corners[:, 0]), [220, 220, 400, 400], atol=3)
    assert len(sr.recognize(outline(TRIANGLE)).points) == 3
    a, b = sr.recognize(np.linspace([100, 200], [500, 260], 40)).points
    assert np.allclose(a, [100, 200]) and np.allclose(b, [500, 260])


def test_draw_shape_paints_the_outline():
    canvas = np.zeros((480, 640, 3), np.uint8)
    for stroke in (circle(), outline(TRIANGLE), np.linspace([100, 200], [500, 260], 40)):
        sr.draw_shape(canvas, sr.recognize(stroke))
    assert (canvas == sr.config.SHAPE_COLOUR).all(axis=2).sum() > 1000


# --- Drawing a shape with a pinch ------------------------------------------------------

PALM = 80


def pen_at(x, y, pinched=True):
    pts = make_hand(("index",))
    if pinched:
        pts[4] = pts[8] + [-0.05, 0.0, 0.0]     # thumb tip on the index tip
    image = pts * PALM + [x, y, 0]
    # Put the pen (between thumb and index tips) at (x, y).
    image[:, :2] -= image[[4, 8], :2].mean(axis=0) - [x, y]
    hand = DetectedHand("Right", 0.98, image / [640, 480, 640], pts * 0.09, image)
    return hand, extract_features(hand.world_landmarks, hand.image_points, "Right")


def pose(gesture=ga.UNKNOWN, newly_stable=False):
    return ga.GestureResult("Right", gesture, gesture, ga.STABLE, 9, 10, newly_stable)


def draw(canvas, points, pinched=True):
    shapes = []
    for x, y in points:
        hand, f = pen_at(x, y, pinched)
        shapes += canvas.update((480, 640, 3), [hand], [f], [pose()])
    return shapes


def lift(canvas, at=(320, 240)):
    """Open the fingers and keep them open until the stroke is recognised."""
    return draw(canvas, [at] * (config.DRAW_JOIN_FRAMES + 1), pinched=False)


def test_lifting_the_pen_turns_a_drawn_circle_into_a_clean_one():
    canvas = AirCanvas()
    assert draw(canvas, circle(120, overshoot=0.15)) == []    # nothing until the pen lifts
    assert canvas.pen_down
    shapes = lift(canvas, (320, 240))
    assert [(s.hand, s.shape.name) for s in shapes] == [("Right", sr.CIRCLE)]
    green = (canvas.canvas == sr.config.SHAPE_COLOUR).all(axis=2).sum()
    cyan = (canvas.canvas == sr.config.DRAW_COLOUR).all(axis=2).sum()
    assert green > 500 and cyan == 0    # the shaky stroke was replaced by the clean outline


def test_hand_leaving_the_picture_also_finishes_the_shape():
    canvas = AirCanvas()
    draw(canvas, outline(SQUARE, points=80))
    shapes = []
    for _ in range(config.DRAW_JOIN_FRAMES + 1):
        shapes += canvas.update((480, 640, 3), [], [], [])
    assert [s.shape.name for s in shapes] == [sr.SQUARE]


def test_unrecognised_strokes_stay_as_drawn():
    canvas = AirCanvas()
    zigzag = [(100 + 40 * i, 200 + (150 if i % 2 else 0)) for i in range(8)]
    draw(canvas, zigzag)
    assert lift(canvas, (400, 200)) == []
    assert (canvas.canvas == sr.config.DRAW_COLOUR).all(axis=2).any()


def test_shapes_can_be_turned_off():
    canvas = AirCanvas(shapes=False)
    draw(canvas, circle(120))
    assert lift(canvas, (320, 240)) == []


def test_earlier_drawing_is_kept_when_a_shape_is_cleaned_up():
    canvas = AirCanvas()
    draw(canvas, outline(TRIANGLE))
    lift(canvas, (0, 0))
    first = canvas.canvas.copy()
    draw(canvas, circle(60) - [200, 0])   # beside the triangle, not over it
    lift(canvas, (0, 0))
    assert (canvas.canvas[first.any(axis=2)] == first[first.any(axis=2)]).all()


def test_reasons_when_a_stroke_is_not_a_shape():
    assert sr.analyse(outline(RECTANGLE)).reason == sr.RECTANGLE
    assert sr.analyse(outline(RECTANGLE)[:40]).reason == "not closed: end where you started"
    assert sr.analyse(circle(10)).reason == "too small"
    star = np.array([[320 + r * np.cos(a), 240 + r * np.sin(a)]
                     for a, r in zip(np.linspace(0, 2 * np.pi, 11), [120, 50] * 5 + [120])])
    assert sr.analyse(star).reason.startswith(("unclear", "scribble"))


def test_a_pinch_that_loosens_at_a_corner_keeps_the_same_stroke():
    canvas = AirCanvas()
    stroke = outline(RECTANGLE, points=80)
    draw(canvas, stroke[:30])
    assert draw(canvas, [tuple(stroke[30])] * 3, pinched=False) == []   # pinch opens for 3 frames
    draw(canvas, stroke[31:])
    shapes = lift(canvas)
    assert [s.shape.name for s in shapes] == [sr.RECTANGLE]
    assert canvas.last_stroke == sr.RECTANGLE


def test_a_long_pause_starts_a_new_stroke():
    canvas = AirCanvas()
    stroke = outline(RECTANGLE, points=80)
    draw(canvas, stroke[:30])
    assert lift(canvas, tuple(stroke[30])) == []          # an L shape on its own: not a shape
    assert canvas.last_stroke == "not closed: end where you started"
    draw(canvas, stroke[31:])
    assert lift(canvas) == []


def test_a_one_frame_tracking_glitch_is_skipped():
    canvas = AirCanvas()
    points = list(map(tuple, circle(100)))
    points.insert(20, (620.0, 20.0))                      # the pen jumps to a corner for one frame
    draw(canvas, points)
    assert [s.shape.name for s in lift(canvas)] == [sr.CIRCLE]


def test_a_real_jump_starts_a_new_stroke():
    canvas = AirCanvas()
    draw(canvas, np.linspace([50, 100], [250, 100], 20))
    draw(canvas, np.linspace([400, 400], [600, 400], 20))   # the hand reappears far away
    assert canvas.pen_down
    assert len(canvas._stroke["Right"]) < 25
