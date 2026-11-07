"""Shape recognition in the draw mode (drawing itself: test_air_draw.py)."""
import numpy as np
import pytest

import shape_recognizer as sr

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


def test_reasons_when_a_stroke_is_not_a_shape():
    assert sr.analyse(outline(RECTANGLE)).reason == sr.RECTANGLE
    assert sr.analyse(outline(RECTANGLE)[:40]).reason == "not closed: end where you started"
    assert sr.analyse(circle(10)).reason == "too small"
    star = np.array([[320 + r * np.cos(a), 240 + r * np.sin(a)]
                     for a, r in zip(np.linspace(0, 2 * np.pi, 11), [120, 50] * 5 + [120])])
    assert sr.analyse(star).reason.startswith(("unclear", "scribble"))
