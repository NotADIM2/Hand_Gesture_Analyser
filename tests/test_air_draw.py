"""Drawing with a pinch: strokes, shapes in one or several strokes, undo, clearing."""
import numpy as np

import config
import gesture_analyzer as ga
import shape_recognizer as sr
from air_draw import AirCanvas, chain
from feature_extractor import extract_features
from hand_detector import DetectedHand
from synthetic_hand import make_hand

PALM, FPS, SHAPE = 80, 30, (480, 640, 3)
CYAN, GREEN = config.DRAW_COLOUR, config.SHAPE_COLOUR


def pen_at(x, y, pinched=True):
    pts = make_hand(("index",))
    if pinched:
        pts[4] = pts[8] + [-0.05, 0.0, 0.0]     # thumb tip on the index tip
    image = pts * PALM + [x, y, 0]
    image[:, :2] -= image[[4, 8], :2].mean(axis=0) - [x, y]   # the pen (between the tips) at (x, y)
    hand = DetectedHand("Right", 0.98, image / [640, 480, 640], pts * 0.09, image)
    return hand, extract_features(hand.world_landmarks, hand.image_points, "Right")


class Hand:
    """Plays frames into a canvas at 30 a second."""

    def __init__(self, canvas=None):
        self.canvas = canvas or AirCanvas()
        self.t = 0.0
        self.shapes = []

    def frame(self, x, y, pinched, gesture=ga.UNKNOWN, newly_stable=False):
        hand, f = pen_at(x, y, pinched)
        result = ga.GestureResult("Right", gesture, gesture, ga.STABLE, 9, 10, newly_stable)
        self.t += 1 / FPS
        found = self.canvas.update(SHAPE, [hand], [f], [result], self.t)
        self.shapes += found
        return found

    def draw(self, points):
        for x, y in points:
            self.frame(x, y, True)
        return self

    def lift(self, seconds=config.DRAW_JOIN_SECONDS + 0.1, at=None):
        """Open the fingers and wait (by default until the stroke is finished)."""
        x, y = at if at is not None else self.canvas.pens.get("Right", ((320, 240), False))[0]
        found = []
        for _ in range(int(seconds * FPS) + 1):
            found += self.frame(x, y, False)
        return [s.shape.name for s in found]

    def gone(self, seconds=config.DRAW_JOIN_SECONDS + 0.1):
        found = []
        for _ in range(int(seconds * FPS) + 1):
            self.t += 1 / FPS
            found += self.canvas.update(SHAPE, [], [], [], self.t)
        return [s.shape.name for s in found]


def outline(corners, points=80):
    corners = np.vstack([corners, corners[:1]]).astype(float)
    per_side = points // (len(corners) - 1)
    out = [a + (b - a) * t for a, b in zip(corners[:-1], corners[1:])
           for t in np.linspace(0, 1, per_side, endpoint=False)]
    return np.array(out + [corners[0]])


def circle(r=120.0, overshoot=0.1, centre=(320, 240)):
    a = np.linspace(0, 2 * np.pi * (1 + overshoot), 60)
    return np.c_[centre[0] + r * np.cos(a), centre[1] + r * np.sin(a)]


def side(a, b, n=20):
    return np.linspace(a, b, n)


SQUARE = [[220, 140], [400, 140], [400, 320], [220, 320]]
RECTANGLE = [[150, 180], [480, 180], [480, 300], [150, 300]]
TRIANGLE = [[320, 120], [440, 330], [200, 330]]


def rendered(canvas):
    frame = np.zeros(SHAPE, np.uint8)
    canvas.overlay(frame)
    return (frame == CYAN).all(axis=2).sum(), (frame == GREEN).all(axis=2).sum()


# --- Drawing -------------------------------------------------------------------------------

def test_pinch_draws_and_open_fingers_dont():
    hand = Hand().draw(side((100, 300), (400, 300), 30))
    assert hand.canvas.pen_down and rendered(hand.canvas)[0] > 500
    other = Hand()
    for x in range(100, 400, 10):
        other.frame(x, 300, pinched=False)
    assert not other.canvas.pen_down and other.canvas.empty


def test_pen_shows_where_it_is_down_or_up():
    hand = Hand()
    hand.frame(200, 200, pinched=False)
    (_, down), = hand.canvas.pens.values()
    assert not down
    hand.frame(200, 200, pinched=True)
    (_, down), = hand.canvas.pens.values()
    assert down


def test_lifting_the_pen_turns_a_circle_into_a_clean_one():
    hand = Hand().draw(circle(overshoot=0.15))
    assert hand.shapes == []                         # nothing until the pen lifts
    assert hand.lift() == [sr.CIRCLE]
    cyan, green = rendered(hand.canvas)
    assert green > 500 and cyan < 50                 # the shaky stroke became a clean outline
    assert hand.canvas.last_stroke == sr.CIRCLE


def test_hand_leaving_the_picture_also_finishes_the_shape():
    hand = Hand().draw(outline(SQUARE))
    assert hand.gone() == [sr.SQUARE]


def test_unrecognised_strokes_stay_as_drawn():
    hand = Hand().draw([(100 + 40 * i, 200 + (150 if i % 2 else 0)) for i in range(8)])
    assert hand.lift() == []
    assert rendered(hand.canvas)[0] > 300 and len(hand.canvas.items) == 1


def test_shapes_can_be_turned_off():
    hand = Hand(AirCanvas(shapes=False)).draw(circle())
    assert hand.lift() == [] and hand.canvas.items[0].shape is None


def test_earlier_shapes_stay_when_new_ones_are_drawn():
    hand = Hand().draw(outline(TRIANGLE))
    hand.lift(seconds=config.DRAW_GROUP_SECONDS + 0.5)
    hand.draw(circle(60, centre=(520, 380)))
    hand.lift()
    assert [i.shape.name for i in hand.canvas.items] == [sr.TRIANGLE, sr.CIRCLE]


def test_a_fist_no_longer_wipes_the_drawing():
    hand = Hand().draw(circle())
    hand.lift()
    hand.frame(300, 300, pinched=False, gesture=ga.FIST, newly_stable=True)   # relaxing the hand
    assert len(hand.canvas.items) == 1


# --- Strokes that break, and shapes in several strokes ---------------------------------------

def test_a_pinch_that_loosens_at_a_corner_keeps_the_same_stroke():
    stroke = outline(RECTANGLE)
    hand = Hand().draw(stroke[:30])
    assert hand.lift(seconds=0.1, at=stroke[30]) == []   # opens for a moment
    hand.draw(stroke[31:])
    assert hand.lift() == [sr.RECTANGLE]
    assert hand.canvas.last_stroke == sr.RECTANGLE


def test_rectangle_in_two_strokes_with_a_pause():
    stroke = outline(RECTANGLE)
    hand = Hand().draw(stroke[:30])
    assert hand.lift(seconds=1.0, at=stroke[30]) == []    # an L on its own: not a shape yet
    hand.draw(stroke[30:])
    assert hand.lift() == [sr.RECTANGLE]
    assert len(hand.canvas.items) == 1 and hand.canvas.last_stroke == "RECTANGLE (2 strokes)"


def test_square_from_four_separate_sides():
    a, b, c, d = SQUARE
    hand = Hand()
    found = []
    for start, end in ((a, b), (b, c), (c, d), (d, a)):
        hand.draw(side(start, end))
        found += hand.lift(seconds=0.8)
    assert found == [sr.LINE, sr.LINE, sr.LINE, sr.SQUARE]   # each side is a line until the last
    assert [i.shape.name for i in hand.canvas.items] == [sr.SQUARE]
    cyan, green = rendered(hand.canvas)
    assert green > 1000 and cyan < 50


def test_triangle_from_sides_drawn_in_any_direction():
    a, b, c = TRIANGLE
    hand = Hand()
    for start, end in ((a, b), (a, c), (b, c)):              # second and third drawn "backwards"
        hand.draw(side(start, end))
        hand.lift(seconds=0.8)
    assert [i.shape.name for i in hand.canvas.items] == [sr.TRIANGLE]


def test_lines_that_dont_meet_stay_lines():
    hand = Hand()
    for y in (100, 200, 300, 400):
        hand.draw(side((100, y), (400, y)))
        hand.lift(seconds=0.8)
    assert [i.shape.name for i in hand.canvas.items] == [sr.LINE] * 4


def test_sides_drawn_too_far_apart_in_time_dont_join():
    a, b, c = TRIANGLE
    hand = Hand()
    for start, end in ((a, b), (b, c), (c, a)):
        hand.draw(side(start, end))
        hand.lift(seconds=config.DRAW_GROUP_SECONDS + 0.5)
    assert [i.shape.name for i in hand.canvas.items] == [sr.LINE] * 3


def test_chain_turns_strokes_round_to_meet():
    path = chain([side((0, 0), (100, 0)), side((100, 100), (100, 0)), side((0, 100), (100, 100))])
    assert np.allclose(path[0], (0, 0)) and np.allclose(path[-1], (0, 100))
    assert chain([side((0, 0), (100, 0)), side((300, 300), (400, 300))]) is None


# --- Glitches, undo, clear ---------------------------------------------------------------------

def test_a_one_frame_tracking_glitch_is_skipped():
    points = list(map(tuple, circle(100)))
    points.insert(20, (620.0, 20.0))                       # the pen jumps to a corner for one frame
    hand = Hand().draw(points)
    assert hand.lift() == [sr.CIRCLE]


def test_a_real_jump_starts_a_new_stroke():
    hand = Hand().draw(side((50, 100), (250, 100))).draw(side((400, 400), (600, 400)))
    assert hand.canvas.pen_down and len(hand.canvas._stroke["Right"]) < 25


def test_undo_and_clear():
    hand = Hand().draw(circle())
    hand.lift(seconds=config.DRAW_GROUP_SECONDS + 0.5)
    hand.draw(side((50, 450), (300, 450)))
    hand.lift()
    assert [i.shape.name for i in hand.canvas.items] == [sr.CIRCLE, sr.LINE]
    assert hand.canvas.undo() and [i.shape.name for i in hand.canvas.items] == [sr.CIRCLE]
    hand.canvas.clear()
    assert hand.canvas.empty and not hand.canvas.undo()


def test_overlay_draws_on_any_frame_size():
    hand = Hand().draw(circle())
    hand.lift()
    small = np.zeros((240, 320, 3), np.uint8)
    hand.canvas.overlay(small)                             # no fixed-size canvas to match
    assert small.any()
