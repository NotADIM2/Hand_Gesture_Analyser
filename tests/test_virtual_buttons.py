"""Virtual buttons: hover and pinch presses, the three button sets, and how the app uses them."""
import numpy as np

import app
import config
from air_draw import AirCanvas
from buttons import Button, VirtualButtons, mode_bar, paint_bar, talk_board
from feature_extractor import extract_features
from hand_detector import DetectedHand
from synthetic_hand import make_hand
from test_air_draw import Hand, outline
from ui import DRAW, MODES, POSES, WORDS

PALM, FPS, SHAPE = 80, 30, (480, 640, 3)
HOLD = config.BUTTON_HOLD_SECONDS


def finger_at(x, y, pinched=False, handedness="Right"):
    """A pointing hand with its index fingertip at (x, y)."""
    pts = make_hand(("index",))
    if pinched:
        pts[4] = pts[8] + [-0.05, 0.0, 0.0]
    image = pts * PALM
    image[:, :2] += np.array([x, y]) - image[8, :2]
    hand = DetectedHand(handedness, 0.98, image / [640, 480, 640], pts * 0.09, image)
    return hand, extract_features(hand.world_landmarks, hand.image_points, handedness)


A = Button("a", "A", (100, 100, 80, 50))
B = Button("b", "B", (100, 200, 80, 50))
QUIT = Button("quit", "QUIT", (300, 100, 80, 50), hold=config.BUTTON_QUIT_SECONDS)


class Finger:
    """Plays frames of one fingertip into a VirtualButtons at 30 a second."""

    def __init__(self, buttons=(A, B, QUIT)):
        self.panel, self.vb, self.t = list(buttons), VirtualButtons(), 0.0

    def at(self, x, y, seconds=0.0, pinched=False):
        clicked = []
        for _ in range(max(1, int(round(seconds * FPS)))):
            self.t += 1 / FPS
            hand, f = finger_at(x, y, pinched)
            clicked += self.vb.update(self.panel, [hand], [f], self.t)
        return clicked


def test_hovering_presses_once_the_bar_fills():
    finger = Finger()
    assert finger.at(140, 125, seconds=HOLD * 0.6) == []
    assert 0.4 < finger.vb.progress["a"] < 0.8              # the bar is filling up
    assert finger.at(140, 125, seconds=HOLD * 0.6) == ["a"]


def test_holding_still_does_not_repeat_until_the_finger_leaves():
    finger = Finger()
    assert finger.at(140, 125, seconds=HOLD * 3) == ["a"]
    finger.at(400, 400)                                     # off the button
    assert finger.at(140, 125, seconds=HOLD * 1.2) == ["a"]


def test_pinching_presses_at_once():
    finger = Finger()
    finger.at(140, 225)
    assert finger.at(140, 225, pinched=True) == ["b"]
    assert finger.at(140, 225, seconds=HOLD * 2, pinched=True) == []   # still pinched: no repeat


def test_moving_to_another_button_starts_again():
    finger = Finger()
    finger.at(140, 125, seconds=HOLD * 0.7)
    assert finger.at(140, 225, seconds=HOLD * 0.7) == []    # the time on A doesn't count for B
    assert "a" not in finger.vb.progress


def test_quit_needs_a_longer_hold():
    finger = Finger()
    assert finger.at(340, 125, seconds=HOLD * 1.5) == []
    assert finger.at(340, 125, seconds=config.BUTTON_QUIT_SECONDS) == ["quit"]


def test_hands_on_a_button_are_busy():
    finger = Finger()
    finger.at(140, 125)
    assert finger.vb.busy == {"Right"}
    finger.at(400, 400)
    assert finger.vb.busy == set()


def test_a_hand_that_leaves_the_picture_forgets_its_hover():
    vb = VirtualButtons()
    hand, f = finger_at(140, 125)
    vb.update([A], [hand], [f], 0.0)
    vb.update([A], [], [], HOLD * 0.5)
    assert vb.update([A], [hand], [f], HOLD * 1.1) == []    # timing starts again


def test_two_hands_press_their_own_buttons():
    vb = VirtualButtons()
    left, fl = finger_at(140, 125, handedness="Left")
    right, fr = finger_at(140, 225, handedness="Right")
    vb.update([A, B], [left, right], [fl, fr], 0.0)
    assert sorted(vb.update([A, B], [left, right], [fl, fr], HOLD + 0.01)) == ["a", "b"]


# --- The three button sets ---------------------------------------------------------------------

def overlaps(a, b):
    ax, ay, aw, ah = a.rect
    bx, by, bw, bh = b.rect
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def inside(button, shape=SHAPE):
    x, y, w, h = button.rect
    return x >= 0 and y >= 0 and x + w <= shape[1] and y + h <= shape[0]


def test_mode_bar_has_every_mode_full_screen_and_quit():
    bar = mode_bar(SHAPE, MODES, DRAW)
    assert [b.id for b in bar] == [f"mode:{m}" for m in MODES] + ["fullscreen", "quit"]
    assert [b.id for b in bar if b.active] == ["mode:draw"]
    assert bar[-1].hold == config.BUTTON_QUIT_SECONDS
    assert all(inside(b) for b in bar)
    assert not any(overlaps(a, b) for i, a in enumerate(bar) for b in bar[i + 1:])
    assert min(b.rect[1] for b in bar) > 40                 # below the FPS counter


def test_paint_bar_and_talk_board_sit_beside_the_mode_bar():
    bar = mode_bar(SHAPE, MODES, DRAW)
    for side in (paint_bar(SHAPE, config.DRAW_COLOUR, 6), talk_board(SHAPE)):
        assert all(inside(b) for b in side)
        assert not any(overlaps(a, b) for a in side for b in bar)
        assert max(b.rect[0] + b.rect[2] for b in side) < min(b.rect[0] for b in bar)


def test_paint_bar_shows_the_pen_colour_and_size():
    bar = paint_bar(SHAPE, config.PAINT_COLOURS["RED"], 10)
    assert [b.id for b in bar if b.active] == ["colour:RED"]
    assert "SIZE 10" in [b.label for b in bar]
    assert {"undo", "clear", "save"} <= {b.id for b in bar}


def test_talk_board_words_and_help_in_red():
    board = talk_board(SHAPE)
    assert [b.label for b in board] == list(config.TALK_WORDS)
    assert {b.label for b in board if b.swatch} == set(config.TALK_WORDS) & set(config.ALERT_WORDS)


def test_buttons_draw_with_hover_bar_and_flash():
    finger = Finger()
    finger.at(140, 125, seconds=HOLD * 0.5)
    frame = np.full(SHAPE, 128, np.uint8)
    finger.vb.draw(frame, finger.panel, finger.t)
    assert (frame[148, 105] == (80, 220, 80)).all()         # the green bar along the bottom of A
    finger.at(140, 125, seconds=HOLD * 0.6)                   # pressed
    frame = np.full(SHAPE, 128, np.uint8)
    finger.vb.draw(frame, finger.panel, finger.t)
    assert (frame[110, 105] == 255).all()                    # A flashes white


# --- Pen colour and size -----------------------------------------------------------------------

SQUARE = [[220, 140], [400, 140], [400, 320], [220, 320]]


def count(frame, colour):
    return (frame == colour).all(axis=2).sum()


def test_strokes_and_shapes_keep_their_pen_colour():
    canvas = AirCanvas()
    hand = Hand(canvas)
    canvas.colour = config.PAINT_COLOURS["RED"]
    hand.draw(outline(SQUARE)).lift()
    frame = np.zeros(SHAPE, np.uint8)
    canvas.overlay(frame)
    assert count(frame, config.PAINT_COLOURS["RED"]) > 1000 and count(frame, config.SHAPE_COLOUR) == 0
    canvas.colour = config.DRAW_COLOUR                         # the default pen still turns shapes green
    hand.lift(seconds=config.DRAW_GROUP_SECONDS + 0.5)
    hand.draw(outline([[20, 20], [120, 20], [120, 120], [20, 120]])).lift()
    frame = np.zeros(SHAPE, np.uint8)
    canvas.overlay(frame)
    assert count(frame, config.SHAPE_COLOUR) > 500 and count(frame, config.PAINT_COLOURS["RED"]) > 1000


def test_pen_size_cycles_and_thicker_lines_cover_more():
    canvas = AirCanvas()
    sizes = [canvas.next_size() for _ in config.PAINT_SIZES]
    assert sizes == list(config.PAINT_SIZES[1:]) + [config.PAINT_SIZES[0]]
    covered = []
    for size in (3, 16):
        c = AirCanvas(shapes=False)
        c.thickness = size
        Hand(c).draw(np.linspace((100, 300), (500, 300), 30)).lift()
        frame = np.zeros(SHAPE, np.uint8)
        c.overlay(frame)
        covered.append(frame.any(axis=2).sum())
    assert covered[1] > 2 * covered[0]


# --- The app's use of them ---------------------------------------------------------------------

def test_mode_buttons_stand_for_the_number_keys():
    assert app.BUTTON_KEYS["mode:poses"] == ord("1") and app.BUTTON_KEYS["mode:practice"] == ord("7")
    assert app.BUTTON_KEYS["quit"] in app.KEY_QUIT
    ids = {b.id for b in mode_bar(SHAPE, MODES, POSES) + paint_bar(SHAPE, config.DRAW_COLOUR, 6)}
    keyless = {i for i in ids if i not in app.BUTTON_KEYS}
    assert keyless == {f"colour:{name}" for name in config.PAINT_COLOURS} | {"size", "clear"}


def test_button_panel_per_mode():
    canvas = AirCanvas()
    assert len(app.button_panel(SHAPE, POSES, canvas)) == len(MODES) + 2
    assert any(b.id == "size" for b in app.button_panel(SHAPE, DRAW, canvas))
    assert any(b.id == "say:HELP" for b in app.button_panel(SHAPE, WORDS, canvas))


def test_hands_on_buttons_are_left_out_of_the_mode():
    left, fl = finger_at(140, 125, handedness="Left")
    right, fr = finger_at(400, 400, handedness="Right")
    hands, features, results = app.free_hands({"Left"}, [left, right], [fl, fr], ["l", "r"])
    assert hands == [right] and features == [fr] and results == ["r"]
