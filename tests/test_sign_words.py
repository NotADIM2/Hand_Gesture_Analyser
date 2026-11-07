"""Sign words: each sign played as a short synthetic video, 30 frames a second."""
import numpy as np
import pytest

import sign_words as sw
from feature_extractor import extract_features
from hand_detector import DetectedHand
from synthetic_hand import ALL_FINGERS, gesture_hand, make_hand, rotate_3d

W, H, PALM, FPS = 640, 480, 70, 30


def placed(pts, x, y, hand="Right"):
    if hand == "Left":
        pts = pts * [-1, 1, 1]      # mirror image of a right hand
    image = pts * PALM + [x, y, 0]
    return DetectedHand(hand, 0.98, image / [W, H, W], pts * 0.09, image)


def play(scenes, signer=None):
    """scenes: one list of (points, x, y[, hand]) per frame. Returns the words recognised."""
    signer = signer or sw.SignWords()
    words = []
    for i, scene in enumerate(scenes):
        hands = [placed(*s) for s in scene]
        features = [extract_features(h.world_landmarks, h.image_points, h.handedness) for h in hands]
        word = signer.update(hands, features, (H, W, 3), i / FPS)
        if word:
            words.append(word)
    return words


def seconds(s):
    return np.arange(int(s * FPS)) / FPS


def wave(t, amplitude, hz):
    return amplitude * np.sin(2 * np.pi * hz * t)


# Hand shapes
FLAT = make_hand(ALL_FINGERS, thumb="out")
FIST = make_hand()
A_SHAPE = gesture_hand("THUMBS UP")
INDEX = make_hand(("index",))
L_SHAPE = make_hand(("index",), thumb="out")
Y_SHAPE = make_hand(("pinky",), thumb="out")
W_SHAPE = make_hand(("index", "middle", "ring"))
V_OPEN = make_hand(("index", "middle"), thumb="out")


def flat_o():
    pts = make_hand(bends={n: (50.0, 40.0, 20.0) for n in ALL_FINGERS})
    pts[4] = pts[[8, 12, 16, 20]].mean(axis=0) + [0.0, 0.0, -0.1]   # thumb tip among the fingertips
    return pts


def snapped():
    pts = V_OPEN.copy()
    pts[8] = pts[4] + [0.05, 0.0, 0.0]
    pts[12] = pts[4] + [0.1, 0.0, 0.0]
    return pts


def flapped():
    return make_hand(ALL_FINGERS, thumb="out", bends={n: (80.0, 10.0, 5.0) for n in ALL_FINGERS})


def hooked():
    return make_hand(("index",), thumb="out", bends={"index": (20.0, 90.0, 60.0)})


def wiggling(t, phase):
    return make_hand(ALL_FINGERS, thumb="out", bends={
        n: (20 + 20 * np.sin(2 * np.pi * 2.5 * t + phase + i), 0.0, 0.0) for i, n in enumerate(ALL_FINGERS)})


def alternate(a, b, frames_each=5, total=40):
    return [a if (i // frames_each) % 2 == 0 else b for i in range(total)]


# --- One hand ----------------------------------------------------------------------------

def one(shapes_xy):
    return [[s] for s in shapes_xy]


ONE_HAND = {
    sw.HELLO: lambda: one([(FLAT, 300 + 6 * i, 150) for i in range(24)]),
    sw.GOODBYE: lambda: one([(p, 320, 200) for p in alternate(FLAT, flapped())]),
    sw.PLEASE: lambda: one([(FLAT, 320 + 40 * np.cos(a), 360 + 40 * np.sin(a))
                            for a in np.linspace(0, 2 * np.pi, 45)]),
    sw.SORRY: lambda: one([(FIST, 320 + 40 * np.cos(a), 360 + 40 * np.sin(a))
                           for a in np.linspace(0, 2 * np.pi, 45)]),
    sw.THANK_YOU: lambda: one([(FLAT, 320, 200 + 5 * i) for i in range(20)]),
    sw.BAD: lambda: one([(rotate_3d(FLAT, 150 * i / 19, 0), 320, 200 + 5 * i) for i in range(20)]),
    sw.YES: lambda: one([(rotate_3d(FIST, 25 * np.sin(2 * np.pi * 2 * t), 0), 320, 300) for t in seconds(1.5)]),
    sw.NO: lambda: one([(p, 320, 300) for p in alternate(V_OPEN, snapped())]),
    sw.WHO: lambda: one([(p, 320, 250) for p in alternate(L_SHAPE, hooked())]),
    sw.WHERE: lambda: one([(INDEX, 320 + wave(t, 30, 2), 250) for t in seconds(1.5)]),
    sw.SAME: lambda: one([(Y_SHAPE, 320 + wave(t, 40, 1.5), 300) for t in seconds(1.5)]),
    sw.WATER: lambda: one([(W_SHAPE, 320, 250 + wave(t, 10, 3)) for t in seconds(1.5)]),
    sw.HOME: lambda: one([(flat_o(), 300 - min(i, 25) * 1.6, 200 - min(i, 25) * 1.2) for i in range(35)]),
    sw.ME: lambda: one([(rotate_3d(INDEX, -70, 0), 320, 300)] * 25),
    sw.YOU: lambda: one([(rotate_3d(INDEX, 70, 0), 320, 300)] * 25),
}


@pytest.mark.parametrize("word", ONE_HAND)
def test_one_hand_signs(word):
    assert play(ONE_HAND[word]()) == [word]


# --- Two hands -----------------------------------------------------------------------------

def stop_or_good(mover):
    base = (FLAT, 320, 330, "Left")
    scenes = [[base, (mover, 330, 150 + 130 * i / 14)] for i in range(15)]
    return scenes + [[base, (mover, 330, 280)]] * 6


TWO_HANDS = {
    sw.MORE: lambda: [[(flat_o(), 320 - d, 300, "Left"), (flat_o(), 320 + d, 300)]
                      for d in 65 + wave(seconds(1.5), 35, 2)],
    sw.WAIT: lambda: [[(wiggling(t, 0.0), 220, 300, "Left"), (wiggling(t, 1.0), 420, 300)] for t in seconds(1.5)],
    sw.WHAT: lambda: [[(FLAT, 220 + wave(t, 30, 2), 330, "Left"), (FLAT, 420 + wave(t, 30, 2), 330)]
                      for t in seconds(1.5)],
    sw.STOP: lambda: stop_or_good(rotate_3d(FLAT, 0, 90)),     # edge of the hand chops down
    sw.GOOD: lambda: stop_or_good(FLAT),                       # palm comes down flat
    sw.HELP: lambda: [[(FLAT, 320, 360 - 5 * i, "Left"), (A_SHAPE, 320, 300 - 5 * i)] for i in range(22)],
    sw.GO: lambda: [[(INDEX, 200 + 7 * i, 300, "Left"), (INDEX, 400 + 7 * i, 300)] for i in range(18)],
    sw.DANGER: lambda: [[(FIST, 300, 320, "Left"), (FIST, 345, 300 + wave(t, 12, 3))] for t in seconds(1.5)],
}


@pytest.mark.parametrize("word", TWO_HANDS)
def test_two_hand_signs(word):
    assert play(TWO_HANDS[word]()) == [word]


def test_every_word_has_a_test():
    assert set(ONE_HAND) | set(TWO_HANDS) == set(sw.WORDS) and len(sw.WORDS) == 23


# --- Nothing by accident --------------------------------------------------------------------

@pytest.mark.parametrize("scene", [
    [(FLAT, 320, 300)], [(FIST, 320, 300)], [(INDEX, 320, 300)], [(Y_SHAPE, 320, 300)],
    [(FLAT, 220, 300, "Left"), (FLAT, 420, 300)], [(FIST, 220, 300, "Left"), (INDEX, 420, 300)],
], ids=["flat", "fist", "index up", "Y", "two flat", "fist and index"])
def test_hands_held_still_say_nothing(scene):
    assert play([scene] * 60) == []


def test_slow_drift_says_nothing():
    assert play(one([(FLAT, 300 + i, 300 + i * 0.5) for i in range(60)])) == []


def test_words_build_a_sentence_with_a_pause_between_signs():
    signer = sw.SignWords()
    scenes = ONE_HAND[sw.ME]() + one([(FLAT, 320, 450)] * 35) + TWO_HANDS[sw.HELP]()
    assert play(scenes, signer) == [sw.ME, sw.HELP]
    assert signer.text == "ME HELP"
    signer.clear()
    assert signer.text == ""


def test_a_held_sign_is_said_once():
    assert play(ONE_HAND[sw.ME]() * 4) == [sw.ME]          # held for over 3 seconds
