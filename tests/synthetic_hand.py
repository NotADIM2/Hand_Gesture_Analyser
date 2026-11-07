"""Build synthetic 21-point hands for tests, so gesture rules can be checked without a webcam.

Coordinates follow the image convention (x right, y down, z towards the camera is
negative) with the palm about 1 unit long. The hand is drawn upright, palm facing
the camera, and can then be rotated or scaled.
"""
import numpy as np

SEGMENTS = (0.40, 0.25, 0.20)                  # MCP->PIP, PIP->DIP, DIP->TIP
MCP_X = {"index": -0.30, "middle": -0.10, "ring": 0.10, "pinky": 0.28}
MCP_Y = {"index": -0.95, "middle": -1.00, "ring": -0.95, "pinky": -0.85}
FINGER_BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
CURL = (80.0, 100.0, 60.0)                     # bend at MCP, PIP, DIP for a folded finger
THUMB_CMC = np.array([-0.30, -0.25, 0.0])
THUMB_SEGMENTS = (0.30, 0.25, 0.20)            # CMC->MCP, MCP->IP, IP->TIP
THUMB_POSES = {
    "folded": np.array([0.60, -0.50, -0.60]),  # across the palm
    "out": np.array([-0.60, -0.80, 0.0]),      # spread sideways, like an open palm
    "side": np.array([-0.99, -0.15, 0.0]),     # straight out to the side (thumbs up/down)
}


def make_hand(extended=(), thumb="folded", spread=0.0, bends=None, curl=1.0) -> np.ndarray:
    """Hand with the given non-thumb fingers extended.

    spread: degrees the index and middle fingers lean apart (each moves half).
    bends: optional finger name -> (MCP, PIP, DIP) bend, overriding straight/curled.
    curl: how tightly folded fingers are bent (1 = fist, ~0.6 = loosely folded).
    """
    pts = np.zeros((21, 3))
    lean = {"index": -spread / 2, "middle": spread / 2}
    for name, base in FINGER_BASE.items():
        mcp = np.array([MCP_X[name], MCP_Y[name], 0.0])
        pts[base] = mcp
        finger_bends = (0.0, 0.0, 0.0) if name in extended else tuple(c * curl for c in CURL)
        finger_bends = (bends or {}).get(name, finger_bends)
        side = np.radians(lean.get(name, 0.0))
        total = 0.0
        pos = mcp
        for i, (length, bend) in enumerate(zip(SEGMENTS, finger_bends)):
            total += np.radians(bend)
            # Start pointing up (0, -1, 0), lean sideways, curl towards the camera.
            direction = np.array([np.sin(side) * np.cos(total), -np.cos(side) * np.cos(total), -np.sin(total)])
            pos = pos + length * direction
            pts[base + 1 + i] = pos

    direction = THUMB_POSES[thumb] / np.linalg.norm(THUMB_POSES[thumb])
    pts[1] = THUMB_CMC
    pos = THUMB_CMC
    for i, length in enumerate(THUMB_SEGMENTS):
        pos = pos + length * direction
        pts[2 + i] = pos
    return pts


def rotate_in_image(pts: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate the hand in the image plane (positive = clockwise on screen)."""
    a = np.radians(degrees)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return pts @ rot.T


def rotate_3d(pts: np.ndarray, x_deg: float, y_deg: float) -> np.ndarray:
    """Tilt the hand towards/away from the camera."""
    ax, ay = np.radians(x_deg), np.radians(y_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    return pts @ (ry @ rx).T


ALL_FINGERS = ("index", "middle", "ring", "pinky")


def gesture_hand(name: str, **options) -> np.ndarray:
    """A synthetic hand showing one of the recognised gestures; options go to make_hand (e.g. curl)."""
    if name == "OPEN PALM":
        return make_hand(ALL_FINGERS, thumb="out", spread=10, **options)
    if name == "FIST":
        return make_hand(**options)
    if name == "THUMBS UP":
        return rotate_in_image(make_hand(thumb="side", **options), 90)    # thumb now points up
    if name == "THUMBS DOWN":
        return rotate_in_image(make_hand(thumb="side", **options), -90)   # thumb now points down
    if name == "ONE FINGER":
        return make_hand(("index",), **options)
    if name == "TWO FINGERS":
        return make_hand(("index", "middle"), spread=0, **options)
    if name == "PEACE":
        return make_hand(("index", "middle"), spread=30, **options)
    if name == "THREE FINGERS":
        return make_hand(("index", "middle", "ring"), **options)
    if name == "OK SIGN":
        pts = make_hand(("middle", "ring", "pinky"), **options)
        touch = np.array([-0.45, -1.05, -0.35])
        pts[4] = touch                      # thumb tip
        pts[8] = touch + [0.05, 0.0, 0.0]   # index tip, just touching
        return pts
    if name == "FOUR FINGERS":
        return make_hand(ALL_FINGERS, **options)
    if name == "ROCK":
        return make_hand(("index", "pinky"), **options)
    if name == "CALL ME":
        return make_hand(("pinky",), thumb="out", **options)
    if name == "I LOVE YOU":
        return make_hand(("index", "pinky"), thumb="out", **options)
    if name == "L SIGN":
        return make_hand(("index",), thumb="out", **options)
    if name == "POINT LEFT":
        return rotate_in_image(make_hand(("index",), **options), -90)    # index now points left
    if name == "POINT RIGHT":
        return rotate_in_image(make_hand(("index",), **options), 90)     # index now points right
    if name == "PINCH":
        # Index half bent (not curled into a fist), thumb tip meeting its tip.
        pts = make_hand(bends={"index": (30.0, 50.0, 30.0)}, **options)
        pts[4] = pts[8] + [-0.05, 0.0, 0.0]
        return pts
    raise ValueError(name)
