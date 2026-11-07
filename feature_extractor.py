"""Convert hand landmarks into gesture-related features (architecture step: FEATURE EXTRACTION).

Works on plain landmark arrays, never on camera frames, so it can be tested without
a webcam.

Two views of the same 21 landmarks are used:
- world landmarks (3D, metres) for angles and distances, which don't depend on how
  far the hand is from the camera;
- image points (pixels, y pointing down) for directions on screen, e.g. thumb up/down.
"""
from dataclasses import dataclass

import numpy as np

import config

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
PINKY_MCP = 17

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
# (MCP, PIP, DIP, TIP) landmark indices for the four non-thumb fingers.
FINGER_JOINTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


@dataclass(frozen=True)
class HandFeatures:
    handedness: str
    extended: dict[str, bool]          # finger name -> extended?
    finger_angles: dict[str, float]    # bend angle at the middle joint, 180 = straight
    thumb_lateral: float               # how far the thumb tip sticks out past the index knuckle
    thumb_index_distance: float        # thumb tip to index tip
    index_middle_spread: float         # degrees between index and middle fingers
    thumb_direction: str | None        # "up", "down" or None
    hand_direction: str                # where the fingers point on screen: up/down/left/right

    @property
    def extended_fingers(self) -> frozenset[str]:
        return frozenset(name for name in FINGER_NAMES if self.extended[name])

    @property
    def extended_count(self) -> int:
        return len(self.extended_fingers)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b (degrees) formed by a-b-c."""
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def vector_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two vectors (degrees)."""
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1.0, 1.0))))


def extract_features(
    world_landmarks: np.ndarray,
    image_points: np.ndarray,
    handedness: str = "Unknown",
) -> HandFeatures:
    world = _as_landmarks(world_landmarks, "world_landmarks")
    image = _as_landmarks(image_points, "image_points")

    # All distances are divided by the palm size (wrist to middle knuckle) so they
    # are comparable between big and small hands.
    palm_size = float(np.linalg.norm(world[MIDDLE_MCP] - world[WRIST])) or 1.0

    extended: dict[str, bool] = {}
    angles: dict[str, float] = {}
    for name, (mcp, pip, _dip, tip) in FINGER_JOINTS.items():
        angles[name] = joint_angle(world[mcp], world[pip], world[tip])
        tip_further_than_joint = (
            np.linalg.norm(world[tip] - world[WRIST]) > np.linalg.norm(world[pip] - world[WRIST])
        )
        extended[name] = angles[name] >= config.FINGER_STRAIGHT_ANGLE and bool(tip_further_than_joint)

    # Thumb: measure how far its tip reaches beyond the index knuckle along the
    # palm's width (pinky knuckle -> index knuckle). Folded across the palm this is
    # negative; sticking out to the side or up it is clearly positive. This works
    # however the hand is rotated.
    across_palm = world[INDEX_MCP] - world[PINKY_MCP]
    across_palm /= np.linalg.norm(across_palm) or 1.0
    thumb_lateral = float(np.dot(world[THUMB_TIP] - world[INDEX_MCP], across_palm)) / palm_size
    angles["thumb"] = joint_angle(world[THUMB_MCP], world[THUMB_IP], world[THUMB_TIP])
    extended["thumb"] = thumb_lateral >= config.THUMB_EXTENDED_LATERAL

    return HandFeatures(
        handedness=handedness,
        extended=extended,
        finger_angles=angles,
        thumb_lateral=thumb_lateral,
        thumb_index_distance=float(np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP])) / palm_size,
        index_middle_spread=vector_angle(
            world[INDEX_TIP] - world[INDEX_MCP], world[MIDDLE_TIP] - world[MIDDLE_MCP]
        ),
        thumb_direction=_thumb_direction(image),
        hand_direction=_screen_direction(image[MIDDLE_MCP, :2] - image[WRIST, :2]),
    )


def _thumb_direction(image: np.ndarray) -> str | None:
    """'up'/'down' when the thumb points vertically and its tip is the highest/lowest point."""
    thumb = image[THUMB_TIP, :2] - image[THUMB_MCP, :2]
    others_y = np.delete(image[:, 1], THUMB_TIP)
    tip_y = image[THUMB_TIP, 1]
    # Image y grows downwards, so "up" is (0, -1).
    if vector_angle(thumb, np.array([0.0, -1.0])) <= config.THUMB_VERTICAL_MAX_ANGLE and tip_y < others_y.min():
        return "up"
    if vector_angle(thumb, np.array([0.0, 1.0])) <= config.THUMB_VERTICAL_MAX_ANGLE and tip_y > others_y.max():
        return "down"
    return None


def _screen_direction(vector: np.ndarray) -> str:
    dx, dy = float(vector[0]), float(vector[1])
    if abs(dy) >= abs(dx):
        return "up" if dy < 0 else "down"
    return "right" if dx > 0 else "left"


def _as_landmarks(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape != (21, 3):
        raise ValueError(f"{name} must have shape (21, 3), got {arr.shape}")
    return arr


if __name__ == "__main__":
    # Module check: print the features of every hand in view. Press q to quit.
    import cv2

    from camera import Camera, to_rgb
    from hand_detector import HandDetector
    from ui import draw_hand

    with Camera() as cam, HandDetector() as detector:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            for hand in detector.detect(to_rgb(frame)):
                f = extract_features(hand.world_landmarks, hand.image_points, hand.handedness)
                fingers = "".join("1" if f.extended[n] else "0" for n in FINGER_NAMES)
                draw_hand(frame, hand, f"{f.handedness} {fingers} thumb:{f.thumb_direction}")
            cv2.imshow("feature extractor check", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
