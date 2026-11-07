"""Central configuration: paths, camera settings, detection settings and gesture thresholds.

Every other module reads its constants from here, so behaviour can be tuned in one place.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Paths -------------------------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DB_PATH = BASE_DIR / "data" / "gesture_history.db"

# --- Webcam (camera.py) --------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Selfie view. MediaPipe's Left/Right labels assume a mirrored image, so this also
# keeps the reported hand correct.
MIRROR_FRAME = True
# Consecutive failed reads before the camera is treated as disconnected.
MAX_FAILED_READS = 30

# --- Hand detection (hand_detector.py) ---------------------------------------
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- Feature extraction (feature_extractor.py) -------------------------------
# A finger counts as extended when the angle at its middle joint (MCP-PIP-TIP)
# is at least this many degrees (180 = perfectly straight).
FINGER_STRAIGHT_ANGLE = 150.0
# The thumb counts as extended when its tip sticks out past the index knuckle,
# sideways across the palm, by this fraction of the palm size.
THUMB_EXTENDED_LATERAL = 0.25
# Max angle (degrees) between the thumb and the vertical for "thumb up/down".
THUMB_VERTICAL_MAX_ANGLE = 45.0

# --- Gesture rules (gesture_analyzer.py) -------------------------------------
# Thumb tip and index tip closer than this (fraction of palm size) = touching.
OK_TOUCH_RATIO = 0.35
# Pinch: the index finger's middle-joint angle must be at least this many degrees.
# In a fist it is tightly curled (well below this); in a pinch it is only half bent.
PINCH_MIN_INDEX_ANGLE = 90.0
# Index and middle fingers spread at least this many degrees apart = Peace/Victory;
# held closer together = Two Fingers.
PEACE_MIN_SPREAD = 15.0
# Stability: a gesture is STABLE once it wins this many of the last N frames.
STABILITY_WINDOW = 10
STABILITY_MIN_AGREEMENT = 7
# MediaPipe sometimes loses a hand for a frame or two. Remember it this many frames
# (~0.5 s at 30 FPS) so a brief dropout doesn't log the same gesture again.
HAND_LOST_GRACE_FRAMES = 15

# --- Interface (ui.py) -------------------------------------------------------
WINDOW_NAME = "Hand Gesture Analyser"
HISTORY_REFRESH_SECONDS = 0.5
HISTORY_RECENT_ROWS = 5

# The 21 MediaPipe hand landmarks joined into a skeleton (landmark index pairs).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (9, 10), (10, 11), (11, 12),             # middle
    (13, 14), (14, 15), (15, 16),            # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
)
