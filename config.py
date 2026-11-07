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

# --- Moving and two-hand gestures (motion_analyzer.py) -----------------------
# One-hand positions are in frame widths (1.0 = the full width of the frame).
MOTION_HISTORY_SECONDS = 2.0    # how much of each hand's path is kept
MOTION_COOLDOWN_SECONDS = 0.8   # after a moving gesture, ignore that hand this long
# A jump bigger than this between two frames is a tracking glitch, not movement.
MOTION_MAX_JUMP = 0.2
# Swipe: at least this far within this time, fairly straight, in one main direction.
SWIPE_MAX_SECONDS = 0.6
SWIPE_MIN_DISTANCE = 0.25
SWIPE_MAX_PATH_RATIO = 1.3      # path length / straight-line distance
SWIPE_MIN_DOMINANCE = 1.5       # main direction vs sideways drift
# Wave: this many left/right swings of at least this size within this time.
WAVE_SECONDS = 1.5
WAVE_MIN_AMPLITUDE = 0.04
WAVE_MIN_SWINGS = 4
# Circle: the palm turns at least this many degrees round a path of at least this
# radius within this time. Roundness: closest point / average distance from the centre.
CIRCLE_SECONDS = 2.0
CIRCLE_MIN_RADIUS = 0.04
CIRCLE_MIN_ROUNDNESS = 0.4
CIRCLE_MIN_TURN = 330.0
# Two-hand distances are in palm sizes (wrist to middle knuckle on screen).
# Clap: palm centres at least CLAP_START_DISTANCE apart, then within
# CLAP_TOUCH_DISTANCE, all within CLAP_SECONDS.
CLAP_SECONDS = 0.6
CLAP_START_DISTANCE = 2.0
CLAP_TOUCH_DISTANCE = 0.8
# Heart: index tips touching and thumb tips touching, wrists apart, held a few frames.
# MediaPipe's fingertips are rough when two hands touch, so "touching" is generous.
# Press f in the app to see the live values.
HEART_TOUCH_DISTANCE = 0.6
HEART_MIN_WRIST_GAP = 0.8
HEART_HOLD_FRAMES = 5

# --- Interface (ui.py) -------------------------------------------------------
WINDOW_NAME = "Hand Gesture Analyser"
HISTORY_REFRESH_SECONDS = 0.5
HISTORY_RECENT_ROWS = 5
# How long a moving or two-hand gesture stays on screen after it happens.
MOTION_DISPLAY_SECONDS = 2.5

# The 21 MediaPipe hand landmarks joined into a skeleton (landmark index pairs).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (9, 10), (10, 11), (11, 12),             # middle
    (13, 14), (14, 15), (15, 16),            # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
)
