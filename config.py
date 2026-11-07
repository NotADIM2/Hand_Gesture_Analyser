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
# Trained k-NN classifier (python scripts/train_classifier.py).
ML_MODEL_PATH = BASE_DIR / "models" / "gesture_knn.npz"
# Poses taught live with the t key (teach.py), one file per pose. Committed, so a
# pose taught on one computer is recognised by everyone who pulls it.
TAUGHT_POSES_DIR = BASE_DIR / "taught_poses"
BENCHMARK_REPORT_PATH = BASE_DIR / "reports" / "benchmark.md"

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

# --- Machine-learning classifier (ml_classifier.py) ---------------------------
# Neighbours that vote on each hand's label.
ML_K = 5
# A hand further than this from every training example is UNKNOWN.
ML_MAX_DISTANCE = 0.6
# Augmented synthetic hands per gesture in the training set.
ML_SYNTHETIC_PER_GESTURE = 40

# --- Teaching new poses (teach.py) ---------------------------------------------
TEACH_EXAMPLES = 30            # examples recorded per teaching (about 2 seconds)
TEACH_COUNTDOWN_SECONDS = 3.0  # time to get the pose ready after pressing Enter
TEACH_MAX_NAME = 16
# A hand counts as a taught pose only when this close to its examples (same units as
# ML_MAX_DISTANCE). Strict, so taught poses don't take over the built-in ones.
TAUGHT_MAX_DISTANCE = 0.3
# Two taught poses clash when more than this share of one's examples look like the other.
TAUGHT_MAX_CONFUSION = 0.1

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

# --- Gesture control (controller.py) ------------------------------------------
# Each profile maps a gesture to (label shown on screen, key to press[, True to keep
# pressing while the pose is held]). Keys are pynput Key names (right, left, f5, esc,
# media_play_pause, media_volume_up, ...) or a single character. Press m in the app
# to switch between off and these profiles, in this order.
# Open Palm is deliberately unused: it's how a hand looks just before a swipe.
CONTROL_PROFILES = {
    "presentation": {
        "SWIPE LEFT": ("NEXT SLIDE", "right"),
        "SWIPE RIGHT": ("PREVIOUS SLIDE", "left"),
        "THUMBS UP": ("START SLIDESHOW", "f5"),
        "THUMBS DOWN": ("END SLIDESHOW", "esc"),
        "FIST": ("BLACK SCREEN", "b"),
    },
    "media": {
        "SWIPE LEFT": ("NEXT TRACK", "media_next"),
        "SWIPE RIGHT": ("PREVIOUS TRACK", "media_previous"),
        "OK SIGN": ("PLAY / PAUSE", "media_play_pause"),
        "THUMBS UP": ("VOLUME UP", "media_volume_up", True),
        "THUMBS DOWN": ("VOLUME DOWN", "media_volume_down", True),
        "FIST": ("MUTE", "media_volume_mute"),
    },
}
# Gesture password: these poses, in order, unlock control mode after it is switched on.
# Each must follow the previous within PASSWORD_STEP_SECONDS. Use () for no password.
CONTROL_PASSWORD = ("FIST", "PEACE", "THUMBS UP")
PASSWORD_STEP_SECONDS = 4.0
# Minimum time between two actions, so one gesture can't press a key twice.
CONTROL_COOLDOWN_SECONDS = 0.5
# How often a held pose repeats an action marked to repeat (e.g. volume).
CONTROL_REPEAT_SECONDS = 0.4

# --- Air drawing (air_draw.py) -----------------------------------------------
# The pen goes down when thumb and index tips are closer than DRAW_TOUCH_RATIO
# (fraction of palm size) and lifts only when they open past DRAW_LIFT_RATIO.
DRAW_TOUCH_RATIO = 0.35
DRAW_LIFT_RATIO = 0.5
DRAW_SMOOTHING = 0.5           # 0 = raw fingertip, closer to 1 = smoother but laggier
DRAW_COLOUR = (255, 255, 0)    # BGR: cyan
DRAW_THICKNESS = 6
DRAW_JOIN_FRAMES = 12          # pen up this many frames (~0.4 s) = stroke finished, recognise it
DRAW_JOIN_DISTANCE = 100.0     # pen down again within this many pixels before then = same stroke
DRAW_MAX_JUMP = 0.25           # pen jumping this far (fraction of frame width) in one frame = glitch
DRAW_GLITCH_FRAMES = 2         # ...unless it stays there longer than this: then the hand really moved
SNAPSHOT_DIR = BASE_DIR / "data" / "snapshots"

# --- Sign letters (sign_language.py) -----------------------------------------
# Poses that match American Sign Language (ASL) fingerspelling letters.
SIGN_LETTERS = {
    "FOUR FINGERS": "B",
    "ONE FINGER": "D",
    "OK SIGN": "F",
    "L SIGN": "L",
    "FIST": "S",
    "TWO FINGERS": "U",
    "PEACE": "V",
    "THREE FINGERS": "W",
    "CALL ME": "Y",
}
SIGN_MAX_LENGTH = 24

# --- Exercise coach (exercise_coach.py) -----------------------------------------
# (exercise, repetitions) in order. Names: OPEN AND CLOSE, THUMB TOUCHES, COUNT TO FIVE.
EXERCISE_ROUTINE = (("OPEN AND CLOSE", 10), ("THUMB TOUCHES", 3), ("COUNT TO FIVE", 3))
# Average finger bend angle: at least this = hand open, at most this = fist.
EXERCISE_OPEN_ANGLE = 150.0
EXERCISE_CLOSED_ANGLE = 110.0
# Thumb touches (fraction of palm size): touching below TOUCH, let go above RELEASE.
EXERCISE_TOUCH_RATIO = 0.35
EXERCISE_RELEASE_RATIO = 0.55
# Count to five: frames a finger count must be held before it counts.
EXERCISE_HOLD_FRAMES = 5

# --- Interface (ui.py) -------------------------------------------------------
WINDOW_NAME = "Hand Gesture Analyser"
HISTORY_REFRESH_SECONDS = 0.5
HISTORY_RECENT_ROWS = 5
# How long a moving or two-hand gesture stays on screen after it happens.
MOTION_DISPLAY_SECONDS = 2.5
# How long the last control action (e.g. "NEXT SLIDE") stays on screen.
CONTROL_DISPLAY_SECONDS = 1.5

# The 21 MediaPipe hand landmarks joined into a skeleton (landmark index pairs).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (9, 10), (10, 11), (11, 12),             # middle
    (13, 14), (14, 15), (15, 16),            # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
)

# --- Shape recognition in draw mode (shape_recognizer.py) ----------------------
SHAPE_MIN_SIZE = 80.0                # pixels of stroke; shorter strokes are ignored
SHAPE_CLOSED_RATIO = 0.2             # closed when the ends are this close (fraction of stroke length)
SHAPE_LINE_MAX_BEND = 0.12           # open stroke straying at most this (fraction of length) = LINE
SHAPE_MAX_PATH_RATIO = 1.4           # stroke longer than this x its outline = scribble
SHAPE_SMOOTHING = 3                  # points averaged when measuring stroke length
SHAPE_CIRCLE_ROUNDNESS = 0.92        # circularity for CIRCLE (1 = perfect; hand-drawn square ~0.85)
SHAPE_BOX_FILL = 0.82                # outline fills this much of its smallest box = SQUARE/RECTANGLE (circle 0.79)
SHAPE_TRIANGLE_FILL = 0.82           # outline fills this much of its smallest triangle = TRIANGLE (square ~0.6)
SHAPE_SQUARE_RATIO = 1.3             # width/height within this (either way) = SQUARE
SHAPE_COLOUR = (80, 220, 80)         # BGR: green outline of the recognised shape

# --- Sign words (sign_words.py, mode 6) -------------------------------------------
# Positions are in frame widths, hand measurements in palm lengths.
SIGN_WINDOW_SECONDS = 1.6        # how much of each hand's recent movement is looked at
SIGN_COOLDOWN_SECONDS = 1.0      # after a word, wait this long before the next
SIGN_REPEAT_SECONDS = 3.0        # the same word again only after this long (not while held)
SIGN_SHAPE_SHARE = 0.6           # a sign's hand shape must show in this share of the frames
SIGN_HOLD_SECONDS = 0.6          # still signs (ME, YOU) are held this long
SIGN_STILL = 0.04                # a hand moving less than this is still
SIGN_SMOOTHING = 3               # frames averaged before counting movements (hides jitter)
SIGN_MAX_STRAY = 0.25            # a straight movement strays at most this much (fraction of its length)
SIGN_FLAT_O_TIPS = 0.5           # fingertips this close to the thumb tip = flattened O
SIGN_FLAT_O_MIN_ANGLE = 110.0    # ...with fingers only partly bent (a fist is tighter)
SIGN_FLAP = 0.3                  # GOODBYE: fingertips move this much towards the wrist and back
SIGN_FLAP_SWINGS = 3
SIGN_SNAP = 0.35                 # NO: index and middle tips close on the thumb and open again
SIGN_SNAP_SWINGS = 3
SIGN_HOOK = 0.25                 # WHO: the index tip bends in and out
SIGN_HOOK_SWINGS = 3
SIGN_WIGGLE = 0.2                # WAIT: fingertips moving in and out of the palm's plane
SIGN_WIGGLE_SWINGS = 6           # ...swings counted over the four fingers of each hand
SIGN_SHAKE = 0.03                # WHERE, SAME, WHAT: side to side
SIGN_SHAKE_SWINGS = 3
SIGN_NOD = 0.25                  # YES: the fist tilts forward and back (change in tilt)
SIGN_NOD_SWINGS = 2
SIGN_TAP = 0.015                 # WATER, DANGER: small up and down taps or slides
SIGN_TAP_SWINGS = 3
SIGN_CIRCLE_MIN_RADIUS = 0.02    # PLEASE, SORRY: rubbing in a circle
SIGN_CIRCLE_MIN_TURN = 300.0
SIGN_MOVE = 0.1                  # HELLO, THANK YOU, BAD, STOP, GOOD, HELP, GO: a clear movement
SIGN_MOVE_SECONDS = 1.0          # ...made within this time
SIGN_HIGH = 0.5                  # HELLO, HOME start in the top half of the picture (fraction of height)
SIGN_TOWARDS = 1.3               # HELLO: or the hand comes this much closer (grows on screen)
SIGN_FLIP_ANGLE = 100.0          # BAD: the palm turns over by at least this much (THANK YOU: under half)
SIGN_EDGE = 0.7                  # STOP: the chopping hand's palm faces sideways
SIGN_NEAR = 1.6                  # two hands this close (palm lengths) are touching or stacked
SIGN_POINT_DEPTH = 0.6           # ME / YOU: the index points this much away from / towards the camera
SIGN_TOGETHER = 0.8              # MORE: the hands come together and apart this much (palm lengths)
SIGN_MAX_WORDS = 8               # words kept in the sentence on screen

# --- Emergency alert (alarm.py) ----------------------------------------------------
ALERT_WORDS = ("DANGER", "HELP")   # signed words that raise the alert
ALERT_SECONDS = 6.0                # how long the red screen and siren last (x dismisses)
ALERT_FLASH_HZ = 2.0               # red screen flashes this many times a second
ALERT_TONES = (880, 1320)          # siren: two alternating tones (Hz)
ALERT_BEEP_MS = 250                # length of each tone
