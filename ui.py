"""Display and user interaction (architecture step: RESULT / UI).

The app works in one mode at a time, chosen with the number keys, so the screen only
shows what that mode is about:

    1 POSES            hand poses (labels on the hands, gesture details in the panel)
    2 MOVING GESTURES  palm trails and a banner for swipes, wave, circle, clap, heart
    3 DRAW SHAPES      pinch to draw; drawn shapes are recognised (z undoes, wave clears)
    4 SIGN LETTERS     poses spell words
    5 EXERCISES        the hand exercise coach
    6 SIGN WORDS       basic sign-language words, made into sentences and spoken
    7 SIGN PRACTICE    the app asks for a sign, you sign it, it keeps score

Draws onto OpenCV frames: hand skeletons, the result panel, the mode's output, an
optional gesture history panel and an optional feature panel for tuning thresholds.
This module only draws what it is given; it never calls the detector or the
database itself.
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from feature_extractor import FINGER_NAMES, HandFeatures
from gesture_analyzer import STABLE, UNCERTAIN, GestureResult
from hand_detector import DetectedHand
from motion_analyzer import HeartMeasurements, MotionEvent

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (80, 220, 80)
YELLOW = (0, 215, 255)
ORANGE = (0, 140, 255)
RED = (60, 60, 230)
SKELETON = (255, 200, 0)
TRAIL = (255, 0, 255)  # magenta
JOINT = (0, 0, 255)
ALERT_RED = (0, 0, 255)

STATUS_COLOURS = {STABLE: GREEN, UNCERTAIN: YELLOW}

KEY_QUIT = (ord("q"), 27)  # q or Esc
KEY_HISTORY = ord("h")
KEY_FEATURES = ord("f")
KEY_CLEAR = ord("c")
KEY_CONTROL = ord("m")
KEY_DRAW = ord("d")
KEY_SIGN = ord("a")
KEY_SAVE = ord("s")
KEY_TEACH = ord("t")
KEY_FULLSCREEN = ord("w")
KEY_EXERCISE = ord("e")
KEY_DISMISS = ord("x")
KEY_SKIP = ord("n")
KEY_UNDO = ord("z")
KEY_BUTTONS = ord("b")
KEY_ESC = 27

POSES, MOTION, DRAW, SIGN, EXERCISE, WORDS, PRACTICE = (
    "poses", "motion", "draw", "sign", "exercise", "words", "practice")
MODES = (POSES, MOTION, DRAW, SIGN, EXERCISE, WORDS, PRACTICE)    # in key order: 1 to 7
MODE_TITLES = {POSES: "POSES", MOTION: "MOVING GESTURES", DRAW: "DRAW SHAPES", SIGN: "SIGN LETTERS",
               EXERCISE: "EXERCISES", WORDS: "SIGN WORDS", PRACTICE: "SIGN PRACTICE"}
# Number keys pick a mode; the old letter keys still work (d, a, e).
MODE_KEYS = {ord(str(i)): mode for i, mode in enumerate(MODES, start=1)}
MODE_KEYS.update({KEY_DRAW: DRAW, KEY_SIGN: SIGN, KEY_EXERCISE: EXERCISE})
SHOWS_POSES = (POSES, SIGN)   # modes that label hand poses

HELP_LINES = (
    "1: poses  2: moving gestures  3: draw shapes  4: sign letters  5: exercises  6: sign words  7: practice",
    "m: control  t: teach  h: history  f: features  b: buttons  c: clear history  s: save  w: full screen  q: quit",
)


def mode_title(mode: str) -> str:
    return f"MODE {MODES.index(mode) + 1}: {MODE_TITLES[mode]}"


@dataclass
class HistoryView:
    """What the history panel shows; filled in by app.py from the database."""
    session_counts: dict[str, int] = field(default_factory=dict)
    recent: list = field(default_factory=list)   # GestureRecord objects
    all_time_total: int = 0


class FPSCounter:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            instant = 1.0 / max(now - self._last, 1e-6)
            self.fps = instant if self.fps == 0 else self.smoothing * self.fps + (1 - self.smoothing) * instant
        self._last = now
        return self.fps


def draw_hand(frame: np.ndarray, hand: DetectedHand, label: str | None = None, colour=SKELETON) -> None:
    points = [tuple(int(v) for v in p[:2]) for p in hand.image_points]
    for start, end in config.HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], colour, 2, cv2.LINE_AA)
    for p in points:
        cv2.circle(frame, p, 4, JOINT, -1, cv2.LINE_AA)
    if label:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        _text_box(frame, label, (min(xs), max(min(ys) - 10, 20)), colour)


def draw_panel(frame: np.ndarray, lines: list[tuple[str, tuple]], origin: tuple[int, int],
               scale: float = 0.55, min_width: int = 0) -> tuple[int, int]:
    """Translucent box with one coloured line of text per entry. Returns its (width, height)."""
    if not lines:
        return 0, 0
    pad, gap = 10, 8
    sizes = [cv2.getTextSize(text, FONT, scale, 1)[0] for text, _ in lines]
    line_h = max(h for _, h in sizes) + gap
    width = max(min_width, max(w for w, _ in sizes) + 2 * pad)
    height = line_h * len(lines) + pad
    x, y = origin
    x = max(0, min(x, frame.shape[1] - width))
    y = max(0, min(y, frame.shape[0] - height))

    region = frame[y:y + height, x:x + width]
    region[:] = cv2.addWeighted(region, 0.35, np.zeros_like(region), 0.65, 0)
    for i, (text, colour) in enumerate(lines):
        baseline = y + pad + line_h * i + line_h - gap
        cv2.putText(frame, text, (x + pad, baseline), FONT, scale, colour, 1, cv2.LINE_AA)
    return width, height


def move_text(move: MotionEvent) -> str:
    return f"{move.gesture} ({move.hand.upper()} HAND{'S' if move.hand == 'Both' else ''})"


def draw_banner(frame: np.ndarray, text: str, colour=GREEN, scale: float = 1.1, lift: int = 65) -> None:
    """Large outlined text, centred `lift` pixels above the bottom of the frame."""
    (w, _), _ = cv2.getTextSize(text, FONT, scale, 2)
    origin = (max(0, (frame.shape[1] - w) // 2), frame.shape[0] - lift)
    cv2.putText(frame, text, origin, FONT, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, origin, FONT, scale, colour, 2, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, trail: np.ndarray) -> None:
    """The palm's recent path, with a dot where the palm is now."""
    points = trail.astype(np.int32)
    cv2.polylines(frame, [points], False, TRAIL, 4, cv2.LINE_AA)
    cv2.circle(frame, tuple(int(v) for v in points[-1]), 7, TRAIL, -1, cv2.LINE_AA)


def finger_count_text(features: list[HandFeatures]) -> str:
    """Fingers up across all hands, 0-10, e.g. "Fingers up: 7 (LEFT 5 + RIGHT 2)"."""
    total = sum(f.extended_count for f in features)
    if len(features) < 2:
        return f"Fingers up: {total}"
    parts = " + ".join(f"{f.handedness.upper()} {f.extended_count}" for f in features)
    return f"Fingers up: {total} ({parts})"


def result_lines(hand_count: int, results: list[GestureResult],
                 last_move: MotionEvent | None = None,
                 features: list[HandFeatures] = (), mode: str = POSES) -> list[tuple[str, tuple]]:
    lines = [(mode_title(mode), YELLOW), (f"Hands detected: {hand_count}", WHITE)]
    if features and mode == POSES:
        lines.append((finger_count_text(features), WHITE))
    if last_move is not None and mode == MOTION:
        lines.append((f"Last move: {move_text(last_move)}", TRAIL))
    if not results:
        lines.append(("Status: NO HAND DETECTED", RED))
        return lines
    if mode not in SHOWS_POSES:
        return lines
    for r in results:
        colour = STATUS_COLOURS.get(r.status, ORANGE)
        lines += [
            ("", WHITE),
            (f"Detected Gesture: {r.gesture}", colour),
            (f"Hand: {r.hand.upper()}", WHITE),
            ("Status: DETECTED", WHITE),
            (f"Recognition: {r.status} ({r.agreement}/{r.window} frames agree)", colour),
        ]
    return lines


def history_lines(history: HistoryView) -> list[tuple[str, tuple]]:
    lines = [("GESTURE HISTORY - this session", WHITE)]
    if history.session_counts:
        lines += [(f"{name}: {count}", GREEN) for name, count in history.session_counts.items()]
    else:
        lines.append(("(nothing recorded yet)", GREY))
    lines.append(("Recent:", WHITE))
    for rec in history.recent:
        clock = rec.timestamp[11:19] if len(rec.timestamp) >= 19 else rec.timestamp
        lines.append((f"{clock} {(rec.hand or '-').upper():5} {rec.gesture}", GREY))
    lines.append((f"All sessions: {history.all_time_total} records", WHITE))
    return lines


def heart_lines(m: HeartMeasurements) -> list[tuple[str, tuple]]:
    def mark(ok: bool) -> str:
        return "ok" if ok else "NO"
    index_ok = m.index_gap < config.HEART_TOUCH_DISTANCE
    thumb_ok = m.thumb_gap < config.HEART_TOUCH_DISTANCE
    wrist_ok = m.wrist_gap > config.HEART_MIN_WRIST_GAP
    return [(
        f"HEART: index tips {m.index_gap:.2f} {mark(index_ok)}  thumb tips {m.thumb_gap:.2f} {mark(thumb_ok)}  "
        f"wrists {m.wrist_gap:.2f} {mark(wrist_ok)}  index above {mark(m.index_above_thumbs)}",
        GREEN if m.is_heart else ORANGE,
    )]


def feature_lines(features: list[HandFeatures], heart: HeartMeasurements | None = None) -> list[tuple[str, tuple]]:
    lines = [("FEATURES  (T I M R P = extended fingers)", WHITE)]
    for f in features:
        fingers = " ".join("1" if f.extended[n] else "0" for n in FINGER_NAMES)
        angles = " ".join(f"{f.finger_angles[n]:.0f}" for n in FINGER_NAMES)
        lines += [
            (f"{f.handedness.upper()}: {fingers}   angles {angles}", YELLOW),
            (f"  spread {f.index_middle_spread:.0f}  thumb-index {f.thumb_index_distance:.2f}  "
             f"thumb-out {f.thumb_lateral:.2f}  thumb {f.thumb_direction or '-'}", YELLOW),
        ]
    if heart is not None:
        lines += heart_lines(heart)
    return lines


def draw_alert(frame: np.ndarray, word: str, flash_on: bool, when: str = "") -> None:
    """Emergency alert over the whole picture: a flashing red tint, a red border and the word."""
    h, w = frame.shape[:2]
    if flash_on:
        red = np.zeros_like(frame)
        red[:] = ALERT_RED
        frame[:] = cv2.addWeighted(frame, 0.4, red, 0.6, 0)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), ALERT_RED, 18)
    # Above the middle, so it stays clear of the sentence and action banners at the bottom.
    for text, scale, y in ((f"ALERT: {word}", 1.8, h // 2 - 45),
                           (f"signed at {when}" if when else "", 0.8, h // 2 - 2),
                           ("press x to dismiss", 0.7, h // 2 + 30)):
        if not text:
            continue
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, 3)
        origin = (max(0, (w - tw) // 2), y)
        cv2.putText(frame, text, origin, FONT, scale, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(frame, text, origin, FONT, scale, WHITE, 3, cv2.LINE_AA)


def fit_to_window(frame: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """Scale the frame to fill a (width, height) window without stretching it; black bars fill the rest."""
    if not size:
        return frame
    width, height = size
    h, w = frame.shape[:2]
    if (w, h) == (width, height) or width <= 0 or height <= 0:
        return frame
    scale = min(width / w, height / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), frame.dtype)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


class Interface:
    def __init__(self, window_name: str = config.WINDOW_NAME, fullscreen: bool = False):
        self.window_name = window_name
        self.show_history = False
        self.show_features = False
        self.show_buttons = True      # virtual buttons (buttons.py); b hides them
        self.fullscreen = False
        # A normal (resizable) window. Frames are still processed at the camera size;
        # show() scales the finished picture to whatever size the window is.
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        if fullscreen:
            self.set_fullscreen(True)

    def set_fullscreen(self, on: bool) -> None:
        mode = cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, mode)
        self.fullscreen = on

    def window_size(self) -> tuple[int, int] | None:
        try:
            _, _, width, height = cv2.getWindowImageRect(self.window_name)
        except cv2.error:
            return None
        return (width, height) if width > 0 and height > 0 else None

    def render(
        self,
        frame: np.ndarray,
        hands: list[DetectedHand],
        results: list[GestureResult],
        features: list[HandFeatures],
        fps: float,
        history: HistoryView | None = None,
        moves: list[MotionEvent] = (),
        trails: list[np.ndarray] = (),
        last_move: MotionEvent | None = None,
        heart: HeartMeasurements | None = None,
        status: list[str] = (),
        action: str | None = None,
        sign_text: str | None = None,
        mode: str = POSES,
        prompt: str | None = None,
        right_edge: int | None = None,
    ) -> np.ndarray:
        """right_edge: where the right-hand panels end (left of the virtual buttons)."""
        right_edge = frame.shape[1] if right_edge is None else right_edge
        # Only the current mode's output is drawn: pose labels in the pose modes,
        # trails and the move banner in the moving gestures mode.
        for hand, result in zip(hands, results):
            if mode in SHOWS_POSES:
                colour = STATUS_COLOURS.get(result.status, ORANGE)
                draw_hand(frame, hand, f"{result.hand.upper()}: {result.gesture}", colour)
            else:
                draw_hand(frame, hand)
        if mode == MOTION:
            for trail in trails:
                draw_trail(frame, trail)

        lines = result_lines(len(hands), results, last_move, features, mode)
        _, result_height = draw_panel(frame, lines, (10, 10), min_width=260)
        draw_panel(frame, [(f"FPS: {fps:.0f}", GREEN)], (frame.shape[1], 10))
        draw_panel(frame, [(line, ORANGE) for line in status], (10, 10 + result_height + 6), scale=0.5)
        if moves and mode == MOTION:
            draw_banner(frame, move_text(moves[-1]))
        if action:
            draw_banner(frame, f"> {action}", ORANGE, lift=110)
        if sign_text is not None:
            draw_banner(frame, f"{sign_text}_", WHITE, scale=1.3, lift=155)
        if prompt:
            draw_banner(frame, prompt, YELLOW, scale=1.3, lift=155)
        if self.show_history and history is not None:
            # Drawn on a view of the frame left of the buttons, so it right-aligns against them.
            draw_panel(frame[:, :right_edge], history_lines(history), (right_edge, 50), scale=0.5)
        if self.show_features and features:
            draw_panel(frame, feature_lines(features, heart), (10, frame.shape[0] - 200), scale=0.45)
        draw_panel(frame, [(line, GREY) for line in HELP_LINES], (10, frame.shape[0]), scale=0.45)
        return frame

    def show(self, frame: np.ndarray) -> int:
        """Display the frame (scaled to the window) and return the key pressed (-1 if none)."""
        cv2.imshow(self.window_name, fit_to_window(frame, self.window_size()))
        return self.poll()

    def poll(self) -> int:
        """Process window events without a new frame; returns the key pressed (-1 if none)."""
        key = cv2.waitKey(1)
        return -1 if key == -1 else key & 0xFF

    def handle_key(self, key: int) -> None:
        if key == KEY_FULLSCREEN:
            self.set_fullscreen(not self.fullscreen)
        elif key == KEY_HISTORY:
            self.show_history = not self.show_history
        elif key == KEY_FEATURES:
            self.show_features = not self.show_features
        elif key == KEY_BUTTONS:
            self.show_buttons = not self.show_buttons

    @property
    def closed(self) -> bool:
        """True once the user closes the window with its X button."""
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def close(self) -> None:
        cv2.destroyAllWindows()


def _text_box(frame: np.ndarray, text: str, origin: tuple[int, int], colour) -> None:
    (w, h), base = cv2.getTextSize(text, FONT, 0.55, 1)
    x, y = origin
    x = max(0, min(x, frame.shape[1] - w - 8))
    cv2.rectangle(frame, (x, y - h - base - 4), (x + w + 8, y + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 4, y - base + 1), FONT, 0.55, colour, 1, cv2.LINE_AA)
