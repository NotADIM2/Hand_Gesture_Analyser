"""Display and user interaction (architecture step: RESULT / UI).

Draws onto OpenCV frames: hand skeletons, palm trails, the result panel, a banner for
moving and two-hand gestures, an optional gesture history panel and an optional
feature panel for tuning thresholds. This module only
draws what it is given; it never calls the detector or the database itself.
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

STATUS_COLOURS = {STABLE: GREEN, UNCERTAIN: YELLOW}

KEY_QUIT = (ord("q"), 27)  # q or Esc
KEY_HISTORY = ord("h")
KEY_FEATURES = ord("f")
KEY_CLEAR = ord("c")


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


def draw_banner(frame: np.ndarray, text: str, colour=GREEN, scale: float = 1.1) -> None:
    """Large outlined text, centred just above the help line."""
    (w, _), _ = cv2.getTextSize(text, FONT, scale, 2)
    origin = (max(0, (frame.shape[1] - w) // 2), frame.shape[0] - 45)
    cv2.putText(frame, text, origin, FONT, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, origin, FONT, scale, colour, 2, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, trail: np.ndarray) -> None:
    """The palm's recent path, with a dot where the palm is now."""
    points = trail.astype(np.int32)
    cv2.polylines(frame, [points], False, TRAIL, 4, cv2.LINE_AA)
    cv2.circle(frame, tuple(int(v) for v in points[-1]), 7, TRAIL, -1, cv2.LINE_AA)


def result_lines(hand_count: int, results: list[GestureResult],
                 last_move: MotionEvent | None = None) -> list[tuple[str, tuple]]:
    lines = [(f"Hands detected: {hand_count}", WHITE)]
    if last_move is not None:
        lines.append((f"Last move: {move_text(last_move)}", TRAIL))
    if not results:
        lines.append(("Status: NO HAND DETECTED", RED))
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


class Interface:
    def __init__(self, window_name: str = config.WINDOW_NAME):
        self.window_name = window_name
        self.show_history = False
        self.show_features = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

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
    ) -> np.ndarray:
        for hand, result in zip(hands, results):
            colour = STATUS_COLOURS.get(result.status, ORANGE)
            draw_hand(frame, hand, f"{result.hand.upper()}: {result.gesture}", colour)
        for trail in trails:
            draw_trail(frame, trail)

        draw_panel(frame, result_lines(len(hands), results, last_move), (10, 10), min_width=260)
        draw_panel(frame, [(f"FPS: {fps:.0f}", GREEN)], (frame.shape[1], 10))
        if moves:
            draw_banner(frame, move_text(moves[-1]))
        if self.show_history and history is not None:
            draw_panel(frame, history_lines(history), (frame.shape[1], 50), scale=0.5)
        if self.show_features and features:
            draw_panel(frame, feature_lines(features, heart), (10, frame.shape[0] - 180), scale=0.45)
        help_text = "q: quit   h: history   f: features   c: clear history"
        draw_panel(frame, [(help_text, GREY)], (10, frame.shape[0]), scale=0.45)
        return frame

    def show(self, frame: np.ndarray) -> int:
        """Display the frame and return the key pressed (-1 if none)."""
        cv2.imshow(self.window_name, frame)
        return self.poll()

    def poll(self) -> int:
        """Process window events without a new frame; returns the key pressed (-1 if none)."""
        key = cv2.waitKey(1)
        return -1 if key == -1 else key & 0xFF

    def handle_key(self, key: int) -> None:
        if key == KEY_HISTORY:
            self.show_history = not self.show_history
        elif key == KEY_FEATURES:
            self.show_features = not self.show_features

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
