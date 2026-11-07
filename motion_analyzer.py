"""Recognise moving and two-hand gestures (architecture step: MOTION ANALYZER).

The gesture analyzer looks at one hand in one frame. This step looks at how hands
move over the last second or two, and at how two hands relate to each other:

- swipes, wave and circle follow one hand's palm centre over time;
- clap and heart look at both hands together.

One-hand positions are measured in frame widths (x and y both divided by the frame
width), so thresholds don't depend on the camera resolution. Two-hand distances are
measured in palm sizes, so they don't depend on how far the hands are from the camera.

Each gesture is reported once, as an event, and the hand is then ignored for a short
cooldown so one movement isn't counted twice.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config
from gesture_analyzer import track_keys
from hand_detector import DetectedHand

SWIPE_LEFT = "SWIPE LEFT"
SWIPE_RIGHT = "SWIPE RIGHT"
SWIPE_UP = "SWIPE UP"
SWIPE_DOWN = "SWIPE DOWN"
WAVE = "WAVE"
CIRCLE = "CIRCLE"
CLAP = "CLAP"
HEART = "HEART"

MOTION_GESTURES: tuple[str, ...] = (SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN, WAVE, CIRCLE)
TWO_HAND_GESTURES: tuple[str, ...] = (CLAP, HEART)
BOTH_HANDS = "Both"

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17

Point = tuple[float, float, float]  # (time in seconds, x, y)


@dataclass(frozen=True)
class MotionEvent:
    gesture: str
    hand: str     # "Left", "Right" or "Both"
    time: float   # when it was recognised (same clock as passed to update)


@dataclass
class _Track:
    points: deque = field(default_factory=deque)
    cooldown_until: float = 0.0
    frames_missing: int = 0


def palm_centre(hand: DetectedHand) -> np.ndarray:
    """Centre of the palm in pixels: the mean of the wrist and the index and pinky knuckles."""
    return hand.image_points[[WRIST, INDEX_MCP, PINKY_MCP], :2].mean(axis=0)


def palm_size(hand: DetectedHand) -> float:
    """Wrist to middle knuckle on screen, in pixels."""
    return float(np.linalg.norm(hand.image_points[MIDDLE_MCP, :2] - hand.image_points[WRIST, :2])) or 1.0


def _recent(points: list[Point], seconds: float) -> np.ndarray:
    """The points from the last `seconds`, as an (n, 3) array of time, x, y."""
    if not points:
        return np.empty((0, 3))
    arr = np.asarray(points, dtype=np.float64)
    return arr[arr[:, 0] >= arr[-1, 0] - seconds]


def detect_swipe(points: list[Point]) -> str | None:
    """A fast, fairly straight movement over a long distance in one main direction."""
    recent = _recent(points, config.SWIPE_MAX_SECONDS)
    if len(recent) < 3:
        return None
    xy = recent[:, 1:]
    dx, dy = xy[-1] - xy[0]
    distance = float(np.hypot(dx, dy))
    if distance < config.SWIPE_MIN_DISTANCE:
        return None
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if path > config.SWIPE_MAX_PATH_RATIO * distance:
        return None  # wobbly or back-and-forth, not a swipe
    if abs(dx) >= config.SWIPE_MIN_DOMINANCE * abs(dy):
        return SWIPE_RIGHT if dx > 0 else SWIPE_LEFT
    if abs(dy) >= config.SWIPE_MIN_DOMINANCE * abs(dx):
        return SWIPE_DOWN if dy > 0 else SWIPE_UP  # image y grows downwards
    return None  # diagonal


def count_swings(xs: np.ndarray, amplitude: float) -> int:
    """Number of left/right movements in xs, ignoring jitter smaller than `amplitude`."""
    if len(xs) == 0:
        return 0
    swings = 0
    direction = 0
    low = high = extreme = xs[0]
    for x in xs[1:]:
        if direction == 0:
            low, high = min(low, x), max(high, x)
            if high - low >= amplitude:
                direction = 1 if x == high else -1
                extreme = x
                swings = 1
        elif direction > 0:
            if x > extreme:
                extreme = x
            elif extreme - x >= amplitude:
                direction, extreme = -1, x
                swings += 1
        else:
            if x < extreme:
                extreme = x
            elif x - extreme >= amplitude:
                direction, extreme = 1, x
                swings += 1
    return swings


def detect_wave(points: list[Point]) -> str | None:
    """The hand moving left and right several times in quick succession."""
    recent = _recent(points, config.WAVE_SECONDS)
    if count_swings(recent[:, 1], config.WAVE_MIN_AMPLITUDE) >= config.WAVE_MIN_SWINGS:
        return WAVE
    return None


def detect_circle(points: list[Point]) -> str | None:
    """The palm going (almost) all the way round a roughly round path."""
    recent = _recent(points, config.CIRCLE_SECONDS)
    if len(recent) < 8:
        return None
    xy = recent[:, 1:]
    offsets = xy - xy.mean(axis=0)
    radii = np.linalg.norm(offsets, axis=1)
    mean_radius = float(radii.mean())
    if mean_radius < config.CIRCLE_MIN_RADIUS:
        return None
    # A back-and-forth line passes through its own centre; a circle never comes close.
    if radii.min() < config.CIRCLE_MIN_ROUNDNESS * mean_radius:
        return None
    angles = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
    if abs(np.degrees(angles[-1] - angles[0])) >= config.CIRCLE_MIN_TURN:
        return CIRCLE
    return None


# Checked in order, first match wins. A wave or circle covers a lot of ground, so
# they are checked before swipe; a swipe's straightness check also rules them out.
ONE_HAND_DETECTORS = (detect_wave, detect_circle, detect_swipe)


def is_heart(a: DetectedHand, b: DetectedHand) -> bool:
    """Index tips touching at the top, thumb tips touching below, wrists apart."""
    palm = (palm_size(a) + palm_size(b)) / 2
    pa, pb = a.image_points[:, :2], b.image_points[:, :2]
    index_gap = np.linalg.norm(pa[INDEX_TIP] - pb[INDEX_TIP]) / palm
    thumb_gap = np.linalg.norm(pa[THUMB_TIP] - pb[THUMB_TIP]) / palm
    wrist_gap = np.linalg.norm(pa[WRIST] - pb[WRIST]) / palm
    index_above_thumbs = pa[INDEX_TIP, 1] + pb[INDEX_TIP, 1] < pa[THUMB_TIP, 1] + pb[THUMB_TIP, 1]
    return bool(
        index_gap < config.HEART_TOUCH_DISTANCE
        and thumb_gap < config.HEART_TOUCH_DISTANCE
        and wrist_gap > config.HEART_MIN_WRIST_GAP
        and index_above_thumbs
    )


class MotionAnalyzer:
    def __init__(self, grace_frames: int = config.HAND_LOST_GRACE_FRAMES):
        self.grace_frames = grace_frames
        self._tracks: dict[str, _Track] = {}
        self._pair_distances: deque = deque()   # (time, palm-centre distance in palm sizes)
        self._pair_cooldown_until = 0.0
        self._heart_frames = 0
        self._heart_reported = False

    def update(self, hands: list[DetectedHand], frame_width: int, now: float) -> list[MotionEvent]:
        """Add this frame's hands; returns the gestures completed on this frame. Call once per frame."""
        keys = track_keys(hands)
        events = []
        for key, hand in zip(keys, hands):
            gesture = self._update_track(key, palm_centre(hand) / frame_width, now)
            if gesture:
                events.append(MotionEvent(gesture, hand.handedness, now))
        for key in set(self._tracks) - set(keys):
            track = self._tracks[key]
            track.frames_missing += 1
            if track.frames_missing > self.grace_frames:
                del self._tracks[key]
        events += self._update_pair(hands, now)
        return events

    def trails(self, frame_width: int) -> list[np.ndarray]:
        """Recent palm-centre paths in pixels, one (n, 2) array per tracked hand, for drawing."""
        return [
            np.asarray([(x, y) for _, x, y in track.points]) * frame_width
            for track in self._tracks.values()
            if len(track.points) >= 2
        ]

    def reset(self) -> None:
        self._tracks.clear()
        self._pair_distances.clear()
        self._heart_frames = 0
        self._heart_reported = False

    def _update_track(self, key: str, position: np.ndarray, now: float) -> str | None:
        track = self._tracks.setdefault(key, _Track())
        track.frames_missing = 0
        x, y = float(position[0]), float(position[1])
        if track.points:
            _, last_x, last_y = track.points[-1]
            if np.hypot(x - last_x, y - last_y) > config.MOTION_MAX_JUMP:
                # Too far for one frame: a tracking glitch (e.g. the two hands' labels
                # swapped), not real movement. Start this hand's path again.
                track.points.clear()
        track.points.append((now, x, y))
        while track.points and track.points[0][0] < now - config.MOTION_HISTORY_SECONDS:
            track.points.popleft()

        if now < track.cooldown_until:
            return None
        points = list(track.points)
        for detect in ONE_HAND_DETECTORS:
            gesture = detect(points)
            if gesture:
                track.cooldown_until = now + config.MOTION_COOLDOWN_SECONDS
                track.points.clear()
                return gesture
        return None

    def _update_pair(self, hands: list[DetectedHand], now: float) -> list[MotionEvent]:
        if len(hands) != 2:
            self._heart_frames = 0
            self._heart_reported = False
            return []
        a, b = hands
        events = []

        # Heart: a held pose, reported once after a few frames and again only after
        # the hands have moved out of it.
        if is_heart(a, b):
            self._heart_frames += 1
            if self._heart_frames >= config.HEART_HOLD_FRAMES and not self._heart_reported:
                self._heart_reported = True
                events.append(MotionEvent(HEART, BOTH_HANDS, now))
        else:
            self._heart_frames = 0
            self._heart_reported = False

        # Clap: the palms were well apart a moment ago and are now together.
        palm = (palm_size(a) + palm_size(b)) / 2
        distance = float(np.linalg.norm(palm_centre(a) - palm_centre(b))) / palm
        self._pair_distances.append((now, distance))
        while self._pair_distances and self._pair_distances[0][0] < now - config.CLAP_SECONDS:
            self._pair_distances.popleft()
        if (
            now >= self._pair_cooldown_until
            and distance <= config.CLAP_TOUCH_DISTANCE
            and max(d for _, d in self._pair_distances) >= config.CLAP_START_DISTANCE
        ):
            self._pair_cooldown_until = now + config.MOTION_COOLDOWN_SECONDS
            self._pair_distances.clear()
            events.append(MotionEvent(CLAP, BOTH_HANDS, now))
        return events
