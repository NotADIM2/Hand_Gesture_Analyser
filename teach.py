"""Teach the app a new hand pose live (extra mode: TEACH, key t).

Press t, type a name (e.g. PINKY PROMISE) and press Enter. After a 3-second countdown
the app records a few dozen examples of the pose, and from then on recognises it like
any built-in gesture: it appears on screen, is saved to the history and can be
stabilised and logged. Taught gestures are stored in data/taught_gestures.npz, so
they are remembered next time.

How it recognises them: each example is stored as the same feature vector the k-NN
model uses (ml_classifier.feature_vector). A hand that is very close to taught
examples gets their name; otherwise the normal classifier (rules or ML) decides.
The closeness limit is strict, so teaching a CLAW doesn't turn every fist into one.
"""
from collections import Counter
from pathlib import Path

import numpy as np

import config
from feature_extractor import HandFeatures
from gesture_analyzer import GESTURES
from ml_classifier import feature_vector
from motion_analyzer import MOTION_GESTURES, TWO_HAND_GESTURES

KEY_ENTER = (13, 10)
KEY_BACKSPACE = 8
KEY_ESC = 27
RESERVED_NAMES = set(GESTURES) | set(MOTION_GESTURES) | set(TWO_HAND_GESTURES) | {"UNKNOWN"}


class TaughtGestures:
    """The taught examples: storage and nearest-example matching."""

    def __init__(self, path: Path | None = config.TAUGHT_GESTURES_PATH,
                 max_distance: float = config.TAUGHT_MAX_DISTANCE, k: int = 3):
        self.path = Path(path) if path is not None else None
        self.max_distance = max_distance
        self.k = k
        self.vectors = np.empty((0, 19), np.float32)
        self.labels: list[str] = []
        if self.path is not None and self.path.is_file():
            data = np.load(self.path)
            self.vectors, self.labels = data["vectors"], [str(label) for label in data["labels"]]

    @property
    def names(self) -> list[str]:
        return sorted(set(self.labels))

    def add(self, name: str, vectors: list[np.ndarray]) -> None:
        self.vectors = np.vstack([self.vectors, np.asarray(vectors, np.float32)])
        self.labels += [name] * len(vectors)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.path, vectors=self.vectors, labels=np.array(self.labels))

    def forget(self) -> None:
        self.vectors = np.empty((0, 19), np.float32)
        self.labels = []
        if self.path is not None and self.path.is_file():
            self.path.unlink()

    def match(self, features: HandFeatures) -> str | None:
        """The taught name for this hand, or None if it isn't close to any taught example."""
        if not self.labels:
            return None
        distances = np.linalg.norm(self.vectors - feature_vector(features), axis=1)
        nearest = np.argsort(distances)[: self.k]
        close = [self.labels[i] for i in nearest if distances[i] <= self.max_distance]
        return Counter(close).most_common(1)[0][0] if close else None


class WithTaughtGestures:
    """A classifier that checks taught gestures first, then falls back to another classifier."""

    def __init__(self, base, taught: TaughtGestures):
        self.base = base
        self.taught = taught

    def classify(self, features: HandFeatures) -> str:
        return self.taught.match(features) or self.base.classify(features)


class Teacher:
    """The t-key flow: type a name -> countdown -> record examples -> learned."""

    def __init__(self, taught: TaughtGestures, examples: int = config.TEACH_EXAMPLES,
                 countdown: float = config.TEACH_COUNTDOWN_SECONDS):
        self.taught = taught
        self.examples = examples
        self.countdown = countdown
        self.state: str | None = None   # None, "naming", "countdown" or "recording"
        self.name = ""
        self._recorded: list[np.ndarray] = []
        self._started_at = 0.0
        self._frame = 0

    @property
    def naming(self) -> bool:
        return self.state == "naming"

    @property
    def active(self) -> bool:
        return self.state is not None

    def start(self) -> None:
        self.state, self.name = "naming", ""

    def handle_key(self, key: int, now: float) -> str | None:
        """While naming, every key goes here. Returns a message to flash, if any."""
        if key == KEY_ESC:
            self.state = None
            return "TEACH CANCELLED"
        if key == KEY_BACKSPACE:
            self.name = self.name[:-1]
        elif key in KEY_ENTER:
            name = self.name.strip()
            if not name:
                return None
            if name in RESERVED_NAMES:
                return "NAME TAKEN, PICK ANOTHER"
            self.name, self.state = name, "countdown"
            self._started_at, self._recorded, self._frame = now, [], 0
        elif 32 <= key < 127 and len(self.name) < config.TEACH_MAX_NAME:
            char = chr(key).upper()
            if char.isalnum() or (char == " " and self.name and not self.name.endswith(" ")):
                self.name += char
        return None

    def update(self, features: list[HandFeatures], now: float) -> str | None:
        """Call once per frame. Returns a message when teaching finishes."""
        if self.state == "countdown" and now - self._started_at >= self.countdown:
            self.state = "recording"
        if self.state != "recording" or len(features) != 1:
            return None  # record only with exactly one hand in view
        self._frame += 1
        if self._frame % 2:  # every other frame, for a little more variety
            self._recorded.append(feature_vector(features[0]))
        if len(self._recorded) >= self.examples:
            self.taught.add(self.name, self._recorded)
            self.state = None
            return f"LEARNED {self.name}"
        return None

    def status_line(self, now: float) -> str | None:
        if self.state == "naming":
            return f"TEACH: type a name, Enter = start, Esc = cancel: {self.name}_"
        if self.state == "countdown":
            left = max(1, int(np.ceil(self.countdown - (now - self._started_at))))
            return f"TEACH {self.name}: get ready, show the pose in {left}"
        if self.state == "recording":
            return f"TEACH {self.name}: recording {len(self._recorded)}/{self.examples}, move it a little"
        return None
