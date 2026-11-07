"""Teach the app a new hand pose live (extra mode: TEACH, key t).

Press t, type a name (e.g. PINKY PROMISE) and press Enter. After a 3-second countdown
the app records a few dozen examples of the pose, and from then on recognises it like
any built-in gesture: it appears on screen, is saved to the history and can be
stabilised and logged.

Each taught pose is saved as its own file in taught_poses/ (e.g.
taught_poses/pinky_promise.json) with its name, who taught it, when, and the raw hand
landmarks of every example. The app loads every file in that folder at start, so a
pose committed to the repository is recognised by everyone who pulls it. One file per
pose means two people teaching different poses never get a merge conflict, and
storing landmarks (not processed numbers) keeps old poses working if the
measurements change.

How it recognises them: each example is turned into the same feature vector the k-NN
model uses (ml_classifier.feature_vector). A hand that is very close to taught
examples gets their name; otherwise the normal classifier (rules or ML) decides.
The closeness limit is strict, so taught poses don't take over the built-in ones.
"""
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from feature_extractor import HandFeatures, extract_features
from gesture_analyzer import GESTURES
from hand_detector import DetectedHand
from ml_classifier import feature_vector
from motion_analyzer import MOTION_GESTURES, TWO_HAND_GESTURES

KEY_ENTER = (13, 10)
KEY_BACKSPACE = 8
KEY_ESC = 27
RESERVED_NAMES = set(GESTURES) | set(MOTION_GESTURES) | set(TWO_HAND_GESTURES) | {"UNKNOWN"}
VECTOR_SIZE = 19


def pose_filename(name: str) -> str:
    """PINKY PROMISE -> pinky_promise.json"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".json"


def example_from_hand(hand: DetectedHand) -> dict:
    return {
        "handedness": hand.handedness,
        "world_landmarks": hand.world_landmarks.round(5).tolist(),
        "image_points": hand.image_points.round(1).tolist(),
    }


def example_vector(example: dict) -> np.ndarray:
    f = extract_features(np.array(example["world_landmarks"]), np.array(example["image_points"]),
                         example.get("handedness", "Right"))
    return feature_vector(f)


class TaughtGestures:
    """All taught poses: loading and saving pose files, and nearest-example matching."""

    def __init__(self, folder: Path | None = config.TAUGHT_POSES_DIR,
                 max_distance: float = config.TAUGHT_MAX_DISTANCE, k: int = 3):
        self.folder = Path(folder) if folder is not None else None
        self.max_distance = max_distance
        self.k = k
        self.vectors = np.empty((0, VECTOR_SIZE), np.float32)
        self.labels: list[str] = []
        self.taught_by: dict[str, list[str]] = {}
        if self.folder is not None and self.folder.is_dir():
            for path in sorted(self.folder.glob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                self._remember(data["name"], data["examples"], data.get("taught_by", []))

    @property
    def names(self) -> list[str]:
        return sorted(set(self.labels))

    def add(self, name: str, examples: list[dict], author: str = "unknown") -> Path | None:
        """Learn a pose from examples (see example_from_hand); returns the file it was saved to.

        Teaching a name that already exists adds the new examples to it.
        """
        self._remember(name, examples, [author])
        if self.folder is None:
            return None
        path = self.folder / pose_filename(name)
        data = (json.loads(path.read_text(encoding="utf-8")) if path.is_file()
                else {"name": name, "taught_by": [], "created": datetime.now().isoformat(timespec="seconds"),
                      "examples": []})
        if author not in data["taught_by"]:
            data["taught_by"].append(author)
        data["examples"] += examples
        self.folder.mkdir(parents=True, exist_ok=True)
        # One example per line keeps the file readable and diffs small.
        lines = [
            "{",
            f' "name": {json.dumps(data["name"])},',
            f' "taught_by": {json.dumps(data["taught_by"])},',
            f' "created": {json.dumps(data["created"])},',
            ' "examples": [',
            ",\n".join("  " + json.dumps(e, separators=(",", ":")) for e in data["examples"]),
            " ]",
            "}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def match(self, features: HandFeatures) -> str | None:
        """The taught name for this hand, or None if it isn't close to any taught example."""
        if not self.labels:
            return None
        distances = np.linalg.norm(self.vectors - feature_vector(features), axis=1)
        nearest = np.argsort(distances)[: self.k]
        close = [self.labels[i] for i in nearest if distances[i] <= self.max_distance]
        return Counter(close).most_common(1)[0][0] if close else None

    def _remember(self, name: str, examples: list[dict], authors: list[str]) -> None:
        vectors = [example_vector(e) for e in examples]
        if vectors:
            self.vectors = np.vstack([self.vectors, np.asarray(vectors, np.float32)])
            self.labels += [name] * len(vectors)
        known = self.taught_by.setdefault(name, [])
        known += [a for a in authors if a not in known]


class WithTaughtGestures:
    """A classifier that checks taught gestures first, then falls back to another classifier."""

    def __init__(self, base, taught: TaughtGestures):
        self.base = base
        self.taught = taught

    def classify(self, features: HandFeatures) -> str:
        return self.taught.match(features) or self.base.classify(features)


class Teacher:
    """The t-key flow: type a name -> countdown -> record examples -> learned and saved."""

    def __init__(self, taught: TaughtGestures, author: str = "unknown", examples: int = config.TEACH_EXAMPLES,
                 countdown: float = config.TEACH_COUNTDOWN_SECONDS):
        self.taught = taught
        self.author = author
        self.examples = examples
        self.countdown = countdown
        self.state: str | None = None   # None, "naming", "countdown" or "recording"
        self.name = ""
        self.saved_to: Path | None = None
        self._recorded: list[dict] = []
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

    def update(self, hands: list[DetectedHand], now: float) -> str | None:
        """Call once per frame. Returns a message when teaching finishes."""
        if self.state == "countdown" and now - self._started_at >= self.countdown:
            self.state = "recording"
        if self.state != "recording" or len(hands) != 1:
            return None  # record only with exactly one hand in view
        self._frame += 1
        if self._frame % 2:  # every other frame, for a little more variety
            self._recorded.append(example_from_hand(hands[0]))
        if len(self._recorded) >= self.examples:
            self.saved_to = self.taught.add(self.name, self._recorded, self.author)
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
