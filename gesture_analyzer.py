"""Classify hand features into gestures (architecture step: GESTURE ANALYZER).

Two layers:
1. A classifier labels one hand in one frame. The default is rule-based: each gesture
   is a small GestureRule, so adding a gesture means adding a rule. Anything with a
   `classify(features) -> str` method (e.g. a trained ML model) can replace it.
2. GestureAnalyzer smooths those per-frame labels over the last few frames and
   reports how many agreed. That agreement count is what the UI shows instead of a
   made-up confidence percentage: rules don't produce a probability.
"""
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import config
from feature_extractor import HandFeatures

OPEN_PALM = "OPEN PALM"
FIST = "FIST"
THUMBS_UP = "THUMBS UP"
THUMBS_DOWN = "THUMBS DOWN"
ONE_FINGER = "ONE FINGER"
TWO_FINGERS = "TWO FINGERS"
THREE_FINGERS = "THREE FINGERS"
PEACE = "PEACE"
OK_SIGN = "OK SIGN"
UNKNOWN = "UNKNOWN"

# Recognition status for one hand.
STABLE = "STABLE"              # the same gesture won most of the recent frames
UNCERTAIN = "UNCERTAIN"        # recent frames disagree; the gesture shown is the best guess
UNRECOGNISED = "UNRECOGNISED"  # the hand is visible but matches no known gesture


@dataclass(frozen=True)
class GestureRule:
    name: str
    matches: Callable[[HandFeatures], bool]


def only(f: HandFeatures, *fingers: str) -> bool:
    """True when exactly these fingers are extended and all others are folded."""
    return f.extended_fingers == frozenset(fingers)


# Checked in order, first match wins, so more specific gestures come first
# (OK before the finger counts, Peace before Two Fingers).
DEFAULT_RULES: tuple[GestureRule, ...] = (
    GestureRule(
        OK_SIGN,
        lambda f: f.thumb_index_distance < config.OK_TOUCH_RATIO
        and all(f.extended[n] for n in ("middle", "ring", "pinky")),
    ),
    GestureRule(THUMBS_UP, lambda f: only(f, "thumb") and f.thumb_direction == "up"),
    GestureRule(THUMBS_DOWN, lambda f: only(f, "thumb") and f.thumb_direction == "down"),
    GestureRule(OPEN_PALM, lambda f: only(f, "thumb", "index", "middle", "ring", "pinky")),
    GestureRule(FIST, lambda f: only(f)),
    GestureRule(ONE_FINGER, lambda f: only(f, "index")),
    GestureRule(
        PEACE,
        lambda f: only(f, "index", "middle") and f.index_middle_spread >= config.PEACE_MIN_SPREAD,
    ),
    GestureRule(TWO_FINGERS, lambda f: only(f, "index", "middle")),
    GestureRule(THREE_FINGERS, lambda f: only(f, "index", "middle", "ring")),
)

GESTURES: tuple[str, ...] = tuple(rule.name for rule in DEFAULT_RULES)


class GestureClassifier(Protocol):
    def classify(self, features: HandFeatures) -> str: ...


class RuleBasedClassifier:
    def __init__(self, rules: Iterable[GestureRule] = DEFAULT_RULES):
        self.rules = list(rules)

    def classify(self, features: HandFeatures) -> str:
        for rule in self.rules:
            if rule.matches(features):
                return rule.name
        return UNKNOWN


@dataclass(frozen=True)
class GestureResult:
    hand: str             # "Left" / "Right"
    gesture: str          # smoothed gesture (most common label in the window)
    frame_gesture: str    # label for this frame alone
    status: str           # STABLE / UNCERTAIN / UNRECOGNISED
    agreement: int        # frames in the window that agree with `gesture`
    window: int           # frames currently in the window
    newly_stable: bool    # True on the frame a new gesture becomes STABLE (log it once)


@dataclass
class _HandTrack:
    labels: deque
    last_logged: str | None = None


class GestureAnalyzer:
    def __init__(
        self,
        classifier: GestureClassifier | None = None,
        window: int = config.STABILITY_WINDOW,
        min_agreement: int = config.STABILITY_MIN_AGREEMENT,
    ):
        if not 1 <= min_agreement <= window:
            raise ValueError("min_agreement must be between 1 and window")
        self.classifier = classifier or RuleBasedClassifier()
        self.window = window
        self.min_agreement = min_agreement
        self._tracks: dict[str, _HandTrack] = {}

    def analyze(self, hands: list[HandFeatures]) -> list[GestureResult]:
        """Classify every hand in the current frame; call once per frame."""
        keys = _track_keys(hands)
        results = [self._update(key, features) for key, features in zip(keys, hands)]
        # Forget hands that left the frame so they start fresh when they return.
        for key in set(self._tracks) - set(keys):
            del self._tracks[key]
        return results

    def reset(self) -> None:
        self._tracks.clear()

    def _update(self, key: str, features: HandFeatures) -> GestureResult:
        track = self._tracks.setdefault(key, _HandTrack(deque(maxlen=self.window)))
        frame_gesture = self.classifier.classify(features)
        track.labels.append(frame_gesture)

        gesture, agreement = Counter(track.labels).most_common(1)[0]
        settled = agreement >= self.min_agreement
        if not settled:
            status = UNCERTAIN
        elif gesture == UNKNOWN:
            status = UNRECOGNISED
            track.last_logged = None
        else:
            status = STABLE

        newly_stable = status == STABLE and gesture != track.last_logged
        if newly_stable:
            track.last_logged = gesture

        return GestureResult(
            hand=features.handedness,
            gesture=gesture,
            frame_gesture=frame_gesture,
            status=status,
            agreement=agreement,
            window=len(track.labels),
            newly_stable=newly_stable,
        )


def _track_keys(hands: list[HandFeatures]) -> list[str]:
    # Hands are tracked by Left/Right. If MediaPipe labels both hands the same,
    # number the duplicates so their histories don't mix.
    seen: Counter = Counter()
    keys = []
    for features in hands:
        seen[features.handedness] += 1
        n = seen[features.handedness]
        keys.append(features.handedness if n == 1 else f"{features.handedness}#{n}")
    return keys
