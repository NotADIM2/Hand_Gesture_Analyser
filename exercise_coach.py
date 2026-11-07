"""Guided hand exercises with repetition counting (extra mode: EXERCISES, key e).

A short routine of three exercises, one after the other:

    OPEN AND CLOSE   open the hand wide, then make a fist      (counts each fist -> open)
    THUMB TOUCHES    touch the thumb to each fingertip in turn   (counts each full round)
    COUNT TO FIVE    show 1, 2, 3, 4, then 5 fingers             (counts each 1 -> 5)

It counts repetitions, times them, and for open-and-close measures how far the
fingers open and close (the bend angle at the middle finger joints). Each finished
exercise is saved to the history database, and the session report shows progress.

This is an exercise tracker, not a medical device: it doesn't diagnose anything.
"""
from dataclasses import dataclass

import numpy as np

import config
from feature_extractor import HandFeatures
from hand_detector import DetectedHand

THUMB_TIP = 4
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
WRIST, MIDDLE_MCP = 0, 9
FOUR_FINGERS = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class ExerciseResult:
    exercise: str
    reps: int
    seconds: float
    detail: str = ""

    @property
    def seconds_per_rep(self) -> float:
        return self.seconds / self.reps if self.reps else 0.0


def mean_finger_angle(f: HandFeatures) -> float:
    """Average bend angle of the four fingers: about 180 when open, well under 100 in a fist."""
    return float(np.mean([f.finger_angles[n] for n in FOUR_FINGERS]))


class OpenClose:
    name = "OPEN AND CLOSE"
    instruction = "open your hand wide, then make a fist"

    def __init__(self):
        self.phase = None           # "open" or "closed"
        self.opened, self.closed = [], []   # widest / tightest angle of each repetition
        self._widest, self._tightest = 0.0, 180.0

    def update(self, hand: DetectedHand, f: HandFeatures) -> bool:
        angle = mean_finger_angle(f)
        self._widest, self._tightest = max(self._widest, angle), min(self._tightest, angle)
        if angle >= config.EXERCISE_OPEN_ANGLE and self.phase != "open":
            rep = self.phase == "closed"
            if rep:
                self.opened.append(self._widest)
                self.closed.append(self._tightest)
                self._widest, self._tightest = angle, angle
            self.phase = "open"
            return rep
        if angle <= config.EXERCISE_CLOSED_ANGLE and self.phase == "open":
            self.phase = "closed"
        return False

    def detail(self) -> str:
        if not self.opened:
            return ""
        return f"opens to {np.mean(self.opened):.0f} deg, closes to {np.mean(self.closed):.0f} deg"


class ThumbTouches:
    name = "THUMB TOUCHES"
    instruction = "touch your thumb to each fingertip: index, middle, ring, pinky"

    def __init__(self):
        self.target = 0                  # index into FOUR_FINGERS
        self._released = True            # the last touched finger has moved away again
        self._last: str | None = None

    def update(self, hand: DetectedHand, f: HandFeatures) -> bool:
        world = hand.world_landmarks
        palm = float(np.linalg.norm(world[MIDDLE_MCP] - world[WRIST])) or 1.0
        gap = {n: float(np.linalg.norm(world[THUMB_TIP] - world[tip])) / palm for n, tip in FINGER_TIPS.items()}
        if self._last is not None and gap[self._last] > config.EXERCISE_RELEASE_RATIO:
            self._released = True
        finger = FOUR_FINGERS[self.target]
        # Fingertips are close together, so the target must also be the nearest one:
        # touching the middle fingertip mustn't count as touching the index.
        nearest = min(gap, key=gap.get)
        if self._released and nearest == finger and gap[finger] < config.EXERCISE_TOUCH_RATIO:
            self._last, self._released = finger, False
            self.target = (self.target + 1) % len(FOUR_FINGERS)
            return self.target == 0      # back to the index: one full round
        return False

    def detail(self) -> str:
        return ""

    @property
    def next_finger(self) -> str:
        return FOUR_FINGERS[self.target]


class CountToFive:
    name = "COUNT TO FIVE"
    instruction = "show 1, 2, 3, 4, then 5 fingers"

    def __init__(self):
        self.target = 1
        self._held = 0

    def update(self, hand: DetectedHand, f: HandFeatures) -> bool:
        self._held = self._held + 1 if f.extended_count == self.target else 0
        if self._held < config.EXERCISE_HOLD_FRAMES:
            return False
        self._held = 0
        if self.target == 5:
            self.target = 1
            return True
        self.target += 1
        return False

    def detail(self) -> str:
        return ""


EXERCISES = {cls.name: cls for cls in (OpenClose, ThumbTouches, CountToFive)}


class ExerciseCoach:
    def __init__(self, routine=config.EXERCISE_ROUTINE):
        self.routine = tuple(routine)
        self.index = 0
        self.results: list[ExerciseResult] = []
        self.done = False
        self._new: list[ExerciseResult] = []
        self._start_exercise()

    def start(self) -> str:
        """Begin the routine from the first exercise; returns what to announce."""
        self.index, self.results, self.done, self._new = 0, [], False, []
        self._start_exercise()
        return f"{self.exercise.name}: {self.exercise.instruction}"

    def update(self, hands: list[DetectedHand], features: list[HandFeatures], now: float) -> list[str]:
        """Call once per frame; returns messages ("3/10", "NEXT: ...", "ROUTINE DONE")."""
        if self.done or not hands:
            return []
        if self._started_at is None:
            self._started_at = now   # time each exercise from when a hand first shows up
        messages = []
        if self.exercise.update(hands[0], features[0]):
            self.reps += 1
            messages.append(f"{self.reps}/{self.target}")
        if self.reps >= self.target:
            result = ExerciseResult(self.exercise.name, self.reps, now - self._started_at, self.exercise.detail())
            self.results.append(result)
            self._new.append(result)
            self.index += 1
            if self.index < len(self.routine):
                self._start_exercise()
                messages.append(f"NEXT: {self.exercise.name}")
            else:
                self.done = True
                messages.append("ROUTINE DONE")
        return messages

    def take_results(self) -> list[ExerciseResult]:
        """Exercises finished since the last call (to save to the database)."""
        new, self._new = self._new, []
        return new

    def status_lines(self) -> list[str]:
        if self.done:
            return ["EXERCISES DONE:"] + [
                f"  {r.exercise}: {r.reps} in {r.seconds:.1f} s"
                + (f", {r.detail}" if r.detail else "")
                for r in self.results
            ]
        ex = self.exercise
        instruction = ex.instruction
        if isinstance(ex, ThumbTouches):
            instruction += f" (next: {ex.next_finger})"
        elif isinstance(ex, CountToFive):
            instruction += f" (now: {ex.target})"
        return [
            f"EXERCISE {self.index + 1}/{len(self.routine)}: {ex.name}  {self.reps}/{self.target}",
            instruction,
        ]

    def _start_exercise(self) -> None:
        name, self.target = self.routine[self.index]
        self.exercise = EXERCISES[name]()
        self.reps = 0
        self._started_at = None
