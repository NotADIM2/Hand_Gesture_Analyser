"""Practise the sign words (mode 7: SIGN PRACTICE).

The app asks for a word ("SIGN: HELP") with a hint on how to sign it, and checks
what you sign with the same recogniser as the sign words mode. A correct sign scores
a point, adds to your streak and moves on to the next word; a different word says
what it saw so you can try again; n skips the word. Words come in a shuffled order,
each once before any repeats.

When you leave the mode, the result (signs correct, attempts, best streak, time) is
saved with the exercise results, so the session report shows practice over time.
"""
import random
from dataclasses import dataclass

from sign_words import HINTS, WORDS


@dataclass(frozen=True)
class Attempt:
    target: str
    signed: str | None   # None = skipped
    seconds: float       # from when the word was asked

    @property
    def correct(self) -> bool:
        return self.signed == self.target


class SignPractice:
    name = "SIGN PRACTICE"

    def __init__(self, words=WORDS, rng: random.Random | None = None):
        self.all_words = tuple(words)
        self.rng = rng or random.Random()
        self.attempts: list[Attempt] = []
        self.streak = self.best_streak = 0
        self.target: str | None = None
        self.started_at = self.asked_at = 0.0
        self._queue: list[str] = []

    def start(self, now: float) -> str:
        """Begin a new round; returns what to announce."""
        self.attempts, self.streak, self.best_streak = [], 0, 0
        self._queue, self.target = [], None
        self.started_at = now
        return self._next(now)

    def check(self, word: str | None, now: float) -> str | None:
        """Feed each recognised word; returns a message to show/say, or None."""
        if word is None or self.target is None:
            return None
        self.attempts.append(Attempt(self.target, word, now - self.asked_at))
        if word != self.target:
            self.streak = 0
            return f"THAT WAS {word}, TRY AGAIN"
        self.streak += 1
        self.best_streak = max(self.best_streak, self.streak)
        self._next(now)
        return "CORRECT!"

    def skip(self, now: float) -> str:
        if self.target is not None:
            self.attempts.append(Attempt(self.target, None, now - self.asked_at))
        self.streak = 0
        return self._next(now)

    @property
    def correct(self) -> int:
        return sum(a.correct for a in self.attempts)

    @property
    def tries(self) -> int:
        """Signs attempted (skips not counted)."""
        return sum(a.signed is not None for a in self.attempts)

    def summary(self) -> str:
        return f"{self.correct}/{self.tries} signs correct, best streak {self.best_streak}"

    def status_lines(self) -> list[str]:
        if self.target is None:
            return []
        return [
            f"HOW: {HINTS[self.target]}",
            f"SCORE: {self.correct} correct of {self.tries} tried   STREAK: {self.streak} (best {self.best_streak})",
            "n: skip this word",
        ]

    def _next(self, now: float) -> str:
        if not self._queue:
            order = list(self.all_words)
            self.rng.shuffle(order)
            if len(order) > 1 and order[0] == self.target:   # never the same word twice in a row
                order.append(order.pop(0))
            self._queue = order
        self.target = self._queue.pop(0)
        self.asked_at = now
        return f"SIGN: {self.target}"
