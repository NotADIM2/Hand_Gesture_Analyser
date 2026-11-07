"""Spell words with sign-language letters (extra mode: SIGN LETTERS).

Several of the recognised poses match letters of the American Sign Language (ASL)
fingerspelling alphabet; config.SIGN_LETTERS lists them. Each letter is added once,
when its pose becomes STABLE. To repeat a letter, drop the hand or change pose and
come back.

    OPEN PALM  -> space        SWIPE LEFT -> delete the last letter
    WAVE       -> clear everything

This is a subset of the alphabet: letters that need motion (J, Z) or poses the rules
can't tell apart yet are not included.
"""
from typing import Iterable

import config
from gesture_analyzer import OPEN_PALM, GestureResult
from motion_analyzer import SWIPE_LEFT, WAVE, MotionEvent


class SignSpeller:
    def __init__(self, letters: dict[str, str] = config.SIGN_LETTERS, max_length: int = config.SIGN_MAX_LENGTH):
        self.letters = letters
        self.max_length = max_length
        self.text = ""

    def update(self, results: Iterable[GestureResult], moves: Iterable[MotionEvent]) -> str | None:
        """Feed this frame's gestures; returns what changed ("A", "SPACE", "DELETE", "CLEAR") or None."""
        change = None
        for move in moves:
            if move.gesture == SWIPE_LEFT and self.text:
                self.text = self.text[:-1]
                change = "DELETE"
            elif move.gesture == WAVE:
                self.text = ""
                change = "CLEAR"
        for result in results:
            if not result.newly_stable:
                continue
            if result.gesture == OPEN_PALM:
                if self.text and not self.text.endswith(" "):
                    self._add(" ")
                    change = "SPACE"
            elif result.gesture in self.letters:
                letter = self.letters[result.gesture]
                self._add(letter)
                change = letter
        return change

    def clear(self) -> None:
        self.text = ""

    def _add(self, text: str) -> None:
        self.text = (self.text + text)[-self.max_length:]
