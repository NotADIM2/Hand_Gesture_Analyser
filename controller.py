"""Turn gestures into key presses (architecture step: CONTROLLER).

A profile in config.CONTROL_PROFILES maps gesture names to actions, e.g. SWIPE LEFT ->
the Right arrow key for "next slide". Control is off until a profile is chosen, so
the app never presses keys unless asked to.

- A hand pose triggers its action once, on the frame it becomes STABLE. Actions
  marked to repeat (e.g. volume) keep pressing while the pose is held.
- A moving or two-hand gesture triggers its action each time it happens.

Key presses go to whichever window is in front, so after turning control on, click
the window to control (PowerPoint, a music player, ...).
"""
from dataclasses import dataclass
from typing import Iterable, Protocol

import config
from gesture_analyzer import STABLE, GestureResult
from motion_analyzer import MotionEvent


@dataclass(frozen=True)
class Action:
    label: str            # shown on screen, e.g. "NEXT SLIDE"
    key: str              # a pynput Key name ("right", "media_play_pause") or one character ("b")
    repeat: bool = False  # keep pressing while the pose is held


class Keyboard(Protocol):
    def press(self, key: str) -> None: ...


class PynputKeyboard:
    """Presses real keys. pynput is imported only when control is first turned on."""

    def __init__(self):
        from pynput.keyboard import Controller, Key

        self._keyboard = Controller()
        self._keys = Key

    def resolve(self, key: str):
        if len(key) == 1:
            return key
        try:
            return getattr(self._keys, key)
        except AttributeError:
            raise ValueError(f"Unknown key {key!r} in config.CONTROL_PROFILES") from None

    def press(self, key: str) -> None:
        resolved = self.resolve(key)
        self._keyboard.press(resolved)
        self._keyboard.release(resolved)


def load_profile(name: str) -> dict[str, Action]:
    try:
        spec = config.CONTROL_PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown control profile {name!r}; choose from {', '.join(config.CONTROL_PROFILES)}") from None
    return {gesture: Action(*action) for gesture, action in spec.items()}


class GestureController:
    def __init__(
        self,
        profile: str | None = None,
        keyboard: Keyboard | None = None,
        cooldown: float = config.CONTROL_COOLDOWN_SECONDS,
        repeat_every: float = config.CONTROL_REPEAT_SECONDS,
    ):
        self.cooldown = cooldown
        self.repeat_every = repeat_every
        self._keyboard = keyboard
        self.profile: str | None = None
        self.actions: dict[str, Action] = {}
        self._last_action_at = float("-inf")
        if profile is not None:
            self.set_profile(profile)

    @property
    def enabled(self) -> bool:
        return self.profile is not None

    def set_profile(self, profile: str | None) -> None:
        """Switch to a profile by name, or None to turn control off."""
        if profile is None:
            self.profile, self.actions = None, {}
            return
        actions = load_profile(profile)
        if self._keyboard is None:
            self._keyboard = PynputKeyboard()
        if isinstance(self._keyboard, PynputKeyboard):
            for action in actions.values():
                self._keyboard.resolve(action.key)  # catch typos now, not mid-presentation
        self.profile, self.actions = profile, actions

    def cycle(self) -> str | None:
        """Off -> first profile -> next profile -> ... -> off. Returns the new profile."""
        names: list[str | None] = [None, *config.CONTROL_PROFILES]
        self.set_profile(names[(names.index(self.profile) + 1) % len(names)])
        return self.profile

    def update(self, results: Iterable[GestureResult], moves: Iterable[MotionEvent], now: float) -> list[Action]:
        """Press the keys for this frame's gestures; returns the actions taken. Call once per frame."""
        if not self.enabled:
            return []
        taken = []
        for move in moves:
            action = self.actions.get(move.gesture)
            if action and self._ready(now, self.cooldown):
                taken.append(self._press(action, now))
        for result in results:
            action = self.actions.get(result.gesture)
            if not action or result.status != STABLE:
                continue
            if result.newly_stable and self._ready(now, self.cooldown):
                taken.append(self._press(action, now))
            elif action.repeat and not result.newly_stable and self._ready(now, self.repeat_every):
                taken.append(self._press(action, now))
        return taken

    def _ready(self, now: float, wait: float) -> bool:
        return now - self._last_action_at >= wait

    def _press(self, action: Action, now: float) -> Action:
        self._keyboard.press(action.key)
        self._last_action_at = now
        return action
