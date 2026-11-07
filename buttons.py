"""Virtual buttons: on-screen buttons pressed with a fingertip in the air.

Two ways to press, both always on:

    hover   hold the index fingertip on a button; a bar fills up and it clicks after
            BUTTON_HOLD_SECONDS (QUIT needs longer, so it can't happen by accident)
    pinch   point at a button and pinch thumb and index: it clicks at once

After a click the finger must leave the button before it can click it again, so
holding still doesn't repeat. Hands over a button are reported in `busy`, so the
app can keep them out of the current mode (pointing at a button isn't drawing or
signing).

Three sets of buttons (layouts below), placed in columns on the right:

    mode bar     every mode, full screen and quit: the whole app without a keyboard
    paint bar    draw shapes: colours, pen size, undo, clear, save
    talk board   sign words: tap a word to say it (HELP raises the alert)
"""
from dataclasses import dataclass

import cv2
import numpy as np

import config
from feature_extractor import HandFeatures
from gesture_analyzer import track_keys
from hand_detector import DetectedHand

INDEX_TIP = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class Button:
    id: str                          # what it does, e.g. "mode:draw", "colour:RED", "say:YES"
    label: str
    rect: tuple[int, int, int, int]  # x, y, width, height in pixels
    swatch: tuple | None = None      # fill colour (paint colours)
    active: bool = False             # the current mode / colour: drawn highlighted
    hold: float = config.BUTTON_HOLD_SECONDS

    def contains(self, point, padding: float = 0.0) -> bool:
        x, y, w, h = self.rect
        return x - padding <= point[0] <= x + w + padding and y - padding <= point[1] <= y + h + padding


class VirtualButtons:
    def __init__(self):
        self.progress: dict[str, float] = {}       # button id -> how far the hover has filled (0-1)
        self.busy: set[str] = set()                # hand keys over a button this frame
        self._hover: dict[str, tuple[str, float]] = {}   # hand key -> (button id, since)
        self._done: dict[str, str] = {}            # hand key -> button it clicked, until it leaves
        self._pinched: dict[str, bool] = {}
        self._flash: dict[str, float] = {}         # button id -> when it was clicked

    def update(self, buttons: list[Button], hands: list[DetectedHand],
               features: list[HandFeatures], now: float) -> list[str]:
        """Call once per frame; returns the ids of buttons clicked this frame."""
        clicked = []
        self.progress, self.busy = {}, set()
        keys = track_keys(hands)
        for key, hand, f in zip(keys, hands, features):
            tip = hand.image_points[INDEX_TIP, :2]
            pinched = f.thumb_index_distance < config.BUTTON_PINCH_RATIO
            was_pinched = self._pinched.get(key, False)
            self._pinched[key] = pinched
            # A little extra room while pinching: the fingertip moves towards the thumb.
            padding = config.BUTTON_PINCH_PADDING if pinched else 0.0
            button = next((b for b in buttons if b.contains(tip, padding)), None)
            if button is None:
                self._hover.pop(key, None)
                self._done.pop(key, None)
                continue
            self.busy.add(key)
            if self._done.get(key) == button.id:
                continue                       # clicked already: leave the button first
            self._done.pop(key, None)
            hovered, since = self._hover.get(key, (None, now))
            if hovered != button.id:
                since = now
            self._hover[key] = (button.id, since)
            held = (now - since) / button.hold
            if (pinched and not was_pinched) or held >= 1.0:
                clicked.append(button.id)
                self._flash[button.id] = now
                self._done[key] = button.id
                self._hover.pop(key, None)
            else:
                self.progress[button.id] = max(self.progress.get(button.id, 0.0), held)
        for key in set(self._hover) - set(keys):
            del self._hover[key]
        return clicked

    def draw(self, frame: np.ndarray, buttons: list[Button], now: float) -> None:
        for b in buttons:
            x, y, w, h = b.rect
            flashing = now - self._flash.get(b.id, -1e9) < config.BUTTON_FLASH_SECONDS
            region = frame[y:y + h, x:x + w]
            fill = np.zeros_like(region)
            fill[:] = b.swatch if b.swatch else (40, 40, 40)
            region[:] = cv2.addWeighted(region, 0.25 if b.swatch else 0.45, fill, 0.75 if b.swatch else 0.55, 0)
            done = self.progress.get(b.id, 0.0)
            if done > 0:   # the hover bar filling up
                cv2.rectangle(frame, (x, y + h - 6), (x + int(w * min(done, 1.0)), y + h), (80, 220, 80), -1)
            if flashing:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
            border = (0, 215, 255) if b.active else (200, 200, 200)
            cv2.rectangle(frame, (x, y), (x + w, y + h), border, 3 if b.active else 1, cv2.LINE_AA)
            if b.label:
                text_colour = (0, 0, 0) if flashing or (b.swatch and sum(b.swatch) > 450) else (255, 255, 255)
                scale = _fit(b.label, w - 6, h)
                (tw, th), _ = cv2.getTextSize(b.label, FONT, scale, 1)
                cv2.putText(frame, b.label, (x + (w - tw) // 2, y + (h + th) // 2), FONT, scale, text_colour, 1, cv2.LINE_AA)


def _fit(text: str, width: int, height: int) -> float:
    scale = min(0.5, height / 45)
    while scale > 0.25 and cv2.getTextSize(text, FONT, scale, 1)[0][0] > width:
        scale -= 0.02
    return scale


# --- Layouts --------------------------------------------------------------------------------

def column(items, frame_shape: tuple, right: float, width: float) -> list[Button]:
    """Buttons stacked top to bottom in a column ending `right` (fraction of the frame width
    from the right edge), `width` wide (fraction), between BUTTON_TOP and BUTTON_BOTTOM.
    items: (id, label, swatch, active, hold) tuples."""
    height, frame_width = frame_shape[:2]
    top, bottom = int(config.BUTTON_TOP * height), int(config.BUTTON_BOTTOM * height)
    gap = max(3, height // 120)
    each = (bottom - top - gap * (len(items) - 1)) // len(items)
    w = int(width * frame_width)
    x = frame_width - int(right * frame_width) - w
    return [Button(i, label, (x, top + n * (each + gap), w, each), swatch, active, hold)
            for n, (i, label, swatch, active, hold) in enumerate(items)]


MODE_BUTTON_LABELS = ("1 POSES", "2 MOVE", "3 DRAW", "4 ABC", "5 EXER", "6 SIGN", "7 PRACT")


def mode_bar(frame_shape: tuple, modes: tuple, current: str) -> list[Button]:
    hold = config.BUTTON_HOLD_SECONDS
    items = [(f"mode:{m}", label, None, m == current, hold) for m, label in zip(modes, MODE_BUTTON_LABELS)]
    items += [("fullscreen", "FULL", None, False, hold), ("quit", "QUIT", (40, 40, 160), False, config.BUTTON_QUIT_SECONDS)]
    return column(items, frame_shape, right=config.BUTTON_MARGIN, width=config.MODE_BAR_WIDTH)


def paint_bar(frame_shape: tuple, colour: tuple, size: int) -> list[Button]:
    items = [(f"colour:{name}", "", bgr, bgr == colour, config.BUTTON_HOLD_SECONDS)
             for name, bgr in config.PAINT_COLOURS.items()]
    items += [(i, label, None, False, config.BUTTON_HOLD_SECONDS)
              for i, label in (("size", f"SIZE {size}"), ("undo", "UNDO"), ("clear", "CLEAR"), ("save", "SAVE"))]
    right = config.BUTTON_MARGIN * 2 + config.MODE_BAR_WIDTH
    return column(items, frame_shape, right=right, width=config.PAINT_BAR_WIDTH)


def talk_board(frame_shape: tuple) -> list[Button]:
    items = [(f"say:{word}", word, (40, 40, 160) if word in config.ALERT_WORDS else None, False,
              config.BUTTON_HOLD_SECONDS) for word in config.TALK_WORDS]
    right = config.BUTTON_MARGIN * 2 + config.MODE_BAR_WIDTH
    return column(items, frame_shape, right=right, width=config.TALK_BOARD_WIDTH)
