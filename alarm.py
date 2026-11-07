"""Emergency alert for urgent signs (DANGER, HELP in the sign words mode).

When an alert word is signed, a siren sounds and the screen flashes red with the
word for ALERT_SECONDS, or until x is pressed. The alert is also printed with the
time and saved in the history as ALERT <WORD>.

The siren is two alternating tones from the computer's own sound (winsound on
Windows, the terminal bell elsewhere), played by a background thread so the video
never waits for it.
"""
import sys
import threading
from datetime import datetime

import config


def system_beep(frequency: int, milliseconds: int) -> None:
    if sys.platform == "win32":
        import winsound
        winsound.Beep(frequency, milliseconds)
    else:
        print("\a", end="", flush=True)
        threading.Event().wait(milliseconds / 1000)


class Alarm:
    def __init__(self, sound: bool = True, beep=system_beep):
        self.sound = sound
        self._beep = beep
        self.word: str | None = None
        self.started_at = 0.0
        self.until = 0.0
        self.raised_at: datetime | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def is_alert(self, word: str | None) -> bool:
        return word in config.ALERT_WORDS

    def raise_alert(self, word: str, now: float) -> None:
        """Start (or extend) the alert for this word."""
        if not self.active(now):
            self.started_at = now
            self.raised_at = datetime.now()
        self.word = word
        self.until = now + config.ALERT_SECONDS
        self._stop.clear()   # a siren still finishing after a dismiss carries on
        if self.sound and (self._thread is None or not self._thread.is_alive()):
            self._thread = threading.Thread(target=self._siren, args=(config.ALERT_SECONDS,),
                                            name="alarm", daemon=True)
            self._thread.start()

    def active(self, now: float) -> bool:
        return self.word is not None and now < self.until

    def dismiss(self) -> None:
        self.word = None
        self.until = 0.0
        self._stop.set()

    def update(self, now: float) -> None:
        """Call once per frame: ends the alert (and the siren) when its time is up."""
        if self.word is not None and now >= self.until:
            self.dismiss()

    def flash_on(self, now: float) -> bool:
        """For the flashing red screen: on and off a few times a second."""
        return int((now - self.started_at) * 2 * config.ALERT_FLASH_HZ) % 2 == 0

    def wait(self) -> None:
        """Block until the siren has stopped (used by tests)."""
        if self._thread is not None:
            self._thread.join()

    def _siren(self, seconds: float) -> None:
        low, high = config.ALERT_TONES
        played, tone = 0.0, 0
        while not self._stop.is_set() and played < seconds:
            try:
                self._beep(high if tone else low, config.ALERT_BEEP_MS)
            except Exception:  # sound is an extra; never let it break the app
                return
            played += config.ALERT_BEEP_MS / 1000
            tone = 1 - tone
