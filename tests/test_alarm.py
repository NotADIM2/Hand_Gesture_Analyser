"""Emergency alert: raised by DANGER / HELP, a siren, a flashing red screen, x to dismiss."""
import threading

import numpy as np

import config
import ui
from alarm import Alarm


class FakeBeep:
    """Records tones instead of making a sound."""

    def __init__(self, seconds_each=0.0):
        self.tones = []
        self.seconds_each = seconds_each   # a real beep takes time; 0 = instant

    def __call__(self, frequency, milliseconds):
        self.tones.append(frequency)
        threading.Event().wait(self.seconds_each)


def test_only_urgent_words_raise_the_alert():
    alarm = Alarm(sound=False)
    assert alarm.is_alert("DANGER") and alarm.is_alert("HELP")
    assert not alarm.is_alert("HELLO") and not alarm.is_alert(None)


def test_alert_lasts_its_time_then_ends():
    alarm = Alarm(sound=False)
    alarm.raise_alert("DANGER", now=10.0)
    assert alarm.active(10.0) and alarm.word == "DANGER" and alarm.raised_at is not None
    alarm.update(10.0 + config.ALERT_SECONDS - 0.1)
    assert alarm.active(10.0 + config.ALERT_SECONDS - 0.1)
    alarm.update(10.0 + config.ALERT_SECONDS)
    assert not alarm.active(10.0 + config.ALERT_SECONDS) and alarm.word is None


def test_signing_again_extends_the_alert():
    alarm = Alarm(sound=False)
    alarm.raise_alert("HELP", now=0.0)
    alarm.raise_alert("DANGER", now=4.0)
    assert alarm.word == "DANGER" and alarm.active(config.ALERT_SECONDS + 1.0)
    assert alarm.started_at == 0.0     # still the same alert, flashing on from the start


def test_x_dismisses_and_stops_the_siren():
    beep = FakeBeep(seconds_each=0.02)
    alarm = Alarm(beep=beep)
    alarm.raise_alert("DANGER", now=0.0)
    alarm.dismiss()
    alarm.wait()
    assert not alarm.active(0.1)
    assert len(beep.tones) < config.ALERT_SECONDS * 1000 / config.ALERT_BEEP_MS   # cut short


def test_siren_alternates_two_tones_for_the_alert_time():
    beep = FakeBeep()
    alarm = Alarm(beep=beep)
    alarm.raise_alert("HELP", now=0.0)
    alarm.wait()
    low, high = config.ALERT_TONES
    assert beep.tones[:4] == [low, high, low, high]
    assert len(beep.tones) == round(config.ALERT_SECONDS * 1000 / config.ALERT_BEEP_MS)


def test_muted_alarm_is_silent_but_still_shows():
    beep = FakeBeep()
    alarm = Alarm(sound=False, beep=beep)
    alarm.raise_alert("DANGER", now=0.0)
    alarm.wait()
    assert beep.tones == [] and alarm.active(0.0)


def test_broken_sound_never_breaks_the_app():
    def no_sound(frequency, milliseconds):
        raise RuntimeError("no sound device")
    alarm = Alarm(beep=no_sound)
    alarm.raise_alert("DANGER", now=0.0)
    alarm.wait()
    assert alarm.active(0.0)


def test_flashing_turns_on_and_off():
    alarm = Alarm(sound=False)
    alarm.raise_alert("DANGER", now=0.0)
    step = 1 / (2 * config.ALERT_FLASH_HZ)
    assert [alarm.flash_on(t * step + 0.01) for t in range(4)] == [True, False, True, False]


def test_red_alert_screen():
    frame = np.full((480, 640, 3), 100, np.uint8)
    ui.draw_alert(frame, "DANGER", flash_on=True, when="14:05:09")
    blue, green, red = (frame[..., i].astype(int) for i in range(3))
    assert (red > blue + 80).mean() > 0.5          # most of the picture is red
    assert (frame[:10, :10] == ui.ALERT_RED).all()  # red border
    dim = np.full((480, 640, 3), 100, np.uint8)
    ui.draw_alert(dim, "DANGER", flash_on=False)
    assert (dim[240, 5] == ui.ALERT_RED).all() and (dim[100, 320] == 100).all()   # border only
