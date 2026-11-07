import pytest

import config
import gesture_analyzer as ga
import motion_analyzer as ma
from controller import Action, GestureController, PynputKeyboard, load_profile


class FakeKeyboard:
    """Records key presses instead of sending them, so tests never press real keys."""

    def __init__(self):
        self.pressed = []

    def press(self, key):
        self.pressed.append(key)


def pose(gesture, newly_stable=False, status=ga.STABLE, hand="Right"):
    return ga.GestureResult(hand, gesture, gesture, status, 9, 10, newly_stable)


def move(gesture, hand="Right"):
    return ma.MotionEvent(gesture, hand, 0.0)


def controller(profile="presentation"):
    keyboard = FakeKeyboard()
    return GestureController(profile, keyboard=keyboard), keyboard


def test_off_by_default_presses_nothing():
    c = GestureController(keyboard=FakeKeyboard())
    assert not c.enabled
    assert c.update([pose(ga.THUMBS_UP, newly_stable=True)], [move(ma.SWIPE_LEFT)], 0.0) == []


def test_swipe_left_is_next_slide():
    c, keyboard = controller()
    actions = c.update([], [move(ma.SWIPE_LEFT)], 0.0)
    assert [a.label for a in actions] == ["NEXT SLIDE"]
    assert keyboard.pressed == ["right"]


def test_pose_triggers_once_when_it_becomes_stable():
    c, keyboard = controller()
    c.update([pose(ga.THUMBS_UP, newly_stable=True)], [], 0.0)
    for i in range(1, 30):  # still holding it
        c.update([pose(ga.THUMBS_UP)], [], i / 30)
    assert keyboard.pressed == ["f5"]


def test_uncertain_pose_does_nothing():
    c, keyboard = controller()
    c.update([pose(ga.THUMBS_UP, newly_stable=True, status=ga.UNCERTAIN)], [], 0.0)
    assert keyboard.pressed == []


def test_unmapped_gestures_do_nothing():
    c, keyboard = controller()
    c.update([pose(ga.OPEN_PALM, newly_stable=True)], [move(ma.WAVE)], 0.0)
    assert keyboard.pressed == []


def test_held_volume_repeats():
    c, keyboard = controller("media")
    c.update([pose(ga.THUMBS_UP, newly_stable=True)], [], 0.0)
    for i in range(1, 31):  # held for one more second
        c.update([pose(ga.THUMBS_UP)], [], i / 30)
    # once at the start, then every CONTROL_REPEAT_SECONDS (0.4 s) -> 0.4, 0.8
    assert keyboard.pressed == ["media_volume_up"] * 3


def test_cooldown_stops_double_presses():
    c, keyboard = controller()
    c.update([], [move(ma.SWIPE_LEFT, "Left"), move(ma.SWIPE_LEFT, "Right")], 0.0)
    c.update([], [move(ma.SWIPE_LEFT)], 0.2)
    c.update([], [move(ma.SWIPE_LEFT)], 0.2 + config.CONTROL_COOLDOWN_SECONDS + 0.1)
    assert keyboard.pressed == ["right", "right"]


def test_cycle_goes_through_profiles_then_off():
    c = GestureController(keyboard=FakeKeyboard())
    seen = [c.cycle() for _ in range(len(config.CONTROL_PROFILES) + 1)]
    assert seen == [*config.CONTROL_PROFILES, None]


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        GestureController("karaoke", keyboard=FakeKeyboard())


def test_profiles_only_use_known_gestures():
    known = set(ga.GESTURES) | set(ma.MOTION_GESTURES) | set(ma.TWO_HAND_GESTURES)
    for name in config.CONTROL_PROFILES:
        assert set(load_profile(name)) <= known, name


def test_open_palm_is_never_mapped():
    # It's how a hand looks right before a swipe, so it would fire by accident.
    for name in config.CONTROL_PROFILES:
        assert ga.OPEN_PALM not in load_profile(name)


def test_profile_keys_exist_in_pynput():
    keyboard = PynputKeyboard()  # only resolves names here; nothing is pressed
    for name in config.CONTROL_PROFILES:
        for action in load_profile(name).values():
            keyboard.resolve(action.key)
    with pytest.raises(ValueError):
        keyboard.resolve("not_a_key")


def test_action_from_config_tuple():
    assert Action("VOLUME UP", "media_volume_up", True).repeat
    assert not Action("NEXT SLIDE", "right").repeat
