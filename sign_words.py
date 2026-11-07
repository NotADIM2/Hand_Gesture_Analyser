"""Recognise basic sign-language words (mode 6: SIGN WORDS).

23 everyday signs, recognised from the hand shape and how the hand moves over the
last second or so:

    Greetings   HELLO, GOODBYE, PLEASE, THANK YOU, SORRY, YES, NO
    Questions   HELP, WHO, WHAT, WHERE, GOOD, BAD
    Words       ME, YOU, SAME, MORE, HOME, WATER
    Urgent      STOP, WAIT, GO, DANGER

The app only tracks hands, not the face or body. Where a sign is made (forehead,
chin, chest) is approximated by how high the hand is in the picture, or not used;
the hand shape and the movement do most of the work. One-hand signs are read when
only one hand is in view, two-hand signs when both are.

Hand shapes are read from the finger features (which fingers are extended) plus two
of their own: a flattened O (fingertips gathered on the thumb) and where the index
points (towards or away from the camera, from MediaPipe's 3D landmarks). Movements
are counted with the same swing counter the wave uses.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

import config
from feature_extractor import HandFeatures
from gesture_analyzer import track_keys
from hand_detector import DetectedHand
from motion_analyzer import count_swings, palm_centre, palm_size

HELLO, GOODBYE, PLEASE, THANK_YOU, SORRY, YES, NO = "HELLO", "GOODBYE", "PLEASE", "THANK YOU", "SORRY", "YES", "NO"
HELP, WHO, WHAT, WHERE, GOOD, BAD = "HELP", "WHO", "WHAT", "WHERE", "GOOD", "BAD"
ME, YOU, SAME, MORE, HOME, WATER = "ME", "YOU", "SAME", "MORE", "HOME", "WATER"
STOP, WAIT, GO, DANGER = "STOP", "WAIT", "GO", "DANGER"
WORDS = (HELLO, GOODBYE, PLEASE, THANK_YOU, SORRY, YES, NO, HELP, WHO, WHAT, WHERE, GOOD, BAD,
         ME, YOU, SAME, MORE, HOME, WATER, STOP, WAIT, GO, DANGER)

# How to sign each word for this app (shown in the practice mode; the README has the same).
HINTS = {
    HELLO: "flat hand at your forehead, then straight out to the side",
    GOODBYE: "open hand up, fingers fold down and up together",
    PLEASE: "flat hand rubbing a circle on your chest",
    THANK_YOU: "flat hand from your chin straight down",
    SORRY: "fist rubbing a circle on your chest",
    YES: "fist nodding forward and back",
    NO: "index and middle fingers snap onto the thumb, twice",
    WHO: "thumb and index out (L), index bends and straightens",
    WHAT: "both flat hands shaking side to side",
    WHERE: "index finger up, shaken side to side",
    GOOD: "two hands: flat hand comes down onto the other",
    BAD: "flat hand from your lips down, turning the palm over",
    ME: "point at your own chest, hold still",
    YOU: "point at the camera, hold still",
    SAME: "thumb and pinky out (Y), side to side",
    MORE: "both hands: fingertips on thumbs, tap them together",
    HOME: "fingertips on the thumb, move up your cheek",
    WATER: "index, middle and ring up (W), tap small up and down",
    STOP: "two hands: edge of one hand chops onto the other palm",
    WAIT: "both flat hands still, fingers wiggling",
    GO: "both index fingers thrust forward together",
    DANGER: "two fists, one slides up the other again and again",
    HELP: "two hands: thumbs-up fist on a flat hand, lift both",
}

# Hand shapes used by the signs.
FLAT, FIST, INDEX, Y_SHAPE, W_SHAPE, FLAT_O, OTHER = "flat", "fist", "index", "Y", "W", "flat O", "other"
SHAPE_NAMES = {FLAT: "flat hand", FIST: "fist", INDEX: "index finger", Y_SHAPE: "Y shape",
               W_SHAPE: "W shape", FLAT_O: "flattened O", OTHER: "other shape"}

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP = 0, 4, 5, 8, 9, 12, 17
FINGER_TIPS = (8, 12, 16, 20)
KNUCKLES = (5, 9, 13, 17)


@dataclass(frozen=True)
class HandFrame:
    """One hand in one frame, reduced to what the signs need."""
    t: float
    x: float                  # palm centre, in frame widths
    y: float
    kx: float                 # knuckles, in frame widths (they move more than the palm when the hand tilts)
    ky: float
    height: float             # palm centre from the top of the picture, as a fraction of its height
    size: float               # palm length on screen, in frame widths
    shape: str
    extended: frozenset
    reach: tuple              # each fingertip's distance from the wrist, in palm lengths
    curl: tuple               # each fingertip's distance out of the palm's plane, in palm lengths
    snap: float               # index and middle tips to the thumb tip, in palm lengths
    tilt: float               # how far the hand leans towards (-) or away from (+) the camera
    normal: np.ndarray        # which way the palm faces (unit vector, 3D)
    pointing: np.ndarray      # which way the index points (unit vector, 3D; +z = away from the camera)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def palm_length(world: np.ndarray) -> float:
    """Wrist to the index, middle and pinky knuckles, averaged: steadier than one bone."""
    return float(np.mean([np.linalg.norm(world[k] - world[WRIST]) for k in (INDEX_MCP, MIDDLE_MCP, PINKY_MCP)])) or 1.0


def hand_shape(f: HandFeatures, world: np.ndarray) -> str:
    palm = palm_length(world)
    to_thumb = np.mean([np.linalg.norm(world[t] - world[THUMB_TIP]) for t in FINGER_TIPS]) / palm
    fingers = f.extended_fingers - {"thumb"}
    bend = np.mean([f.finger_angles[n] for n in ("index", "middle", "ring", "pinky")])
    if to_thumb < config.SIGN_FLAT_O_TIPS and bend >= config.SIGN_FLAT_O_MIN_ANGLE:
        return FLAT_O
    if fingers == {"index", "middle", "ring", "pinky"}:
        return FLAT
    if not fingers:
        return FIST           # fist, or A with the thumb up / at the side
    if fingers == {"index"}:
        return INDEX
    if f.extended_fingers == {"thumb", "pinky"}:
        return Y_SHAPE
    if fingers == {"index", "middle", "ring"}:
        return W_SHAPE
    return OTHER


def observe(hand: DetectedHand, f: HandFeatures, frame_shape: tuple, now: float) -> HandFrame:
    width, height = frame_shape[1], frame_shape[0]
    world = hand.world_landmarks
    palm = palm_length(world)
    centre = palm_centre(hand)
    knuckles = hand.image_points[list(KNUCKLES), :2].mean(axis=0)
    normal = _unit(np.cross(world[INDEX_MCP] - world[WRIST], world[PINKY_MCP] - world[WRIST]))
    snap = np.mean([np.linalg.norm(world[t] - world[THUMB_TIP]) for t in (INDEX_TIP, MIDDLE_TIP)]) / palm
    return HandFrame(
        t=now,
        x=float(centre[0] / width), y=float(centre[1] / width),
        kx=float(knuckles[0] / width), ky=float(knuckles[1] / width),
        height=float(centre[1] / height),
        size=palm_size(hand) / width,
        shape=hand_shape(f, world),
        extended=f.extended_fingers,
        reach=tuple(float(np.linalg.norm(world[t] - world[WRIST]) / palm) for t in FINGER_TIPS),
        curl=tuple(float(np.dot(world[t] - world[k], normal) / palm) for t, k in zip(FINGER_TIPS, KNUCKLES)),
        snap=float(snap),
        tilt=float(_unit(world[MIDDLE_MCP] - world[WRIST])[2]),
        normal=normal,
        pointing=_unit(world[INDEX_TIP] - world[INDEX_MCP]),
    )


# --- Measuring a run of frames ----------------------------------------------------------

def share(frames, *shapes) -> float:
    return sum(h.shape in shapes for h in frames) / len(frames) if frames else 0.0


def smooth(values) -> np.ndarray:
    """Moving average over a few frames: landmark jitter isn't counted as movement."""
    values = np.asarray(values, dtype=float)
    n = config.SIGN_SMOOTHING
    if len(values) <= n:
        return values
    padded = np.pad(values, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def swings(values, amplitude: float) -> int:
    return count_swings(smooth(values), amplitude)


def spread(frames) -> float:
    """How far the palm wandered: the larger side of the box around its path."""
    xs, ys = [h.x for h in frames], [h.y for h in frames]
    return max(np.ptp(xs), np.ptp(ys)) if frames else 0.0


def straight(frames) -> bool:
    """The palm went fairly directly from start to end (not round in a circle): it never
    strays far from the straight line between them. Unlike path length, jitter barely
    changes this."""
    return _straight(_path(frames))


def _path(frames) -> np.ndarray:
    return np.column_stack([smooth([h.x for h in frames]), smooth([h.y for h in frames])])


def _straight(xy: np.ndarray) -> bool:
    chord = xy[-1] - xy[0]
    length = float(np.linalg.norm(chord))
    if not length:
        return False
    offsets = xy - xy[0]
    stray = np.abs(offsets[:, 0] * chord[1] - offsets[:, 1] * chord[0]).max() / length
    along = offsets @ chord / length    # how far along the line each point is
    # It mustn't overshoot and come back either (up, then down the same line).
    backtrack = max(-along.min(), along.max() - length)
    return max(stray, backtrack) <= config.SIGN_MAX_STRAY * length


def last_stroke(frames, min_frames: int = 5):
    """The latest straight stretch of movement: the longest run of final frames that is
    straight. A sign like HELLO starts by raising the hand to the forehead; only the
    movement after that counts, so no pause is needed in between."""
    xy = _path(frames)   # smoothed once, then each candidate start is tried
    for start in range(len(frames) - min_frames + 1):
        if _straight(xy[start:]):
            return frames[start:]
    return []


def since(frames, seconds: float):
    return [h for h in frames if h.t >= frames[-1].t - seconds] if frames else []


def turned(frames) -> float:
    """Degrees the palm went round its centre (positive or negative); 0 if the path isn't round."""
    if len(frames) < 8:
        return 0.0
    xy = np.array([(h.x, h.y) for h in frames])
    offsets = xy - xy.mean(axis=0)
    radii = np.linalg.norm(offsets, axis=1)
    if radii.mean() < config.SIGN_CIRCLE_MIN_RADIUS or radii.min() < 0.4 * radii.mean():
        return 0.0
    angles = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
    return float(np.degrees(angles[-1] - angles[0]))


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


# --- One-hand signs -------------------------------------------------------------------------

def one_hand_sign(frames: list[HandFrame]) -> str | None:
    if len(frames) < 8:
        return None
    folded = lambda *names: sum(not (set(names) & h.extended) for h in frames) / len(frames)  # noqa: E731

    # Fingers moving: WHO (index hooks), NO (index and middle snap on the thumb),
    # GOODBYE (all four fingers flap together).
    index_reach = [h.reach[0] for h in frames]
    middle_reach = [h.reach[1] for h in frames]
    if (folded("middle", "ring", "pinky") >= config.SIGN_SHAPE_SHARE
            and swings(index_reach, config.SIGN_HOOK) >= config.SIGN_HOOK_SWINGS
            and swings(middle_reach, config.SIGN_HOOK) < 2):   # only the index moves (in NO, both do)
        return WHO
    if (folded("ring", "pinky") >= config.SIGN_SHAPE_SHARE
            and swings([h.snap for h in frames], config.SIGN_SNAP) >= config.SIGN_SNAP_SWINGS):
        return NO
    flap = [np.mean(h.reach) for h in frames]
    if swings(flap, config.SIGN_FLAP) >= config.SIGN_FLAP_SWINGS and spread(frames) < 2 * config.SIGN_STILL:
        return GOODBYE

    # Rubbing in a circle: PLEASE (flat hand), SORRY (fist).
    if abs(turned(frames)) >= config.SIGN_CIRCLE_MIN_TURN:
        if share(frames, FLAT) >= config.SIGN_SHAPE_SHARE:
            return PLEASE
        if share(frames, FIST) >= config.SIGN_SHAPE_SHARE:
            return SORRY

    # A shape moved back and forth: WHERE (index, side to side), SAME (Y, side to side),
    # YES (fist nodding), WATER (W tapping).
    xs = [h.x for h in frames]
    if share(frames, INDEX) >= config.SIGN_SHAPE_SHARE and swings(xs, config.SIGN_SHAKE) >= config.SIGN_SHAKE_SWINGS:
        return WHERE
    if share(frames, Y_SHAPE) >= config.SIGN_SHAPE_SHARE and swings(xs, config.SIGN_SHAKE) >= 2:
        return SAME
    if share(frames, FIST) >= config.SIGN_SHAPE_SHARE and (
            swings([h.tilt for h in frames], config.SIGN_NOD) >= config.SIGN_NOD_SWINGS
            or swings([h.ky for h in frames], config.SIGN_TAP) >= config.SIGN_TAP_SWINGS):
        return YES
    if share(frames, W_SHAPE) >= config.SIGN_SHAPE_SHARE and swings([h.ky for h in frames], config.SIGN_TAP) >= config.SIGN_TAP_SWINGS:
        return WATER

    # One clear movement of a flat hand: HELLO (from up high, out to the side, or
    # forwards towards the camera), THANK YOU (down and forward), BAD (down, turning
    # the palm over).
    move = last_stroke(since(frames, config.SIGN_MOVE_SECONDS))
    # HELLO counts from where the hand is up high (raising it there isn't part of it).
    high = next((i for i, h in enumerate(move) if h.height < config.SIGN_HIGH), len(move))
    salute = move[high:]
    if len(salute) >= 5 and share(salute, FLAT) >= config.SIGN_SHAPE_SHARE:
        dx, dy = salute[-1].x - salute[0].x, salute[-1].y - salute[0].y
        sideways = abs(dx) >= config.SIGN_MOVE and abs(dy) <= 0.8 * abs(dx)
        forwards = salute[-1].size / salute[0].size >= config.SIGN_TOWARDS and abs(dx) < config.SIGN_MOVE
        if sideways or forwards:
            return HELLO
    if move and share(move, FLAT) >= config.SIGN_SHAPE_SHARE:
        dx, dy = move[-1].x - move[0].x, move[-1].y - move[0].y
        if dy >= config.SIGN_MOVE and abs(dx) <= dy:
            turn = angle_between(move[0].normal, move[-1].normal)
            if turn >= config.SIGN_FLIP_ANGLE:
                return BAD
            if turn <= config.SIGN_FLIP_ANGLE / 2:
                return THANK_YOU
            return None   # the palm is half turned: wait and see whether it turns over

    # HOME: a flattened O moved a short way up near the face.
    if share(frames, FLAT_O) >= 0.7 and frames[0].height < config.SIGN_HIGH + 0.1:
        dx, dy = frames[-1].x - frames[0].x, frames[-1].y - frames[0].y
        if config.SIGN_STILL <= np.hypot(dx, dy) <= 3 * config.SIGN_MOVE and dy < 0:
            return HOME

    # Held still, pointing: ME (at yourself, away from the camera), YOU (at the camera).
    hold = since(frames, config.SIGN_HOLD_SECONDS)
    if (hold[-1].t - frames[0].t >= config.SIGN_HOLD_SECONDS and share(hold, INDEX) >= 0.9
            and spread(hold) < config.SIGN_STILL):
        depth = float(np.mean([h.pointing[2] for h in hold]))
        if depth >= config.SIGN_POINT_DEPTH:
            return ME
        if depth <= -config.SIGN_POINT_DEPTH:
            return YOU
    return None


# --- Two-hand signs -------------------------------------------------------------------------

def two_hand_sign(pairs: list[tuple[HandFrame, HandFrame]]) -> str | None:
    if len(pairs) < 8:
        return None
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    size = float(np.mean([h.size for h in a + b])) or 1.0
    gap = [np.hypot(p.x - q.x, p.y - q.y) / size for p, q in pairs]   # palm lengths apart
    both = lambda *shapes: min(share(a, *shapes), share(b, *shapes)) >= config.SIGN_SHAPE_SHARE  # noqa: E731

    # MORE: both flattened O, tapping the fingertips together.
    if both(FLAT_O) and swings(gap, config.SIGN_TOGETHER) >= config.SIGN_SHAKE_SWINGS:
        return MORE
    if both(FLAT):
        # WHAT: both hands shaking side to side.
        if min(swings([h.x for h in hand], config.SIGN_SHAKE) for hand in (a, b)) >= 2:
            return WHAT
        # WAIT: both hands still, fingers wiggling.
        wiggles = [sum(swings([h.curl[i] for h in hand], config.SIGN_WIGGLE) for i in range(4)) for hand in (a, b)]
        if min(wiggles) >= config.SIGN_WIGGLE_SWINGS and max(spread(a), spread(b)) < config.SIGN_STILL:
            return WAIT
    move = [p for p in pairs if p[0].t >= pairs[-1][0].t - config.SIGN_MOVE_SECONDS]
    ma, mb = [p[0] for p in move], [p[1] for p in move]
    # STOP / GOOD: one flat hand comes down onto the other, flat hand and stops there.
    if share(a, FLAT) >= 0.5 and share(b, FLAT) >= 0.5:
        drop = {0: ma[-1].y - ma[0].y, 1: mb[-1].y - mb[0].y}
        mover = max(drop, key=drop.get)
        moving, base = (ma, mb) if mover == 0 else (mb, ma)
        lands_on = (moving[-1].y < base[-1].y and gap[-1] < config.SIGN_NEAR)
        if drop[mover] >= config.SIGN_MOVE and spread(base) < config.SIGN_STILL * 1.5 and lands_on:
            return STOP if abs(moving[-1].normal[0]) >= config.SIGN_EDGE else GOOD
    # HELP: a fist resting on a flat hand, both lifting together.
    for top, under in ((ma, mb), (mb, ma)):
        if (share(top, FIST) >= config.SIGN_SHAPE_SHARE and share(under, FLAT) >= config.SIGN_SHAPE_SHARE
                and top[-1].y < under[-1].y and gap[-1] < config.SIGN_NEAR
                and min(top[0].y - top[-1].y, under[0].y - under[-1].y) >= config.SIGN_MOVE):
            return HELP
    # GO: both index fingers thrust the same way (or towards the camera).
    if share(ma, INDEX) >= config.SIGN_SHAPE_SHARE and share(mb, INDEX) >= config.SIGN_SHAPE_SHARE:
        da = np.array([ma[-1].x - ma[0].x, ma[-1].y - ma[0].y])
        db = np.array([mb[-1].x - mb[0].x, mb[-1].y - mb[0].y])
        same_way = min(np.linalg.norm(da), np.linalg.norm(db)) >= config.SIGN_MOVE and np.dot(_unit(da), _unit(db)) > 0.7
        closer = min(ma[-1].size / ma[0].size, mb[-1].size / mb[0].size) >= 1.25
        if same_way or closer:
            return GO
    # DANGER: two fists together, one sliding up the other repeatedly.
    if both(FIST) and np.mean(gap) < config.SIGN_NEAR * 1.5:
        taps = [swings([h.ky for h in hand], config.SIGN_TAP) for hand in (a, b)]
        if max(taps) >= config.SIGN_TAP_SWINGS:
            return DANGER
    return None


# --- Following the hands --------------------------------------------------------------------

class SignWords:
    def __init__(self):
        self._tracks: dict[str, deque] = {}
        self._pairs: deque = deque()
        self._last_pair_at = -1e9
        self._quiet_until = 0.0
        self._last_word, self._last_word_at = None, -1e9
        self.words: list[str] = []
        self.seen = ""   # the hand shapes in view, e.g. "flat hand, high" (shown on screen)

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def clear(self) -> None:
        self._tracks.clear()
        self._pairs.clear()
        self.words = []

    def update(self, hands: list[DetectedHand], features: list[HandFeatures],
               frame_shape: tuple, now: float) -> str | None:
        """Call once per frame; returns a word when one is recognised."""
        keys = track_keys(hands)
        frames = {k: observe(h, f, frame_shape, now) for k, h, f in zip(keys, hands, features)}
        self.seen = " + ".join(
            f"{SHAPE_NAMES[h.shape]}, {'high' if h.height < config.SIGN_HIGH else 'low'}"
            for h in sorted(frames.values(), key=lambda h: h.x))
        for key in set(self._tracks) - set(frames):
            del self._tracks[key]
        for key, frame in frames.items():
            track = self._tracks.setdefault(key, deque())
            if track and np.hypot(frame.x - track[-1].x, frame.y - track[-1].y) > config.MOTION_MAX_JUMP:
                track.clear()   # too far for one frame: a tracking glitch, start again
            track.append(frame)
        if len(frames) == 2:
            first, second = sorted(frames)
            self._pairs.append((frames[first], frames[second]))
            self._last_pair_at = now
        else:
            self._pairs.clear()
        oldest = now - config.SIGN_WINDOW_SECONDS
        for track in list(self._tracks.values()) + [self._pairs]:
            while track and _time(track[0]) < oldest:
                track.popleft()

        if now < self._quiet_until:
            return None
        if len(frames) == 2:
            word = two_hand_sign(list(self._pairs))
        elif len(frames) == 1 and now - self._last_pair_at > 0.5:
            # One-hand signs only once the other hand has been out of view a moment,
            # so half of a two-hand sign isn't read as a one-hand one.
            word = one_hand_sign(list(next(iter(self._tracks.values()))))
        else:
            word = None
        if word == self._last_word and now - self._last_word_at < config.SIGN_REPEAT_SECONDS:
            word = None   # still holding the same sign, not signing it again
        if word:
            self._last_word, self._last_word_at = word, now
            self.words = (self.words + [word])[-config.SIGN_MAX_WORDS:]
            self._quiet_until = now + config.SIGN_COOLDOWN_SECONDS
            for track in self._tracks.values():
                track.clear()
            self._pairs.clear()
        return word


def _time(item) -> float:
    return item[0].t if isinstance(item, tuple) else item.t
