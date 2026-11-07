"""Application entry point: connects the modules in pipeline order.

    Webcam -> OpenCV processing -> MediaPipe hand detection -> feature extraction
    -> gesture analyzer (+ motion analyzer) -> controller (optional key presses)
    -> UI result (+ SQLite gesture history)

It works in one mode at a time, picked with the number keys in the window:
1 poses (default), 2 moving gestures, 3 draw shapes (recognised when the pen lifts),
4 sign letters (spoken aloud), 5 hand exercises, 6 sign words (made into sentences
and spoken), 7 sign practice (asks for signs and keeps score). Gesture control (m) works in modes
1 and 2 and is paused in the others, so the same gestures don't press keys.
New poses can be taught live (t); --benchmark times every stage of the pipeline.
Virtual buttons on the right (buttons.py) do the same without a keyboard: pick a mode,
change the pen in draw shapes, tap a word in sign words (b hides them).

Run:  python app.py            (see --help for options)
"""
import argparse
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2

import config
from air_draw import AirCanvas
from alarm import Alarm
from benchmark import StageTimer
from buttons import Button, VirtualButtons, mode_bar, paint_bar, talk_board
from camera import Camera, CameraError, to_rgb
from controller import GestureController, GesturePassword
from database import GestureDatabase
from exercise_coach import ExerciseCoach
from feature_extractor import extract_features
from gesture_analyzer import GestureAnalyzer, RuleBasedClassifier, track_keys
from hand_detector import HandDetector
from ml_classifier import KNNClassifier
from motion_analyzer import WAVE, MotionAnalyzer, measure_heart
from phrases import PhraseBuilder
from sign_language import SignSpeller
from sign_practice import SignPractice
from sign_words import SignWords
from speech import Speaker
from teach import Teacher, TaughtGestures, WithTaughtGestures
from ui import (
    DRAW, EXERCISE, KEY_CLEAR, KEY_CONTROL, KEY_DISMISS, KEY_ESC, KEY_FULLSCREEN, KEY_QUIT, KEY_SKIP, KEY_UNDO,
    KEY_SAVE, KEY_TEACH, MODE_KEYS, MODE_TITLES, MODES, MOTION, POSES, PRACTICE, SHOWS_POSES, SIGN, WORDS,
    FPSCounter, HistoryView, Interface, draw_alert,
)

CONTROL_MODES = (POSES, MOTION)   # modes in which gesture control may press keys
# Virtual buttons that do what a key does.
BUTTON_KEYS = {"fullscreen": KEY_FULLSCREEN, "quit": KEY_QUIT[0], "undo": KEY_UNDO, "save": KEY_SAVE}
BUTTON_KEYS.update({f"mode:{mode}": ord(str(i)) for i, mode in enumerate(MODES, start=1)})


def button_panel(frame_shape: tuple, mode: str, canvas: AirCanvas) -> list[Button]:
    """The virtual buttons in this mode: the mode bar, plus the paint bar (draw shapes)
    or the talk board (sign words)."""
    panel = mode_bar(frame_shape, MODES, mode)
    if mode == DRAW:
        panel += paint_bar(frame_shape, canvas.colour, canvas.thickness)
    elif mode == WORDS:
        panel += talk_board(frame_shape)
    return panel


def free_hands(busy: set[str], hands: list, features: list, results: list) -> tuple[list, list, list]:
    """Leave out the hands on a virtual button: pointing at a button isn't drawing or signing."""
    keep = [i for i, key in enumerate(track_keys(hands)) if key not in busy]
    return [hands[i] for i in keep], [features[i] for i in keep], [results[i] for i in keep]


def load_history(db: GestureDatabase, session_id: str) -> HistoryView:
    return HistoryView(
        session_counts=db.count_gestures(session_id),
        recent=db.get_history(limit=config.HISTORY_RECENT_ROWS),
        all_time_total=db.total(),
    )


def status_lines(controller: GestureController, password: GesturePassword, mode: str) -> list[str]:
    lines = []
    if controller.enabled:
        if mode not in CONTROL_MODES:
            state = "PAUSED (control works in modes 1 and 2)"
        elif password.unlocked:
            state = "ON"
        else:
            state = f"LOCKED {password.progress}/{len(password.sequence)}, show password"
        lines.append(f"CONTROL: {controller.profile.upper()} ({state})")
    if mode == MOTION:
        lines.append("Move one palm: swipe, wave or draw a circle. Two hands: clap or heart")
    if mode == DRAW:
        lines.append("Pinch thumb and index to draw, open them to finish; z undoes, wave clears")
        lines.append("Shapes in one stroke or several: circle, triangle, square, rectangle, line")
    if mode == SIGN:
        lines.append("Palm = space, thumbs up = say it, swipe left = delete, wave = clear")
    if mode == WORDS:
        lines.append("Sign a word: HELLO, THANK YOU, YES, NO, HELP, STOP ... (list in the README)")
        lines.append("Or point at a word on the talk board to say it")
        lines.append("One-hand signs with only that hand in view; pause to finish a sentence")
        lines.append(f"{' or '.join(config.ALERT_WORDS)} raises an emergency alert (x dismisses)")
    return lines


def save_practice(db: GestureDatabase | None, practice: SignPractice, now: float, session_id: str) -> None:
    """Store a practice round with the exercise results (it shows in the session report)."""
    if db is not None and practice.tries:
        db.insert_exercise(practice.name, practice.correct, now - practice.started_at, practice.summary(), session_id)


def save_snapshot(frame) -> Path:
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.SNAPSHOT_DIR / f"gesture_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(str(path), frame)
    return path


def git_user_name() -> str:
    """Who is teaching, for taught pose files: --person, else git's user.name, else 'unknown'."""
    try:
        name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True,
                              cwd=config.BASE_DIR, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    return name or "unknown"


def run(
    camera_index: int = config.CAMERA_INDEX,
    db_path: Path | None = config.DB_PATH,
    control: str | None = None,
    classifier: str = "rules",
    benchmark: bool = False,
    fullscreen: bool = False,
    person: str | None = None,
    speak: bool = True,
) -> int:
    camera = Camera(index=camera_index)
    detector = db = ui = alarm = practice = None
    mode, session_id = POSES, uuid.uuid4().hex[:8]
    try:
        camera.open()
        detector = HandDetector()
        db = GestureDatabase(db_path) if db_path is not None else None
        taught = TaughtGestures()
        base = KNNClassifier.load() if classifier == "ml" else RuleBasedClassifier()
        analyzer = GestureAnalyzer(WithTaughtGestures(base, taught))
        teacher = Teacher(taught, author=person or git_user_name())
        timer = StageTimer()
        motion = MotionAnalyzer()
        recent_moves = []
        last_move = None
        controller = GestureController(control)
        password = GesturePassword()
        canvas = AirCanvas()
        speller = SignSpeller()
        signer = SignWords()
        phrases = PhraseBuilder()
        practice = SignPractice()
        coach = ExerciseCoach()
        speaker = Speaker(enabled=speak)
        alarm = Alarm(sound=speak)
        buttons = VirtualButtons()
        last_action, last_action_at = None, 0.0
        ui = Interface(fullscreen=fullscreen)
        fps = FPSCounter()
        history = HistoryView()
        history_loaded_at = 0.0
        print(f"Running (session {session_id}, {classifier} classifier). Press q in the video window to quit.")
        for skipped in taught.skipped:
            print(f"Warning: skipped taught pose file {skipped}; fix or delete it.", file=sys.stderr)
        if taught.names:
            print("Taught poses: " + ", ".join(f"{n} (by {', '.join(taught.taught_by[n])})" for n in taught.names))
            for clash in taught.clashes():
                print(f"Warning: taught pose {clash}; they may be confused.")
        if controller.enabled:
            print(f"Control mode: {controller.profile}. Show the gesture password, then click the window to control.")

        while True:
            timer.start()
            frame = camera.read()
            timer.lap("camera")
            if frame is None:
                # Keep the window responsive while the camera hiccups.
                if ui.poll() in KEY_QUIT or ui.closed:
                    break
                continue

            hands = detector.detect(to_rgb(frame))
            timer.lap("hand detection")
            features = [extract_features(h.world_landmarks, h.image_points, h.handedness) for h in hands]
            timer.lap("feature extraction")
            results = analyzer.analyze(features)
            timer.lap("gesture classifier")
            now = time.monotonic()
            shown = hands, results, features   # every hand is drawn, even one on a button
            panel, button_key, tapped = [], -1, None
            if ui.show_buttons:
                panel = button_panel(frame.shape, mode, canvas)
                for click in buttons.update(panel, hands, features, now):
                    if click.startswith("colour:"):
                        canvas.colour = config.PAINT_COLOURS[click.split(":")[1]]
                        last_action, last_action_at = f"PEN {click.split(':')[1]}", now
                    elif click == "size":
                        last_action, last_action_at = f"PEN SIZE {canvas.next_size()}", now
                    elif click == "clear":
                        canvas.clear()
                        last_action, last_action_at = "CLEARED", now
                    elif click.startswith("say:"):
                        tapped = click.split(":")[1]
                    elif button_key == -1:   # handled with the keys below
                        button_key = BUTTON_KEYS[click]
                if buttons.busy:
                    hands, features, results = free_hands(buttons.busy, hands, features, results)
            moves = motion.update(hands, frame.shape[1], now)
            timer.lap("motion analyzer")
            learned = teacher.update(hands, now)
            if learned:
                last_action, last_action_at = learned, now
                if teacher.saved_to is not None:
                    print(f"{learned}: saved {teacher.saved_to.relative_to(config.BASE_DIR)}. "
                          "Commit it to share the pose with your team.")
            recent_moves = [m for m in recent_moves + moves if now - m.time < config.MOTION_DISPLAY_SECONDS]
            if moves:
                last_move = moves[-1]
            drawn, signed = [], None
            if mode == DRAW:
                drawn = canvas.update(frame.shape, hands, features, results, now)
                for d in drawn:
                    last_action, last_action_at = f"SHAPE: {d.shape.name}", now
                # Wave to wipe the board (only with the pen up, so a zig-zag line can't do it).
                if not canvas.pen_down and not canvas.empty and any(m.gesture == WAVE for m in moves):
                    canvas.clear()
                    last_action, last_action_at = "CLEARED", now
                canvas.overlay(frame)
            elif mode == SIGN:
                change = speller.update(results, moves)
                if change:
                    last_action, last_action_at = change, now
                if change == "SPACE":
                    speaker.say(speller.last_word)
                elif change == "SAY":
                    speaker.say(speller.text)
            elif mode == WORDS:
                # A word tapped on the talk board counts just like a signed one.
                signed = tapped or signer.update(hands, features, frame.shape, now)
                if signed:
                    last_action, last_action_at = signed, now
                    speaker.say(signed.lower())
                    phrases.add(signed, now)
                sentence = phrases.update(now)
                if sentence:
                    print(f"Signed: {sentence}")
                    if phrases.is_phrase:   # words joined into something new: say it as a sentence
                        speaker.say(sentence)
                if alarm.is_alert(signed):
                    alarm.raise_alert(signed, now)
                    print(f"ALERT: {signed} signed at {alarm.raised_at:%H:%M:%S}")
                    if db is not None:
                        db.insert(f"ALERT {signed}", hands[0].handedness if len(hands) == 1 else "Both", session_id)
            elif mode == PRACTICE:
                message = practice.check(signer.update(hands, features, frame.shape, now), now)
                if message:
                    last_action, last_action_at = message, now
                    if message == "CORRECT!":
                        speaker.say(f"correct. next, {practice.target.lower()}")
                    else:
                        speaker.say(message.lower())
            elif mode == EXERCISE:
                for message in coach.update(hands, features, now):
                    last_action, last_action_at = message, now
                    if not message[0].isdigit():  # announce new exercises, not every repetition
                        speaker.say(message.replace("NEXT: ", "Next: ").capitalize())
                for done in coach.take_results():
                    if db is not None:
                        db.insert_exercise(done.exercise, done.reps, done.seconds, done.detail, session_id)
            elif controller.enabled and not password.unlocked:
                if password.update(results, now):
                    last_action, last_action_at = "CONTROL UNLOCKED", now
            if mode in CONTROL_MODES and password.unlocked:
                for action in controller.update(results, moves, now):
                    last_action, last_action_at = action.label, now

            if db is not None:
                # Log what the current mode is about: poses, moves or drawn shapes.
                if mode in SHOWS_POSES:
                    for result in results:
                        if result.newly_stable:
                            db.insert(result.gesture, result.hand, session_id)
                if mode == MOTION:
                    for move in moves:
                        db.insert(move.gesture, move.hand, session_id)
                for d in drawn:
                    db.insert(f"DRAWN {d.shape.name}", d.hand, session_id)
                if signed:
                    who = shown[0][0].handedness if len(shown[0]) == 1 else "Both"
                    db.insert(f"{'TAP' if tapped else 'SIGN'} {signed}", who, session_id)
                # Only query history while its panel is open, and at most a few times a second.
                if ui.show_history and time.monotonic() - history_loaded_at > config.HISTORY_REFRESH_SECONDS:
                    history = load_history(db, session_id)
                    history_loaded_at = time.monotonic()
            timer.lap("modes, control, history")

            status = status_lines(controller, password, mode)
            if mode == EXERCISE:
                status += coach.status_lines()
            if mode in (WORDS, PRACTICE):
                status.append(f"Seen: {signer.seen or 'no hand'}")
            if mode == PRACTICE:
                status += practice.status_lines()
            if mode == DRAW:
                pen = f"Pen: {'DOWN' if canvas.pen_down else 'UP'}"
                status.append(pen + (f"   Last stroke: {canvas.last_stroke}" if canvas.last_stroke else ""))
            if teacher.active:
                status.append(teacher.status_line(now))
            if benchmark:
                status.append(timer.live_line())
            ui.render(
                frame, *shown, fps.tick(), history if db else None,
                moves=recent_moves, trails=motion.trails(frame.shape[1]), last_move=last_move,
                heart=measure_heart(*hands) if len(hands) == 2 and ui.show_features else None,
                status=status,
                action=last_action if now - last_action_at < config.CONTROL_DISPLAY_SECONDS else None,
                # Sign words: the words being signed (with a cursor), then the finished sentence.
                sign_text=speller.text if mode == SIGN else " ".join(phrases.words) if mode == WORDS and phrases.words else None,
                mode=mode,
                prompt=(f"SIGN: {practice.target}" if mode == PRACTICE
                        else phrases.sentence if mode == WORDS and not phrases.words else None),
                right_edge=min((b.rect[0] for b in panel), default=None),
            )
            buttons.draw(frame, panel, now)
            alarm.update(now)
            if alarm.active(now):
                draw_alert(frame, alarm.word, alarm.flash_on(now), f"{alarm.raised_at:%H:%M:%S}")
            timer.lap("drawing")
            key = ui.show(frame)
            timer.lap("display")
            timer.end_frame()
            if ui.closed:
                break
            if teacher.naming:  # while typing a name, keys are letters, not commands
                message = teacher.handle_key(key, now)
                if message:
                    last_action, last_action_at = message, now
                continue
            if key == -1:   # no key pressed: a virtual button may stand for one
                key = button_key
            if key == KEY_ESC and ui.fullscreen:  # Esc leaves full screen first, then quits
                ui.set_fullscreen(False)
                continue
            if key in KEY_QUIT:
                break
            if key == KEY_CLEAR and db is not None:
                db.clear()
                history_loaded_at = 0.0
            if key == KEY_CONTROL:
                try:
                    profile = controller.cycle()
                except Exception as exc:  # e.g. no keyboard access on this system
                    controller.set_profile(None)
                    print(f"Control mode unavailable: {exc}", file=sys.stderr)
                else:
                    if profile is None:
                        password.lock()
                    print(f"Control mode: {profile or 'off'}")
            if key in MODE_KEYS:
                wanted = MODE_KEYS[key]
                # Pressing the current mode's key again goes back to poses.
                new_mode = POSES if wanted == mode else wanted
                if new_mode != mode:
                    if mode == PRACTICE:
                        save_practice(db, practice, now, session_id)
                        print(f"Sign practice: {practice.summary()}")
                    mode = new_mode
                    canvas.clear()
                    speller.clear()
                    signer.clear()
                    phrases.clear()
                    last_action, last_action_at = MODE_TITLES[mode], now
                    if mode == EXERCISE:
                        speaker.say(coach.start())
                    if mode == PRACTICE:
                        speaker.say(practice.start(now).replace("SIGN: ", "sign ").lower())
            if key == KEY_SKIP and mode == PRACTICE:
                message = practice.skip(now)
                last_action, last_action_at = message, now
                speaker.say(message.replace("SIGN: ", "sign ").lower())
            if key == KEY_UNDO and mode == DRAW and canvas.undo():
                last_action, last_action_at = "UNDO", now
            if key == KEY_DISMISS:
                alarm.dismiss()
            if key == KEY_SAVE:
                print(f"Saved {save_snapshot(frame)}")
            if key == KEY_TEACH and not teacher.active:
                teacher.start()
            ui.handle_key(key)
        return 0

    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        camera.release()
        if alarm is not None:
            alarm.dismiss()
        if mode == PRACTICE and practice is not None:
            save_practice(db, practice, time.monotonic(), session_id)
            print(f"Sign practice: {practice.summary()}")
        if detector is not None:
            detector.close()
        if db is not None:
            db.close()
        if ui is not None:
            ui.close()
        if benchmark and timer.frames:
            report = timer.summary()
            print(report)
            config.BENCHMARK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            config.BENCHMARK_REPORT_PATH.write_text(report, encoding="utf-8")
            print(f"Saved {config.BENCHMARK_REPORT_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-time hand gesture analyser.")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="webcam index (default: %(default)s)")
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="SQLite history file (default: %(default)s)")
    parser.add_argument("--no-history", action="store_true", help="don't record gesture history")
    parser.add_argument(
        "--control", choices=list(config.CONTROL_PROFILES),
        help="start with gesture control on, using this profile (press m in the window to switch)",
    )
    parser.add_argument(
        "--classifier", choices=("rules", "ml"), default="rules",
        help="recognise poses with geometric rules or the trained k-NN model (default: %(default)s)",
    )
    parser.add_argument("--benchmark", action="store_true", help="time each pipeline stage; summary saved on exit")
    parser.add_argument("--fullscreen", action="store_true", help="start full screen (w toggles it in the window)")
    parser.add_argument("--person", help="your name, saved in poses you teach (default: git user.name)")
    parser.add_argument("--mute", action="store_true",
                        help="no speech or alarm sound (sign letters, sign words, exercise coach)")
    args = parser.parse_args(argv)
    return run(
        camera_index=args.camera, db_path=None if args.no_history else args.db,
        control=args.control, classifier=args.classifier, benchmark=args.benchmark,
        fullscreen=args.fullscreen, person=args.person, speak=not args.mute,
    )


if __name__ == "__main__":
    sys.exit(main())
