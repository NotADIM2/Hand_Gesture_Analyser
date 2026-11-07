"""Application entry point: connects the modules in pipeline order.

    Webcam -> OpenCV processing -> MediaPipe hand detection -> feature extraction
    -> gesture analyzer (+ motion analyzer) -> controller (optional key presses)
    -> UI result (+ SQLite gesture history)

Extra modes, switched with keys in the window: air drawing (d) and sign letters (a).
While one is on, gesture control is paused so the same gestures don't press keys.
New poses can be taught live (t); --benchmark times every stage of the pipeline.

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
from benchmark import StageTimer
from camera import Camera, CameraError, to_rgb
from controller import GestureController, GesturePassword
from database import GestureDatabase
from feature_extractor import extract_features
from gesture_analyzer import GestureAnalyzer, RuleBasedClassifier
from hand_detector import HandDetector
from ml_classifier import KNNClassifier
from motion_analyzer import MotionAnalyzer, measure_heart
from sign_language import SignSpeller
from teach import Teacher, TaughtGestures, WithTaughtGestures
from ui import (
    KEY_CLEAR, KEY_CONTROL, KEY_DRAW, KEY_ESC, KEY_QUIT, KEY_SAVE, KEY_SIGN, KEY_TEACH,
    FPSCounter, HistoryView, Interface,
)


def load_history(db: GestureDatabase, session_id: str) -> HistoryView:
    return HistoryView(
        session_counts=db.count_gestures(session_id),
        recent=db.get_history(limit=config.HISTORY_RECENT_ROWS),
        all_time_total=db.total(),
    )


def status_lines(controller: GestureController, password: GesturePassword, mode: str | None) -> list[str]:
    lines = []
    if controller.enabled:
        if mode:
            state = "PAUSED"
        elif password.unlocked:
            state = "ON"
        else:
            state = f"LOCKED {password.progress}/{len(password.sequence)}, show password"
        lines.append(f"CONTROL: {controller.profile.upper()} ({state})")
    if mode == "draw":
        lines.append("AIR DRAW: pinch to draw, fist clears")
    if mode == "sign":
        lines.append("SIGN LETTERS: palm = space, swipe left = delete, wave = clear")
    return lines


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
) -> int:
    session_id = uuid.uuid4().hex[:8]
    camera = Camera(index=camera_index)
    detector = db = ui = None
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
        mode = None  # None, "draw" or "sign"
        last_action, last_action_at = None, 0.0
        ui = Interface(fullscreen=fullscreen)
        fps = FPSCounter()
        history = HistoryView()
        history_loaded_at = 0.0
        print(f"Running (session {session_id}, {classifier} classifier). Press q in the video window to quit.")
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
            if mode == "draw":
                canvas.update(frame.shape, hands, features, results)
                canvas.overlay(frame)
            elif mode == "sign":
                change = speller.update(results, moves)
                if change:
                    last_action, last_action_at = change, now
            elif controller.enabled and not password.unlocked:
                if password.update(results, now):
                    last_action, last_action_at = "CONTROL UNLOCKED", now
            else:
                for action in controller.update(results, moves, now):
                    last_action, last_action_at = action.label, now

            if db is not None:
                for result in results:
                    if result.newly_stable:
                        db.insert(result.gesture, result.hand, session_id)
                for move in moves:
                    db.insert(move.gesture, move.hand, session_id)
                # Only query history while its panel is open, and at most a few times a second.
                if ui.show_history and time.monotonic() - history_loaded_at > config.HISTORY_REFRESH_SECONDS:
                    history = load_history(db, session_id)
                    history_loaded_at = time.monotonic()
            timer.lap("modes, control, history")

            status = status_lines(controller, password, mode)
            if teacher.active:
                status.append(teacher.status_line(now))
            if benchmark:
                status.append(timer.live_line())
            ui.render(
                frame, hands, results, features, fps.tick(), history if db else None,
                moves=recent_moves, trails=motion.trails(frame.shape[1]), last_move=last_move,
                heart=measure_heart(*hands) if len(hands) == 2 and ui.show_features else None,
                status=status,
                action=last_action if now - last_action_at < config.CONTROL_DISPLAY_SECONDS else None,
                sign_text=speller.text if mode == "sign" else None,
            )
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
            if key in (KEY_DRAW, KEY_SIGN):
                wanted = "draw" if key == KEY_DRAW else "sign"
                mode = None if mode == wanted else wanted
                canvas.clear()
                speller.clear()
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
    args = parser.parse_args(argv)
    return run(
        camera_index=args.camera, db_path=None if args.no_history else args.db,
        control=args.control, classifier=args.classifier, benchmark=args.benchmark,
        fullscreen=args.fullscreen, person=args.person,
    )


if __name__ == "__main__":
    sys.exit(main())
