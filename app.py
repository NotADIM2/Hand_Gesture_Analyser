"""Application entry point: connects the modules in pipeline order.

    Webcam -> OpenCV processing -> MediaPipe hand detection -> feature extraction
    -> gesture analyzer (+ motion analyzer) -> controller (optional key presses)
    -> UI result (+ SQLite gesture history)

Run:  python app.py            (see --help for options)
"""
import argparse
import sys
import time
import uuid
from pathlib import Path

import config
from camera import Camera, CameraError, to_rgb
from controller import GestureController
from database import GestureDatabase
from feature_extractor import extract_features
from gesture_analyzer import GestureAnalyzer
from hand_detector import HandDetector
from motion_analyzer import MotionAnalyzer, measure_heart
from ui import KEY_CLEAR, KEY_CONTROL, KEY_QUIT, FPSCounter, HistoryView, Interface


def load_history(db: GestureDatabase, session_id: str) -> HistoryView:
    return HistoryView(
        session_counts=db.count_gestures(session_id),
        recent=db.get_history(limit=config.HISTORY_RECENT_ROWS),
        all_time_total=db.total(),
    )


def run(
    camera_index: int = config.CAMERA_INDEX,
    db_path: Path | None = config.DB_PATH,
    control: str | None = None,
) -> int:
    session_id = uuid.uuid4().hex[:8]
    camera = Camera(index=camera_index)
    detector = db = ui = None
    try:
        camera.open()
        detector = HandDetector()
        db = GestureDatabase(db_path) if db_path is not None else None
        analyzer = GestureAnalyzer()
        motion = MotionAnalyzer()
        recent_moves = []
        last_move = None
        controller = GestureController(control)
        last_action, last_action_at = None, 0.0
        ui = Interface()
        fps = FPSCounter()
        history = HistoryView()
        history_loaded_at = 0.0
        print(f"Running (session {session_id}). Press q in the video window to quit.")
        if controller.enabled:
            print(f"Control mode: {controller.profile}. Click the window you want to control.")

        while True:
            frame = camera.read()
            if frame is None:
                # Keep the window responsive while the camera hiccups.
                if ui.poll() in KEY_QUIT or ui.closed:
                    break
                continue

            hands = detector.detect(to_rgb(frame))
            features = [extract_features(h.world_landmarks, h.image_points, h.handedness) for h in hands]
            results = analyzer.analyze(features)
            now = time.monotonic()
            moves = motion.update(hands, frame.shape[1], now)
            recent_moves = [m for m in recent_moves + moves if now - m.time < config.MOTION_DISPLAY_SECONDS]
            if moves:
                last_move = moves[-1]
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

            ui.render(
                frame, hands, results, features, fps.tick(), history if db else None,
                moves=recent_moves, trails=motion.trails(frame.shape[1]), last_move=last_move,
                heart=measure_heart(*hands) if len(hands) == 2 and ui.show_features else None,
                control=controller.profile,
                action=last_action if now - last_action_at < config.CONTROL_DISPLAY_SECONDS else None,
            )
            key = ui.show(frame)
            if key in KEY_QUIT or ui.closed:
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
                    print(f"Control mode: {profile or 'off'}" + (". Click the window you want to control." if profile else ""))
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-time hand gesture analyser.")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="webcam index (default: %(default)s)")
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="SQLite history file (default: %(default)s)")
    parser.add_argument("--no-history", action="store_true", help="don't record gesture history")
    parser.add_argument(
        "--control", choices=list(config.CONTROL_PROFILES),
        help="start with gesture control on, using this profile (press m in the window to switch)",
    )
    args = parser.parse_args(argv)
    return run(camera_index=args.camera, db_path=None if args.no_history else args.db, control=args.control)


if __name__ == "__main__":
    sys.exit(main())
