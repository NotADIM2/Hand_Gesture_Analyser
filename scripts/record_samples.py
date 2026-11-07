"""Record real, labelled hand landmarks from the webcam for testing the gesture rules.

Hold a gesture in front of the camera and press its key to save the current
hand's landmarks. Samples are appended to tests/fixtures/recorded_hands.json, which
tests/test_gesture_analyzer.py checks automatically.

Keys:  1-9, 0, a-g = save sample for that gesture (shown on screen)   q = quit

Usage:  python scripts/record_samples.py --person <your name>

Then measure accuracy with:  python scripts/evaluate.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from camera import Camera, to_rgb  # noqa: E402
from feature_extractor import extract_features  # noqa: E402
from gesture_analyzer import GESTURES, RuleBasedClassifier  # noqa: E402
from hand_detector import HandDetector  # noqa: E402
from ui import GREY, WHITE, draw_hand, draw_panel  # noqa: E402

OUTPUT = ROOT / "tests" / "fixtures" / "recorded_hands.json"
# One key per gesture, in GESTURES order. q is left free for quitting.
SAVE_KEYS = "1234567890abcdefghijklmnoprstuvwxyz"


def load() -> dict:
    if OUTPUT.is_file():
        return json.loads(OUTPUT.read_text())
    return {"description": "Real landmarks recorded with scripts/record_samples.py", "samples": []}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record labelled hand samples from the webcam.")
    parser.add_argument("--person", default="unknown", help="who is recording (for per-person accuracy)")
    args = parser.parse_args(argv)
    data = load()
    classifier = RuleBasedClassifier()
    keys = {ord(key): name for key, name in zip(SAVE_KEYS, GESTURES)}
    menu = [(f"{key}: {name}", GREY) for key, name in zip(SAVE_KEYS, GESTURES)]

    with Camera() as cam, HandDetector(max_hands=1) as detector:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            hands = detector.detect(to_rgb(frame))
            for hand in hands:
                f = extract_features(hand.world_landmarks, hand.image_points, hand.handedness)
                draw_hand(frame, hand, f"rules say: {classifier.classify(f)}")
            status = (f"Saved: {len(data['samples'])} samples", WHITE)
            draw_panel(frame, [status, ("Press a key to save:", WHITE)] + menu + [("q: quit", GREY)], (10, 10), scale=0.5)
            cv2.imshow("record samples", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in keys and hands:
                hand = hands[0]
                data["samples"].append({
                    "source": "recorded",
                    "person": args.person,
                    "gesture": keys[key],
                    "handedness": hand.handedness,
                    "world_landmarks": hand.world_landmarks.round(5).tolist(),
                    "image_points": hand.image_points.round(2).tolist(),
                })
                OUTPUT.write_text(json.dumps(data, indent=1))
                print(f"saved {keys[key]} ({len(data['samples'])} total)")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
