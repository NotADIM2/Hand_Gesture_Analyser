"""MediaPipe initialisation and landmark detection (architecture step: HAND DETECTION).

Wraps the MediaPipe Hand Landmarker (Tasks API) and returns plain numpy arrays, so
the rest of the app never depends on MediaPipe types.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import config

NUM_LANDMARKS = 21


@dataclass(frozen=True)
class DetectedHand:
    handedness: str               # "Left" or "Right"
    handedness_score: float       # MediaPipe's score for the Left/Right label
    landmarks: np.ndarray         # (21, 3) normalised: x, y in 0..1 of the frame
    world_landmarks: np.ndarray   # (21, 3) metres, origin at the hand's centre
    image_points: np.ndarray      # (21, 3) pixels; z uses the same scale as x


class HandDetector:
    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        max_hands: int = config.MAX_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_presence_confidence: float = config.MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmark model not found at {model_path}.\n"
                "Download it with:  python scripts/download_model.py"
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, rgb_frame: np.ndarray) -> list[DetectedHand]:
        """Detect hands in an RGB frame (use camera.to_rgb on OpenCV frames)."""
        height, width = rgb_frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect_for_video(image, self._next_timestamp_ms())

        hands = []
        for landmarks, world, handedness in zip(
            result.hand_landmarks, result.hand_world_landmarks, result.handedness
        ):
            normalised = np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float64)
            pixels = normalised * np.array([width, height, width], dtype=np.float64)
            hands.append(
                DetectedHand(
                    handedness=handedness[0].category_name,
                    handedness_score=float(handedness[0].score),
                    landmarks=normalised,
                    world_landmarks=np.array([(p.x, p.y, p.z) for p in world], dtype=np.float64),
                    image_points=pixels,
                )
            )
        return hands

    def _next_timestamp_ms(self) -> int:
        # VIDEO mode requires strictly increasing timestamps.
        now = int(time.monotonic() * 1000)
        self._last_timestamp_ms = max(now, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    # Module check: live feed with the hand skeleton and Left/Right label. Press q to quit.
    import cv2

    from camera import Camera, to_rgb
    from ui import draw_hand

    with Camera() as cam, HandDetector() as detector:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            for hand in detector.detect(to_rgb(frame)):
                draw_hand(frame, hand, f"{hand.handedness} ({hand.handedness_score:.2f})")
            cv2.imshow("hand detector check", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
