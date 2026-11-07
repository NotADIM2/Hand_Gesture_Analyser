"""Webcam capture and frame handling (architecture steps: WEBCAM + IMAGE PROCESSING).

Owns the OpenCV VideoCapture, turns raw camera output into consistently sized,
mirrored BGR frames, and converts frames to RGB for MediaPipe.
"""
import sys

import cv2
import numpy as np

import config

# Media Foundation is the native Windows backend; the default would try FFmpeg first
# and print a warning on every start.
_BACKEND = cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or stops delivering frames."""


class Camera:
    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        mirror: bool = config.MIRROR_FRAME,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror
        self._capture = None
        self._failed_reads = 0

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> "Camera":
        capture = cv2.VideoCapture(self.index, _BACKEND)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera {self.index}. Check that a webcam is connected "
                "and not being used by another app (Zoom, Teams, browser tabs...)."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        ok, _ = capture.read()
        if not ok:
            capture.release()
            raise CameraError(f"Camera {self.index} opened but returned no image.")
        self._capture = capture
        self._failed_reads = 0
        return self

    def read(self) -> np.ndarray | None:
        """Return the next processed BGR frame, or None if this read failed.

        Raises CameraError once reads have failed MAX_FAILED_READS times in a row,
        which usually means the camera was unplugged.
        """
        if not self.is_open:
            raise CameraError("Camera is not open.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._failed_reads += 1
            if self._failed_reads >= config.MAX_FAILED_READS:
                raise CameraError("Camera stopped sending frames (disconnected?).")
            return None
        self._failed_reads = 0
        return self.process(frame)

    def process(self, frame: np.ndarray) -> np.ndarray:
        # Some webcams ignore the requested resolution; resize so the rest of the
        # pipeline always sees the same frame size.
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """OpenCV frames are BGR; MediaPipe expects RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    # Module check: show the live feed. Press q to quit.
    import time

    with Camera() as cam:
        last = time.perf_counter()
        while True:
            frame = cam.read()
            if frame is None:
                continue
            now = time.perf_counter()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("camera check", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
