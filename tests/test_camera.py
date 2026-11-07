import numpy as np
import pytest

import camera
import config
from camera import Camera, CameraError, to_rgb


class FakeCapture:
    """Stands in for cv2.VideoCapture so tests don't need a webcam."""

    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        return (frame is not None), frame

    def set(self, *args):
        return True

    def release(self):
        self.released = True


def use_fake(monkeypatch, fake):
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *args: fake)
    return fake


def frame(width=640, height=480):
    f = np.zeros((height, width, 3), np.uint8)
    f[:, : width // 2] = (255, 0, 0)  # left half blue (BGR)
    return f


def test_open_failure_raises_clear_error(monkeypatch):
    use_fake(monkeypatch, FakeCapture([], opened=False))
    with pytest.raises(CameraError, match="Could not open camera"):
        Camera().open()


def test_camera_with_no_image_raises(monkeypatch):
    use_fake(monkeypatch, FakeCapture([None]))
    with pytest.raises(CameraError, match="returned no image"):
        Camera().open()


def test_read_mirrors_frame(monkeypatch):
    use_fake(monkeypatch, FakeCapture([frame(), frame()]))
    with Camera(mirror=True) as cam:
        out = cam.read()
    assert out.shape == (480, 640, 3)
    assert tuple(out[0, -1]) == (255, 0, 0)  # blue moved to the right edge
    assert tuple(out[0, 0]) == (0, 0, 0)


def test_read_resizes_unexpected_resolution(monkeypatch):
    use_fake(monkeypatch, FakeCapture([frame(1280, 720), frame(1280, 720)]))
    with Camera(width=640, height=480) as cam:
        assert cam.read().shape == (480, 640, 3)


def test_single_failed_read_returns_none(monkeypatch):
    use_fake(monkeypatch, FakeCapture([frame(), None, frame()]))
    with Camera() as cam:
        assert cam.read() is None
        assert cam.read() is not None


def test_disconnect_raises_after_repeated_failures(monkeypatch):
    use_fake(monkeypatch, FakeCapture([frame()] + [None] * config.MAX_FAILED_READS))
    with Camera() as cam:
        with pytest.raises(CameraError, match="stopped sending frames"):
            for _ in range(config.MAX_FAILED_READS):
                cam.read()


def test_release_on_exit(monkeypatch):
    fake = use_fake(monkeypatch, FakeCapture([frame()]))
    with Camera() as cam:
        assert cam.is_open
    assert fake.released and not cam.is_open


def test_read_when_closed_raises():
    with pytest.raises(CameraError):
        Camera().read()


def test_to_rgb_swaps_channels():
    assert tuple(to_rgb(frame())[0, 0]) == (0, 0, 255)
