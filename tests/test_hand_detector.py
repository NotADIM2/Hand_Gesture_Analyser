import numpy as np
import pytest

import config
from hand_detector import HandDetector

needs_model = pytest.mark.skipif(
    not config.MODEL_PATH.is_file(), reason="run scripts/download_model.py first"
)


def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_model.py"):
        HandDetector(model_path=tmp_path / "missing.task")


@needs_model
def test_blank_frame_has_no_hands():
    with HandDetector() as detector:
        for _ in range(3):  # also checks repeated calls get increasing timestamps
            assert detector.detect(np.zeros((480, 640, 3), np.uint8)) == []


@needs_model
def test_accepts_any_frame_size():
    with HandDetector() as detector:
        assert detector.detect(np.full((240, 320, 3), 128, np.uint8)) == []
        assert detector.detect(np.full((720, 1280, 3), 128, np.uint8)) == []
