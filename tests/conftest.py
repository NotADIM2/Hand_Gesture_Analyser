import json
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent))  # project modules
sys.path.insert(0, str(TESTS_DIR))         # synthetic_hand helper

FIXTURES = TESTS_DIR / "fixtures"


def load_real_samples() -> list[dict]:
    """Real landmarks: MediaPipe's sample photos plus anything recorded with scripts/record_samples.py."""
    samples = []
    for name in ("sample_hands.json", "recorded_hands.json"):
        path = FIXTURES / name
        if path.is_file():
            for sample in json.loads(path.read_text())["samples"]:
                sample["world_landmarks"] = np.array(sample["world_landmarks"])
                sample["image_points"] = np.array(sample["image_points"])
                samples.append(sample)
    return samples


@pytest.fixture
def real_samples() -> list[dict]:
    return load_real_samples()
