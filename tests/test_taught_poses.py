"""Checks every committed pose file in taught_poses/, so a bad pose fails the pull request."""
import json

import numpy as np
import pytest

import config
import gesture_analyzer as ga
from feature_extractor import extract_features
from synthetic_hand import gesture_hand
from teach import RESERVED_NAMES, TaughtGestures, WithTaughtGestures, pose_filename

POSE_FILES = sorted(config.TAUGHT_POSES_DIR.glob("*.json"))


@pytest.mark.parametrize("path", POSE_FILES, ids=lambda p: p.name)
def test_pose_file_is_valid(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    name = data["name"]
    assert name == name.upper().strip() and name, "names are upper case"
    assert name not in RESERVED_NAMES, f"{name} is a built-in gesture name"
    assert path.name == pose_filename(name), f"file should be called {pose_filename(name)}"
    assert data["taught_by"], "say who taught it"
    assert len(data["examples"]) >= 10, "teach it with at least 10 examples"
    for example in data["examples"]:
        assert np.array(example["world_landmarks"]).shape == (21, 3)
        assert np.array(example["image_points"]).shape == (21, 3)


def test_pose_names_are_unique():
    names = [json.loads(p.read_text(encoding="utf-8"))["name"] for p in POSE_FILES]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("gesture", ga.GESTURES)
def test_committed_poses_dont_take_over_built_in_gestures(gesture):
    classifier = WithTaughtGestures(ga.RuleBasedClassifier(), TaughtGestures())
    for curl in (0.7, 1.0):
        pts = gesture_hand(gesture, curl=curl)
        assert classifier.classify(extract_features(pts, pts)) == gesture
