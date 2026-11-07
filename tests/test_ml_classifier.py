import sys
from pathlib import Path

import numpy as np
import pytest

import config
import gesture_analyzer as ga
from feature_extractor import extract_features
from ml_classifier import KNNClassifier, feature_vector
from synthetic_hand import gesture_hand, make_hand, rotate_3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate  # noqa: E402
import train_classifier  # noqa: E402


def features(pts):
    return extract_features(pts, pts)


@pytest.fixture(scope="module")
def model():
    return train_classifier.train(samples=[])  # synthetic only


def test_feature_vector_is_fixed_length():
    assert feature_vector(features(make_hand())).shape == (19,)


@pytest.mark.parametrize("gesture", ga.GESTURES)
def test_ml_recognises_every_gesture(model, gesture):
    assert model.classify(features(gesture_hand(gesture))) == gesture


@pytest.mark.parametrize("gesture", [ga.FIST, ga.PEACE, ga.ROCK, ga.THUMBS_UP])
def test_ml_handles_tilt_and_loose_fingers(model, gesture):
    assert model.classify(features(rotate_3d(gesture_hand(gesture, curl=0.65), 15, -15))) == gesture


def test_ml_says_unknown_for_other_patterns(model):
    assert model.classify(features(make_hand(("ring", "pinky")))) == ga.UNKNOWN


def test_far_from_everything_is_unknown():
    knn = KNNClassifier(np.zeros((3, 2)), ["A", "A", "B"], k=3, max_distance=0.5)
    assert knn.predict(np.array([0.1, 0.0])) == "A"
    assert knn.predict(np.array([5.0, 5.0])) == ga.UNKNOWN


def test_save_and_load(tmp_path):
    knn = KNNClassifier(np.eye(3), ["A", "B", "C"], k=1)
    knn.save(tmp_path / "m.npz")
    loaded = KNNClassifier.load(tmp_path / "m.npz")
    assert loaded.predict(np.array([0.0, 1.0, 0.0])) == "B"


def test_missing_model_says_how_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_classifier.py"):
        KNNClassifier.load(tmp_path / "missing.npz")


@pytest.mark.skipif(not config.ML_MODEL_PATH.is_file(), reason="run scripts/train_classifier.py")
def test_committed_model_agrees_with_rules():
    knn = KNNClassifier.load()
    for gesture in ga.GESTURES:
        assert knn.classify(features(gesture_hand(gesture))) == gesture


def test_works_inside_the_gesture_analyzer(model):
    analyzer = ga.GestureAnalyzer(model, window=3, min_agreement=2)
    for _ in range(3):
        (result,) = analyzer.analyze([extract_features(*(gesture_hand(ga.PEACE),) * 2, "Right")])
    assert (result.gesture, result.status) == (ga.PEACE, ga.STABLE)


def test_leave_one_out_never_sees_the_test_sample():
    # A fist mislabelled PEACE: if the model had trained on it, it would answer PEACE.
    pts = make_hand()
    sample = {"gesture": "PEACE", "world_landmarks": pts, "image_points": pts,
              "handedness": "Right", "source": "recorded", "person": "alex"}
    ev = evaluate.evaluate_ml([sample])
    assert ev.confusion[("PEACE", ga.FIST)] == 1


def test_comparison_table():
    pts = make_hand()
    samples = [{"gesture": "FIST", "world_landmarks": pts, "image_points": pts,
                "handedness": "Right", "source": "recorded", "person": "alex"}]
    text = evaluate.format_comparison(evaluate.evaluate(samples), evaluate.evaluate_ml(samples))
    assert "**Rules: 100%   ML: 100%**" in text and "| FIST | 1/1 | 1/1 |" in text
