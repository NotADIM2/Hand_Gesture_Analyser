"""Train the k-NN gesture classifier and save it to models/gesture_knn.npz.

Training data:
- synthetic hands for every gesture, randomly tilted, rotated and jittered so the
  model sees many variations (plus a few finger patterns that mean UNKNOWN);
- every real labelled sample: MediaPipe's sample photos and anything recorded with
  scripts/record_samples.py.

Usage:  python scripts/train_classifier.py
Then:   python app.py --classifier ml
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import config  # noqa: E402
from conftest import load_real_samples  # noqa: E402
from feature_extractor import extract_features  # noqa: E402
from gesture_analyzer import GESTURES, UNKNOWN  # noqa: E402
from ml_classifier import KNNClassifier, feature_vector  # noqa: E402
from synthetic_hand import gesture_hand, make_hand, rotate_3d, rotate_in_image  # noqa: E402

# Finger patterns that aren't a gesture, so the model also learns UNKNOWN.
UNKNOWN_HANDS = (
    dict(extended=("middle",)),
    dict(extended=("ring",)),
    dict(extended=("pinky",)),
    dict(extended=("ring", "pinky")),
    dict(extended=("middle", "ring")),
    dict(extended=("index", "ring")),
    dict(extended=("middle", "ring", "pinky")),
    dict(extended=("middle",), thumb="out"),
    dict(extended=("ring", "pinky"), thumb="out"),
)


def augment(pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pts = rotate_3d(pts, rng.uniform(-25, 25), rng.uniform(-25, 25))
    pts = rotate_in_image(pts, rng.uniform(-15, 15))
    return pts + rng.normal(0.0, 0.015, pts.shape)


def synthetic_samples(per_gesture: int = config.ML_SYNTHETIC_PER_GESTURE, seed: int = 0):
    """(label, vector) pairs from augmented synthetic hands."""
    rng = np.random.default_rng(seed)
    hands = [(g, lambda curl, g=g: gesture_hand(g, curl=curl)) for g in GESTURES]
    hands += [(UNKNOWN, lambda curl, kw=kw: make_hand(**kw, curl=curl)) for kw in UNKNOWN_HANDS]
    for label, build in hands:
        for _ in range(per_gesture):
            # Real folded fingers are often looser than a tight fist.
            pts = augment(build(curl=rng.uniform(0.55, 1.1)), rng)
            yield label, feature_vector(extract_features(pts, pts))


def real_vectors(samples: list[dict]) -> list[tuple[str, np.ndarray]]:
    return [
        (s["gesture"], feature_vector(extract_features(s["world_landmarks"], s["image_points"], s["handedness"])))
        for s in samples
        if s.get("gesture")
    ]


def train(samples: list[dict] | None = None) -> KNNClassifier:
    pairs = list(synthetic_samples()) + real_vectors(load_real_samples() if samples is None else samples)
    labels, vectors = zip(*pairs)
    return KNNClassifier(np.stack(vectors), labels)


def main() -> int:
    samples = load_real_samples()
    model = train(samples)
    model.save()
    counts = Counter(str(label) for label in model.labels)
    real = len(real_vectors(samples))
    print(f"Trained on {len(model.labels)} examples ({real} real, {len(model.labels) - real} synthetic), "
          f"{len(counts)} classes")
    print(f"Saved {config.ML_MODEL_PATH}")
    if real < 50:
        print("Tip: record real samples (python scripts/record_samples.py --person <name>) and train again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
