"""Machine-learning gesture classifier: k-nearest neighbours (alternative to the rules).

A drop-in replacement for RuleBasedClassifier: it implements classify(features) -> str
and sees the same HandFeatures, never raw frames. Each hand becomes a short vector of
numbers (which fingers are extended, joint angles, distances, directions); a new hand
gets the label most common among the K most similar hands it was trained on.

k-NN was chosen because it needs no training step beyond storing examples, so new
recordings are used immediately, and every answer can be explained by looking at its
nearest examples. Train it with:  python scripts/train_classifier.py
"""
from collections import Counter
from pathlib import Path

import numpy as np

import config
from feature_extractor import FINGER_NAMES, HandFeatures
from gesture_analyzer import UNKNOWN

DIRECTIONS = ("up", "down", "left", "right")


def feature_vector(f: HandFeatures) -> np.ndarray:
    """The numbers the model compares; each one scaled to roughly 0..1."""
    values = [float(f.extended[n]) for n in FINGER_NAMES]
    values += [f.finger_angles[n] / 180.0 for n in FINGER_NAMES]
    values += [
        float(np.clip(f.thumb_lateral, -1.5, 1.5)),
        float(min(f.thumb_index_distance, 2.0)),
        f.index_middle_spread / 90.0,
    ]
    values += [float(f.thumb_direction == d) for d in ("up", "down")]
    # Where the index points only means something when it is extended.
    values += [float(f.extended["index"] and f.index_direction == d) for d in DIRECTIONS]
    return np.array(values, dtype=np.float32)


class KNNClassifier:
    def __init__(self, vectors: np.ndarray, labels, k: int = config.ML_K,
                 max_distance: float = config.ML_MAX_DISTANCE):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.labels = np.asarray(labels)
        if len(self.vectors) != len(self.labels) or not len(self.labels):
            raise ValueError("need the same, non-zero number of vectors and labels")
        self.k = min(k, len(self.labels))
        self.max_distance = max_distance

    def classify(self, features: HandFeatures) -> str:
        return self.predict(feature_vector(features))

    def predict(self, vector: np.ndarray) -> str:
        distances = np.linalg.norm(self.vectors - vector, axis=1)
        nearest = np.argpartition(distances, self.k - 1)[: self.k]
        if distances[nearest].min() > self.max_distance:
            return UNKNOWN  # nothing like this was in the training data
        votes: Counter = Counter()
        for i in nearest:
            votes[str(self.labels[i])] += 1.0 / (distances[i] + 1e-3)  # closer examples count more
        return votes.most_common(1)[0][0]

    def save(self, path: Path = config.ML_MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self.vectors, labels=self.labels)

    @classmethod
    def load(cls, path: Path = config.ML_MODEL_PATH) -> "KNNClassifier":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Trained model not found at {path}.\nTrain it with:  python scripts/train_classifier.py"
            )
        data = np.load(path)
        return cls(data["vectors"], data["labels"])
