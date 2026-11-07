"""Measure how accurate the gesture rules are on real hands.

Runs the classifier over every labelled real sample (MediaPipe's sample photos plus
everything recorded with scripts/record_samples.py) and reports accuracy per gesture,
per person, and a confusion matrix showing which gestures get mixed up.

It also compares the rules with the k-NN model (ml_classifier.py). The model is
tested leave-one-out: for each real sample it is trained on everything except that
sample, so it is never graded on a hand it has already seen.

Usage:
    python scripts/evaluate.py                              # print the report
    python scripts/evaluate.py --save reports/accuracy.md   # also save it as Markdown
"""
import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from conftest import load_real_samples  # noqa: E402
from feature_extractor import extract_features  # noqa: E402
from gesture_analyzer import GESTURES, RuleBasedClassifier  # noqa: E402
from ml_classifier import KNNClassifier  # noqa: E402
from train_classifier import real_vectors, synthetic_samples  # noqa: E402


@dataclass
class Evaluation:
    confusion: Counter = field(default_factory=Counter)        # (true, predicted) -> count
    by_person: dict = field(default_factory=lambda: defaultdict(Counter))  # person -> {"right"/"total"}

    @property
    def total(self) -> int:
        return sum(self.confusion.values())

    @property
    def correct(self) -> int:
        return sum(n for (true, predicted), n in self.confusion.items() if true == predicted)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def per_gesture(self) -> dict[str, tuple[int, int]]:
        """Gesture -> (correct, total), for gestures that have samples."""
        result = {}
        for gesture in GESTURES:
            total = sum(n for (true, _), n in self.confusion.items() if true == gesture)
            if total:
                result[gesture] = (self.confusion[(gesture, gesture)], total)
        return result


def evaluate(samples: list[dict], classifier=None) -> Evaluation:
    classifier = classifier or RuleBasedClassifier()
    ev = Evaluation()
    for sample in samples:
        if not sample.get("gesture"):
            continue  # unlabelled (e.g. a photo with no particular gesture)
        f = extract_features(sample["world_landmarks"], sample["image_points"], sample["handedness"])
        predicted = classifier.classify(f)
        ev.confusion[(sample["gesture"], predicted)] += 1
        person = ev.by_person[sample.get("person") or _source_name(sample.get("source", ""))]
        person["total"] += 1
        person["right"] += predicted == sample["gesture"]
    return ev


def evaluate_ml(samples: list[dict]) -> Evaluation:
    """k-NN accuracy on the real samples, leave-one-out: each is held out of its own training set."""
    synthetic = list(synthetic_samples())
    labelled = [s for s in samples if s.get("gesture")]
    real = real_vectors(labelled)
    ev = Evaluation()
    for i, (sample, (true, vector)) in enumerate(zip(labelled, real)):
        labels, vectors = zip(*(synthetic + real[:i] + real[i + 1:]))
        predicted = KNNClassifier(np.stack(vectors), labels).predict(vector)
        ev.confusion[(true, predicted)] += 1
        person = ev.by_person[sample.get("person") or _source_name(sample.get("source", ""))]
        person["total"] += 1
        person["right"] += predicted == true
    return ev


def format_comparison(rules: Evaluation, ml: Evaluation) -> str:
    lines = [
        "## Rules vs machine learning (k-NN, leave-one-out)",
        "",
        f"**Rules: {rules.accuracy:.0%}   ML: {ml.accuracy:.0%}**  ({rules.total} real samples)",
        "",
        "| Gesture | Rules | ML |",
        "|---|---|---|",
    ]
    ml_per = ml.per_gesture()
    for gesture, (right, total) in rules.per_gesture().items():
        ml_right, _ = ml_per.get(gesture, (0, total))
        lines.append(f"| {gesture} | {right}/{total} | {ml_right}/{total} |")
    return "\n".join(lines) + "\n"


def _source_name(source: str) -> str:
    return "MediaPipe sample photos" if source.startswith("mediapipe-sample") else "unknown"


def format_report(ev: Evaluation) -> str:
    if not ev.total:
        return ("No labelled samples yet. Record some with:\n\n"
                "    python scripts/record_samples.py --person <your name>\n")
    lines = [
        "# Gesture recognition accuracy",
        "",
        f"**Overall: {ev.correct}/{ev.total} correct ({ev.accuracy:.0%})**",
        "",
        "## Per gesture",
        "",
        "| Gesture | Correct | Samples | Accuracy |",
        "|---|---|---|---|",
    ]
    for gesture, (right, total) in ev.per_gesture().items():
        lines.append(f"| {gesture} | {right} | {total} | {right / total:.0%} |")

    lines += ["", "## Per person", "", "| Person | Correct | Samples | Accuracy |", "|---|---|---|---|"]
    for person, c in sorted(ev.by_person.items()):
        lines.append(f"| {person} | {c['right']} | {c['total']} | {c['right'] / c['total']:.0%} |")

    mistakes = sorted(((n, t, p) for (t, p), n in ev.confusion.items() if t != p), reverse=True)
    lines += ["", "## Mistakes (confusion)", ""]
    if mistakes:
        lines += ["| Shown | Recognised as | Times |", "|---|---|---|"]
        lines += [f"| {t} | {p} | {n} |" for n, t, p in mistakes]
    else:
        lines.append("None: every sample was recognised correctly.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", type=Path, help="also write the report to this Markdown file")
    parser.add_argument("--no-ml", action="store_true", help="skip the rules vs machine learning comparison")
    args = parser.parse_args(argv)
    samples = load_real_samples()
    rules = evaluate(samples)
    report = format_report(rules)
    if rules.total and not args.no_ml:
        report += "\n" + format_comparison(rules, evaluate_ml(samples))
    print(report)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(report, encoding="utf-8")
        print(f"Saved to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
