"""Measure how accurate the gesture rules are on real hands.

Runs the classifier over every labelled real sample (MediaPipe's sample photos plus
everything recorded with scripts/record_samples.py) and reports accuracy per gesture,
per person, and a confusion matrix showing which gestures get mixed up.

Usage:
    python scripts/evaluate.py                              # print the report
    python scripts/evaluate.py --save reports/accuracy.md   # also save it as Markdown
"""
import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import load_real_samples  # noqa: E402
from feature_extractor import extract_features  # noqa: E402
from gesture_analyzer import GESTURES, RuleBasedClassifier  # noqa: E402


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
    args = parser.parse_args(argv)
    report = format_report(evaluate(load_real_samples()))
    print(report)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(report, encoding="utf-8")
        print(f"Saved to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
