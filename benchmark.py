"""Measure how long each pipeline stage takes (python app.py --benchmark).

Each frame is timed stage by stage: camera, hand detection, feature extraction,
classification, motion analysis, modes / control / history, drawing and display.
The window shows the live per-frame time; on exit a table of averages is printed and
saved to reports/benchmark.md.
"""
import time
from collections import defaultdict

STAGE_NOTES = {
    "camera": "includes waiting for the webcam's next frame",
    "display": "includes the window's key check (waitKey)",
}


class StageTimer:
    def __init__(self, clock=time.perf_counter, smoothing: float = 0.9):
        self.clock = clock
        self.smoothing = smoothing
        self.totals: dict[str, float] = defaultdict(float)   # seconds per stage, all frames
        self.recent: dict[str, float] = {}                    # smoothed seconds per stage
        self.frames = 0
        self._last = None
        self._frame: dict[str, float] = {}

    def start(self) -> None:
        self._last = self.clock()
        self._frame = {}

    def lap(self, stage: str) -> None:
        """Time since the previous lap (or start) is charged to `stage`."""
        now = self.clock()
        self._frame[stage] = self._frame.get(stage, 0.0) + now - self._last
        self._last = now

    def end_frame(self) -> None:
        self.frames += 1
        for stage, seconds in self._frame.items():
            self.totals[stage] += seconds
            old = self.recent.get(stage)
            self.recent[stage] = seconds if old is None else self.smoothing * old + (1 - self.smoothing) * seconds

    def averages_ms(self) -> dict[str, float]:
        return {stage: 1000 * total / self.frames for stage, total in self.totals.items()} if self.frames else {}

    def live_line(self) -> str:
        """Short on-screen summary of the current per-frame times."""
        if not self.recent:
            return "SPEED: measuring..."
        total = 1000 * sum(self.recent.values())
        detect = 1000 * self.recent.get("hand detection", 0.0)
        return f"SPEED: {total:.1f} ms/frame, hand detection {detect:.1f} ms"

    def summary(self) -> str:
        averages = self.averages_ms()
        if not averages:
            return "No frames measured."
        total = sum(averages.values())
        processing = total - averages.get("camera", 0.0)
        lines = [
            "# Speed benchmark",
            "",
            f"{self.frames} frames. Average {total:.1f} ms per frame ({1000 / total:.0f} FPS); "
            f"without waiting for the camera {processing:.1f} ms "
            f"({1000 / processing:.0f} FPS possible)." if processing > 0 else "",
            "",
            "| Stage | Average ms | Share |",
            "|---|---|---|",
        ]
        for stage, ms in sorted(averages.items(), key=lambda item: -item[1]):
            note = f" ({STAGE_NOTES[stage]})" if stage in STAGE_NOTES else ""
            lines.append(f"| {stage}{note} | {ms:.2f} | {ms / total:.0%} |")
        return "\n".join(lines) + "\n"
