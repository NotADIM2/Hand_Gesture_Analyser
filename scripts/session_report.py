"""Turn the gesture history database into an HTML report with charts.

Shows totals, the most used gestures (hand poses vs moving / two-hand gestures),
which hands were used, activity per session, and a table of sessions. The page is a
single self-contained file (charts are inline SVG), so it opens offline in any browser.

Usage:
    python scripts/session_report.py                    # all sessions
    python scripts/session_report.py --session latest   # only the most recent run
    python scripts/session_report.py --out my_report.html
"""
import argparse
import html
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from database import GestureDatabase, GestureRecord  # noqa: E402
from gesture_analyzer import GESTURES  # noqa: E402
from motion_analyzer import MOTION_GESTURES, TWO_HAND_GESTURES  # noqa: E402

DEFAULT_OUT = ROOT / "reports" / "session_report.html"
KINDS = {**{g: "pose" for g in GESTURES}, **{g: "moving" for g in MOTION_GESTURES},
         **{g: "two-hand" for g in TWO_HAND_GESTURES}}


@dataclass
class Session:
    session_id: str
    start: datetime
    end: datetime
    records: list[GestureRecord]

    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60

    def top(self, n: int = 3) -> list[tuple[str, int]]:
        return Counter(r.gesture for r in self.records).most_common(n)


def load_records(db_path: Path) -> list[GestureRecord]:
    """Every record, oldest first."""
    if not Path(db_path).is_file():
        return []
    with GestureDatabase(db_path) as db:
        return list(reversed(db.get_history(limit=10**9)))


def group_sessions(records: list[GestureRecord]) -> list[Session]:
    by_id: dict[str, list[GestureRecord]] = {}
    for r in records:
        by_id.setdefault(r.session_id or "-", []).append(r)
    sessions = []
    for sid, recs in by_id.items():
        times = [datetime.fromisoformat(r.timestamp) for r in recs]
        sessions.append(Session(sid, min(times), max(times), recs))
    return sorted(sessions, key=lambda s: s.start)


def bar_chart(rows: list[tuple[str, int, str]], unit: str = "") -> str:
    """Horizontal bars as inline SVG. rows: (label, value, css class)."""
    if not rows:
        return '<p class="muted">Nothing recorded yet.</p>'
    top = max(v for _, v, _ in rows)
    row_h, label_w, bar_w = 26, 170, 380
    parts = [f'<svg viewBox="0 0 {label_w + bar_w + 60} {row_h * len(rows)}" role="img" class="chart">']
    for i, (label, value, cls) in enumerate(rows):
        y = i * row_h
        w = max(2, bar_w * value / top)
        parts.append(
            f'<text x="{label_w - 8}" y="{y + 17}" text-anchor="end" class="label">{html.escape(label)}</text>'
            f'<rect x="{label_w}" y="{y + 5}" width="{w:.1f}" height="16" rx="3" class="bar {cls}"/>'
            f'<text x="{label_w + w + 6:.1f}" y="{y + 17}" class="value">{value}{unit}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def build_report(records: list[GestureRecord], title: str = "Gesture session report") -> str:
    sessions = group_sessions(records)
    counts = Counter(r.gesture for r in records)
    hands = Counter((r.hand or "-") for r in records)
    kinds = Counter(KINDS.get(g, "other") for g in (r.gesture for r in records))

    if records:
        first, last = sessions[0].start, sessions[-1].end
        period = f"{first:%d %b %Y %H:%M} to {last:%d %b %Y %H:%M}"
    else:
        period = "no sessions yet"

    stats = [
        ("Gestures recorded", len(records)),
        ("Sessions", len(sessions)),
        ("Different gestures", len(counts)),
        ("Moving / two-hand", kinds["moving"] + kinds["two-hand"]),
    ]
    stat_html = "".join(f'<div class="stat"><div class="num">{v}</div><div class="muted">{k}</div></div>' for k, v in stats)

    gesture_rows = [(g, n, KINDS.get(g, "other")) for g, n in counts.most_common()]
    hand_rows = [(h.upper(), n, "hand") for h, n in hands.most_common()]
    session_rows = [(f"{s.start:%d %b %H:%M}", len(s.records), "session") for s in sessions[-15:]]

    table = "".join(
        f"<tr><td>{s.start:%d %b %Y %H:%M}</td><td>{s.minutes:.1f}</td><td>{len(s.records)}</td>"
        f"<td>{html.escape(', '.join(f'{g} ({n})' for g, n in s.top()))}</td></tr>"
        for s in reversed(sessions)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#f7f7f8; --card:#fff; --text:#1d1d22; --muted:#6b6b76; --line:#e3e3e8;
         --pose:#2f6fdb; --moving:#d9488f; --two:#e07b00; --other:#888; --hand:#2a9d6f; --session:#6b5bd2; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#16161a; --card:#202026; --text:#ececf1; --muted:#9d9daa; --line:#33333c;
           --pose:#6c9cf0; --moving:#ee77b2; --two:#f5a54a; --hand:#4fc493; --session:#9d90ee; }}
}}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:900px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ margin:0 0 4px; font-size:26px; }}
h2 {{ font-size:17px; margin:0 0 12px; }}
.muted {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; margin-top:18px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin-top:18px; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }}
.num {{ font-size:28px; font-weight:650; }}
.chart {{ width:100%; height:auto; }}
.label, .value {{ font-size:12px; fill:var(--text); }}
.bar.pose {{ fill:var(--pose); }} .bar.moving {{ fill:var(--moving); }} .bar.two-hand {{ fill:var(--two); }}
.bar.other {{ fill:var(--other); }} .bar.hand {{ fill:var(--hand); }} .bar.session {{ fill:var(--session); }}
.legend span {{ display:inline-block; margin-right:14px; font-size:13px; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:middle; }}
.table-wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<div class="muted">{period} · generated {datetime.now():%d %b %Y %H:%M}</div>
<div class="stats">{stat_html}</div>
<section class="card"><h2>Most used gestures</h2>
<div class="legend"><span><i class="dot" style="background:var(--pose)"></i>Hand pose</span>
<span><i class="dot" style="background:var(--moving)"></i>Moving</span>
<span><i class="dot" style="background:var(--two)"></i>Two-hand</span></div>
{bar_chart(gesture_rows)}</section>
<section class="card"><h2>Hands used</h2>{bar_chart(hand_rows)}</section>
<section class="card"><h2>Gestures per session{' (last 15)' if len(sessions) > 15 else ''}</h2>{bar_chart(session_rows)}</section>
<section class="card"><h2>Sessions</h2><div class="table-wrap"><table>
<thead><tr><th>Started</th><th>Minutes</th><th>Gestures</th><th>Most used</th></tr></thead>
<tbody>{table}</tbody></table></div></section>
</main></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="history database (default: %(default)s)")
    parser.add_argument("--session", help="a session id, or 'latest' for the most recent run")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="HTML file to write (default: %(default)s)")
    args = parser.parse_args(argv)

    records = load_records(args.db)
    title = "Gesture session report"
    if args.session:
        sessions = group_sessions(records)
        wanted = sessions[-1].session_id if args.session == "latest" and sessions else args.session
        records = [r for r in records if r.session_id == wanted]
        title += f" (session {wanted})"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(records, title), encoding="utf-8")
    print(f"{len(records)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
