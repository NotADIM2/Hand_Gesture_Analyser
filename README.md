# Hand_Gesture_Analyser

Team_Project. A real-time hand gesture analyser: it reads your webcam, finds your
hands with MediaPipe, recognises 17 hand poses plus 8 moving and two-hand
gestures (25 in total) with geometric rules, and keeps a gesture history in SQLite.

## Architecture

```
USER → WEBCAM → IMAGE PROCESSING (OpenCV) → HAND DETECTION (MediaPipe)
     → FEATURE EXTRACTION → GESTURE ANALYZER ─┐
                          → MOTION ANALYZER ──┴→ RESULT / UI + GESTURE HISTORY (SQLite)
```

| Step | Module | Does |
|---|---|---|
| Configuration | `config.py` | All constants and thresholds |
| Webcam + image processing | `camera.py` | Capture, resize, mirror, BGR→RGB, camera errors |
| Hand detection | `hand_detector.py` | MediaPipe Hand Landmarker: up to 2 hands, 21 landmarks, Left/Right |
| Feature extraction | `feature_extractor.py` | Landmarks → extended fingers, joint angles, distances, directions |
| Gesture analyzer | `gesture_analyzer.py` | Features → gesture (rule-based), smoothed over recent frames |
| Motion analyzer | `motion_analyzer.py` | Palm paths over time → swipes, wave, circle; both hands → clap, heart |
| Gesture history | `database.py` | SQLite, written from a background thread |
| Result / UI | `ui.py` | Skeleton, result panel, history and feature panels, keys |
| Entry point | `app.py` | Connects the modules in pipeline order |

Feature extraction only sees landmark arrays (never frames), and the analyzer only
sees features (never landmarks), so a trained ML classifier can replace the rules
later by implementing `classify(features) -> str`.

## Setup (Windows, Python 3.12)

```powershell
git clone https://github.com/NotADIM2/Hand_Gesture_Analyser.git
cd Hand_Gesture_Analyser
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/download_model.py
```

## Run

```powershell
python app.py                 # default webcam
python app.py --camera 1      # another webcam
python app.py --no-history    # don't save to the database
```

| Key | Action |
|---|---|
| `q` / `Esc` | Quit (closing the window also works) |
| `h` | Show/hide gesture history |
| `f` | Show/hide the feature panel (for tuning) |
| `c` | Clear all saved history |

History is saved in `data/gesture_history.db`, one row each time a new gesture
becomes stable (not once per frame).

## Gestures

| Gesture | Rule |
|---|---|
| OK SIGN | Thumb and index tips touching; middle, ring and pinky extended |
| PINCH | Thumb and index tips touching; index half bent; middle, ring and pinky folded |
| THUMBS UP | Only the thumb extended, pointing up and above the rest of the hand |
| THUMBS DOWN | Only the thumb extended, pointing down and below the rest of the hand |
| OPEN PALM | All five fingers extended |
| FOUR FINGERS | Index, middle, ring and pinky extended; thumb folded |
| FIST | No fingers extended |
| POINT LEFT | Only the index finger extended, pointing left on screen |
| POINT RIGHT | Only the index finger extended, pointing right on screen |
| ONE FINGER | Only the index finger extended, pointing up or down |
| L SIGN | Thumb and index extended |
| PEACE | Index and middle extended and spread apart (≥ 15°) |
| TWO FINGERS | Index and middle extended and held together |
| THREE FINGERS | Index, middle and ring extended |
| ROCK | Index and pinky extended |
| I LOVE YOU | Thumb, index and pinky extended |
| CALL ME | Thumb and pinky extended |

Rules are checked in this order. Anything else is shown as UNKNOWN.

### Moving and two-hand gestures

These follow the palm over the last second or two, so they work with any hand shape.
Each one appears in a banner at the bottom of the window, is saved to the history once,
and the pink line on screen shows the path the palm has taken.

| Gesture | How |
|---|---|
| SWIPE LEFT / RIGHT / UP / DOWN | Move your hand quickly and fairly straight across at least a quarter of the frame |
| WAVE | Move your hand left and right about twice, quickly |
| CIRCLE | Draw a full circle in the air with your hand |
| CLAP | Bring both hands together from well apart |
| HEART | Both hands: index tips touching at the top, thumb tips touching below, wrists apart; hold it briefly |

### Why no confidence percentage?

Rules don't produce a probability, so the UI doesn't invent one. Instead it shows
how many of the last 10 frames agreed on the gesture:

- **STABLE**: at least 7 of the last 10 frames agree.
- **UNCERTAIN**: recent frames disagree.
- **UNRECOGNISED**: a hand is visible but matches no gesture.

## Testing

```powershell
python -m pytest
```

The tests don't need a webcam. They use synthetic hands plus real landmarks taken
from MediaPipe's public sample photos (`tests/fixtures/sample_hands.json`).

Each module can also be checked on its own with the live camera:

```powershell
python camera.py              # live feed + FPS
python hand_detector.py       # skeleton + Left/Right
python feature_extractor.py   # extended fingers + thumb direction
```

### Testing with your own hands

```powershell
python scripts/record_samples.py
```

Hold a gesture and press its key (shown on screen) to save it. Samples go to
`tests/fixtures/recorded_hands.json` and are checked by `pytest`. Commit them so
every teammate's hands are covered.

## Tuning

All thresholds are in `config.py`. Press `f` in the app to see the measured values
live, then adjust:

- `FINGER_STRAIGHT_ANGLE`: lower it if relaxed open fingers count as folded.
- `THUMB_EXTENDED_LATERAL`: raise it if a tucked thumb counts as extended.
- `OK_TOUCH_RATIO`: raise it if the OK sign or Pinch isn't detected.
- `PINCH_MIN_INDEX_ANGLE`: lower it if a Pinch shows as Fist; raise it if a Fist shows as Pinch.
- `PEACE_MIN_SPREAD`: the angle that separates Peace from Two Fingers.
- `STABILITY_WINDOW` / `STABILITY_MIN_AGREEMENT`: smoothing.
- `SWIPE_MIN_DISTANCE` / `SWIPE_MAX_SECONDS`: lower the distance or allow more time if
  swipes are missed; the opposite if ordinary movement triggers them.
- `WAVE_MIN_SWINGS` / `WAVE_MIN_AMPLITUDE`, `CIRCLE_MIN_TURN`: how much waving or circling counts.
- `CLAP_START_DISTANCE`, `HEART_TOUCH_DISTANCE`: two-hand gestures, in palm sizes.

## Troubleshooting

- **"Could not open camera"**: close other apps that use the webcam (Zoom, Teams,
  browser tabs), or try `--camera 1`.
- **Left/Right swapped**: `MIRROR_FRAME` in `config.py` must stay `True`, because
  MediaPipe's labels assume a mirrored (selfie) image.
- **`INFO`/`W0000` lines at start-up**: MediaPipe's own log messages. They are
  harmless and can be ignored.
- **Old tutorials use `mp.solutions.hands`**: that module was removed in current
  MediaPipe. This project uses the Tasks API (`mediapipe.tasks.python.vision`).
