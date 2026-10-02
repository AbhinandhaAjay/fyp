# Input Processing (Modules 1 & 3 input stage)

Records the webcam, keyboard and mouse/touchpad in 15–30 s windows. Each window is
uploaded once and processed in the background into:

```
                       one window (15 s)
              ┌───────────────┴───────────────┐
           webcam                  keyboard + pointer + focus events
              ↓                                 ↓   (buffered in the browser,
   frames @10 fps → MediaPipe                   ↓    sent with the window)
   Face Landmarker → gaze + head pose     behaviour features
              ↓                                 ↓
   sequence_fixed 150 × 29 + mask         behavior 1 × 41
```

## Run
```
pip install -r requirements.txt
python app.py
```
Open http://localhost:8000. Click **Start camera**, then **Start recording**.
The model file is `models/face_landmarker.task`. If it's missing, download it from
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

## Per-frame vector (29 dims)
| group | features |
|---|---|
| presence | `face_present` (0 → all other values are 0) |
| head pose | `head_yaw`, `head_pitch`, `head_roll` (deg), `head_tx/ty/tz` (cm, from the face transform matrix) |
| iris gaze | `l_iris_h/v`, `r_iris_h/v` (0.5 = centred in the eye), `gaze_h/v` (averaged, centred at 0) |
| eye openness | `l_ear`, `r_ear` (eye aspect ratio) |
| eye blendshapes | `bs_eyeBlink*`, `bs_eyeLook{In,Out,Up,Down}*`, `bs_eyeSquint*`, `bs_eyeWide*` |

## Screen gaze (calibration)
Click **Calibrate gaze** (camera on, not recording). A dot visits 9 positions, then 4
held-out test positions, ~2 s each, while the webcam records (`static/calibration.js`).
The server (`processing/calibration.py`) runs MediaPipe on the clip, drops the first
0.7 s at each dot (eyes still moving) and blinks, and fits a ridge regression
(numpy, strength chosen by leave-one-dot-out error) from 20 per-frame features
(iris position, gaze, eye-look blendshapes, head pose and position) to
`(screen_x, screen_y)`: fractions of the browser window, (0,0) top-left, (1,1)
bottom-right; values outside 0..1 are looks off the page.

Accuracy is measured only on the 4 test dots, and rated against always guessing the
centre (`skill` = 1 - error / baseline): **good** needs skill ≥ 0.6 and error < 10% of
the window diagonal, **fair** skill ≥ 0.35 and < 18%. Expect fair-to-good with a
webcam: enough for regions of the page (passage vs answer box vs off-page), not
single words. Calibrations are saved in `data/calibrations/<id>.json` (+ clip).

Recordings started after accepting a calibration get, per window:
`gaze_screen` (N×2) + `gaze_screen_valid` on the model-input grid, `gaze_screen_raw`
for raw frames, `calibration_id`, and summary fields `screen_x/y_mean`,
`screen_x/y_std`, `offscreen_ratio_calibrated`. The session folder keeps a copy as
`calibration.json`. Don't move or resize the browser window after calibrating; the
page warns and stops using a calibration whose window size no longer matches.

```python
X, mask, names = load_window(path, screen_gaze=True)   # (150, 32): 29 + screen_x, screen_y, screen_valid
```

## Areas of interest (needs calibration)
Mark page regions with `data-aoi="name"` (here: `passage`, `answer`, `quiz`).
`behavior.js` records their rectangles (fractions of the viewport) at each window
start, every 0.5 s and after scrolling/resizing, so regions that move with the page
are tracked. `processing/aoi.py` labels every calibrated gaze point with the region
it falls in (or within 0.03 of, to allow for webcam error), else `other` (on the page)
or `offpage`. Runs shorter than 2 frames are merged into the previous label so gaze
jitter doesn't count as switching.

Per window (`load_aoi(path)` -> vector, names, has_aoi, detail):

| feature | meaning |
|---|---|
| `on_aoi_ratio`, `other_ratio`, `offpage_ratio` | share of gaze time on regions / elsewhere on the page / off the page |
| `aoi_visits`, `aoi_switches`, `aoi_revisits` | separate visits, moves between different regions, returns to a region already visited |
| `mean_visit_s`, `longest_visit_s` | how long attention stayed on a region |
| `aoi_entropy` | 0 = all time on one region, 1 = spread evenly across regions |

`detail` has per-region `dwell` (share of time) and `visits`, since region names
depend on the page. The `.npz` also has `aoi_labels` (per raw frame),
`aoi_labels_fixed` (per model-input step) and `aoi_snapshots`. Regions scrolled out
of view can't be looked at: looks there count as `offpage`.

## Behaviour vector (41 dims)
Captured by `static/behavior.js` (drop it into any page; call `Behavior.start()` /
`Behavior.cut()` per window). Computed in `processing/behavior.py`.

| group | features | what it can indicate |
|---|---|---|
| keyboard (17) | `key_count`, `keys_per_sec`, `typing_speed_burst`, `typing_active_ratio`, `iki_mean_s`, `iki_std_s`, `dwell_mean_s`, `dwell_std_s`, `backspace_count`, `delete_count`, `correction_ratio`, `max_correction_run`, `correction_episodes`, `enter_count`, `pause_count`, `longest_key_pause_s`, `held_key_repeats` | errors / re-typing (corrections), hesitation (pauses), irregular rhythm and long key holds (stress, fatigue) |
| pointer (16) | `move_events`, `distance_norm`, `speed_mean`, `speed_std`, `speed_max`, `direction_changes`, `path_straightness`, `pointer_active_ratio`, `longest_pointer_idle_s`, `click_count`, `rapid_click_count`, `scroll_gestures`, `scroll_distance_norm`, `scroll_reversals`, `zoom_gestures`, `touch_ratio` | searching / jittery movement, repeated frustrated clicks, re-reading (scroll reversals), inactivity |
| focus (3) | `tab_switches`, `hidden_ratio`, `window_blur_count` | distraction, leaving the task |
| spelling (5) | `words_typed`, `misspelled_words`, `misspelling_ratio`, `misspellings_fixed`, `misspellings_remaining` | spelling difficulty, noticing and fixing own errors vs leaving them |

**Mouse vs touchpad vs touchscreen:** browsers report a touchpad as `"mouse"`, so the
two can't be told apart. Scrolling is therefore measured from how far content actually
moved, grouped into gestures (events < 0.3 s apart, split when direction flips). A
mouse wheel's few big notches, a touchpad's many tiny events plus coasting, a
touchscreen swipe, keyboard scrolling and scrollbar drags all count the same.
Touchpad pinch (Ctrl+wheel) is counted as `zoom_gestures`, not scrolling. Pointer
movement speed still differs by device; compare each student to their own baseline.

**Spelling** (`static/spelling.js` + `static/spell-worker.js`): a Hunspell spell
checker (Typo.js, en_US + en_GB dictionaries, vendored in `static/vendor/`) runs in a
Web Worker in the student's browser. Words are judged when finished (space,
punctuation, Enter, or leaving the box); the word still being typed is not judged.
A word is correct if either dictionary accepts it ("colour" and "color" both pass).
Only the five counts are sent. Pasted text is a baseline and never counts as the
student's errors. `has_spelling` in the `.npz` is False when the checker wasn't
ready; the spelling values are then zeros, not "no errors".
Limits: English only; real-word mistakes ("wen" for "when", "there" for "their")
pass; every word is checked, including capitalised and ALL-CAPS ones, so names
are flagged unless they appear in the lesson text (elements marked
`data-spell-vocab`) or are added with `Spelling.allow(["Priya", ...])`; missing
apostrophes ("dont") count as errors on purpose. The browser's own red
underline also shows on the practice box; turn it off (`spellcheck="false"`) if you
want to measure whether students notice errors unaided.

**Privacy:** keys are stored only as a category (`char`, `bs`, `del`, `enter`, `space`,
`nav`, `tab`, `mod`, `other`) with timings. No key identity or typed text is stored,
and password fields are skipped entirely. Pointer positions are viewport coordinates;
no screenshots.

**Limit:** events are only seen inside the page that runs `behavior.js`. Time spent in
another tab or app shows up as `tab_switches` / `hidden_ratio` / `window_blur_count`,
not as typing or clicks.

## Outputs per window
- `window_NNNN.webm`: the raw clip
- `window_NNNN.npz`:
  - `sequence_fixed` (N×29) + `mask` (N): **the model input**. N = window_seconds × fps
    (150 for 15 s at 10 fps), always the configured length, even for a shorter final clip.
    Values are interpolated by timestamp onto a uniform 100 ms grid (`grid_ms`).
    `mask` = 1 where real frames back the step, 0 for padding (zeros, never a repeated
    last frame) and for gaps > 250 ms. A step with no face is still observed
    (mask 1, `face_present` 0).
  - `sequence` (T×29) + `timestamps_ms`: the raw frames at their real, uneven times
  - `feature_names`, `feature_groups` (JSON), `fps`
  - `behavior` (41,), `behavior_names`, `behavior_groups` (JSON), `has_behavior`
    (False → no events were sent and the vector is zeros)
- `window_NNNN_events.json`: the anonymised raw events, so features can be recomputed
- `window_NNNN.json`: window summary: face presence, off-screen ratio, head pose
  mean/std, blinks, approximate fixations, gaze transitions, observed/padded steps

```python
from processing.dataset import load_window, load_session
X, mask, names = load_window("data/sessions/<id>/window_0000.npz")   # (150, 29), (150,)
X, M, names = load_session("data/sessions/<id>")                      # (W, 150, 29), (W, 150)

from processing.dataset import load_behavior, load_session_behavior
b, names, recorded = load_behavior("data/sessions/<id>/window_0000.npz")  # (41,)
kb, names, _ = load_behavior(path, groups=("keyboard",))                 # (17,)
B, names = load_session_behavior("data/sessions/<id>")                   # (W, 41)
```

## Feature ablation
Keep all 29 features, then compare which signals actually help:

| preset | groups | dims |
|---|---|---|
| `A_head` | presence + head pose | 7 |
| `B_gaze_head` | + iris gaze | 13 |
| `C_gaze_head_eye` | + eye openness (EAR) | 15 |
| `D_all` | + eye blendshapes | 29 |

```python
X, mask, names = load_window(path, ablation="B_gaze_head")
```

Thresholds for off-screen, blinks and fixations are at the top of
`processing/features.py`. They are heuristics. Calibrate them per user and setup.
