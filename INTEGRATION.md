# Input Processing: integration spec

Handoff for integrating this project into the main platform ("Personalised Learning
Platform for Students with Special Needs"). It covers what the system does, every
input and output, every feature, the APIs, the requirements, and the known gaps.
`README.md` is the shorter user-facing summary; this file is the full contract.

---

## 1. What this is

The **input-processing stage** of the platform's architecture. It covers:

- **Module 1, Eye-Gaze and Attention Analysis** (input side): per-frame gaze and head
  pose from the webcam, calibrated screen gaze, areas-of-interest attention metrics.
- **Module 3, Learner Interaction Behaviour Analysis** (input side): keyboard,
  mouse/touchpad/touch, scrolling, tab focus, spelling.

It **does not** contain: facial affect / emotion recognition (Module 2), cognitive-state
estimation or any ML model (Modules 4–5), the cognitive twin, adaptation, or a
database. Its outputs are the **feature tensors those modules consume**.

**Core idea:** the session is cut into fixed **windows** (15–30 s, default 15 s). For each
window the browser uploads one webcam clip plus that window's anonymised interaction
events. The server turns them into:

```
                             one window (15 s)
         ┌─────────────────────────┼──────────────────────────┐
      webcam clip          keyboard/pointer/focus       (if calibrated) gaze
         │                 + spelling counts             mapped to the page
         ▼                         ▼                             ▼
  MediaPipe Face Landmarker   behaviour features         screen gaze + areas of interest
         ▼                         ▼                             ▼
  sequence_fixed 150×29      behavior 1×41               gaze_screen 150×2 (+valid)
  + mask 150                                             aoi_features 1×9 (+per-region detail)
```

Everything runs locally: Python (FastAPI) server + plain JavaScript in the page.
No cloud services, no AI/LLM calls, no Chrome extension.

---

## 2. Requirements

### Server
- **Python 3.10** (developed and tested on 3.10.11, Windows 11).
- `pip install -r requirements.txt`: `fastapi`, `uvicorn`, `python-multipart`,
  `mediapipe>=0.10.14`, `opencv-python`, `numpy`.
  - Tested versions: mediapipe 1.0.1, opencv 5.0.0, numpy 2.2.6, fastapi 0.122, uvicorn 0.38.
  - Installing mediapipe upgraded numpy to 2.x and OpenCV to 5.x; use a separate
    virtualenv if the platform pins older versions.
  - OpenCV must be built with FFmpeg (the pip wheel is) to decode browser WebM/VP8.
- **Model file:** `models/face_landmarker.task` (MediaPipe Face Landmarker, float16,
  ~3.7 MB). Present in the repo. Source:
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
- CPU only. Processing a 15 s window takes ~3–12 s on a laptop CPU (2 worker threads),
  i.e. faster than real time.
- Run: `python app.py` → serves on `http://127.0.0.1:8000` (API + static page).

### Browser (student side)
- **Chromium-based browser (Chrome / Edge) recommended**; that's what was tested.
  Uses: `getUserMedia` (camera), `MediaRecorder` with WebM (VP8 preferred; falls back
  to VP9 / generic webm / mp4), Web Workers, `PointerEvent`, `performance.now()`.
  Firefox should work but is untested; Safari's MediaRecorder produces MP4 (the server
  accepts `.mp4` if the content type says so, untested).
- **Webcam** and the student's permission to use it.
- **Secure context:** `getUserMedia` only works on `https://` or `http://localhost` /
  `127.0.0.1`. When the platform is deployed on a real domain it **must be HTTPS**.
- **No Chrome extension is needed.** All capture is done by JavaScript running in the
  platform's own pages. (The "Claude in Chrome" extension mentioned during development
  was only a testing tool and is not part of the system.) An extension would only be
  required if you wanted to capture keyboard/mouse activity **outside** the platform
  (other tabs/sites); that is not built. Leaving the page is still detected
  (tab switches, window blur, hidden time).

### Third-party code bundled (offline, no CDN)
- `static/vendor/typo/typo.js`: Typo.js 1.3.2 (BSD-3-Clause), Hunspell spell checker in JS.
- `static/vendor/dictionaries/en_US/*` and `en_GB/*`: Hunspell dictionaries from npm
  `dictionary-en@4.0.0` / `dictionary-en-gb@3.0.0` (MIT AND BSD), ~550 KB each, with licenses.

---

## 3. File map

| Path | Role |
|---|---|
| `app.py` | FastAPI server: REST API, background processing, static file serving |
| `processing/features.py` | Video → frames → MediaPipe → 29 per-frame features; resampling + mask; window summary |
| `processing/behavior.py` | Interaction events → 41 behaviour features |
| `processing/calibration.py` | Calibration fitting (ridge regression), applying it, accuracy rating, storage |
| `processing/aoi.py` | Areas of interest: label gaze with page regions → 9 features + per-region detail |
| `processing/dataset.py` | Loaders for training: `load_window`, `load_session`, `load_behavior`, `load_session_behavior`, `load_aoi` |
| `show_vectors.py` | CLI: print a window's vectors (`python show_vectors.py [file.npz or video]`) |
| `static/index.html` | Demo/test page: camera, recording loop, calibration button, practice activity, results UI |
| `static/behavior.js` | Browser collector: keyboard, pointer, wheel, scroll, focus, AOI rectangles (reusable) |
| `static/spelling.js` | Browser spelling tracker for text fields (reusable) |
| `static/spell-worker.js` | Web Worker running Typo.js with US + UK dictionaries |
| `static/calibration.js` | Calibration overlay UI (reusable): `Calibration.run(stream)` |
| `models/face_landmarker.task` | MediaPipe model |
| `data/sessions/<sid>/` | Per-session outputs (created at runtime) |
| `data/calibrations/<cid>.json/.webm` | Saved calibrations (created at runtime) |

---

## 4. End-to-end flow

1. **Camera on:** page calls `getUserMedia({video: {width: 640, height: 480, frameRate: 30}})` (ideal values).
2. **Optional: calibrate gaze:** `Calibration.run(stream)` shows 13 dots, records the
   webcam, POSTs to `/api/calibrations`, shows accuracy; the user accepts → page keeps
   the calibration object (in memory only; lost on reload).
3. **Start recording:** `POST /api/sessions` with `{"calibration_id": <id or null>}` →
   `session_id`. Page calls `Behavior.start()`.
4. **Recording loop** (page logic in `index.html`, must be replicated by the platform):
   - Each window uses a **new `MediaRecorder`** so every clip is a standalone,
     decodable file (`videoBitsPerSecond: 2_000_000`, `start(1000)`).
   - When the window length elapses: `events = Behavior.cut()` (returns this window's
     events and starts the next window's buffer at the same instant), stop that
     recorder, immediately start the next recorder.
   - On the old recorder's `stop`: `POST /api/sessions/{sid}/windows` with the clip +
     `events` JSON. One request per window; nothing is sent per keystroke.
   - On stop: `Behavior.stop()` (flushes the final partial window). Final fragments
     shorter than 2 s are discarded.
5. **Server** saves the clip + events, computes behaviour features immediately, then
   queues video processing on a thread pool (2 workers). Status goes
   `queued → processing → done | error`.
6. **Results:** poll `GET /api/sessions/{sid}/windows` (the page polls every 1.5 s), or
   read files from `data/sessions/<sid>/`.

Alignment: video time 0 = recorder start; event time 0 = `Behavior.cut()`; these happen
within the same JS task, so they line up to within tens of milliseconds.

---

## 5. REST API

All JSON unless noted. No authentication (see §13).

### `POST /api/sessions`
Body (optional JSON): `{"calibration_id": "a1b2c3d4e5f6" | null}`
Response:
```json
{"session_id": "500c6f4efaf9", "target_fps": 10.0,
 "feature_names": ["face_present", "..."],            // 29 per-frame feature names
 "behavior_features": ["key_count", "..."],           // 41 behaviour feature names
 "calibration_id": "a1b2c3d4e5f6" | null}
```
If the calibration id exists, a copy is written to `data/sessions/<sid>/calibration.json`
and every window of the session gets screen gaze + AOI outputs.
The demo page compares `behavior_features` with its own list to detect a server
running stale code and warns the user.

### `POST /api/sessions/{sid}/windows`  (multipart/form-data)
| field | type | meaning |
|---|---|---|
| `video` | file | the window's clip (WebM; `.mp4` if content type contains "mp4") |
| `index` | int | window number, 0-based |
| `window_seconds` | float | **configured** window length → number of fixed rows (`round(window_seconds × 10)`) |
| `recorded_seconds` | float | actual clip length (last window is usually shorter) |
| `started_at` | float | window start, Unix epoch seconds (informational) |
| `events` | string (JSON) | behaviour events, see §7.1; may be empty → no behaviour data |

Response: `{"index": 0, "status": "queued"}`. 404 if the session id is unknown
(e.g. server restarted, see §13).

### `GET /api/sessions/{sid}/windows`
Array of window status objects (in memory), each:
```json
{"index": 0, "status": "queued|processing|done|error", "error": "...",
 "started_at": 1790000000.0, "window_seconds": 15, "recorded_seconds": 15.0, "bytes": 2345710,
 "behavior": {"key_count": 23.0, "...": 0.0} | null,     // 41 features, null if no events
 "has_spelling": true,
 "processing_seconds": 3.04,
 "shape": [150, 29],           // raw frames × features
 "shape_fixed": [150, 29],     // model input steps × features
 "observed_steps": 150,        // mask sum
 "summary": { ... },           // §6.4 (+ §9.3 screen fields if calibrated)
 "preview": {"t": [...], "face": [...], "yaw": [...], "pitch": [...], "gaze_h": [...], "gaze_v": [...]},
 // only if the session is calibrated:
 "calibration": {"id": "...", "quality": "good|fair|poor", "error_px": 84, "viewport": [1920, 937]},
 "screen_points": [[x, y, region_or_null], ...],          // ≤150 points for the gaze map
 "aoi": { 9 AOI features, "dwell": {...}, "visits": {...}, "names": [...], "rects": {...} }  // §10
}
```

### `GET /api/sessions/{sid}/windows/{index}/npz`
Downloads `window_NNNN.npz` (§11).

### `GET /api/sessions/{sid}/windows/{index}/vectors?kind=fixed|raw`
For display/debugging. `fixed` (default): `{"kind", "feature_names", "timestamps_ms"
(grid), "rows" (N×D), "mask"}`; `raw`: real frames, no mask. If calibrated, two extra
columns `screen_x`, `screen_y` are appended. Old windows without a mask fall back to `raw`.

### `POST /api/calibrations`  (multipart/form-data)
| field | type | meaning |
|---|---|---|
| `video` | file | calibration clip (WebM) |
| `points` | string (JSON) | `[{"x": 0.08, "y": 0.5, "t0": 1234, "t1": 3034, "phase": "cal"|"val"}, ...]`, x/y = viewport fractions, t0/t1 = ms from recorder start |
| `viewport` | string (JSON) | `[innerWidth, innerHeight]` in CSS px |

Processing runs synchronously (~5–15 s). Response = the saved calibration (§9.2).
**422** with a message if too few dots had usable frames (face not found / eyes closed).

### `GET /api/calibrations/{cid}`
Returns a saved calibration (§9.2) or 404.

### Static
`/` → `static/index.html`; all files in `static/` served at the root
(`/behavior.js`, `/spelling.js`, `/spell-worker.js`, `/calibration.js`, `/vendor/...`).

---

## 6. Video features (Module 1 input)

### 6.1 Pipeline (`processing/features.py`)
1. Decode the clip with OpenCV. Browser WebM often has unreliable timestamps; if a
   frame's timestamp doesn't increase, it is replaced by `index / nominal_fps`
   (nominal 30 if unknown).
2. **Sample at 10 fps on a fixed grid** (the next target time advances by exactly
   100 ms; a frame is taken if within half a source-frame period of the target), so
   sampling error doesn't accumulate.
3. Each sampled frame → **MediaPipe Face Landmarker** (VIDEO mode, 1 face, 478 landmarks
   incl. iris, blendshapes on, facial transformation matrix on). A fresh landmarker is
   created per window.
4. → 29-number vector per frame (§6.2) → raw sequence `(T, 29)` with real timestamps.
5. → **resampled model input** `(N, 29)` on a uniform 100 ms grid + `mask (N,)` (§6.3).
6. → window summary (§6.4).

### 6.2 Per-frame vector: 29 features (order is fixed)

| # | name | meaning / units |
|---|---|---|
| 0 | `face_present` | 1 if a face was detected, else 0 (then **all other values are 0**) |
| 1 | `head_yaw` | head turn left/right, degrees (from the facial transformation matrix) |
| 2 | `head_pitch` | head nod up/down, degrees |
| 3 | `head_roll` | head tilt, degrees |
| 4–6 | `head_tx`, `head_ty`, `head_tz` | head translation from the transform matrix (MediaPipe canonical-face units ≈ cm; `tz` ≈ distance, negative) |
| 7, 8 | `l_iris_h`, `l_iris_v` | left iris position inside the eye opening: horizontal along the corner-to-corner axis, vertical between the lids; 0..1, 0.5 = centred |
| 9, 10 | `r_iris_h`, `r_iris_v` | same, right eye |
| 11 | `gaze_h` | mean of both `iris_h` − 0.5 (eye-in-head horizontal direction, ≈ −0.5..0.5) |
| 12 | `gaze_v` | mean of both `iris_v` − 0.5 |
| 13, 14 | `l_ear`, `r_ear` | eye aspect ratio (6-point) = eye openness |
| 15–28 | `bs_eyeBlinkLeft`, `bs_eyeBlinkRight`, `bs_eyeLookDownLeft`, `bs_eyeLookDownRight`, `bs_eyeLookInLeft`, `bs_eyeLookInRight`, `bs_eyeLookOutLeft`, `bs_eyeLookOutRight`, `bs_eyeLookUpLeft`, `bs_eyeLookUpRight`, `bs_eyeSquintLeft`, `bs_eyeSquintRight`, `bs_eyeWideLeft`, `bs_eyeWideRight` | MediaPipe eye blendshape scores, 0..1 |

"Left/right" eye = the subject's own. Landmark indices are in `features.py`
(e.g. iris centres 468/473). These are **directions**, not screen coordinates; screen
position needs calibration (§9). The sign convention of yaw/pitch has not been
verified against real head movement.

Column groups (`FEATURE_GROUPS`) and ablation presets (`ABLATIONS`) for experiments:

| preset | groups | dims |
|---|---|---|
| `A_head` | presence + head | 7 |
| `B_gaze_head` | + iris/gaze | 13 |
| `C_gaze_head_eye` | + eye openness (EAR) | 15 |
| `D_all` | + blendshapes | 29 |

### 6.3 Model input: fixed length + mask
- `N = round(window_seconds × 10)`: **150 for 15 s**, always the configured length,
  even when the clip is shorter (last window).
- Grid times `grid_ms = [0, 100, 200, ...]`.
- For each grid time: if real frames exist on both sides within **250 ms**, values are
  **linearly interpolated by timestamp**; if either side has no face, the nearest frame is
  copied instead (never blend "face" with "no face"). Else, a real frame within 50 ms is
  copied. Else → **zeros and `mask = 0`** (padding). The last frame is **never repeated**
  to fill the end.
- `mask = 1` means the step is backed by real frames. A step with no face detected is
  still a real observation: `mask = 1, face_present = 0` (the learner being absent is
  information).

### 6.4 Window summary (`summary` in status/JSON)
| key | meaning |
|---|---|
| `face_presence_ratio` | share of sampled frames with a face |
| `head_yaw/pitch/roll_mean`, `_std` | over face frames |
| `gaze_h_mean/_std`, `gaze_v_mean/_std` | over face frames |
| `off_screen_ratio`, `off_screen_episodes` | **uncalibrated heuristic**: off-screen if no face, \|yaw\| ≥ 25°, \|pitch\| ≥ 20°, or \|gaze_h\| ≥ 0.18 |
| `blink_count`, `blink_rate_per_min` | rising edges of mean eyeBlink blendshape > 0.5 |
| `ear_mean` | mean eye aspect ratio |
| `fixation_count`, `fixation_mean_duration_s` | approximate: runs ≥ 200 ms where combined (gaze + head/90) velocity < 0.6 units/s |
| `gaze_transitions`, `gaze_velocity_mean` | frames above / mean of that velocity |
| `frames_decoded`, `frames_processed` | decoder / sampled frame counts |
| `grid_steps`, `observed_steps`, `padded_steps` | mask statistics |
| + `screen_*`, `offscreen_ratio_calibrated` | only if calibrated (§9.3) |

Thresholds are constants at the top of `features.py` and are uncalibrated heuristics.

---

## 7. Interaction capture (Module 3 input), browser side

### 7.1 `static/behavior.js`: `window.Behavior`
```js
Behavior.start()        // begin capturing; resets buffers
Behavior.cut()  -> ev   // return this window's events, start the next window
Behavior.stop() -> ev   // return the final window's events, stop capturing
Behavior.live() -> {keys, corrections, clicks, scrolls, zooms, tab_switches,
                    words, misspelled, spell_status}   // running counts for UI
```
Listens on the whole document (capture phase). Event object sent as the `events` field:
```js
{
  "duration_ms": 15006,
  "viewport": [innerWidth, innerHeight],
  "initially_hidden": false,
  "keys":    [[t_ms, "d"|"u", category, x], ...],
       // category: char | bs | del | enter | space | tab | nav | mod | other
       // x on "d" = 1 if auto-repeat else 0; on "u" = hold time (ms)
  "pointer": [[t_ms, "m"|"d", clientX, clientY, "m"|"t"|"p", button], ...],
       // m = move (throttled to ~60 Hz), d = pointerdown; m/t/p = mouse/touch/pen
  "wheel":   [[t_ms, dx_px, dy_px, ctrl], ...],   // raw wheel; ctrl=1 = pinch/Ctrl+wheel zoom
  "scroll":  [[t_ms, dx_px, dy_px], ...],         // actual content movement of any scroller
  "focus":   [[t_ms, "hidden"|"visible"|"blur"|"focus"], ...],
  "aoi":     [[t_ms, {"passage": [[x0,y0,x1,y1], ...], ...}], ...],  // §10
  "spell":   {"ready": true, "words": 8, "misspelled": 2, "fixed": 1, "remaining": 1} | null
}
```
- **Privacy:** keys are stored only as a category + timing, never the key or text.
  Events whose target is `input[type=password]` are ignored entirely. No screenshots.
- **Mouse vs touchpad:** browsers report both as `pointerType "mouse"`; they can't be
  told apart. Scrolling is therefore measured from **actual content movement** (scroll
  events from any scroll container: wheel, touchpad two-finger, touchscreen swipe,
  keyboard, scrollbar drag), so devices are comparable. Touchpad pinch arrives as
  Ctrl+wheel and is counted as zoom, not scroll. Scroll positions are seeded on
  pointerdown/wheel/keydown so the first scroll has a starting point.
- **Scope:** only the document running the script is captured (not iframes, other
  tabs or other apps). Leaving is captured via focus events.

### 7.2 `static/spelling.js` + `static/spell-worker.js`: `window.Spelling`
```js
Spelling.status   // "loading" | "ready" | "unavailable"
Spelling.take()   // -> {ready, words, misspelled, fixed, remaining}; resets counters
                  //    (Behavior.cut() calls this and puts the result in ev.spell)
Spelling.live()   // -> {words, misspelled, fixed}
Spelling.allow(["Priya", ...])   // accept extra words (names, subject terms)
```
- Tracks `<textarea>` and `<input type="text|search">` only. **Not** `contenteditable`
  / rich-text editors (would need adapting).
- A word is judged when **finished**: followed by space/punctuation/Enter, or when the
  field loses focus. The word under the caret is never judged. Words < 2 letters skipped.
  Also re-checked after 1 s of no typing (to notice fixes).
- Checked locally in a Web Worker with Typo.js; a word is correct if **either** the US or
  the UK dictionary accepts it (as typed or lower-cased), or it's in the allow list:
  built-in (`ok, wifi, youtube, covid, tv, pc, eg, ie, lol, btw, pls, app, apps, online,
  email`) + all words in the page's lesson text (elements marked `data-spell-vocab`, or
  the whole page if none are marked; rendered text only, never field contents) +
  `Spelling.allow(...)`.
- Every finished word is checked, including Capitalised and ALL-CAPS ones.
- Counting is a per-field diff of misspelled-word multisets: a misspelling appearing =
  `misspelled`; one disappearing = `fixed` (corrected or deleted); misspellings present
  at window end = `remaining`. **Pasted/dropped text and text already in the field on
  first focus are a baseline** and never count as the student's errors.
- Only counts leave the browser; words are never sent.
- Loading: ~1 s, ~50 MB in the worker. `spell-worker.js` is loaded with a **relative
  URL**; see §12.

### 7.3 Behaviour vector: 41 features (`processing/behavior.py`)
Computed server-side from the events at upload. Durations use the window's actual
`duration_ms`.

**Keyboard (17)**: modifier keys excluded everywhere
| feature | definition |
|---|---|
| `key_count` | key presses, auto-repeat excluded |
| `keys_per_sec` | key_count / window duration |
| `typing_speed_burst` | 1 / mean inter-key interval within bursts (gaps < 2 s) |
| `typing_active_ratio` | share of 1-s slots containing a key press |
| `iki_mean_s`, `iki_std_s` | inter-key interval within bursts (rhythm) |
| `dwell_mean_s`, `dwell_std_s` | key hold time |
| `backspace_count`, `delete_count` | incl. auto-repeat while held |
| `correction_ratio` | deletions / (fresh keys + deletion repeats) |
| `max_correction_run` | longest run of consecutive deletion presses |
| `correction_episodes` | number of separate deletion runs |
| `enter_count` | |
| `pause_count` | inter-key gaps ≥ 2 s |
| `longest_key_pause_s` | longest gap incl. window edges (= window length if no keys) |
| `held_key_repeats` | auto-repeat events |

**Pointer (16)**: distances normalised by viewport diagonal
| feature | definition |
|---|---|
| `move_events` | pointer moves (~60 Hz max) |
| `distance_norm` | path length in viewport diagonals |
| `speed_mean`, `speed_std`, `speed_max` | diagonals/s, between consecutive moves within strokes |
| `direction_changes` | turns > 90° between move segments > 2 px |
| `path_straightness` | mean (straight distance / path length) of strokes (split at 0.3 s pauses, > 0.02 diag); 1 = direct, low = wandering |
| `pointer_active_ratio` | share of 1-s slots with pointer/wheel/scroll activity |
| `longest_pointer_idle_s` | longest gap incl. edges |
| `click_count` | pointerdowns |
| `rapid_click_count` | clicks < 0.6 s after the previous one within 30 px (repeated/frustrated clicking) |
| `scroll_gestures` | scroll movements: events < 0.3 s apart grouped, split on direction flip, ≥ 10 px |
| `scroll_distance_norm` | total scrolled distance in viewport heights |
| `scroll_reversals` | direction changes between gestures (re-reading/searching) |
| `zoom_gestures` | Ctrl+wheel (pinch) gestures |
| `touch_ratio` | share of pointer events from touch |

**Focus (3)**: `tab_switches` (page became hidden), `hidden_ratio` (share of window
hidden), `window_blur_count` (focus left the browser window).

**Spelling (5)**: `words_typed`, `misspelled_words`, `misspelling_ratio`
(= misspelled / max(words, misspelled, 1)), `misspellings_fixed`,
`misspellings_remaining`. All zeros when the checker wasn't ready; check `has_spelling`.

**Zeros are ambiguous**: with no typing, `iki_mean_s = 0`, `typing_speed_burst = 0`
etc. `key_count == 0` / `move_events == 0` identify those windows; add explicit
activity flags or mask those features before training.

Old event files (before scroll capture) fall back to wheel events for scroll features.

---

## 8. Spelling: what it can't do
Real-word errors ("wen", "their/there") pass; names not on the page or allowed are
flagged; English only (Hinglish/other languages flagged); missing apostrophes
("dont", "im") count as errors deliberately. Chrome's own red underline is still
active on the demo textarea; set `spellcheck="false"` for unaided-error studies and
keep it consistent.

---

## 9. Gaze calibration → screen gaze

### 9.1 Procedure (`static/calibration.js`: `await Calibration.run(stream)` → result | null)
- Full-viewport overlay (dark), instructions, Start/Cancel; Esc aborts.
- Records the webcam (own MediaRecorder on the same stream); waits 600 ms after start.
- **9 calibration dots** at x, y ∈ {0.08, 0.5, 0.92} (shuffled), then **4 test dots** at
  (0.25|0.75, 0.25|0.75) (shuffled), **1.8 s each** (~25 s total). Shrinking yellow ring.
- Uploads clip + `points` (with t0/t1 relative to recorder start) + viewport to
  `/api/calibrations`; shows a target-vs-prediction map, error, verdict; buttons
  Use / Redo / Cancel. Returns the calibration JSON if accepted.
- The demo page disables calibration while recording, stores the accepted calibration
  in memory, and **stops using it if the window size changes** (> 4 px).

### 9.2 Fitting (`processing/calibration.py`) and saved calibration JSON
- Decode at 15 fps; per dot keep frames from `t0 + 700 ms` to `t1 − 100 ms` with a face
  and not blinking (mean eyeBlink ≤ 0.5). Needs ≥ 3 frames per dot and ≥ max(5, 9 − 2)
  usable calibration dots, else 422.
- Inputs (20): `gaze_h, gaze_v, l_iris_h, l_iris_v, r_iris_h, r_iris_v, head_yaw,
  head_pitch, head_roll, head_tx, head_ty, head_tz, bs_eyeLook{Down,In,Out,Up}{Left,Right}`.
- Model: standardised **ridge regression** (numpy closed form) → (x, y). Ridge strength
  α ∈ {0.01, 0.1, 1, 10, 100} chosen by leave-one-dot-out pixel error.
- Accuracy: on the 4 held-out test dots (median prediction per dot). **Rated against a
  baseline of always guessing the centre**: `skill = 1 − error / baseline_error`.
  `good`: skill ≥ 0.6 and error < 10% of viewport diagonal; `fair`: skill ≥ 0.35 and
  < 18%; else `poor`.
- Saved to `data/calibrations/<id>.json` (+ `<id>.webm` clip):
```json
{"id": "a1b2c3d4e5f6", "created": 1790000000.0,
 "model": {"features": [20 names], "mean": [...], "std": [...],
           "coef": [[cx, cy] × 20], "intercept": [x0, y0], "alpha": 1.0},
 "viewport": [1920, 937], "points_used": 9, "points_total": 9,
 "calibration_points": [{"x", "y", "pred_x", "pred_y", "error_px", "frames"}, ...],
 "validation_points":  [ same ],
 "leave_one_out_error_px": 95.2, "validation_error_px": 84.1,
 "baseline_error_px": 377.0, "skill": 0.78,
 "validation_error_pct_of_diagonal": 4.0, "quality": "good",
 "frames_processed": 401}
```
Prediction: `screen = ((features − mean) / std) @ coef + intercept`.

### 9.3 Outputs when a session is calibrated
- **Screen coordinates are fractions of the browser viewport**: (0,0) top-left,
  (1,1) bottom-right; outside 0..1 = looking off the page (> 0.05 outside counts as off-screen).
- `.npz`: `gaze_screen (N,2)` + `gaze_screen_valid (N,)` on the model-input grid
  (valid = face present and mask 1), `gaze_screen_raw (T,2)` + `gaze_screen_raw_valid`,
  `calibration_id`. Invalid rows are 0.
- Summary adds: `screen_gaze_frames`, `screen_x_mean`, `screen_y_mean`, `screen_x_std`,
  `screen_y_std`, `offscreen_ratio_calibrated`.
- Realistic webcam accuracy ~50–150 px: enough for page **regions**, not words.
  Recalibrate after moving seat, big lighting change, or resizing/moving the window.

---

## 10. Areas of interest (AOI): needs calibration

### 10.1 Marking regions (page side)
Add `data-aoi="<name>"` to any element (several elements may share a name). The demo
marks `passage` (reading div), `answer` (textarea), `quiz` (option buttons).
`behavior.js` records each region's rectangles as viewport fractions
`[x0, y0, x1, y1]` (4 decimals) at window start, every 0.5 s, and 100 ms after any
scroll/resize, **only when they change** → `events.aoi`. Regions moving with page
scroll are therefore tracked; rectangles may extend outside 0..1 (scrolled out of view).

### 10.2 Labelling (`processing/aoi.py`)
For each valid calibrated gaze point, using the latest rectangle snapshot at that time:
off-page (outside viewport ± 0.05) → `"offpage"`; else the nearest region whose
rectangle is within **0.03** (viewport units, allowance for gaze error) → its name;
else `"other"`; no gaze → none. Runs shorter than **2 frames** are merged into the
previous label (jitter). Frames without gaze don't break a visit.

### 10.3 AOI features: 9 (fixed, page-independent)
| feature | meaning |
|---|---|
| `on_aoi_ratio` | share of gaze time on any region |
| `other_ratio` | on the page but not on a region |
| `offpage_ratio` | outside the browser window |
| `aoi_visits` | separate visits to regions |
| `aoi_switches` | moves from one region to a different one (A→other→B counts) |
| `aoi_revisits` | returns to a region already visited this window |
| `mean_visit_s`, `longest_visit_s` | visit durations |
| `aoi_entropy` | 0 = all region time on one region, 1 = spread evenly over all regions |

Plus page-specific **detail**: `dwell` {region: share of gaze time}, `visits` {region: count}.
Frame durations come from timestamps, capped at 0.25 s across gaps.

`.npz`: `has_aoi`, `aoi_features (9,)`, `aoi_feature_names`, `aoi_names`,
`aoi_detail` (JSON), `aoi_labels (T,)` per raw frame (`""` = none),
`aoi_labels_fixed (N,)` per model-input step, `aoi_snapshots` (JSON).
Status JSON `aoi` = the 9 features + `dwell`, `visits`, `names`, `rects` (last snapshot).

Not built: reading-line regressions/word-level fixations (accuracy insufficient).

---

## 11. Stored outputs per window: `data/sessions/<sid>/`

| file | content |
|---|---|
| `window_NNNN.webm` | raw webcam clip (**identifiable biometric data**, see §13) |
| `window_NNNN_events.json` | anonymised events (§7.1) |
| `window_NNNN.json` | status object without `preview`/`screen_points` (incl. summary, behaviour, AOI) |
| `window_NNNN.npz` | all arrays below |
| `calibration.json` | copy of the session's calibration (if any) |

`window_NNNN.npz` keys:
| key | shape / type | notes |
|---|---|---|
| `sequence_fixed` | (N, 29) float32 | **model input** (N = 150 for 15 s) |
| `mask` | (N,) | 1 = real, 0 = padding |
| `grid_ms` | (N,) | 0, 100, 200, … |
| `sequence` | (T, 29) | raw sampled frames |
| `timestamps_ms` | (T,) | real frame times |
| `feature_names` | (29,) str | |
| `feature_groups` | JSON str | §6.2 groups |
| `fps` | scalar | 10.0 |
| `behavior` | (41,) float32 | zeros if no events |
| `behavior_names`, `behavior_groups` (JSON: keyboard/pointer/focus/spelling) | | |
| `has_behavior`, `has_spelling` | bool | |
| `gaze_screen`, `gaze_screen_valid`, `gaze_screen_raw`, `gaze_screen_raw_valid`, `calibration_id` | | only if calibrated |
| `has_aoi`, `aoi_features`, `aoi_feature_names`, `aoi_names`, `aoi_detail`, `aoi_labels`, `aoi_labels_fixed`, `aoi_snapshots` | | only if calibrated **and** the page has `data-aoi` regions |

Windows recorded with earlier versions of the code may lack `mask`, behaviour, or
screen/AOI keys; the loaders handle that.

---

## 12. Python loaders (`processing/dataset.py`)

```python
from processing.dataset import (load_window, load_session, load_behavior,
                                load_session_behavior, load_aoi)

X, mask, names = load_window(path)                          # (150, 29), (150,)
X, mask, names = load_window(path, ablation="B_gaze_head")  # (150, 13)
X, mask, names = load_window(path, screen_gaze=True)        # (150, 32): + screen_x, screen_y, screen_valid
X, M, names = load_session("data/sessions/<sid>")           # (W, 150, D), (W, 150); skips windows of other lengths

b, names, has_behavior = load_behavior(path)                       # (41,)
b, names, _ = load_behavior(path, groups=("keyboard", "spelling")) # subset
B, names = load_session_behavior("data/sessions/<sid>")            # (W, 41)

a, names, has_aoi, detail = load_aoi(path)   # (9,), detail = {"dwell": {...}, "visits": {...}}
```

Suggested model input per window (matches the report's fusion design):
`sequence_fixed (150×29) + mask` (+ optional screen gaze 150×3) → temporal model
(GRU/LSTM/Transformer); `behavior (41)` and `aoi_features (9)` → MLPs; fuse;
predict cognitive state. Normalise features per student (baseline) before training:
devices and people differ (e.g. touchpad vs mouse speed).

---

## 13. Integration notes, constraints, known gaps

**Deployment / architecture**
- Can run as a **separate microservice** (keep `app.py`) or have its processing modules
  imported into the platform backend (`extract_window`, `behavior_features`,
  `calibrate_video`, `apply_calibration`, `label_frames`, `aoi_features` are plain
  functions).
- **State is in memory**: `windows` and `session_cal` dicts. After a server restart
  the API forgets sessions (`GET .../windows` → 404, uploads to old sessions → 404);
  files on disk remain. Calibrations persist on disk; the page keeps the accepted one
  only in memory (reload = recalibrate). For production move status to the platform
  database (Module 13).
- **Python code changes need a server restart**; static files are read fresh. The demo
  page detects stale servers via `behavior_features`.
- Single process, `ThreadPoolExecutor(max_workers=2)`; MediaPipe is CPU-heavy, so scale
  workers/processes for many concurrent students.
- Binds `127.0.0.1:8000`. No auth, no CORS, no upload size limits, no rate limiting.
  Session ids are 12 hex chars (checked alphanumeric). Add authentication, link
  sessions to platform users, add limits before any non-local deployment.
- **Same-origin assumption:** the JS uses root-relative API paths (`/api/...`) and the
  spelling worker is created with a **relative** URL (`new Worker("spell-worker.js")`,
  which then loads `vendor/...` relative to itself). If platform pages live at other
  paths or on another origin: serve these files from the platform origin (or make the
  URLs configurable / absolute) and enable CORS on the API. Web Workers must be
  same-origin.
- HTTPS required for camera on non-localhost origins.

**What the platform must provide in its pages**
1. Include `behavior.js`, `spelling.js` (+ `spell-worker.js`, `vendor/`), and
   `calibration.js` if using screen gaze.
2. Camera stream via `getUserMedia`, and the recording loop of §4 (copy from
   `index.html`: `startRecording`, `startSegment`, `endSegment`, `upload`, `stopRecording`).
3. `data-aoi="..."` on the regions to analyse; `data-spell-vocab` on lesson content;
   `Spelling.allow([...])` for student/class names.
4. A calibration step before lessons if screen gaze/AOI is wanted; keep the window
   size fixed afterwards (or recalibrate).

**Privacy / ethics**
- Raw webcam clips (windows and calibration) are **stored on disk indefinitely**; they
  are identifiable biometric data of minors/special-needs students. Define consent,
  retention and deletion, or delete clips after feature extraction.
- Keystroke timing and pointer paths are behavioural biometrics even without content.
- No key content or typed text leaves the browser; spelling is checked locally.

**Accuracy / validation status (be honest in the report)**
- Verified: synthetic tests for resampling/mask, behaviour features (simulated calm vs
  stressed users), scroll gestures across mouse/touchpad/touchscreen simulations,
  spelling (real headless Chrome, scripted typing), calibration fitting (synthetic gaze)
  and a full real-Chrome run with a fake webcam (a still photo) through calibration →
  recording → AOI outputs.
- **Not yet validated on real students**: real calibration accuracy, AOI accuracy,
  the uncalibrated off-screen/blink/fixation heuristics, yaw/pitch sign conventions.
  Webcam gaze is coarse (regions, not words).
- Real webcam windows were recorded by the developer and processed successfully; the
  face/feature extraction works on real video.

**Not built (other modules)**: facial affect/emotion (Module 2; only eye blendshapes
exist), cognitive-state models (4–5), cognitive twin (6), adaptation (7–11), dashboard
(12), database/auth (13), per-student normalisation, activity flags for ambiguous zeros,
contenteditable support for spelling, capture outside the platform's pages.

**Tunable constants**
- `features.py`: `TARGET_FPS` (app.py, 10), `MAX_INTERP_GAP_MS` 250, off-screen 25°/20°/0.18, blink 0.5, fixation 0.6.
- `behavior.py`: burst gap 2 s, stroke gap 0.3 s, rapid click 0.6 s / 30 px, scroll gesture 0.3 s / 10 px.
- `calibration.py`: settle 700 ms, tail 100 ms, alphas, 15 fps, quality thresholds, off-screen margin 0.05.
- `aoi.py`: margin 0.03, off-page 0.05, min visit 2 frames, max frame 0.25 s.
- `calibration.js`: 1.8 s per dot, dot positions. `spelling.js`: 1 s idle re-check.
- Window length: 15–30 s slider in the demo page (`window_seconds`).
