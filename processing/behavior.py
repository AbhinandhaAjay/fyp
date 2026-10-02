"""
Keyboard + pointer (mouse / touchpad / touch) + focus events -> one behaviour
vector per window.

Events come from static/behavior.js, already anonymised: keys are reduced to a
category (char, backspace, enter, ...) and never identify what was typed.
Times are ms relative to the window start.

    events = {
      "duration_ms": 15000, "viewport": [w, h], "initially_hidden": false,
      "keys":    [[t, "d"|"u", cat, repeat_or_dwell_ms], ...],
      "pointer": [[t, "m"|"d", x, y, ptype, button], ...],   ptype m/t/p
      "wheel":   [[t, dx_px, dy_px, ctrl], ...],   raw wheel; ctrl=1 is pinch/zoom
      "scroll":  [[t, dx_px, dy_px], ...],         actual content movement, any device
      "focus":   [[t, "hidden"|"visible"|"blur"|"focus"], ...],
      "spell":   {"ready": bool, "words", "misspelled", "fixed", "remaining"},
      "aoi":     [[t, {"passage": [[x0, y0, x1, y1], ...], ...}], ...],  region rects
                 (viewport fractions); used by processing/aoi.py, not by this module
    }
"""
from __future__ import annotations

import math

import numpy as np

KEYBOARD_FEATURES = [
    "key_count",            # non-modifier key presses (auto-repeat excluded)
    "keys_per_sec",         # over the whole window
    "typing_speed_burst",   # keys/s while actually typing (pauses excluded)
    "typing_active_ratio",  # share of 1-s slots with any key press
    "iki_mean_s",           # inter-key interval inside typing bursts
    "iki_std_s",            #   (rhythm irregularity)
    "dwell_mean_s",         # how long keys are held down
    "dwell_std_s",
    "backspace_count",      # incl. auto-repeat while held
    "delete_count",
    "correction_ratio",     # deletions / all keystrokes: typing errors
    "max_correction_run",   # longest run of consecutive deletions
    "correction_episodes",  # how many separate times they went back to fix
    "enter_count",
    "pause_count",          # gaps >= 2 s between key presses
    "longest_key_pause_s",
    "held_key_repeats",     # auto-repeat events (holding a key)
]
POINTER_FEATURES = [
    "move_events",
    "distance_norm",        # path length in viewport diagonals
    "speed_mean",           # diagonals / s
    "speed_std",
    "speed_max",
    "direction_changes",    # turns > 90 deg (jittery / searching movement)
    "path_straightness",    # 1 = straight to target, low = wandering
    "pointer_active_ratio", # share of 1-s slots with any pointer activity
    "longest_pointer_idle_s",
    "click_count",
    "rapid_click_count",    # repeat clicks < 0.6 s apart, same spot (frustration)
    "scroll_gestures",      # separate scroll movements (device-independent)
    "scroll_distance_norm", # in viewport heights
    "scroll_reversals",     # down <-> up changes (re-reading / searching)
    "zoom_gestures",        # touchpad pinch / Ctrl+wheel zooms
    "touch_ratio",          # share of pointer events from touch
]
FOCUS_FEATURES = [
    "tab_switches",         # page hidden (switched tab / minimised)
    "hidden_ratio",         # share of the window the page was hidden
    "window_blur_count",    # focus left the browser window
]
SPELLING_FEATURES = [
    "words_typed",          # words finished (space / punctuation / Enter after them)
    "misspelled_words",     # misspelled words that appeared in the text
    "misspelling_ratio",    # misspelled / words typed
    "misspellings_fixed",   # misspelled words later corrected or deleted
    "misspellings_remaining",  # misspelled words left in the text at window end
]
BEHAVIOR_FEATURES = KEYBOARD_FEATURES + POINTER_FEATURES + FOCUS_FEATURES + SPELLING_FEATURES

BURST_GAP_S = 2.0           # key gap that counts as a pause
STROKE_GAP_S = 0.3          # pointer gap that ends a movement stroke
RAPID_CLICK_S = 0.6
RAPID_CLICK_PX = 30.0
GESTURE_GAP_S = 0.3         # scroll events closer than this form one gesture
MIN_GESTURE_PX = 10.0       # smaller movements are noise, not a gesture
DELETIONS = ("bs", "del")


def _active_ratio(times_s: list[float], duration_s: float) -> float:
    n = max(1, math.ceil(duration_s))
    slots = {min(int(t), n - 1) for t in times_s if 0 <= t <= duration_s}
    return len(slots) / n


def _longest_gap(times_s: list[float], duration_s: float) -> float:
    """Longest stretch without events, counting the window edges."""
    pts = [0.0] + sorted(t for t in times_s if 0 <= t <= duration_s) + [duration_s]
    return float(max(np.diff(pts))) if len(pts) > 1 else duration_s


def keyboard_features(keys: list, duration_s: float) -> dict:
    downs = [k for k in keys if k[1] == "d" and k[2] != "mod"]
    fresh = [k for k in downs if not k[3]]          # not auto-repeat
    t = [k[0] / 1000.0 for k in fresh]
    f = dict.fromkeys(KEYBOARD_FEATURES, 0.0)

    f["key_count"] = len(fresh)
    f["keys_per_sec"] = len(fresh) / duration_s if duration_s else 0.0
    f["typing_active_ratio"] = _active_ratio(t, duration_s)
    f["longest_key_pause_s"] = _longest_gap(t, duration_s)
    f["held_key_repeats"] = sum(1 for k in downs if k[3])

    iki = np.diff(t) if len(t) > 1 else np.array([])
    burst = iki[iki < BURST_GAP_S]
    f["pause_count"] = int(np.sum(iki >= BURST_GAP_S))
    if len(burst):
        f["iki_mean_s"] = float(burst.mean())
        f["iki_std_s"] = float(burst.std())
        f["typing_speed_burst"] = float(1.0 / max(burst.mean(), 1e-3))

    dwell = [k[3] / 1000.0 for k in keys if k[1] == "u" and k[2] != "mod" and k[3] > 0]
    if dwell:
        f["dwell_mean_s"] = float(np.mean(dwell))
        f["dwell_std_s"] = float(np.std(dwell))

    # Corrections: count every deletion keystroke, including held-key repeats
    f["backspace_count"] = sum(1 for k in downs if k[2] == "bs")
    f["delete_count"] = sum(1 for k in downs if k[2] == "del")
    deletions = f["backspace_count"] + f["delete_count"]
    all_strokes = len(fresh) + sum(1 for k in downs if k[3] and k[2] in DELETIONS)
    f["correction_ratio"] = deletions / all_strokes if all_strokes else 0.0
    run = best = episodes = 0
    for k in downs:
        if k[2] in DELETIONS:
            run += 1
            if run == 1:
                episodes += 1
            best = max(best, run)
        else:
            run = 0
    f["max_correction_run"] = best
    f["correction_episodes"] = episodes
    f["enter_count"] = sum(1 for k in fresh if k[2] == "enter")
    return f


def _gestures(events: list) -> list[list]:
    """Group events into gestures: split on gaps > GESTURE_GAP_S."""
    groups = []
    for e in events:
        if groups and (e[0] - groups[-1][-1][0]) / 1000.0 <= GESTURE_GAP_S:
            groups[-1].append(e)
        else:
            groups.append([e])
    return groups


def scroll_features(scroll: list, wheel: list, viewport_h: float) -> dict:
    """Scroll gestures from actual content movement, so a touchpad's stream of
    tiny events and a mouse wheel's few large notches count the same way."""
    # A gesture also ends when the vertical direction flips
    segs = []                         # [direction, total px]
    last_t = None
    for t, dx, dy in scroll:
        d = (dy > 0) - (dy < 0)
        gap = last_t is None or (t - last_t) / 1000.0 > GESTURE_GAP_S
        if segs and not gap and (d == 0 or segs[-1][0] in (0, d)):
            segs[-1][0] = segs[-1][0] or d
            segs[-1][1] += abs(dx) + abs(dy)
        else:
            segs.append([d, abs(dx) + abs(dy)])
        last_t = t
    real = [sg for sg in segs if sg[1] >= MIN_GESTURE_PX]
    dirs = [sg[0] for sg in real if sg[0] != 0]
    return {
        "scroll_gestures": len(real),
        "scroll_distance_norm": sum(abs(dx) + abs(dy) for _, dx, dy in scroll) / (viewport_h or 1),
        "scroll_reversals": sum(1 for a, b in zip(dirs, dirs[1:]) if a != b),
        "zoom_gestures": len(_gestures([w for w in wheel if len(w) > 3 and w[3]])),
    }


def pointer_features(pointer: list, wheel: list, scroll: list | None, viewport,
                     duration_s: float) -> dict:
    w, h = (viewport or [1920, 1080])[:2]
    diag = math.hypot(w, h) or 1.0
    f = dict.fromkeys(POINTER_FEATURES, 0.0)

    moves = [p for p in pointer if p[1] == "m"]
    clicks = [p for p in pointer if p[1] == "d"]
    f["move_events"] = len(moves)
    f["click_count"] = len(clicks)
    if scroll is None:  # recorded before scroll capture: fall back to non-zoom wheel
        scroll = [w[:3] for w in wheel if not (len(w) > 3 and w[3])]
    f.update(scroll_features(scroll, wheel, h))
    if pointer:
        f["touch_ratio"] = sum(1 for p in pointer if p[4] == "t") / len(pointer)

    all_t = [x[0] / 1000.0 for x in pointer + wheel + scroll]
    f["pointer_active_ratio"] = _active_ratio(all_t, duration_s)
    f["longest_pointer_idle_s"] = _longest_gap(all_t, duration_s)

    # Movement kinematics, split into strokes at pauses
    speeds, turns, straight = [], 0, []
    stroke, total = [], 0.0
    prev_vec = None

    def close_stroke(s):
        if len(s) < 2:
            return
        path = sum(math.dist(s[i][1:], s[i + 1][1:]) for i in range(len(s) - 1))
        if path / diag > 0.02:  # ignore tiny wiggles
            straight.append(math.dist(s[0][1:], s[-1][1:]) / path)

    for i, p in enumerate(moves):
        pt = (p[0] / 1000.0, p[2], p[3])
        if stroke and pt[0] - stroke[-1][0] > STROKE_GAP_S:
            close_stroke(stroke)
            stroke, prev_vec = [], None
        if stroke:
            dt = pt[0] - stroke[-1][0]
            dx, dy = pt[1] - stroke[-1][1], pt[2] - stroke[-1][2]
            d = math.hypot(dx, dy)
            total += d
            if dt > 0:
                speeds.append(d / diag / dt)
            if d > 2:
                if prev_vec is not None:
                    dot = dx * prev_vec[0] + dy * prev_vec[1]
                    if dot < 0:      # angle > 90 deg
                        turns += 1
                prev_vec = (dx, dy)
        stroke.append(pt)
    close_stroke(stroke)

    f["distance_norm"] = total / diag
    if speeds:
        f["speed_mean"] = float(np.mean(speeds))
        f["speed_std"] = float(np.std(speeds))
        f["speed_max"] = float(np.max(speeds))
    f["direction_changes"] = turns
    f["path_straightness"] = float(np.mean(straight)) if straight else 0.0

    rapid = 0
    for a, b in zip(clicks, clicks[1:]):
        if (b[0] - a[0]) / 1000.0 < RAPID_CLICK_S and math.dist(a[2:4], b[2:4]) < RAPID_CLICK_PX:
            rapid += 1
    f["rapid_click_count"] = rapid

    return f


def focus_features(focus: list, initially_hidden: bool, duration_s: float) -> dict:
    f = dict.fromkeys(FOCUS_FEATURES, 0.0)
    hidden_since = 0.0 if initially_hidden else None
    hidden_total = 0.0
    for t_ms, kind in focus:
        t = t_ms / 1000.0
        if kind == "hidden":
            f["tab_switches"] += 1
            if hidden_since is None:
                hidden_since = t
        elif kind == "visible" and hidden_since is not None:
            hidden_total += t - hidden_since
            hidden_since = None
        elif kind == "blur":
            f["window_blur_count"] += 1
    if hidden_since is not None:
        hidden_total += duration_s - hidden_since
    f["hidden_ratio"] = min(1.0, hidden_total / duration_s) if duration_s else 0.0
    return f


def spelling_features(spell: dict | None) -> dict:
    """Counts come from the browser's local spell checker; all zeros if it
    wasn't running (check `has_spelling` before trusting them)."""
    f = dict.fromkeys(SPELLING_FEATURES, 0.0)
    if not spell or not spell.get("ready"):
        return f
    f["words_typed"] = spell.get("words", 0)
    f["misspelled_words"] = spell.get("misspelled", 0)
    f["misspellings_fixed"] = spell.get("fixed", 0)
    f["misspellings_remaining"] = spell.get("remaining", 0)
    # a misspelling can also be found on leaving a field, without a counted word end
    f["misspelling_ratio"] = f["misspelled_words"] / max(f["words_typed"], f["misspelled_words"], 1)
    return f


def has_spelling(events: dict) -> bool:
    return bool((events.get("spell") or {}).get("ready"))


def behavior_features(events: dict, duration_s: float | None = None) -> dict:
    """All behaviour features for one window, as {name: value}."""
    duration_s = duration_s or events.get("duration_ms", 15000) / 1000.0
    out = {}
    out.update(keyboard_features(events.get("keys", []), duration_s))
    out.update(pointer_features(events.get("pointer", []), events.get("wheel", []),
                                events.get("scroll"),
                                events.get("viewport"), duration_s))
    out.update(focus_features(events.get("focus", []), events.get("initially_hidden", False),
                              duration_s))
    out.update(spelling_features(events.get("spell")))
    return {k: float(out[k]) for k in BEHAVIOR_FEATURES}


def behavior_vector(events: dict, duration_s: float | None = None) -> np.ndarray:
    f = behavior_features(events, duration_s)
    return np.array([f[k] for k in BEHAVIOR_FEATURES], dtype=np.float32)
