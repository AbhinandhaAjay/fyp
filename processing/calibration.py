"""
Gaze calibration: map per-frame gaze + head features to a point on the screen.

During calibration the student follows a dot shown at known positions. For each
dot we keep the frames after the eyes have settled, then fit a ridge regression

    [iris position, gaze, eye blendshapes, head pose]  ->  (screen_x, screen_y)

Screen coordinates are fractions of the browser viewport: (0, 0) top-left,
(1, 1) bottom-right. Values outside 0..1 mean looking off the page.

A few extra dots are held out of the fit and used only to measure accuracy.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

from processing.features import IDX, extract_window

INPUT_FEATURES = [
    "gaze_h", "gaze_v", "l_iris_h", "l_iris_v", "r_iris_h", "r_iris_v",
    "head_yaw", "head_pitch", "head_roll", "head_tx", "head_ty", "head_tz",
    "bs_eyeLookDownLeft", "bs_eyeLookDownRight", "bs_eyeLookInLeft", "bs_eyeLookInRight",
    "bs_eyeLookOutLeft", "bs_eyeLookOutRight", "bs_eyeLookUpLeft", "bs_eyeLookUpRight",
]
COLS = [IDX[n] for n in INPUT_FEATURES]

SETTLE_MS = 700          # ignore the start of each dot: eyes still moving to it
TAIL_MS = 100            # and the very end: eyes may leave early
BLINK_SCORE = 0.5
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
CAL_FPS = 15.0
OFFSCREEN_MARGIN = 0.05  # predictions this far outside the viewport count as off-screen
MIN_FRAMES_PER_POINT = 3


# --------------------------------------------------------------------------
def _ridge(X: np.ndarray, Y: np.ndarray, alpha: float):
    """Ridge on standardised X. Returns (mean, std, coef (D,2), intercept (2,))."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0
    Z = (X - mu) / sd
    y0 = Y.mean(0)
    coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - y0))
    return mu, sd, coef, y0


def _predict(model: dict, X: np.ndarray) -> np.ndarray:
    Z = (X - np.asarray(model["mean"])) / np.asarray(model["std"])
    return Z @ np.asarray(model["coef"]) + np.asarray(model["intercept"])


def _px_error(pred: np.ndarray, target: np.ndarray, viewport) -> np.ndarray:
    w, h = viewport
    d = (pred - target) * np.array([w, h])
    return np.hypot(d[:, 0], d[:, 1])


def _point_samples(ts_ms, seq, points):
    """Frames belonging to each dot, after settling, face visible, not blinking."""
    face = seq[:, IDX["face_present"]] > 0.5
    blink = (seq[:, IDX["bs_eyeBlinkLeft"]] + seq[:, IDX["bs_eyeBlinkRight"]]) / 2 > BLINK_SCORE
    out = []
    for p in points:
        sel = ((ts_ms >= p["t0"] + SETTLE_MS) & (ts_ms <= p["t1"] - TAIL_MS) & face & ~blink)
        out.append(np.nonzero(sel)[0])
    return out


def fit_calibration(ts_ms: np.ndarray, seq: np.ndarray, points: list[dict], viewport) -> dict:
    """points: [{"x","y","t0","t1","phase": "cal"|"val"}], times in ms from video start."""
    idx = _point_samples(ts_ms, seq, points)
    cal = [(p, i) for p, i in zip(points, idx) if p["phase"] == "cal" and len(i) >= MIN_FRAMES_PER_POINT]
    val = [(p, i) for p, i in zip(points, idx) if p["phase"] == "val" and len(i) >= MIN_FRAMES_PER_POINT]
    n_cal = sum(p["phase"] == "cal" for p in points)
    if len(cal) < max(5, n_cal - 2):
        raise ValueError(f"only {len(cal)} of {n_cal} calibration dots had usable frames "
                         "(face not found or eyes closed). Check lighting and try again.")

    X = np.concatenate([seq[i][:, COLS] for _, i in cal])
    Y = np.concatenate([np.tile([p["x"], p["y"]], (len(i), 1)) for p, i in cal])
    groups = np.concatenate([np.full(len(i), k) for k, (_, i) in enumerate(cal)])

    # Pick ridge strength by leave-one-dot-out error (in pixels)
    best = None
    for a in ALPHAS:
        errs = []
        for k in range(len(cal)):
            tr = groups != k
            mu, sd, coef, y0 = _ridge(X[tr], Y[tr], a)
            m = {"mean": mu, "std": sd, "coef": coef, "intercept": y0}
            pred = np.median(_predict(m, X[~tr]), axis=0, keepdims=True)
            errs.append(_px_error(pred, Y[~tr][:1], viewport)[0])
        score = float(np.mean(errs))
        if best is None or score < best[1]:
            best = (a, score)
    alpha, loo_px = best
    mu, sd, coef, y0 = _ridge(X, Y, alpha)
    model = {"features": INPUT_FEATURES, "mean": mu.tolist(), "std": sd.tolist(),
             "coef": coef.tolist(), "intercept": y0.tolist(), "alpha": alpha}

    def report(pairs):
        rows = []
        for p, i in pairs:
            pred = np.median(_predict(model, seq[i][:, COLS]), axis=0)
            err = float(_px_error(pred[None], np.array([[p["x"], p["y"]]]), viewport)[0])
            rows.append({"x": p["x"], "y": p["y"], "pred_x": float(pred[0]),
                         "pred_y": float(pred[1]), "error_px": err, "frames": int(len(i))})
        return rows

    cal_rows, val_rows = report(cal), report(val)
    diag = float(np.hypot(*viewport))
    val_err = [r["error_px"] for r in val_rows] or [loo_px]
    mean_px = float(np.mean(val_err))
    pct = mean_px / diag * 100
    # Baseline: always guessing the centre of the calibration dots. A model that
    # doesn't clearly beat this hasn't learnt anything about where the eyes look.
    centre = np.array([[np.mean([p["x"] for p, _ in cal]), np.mean([p["y"] for p, _ in cal])]])
    targets = np.array([[r["x"], r["y"]] for r in (val_rows or cal_rows)])
    baseline_px = float(np.mean(_px_error(np.repeat(centre, len(targets), 0), targets, viewport)))
    skill = 1.0 - mean_px / baseline_px if baseline_px > 0 else 0.0
    quality = ("good" if skill >= 0.6 and pct < 10 else
               "fair" if skill >= 0.35 and pct < 18 else "poor")
    return {
        "model": model,
        "viewport": list(viewport),
        "points_used": len(cal), "points_total": n_cal,
        "calibration_points": cal_rows,
        "validation_points": val_rows,
        "leave_one_out_error_px": loo_px,
        "validation_error_px": mean_px,
        "baseline_error_px": baseline_px,   # error of always guessing the centre
        "skill": skill,                     # 1 = perfect, 0 = no better than guessing
        "validation_error_pct_of_diagonal": pct,
        "quality": quality,
    }


def calibrate_video(video_path: str | Path, points: list[dict], viewport) -> dict:
    r = extract_window(video_path, CAL_FPS)
    result = fit_calibration(r.timestamps_ms, r.sequence, points, viewport)
    result["frames_processed"] = int(len(r.sequence))
    return result


# --------------------------------------------------------------------------
def apply_calibration(model: dict, seq: np.ndarray, valid: np.ndarray | None = None):
    """Per-frame screen gaze. Returns (xy (N,2), valid (N,)); invalid rows are 0."""
    face = seq[:, IDX["face_present"]] > 0.5
    ok = face if valid is None else (face & (valid > 0.5))
    xy = np.zeros((len(seq), 2), dtype=np.float32)
    if ok.any():
        xy[ok] = _predict(model, seq[ok][:, COLS]).astype(np.float32)
    return xy, ok.astype(np.float32)


def screen_summary(xy: np.ndarray, ok: np.ndarray) -> dict:
    """Window-level metrics from calibrated gaze."""
    v = ok > 0.5
    if not v.any():
        return {"screen_gaze_frames": 0}
    p = xy[v]
    m = OFFSCREEN_MARGIN
    off = (p[:, 0] < -m) | (p[:, 0] > 1 + m) | (p[:, 1] < -m) | (p[:, 1] > 1 + m)
    return {
        "screen_gaze_frames": int(v.sum()),
        "screen_x_mean": float(p[:, 0].mean()), "screen_y_mean": float(p[:, 1].mean()),
        "screen_x_std": float(p[:, 0].std()), "screen_y_std": float(p[:, 1].std()),
        "offscreen_ratio_calibrated": float(off.mean()),
    }


# --------------------------------------------------------------------------
class CalibrationStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def save(self, result: dict) -> str:
        cid = uuid.uuid4().hex[:12]
        result = {"id": cid, "created": time.time(), **result}
        (self.root / f"{cid}.json").write_text(json.dumps(result, indent=2))
        return cid

    def load(self, cid: str) -> dict | None:
        if not cid or not cid.isalnum():
            return None
        p = self.root / f"{cid}.json"
        return json.loads(p.read_text()) if p.exists() else None
