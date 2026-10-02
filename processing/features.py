"""
Video window -> per-frame feature vectors (eye gaze + head pose) using
MediaPipe Face Landmarker.

Pipeline (one window, e.g. 15 s):
    video file -> sampled frames (target_fps) -> Face Landmarker
      -> per frame: gaze features (iris position, EAR, eye blendshapes)
                    head pose (yaw/pitch/roll + translation)
      -> sequence of vectors, shape (T, D)
      -> window summary (attention metrics for Module 1)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# --- MediaPipe 478-point face mesh indices -------------------------------
# "Right eye" = subject's right = left side of the (non-mirrored) image.
R_EYE_CORNERS = (33, 133)       # image-left corner, image-right corner
L_EYE_CORNERS = (362, 263)
R_EYE_LIDS = (159, 145)         # upper, lower
L_EYE_LIDS = (386, 374)
R_IRIS_CENTER = 468
L_IRIS_CENTER = 473
# 6-point eye aspect ratio: p1..p6 (corner, top, top, corner, bottom, bottom)
R_EAR_PTS = (33, 160, 158, 133, 153, 144)
L_EAR_PTS = (362, 385, 387, 263, 373, 380)

EYE_BLENDSHAPES = [
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
]

FEATURE_NAMES = (
    ["face_present"]
    + ["head_yaw", "head_pitch", "head_roll", "head_tx", "head_ty", "head_tz"]
    + ["l_iris_h", "l_iris_v", "r_iris_h", "r_iris_v", "gaze_h", "gaze_v"]
    + ["l_ear", "r_ear"]
    + [f"bs_{n}" for n in EYE_BLENDSHAPES]
)
FEATURE_DIM = len(FEATURE_NAMES)
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

# Column groups, for feature ablation (which signals actually help prediction)
FEATURE_GROUPS = {
    "presence": ["face_present"],
    "head": [n for n in FEATURE_NAMES if n.startswith("head_")],
    "gaze": [n for n in FEATURE_NAMES if "iris" in n or n.startswith("gaze_")],
    "eye": ["l_ear", "r_ear"],
    "blendshapes": [n for n in FEATURE_NAMES if n.startswith("bs_")],
}
ABLATIONS = {
    "A_head": ["presence", "head"],
    "B_gaze_head": ["presence", "head", "gaze"],
    "C_gaze_head_eye": ["presence", "head", "gaze", "eye"],
    "D_all": list(FEATURE_GROUPS),
}

# Resampling: a grid step counts as observed only if real frames lie close by
MAX_INTERP_GAP_MS = 250.0       # bridge at most ~1 missing sample at 10 fps

# Heuristic thresholds (webcam-grade; tune per setup / calibrate later)
OFFSCREEN_YAW_DEG = 25.0
OFFSCREEN_PITCH_DEG = 20.0
OFFSCREEN_GAZE = 0.18           # |iris offset from eye centre| (0..0.5)
BLINK_SCORE = 0.5              # eyeBlink blendshape (more robust than EAR)
FIXATION_VEL = 0.6              # gaze units / second (head-angle-normalised)


@dataclass
class WindowResult:
    timestamps_ms: np.ndarray   # (T,) real frame times
    sequence: np.ndarray        # (T, D) float32, one row per real frame
    sequence_fixed: np.ndarray  # (N, D) on a uniform 1/fps grid, model input
    mask: np.ndarray            # (N,) 1 = backed by real frames, 0 = padding
    grid_ms: np.ndarray         # (N,) grid times
    summary: dict


# --------------------------------------------------------------------------
def _create_landmarker() -> vision.FaceLandmarker:
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def _rotation_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Rotation matrix -> (yaw, pitch, roll) in degrees."""
    sy = math.hypot(R[2, 1], R[2, 2])
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def _iris_ratio(pts: np.ndarray, corners, lids, iris) -> tuple[float, float]:
    """Iris position inside the eye: h,v in [0,1], 0.5 = centred."""
    a, b = pts[corners[0]], pts[corners[1]]
    c = pts[iris]
    ab = b - a
    h = float(np.dot(c - a, ab) / (np.dot(ab, ab) + 1e-9))
    up, lo = pts[lids[0]], pts[lids[1]]
    v = float((c[1] - up[1]) / ((lo[1] - up[1]) + 1e-9))
    return h, v


def _ear(pts: np.ndarray, ids) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in ids)
    return float(
        (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
        / (2.0 * np.linalg.norm(p1 - p4) + 1e-9)
    )


def frame_features(result, width: int, height: int) -> np.ndarray:
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    if not result.face_landmarks:
        return vec

    lm = result.face_landmarks[0]
    # pixel coordinates so ratios are aspect-correct
    pts = np.array([(p.x * width, p.y * height) for p in lm], dtype=np.float64)

    vec[IDX["face_present"]] = 1.0

    if result.facial_transformation_matrixes:
        M = np.asarray(result.facial_transformation_matrixes[0])
        yaw, pitch, roll = _rotation_to_euler(M[:3, :3])
        vec[IDX["head_yaw"]], vec[IDX["head_pitch"]], vec[IDX["head_roll"]] = yaw, pitch, roll
        vec[IDX["head_tx"]], vec[IDX["head_ty"]], vec[IDX["head_tz"]] = M[:3, 3]

    if len(pts) > L_IRIS_CENTER:  # iris landmarks present
        lh, lv = _iris_ratio(pts, L_EYE_CORNERS, L_EYE_LIDS, L_IRIS_CENTER)
        rh, rv = _iris_ratio(pts, R_EYE_CORNERS, R_EYE_LIDS, R_IRIS_CENTER)
        vec[IDX["l_iris_h"]], vec[IDX["l_iris_v"]] = lh, lv
        vec[IDX["r_iris_h"]], vec[IDX["r_iris_v"]] = rh, rv
        vec[IDX["gaze_h"]] = (lh + rh) / 2.0 - 0.5
        vec[IDX["gaze_v"]] = (lv + rv) / 2.0 - 0.5

    vec[IDX["l_ear"]] = _ear(pts, L_EAR_PTS)
    vec[IDX["r_ear"]] = _ear(pts, R_EAR_PTS)

    if result.face_blendshapes:
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        for n in EYE_BLENDSHAPES:
            vec[IDX[f"bs_{n}"]] = scores.get(n, 0.0)

    return vec


# --------------------------------------------------------------------------
def extract_window(video_path: str | Path, target_fps: float = 10.0,
                   window_seconds: float | None = None) -> WindowResult:
    """Decode a video window, sample frames at target_fps, run Face Landmarker."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    nominal_fps = cap.get(cv2.CAP_PROP_FPS)
    if not nominal_fps or nominal_fps > 120 or nominal_fps < 1:
        nominal_fps = 30.0
    step_ms = 1000.0 / target_fps
    tol_ms = 500.0 / nominal_fps    # half a source-frame period

    landmarker = _create_landmarker()
    timestamps, rows = [], []
    next_t, last_ts, idx = 0.0, -1.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            # Browser-recorded WebM often lacks reliable timestamps
            if ts <= last_ts:
                ts = idx * 1000.0 / nominal_fps
            last_ts = ts
            idx += 1
            if ts < next_t - tol_ms:
                continue
            # Advance along a fixed grid so sampling error doesn't accumulate
            while next_t <= ts + tol_ms:
                next_t += step_ms

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(image, int(ts))
            rows.append(frame_features(res, w, h))
            timestamps.append(ts)
    finally:
        landmarker.close()
        cap.release()

    ts_arr = np.asarray(timestamps, dtype=np.float64)
    seq = np.stack(rows) if rows else np.zeros((0, FEATURE_DIM), np.float32)

    duration_s = window_seconds or (ts_arr[-1] / 1000.0 + 1.0 / target_fps if len(ts_arr) else 0.0)
    fixed, mask, grid = resample_fixed(ts_arr, seq, duration_s, target_fps)
    summary = summarise(ts_arr, seq, target_fps)
    summary["frames_decoded"] = idx
    summary["frames_processed"] = int(len(seq))
    summary["grid_steps"] = int(len(mask))
    summary["observed_steps"] = int(mask.sum())
    summary["padded_steps"] = int(len(mask) - mask.sum())
    return WindowResult(ts_arr, seq, fixed, mask, grid, summary)


def resample_fixed(ts_ms: np.ndarray, seq: np.ndarray, duration_s: float,
                   fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample real frames onto exactly round(duration*fps) uniform steps.

    For each grid time g:
      - real frames on both sides within MAX_INTERP_GAP_MS -> linear
        interpolation by timestamp (nearest frame if either side has no face,
        so "no face" is never blended into a fake half-face)
      - otherwise a real frame within half a step -> that frame
      - otherwise -> zeros and mask 0 (padding; never a repeated frame)

    A step with no face detected is still a real observation (mask 1,
    face_present 0): the learner being absent is information.
    """
    step = 1000.0 / fps
    n = max(1, int(round(duration_s * fps)))
    grid = np.arange(n) * step
    out = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    if len(seq) == 0:
        return out, mask, grid

    face = seq[:, IDX["face_present"]] > 0.5
    for k, g in enumerate(grid):
        j = int(np.searchsorted(ts_ms, g))      # first frame at/after g
        i = j - 1                               # last frame before g
        has_i, has_j = i >= 0, j < len(ts_ms)
        if has_i and has_j and ts_ms[j] - ts_ms[i] <= MAX_INTERP_GAP_MS:
            if face[i] and face[j]:
                w = (g - ts_ms[i]) / max(ts_ms[j] - ts_ms[i], 1e-6)
                out[k] = (1.0 - w) * seq[i] + w * seq[j]
            else:
                out[k] = seq[i] if g - ts_ms[i] < ts_ms[j] - g else seq[j]
            mask[k] = 1.0
            continue
        cands = [c for c in (i, j) if 0 <= c < len(ts_ms)]
        near = min(cands, key=lambda c: abs(ts_ms[c] - g))
        if abs(ts_ms[near] - g) <= step / 2:
            out[k] = seq[near]
            mask[k] = 1.0
    return out, mask, grid


def _runs(mask: np.ndarray) -> list[int]:
    """Lengths of consecutive True runs."""
    runs, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def summarise(ts_ms: np.ndarray, seq: np.ndarray, fps: float) -> dict:
    """Window-level attention metrics (approximate, webcam-based)."""
    T = len(seq)
    if T == 0:
        return {"face_presence_ratio": 0.0}

    face = seq[:, IDX["face_present"]] > 0.5
    s = {"face_presence_ratio": float(face.mean())}
    if not face.any():
        s["off_screen_ratio"] = 1.0
        return s

    f = seq[face]
    for name in ["head_yaw", "head_pitch", "head_roll", "gaze_h", "gaze_v"]:
        col = f[:, IDX[name]]
        s[f"{name}_mean"] = float(col.mean())
        s[f"{name}_std"] = float(col.std())

    yaw, pitch = seq[:, IDX["head_yaw"]], seq[:, IDX["head_pitch"]]
    gh, gv = seq[:, IDX["gaze_h"]], seq[:, IDX["gaze_v"]]
    on_screen = (
        face
        & (np.abs(yaw) < OFFSCREEN_YAW_DEG)
        & (np.abs(pitch) < OFFSCREEN_PITCH_DEG)
        & (np.abs(gh) < OFFSCREEN_GAZE)
    )
    s["off_screen_ratio"] = float(1.0 - on_screen.mean())
    s["off_screen_episodes"] = len(_runs(~on_screen))

    # Blinks: eyeBlink blendshape above threshold (rising edges)
    ear = (seq[:, IDX["l_ear"]] + seq[:, IDX["r_ear"]]) / 2.0
    blink = (seq[:, IDX["bs_eyeBlinkLeft"]] + seq[:, IDX["bs_eyeBlinkRight"]]) / 2.0
    closed = face & (blink > BLINK_SCORE)
    blinks = int(np.sum(closed[1:] & ~closed[:-1]) + (1 if closed[0] else 0))
    dur_min = max((ts_ms[-1] - ts_ms[0]) / 60000.0, 1e-6) if T > 1 else 1e-6
    s["blink_count"] = blinks
    s["blink_rate_per_min"] = float(blinks / dur_min) if T > 1 else 0.0
    s["ear_mean"] = float(ear[face].mean())

    # Fixations / gaze transitions from combined head + eye direction
    combo = np.stack([gh + yaw / 90.0, gv + pitch / 90.0], axis=1)
    if T > 1:
        dt = np.diff(ts_ms) / 1000.0
        vel = np.linalg.norm(np.diff(combo, axis=0), axis=1) / np.maximum(dt, 1e-3)
        valid = face[1:] & face[:-1] & ~closed[1:]
        fix_mask = valid & (vel < FIXATION_VEL)
        runs = [r for r in _runs(fix_mask) if r >= max(1, int(0.2 * fps))]  # >=200ms
        s["fixation_count"] = len(runs)
        s["fixation_mean_duration_s"] = float(np.mean(runs) / fps) if runs else 0.0
        s["gaze_transitions"] = int(np.sum(valid & (vel >= FIXATION_VEL)))
        s["gaze_velocity_mean"] = float(vel[valid].mean()) if valid.any() else 0.0
    return s
