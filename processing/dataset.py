"""
Load saved windows as model input.

    from processing.dataset import load_window, load_session
    X, mask, names = load_window("data/sessions/<id>/window_0000.npz", ablation="B_gaze_head")
    X, M, names = load_session("data/sessions/<id>")      # (W, 150, D), (W, 150)

Ablations (see features.ABLATIONS):
    A_head          presence + head pose
    B_gaze_head     + iris gaze
    C_gaze_head_eye + eye openness (EAR)
    D_all           + eye blendshapes  (all 29)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from processing.features import ABLATIONS


def _columns(names: list[str], groups: dict, ablation: str | None) -> list[int]:
    if ablation is None:
        return list(range(len(names)))
    wanted = {n for g in ABLATIONS[ablation] for n in groups[g]}
    return [i for i, n in enumerate(names) if n in wanted]


def load_window(path: str | Path, ablation: str | None = None, screen_gaze: bool = False):
    """-> X (N, D'), mask (N,), feature names. Padded steps are zeros with mask 0.

    screen_gaze=True appends calibrated `screen_x`, `screen_y` (fractions of the
    browser window) and `screen_valid` (1 where a face was seen). Windows recorded
    without calibration get zeros there, with screen_valid 0."""
    d = np.load(path)
    names = d["feature_names"].tolist()
    cols = _columns(names, json.loads(str(d["feature_groups"])), ablation)
    X, names = d["sequence_fixed"][:, cols], [names[i] for i in cols]
    if screen_gaze:
        if "gaze_screen" in d.files:
            xy, ok = d["gaze_screen"], d["gaze_screen_valid"][:, None]
        else:
            xy, ok = np.zeros((len(X), 2), np.float32), np.zeros((len(X), 1), np.float32)
        X = np.concatenate([X, xy, ok], axis=1)
        names = names + ["screen_x", "screen_y", "screen_valid"]
    return X, d["mask"], names


def load_behavior(path: str | Path, groups=("keyboard", "pointer", "focus", "spelling")):
    """-> behaviour vector (B,), names, has_behavior. Zeros if none was recorded.
    Spelling values are zeros when the spell checker wasn't running; see
    `has_spelling` in the .npz."""
    d = np.load(path)
    if "behavior" not in d.files:
        raise KeyError(f"{path} was saved before behaviour capture existed")
    names = d["behavior_names"].tolist()
    available = json.loads(str(d["behavior_groups"]))
    wanted = {n for g in groups if g in available for n in available[g]}
    cols = [i for i, n in enumerate(names) if n in wanted]
    return d["behavior"][cols], [names[i] for i in cols], bool(d["has_behavior"])


def load_session_behavior(session_dir: str | Path,
                          groups=("keyboard", "pointer", "focus", "spelling")):
    """All windows of a session -> B (W, D), names. Row i matches window i."""
    rows, names = [], []
    for p in sorted(Path(session_dir).glob("window_*.npz")):
        b, names, _ = load_behavior(p, groups)
        rows.append(b)
    if not rows:
        raise FileNotFoundError(f"no processed windows in {session_dir}")
    return np.stack(rows), names


def load_aoi(path: str | Path):
    """-> AOI vector (9,), names, has_aoi, detail {"dwell": {region: share}, "visits": {...}}.
    Zeros with has_aoi False when the window wasn't calibrated or the page had no
    marked regions."""
    from processing.aoi import AOI_FEATURES
    d = np.load(path)
    if "aoi_features" not in d.files:
        return np.zeros(len(AOI_FEATURES), np.float32), list(AOI_FEATURES), False, {}
    return (d["aoi_features"], d["aoi_feature_names"].tolist(), True,
            json.loads(str(d["aoi_detail"])))


def load_session(session_dir: str | Path, ablation: str | None = None):
    """All windows of a session stacked -> X (W, N, D'), mask (W, N), names.
    Windows with a different length (changed slider mid-study) are skipped."""
    xs, ms, names = [], [], []
    for p in sorted(Path(session_dir).glob("window_*.npz")):
        X, m, names = load_window(p, ablation)
        if xs and X.shape != xs[0].shape:
            continue
        xs.append(X)
        ms.append(m)
    if not xs:
        raise FileNotFoundError(f"no processed windows in {session_dir}")
    return np.stack(xs), np.stack(ms), names
