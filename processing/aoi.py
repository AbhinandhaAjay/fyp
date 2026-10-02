"""
Areas of interest (AOI): which part of the page the student was looking at.

The page marks regions with data-aoi="passage" etc.; behavior.js records their
rectangles over time (they move when the page scrolls). Each calibrated gaze
point is labelled with the region it falls in at that moment:

    "<aoi name>"   inside (or within AOI_MARGIN of) a marked region
    "other"        on the page, but not near any region
    "offpage"      outside the browser window
    None           no usable gaze (no face / not calibrated)

Rectangles and gaze are both fractions of the browser viewport.
"""
from __future__ import annotations

import math

import numpy as np

AOI_MARGIN = 0.03        # count gaze this close to a region as on it (webcam error)
OFFPAGE_MARGIN = 0.05
MIN_VISIT_FRAMES = 2     # shorter runs are gaze jitter, merged into the previous label
MAX_FRAME_S = 0.25       # cap per-frame duration across gaps

AOI_FEATURES = [
    "on_aoi_ratio",      # share of gaze time on any marked region
    "other_ratio",       # on the page but not on a marked region
    "offpage_ratio",     # outside the browser window
    "aoi_visits",        # separate visits to regions
    "aoi_switches",      # moves from one region to a different one
    "aoi_revisits",      # returns to a region already visited this window
    "mean_visit_s",
    "longest_visit_s",
    "aoi_entropy",       # 0 = all time on one region, 1 = spread evenly
]


def _rects_at(snapshots: list, t_ms: float) -> dict:
    """Latest snapshot at or before t (or the first one)."""
    cur = snapshots[0][1] if snapshots else {}
    for t, rects in snapshots:
        if t > t_ms:
            break
        cur = rects
    return cur


def _hit(x: float, y: float, rects: dict) -> str:
    if x < -OFFPAGE_MARGIN or x > 1 + OFFPAGE_MARGIN or y < -OFFPAGE_MARGIN or y > 1 + OFFPAGE_MARGIN:
        return "offpage"
    best, best_d = None, None
    for name, boxes in rects.items():
        for x0, y0, x1, y1 in boxes:
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - y, 0.0, y - y1)
            d = math.hypot(dx, dy)          # 0 when inside
            if d <= AOI_MARGIN and (best_d is None or d < best_d):
                best, best_d = name, d
    return best or "other"


def _smooth(labels: list) -> list:
    """Merge runs shorter than MIN_VISIT_FRAMES into the previous label.
    Frames without gaze (None) are left as they are."""
    out = list(labels)
    idx = [i for i, l in enumerate(out) if l is not None]
    runs, start = [], 0
    for k in range(1, len(idx) + 1):
        if k == len(idx) or out[idx[k]] != out[idx[start]]:
            runs.append((start, k))
            start = k
    prev = None
    for a, b in runs:
        if b - a < MIN_VISIT_FRAMES and prev is not None:
            for k in range(a, b):
                out[idx[k]] = prev
        else:
            prev = out[idx[a]]
    return out


def label_frames(ts_ms: np.ndarray, xy: np.ndarray, ok: np.ndarray, snapshots: list) -> list:
    labels = [None] * len(ts_ms)
    for i, t in enumerate(ts_ms):
        if ok[i] > 0.5:
            labels[i] = _hit(float(xy[i, 0]), float(xy[i, 1]), _rects_at(snapshots, float(t)))
    return _smooth(labels)


def aoi_features(ts_ms: np.ndarray, labels: list, aoi_names: list[str]) -> tuple[dict, dict]:
    """-> (fixed features, per-region details {"dwell": {name: share}, "visits": {name: n}})."""
    f = dict.fromkeys(AOI_FEATURES, 0.0)
    detail = {"dwell": {n: 0.0 for n in aoi_names}, "visits": {n: 0 for n in aoi_names}}
    v = [i for i, l in enumerate(labels) if l is not None]
    if not v:
        return f, detail

    # per-frame duration from timestamps, capped across gaps
    step = float(np.median(np.diff(ts_ms))) if len(ts_ms) > 1 else 100.0
    dt = np.diff(ts_ms, append=ts_ms[-1] + step) / 1000.0
    dt = np.clip(dt, 0, MAX_FRAME_S)
    total = float(sum(dt[i] for i in v)) or 1.0
    time_on = {}
    for i in v:
        time_on[labels[i]] = time_on.get(labels[i], 0.0) + float(dt[i])

    on_aoi = {k: t for k, t in time_on.items() if k not in ("other", "offpage")}
    f["on_aoi_ratio"] = sum(on_aoi.values()) / total
    f["other_ratio"] = time_on.get("other", 0.0) / total
    f["offpage_ratio"] = time_on.get("offpage", 0.0) / total
    for n in aoi_names:
        detail["dwell"][n] = on_aoi.get(n, 0.0) / total

    # visits: consecutive runs on the same region (frames without gaze don't break a run)
    visits, cur, dur = [], None, 0.0
    for i in v:
        if labels[i] != cur:
            if cur is not None:
                visits.append((cur, dur))
            cur, dur = labels[i], 0.0
        dur += float(dt[i])
    visits.append((cur, dur))
    aoi_visits = [(n, d) for n, d in visits if n not in ("other", "offpage")]
    f["aoi_visits"] = len(aoi_visits)
    for n, _ in aoi_visits:
        detail["visits"][n] = detail["visits"].get(n, 0) + 1
    seen, revisits = set(), 0
    for n, _ in aoi_visits:
        revisits += n in seen
        seen.add(n)
    f["aoi_revisits"] = revisits
    f["aoi_switches"] = sum(1 for (a, _), (b, _) in zip(aoi_visits, aoi_visits[1:]) if a != b)
    if aoi_visits:
        ds = [d for _, d in aoi_visits]
        f["mean_visit_s"] = float(np.mean(ds))
        f["longest_visit_s"] = float(np.max(ds))
    if len(aoi_names) > 1 and on_aoi:
        p = np.array([on_aoi.get(n, 0.0) for n in aoi_names])
        p = p[p > 0] / p.sum()
        f["aoi_entropy"] = float(-(p * np.log(p)).sum() / math.log(len(aoi_names)))
    return {k: float(x) for k, x in f.items()}, detail


def aoi_names_in(snapshots: list) -> list[str]:
    names = []
    for _, rects in snapshots:
        for n in rects:
            if n not in names:
                names.append(n)
    return names
