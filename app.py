"""
Real-time webcam input processing server.

The browser records the webcam in fixed windows (15-30 s) and uploads each
window as soon as it ends. Each window is processed in a background worker:
frames -> MediaPipe Face Landmarker -> per-frame gaze/head-pose vectors.

Run:  python app.py      then open http://localhost:8000
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from processing.aoi import AOI_FEATURES, aoi_features, aoi_names_in, label_frames
from processing.behavior import (BEHAVIOR_FEATURES, FOCUS_FEATURES, KEYBOARD_FEATURES,
                                 POINTER_FEATURES, SPELLING_FEATURES, behavior_features,
                                 has_spelling)
from processing.calibration import (CalibrationStore, apply_calibration, calibrate_video,
                                    screen_summary)
from processing.features import FEATURE_GROUPS, FEATURE_NAMES, IDX, extract_window

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TARGET_FPS = 10.0
calibrations = CalibrationStore(ROOT / "data" / "calibrations")

app = FastAPI(title="Input Processing")
executor = ThreadPoolExecutor(max_workers=2)
lock = threading.Lock()
# session_id -> {window_index -> status dict}
windows: dict[str, dict[int, dict]] = {}
# session_id -> calibration (or None)
session_cal: dict[str, dict | None] = {}


def _session_dir(sid: str) -> Path:
    if not sid.isalnum():
        raise HTTPException(400, "bad session id")
    return DATA_DIR / sid


def _preview(ts: np.ndarray, seq: np.ndarray, n: int = 60) -> dict:
    """Downsampled series for the UI sparklines."""
    if len(seq) == 0:
        return {}
    pick = np.linspace(0, len(seq) - 1, min(n, len(seq))).astype(int)
    return {
        "t": (ts[pick] / 1000.0).round(2).tolist(),
        "face": seq[pick, IDX["face_present"]].tolist(),
        "yaw": seq[pick, IDX["head_yaw"]].round(1).tolist(),
        "pitch": seq[pick, IDX["head_pitch"]].round(1).tolist(),
        "gaze_h": seq[pick, IDX["gaze_h"]].round(3).tolist(),
        "gaze_v": seq[pick, IDX["gaze_v"]].round(3).tolist(),
    }


def _screen_preview(xy: np.ndarray, ok: np.ndarray, labels: list | None = None,
                    n: int = 150) -> list:
    """Calibrated gaze points [x, y, region] for the per-window gaze map."""
    idx = np.nonzero(ok > 0.5)[0]
    if len(idx) == 0:
        return []
    pick = idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
    return [[round(float(xy[i, 0]), 3), round(float(xy[i, 1]), 3),
             labels[i] if labels else None] for i in pick]


def _process(sid: str, index: int, video_path: Path, window_seconds: float,
             aoi_snapshots: list | None = None):
    info = windows[sid][index]
    info["status"] = "processing"
    t0 = time.time()
    try:
        r = extract_window(video_path, TARGET_FPS, window_seconds)
        out = video_path.with_suffix("")
        b = info.get("behavior") or dict.fromkeys(BEHAVIOR_FEATURES, 0.0)

        cal = session_cal.get(sid)
        screen = {}
        if cal:
            xy, ok = apply_calibration(cal["model"], r.sequence_fixed, r.mask)
            xy_raw, ok_raw = apply_calibration(cal["model"], r.sequence)
            screen = dict(gaze_screen=xy, gaze_screen_valid=ok,
                          gaze_screen_raw=xy_raw, gaze_screen_raw_valid=ok_raw,
                          calibration_id=cal["id"])
            r.summary.update(screen_summary(xy_raw, ok_raw))
            labels_raw = None
            if aoi_snapshots:
                names = aoi_names_in(aoi_snapshots)
                labels_raw = label_frames(r.timestamps_ms, xy_raw, ok_raw, aoi_snapshots)
                labels_fixed = label_frames(r.grid_ms, xy, ok, aoi_snapshots)
                feats, detail = aoi_features(r.timestamps_ms, labels_raw, names)
                screen.update(
                    has_aoi=True,
                    aoi_features=np.array([feats[k] for k in AOI_FEATURES], dtype=np.float32),
                    aoi_feature_names=np.array(AOI_FEATURES),
                    aoi_names=np.array(names),
                    aoi_detail=json.dumps(detail),
                    aoi_labels=np.array([l or "" for l in labels_raw]),
                    aoi_labels_fixed=np.array([l or "" for l in labels_fixed]),
                    aoi_snapshots=json.dumps(aoi_snapshots),
                )
                info["aoi"] = {**feats, **detail, "names": names, "rects": aoi_snapshots[-1][1]}
            info["screen_points"] = _screen_preview(xy_raw, ok_raw, labels_raw)
            info["calibration"] = {"id": cal["id"], "quality": cal["quality"],
                                   "error_px": round(cal["validation_error_px"]),
                                   "viewport": cal["viewport"]}
        np.savez_compressed(
            f"{out}.npz",
            **screen,
            behavior=np.array([b[k] for k in BEHAVIOR_FEATURES], dtype=np.float32),
            behavior_names=np.array(BEHAVIOR_FEATURES),
            behavior_groups=json.dumps({"keyboard": KEYBOARD_FEATURES,
                                        "pointer": POINTER_FEATURES,
                                        "focus": FOCUS_FEATURES,
                                        "spelling": SPELLING_FEATURES}),
            has_behavior=info.get("behavior") is not None,
            has_spelling=bool(info.get("has_spelling")),
            sequence=r.sequence,
            timestamps_ms=r.timestamps_ms,
            sequence_fixed=r.sequence_fixed,
            mask=r.mask,
            grid_ms=r.grid_ms,
            feature_names=np.array(FEATURE_NAMES),
            feature_groups=json.dumps(FEATURE_GROUPS),
            fps=TARGET_FPS,
        )
        info.update(
            status="done",
            processing_seconds=round(time.time() - t0, 2),
            shape=list(r.sequence.shape),
            shape_fixed=list(r.sequence_fixed.shape),
            observed_steps=int(r.mask.sum()),
            summary=r.summary,
            preview=_preview(r.timestamps_ms, r.sequence),
        )
        Path(f"{out}.json").write_text(json.dumps(
            {k: v for k, v in info.items() if k not in ("preview", "screen_points")}, indent=2))
    except Exception as e:
        traceback.print_exc()
        info.update(status="error", error=str(e))


@app.post("/api/calibrations")
async def create_calibration(
    video: UploadFile = File(...),
    points: str = Form(...),       # [{"x","y","t0","t1","phase"}], ms from recording start
    viewport: str = Form(...),     # [width, height] in CSS px
):
    tmp = calibrations.root / f"upload_{uuid.uuid4().hex[:8]}.webm"
    tmp.write_bytes(await video.read())
    try:
        result = await run_in_threadpool(
            calibrate_video, tmp, json.loads(points), json.loads(viewport))
    except ValueError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(422, str(e))
    cid = calibrations.save(result)
    tmp.rename(calibrations.root / f"{cid}.webm")
    return calibrations.load(cid)


@app.get("/api/calibrations/{cid}")
def get_calibration(cid: str):
    cal = calibrations.load(cid)
    if not cal:
        raise HTTPException(404, "unknown calibration")
    return cal


@app.post("/api/sessions")
def create_session(body: dict = Body(default={})):
    sid = uuid.uuid4().hex[:12]
    _session_dir(sid).mkdir(parents=True)
    cal = calibrations.load(body.get("calibration_id", "")) if body else None
    if cal:
        (_session_dir(sid) / "calibration.json").write_text(json.dumps(cal, indent=2))
    with lock:
        windows[sid] = {}
        session_cal[sid] = cal
    return {"session_id": sid, "target_fps": TARGET_FPS, "feature_names": FEATURE_NAMES,
            "behavior_features": BEHAVIOR_FEATURES, "calibration_id": cal["id"] if cal else None}


@app.post("/api/sessions/{sid}/windows")
async def upload_window(
    sid: str,
    video: UploadFile = File(...),
    index: int = Form(...),
    window_seconds: float = Form(...),       # configured length -> fixed rows
    recorded_seconds: float = Form(0.0),     # actual clip length (may be shorter)
    started_at: float = Form(0.0),
    events: str = Form(""),                  # keyboard/pointer/focus events (JSON)
):
    if sid not in windows:
        raise HTTPException(404, "unknown session")
    ext = ".mp4" if "mp4" in (video.content_type or "") else ".webm"
    path = _session_dir(sid) / f"window_{index:04d}{ext}"
    path.write_bytes(await video.read())

    behavior, spelling_ok, aoi_snapshots = None, False, None
    if events:
        try:
            ev = json.loads(events)
            (path.parent / f"{path.stem}_events.json").write_text(json.dumps(ev))
            behavior = behavior_features(ev)
            spelling_ok = has_spelling(ev)
            aoi_snapshots = ev.get("aoi") or None
        except (ValueError, TypeError, KeyError, IndexError) as e:
            print(f"bad events for window {index}: {e}")
    with lock:
        windows[sid][index] = {
            "index": index,
            "status": "queued",
            "started_at": started_at,
            "window_seconds": window_seconds,
            "recorded_seconds": recorded_seconds or window_seconds,
            "bytes": path.stat().st_size,
            "behavior": behavior,
            "has_spelling": spelling_ok,
        }
    executor.submit(_process, sid, index, path, window_seconds, aoi_snapshots)
    return {"index": index, "status": "queued"}


@app.get("/api/sessions/{sid}/windows")
def list_windows(sid: str):
    if sid not in windows:
        raise HTTPException(404, "unknown session")
    return [windows[sid][i] for i in sorted(windows[sid])]


@app.get("/api/sessions/{sid}/windows/{index}/npz")
def download_npz(sid: str, index: int):
    path = _session_dir(sid) / f"window_{index:04d}.npz"
    if not path.exists():
        raise HTTPException(404, "not ready")
    return FileResponse(path, filename=f"{sid}_window_{index:04d}.npz")


@app.get("/api/sessions/{sid}/windows/{index}/vectors")
def get_vectors(sid: str, index: int, kind: str = "fixed"):
    """kind=fixed: model input on the uniform grid + mask; kind=raw: real frames."""
    path = _session_dir(sid) / f"window_{index:04d}.npz"
    if not path.exists():
        raise HTTPException(404, "not ready")
    d = np.load(path)
    if "mask" not in d.files:   # window saved before masks existed
        kind = "raw"
    names = d["feature_names"].tolist()
    if kind == "raw":
        ts, rows, scr = d["timestamps_ms"], d["sequence"], "gaze_screen_raw"
    else:
        ts, rows, scr = d["grid_ms"], d["sequence_fixed"], "gaze_screen"
    if scr in d.files:  # calibrated: append the screen position columns
        rows = np.concatenate([rows, d[scr]], axis=1)
        names = names + ["screen_x", "screen_y"]
    out = {"kind": kind, "feature_names": names,
           "timestamps_ms": ts.round(1).tolist(), "rows": rows.round(4).tolist()}
    if kind != "raw":
        out["mask"] = d["mask"].astype(int).tolist()
    return out


app.mount("/",StaticFiles(directory=ROOT / "static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
