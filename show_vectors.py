"""
Print the vectors saved for a window.

    python show_vectors.py                      # latest window of the latest session
    python show_vectors.py path/to/window.npz   # a specific window
    python show_vectors.py path/to/clip.webm    # run extraction on any video file
"""
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data" / "sessions"
SHOW = ["face_present", "head_yaw", "head_pitch", "head_roll", "gaze_h", "gaze_v",
        "bs_eyeBlinkLeft", "bs_eyeBlinkRight"]


def latest_npz() -> Path:
    files = sorted(DATA_DIR.glob("*/window_*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit("No processed windows yet. Record something in the UI first.")
    return files[-1]


def table(title, ts, rows, names, mask=None, n=10):
    cols = [names.index(c) for c in SHOW]
    print(f"\n{title}")
    print("  t(s) " + (" mask " if mask is not None else "")
          + " ".join(f"{c[:10]:>10}" for c in SHOW))
    for k in range(min(n, len(rows))):
        m = f" {int(mask[k]):4d}" if mask is not None else ""
        print(f"{ts[k] / 1000:6.2f}{m} " + " ".join(f"{rows[k][c]:10.3f}" for c in cols))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_npz()
    if path.suffix == ".npz":
        d = np.load(path)
        names = list(d["feature_names"])
        raw, ts = d["sequence"], d["timestamps_ms"]
        X, mask, grid = d["sequence_fixed"], d["mask"], d["grid_ms"]
    else:
        from processing.features import FEATURE_NAMES, extract_window
        r = extract_window(path, window_seconds=15)
        names, raw, ts = FEATURE_NAMES, r.sequence, r.timestamps_ms
        X, mask, grid = r.sequence_fixed, r.mask, r.grid_ms

    print(path)
    print(f"model input: {X.shape} + mask {mask.shape}  "
          f"({int(mask.sum())} observed, {int(len(mask) - mask.sum())} padded)")
    print(f"raw frames : {raw.shape}  (real timestamps, uneven spacing)")

    print("\nOne model-input vector (step 0):")
    for name, v in zip(names, X[0]):
        print(f"  {name:<20} {v:9.3f}")

    table("Model input, first 10 steps (uniform grid):", grid, X, names, mask)
    table("Raw frames, first 10 (actual timestamps):", ts, raw, names)
    if mask.sum() < len(mask):
        last = int(np.nonzero(mask)[0][-1]) if mask.any() else -1
        print(f"\nPadding starts at step {last + 1} (t={grid[last + 1] / 1000:.1f}s); "
              "padded rows are zeros, not repeated frames.")


if __name__ == "__main__":
    main()
