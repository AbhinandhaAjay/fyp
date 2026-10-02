/*
 * Gaze calibration screen.
 *
 *   const result = await Calibration.run(stream);   // null if cancelled
 *
 * Shows a dot at 9 positions (used to fit the mapping) and then 4 more
 * (held out, only used to measure accuracy) while recording the webcam.
 * Uploads the clip plus when each dot was where to /api/calibrations, then shows
 * targets vs predictions so the user can accept or redo.
 *
 * Positions are fractions of the browser viewport, so the window must not be
 * moved or resized after calibrating.
 */
(function () {
  const DWELL_MS = 1800;
  const CAL = [0.08, 0.5, 0.92].flatMap(y => [0.08, 0.5, 0.92].map(x => [x, y]));
  const VAL = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]];
  const shuffle = a => a.map(v => [Math.random(), v]).sort((p, q) => p[0] - q[0]).map(p => p[1]);
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  const CSS = `
  .cal-overlay { position: fixed; inset: 0; z-index: 1000; background: #23262d; color: #e8eaf0;
    font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; cursor: none; }
  .cal-overlay.show-cursor { cursor: default; }
  .cal-panel { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
    max-width: 520px; width: calc(100% - 32px); background: #2d3139; border-radius: 12px; padding: 24px; }
  .cal-panel h2 { margin: 0 0 10px; font-size: 19px; }
  .cal-panel ul { margin: 8px 0 16px; padding-left: 20px; color: #c3c8d4; }
  .cal-panel .row { display: flex; gap: 8px; flex-wrap: wrap; }
  .cal-panel button { font: inherit; padding: 8px 16px; border-radius: 8px; border: 1px solid #4a505c;
    background: #3a3f49; color: #e8eaf0; cursor: pointer; }
  .cal-panel button.go { background: #5b8ff0; border-color: #5b8ff0; color: #fff; }
  .cal-dot { position: absolute; width: 44px; height: 44px; margin: -22px 0 0 -22px; }
  .cal-dot::before { content: ""; position: absolute; inset: 0; border-radius: 50%; border: 3px solid #ffcc4d;
    animation: cal-shrink ${DWELL_MS}ms linear forwards; }
  .cal-dot::after { content: ""; position: absolute; left: 50%; top: 50%; width: 8px; height: 8px;
    margin: -4px 0 0 -4px; border-radius: 50%; background: #ffcc4d; }
  @keyframes cal-shrink { from { transform: scale(1); opacity: 1; } to { transform: scale(.25); opacity: .4; } }
  .cal-progress { position: absolute; left: 50%; bottom: 18px; transform: translateX(-50%);
    font-size: 13px; color: #8f97a8; }
  .cal-spinner { width: 28px; height: 28px; border: 3px solid #4a505c; border-top-color: #5b8ff0;
    border-radius: 50%; animation: cal-spin 1s linear infinite; margin-bottom: 10px; }
  @keyframes cal-spin { to { transform: rotate(360deg); } }`;

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function panel(overlay, html, buttons) {
    overlay.classList.add("show-cursor");
    overlay.querySelectorAll(".cal-panel").forEach(p => p.remove());
    const p = el("div", "cal-panel", html);
    const row = el("div", "row");
    return new Promise(resolve => {
      for (const [label, value, cls] of buttons) {
        const b = el("button", cls || "", label);
        b.onclick = () => resolve(value);
        row.appendChild(b);
      }
      if (buttons.length) p.appendChild(row);
      overlay.appendChild(p);
      if (!buttons.length) resolve(null);
    });
  }

  async function collect(overlay, stream) {
    overlay.classList.remove("show-cursor");
    overlay.querySelectorAll(".cal-panel").forEach(p => p.remove());
    const mime = ["video/webm;codecs=vp8", "video/webm"].find(m => MediaRecorder.isTypeSupported(m)) || "";
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks = [];
    rec.ondataavailable = e => e.data.size && chunks.push(e.data);
    const started = new Promise(r => (rec.onstart = r));
    const stopped = new Promise(r => (rec.onstop = r));
    rec.start(500);
    await started;
    const t0 = performance.now();
    await sleep(600);                               // let the recorder settle

    const plan = [...shuffle(CAL).map(p => [...p, "cal"]), ...shuffle(VAL).map(p => [...p, "val"])];
    const points = [];
    const progress = el("div", "cal-progress");
    overlay.appendChild(progress);
    for (let i = 0; i < plan.length; i++) {
      if (overlay.dataset.cancel) break;
      const [x, y, phase] = plan[i];
      progress.textContent = `${i + 1} / ${plan.length}${phase === "val" ? " · checking accuracy" : ""}`;
      const dot = el("div", "cal-dot");
      dot.style.left = x * innerWidth + "px";
      dot.style.top = y * innerHeight + "px";
      overlay.appendChild(dot);
      const start = performance.now() - t0;
      await sleep(DWELL_MS);
      points.push({ x, y, phase, t0: Math.round(start), t1: Math.round(performance.now() - t0) });
      dot.remove();
    }
    progress.remove();
    rec.stop();
    await stopped;
    if (overlay.dataset.cancel) return null;
    return { blob: new Blob(chunks, { type: rec.mimeType || "video/webm" }), points };
  }

  // Targets vs predictions as a small map of the browser window, for the result panel.
  function resultMap(r) {
    const [w, h] = r.viewport, pad = 0.12;
    const vb = [-pad * w, -pad * h, w * (1 + 2 * pad), h * (1 + 2 * pad)].join(" ");
    const s = Math.max(w, h) / 220;                 // keep marks readable when scaled down
    let g = `<rect x="0" y="0" width="${w}" height="${h}" fill="none" stroke="#6b7385" stroke-width="${s}"/>`;
    const pts = [...r.calibration_points.map(p => [p, "#8f97a8"]), ...r.validation_points.map(p => [p, "#ffcc4d"])];
    for (const [p, col] of pts) {
      const tx = p.x * w, ty = p.y * h;
      const px = Math.max(-pad, Math.min(1 + pad, p.pred_x)) * w, py = Math.max(-pad, Math.min(1 + pad, p.pred_y)) * h;
      g += `<line x1="${tx}" y1="${ty}" x2="${px}" y2="${py}" stroke="${col}" stroke-width="${s}" stroke-dasharray="${3 * s} ${2 * s}"/>
            <circle cx="${tx}" cy="${ty}" r="${5 * s}" fill="none" stroke="${col}" stroke-width="${1.3 * s}"/>
            <circle cx="${px}" cy="${py}" r="${2.6 * s}" fill="${col}"/>`;
    }
    return `<svg viewBox="${vb}" style="width:100%;height:auto;display:block;margin:6px 0 12px;
              background:#23262d;border-radius:8px">${g}</svg>`;
  }

  async function run(stream) {
    if (!document.getElementById("cal-style")) {
      const st = el("style"); st.id = "cal-style"; st.textContent = CSS; document.head.appendChild(st);
    }
    const overlay = el("div", "cal-overlay");
    document.body.appendChild(overlay);
    const onKey = e => { if (e.key === "Escape") overlay.dataset.cancel = "1"; };
    addEventListener("keydown", onKey);
    try {
      for (;;) {
        const go = await panel(overlay, `
          <h2>Gaze calibration</h2>
          <p>A yellow dot will appear in ${CAL.length + VAL.length} places, about 2 seconds each (≈ 25 s).</p>
          <ul>
            <li>Sit as you normally will while studying, face the camera, good light on your face.</li>
            <li><b>Follow the dot with your eyes</b> and look at its centre until it moves.</li>
            <li>Keep your head fairly still. Don't move or resize this browser window afterwards.</li>
            <li>Press Esc to stop.</li>
          </ul>`, [["Start", true, "go"], ["Cancel", false]]);
        if (!go) return null;

        const data = await collect(overlay, stream);
        if (!data) return null;

        panel(overlay, `<div class="cal-spinner"></div>Working out where you were looking…`, []);
        const fd = new FormData();
        fd.append("video", data.blob, "calibration.webm");
        fd.append("points", JSON.stringify(data.points));
        fd.append("viewport", JSON.stringify([innerWidth, innerHeight]));
        let r;
        try {
          const resp = await fetch("/api/calibrations", { method: "POST", body: fd });
          r = await resp.json();
          if (!resp.ok) throw new Error(r.detail || resp.status);
        } catch (err) {
          const again = await panel(overlay, `<h2>Calibration failed</h2><p>${err.message}</p>`,
                                    [["Try again", true, "go"], ["Cancel", false]]);
          if (!again) return null;
          continue;
        }

        const pct = r.validation_error_pct_of_diagonal;
        const verdict = { good: "Good", fair: "Fair: fine for regions of the page", poor: "Poor: try again" }[r.quality];
        const keep = await panel(overlay, `
          <h2>${verdict}</h2>
          <p>Average error on the 4 test dots: <b>${Math.round(r.validation_error_px)} px</b>
             (${pct.toFixed(1)}% of the screen diagonal). Always guessing the centre would be off by
             ${Math.round(r.baseline_error_px)} px, so this is <b>${Math.round(Math.max(0, r.skill) * 100)}%</b>
             better than guessing.</p>
          ${resultMap(r)}
          <p style="color:#c3c8d4;font-size:13px">Rings are where the dot was; filled dots are where we estimated
             you looked; the box is this window. Yellow = test dots (not used for fitting), grey = calibration dots.
             ${r.points_used < r.points_total ? `${r.points_total - r.points_used} dot(s) had no usable frames.` : ""}</p>`,
          [["Use this calibration", "keep", "go"], ["Redo", "redo"], ["Cancel", "cancel"]]);
        if (keep === "keep") return r;
        if (keep === "cancel") return null;
      }
    } finally {
      removeEventListener("keydown", onKey);
      overlay.remove();
    }
  }

  window.Calibration = { run };
})();
