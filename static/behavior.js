/*
 * Behaviour collector: keyboard, pointer (mouse / touchpad / touch / pen),
 * scrolling and tab focus. Buffers events in the page; nothing is sent per event.
 *
 * Privacy: keys are reduced to a category (char, bs, enter, ...). The actual
 * key or text is never stored, and password fields are ignored entirely.
 *
 * Scrolling is measured from how far content actually moved (scroll events),
 * so mouse wheel, touchpad, touchscreen swipe, keyboard and scrollbar drags all
 * count the same way. Raw wheel events are kept separately; Ctrl+wheel is a
 * touchpad pinch / zoom, not a scroll.
 *
 * If spelling.js is loaded, each window also carries spelling counts.
 *
 * Areas of interest: elements marked data-aoi="name" have their rectangles
 * (fractions of the viewport) recorded at the window start, every 0.5 s and
 * after scrolling/resizing, whenever they change.
 *
 *   Behavior.start()        begin a window
 *   Behavior.cut()  -> {}   return this window's events and start the next one
 *   Behavior.stop()
 *   Behavior.live() -> {}   running counts for the current window (for the UI)
 */
(function () {
  const now = () => performance.now();
  const GESTURE_GAP_MS = 300;   // scroll events closer than this belong to one gesture
  let on = false, t0 = 0, buf = null, live = null;
  const downAt = new Map();     // key code -> press time, only to compute hold time
  const lastPos = new WeakMap(); // scrollable element -> last scroll position
  let lastMove = 0, lastScroll = -1e9, lastZoom = -1e9, lastAoi = "", aoiTimer = 0;

  function fresh() {
    t0 = now();
    buf = { keys: [], pointer: [], wheel: [], scroll: [], focus: [], aoi: [],
            viewport: [innerWidth, innerHeight], initially_hidden: document.hidden };
    live = { keys: 0, corrections: 0, clicks: 0, scrolls: 0, zooms: 0, tab_switches: 0 };
    lastAoi = "";
    snapAoi();
  }
  const t = () => Math.round(now() - t0);

  // Rectangles of the marked regions, as fractions of the viewport
  function snapAoi() {
    const rects = {};
    for (const el of document.querySelectorAll("[data-aoi]")) {
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      (rects[el.dataset.aoi] = rects[el.dataset.aoi] || []).push(
        [r.left / innerWidth, r.top / innerHeight, r.right / innerWidth, r.bottom / innerHeight]
          .map(v => Math.round(v * 1e4) / 1e4));
    }
    const key = JSON.stringify(rects);
    if (key !== lastAoi) { buf.aoi.push([t(), rects]); lastAoi = key; }
  }
  const soonAoi = () => { clearTimeout(aoiTimer); aoiTimer = setTimeout(() => on && snapAoi(), 100); };
  setInterval(() => on && snapAoi(), 500);
  addEventListener("resize", soonAoi);

  function category(e) {
    const k = e.key;
    if (k === "Backspace") return "bs";
    if (k === "Delete") return "del";
    if (k === "Enter") return "enter";
    if (k === " ") return "space";
    if (k === "Tab") return "tab";
    if (["Shift", "Control", "Alt", "Meta", "CapsLock", "AltGraph", "Fn"].includes(k)) return "mod";
    if (/^(Arrow|Home|End|Page)/.test(k)) return "nav";
    if (k && k.length === 1) return "char";
    return "other";
  }
  const isPassword = e => e.target && e.target.type === "password";
  const ptype = e => ({ mouse: "m", touch: "t", pen: "p" })[e.pointerType] || "m";

  // Remember scroll positions of the element under an input and its ancestors,
  // so the first scroll after it has a known starting point.
  function seed(target) {
    for (let el = target; el && el.nodeType === 1; el = el.parentElement)
      if (!lastPos.has(el)) lastPos.set(el, [el.scrollLeft, el.scrollTop]);
    const root = document.scrollingElement;
    if (root && !lastPos.has(root)) lastPos.set(root, [root.scrollLeft, root.scrollTop]);
  }

  addEventListener("keydown", e => {
    seed(e.target);
    if (!on || isPassword(e)) return;
    const cat = category(e);
    buf.keys.push([t(), "d", cat, e.repeat ? 1 : 0]);
    if (!e.repeat) downAt.set(e.code, now());
    if (cat !== "mod" && !e.repeat) live.keys++;
    if (cat === "bs" || cat === "del") live.corrections++;
  }, true);

  addEventListener("keyup", e => {
    if (!on || isPassword(e)) return;
    const start = downAt.get(e.code);
    downAt.delete(e.code);
    buf.keys.push([t(), "u", category(e), start ? Math.round(now() - start) : 0]);
  }, true);

  addEventListener("pointermove", e => {
    if (!on) return;
    const n = now();
    if (n - lastMove < 16) return;          // ~60 Hz is plenty
    lastMove = n;
    buf.pointer.push([t(), "m", Math.round(e.clientX), Math.round(e.clientY), ptype(e), 0]);
  }, { capture: true, passive: true });

  addEventListener("pointerdown", e => {
    seed(e.target);
    if (!on) return;
    buf.pointer.push([t(), "d", Math.round(e.clientX), Math.round(e.clientY), ptype(e), e.button]);
    live.clicks++;
  }, { capture: true, passive: true });

  addEventListener("wheel", e => {
    seed(e.target);
    if (!on) return;
    const k = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? innerHeight : 1;  // lines/pages -> px
    const zoom = e.ctrlKey ? 1 : 0;          // touchpad pinch arrives as Ctrl+wheel
    buf.wheel.push([t(), Math.round(e.deltaX * k), Math.round(e.deltaY * k), zoom]);
    if (zoom) {
      const n = now();
      if (n - lastZoom > GESTURE_GAP_MS) live.zooms++;
      lastZoom = n;
    }
  }, { capture: true, passive: true });

  // Actual content movement, whatever caused it. Scroll events don't bubble,
  // but a capturing listener on document still sees scrolls inside elements.
  document.addEventListener("scroll", e => {
    const el = e.target === document ? document.scrollingElement : e.target;
    if (!el) return;
    const pos = [el.scrollLeft, el.scrollTop];
    const prev = lastPos.get(el);
    lastPos.set(el, pos);
    if (!on || !prev) return;
    const dx = Math.round(pos[0] - prev[0]), dy = Math.round(pos[1] - prev[1]);
    if (!dx && !dy) return;
    buf.scroll.push([t(), dx, dy]);
    const n = now();
    if (n - lastScroll > GESTURE_GAP_MS) live.scrolls++;
    lastScroll = n;
    soonAoi();
  }, { capture: true, passive: true });

  document.addEventListener("visibilitychange", () => {
    if (!on) return;
    buf.focus.push([t(), document.hidden ? "hidden" : "visible"]);
    if (document.hidden) live.tab_switches++;
  });
  addEventListener("blur", () => on && buf.focus.push([t(), "blur"]));
  addEventListener("focus", () => on && buf.focus.push([t(), "focus"]));

  window.Behavior = {
    start() { fresh(); seed(document.body); if (window.Spelling) Spelling.take(); on = true; },
    stop() { const out = this.cut(); on = false; return out; },
    cut() {
      const out = buf ? { ...buf, duration_ms: t(),
                          spell: window.Spelling ? Spelling.take() : null } : null;
      fresh();
      return out;
    },
    live: () => ({ ...live, ...(window.Spelling ? { words: Spelling.live().words, misspelled: Spelling.live().misspelled,
                                                    spell_status: Spelling.status } : {}) }),
  };
})();
