/*
 * Spelling tracker for text fields (textarea, text/search inputs).
 *
 * Words are checked locally by spell-worker.js. Only counts leave this file:
 *   words      words finished (followed by a space, punctuation or Enter)
 *   misspelled misspelled words that appeared in the text
 *   fixed      misspelled words that later disappeared (corrected or deleted)
 *   remaining  misspelled words sitting in the text when the window ended
 *
 * The word still under the caret is never judged (it's being typed). Every
 * finished word is checked, capitalised or not. Words that appear in the
 * page's own content (elements marked data-spell-vocab, or the whole page if
 * none are marked) are accepted, so lesson names and terms aren't flagged.
 *
 *   Spelling.take() -> {ready, words, misspelled, fixed, remaining}, resets counts
 *   Spelling.allow(["Priya", ...])   accept extra words (e.g. student names)
 *   Spelling.status -> "loading" | "ready" | "unavailable"
 */
(function () {
  const WORD = /[A-Za-z]+(?:'[A-Za-z]+)*/g;
  const BOUNDARY = /[\s.,;:!?)"\]]/;
  const IDLE_CHECK_MS = 1000;
  const fields = new Map();     // field -> { misspelled: Map(word -> count), resync, timer }
  const pending = new Map();    // request id -> callback
  let worker = null, nextId = 1;
  let counts = { words: 0, misspelled: 0, fixed: 0 };

  const api = {
    status: "loading", take, live: () => ({ ...counts }),
    allow: words => worker && worker.postMessage({ allow: [...words] }),
  };
  window.Spelling = api;

  // Words used in the lesson itself (rendered text only; text-box contents
  // are form values and are not part of innerText)
  function pageVocab() {
    const roots = document.querySelectorAll("[data-spell-vocab]");
    const words = new Set();
    for (const root of roots.length ? roots : [document.body])
      for (const m of (root.innerText || "").matchAll(WORD)) words.add(m[0]);
    return words;
  }

  try {
    worker = new Worker("spell-worker.js");
    worker.onmessage = e => {
      const m = e.data;
      if ("ready" in m) {
        api.status = m.ready ? "ready" : "unavailable";
        if (m.ready) {
          api.allow(pageVocab());    // sent first; the worker handles messages in order
          // fields touched while the dictionaries were loading get their baseline now
          for (const el of fields.keys()) check(el, true);
        }
        return;
      }
      const cb = pending.get(m.id);
      pending.delete(m.id);
      if (cb && m.ok) cb(m.ok);
    };
    worker.onerror = () => { api.status = "unavailable"; };
  } catch (err) { api.status = "unavailable"; }

  const isTextField = el => el && (el.tagName === "TEXTAREA" ||
    (el.tagName === "INPUT" && ["text", "search", ""].includes(el.type)));

  function state(el) {
    // base: misspellings that were already there (pasted / pre-filled), not typed
    if (!fields.has(el)) fields.set(el, { misspelled: new Map(), base: new Map(), resync: true, timer: 0 });
    return fields.get(el);
  }

  // Words worth judging: finished ones, at least 2 letters.
  function candidates(el, includeCaretWord) {
    const text = el.value, caret = document.activeElement === el ? el.selectionEnd : -1;
    const out = [];
    for (const m of text.matchAll(WORD)) {
      const w = m[0], end = m.index + w.length;
      if (w.length < 2) continue;
      if (!includeCaretWord && end === caret) continue;
      out.push(w);
    }
    return out;
  }

  function check(el, includeCaretWord = false) {
    if (api.status !== "ready") return;
    const st = state(el);
    clearTimeout(st.timer);
    const words = candidates(el, includeCaretWord);
    const id = nextId++;
    pending.set(id, ok => {
      const now = new Map();
      words.forEach((w, i) => {
        if (!ok[i]) { const k = w.toLowerCase(); now.set(k, (now.get(k) || 0) + 1); }
      });
      if (st.resync) {
        // everything misspelled right now was not typed by the student
        for (const [k, n] of now) st.base.set(k, Math.max(n - (st.misspelled.get(k) || 0), 0) + (st.base.get(k) || 0));
      } else {
        // only count misspellings above the baseline as the student's own
        const keys = new Set([...now.keys(), ...st.misspelled.keys()]);
        for (const k of keys) {
          const base = st.base.get(k) || 0;
          const before = Math.max(0, (st.misspelled.get(k) || 0) - base);
          const after = Math.max(0, (now.get(k) || 0) - base);
          counts.misspelled += Math.max(0, after - before);
          counts.fixed += Math.max(0, before - after);
          if ((now.get(k) || 0) < base) st.base.set(k, now.get(k) || 0);  // pasted error removed
        }
      }
      st.misspelled = now;
      st.resync = false;
    });
    worker.postMessage({ id, words });
  }

  document.addEventListener("input", e => {
    const el = e.target;
    if (!isTextField(el)) return;
    const st = state(el);
    // Pasted or dropped text wasn't typed: take it as the new baseline
    if (/^insertFrom(Paste|Drop)/.test(e.inputType || "")) { st.resync = true; return check(el, true); }
    const ch = e.inputType === "insertLineBreak" ? "\n" : e.data || "";
    if (ch && BOUNDARY.test(ch.slice(-1))) {
      const before = el.value.slice(0, el.selectionEnd - ch.length);
      if (/[A-Za-z]$/.test(before)) counts.words++;
      return check(el);
    }
    clearTimeout(st.timer);
    st.timer = setTimeout(() => check(el), IDLE_CHECK_MS);
  }, true);

  document.addEventListener("focusout", e => { if (isTextField(e.target)) check(e.target, true); }, true);
  document.addEventListener("focusin", e => {
    if (isTextField(e.target) && !fields.has(e.target)) { state(e.target); check(e.target, true); }
  }, true);

  function take() {
    let remaining = 0;
    for (const [el, st] of fields) {
      if (!el.isConnected) { fields.delete(el); continue; }
      for (const [k, n] of st.misspelled) remaining += Math.max(0, n - (st.base.get(k) || 0));
    }
    const out = { ready: api.status === "ready", ...counts, remaining };
    counts = { words: 0, misspelled: 0, fixed: 0 };
    return out;
  }
})();
