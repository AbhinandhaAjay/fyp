/*
 * Spell-check worker. Runs in the student's browser; words are never sent to
 * the server. A word counts as correct if either the US or the British
 * dictionary accepts it, so "colour" and "color" are both fine.
 *
 *   in:  { id, words: ["teh", "cloud", ...] }
 *        { allow: ["Priya", "photosynthesis", ...] }   extra accepted words
 *   out: { ready: true } once loaded, then { id, ok: [false, true, ...] }
 */
importScripts("vendor/typo/typo.js");

const DICTS = ["en_US", "en_GB"];
// Everyday / tech words the dictionaries don't know. Missing apostrophes
// ("dont", "im") are deliberately NOT here: they are real spelling errors.
const ALLOW = new Set(["ok", "wifi", "youtube", "covid", "tv", "pc", "eg", "ie",
                       "lol", "btw", "pls", "app", "apps", "online", "email"]);
let dicts = null;

async function load() {
  const base = "vendor/dictionaries";
  dicts = await Promise.all(DICTS.map(async code => {
    const [aff, dic] = await Promise.all(["aff", "dic"].map(ext =>
      fetch(`${base}/${code}/${code}.${ext}`).then(r => {
        if (!r.ok) throw new Error(`${code}.${ext}: HTTP ${r.status}`);
        return r.text();
      })));
    return new Typo(code, aff, dic);
  }));
}

function correct(word) {
  return ALLOW.has(word.toLowerCase()) ||
    dicts.some(d => d.check(word) || d.check(word.toLowerCase()));
}

const ready = load().then(
  () => postMessage({ ready: true }),
  err => postMessage({ ready: false, error: String(err) }));

onmessage = async e => {
  if (e.data.allow) {
    for (const w of e.data.allow) ALLOW.add(String(w).toLowerCase());
    return;
  }
  await ready;
  if (!dicts) return postMessage({ id: e.data.id, ok: null });
  postMessage({ id: e.data.id, ok: e.data.words.map(correct) });
};
