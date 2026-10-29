/* Application-state tests with a DOM/Canvas double; this is not a browser layout test. */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'),
  manifest = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/manifest.json')));
const context = new Proxy({
  measureText: t => ({
    width: String(t).length * 8
  })
}, {
  get: (o, k) => k in o ? o[k] : () => {},
  set: (o, k, v) => (o[k] = v, true)
});
class Element {
  constructor(id = '') {
    this.id = id;
    this.value = id === 'rate' ? '1' : '';
    this.checked = id === 'predictions';
    this.children = [];
    this.attributes = {};
    this.style = {};
    this.hidden = false;
  }
  append(...children) {
    this.children.push(...children);
    if (['method', 'track', 'seed', 'focus'].includes(this.id) && !this.value && children[0]) this.value = String(children[0].value);
  }
  replaceChildren(...children) {
    this.children = [];
    this.value = '';
    this.append(...children);
  }
  get options() {
    return this.children;
  }
  setAttribute(k, v) {
    this.attributes[k] = v;
  }
  getBoundingClientRect() {
    return {
      width: 1000,
      height: 500
    };
  }
  getContext() {
    return context;
  }
}
const html = fs.readFileSync(path.join(root, 'docs/index.html'), 'utf8'),
  ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map(m => m[1])),
  nodes = new Map();
const get = id => {
  assert.ok(ids.has(id), `Missing HTML node ${id}`);
  if (!nodes.has(id)) nodes.set(id, new Element(id));
  return nodes.get(id);
};
let frame;
const handlers = {};
const document = {
  getElementById: get,
  createElement: () => new Element(),
  createTextNode: text => ({
    textContent: text
  }),
  hidden: false,
  addEventListener: (e, cb) => handlers[e] = cb
};
const window = {
  RACE_MANIFEST: manifest,
  devicePixelRatio: 1,
  addEventListener() {}
};
const sandbox = {
  window,
  document,
  Math,
  Blob,
  Response,
  DecompressionStream,
  Uint8Array,
  TextDecoder,
  requestAnimationFrame: cb => frame = cb,
  fetch: async url => {
    const b = fs.readFileSync(path.join(root, 'docs', url));
    return {
      ok: true,
      arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)
    };
  }
};
for (const name of ['scene.js', 'app.js']) vm.runInNewContext(fs.readFileSync(path.join(root, 'docs', name), 'utf8'), sandbox);
async function ready() {
  for (let i = 0; i < 2000 && !get('load-status').hidden; i++) await new Promise(r => setTimeout(r, 5));
  assert.ok(get('load-status').hidden, get('load-status').textContent);
}
await ready();
assert.equal(window.RaceViewer.getState().playing, false);
assert.equal(window.RaceViewer.getState().time, 0);
let stamp = 0;
const tick = () => frame(stamp += 16.667);
for (const method of [...new Set(manifest.runs.map(r => r.method))]) {
  get('method').value = method;
  get('method').onchange();
  await ready();
  assert.equal(window.RaceViewer.getState().method, method);
  window.RaceViewer.setTime(.4);
  assert.equal(window.RaceViewer.getState().time, .4);
  get('focus').value = '3';
  get('focus').onchange();
  assert.equal(window.RaceViewer.getState().selected, 3);
}
get('play').onclick();
tick();
tick();
assert.ok(window.RaceViewer.getState().time > .4);
document.hidden = true;
handlers.visibilitychange();
const t = window.RaceViewer.getState().time;
tick();
assert.equal(window.RaceViewer.getState().time, t);
document.hidden = false;
window.RaceViewer.setTime(Number(get('timeline').max) - .005);
tick();
tick();
assert.equal(window.RaceViewer.getState().playing, false);
const end = window.RaceViewer.getState().time;
tick();
assert.equal(window.RaceViewer.getState().time, end);
get('track').value = 'Sepang';
get('track').onchange();
await ready();
assert.equal(window.RaceViewer.getState().time, 0);
assert.ok(!html.includes('ghost'));
assert.ok(!ids.has('scenario'));
assert.ok(!ids.has('probe'));
console.log('Verified method/circuit selection, loading compressed replays, focus, seek, playback, hidden-tab time freeze and no automatic loop. Browser layout remains unverified.');
