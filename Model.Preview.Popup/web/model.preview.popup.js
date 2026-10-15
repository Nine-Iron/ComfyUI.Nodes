// Model Preview Popup - frontend half.
// 1. Hovering a model in the Model Library shows its preview files large,
//    with a strip of the extras on top (hover or click one to view it large).
//    The popup sizes to each picture but never moves while you are in it, so
//    the strip stays under the pointer; moving to another model cancels the
//    pictures still loading for the last one.
// 2. When a model has preview files, the thumbnail embedded in the model file
//    is dropped from its metadata, so the sidebar icon keeps showing yours.
// 3. The popup shows the model's base model in large type, and its "Base
//    models" button picks which base models are listed in the Model Library and
//    in loader dropdowns. The choice is saved in ComfyUI's user settings.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const THUMB_KEYS = ["modelspec.thumbnail", "thumbnail", "image", "icon"];
const HIDE_DELAY = 250;
const SHOW_DELAY = 150; // the pointer must rest on a row this long, so sweeping
                        // across rows on the way to the popup does not switch it

const STYLE = `
.mpp-popup { position: fixed; z-index: 10000; display: none; width: 400px; max-width: calc(100vw - 24px);
  padding: 10px; border-radius: 8px; box-sizing: border-box;
  background: var(--comfy-menu-bg, #202020); color: var(--fg-color, #ddd);
  border: 1px solid var(--border-color, #444); box-shadow: 0 8px 24px rgba(0,0,0,.45);
  font: 12px/1.4 Inter, Arial, sans-serif; }
.mpp-head { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.mpp-base { flex: 1; font-size: 20px; font-weight: 800; line-height: 1.15; overflow-wrap: anywhere; }
.mpp-base small { font-size: 10px; font-weight: 400; color: var(--descrip-text, #999); margin-left: 6px; }
.mpp-btn { flex: none; padding: 3px 8px; border-radius: 4px; cursor: pointer; font: inherit; font-size: 11px;
  color: var(--fg-color, #ddd); background: var(--comfy-input-bg, #2a2a2a); border: 1px solid var(--border-color, #555); }
.mpp-btn:hover { border-color: var(--p-primary-color, #3b82f6); }
.mpp-title { font-weight: 600; font-size: 13px; margin-bottom: 2px; overflow-wrap: anywhere; }
.mpp-file { color: var(--descrip-text, #999); font-size: 11px; margin-bottom: 8px; overflow-wrap: anywhere; }
.mpp-stage { position: relative; width: 100%; min-height: 80px; border-radius: 4px;
  background: var(--comfy-input-bg, #111); overflow: hidden; }
.mpp-main { display: block; width: 100%; height: auto; max-height: var(--mpp-max, 52vh); object-fit: contain; }
.mpp-main:not([src]) { display: none; }
.mpp-stage.is-loading { height: 160px; }
.mpp-loading { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  color: var(--descrip-text, #888); font-size: 11px; pointer-events: none; }
.mpp-extras { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.mpp-extras:empty { display: none; }
.mpp-extras .mpp-slot { width: 56px; height: 56px; border-radius: 4px; background: var(--comfy-input-bg, #111); }
.mpp-extras img { width: 56px; height: 56px; object-fit: cover; border-radius: 4px; cursor: pointer;
  border: 2px solid transparent; opacity: .75; }
.mpp-extras img:hover { opacity: 1; }
.mpp-extras img.is-current { border-color: var(--p-primary-color, #3b82f6); opacity: 1; }
.mpp-meta { margin-top: 8px; color: var(--descrip-text, #aaa); overflow-wrap: anywhere; }
.mpp-meta b { color: var(--fg-color, #ddd); font-weight: 600; }
body.mpp-showing .model-lib-model-preview { display: none !important; }
.mpp-dialog-back { position: fixed; inset: 0; z-index: 10001; background: rgba(0,0,0,.45);
  display: flex; align-items: center; justify-content: center; }
.mpp-dialog { width: 380px; max-width: calc(100vw - 32px); max-height: calc(100vh - 64px); display: flex; flex-direction: column;
  background: var(--comfy-menu-bg, #202020); color: var(--fg-color, #ddd); border: 1px solid var(--border-color, #444);
  border-radius: 8px; box-shadow: 0 12px 32px rgba(0,0,0,.5); font: 13px/1.4 Inter, Arial, sans-serif; }
.mpp-dialog h3 { margin: 0; padding: 12px 14px 4px; font-size: 15px; }
.mpp-dialog p { margin: 0; padding: 0 14px 8px; color: var(--descrip-text, #999); font-size: 11px; }
.mpp-dlist { overflow-y: auto; padding: 0 8px; }
.mpp-drow { display: flex; align-items: center; gap: 8px; padding: 5px 6px; border-radius: 4px; cursor: pointer; }
.mpp-drow:hover { background: var(--comfy-input-bg, #2a2a2a); }
.mpp-drow b { flex: 1; }
.mpp-drow span { color: var(--descrip-text, #999); font-size: 11px; }
.mpp-dfoot { display: flex; gap: 6px; padding: 10px 14px; border-top: 1px solid var(--border-color, #444); }
.mpp-dfoot .mpp-btn:last-child { margin-left: auto; }
`;

const LIST_TTL = 30000; // re-check the folder after this, so new preview files show up
const listCache = new Map(); // "folder/filename" -> { at, promise }
const metaCache = new Map();

function previewsFor(folder, filename) {
  const key = `${folder}/${filename}`;
  const hit = listCache.get(key);
  if (hit && Date.now() - hit.at < LIST_TTL) return hit.promise;
  const q = new URLSearchParams({ folder, filename });
  const promise = api.fetchApi(`/Model.Preview.Popup/list?${q}`)
    .then((r) => (r.ok ? r.json() : {}))
    .then((d) => ({ images: Array.isArray(d.images) ? d.images : [], base: d.base || "unknown",
                    baseLabel: d.baseLabel || "Unknown", baseSource: d.baseSource || "none" }))
    .catch(() => { listCache.delete(key); return { images: [], base: "unknown", baseLabel: "Unknown", baseSource: "none" }; });
  listCache.set(key, { at: Date.now(), promise });
  return promise;
}

function imageUrl(folder, filename, name) {
  return api.apiURL(`/Model.Preview.Popup/image?${new URLSearchParams({ folder, filename, name })}`);
}

// --- keep the sidebar icon on the preview file -----------------------------
// The Model Library prefers a thumbnail embedded in the model file over the
// preview beside it. Drop the embedded one whenever a preview file exists.
function patchMetadata() {
  const original = api.viewMetadata?.bind(api);
  if (!original || api.viewMetadata.__mpp) return;
  const patched = async (folder, filename, ...rest) => {
    const meta = await original(folder, filename, ...rest);
    if (!meta || typeof meta !== "object") return meta;
    if (!THUMB_KEYS.some((k) => k in meta)) return meta;
    const { images } = await previewsFor(folder, filename);
    if (!images.length) return meta;
    const copy = { ...meta };
    for (const k of THUMB_KEYS) delete copy[k];
    return copy;
  };
  patched.__mpp = true;
  api.viewMetadata = patched;
}

async function metadataFor(folder, filename) {
  const key = `${folder}/${filename}`;
  if (!metaCache.has(key)) {
    metaCache.set(key, Promise.resolve(api.viewMetadata?.(folder, filename)).catch(() => null));
  }
  return metaCache.get(key);
}

// --- loading pictures, cancellable -------------------------------------------
// Each hovered model gets an AbortController; moving to another model aborts it,
// which cancels the list request and every picture still downloading. Pictures
// that did arrive are kept for the last few models, so going back is instant.
const BLOB_MODELS = 8;
const blobCache = new Map(); // "folder/filename" -> Map(name -> objectURL)
let loadCtl = null;

function blobsFor(key) {
  let m = blobCache.get(key);
  if (m) { blobCache.delete(key); blobCache.set(key, m); return m; } // most recent last
  m = new Map();
  blobCache.set(key, m);
  while (blobCache.size > BLOB_MODELS) {
    const [oldKey, old] = blobCache.entries().next().value;
    if (oldKey === key) break;
    for (const u of old.values()) URL.revokeObjectURL(u);
    blobCache.delete(oldKey);
  }
  return m;
}

async function loadImage(folder, filename, name, signal) {
  const cache = blobsFor(`${folder}/${filename}`);
  if (cache.has(name)) return cache.get(name);
  const r = await fetch(imageUrl(folder, filename, name), { signal, cache: "no-cache" });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const u = URL.createObjectURL(await r.blob());
  if (signal.aborted) { URL.revokeObjectURL(u); throw new DOMException("aborted", "AbortError"); }
  cache.set(name, u);
  return u;
}

function cancelLoading() {
  loadCtl?.abort();
  loadCtl = null;
}

// --- base model filter ------------------------------------------------------
const SETTING = "ModelPreviewPopup.HiddenBases";
let bases = { folders: {}, labels: {} }; // from /Model.Preview.Popup/bases
let byFile = new Map();                  // "folder/filename" and "filename" -> base tag
let basesLoaded = null;

const hiddenSet = () => new Set(app.extensionManager?.setting?.get(SETTING) || []);

async function loadBases() {
  const r = await api.fetchApi("/Model.Preview.Popup/bases");
  bases = r.ok ? await r.json() : { folders: {}, labels: {} };
  byFile = new Map();
  for (const [folder, files] of Object.entries(bases.folders || {})) {
    for (const [name, tag] of Object.entries(files)) {
      byFile.set(`${folder}/${name}`, tag);
      if (!byFile.has(name)) byFile.set(name, tag);
    }
  }
}
function ensureBases() {
  basesLoaded ??= loadBases().catch((e) => { basesLoaded = null; console.warn("[Model Preview Popup]", e); });
  return basesLoaded;
}

function isHidden(folder, name, hidden = hiddenSet()) {
  if (!hidden.size) return false;
  const tag = byFile.get(`${folder}/${name}`) ?? (folder ? undefined : byFile.get(name));
  return tag !== undefined && hidden.has(tag);
}

// Model Library: drop hidden models from the lists it loads.
function patchModelLists() {
  if (!api.getModels || api.getModels.__mpp) return;
  const original = api.getModels.bind(api);
  const patched = async (folder, ...rest) => {
    const list = await original(folder, ...rest);
    if (!Array.isArray(list) || !(folder in (bases.folders || {}))) return list;
    await ensureBases();
    const hidden = hiddenSet();
    return list.filter((m) => !isHidden(folder, typeof m === "string" ? m : m?.name, hidden));
  };
  patched.__mpp = true;
  api.getModels = patched;
}

// Loader dropdowns: show only visible models (the one already chosen stays).
function filterCombo(widget) {
  if (widget.type !== "combo" || !widget.options || widget.options.__mpp) return;
  let source = widget.options.values;
  Object.defineProperty(widget.options, "values", {
    configurable: true, enumerable: true,
    get() {
      const all = typeof source === "function" ? source() : source;
      if (!Array.isArray(all) || !byFile.size) return all;
      const hidden = hiddenSet();
      if (!hidden.size) return all;
      return all.filter((v) => typeof v !== "string" || v === widget.value || !isHidden("", v, hidden));
    },
    set(v) { source = v; },
  });
  widget.options.__mpp = true;
}
const filterNode = (node) => node?.widgets?.forEach(filterCombo);

function refreshLists() {
  // Reload the Model Library so it re-reads the (filtered) lists.
  const btn = [...document.querySelectorAll("button")].find((b) =>
    /refresh/i.test(b.getAttribute("aria-label") || b.title || "") && b.closest("[class*=model], .side-bar-panel, .sidebar-content-container"));
  btn?.click();
  app.graph?.setDirtyCanvas(true, true);
}

async function openBaseDialog() {
  hide();
  await (basesLoaded = loadBases()); // fresh counts every time
  const counts = new Map();
  for (const files of Object.values(bases.folders || {})) {
    for (const tag of Object.values(files)) counts.set(tag, (counts.get(tag) || 0) + 1);
  }
  const hidden = hiddenSet();
  const tags = [...counts.keys()].sort((a, b) => (a === "unknown") - (b === "unknown") || counts.get(b) - counts.get(a) || a.localeCompare(b));
  const back = Object.assign(document.createElement("div"), { className: "mpp-dialog-back" });
  const dlg = Object.assign(document.createElement("div"), { className: "mpp-dialog" });
  const h = Object.assign(document.createElement("h3"), { textContent: "Base models shown" });
  const p = Object.assign(document.createElement("p"), { textContent:
    "Unticked base models are hidden from the Model Library and loader dropdowns. Text encoders, VAEs and upscalers are always shown." });
  const list = Object.assign(document.createElement("div"), { className: "mpp-dlist" });
  const boxes = [];
  for (const tag of tags) {
    const row = Object.assign(document.createElement("label"), { className: "mpp-drow" });
    const box = Object.assign(document.createElement("input"), { type: "checkbox", checked: !hidden.has(tag) });
    box.dataset.tag = tag;
    const name = Object.assign(document.createElement("b"), { textContent: bases.labels?.[tag] || tag });
    const meta = Object.assign(document.createElement("span"), { textContent: `${tag} · ${counts.get(tag)}` });
    row.append(box, name, meta);
    list.appendChild(row);
    boxes.push(box);
  }
  const save = async () => {
    const keep = new Set(hiddenSet()); // keep hidden tags that are not on disk right now
    for (const b of boxes) { if (b.checked) keep.delete(b.dataset.tag); else keep.add(b.dataset.tag); }
    await app.extensionManager.setting.set(SETTING, [...keep].sort());
    refreshLists();
  };
  boxes.forEach((b) => b.addEventListener("change", save));
  const btn = (label, fn) => { const b = Object.assign(document.createElement("button"), { className: "mpp-btn", textContent: label }); b.addEventListener("click", fn); return b; };
  const close = () => back.remove();
  const foot = Object.assign(document.createElement("div"), { className: "mpp-dfoot" });
  foot.append(
    btn("All", () => { boxes.forEach((b) => (b.checked = true)); save(); }),
    btn("None", () => { boxes.forEach((b) => (b.checked = false)); save(); }),
    btn("Close", close));
  dlg.append(h, p, list, foot);
  back.appendChild(dlg);
  back.addEventListener("mousedown", (e) => { if (e.target === back) close(); });
  document.addEventListener("keydown", function esc(e) { if (e.key === "Escape") { close(); document.removeEventListener("keydown", esc); } });
  document.body.appendChild(back);
}

// --- the popup ---------------------------------------------------------------
let popup, baseEl, titleEl, fileEl, stage, mainImg, loadingEl, extrasEl, metaEl;
let hideTimer = null;
let currentKey = null;
let anchor = null;

function build() {
  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.appendChild(style);
  popup = document.createElement("div");
  popup.className = "mpp-popup";
  const head = Object.assign(document.createElement("div"), { className: "mpp-head" });
  baseEl = Object.assign(document.createElement("div"), { className: "mpp-base" });
  const basesBtn = Object.assign(document.createElement("button"), { className: "mpp-btn", textContent: "Base models",
    title: "Choose which base models are listed" });
  basesBtn.addEventListener("click", () => openBaseDialog().catch((e) => console.warn("[Model Preview Popup]", e)));
  head.append(baseEl, basesBtn);
  titleEl = Object.assign(document.createElement("div"), { className: "mpp-title" });
  fileEl = Object.assign(document.createElement("div"), { className: "mpp-file" });
  extrasEl = Object.assign(document.createElement("div"), { className: "mpp-extras" });
  stage = Object.assign(document.createElement("div"), { className: "mpp-stage" });
  mainImg = Object.assign(document.createElement("img"), { className: "mpp-main", alt: "" });
  loadingEl = Object.assign(document.createElement("div"), { className: "mpp-loading", textContent: "loading…" });
  stage.append(mainImg, loadingEl);
  metaEl = Object.assign(document.createElement("div"), { className: "mpp-meta" });
  popup.append(head, titleEl, fileEl, extrasEl, stage, metaEl);
  popup.addEventListener("mouseenter", () => clearTimeout(hideTimer));
  popup.addEventListener("mouseleave", scheduleHide);
  document.body.appendChild(popup);
}

const MAX_IMAGE = () => Math.min(window.innerHeight * 0.52, 520);

// Positioned once per model: the top edge is chosen so the popup fits even with
// the tallest picture, and the picture is capped to the room left below. The
// popup then only ever grows or shrinks at the bottom, never moves.
function place() {
  if (!anchor || popup.style.display === "none") return;
  const row = anchor.getBoundingClientRect();
  const panel = (anchor.closest(".p-tree, [class*=sidebar], .side-bar-panel") || anchor.parentElement).getBoundingClientRect();
  const chrome = popup.offsetHeight - stage.offsetHeight; // everything but the picture
  const w = popup.offsetWidth;
  const h = chrome + MAX_IMAGE();
  let left = panel.right + 8;
  if (left + w > window.innerWidth - 8) left = Math.max(8, panel.left - w - 8);
  const top = Math.max(8, Math.min(row.top - 20, window.innerHeight - h - 8));
  popup.style.left = `${left}px`;
  popup.style.top = `${top}px`;
  const room = Math.max(120, Math.min(MAX_IMAGE(), window.innerHeight - top - chrome - 8));
  popup.style.setProperty("--mpp-max", `${room}px`);
}

function hide() {
  cancelLoading();
  popup.style.display = "none";
  document.body.classList.remove("mpp-showing");
  currentKey = null;
  anchor = null;
}

function scheduleHide() {
  clearTimeout(hideTimer);
  hideTimer = setTimeout(hide, HIDE_DELAY);
}

function text(meta, ...keys) {
  for (const k of keys) if (meta && typeof meta[k] === "string" && meta[k].trim()) return meta[k].trim();
  return "";
}

async function show(leaf, folder, filename) {
  const key = `${folder}/${filename}`;
  clearTimeout(hideTimer);
  if (currentKey === key) return;
  cancelLoading(); // stop whatever the last model was still fetching
  const ctl = new AbortController();
  loadCtl = ctl;
  currentKey = key;
  const info = await previewsFor(folder, filename);
  if (ctl.signal.aborted) return; // moved on meanwhile
  const images = info.images;
  const meta = await metadataFor(folder, filename);
  if (ctl.signal.aborted) return;
  // No preview files: fall back to a thumbnail embedded in the model, if any.
  const embedded = images.length ? "" : text(meta, "modelspec.thumbnail", "thumbnail", "image", "icon");

  baseEl.replaceChildren(info.baseLabel);
  if (info.base !== "unknown" && info.baseLabel !== info.base) {
    baseEl.append(Object.assign(document.createElement("small"), { textContent: info.base }));
  }
  if (info.baseSource === "guess") baseEl.title = "Guessed from the file's contents";
  else baseEl.removeAttribute("title");

  const base = filename.split("/").pop();
  titleEl.textContent = text(meta, "modelspec.title", "title") || base.replace(/\.[^.]+$/, "");
  fileEl.textContent = `${folder}/${filename}`;

  // Thumbnail strip on top: one slot per picture, filled as each arrives.
  let current = 0;
  const urls = new Array(images.length).fill(null);
  const slots = images.map((name, i) => {
    const slot = document.createElement("div");
    slot.className = "mpp-slot";
    slot.title = name;
    return slot;
  });
  const select = (i) => {
    current = i;
    if (urls[i]) { mainImg.src = urls[i]; loadingEl.style.display = "none"; stage.classList.remove("is-loading"); }
    else { mainImg.removeAttribute("src"); loadingEl.style.display = ""; stage.classList.add("is-loading"); }
    extrasEl.querySelectorAll("img").forEach((el) => el.classList.toggle("is-current", Number(el.dataset.i) === i));
  };
  extrasEl.replaceChildren(...(images.length > 1 ? slots : []));
  mainImg.removeAttribute("src");
  loadingEl.style.display = "";
  stage.classList.add("is-loading");
  stage.style.display = images.length || embedded.startsWith("data:image") ? "" : "none";
  if (!images.length && embedded.startsWith("data:image")) {
    mainImg.src = embedded;
    loadingEl.style.display = "none";
    stage.classList.remove("is-loading");
  }

  const bits = [];
  const trigger = text(meta, "modelspec.trigger_phrase", "trigger_phrase");
  const author = text(meta, "modelspec.author", "author");
  const desc = text(meta, "modelspec.description", "description");
  metaEl.replaceChildren();
  for (const [label, value] of [["Trigger", trigger], ["Author", author], ["Description", desc]]) {
    if (!value) continue;
    const line = document.createElement("div");
    const b = document.createElement("b");
    b.textContent = `${label}: `;
    line.append(b, value.length > 400 ? value.slice(0, 400) + "…" : value);
    bits.push(line);
  }
  metaEl.append(...bits);
  metaEl.style.display = bits.length ? "" : "none";

  anchor = leaf;
  popup.style.display = "block";
  document.body.classList.add("mpp-showing");
  place();

  // Main picture first, then the extras in order, one at a time.
  for (let i = 0; i < images.length; i++) {
    let u;
    try {
      u = await loadImage(folder, filename, images[i], ctl.signal);
    } catch (err) {
      if (ctl.signal.aborted) return;
      continue; // a broken file just leaves its slot empty
    }
    if (ctl.signal.aborted) return;
    urls[i] = u;
    if (images.length > 1) {
      const t = Object.assign(document.createElement("img"), { src: u, title: images[i], alt: "" });
      t.dataset.i = String(i);
      t.addEventListener("click", () => select(i));
      t.addEventListener("mouseenter", () => select(i));
      slots[i].replaceWith(t);
    }
    if (i === current) select(i);
  }
}

// Model rows carry data-testid="tree-node-root/<folder>/<file path>"; only
// Model Library rows have the model icon container.
function modelOf(target) {
  const leaf = target?.closest?.('.tree-leaf[data-testid^="tree-node-root/"]');
  if (!leaf || !leaf.querySelector(".model-lib-model-icon-container")) return null;
  const path = leaf.dataset.testid.slice("tree-node-root/".length);
  const slash = path.indexOf("/");
  if (slash <= 0 || slash === path.length - 1) return null;
  return { leaf, folder: path.slice(0, slash), filename: path.slice(slash + 1) };
}

app.registerExtension({
  name: "Model.Preview.Popup",
  settings: [{ id: SETTING, name: "Model Preview Popup: hidden base models", type: "hidden", defaultValue: [] }],
  nodeCreated(node) { filterNode(node); },
  loadedGraphNode(node) { filterNode(node); },
  setup() {
    patchMetadata();
    patchModelLists();
    ensureBases();
    build();
    app.graph?.nodes?.forEach(filterNode);
    let showTimer = null;
    document.addEventListener("mouseover", (e) => {
      const m = modelOf(e.target);
      if (!m) return;
      clearTimeout(showTimer);
      const go = () => show(m.leaf, m.folder, m.filename).catch((err) => console.warn("[Model Preview Popup]", err));
      if (popup.style.display === "block") showTimer = setTimeout(go, SHOW_DELAY);
      else go();
    });
    document.addEventListener("mouseout", (e) => {
      const m = modelOf(e.target);
      if (m && !m.leaf.contains(e.relatedTarget)) {
        clearTimeout(showTimer);
        scheduleHide();
      }
    });
    window.addEventListener("scroll", (e) => { if (!popup.contains(e.target)) place(); }, true);
  },
});
