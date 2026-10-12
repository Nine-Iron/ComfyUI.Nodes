// Model Preview Popup - frontend half.
// 1. Hovering a model in the Model Library shows its preview files large,
//    extras included (click one to view it large).
// 2. When a model has preview files, the thumbnail embedded in the model file
//    is dropped from its metadata, so the sidebar icon keeps showing yours.

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
.mpp-title { font-weight: 600; font-size: 13px; margin-bottom: 2px; overflow-wrap: anywhere; }
.mpp-file { color: var(--descrip-text, #999); font-size: 11px; margin-bottom: 8px; overflow-wrap: anywhere; }
.mpp-main { display: block; width: 100%; max-height: 52vh; object-fit: contain; border-radius: 4px;
  background: var(--comfy-input-bg, #111); }
.mpp-extras { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.mpp-extras img { width: 56px; height: 56px; object-fit: cover; border-radius: 4px; cursor: pointer;
  border: 2px solid transparent; opacity: .75; }
.mpp-extras img:hover { opacity: 1; }
.mpp-extras img.is-current { border-color: var(--p-primary-color, #3b82f6); opacity: 1; }
.mpp-meta { margin-top: 8px; color: var(--descrip-text, #aaa); overflow-wrap: anywhere; }
.mpp-meta b { color: var(--fg-color, #ddd); font-weight: 600; }
body.mpp-showing .model-lib-model-preview { display: none !important; }
`;

const LIST_TTL = 30000; // re-check the folder after this, so new preview files show up
const listCache = new Map(); // "folder/filename" -> { at, promise }
const metaCache = new Map();

function previewsFor(folder, filename) {
  const key = `${folder}/${filename}`;
  const hit = listCache.get(key);
  if (hit && Date.now() - hit.at < LIST_TTL) return hit.promise;
  const q = new URLSearchParams({ folder, filename });
  const promise = api.fetchApi(`/model_preview_popup/list?${q}`)
    .then((r) => (r.ok ? r.json() : { images: [] }))
    .then((d) => (Array.isArray(d.images) ? d.images : []))
    .catch(() => { listCache.delete(key); return []; });
  listCache.set(key, { at: Date.now(), promise });
  return promise;
}

function imageUrl(folder, filename, name) {
  return api.apiURL(`/model_preview_popup/image?${new URLSearchParams({ folder, filename, name })}`);
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
    const images = await previewsFor(folder, filename);
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

// --- the popup ---------------------------------------------------------------
let popup, titleEl, fileEl, mainImg, extrasEl, metaEl;
let hideTimer = null;
let currentKey = null;

function build() {
  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.appendChild(style);
  popup = document.createElement("div");
  popup.className = "mpp-popup";
  titleEl = Object.assign(document.createElement("div"), { className: "mpp-title" });
  fileEl = Object.assign(document.createElement("div"), { className: "mpp-file" });
  mainImg = Object.assign(document.createElement("img"), { className: "mpp-main", alt: "" });
  extrasEl = Object.assign(document.createElement("div"), { className: "mpp-extras" });
  metaEl = Object.assign(document.createElement("div"), { className: "mpp-meta" });
  popup.append(titleEl, fileEl, mainImg, extrasEl, metaEl);
  popup.addEventListener("mouseenter", () => clearTimeout(hideTimer));
  popup.addEventListener("mouseleave", scheduleHide);
  mainImg.addEventListener("load", () => place());
  document.body.appendChild(popup);
}

let anchor = null;
function place() {
  if (!anchor || popup.style.display === "none") return;
  const row = anchor.getBoundingClientRect();
  const panel = (anchor.closest(".p-tree, [class*=sidebar], .side-bar-panel") || anchor.parentElement).getBoundingClientRect();
  const w = popup.offsetWidth, h = popup.offsetHeight;
  let left = panel.right + 8;
  if (left + w > window.innerWidth - 8) left = Math.max(8, panel.left - w - 8);
  const top = Math.max(8, Math.min(row.top - 20, window.innerHeight - h - 8));
  popup.style.left = `${left}px`;
  popup.style.top = `${top}px`;
}

function hide() {
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
  currentKey = key;
  const images = await previewsFor(folder, filename);
  if (currentKey !== key) return; // moved on meanwhile
  if (!images.length) { // nothing of ours to show: leave ComfyUI's own popup alone
    popup.style.display = "none";
    document.body.classList.remove("mpp-showing");
    return;
  }
  const meta = await metadataFor(folder, filename);
  if (currentKey !== key) return;

  const base = filename.split("/").pop();
  titleEl.textContent = text(meta, "modelspec.title", "title") || base.replace(/\.[^.]+$/, "");
  fileEl.textContent = `${folder}/${filename}`;
  const urls = images.map((n) => imageUrl(folder, filename, n));
  const select = (i) => {
    mainImg.src = urls[i];
    [...extrasEl.children].forEach((el, j) => el.classList.toggle("is-current", j === i));
  };
  extrasEl.replaceChildren();
  if (urls.length > 1) {
    urls.forEach((u, i) => {
      const t = Object.assign(document.createElement("img"), { src: u, title: images[i], alt: "" });
      t.addEventListener("click", () => select(i));
      t.addEventListener("mouseenter", () => select(i));
      extrasEl.appendChild(t);
    });
  }
  select(0);
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
  name: "model_preview_popup",
  setup() {
    patchMetadata();
    build();
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
    window.addEventListener("scroll", () => place(), true);
  },
});
