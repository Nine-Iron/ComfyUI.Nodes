import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Image Recall settings panel.
// - Shown on the "Image Recall - Outpaint & Upscale" subgraph node (or on a plain Image Recall node).
// - Drop an image anywhere in the window (Shift+drop = ComfyUI's normal "load workflow").
// - Every setting the image recorded is listed and editable; edits go into the inner nodes' widgets.

const RECALL = "ImageRecall";
const FROM_IMAGE = "(from image)";
const MODES = ["outpaint", "outpaint + upscale", "upscale"];
const ASPECTS = ["16:9", "9:16", "21:9", "9:21", "4:3", "3:4", "3:2", "2:3", "1:1"];
const ANCHORS = ["center", "left", "right", "top", "bottom"];
const BUCKETS = { "16:9": [1344, 768], "9:16": [768, 1344], "21:9": [1536, 640], "9:21": [640, 1536],
  "4:3": [1152, 896], "3:4": [896, 1152], "3:2": [1216, 832], "2:3": [832, 1216], "1:1": [1024, 1024] };
const TEXT_DEFAULT_H = 60;               // the frontend's default multiline height (px)
const TEXT_MIN_H = TEXT_DEFAULT_H * 3;   // every text field: at least 3x that
const SPEED_KEY = "ImageRecall.secPerMPStep";
const THUMB_KEY = "ImageRecall.thumbSize";
const HISTORY_FILE = "image_recall_history.json"; // ~/ComfyUI/user/default/image_recall_history.json

// ---------------------------------------------------------------- helpers
const widget = (node, name) => node?.widgets?.find((w) => w.name === name);
const innerNodes = (node) => node?.subgraph?.nodes ?? node?.subgraph?._nodes ?? [];

function setWidget(node, name, value) {
  const w = widget(node, name);
  if (!w) return;
  if (w.options?.values && Array.isArray(w.options.values) && !w.options.values.includes(value)) {
    w.options.values.push(value);
  }
  w.value = value;
  node.graph?.setDirtyCanvas?.(true, true);
}

function targetOf(node) {
  if (!node) return null;
  if (node.type === RECALL) return { host: node, nodes: [node], recall: node };
  if (node.isSubgraphNode?.()) {
    const nodes = innerNodes(node);
    const recall = nodes.find((n) => n.type === RECALL);
    if (recall) return { host: node, nodes, recall };
  }
  return null;
}

function findTarget() {
  for (const graph of [app.canvas?.graph, app.graph]) {
    for (const n of graph?.nodes ?? graph?._nodes ?? []) {
      const t = targetOf(n);
      if (t) return t;
    }
  }
  return null;
}

const byType = (t, type) => t.nodes.find((n) => n.type === type);
const step2 = (t) => t.nodes.find((n) => n.type === "KSampler" && /step 2/i.test(n.title ?? ""));
const step1 = (t) => t.nodes.find((n) => n.type === "KSampler" && /step 1/i.test(n.title ?? ""));

function readSettings(t) {
  try {
    const s = JSON.parse(widget(t.recall, "settings")?.value || "{}");
    return s && typeof s === "object" ? s : {};
  } catch {
    return {};
  }
}
function writeSettings(t, s) {
  setWidget(t.recall, "settings", JSON.stringify(s));
}

// 2 model calls per step for these samplers (heunpp2: 3)
function callsPerStep(sampler) {
  if (/^heunpp2$/.test(sampler)) return 3;
  return /^(dpmpp_2s_ancestral|dpmpp_sde|dpm_2|heun)/.test(sampler ?? "") ? 2 : 1;
}
const ceil64 = (v) => Math.ceil(v / 64) * 64;
const round8 = (v) => Math.max(8, Math.round(v / 8) * 8);
// Mirrors layout() in __init__.py: the source's size on the outpaint canvas.
function placedSize(srcW, srcH, aspect, rw, rh) {
  const [W, H] = BUCKETS[aspect] ?? [1344, 768];
  let tw, th;
  if (rw > 0 && rh > 0) [tw, th] = [rw, rh];
  else if (rw > 0) [tw, th] = [rw, (rw * srcH) / srcW];
  else if (rh > 0) [tw, th] = [(rh * srcW) / srcH, rh];
  else [tw, th] = [srcW * 1e9, srcH * 1e9];
  const k = Math.min(1, W / tw, H / th);
  return { w: Math.min(W, round8(tw * k)), h: Math.min(H, round8(th * k)), W, H };
}
function upscaleSize(srcW, srcH, aspect, longSide) {
  let [ra, rb] = aspect in BUCKETS ? aspect.split(":").map(Number) : [srcW, srcH];
  const [w, h] = ra >= rb ? [longSide, round8((longSide * rb) / ra)] : [round8((longSide * ra) / rb), longSide];
  return { rw: ceil64(w), rh: ceil64(h), w, h };
}

// ---------------------------------------------------------------- LoRA tags (via LoRA Tag Picker)
const PICKER = "LoraTagPicker";
const AUTO_ON = 3; // same rule as the picker: first 3 triggers up front start on
const tagCache = new Map();
async function loraTags(name) {
  if (!tagCache.has(name)) {
    tagCache.set(name, api.fetchApi(`/Lora.Tag.Picker/tags?name=${encodeURIComponent(name)}`)
      .then((r) => (r.ok ? r.json() : { tags: [] }))
      .catch(() => ({ tags: [] })));
  }
  return tagCache.get(name);
}
function pickerState(picker) {
  try {
    const v = JSON.parse(widget(picker, "selection")?.value || "{}");
    return { v: 1, overrides: v?.overrides && typeof v.overrides === "object" ? v.overrides : {} };
  } catch {
    return { v: 1, overrides: {} };
  }
}
function frontOf(info, maxTags) {
  const front = (info.tags ?? []).filter((t) => t.tier !== "rare").slice(0, maxTags);
  const auto = new Set(front.filter((t) => t.tier === "trigger").slice(0, AUTO_ON).map((t) => t.tag));
  return { front, auto };
}

// ---------------------------------------------------------------- recall + upload
async function fetchRecall(t, filename) {
  const s = readSettings(t);
  const q = new URLSearchParams({
    filename,
    prompt_prefix: String(widget(t.recall, "prompt_prefix")?.value ?? ""),
    subject: String(widget(t.recall, "subject")?.value ?? ""),
    subject_weight: String(widget(t.recall, "subject_weight")?.value ?? 0.4),
  });
  const res = await api.fetchApi(`/image_recall/recall?${q}`);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

// Fresh settings from what the image recorded (drop / "Reset to image").
function settingsFromRecall(t, r) {
  const mode = widget(t.recall, "mode")?.value ?? MODES[1];
  return {
    checkpoint: r.checkpoint || FROM_IMAGE,
    loras: r.loras.map((l) => ({ name: l.name, on: l.on, model: l.model, clip: l.clip })),
    positive: mode === "upscale" ? r.positive : r.outpaint_positive,
    negative: r.negative,
    seed: r.seed, steps: r.steps, cfg: r.cfg, sampler: r.sampler, scheduler: r.scheduler,
  };
}

// Turn on exactly the LoRA tags the image's own LoRA Tag Picker used (and its before/after setting).
// No picker in the image: back to the picker's defaults.
async function applyRecalledTags(t, r) {
  const picker = byType(t, PICKER);
  if (!picker) return;
  const lt = r.lora_tags;
  if (!lt?.resolved?.length) {
    setWidget(picker, "selection", JSON.stringify({ v: 1, overrides: {} }));
    return;
  }
  if (lt.position) setWidget(picker, "tag_position", lt.position);
  const maxTags = Number(widget(picker, "max_tags")?.value ?? 8);
  const overrides = {};
  for (const e of lt.resolved) {
    const { auto } = frontOf(await loraTags(e.lora), maxTags);
    const want = new Set(e.tags);
    const o = {};
    for (const tag of want) if (!auto.has(tag)) o[tag] = true;
    for (const tag of auto) if (!want.has(tag)) o[tag] = false;
    if (Object.keys(o).length) overrides[e.lora] = o;
  }
  setWidget(picker, "selection", JSON.stringify({ v: 1, overrides }));
}

function applyDefaultsForImage(t, r) {
  const s2 = step2(t);
  if (s2) setWidget(s2, "steps", r.steps);
  syncUpscaleAspect(t);
}

// Step 2's shape: "auto" = the picture's own shape in upscale mode, the outpaint aspect otherwise.
const UP_AUTO = "auto";
const upShape = (t) => t.host.properties?.imageRecallUpAspect ?? UP_AUTO;
const renderScale = (t) => t.nodes.find((n) => n.type === "ImageScale" && /render size/i.test(n.title ?? ""));
function syncUpscaleAspect(t) {
  const mode = widget(t.recall, "mode")?.value;
  const canvas = byType(t, "OutpaintCanvas");
  const up = byType(t, "UpscaleSize");
  const auto = mode === "upscale" ? FROM_IMAGE : widget(canvas, "aspect")?.value ?? "16:9";
  if (up) setWidget(up, "aspect", upShape(t) === UP_AUTO ? auto : upShape(t));
}

// Blend slider (0..100): keep the dropped picture <-> let the generated scene eat it.
// 0..70 widens the feather from 8 px to half the picture; 70..100 also redraws the picture itself (up to 0.85).
const BLEND_FEATHER_END = 70;
const BLEND_REPAINT_MAX = 0.85;
function blendParams(b, placed) {
  const maxF = Math.max(8, Math.floor(Math.min(placed.w, placed.h) / 2 / 8) * 8);
  const f = Math.min(1, b / BLEND_FEATHER_END);
  const feather = Math.round((8 + (maxF - 8) * f) / 8) * 8;
  const repaint = b <= BLEND_FEATHER_END ? 0
    : Math.round(((b - BLEND_FEATHER_END) / (100 - BLEND_FEATHER_END)) * BLEND_REPAINT_MAX * 100) / 100;
  return { feather, repaint, maxF };
}
// Slider position for an existing feather/repaint (workflows saved before the slider existed).
function blendFrom(feather, repaint, placed) {
  if (repaint > 0) return BLEND_FEATHER_END + (repaint / BLEND_REPAINT_MAX) * (100 - BLEND_FEATHER_END);
  const { maxF } = blendParams(0, placed);
  return Math.max(0, Math.min(BLEND_FEATHER_END, ((feather - 8) / Math.max(1, maxF - 8)) * BLEND_FEATHER_END));
}

async function uploadImage(file) {
  const body = new FormData();
  body.append("image", file);
  const res = await api.fetchApi("/upload/image", { method: "POST", body });
  if (!res.ok) throw new Error(`upload failed: ${res.status}`);
  const j = await res.json();
  return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
}

// ---------------------------------------------------------------- history + lock
// History: every imported image with the settings it was last used with, kept in ComfyUI's user data.
let history = null;
let historyLoad = null;
let historySave = null;
async function loadHistory() {
  if (history) return history;
  historyLoad ??= (async () => {
    try {
      const res = await api.getUserData(HISTORY_FILE);
      history = res.ok ? await res.json() : [];
    } catch {
      history = [];
    }
    if (!Array.isArray(history)) history = [];
    return history;
  })();
  return historyLoad;
}
function saveHistory() {
  clearTimeout(historySave);
  historySave = setTimeout(() => {
    api.storeUserData(HISTORY_FILE, history, { overwrite: true, stringify: true, throwOnError: false })
      .catch((err) => console.warn("Image Recall: could not save history", err));
  }, 500);
}

function snapshot(t) {
  const v = (n, k) => widget(n, k)?.value;
  const canvas = byType(t, "OutpaintCanvas");
  const up = byType(t, "UpscaleSize");
  const s2 = step2(t);
  const picker = byType(t, PICKER);
  return {
    settings: readSettings(t),
    mode: v(t.recall, "mode"), subject: v(t.recall, "subject"), subject_weight: v(t.recall, "subject_weight"),
    prompt_prefix: v(t.recall, "prompt_prefix"),
    canvas: canvas ? { aspect: v(canvas, "aspect"), anchor: v(canvas, "anchor"), feather: v(canvas, "feather"),
      resize_width: v(canvas, "resize_width"), resize_height: v(canvas, "resize_height"),
      repaint_original: v(canvas, "repaint_original") } : undefined,
    upscale: up ? { long_side: v(up, "long_side") } : undefined,
    render: renderScale(t) ? { crop: v(renderScale(t), "crop") } : undefined,
    props: { imageRecallBlend: t.host.properties?.imageRecallBlend, imageRecallUpAspect: t.host.properties?.imageRecallUpAspect,
      imageRecallSeedMode: t.host.properties?.imageRecallSeedMode },
    step2: s2 ? { steps: v(s2, "steps"), denoise: v(s2, "denoise") } : undefined,
    tags: picker ? { selection: v(picker, "selection"), tag_position: v(picker, "tag_position") } : undefined,
  };
}
function restore(t, snap) {
  const put = (node, obj) => {
    if (!node || !obj) return;
    for (const [k, val] of Object.entries(obj)) if (val !== undefined && val !== null) setWidget(node, k, val);
  };
  writeSettings(t, snap.settings ?? {});
  put(t.recall, { mode: snap.mode, subject: snap.subject, subject_weight: snap.subject_weight, prompt_prefix: snap.prompt_prefix });
  put(byType(t, "OutpaintCanvas"), snap.canvas);
  put(byType(t, "UpscaleSize"), snap.upscale);
  put(step2(t), snap.step2);
  put(renderScale(t), snap.render);
  put(byType(t, PICKER), snap.tags);
  if (snap.props) {
    const props = (t.host.properties ??= {});
    for (const [k, val] of Object.entries(snap.props)) {
      if (val === undefined || val === null) delete props[k];
      else props[k] = val;
    }
  }
  syncUpscaleAspect(t);
}

// Store the current image's settings in its history entry (bump = move it to the top as just used).
function remember(t, { bump = false, size } = {}) {
  if (!history) return;
  const filename = widget(t.recall, "image")?.value;
  if (!filename || filename === "example.png") return;
  const i = history.findIndex((h) => h.filename === filename);
  const now = Date.now();
  const e = i >= 0 ? history[i] : { filename, added: now, used: now };
  e.snapshot = snapshot(t);
  if (size) e.size = size;
  if (bump || i < 0) {
    e.used = now;
    if (i >= 0) history.splice(i, 1);
    history.unshift(e);
  }
  saveHistory();
}

const isLocked = (t) => !!t.host.properties?.imageRecallLock;

// Seed control after each queued run, like ComfyUI's control_after_generate.
const SEED_MODES = ["fixed", "increment", "decrement", "randomize"];
const SEED_MAX = 2 ** 50; // ComfyUI's own random range (safe in JS numbers)
const seedMode = (t) => (SEED_MODES.includes(t.host.properties?.imageRecallSeedMode) ? t.host.properties.imageRecallSeedMode : "fixed");
const randomSeed = () => Math.floor(Math.random() * SEED_MAX);
function stepSeed(t) {
  const mode = seedMode(t);
  if (mode === "fixed") return;
  const s = readSettings(t);
  const cur = Number.isFinite(Number(s.seed)) ? Number(s.seed) : 0;
  s.seed = mode === "randomize" ? randomSeed()
    : mode === "increment" ? (cur + 1) % SEED_MAX
    : (cur - 1 + SEED_MAX) % SEED_MAX;
  writeSettings(t, s);
  remember(t);
  t.host.imageRecallPanel?.render(null);
}
function allTargets() {
  const out = [];
  for (const n of app.graph?.nodes ?? app.graph?._nodes ?? []) {
    const t = n.imageRecallTarget ?? targetOf(n);
    if (t) out.push(t);
  }
  return out;
}
// The run was already built with the current seed; change it for the next one (once per queued run).
if (!api.imageRecallSeedHook) {
  api.imageRecallSeedHook = true;
  const queue = api.queuePrompt.bind(api);
  api.queuePrompt = async (...args) => {
    const res = await queue(...args);
    try {
      for (const t of allTargets()) stepSeed(t);
    } catch (err) {
      console.warn("Image Recall: seed update failed", err);
    }
    return res;
  };
}

// Lock: fill any setting the settings JSON leaves to "the image" from the current image first,
// so the next image's metadata can't change it.
function freezeSettings(t, r) {
  if (!r) return;
  const s = readSettings(t);
  for (const [k, v] of Object.entries(settingsFromRecall(t, r))) if (s[k] === undefined || s[k] === null) s[k] = v;
  writeSettings(t, s);
}

// reset: take the image's own values. snap: restore saved settings. force: ignore Lock ("Reset to image").
async function loadImageInto(t, filename, { reset = false, snap = null, force = false, bump = true } = {}) {
  const panel = t.host.imageRecallPanel;
  const locked = isLocked(t) && !force;
  if (locked) freezeSettings(t, panel?.last?.());
  else if (snap) restore(t, snap);
  setWidget(t.recall, "image", filename);
  try {
    const r = await fetchRecall(t, filename);
    if (reset && !locked && !snap) {
      writeSettings(t, settingsFromRecall(t, r));
      applyDefaultsForImage(t, r);
      await applyRecalledTags(t, r);
    }
    await loadHistory();
    remember(t, { bump, size: r.size });
    panel?.render(r);
  } catch (err) {
    panel?.render(null, String(err));
  }
}

// Large preview that follows the mouse while hovering a history thumbnail.
const POPUP_MAX = 512;
let popup = null;
function showPopup(entry, ev) {
  if (!popup) {
    popup = el("div", { style: { position: "fixed", zIndex: "10000", pointerEvents: "none", display: "none",
      background: "#111", border: "1px solid #555", borderRadius: "6px", padding: "4px",
      boxShadow: "0 4px 16px rgba(0,0,0,0.6)", color: "#ddd", fontSize: "11px" } },
      el("img", { style: { display: "block", maxWidth: `${POPUP_MAX}px`, maxHeight: `${POPUP_MAX}px`, borderRadius: "4px" } }),
      el("div", { style: { marginTop: "3px", opacity: "0.8", whiteSpace: "nowrap", overflow: "hidden",
        textOverflow: "ellipsis", maxWidth: `${POPUP_MAX}px` } }));
    popup.firstChild.onload = () => movePopup();
    document.body.append(popup);
  }
  const src = viewURL(entry.filename, "webp;85");
  if (popup.firstChild.getAttribute("src") !== src) popup.firstChild.src = src;
  const when = new Date(entry.used ?? entry.added ?? 0).toLocaleString();
  popup.lastChild.textContent = `${entry.filename}  ${entry.size ?? ""}  ·  last used ${when}`;
  popup.style.display = "block";
  movePopup(ev);
}
let popupAt = { x: 0, y: 0 };
function movePopup(ev) {
  if (!popup || popup.style.display === "none") return;
  if (ev) popupAt = { x: ev.clientX, y: ev.clientY };
  const gap = 16;
  const w = popup.offsetWidth, h = popup.offsetHeight;
  let x = popupAt.x + gap, y = popupAt.y + gap;
  if (x + w > window.innerWidth) x = Math.max(0, popupAt.x - gap - w);   // flip to the left
  if (y + h > window.innerHeight) y = Math.max(0, window.innerHeight - h); // keep on screen
  popup.style.left = `${x}px`;
  popup.style.top = `${y}px`;
}
function hidePopup() {
  if (popup) popup.style.display = "none";
}

function viewURL(filename, preview) {
  const k = filename.lastIndexOf("/");
  const q = new URLSearchParams({ filename: filename.slice(k + 1), type: "input" });
  if (k > 0) q.set("subfolder", filename.slice(0, k));
  if (preview) q.set("preview", preview);
  return api.apiURL(`/view?${q}`);
}

// ---------------------------------------------------------------- panel
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) e[k] = v;
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined) e.append(kid);
  return e;
}

const css = {
  root: { display: "flex", flexDirection: "column", gap: "6px", padding: "6px", fontSize: "12px",
    color: "var(--input-text, #ddd)", boxSizing: "border-box", width: "100%", overflow: "hidden" },
  head: { fontWeight: "600", marginTop: "6px", borderBottom: "1px solid #555", paddingBottom: "2px" },
  row: { display: "flex", gap: "6px", alignItems: "center", flexWrap: "wrap" },
  label: { minWidth: "84px", opacity: "0.8" },
  input: { background: "var(--comfy-input-bg, #222)", color: "inherit", border: "1px solid #555",
    borderRadius: "4px", padding: "2px 4px", font: "inherit" },
  text: { width: "100%", minHeight: `${TEXT_MIN_H}px`, resize: "vertical", boxSizing: "border-box" },
  note: { color: "#f6a", whiteSpace: "pre-wrap" },
  dim: { opacity: "0.7", whiteSpace: "pre-wrap" },
};

function select(options, value, onchange, width = "auto") {
  const s = el("select", { style: { ...css.input, width, maxWidth: "100%" }, onchange: () => onchange(s.value) });
  const opts = options.includes(value) || value == null ? options : [value, ...options];
  for (const o of opts) s.append(el("option", { value: o, textContent: o, selected: o === value }));
  return s;
}
function num(value, step, onchange, width = "70px") {
  const i = el("input", { type: "number", step, value, style: { ...css.input, width },
    onchange: () => onchange(i.value === "" ? null : Number(i.value)) });
  return i;
}
function textBox(value, onchange) {
  const t = el("textarea", { value: value ?? "", style: { ...css.input, ...css.text },
    onchange: () => onchange(t.value) });
  // keep typing inside the box from triggering canvas shortcuts
  t.addEventListener("keydown", (e) => e.stopPropagation());
  return t;
}
const row = (label, ...kids) => el("div", { style: css.row }, el("span", { style: css.label, textContent: label }), ...kids);

function makePanel(t) {
  const root = el("div", { style: css.root });
  let last = null;     // last recall result
  let lastError = null;

  const changed = () => {
    t.host.graph?.setDirtyCanvas?.(true, true);
    app.graph?.change?.();
    remember(t);
  };
  const edit =(key, value) => {
    const s = readSettings(t);
    s[key] = value;
    writeSettings(t, s);
    changed();
    render(last, lastError);
  };

  function estimate(s) {
    const mode = widget(t.recall, "mode")?.value ?? MODES[1];
    const canvas = byType(t, "OutpaintCanvas");
    const up = byType(t, "UpscaleSize");
    const s2 = step2(t);
    const calls = callsPerStep(s.sampler);
    const lines = [];
    let units = 0;
    const [sw, sh] = (last?.size ?? "1024x1024").split("x").map(Number);
    let base = [sw, sh];
    if (mode !== "upscale") {
      const [W, H] = BUCKETS[widget(canvas, "aspect")?.value ?? "16:9"] ?? [1344, 768];
      const u = ((W * H) / 2 ** 20) * (s.steps ?? 25) * calls;
      units += u;
      base = [W, H];
      lines.push(`Step 1 outpaint: ${W}x${H} x ${s.steps} steps${calls > 1 ? ` x ${calls} calls` : ""}`);
    }
    if (mode !== "outpaint") {
      const z = upscaleSize(base[0], base[1], widget(up, "aspect")?.value ?? FROM_IMAGE,
        Number(widget(up, "long_side")?.value ?? 1920));
      const st = Number(widget(s2, "steps")?.value ?? s.steps ?? 25);
      const u = ((z.rw * z.rh) / 2 ** 20) * st * calls;
      units += u;
      lines.push(`Step 2 upscale: render ${z.rw}x${z.rh} -> ${z.w}x${z.h}, ${st} steps (all run, whatever the denoise)`);
    }
    const normal = (s.steps ?? 25) * calls;
    let text = lines.join("\n") + `\n≈ ${(units / Math.max(1, normal)).toFixed(1)}x one normal 1024x1024 render`;
    const sec = Number(localStorage.getItem(SPEED_KEY));
    if (sec > 0) text += `, about ${Math.max(1, Math.round((units * sec) / 60))} min on this machine (last measured)`;
    if (calls > 1) text += `\n! ${s.sampler} makes ${calls} model calls per step: a 1-call sampler (dpmpp_2m, euler) is ${calls}x faster`;
    return text;
  }

  function render(r, error) {
    last = r ?? last;
    lastError = error ?? null;
    root.replaceChildren();
    const image = widget(t.recall, "image")?.value;
    const mode = widget(t.recall, "mode")?.value ?? MODES[1];
    const s = readSettings(t);

    // image
    const thumb = image
      ? el("img", { src: viewURL(image),
          style: { maxWidth: "96px", maxHeight: "96px", borderRadius: "4px" } })
      : null;
    root.append(el("div", { style: css.row }, thumb,
      el("div", { style: css.dim, textContent: image
        ? `${image}\n${last?.size ?? ""}\nRead from: ${(last?.sources ?? []).join(", ") || "nothing"}`
        : "Drop an image anywhere in the window.\n(Shift+drop loads its workflow instead.)" })));
    if (lastError) root.append(el("div", { style: css.note, textContent: lastError }));
    // Under Lock the notes describe the new picture's own (ignored) metadata, so don't alarm with them.
    if (isLocked(t) && image) {
      root.append(el("div", { style: css.dim, textContent: (last?.notes?.length
        ? "Lock on: this picture's own metadata is ignored (it has none, or only some); the locked settings below are used."
        : "Lock on: the settings below are used, not this picture's own.") }));
    } else {
      for (const n of last?.notes ?? []) root.append(el("div", { style: css.note, textContent: n }));
    }

    // mode
    const modeRow = el("div", { style: css.row }, el("span", { style: css.label, textContent: "Mode" }));
    for (const m of MODES) {
      modeRow.append(el("label", { style: { display: "flex", gap: "3px", alignItems: "center" } },
        el("input", { type: "radio", name: `ir-mode-${t.host.id}`, checked: m === mode, onchange: () => {
          const prev = widget(t.recall, "mode")?.value;
          setWidget(t.recall, "mode", m);
          // swap the prompt if it is still the one filled in from the image
          if (last) {
            const cur = readSettings(t);
            const auto = prev === "upscale" ? last.positive : last.outpaint_positive;
            if (cur.positive === auto) edit("positive", m === "upscale" ? last.positive : last.outpaint_positive);
          }
          syncUpscaleAspect(t);
          changed();
          render(last);
        } }), m));
    }
    modeRow.append(el("label", { title: "Lock: a new image (drop, paste or history) changes only the picture, not the settings",
      style: { display: "flex", gap: "3px", alignItems: "center", marginLeft: "auto" } },
      el("input", { type: "checkbox", checked: isLocked(t), onchange: (e) => {
        t.host.properties ??= {};
        t.host.properties.imageRecallLock = e.target.checked;
        t.host.graph?.setDirtyCanvas?.(true, true);
        app.graph?.change?.();
        render(last, lastError);
      } }), "Lock settings"));
    root.append(modeRow);
    root.append(historySection(image));
    if (!image) {
      requestAnimationFrame(() => fitHost());
      return;
    }

    const choices = last?.choices ?? {};
    // model
    root.append(el("div", { style: css.head, textContent: "Model" }));
    root.append(row("Checkpoint", select(choices.checkpoints ?? [], s.checkpoint, (v) => edit("checkpoint", v), "100%")));
    const loras = Array.isArray(s.loras) ? s.loras : [];
    loras.forEach((l, i) => {
      const upd = (k, v) => { const ls = readSettings(t).loras ?? []; ls[i] = { ...ls[i], [k]: v }; edit("loras", ls); };
      const known = (choices.loras ?? []).includes(l.name);
      root.append(el("div", { style: css.row },
        el("input", { type: "checkbox", checked: l.on !== false, onchange: (e) => upd("on", e.target.checked) }),
        el("span", { style: { flex: "1", minWidth: "120px", color: known ? "inherit" : "#f6a" },
          textContent: (known ? "" : "missing: ") + l.name.split("/").pop() }),
        el("span", { textContent: "model" }), num(l.model, 0.05, (v) => upd("model", v ?? 0), "56px"),
        el("span", { textContent: "clip" }), num(l.clip ?? l.model, 0.05, (v) => upd("clip", v ?? 0), "56px"),
        el("button", { textContent: "✕", style: css.input, onclick: () => {
          const ls = readSettings(t).loras ?? []; ls.splice(i, 1); edit("loras", ls); } })));
    });
    // Add LoRA: a dropdown box like the checkpoint one. It opens ComfyUI's list menu (the same one
    // rgthree's Power Lora Loader uses), where typing filters the LoRAs by name/path.
    const addLora = (name) => edit("loras", [...(readSettings(t).loras ?? []), { name, on: true, model: 1, clip: 1 }]);
    const addBox = el("button", { title: "Add a LoRA (type in the list to search)",
      style: { ...css.input, width: "100%", textAlign: "left", cursor: "pointer", display: "flex",
        justifyContent: "space-between" } },
      el("span", { textContent: "Add LoRA…" }), el("span", { textContent: "▾" }));
    addBox.onclick = (e) => {
      const taken = new Set((readSettings(t).loras ?? []).map((l) => l.name));
      const items = (choices.loras ?? []).filter((n) => !taken.has(n));
      const LG = window.LiteGraph;
      if (!LG?.ContextMenu || !items.length) return;
      new LG.ContextMenu(items, {
        event: e,
        title: "Choose a lora",
        scale: Math.max(1, app.canvas?.ds?.scale ?? 1),
        className: "dark",
        callback: (value) => {
          const name = typeof value === "string" ? value : value?.content;
          if (name) addLora(name);
        },
      });
    };
    root.append(row("Add LoRA", addBox));

    // LoRA tags (the picker inside the subgraph adds the active ones to the prompt at queue time)
    const picker = byType(t, PICKER);
    const onLoras = loras.filter((l) => l.on !== false);
    if (picker && onLoras.length) {
      root.append(el("div", { style: css.head, textContent: "LoRA tags (click to add to the prompt)" }));
      root.append(row("Tags go", select(["after prompt", "before prompt"], widget(picker, "tag_position")?.value,
        (v) => { setWidget(picker, "tag_position", v); changed(); })));
      const maxTags = Number(widget(picker, "max_tags")?.value ?? 8);
      for (const l of onLoras) {
        const box = el("div", { style: { display: "flex", flexWrap: "wrap", gap: "4px", alignItems: "center" } },
          el("span", { style: { ...css.label, minWidth: "0", width: "100%" }, textContent: l.name.split("/").pop() }));
        root.append(box);
        loraTags(l.name).then((info) => {
          const { front, auto } = frontOf(info, maxTags);
          const showAll = t.host.imageRecallAllTags?.has(l.name);
          const list = showAll ? (info.tags ?? []) : front;
          if (!list.length) box.append(el("span", { style: css.dim, textContent: info.note || "no tags in this file" }));
          for (const tag of list) {
            const isOn = () => pickerState(picker).overrides[l.name]?.[tag.tag] ?? auto.has(tag.tag);
            const chip = el("button", { textContent: tag.tag, title: tag.tier ?? "", style: { ...css.input,
              cursor: "pointer", padding: "1px 6px", borderRadius: "10px",
              background: isOn() ? "#2a6" : "var(--comfy-input-bg, #222)" } });
            chip.onclick = () => {
              const st = pickerState(picker);
              const next = !isOn();
              const forLora = (st.overrides[l.name] ??= {});
              if (next === auto.has(tag.tag)) delete forLora[tag.tag];
              else forLora[tag.tag] = next;
              if (!Object.keys(forLora).length) delete st.overrides[l.name];
              setWidget(picker, "selection", JSON.stringify(st));
              chip.style.background = next ? "#2a6" : "var(--comfy-input-bg, #222)";
              changed();
            };
            box.append(chip);
          }
          const more = (info.tags ?? []).length - front.length;
          if (more > 0) {
            box.append(el("button", { textContent: showAll ? "fewer" : `+${more} more`, style: { ...css.input, cursor: "pointer" },
              onclick: () => {
                const set = (t.host.imageRecallAllTags ??= new Set());
                showAll ? set.delete(l.name) : set.add(l.name);
                render(last);
              } }));
          }
          requestAnimationFrame(() => fitHost());
        });
      }
    }

    // prompts
    root.append(el("div", { style: css.head, textContent: "Prompt" }));
    root.append(el("div", { style: css.dim, textContent: mode === "upscale" ? "Prompt"
      : "Prompt for the outpaint (main subject weighted down)" }));
    root.append(textBox(s.positive, (v) => edit("positive", v)));
    root.append(el("div", { style: css.dim, textContent: "Negative" }));
    root.append(textBox(s.negative, (v) => edit("negative", v)));
    if (mode !== "upscale") {
      root.append(el("div", { style: css.dim, textContent: "Subject (weighted down in the new area; empty = auto, 'none' = off)" }));
      root.append(textBox(widget(t.recall, "subject")?.value ?? "", (v) => {
        setWidget(t.recall, "subject", v); refreshPrompt(); }));
      root.append(row("Subject weight", num(widget(t.recall, "subject_weight")?.value ?? 0.4, 0.05, (v) => {
        setWidget(t.recall, "subject_weight", v ?? 0.4); refreshPrompt(); })));
    }

    // sampler
    root.append(el("div", { style: css.head, textContent: "Sampler" }));
    root.append(row("Seed", num(s.seed, 1, (v) => edit("seed", v ?? 0), "170px"),
      el("button", { textContent: "🎲", title: "New random seed now", style: { ...css.input, cursor: "pointer" },
        onclick: () => edit("seed", randomSeed()) }),
      el("span", { textContent: "after each run:" }),
      select(SEED_MODES, seedMode(t), (v) => {
        (t.host.properties ??= {}).imageRecallSeedMode = v; changed(); })));
    root.append(row("Steps", num(s.steps, 1, (v) => edit("steps", v ?? 1)),
      el("span", { textContent: "CFG" }), num(s.cfg, 0.1, (v) => edit("cfg", v ?? 1))));
    root.append(row("Sampler", select(choices.samplers ?? [], s.sampler, (v) => edit("sampler", v)),
      select(choices.schedulers ?? [], s.scheduler, (v) => edit("scheduler", v))));

    // outpaint
    const canvas = byType(t, "OutpaintCanvas");
    if (mode !== "upscale" && canvas) {
      root.append(el("div", { style: css.head, textContent: "Step 1 - Outpaint" }));
      root.append(row("Aspect", select(ASPECTS, widget(canvas, "aspect")?.value, (v) => {
        setWidget(canvas, "aspect", v); syncUpscaleAspect(t); changed(); render(last); }),
        el("span", { textContent: "Original at" }), select(ANCHORS, widget(canvas, "anchor")?.value, (v) => {
          setWidget(canvas, "anchor", v); changed(); render(last); })));
      const [srcW, srcH] = (last?.size ?? "1024x1024").split("x").map(Number);
      const rw = Number(widget(canvas, "resize_width")?.value ?? 0);
      const rh = Number(widget(canvas, "resize_height")?.value ?? 0);
      const p = placedSize(srcW, srcH, widget(canvas, "aspect")?.value ?? "16:9", rw, rh);
      root.append(row("Resize to",
        num(rw, 8, (v) => { setWidget(canvas, "resize_width", Math.max(0, v ?? 0)); changed(); render(last); }),
        el("span", { textContent: "×" }),
        num(rh, 8, (v) => { setWidget(canvas, "resize_height", Math.max(0, v ?? 0)); changed(); render(last); }),
        el("span", { style: css.dim, textContent: `(0 = fit) → ${p.w}x${p.h} on ${p.W}x${p.H}` })));

      // blend slider; it sets feather + repaint_original, re-fitted whenever the placed size changes
      const props = (t.host.properties ??= {});
      let b = Number(props.imageRecallBlend);
      if (!Number.isFinite(b)) {
        b = blendFrom(Number(widget(canvas, "feather")?.value ?? 48), Number(widget(canvas, "repaint_original")?.value ?? 0), p);
      }
      const apply = (val) => {
        const q = blendParams(val, p);
        if (widget(canvas, "feather")?.value !== q.feather) setWidget(canvas, "feather", q.feather);
        if (widget(canvas, "repaint_original") && widget(canvas, "repaint_original").value !== q.repaint) {
          setWidget(canvas, "repaint_original", q.repaint);
        }
        return q;
      };
      const describeBlend = (q) => `feather ${q.feather} px` + (q.repaint > 0
        ? ` (whole picture) · picture redrawn ${Math.round(q.repaint * 100)}%`
        : ` · picture kept${q.feather >= q.maxF ? " (only its centre)" : ""}`);
      const q0 = apply(b);
      const readout = el("span", { style: css.dim, textContent: describeBlend(q0) });
      const slider = el("input", { type: "range", min: 0, max: 100, step: 1, value: Math.round(b),
        title: "Left: keep as much of the dropped picture as possible. Right: let the generated scene take it over.",
        style: { flex: "1", minWidth: "120px" } });
      slider.addEventListener("input", () => { readout.textContent = describeBlend(blendParams(Number(slider.value), p)); });
      slider.addEventListener("change", () => {
        props.imageRecallBlend = Number(slider.value);
        apply(props.imageRecallBlend);
        changed();
        render(last);
      });
      slider.addEventListener("pointerdown", (e) => e.stopPropagation()); // drag the slider, not the canvas
      root.append(el("div", { style: css.row }, el("span", { style: css.label, textContent: "Blend" }),
        el("span", { style: { fontSize: "11px" }, textContent: "keep picture" }), slider,
        el("span", { style: { fontSize: "11px" }, textContent: "scene eats it" })));
      root.append(el("div", { style: { ...css.dim, paddingLeft: "90px" } }, readout));
    }

    // upscale
    const up = byType(t, "UpscaleSize");
    const s2 = step2(t);
    if (mode !== "outpaint" && up && s2) {
      root.append(el("div", { style: css.head, textContent: "Step 2 - Upscale (one SDXL pass)" }));
      const autoLabel = mode === "upscale" ? "auto (picture's own shape)" : "auto (same as Step 1)";
      const shapes = [UP_AUTO, FROM_IMAGE, ...ASPECTS];
      const shapeSel = select(shapes, upShape(t), (v) => {
        (t.host.properties ??= {}).imageRecallUpAspect = v;
        syncUpscaleAspect(t); changed(); render(last); });
      for (const o of shapeSel.options) {
        if (o.value === UP_AUTO) o.textContent = autoLabel;
        if (o.value === FROM_IMAGE) o.textContent = "picture's own shape";
      }
      const rs = renderScale(t);
      const fit = rs ? select(["stretch", "crop"], widget(rs, "crop")?.value === "disabled" ? "stretch" : "crop", (v) => {
        setWidget(rs, "crop", v === "stretch" ? "disabled" : "center"); changed(); }) : null;
      root.append(row("Shape", shapeSel, fit ? el("span", { textContent: "to fit:" }) : null, fit));
      root.append(row("Long side", num(widget(up, "long_side")?.value, 8, (v) => {
        setWidget(up, "long_side", v ?? 1920); changed(); render(last); })));
      root.append(row("Steps", num(widget(s2, "steps")?.value, 1, (v) => {
        setWidget(s2, "steps", v ?? 1); changed(); render(last); }),
        el("span", { textContent: "Denoise" }), num(widget(s2, "denoise")?.value, 0.05, (v) => {
          setWidget(s2, "denoise", v ?? 0.3); changed(); })));
    }

    // estimate + reset
    root.append(el("div", { style: css.head, textContent: "Estimate" }));
    root.append(el("div", { style: css.dim, textContent: estimate(s) }));
    root.append(el("button", { textContent: "Reset to image", style: { ...css.input, alignSelf: "flex-start" },
      onclick: () => loadImageInto(t, image, { reset: true, force: true }) }));

    requestAnimationFrame(() => fitHost());
  }

  // thumbnails of every imported image; click = load, ✕ = remove from history, Ctrl+wheel = size
  function historySection(current) {
    hidePopup(); // the panel is being rebuilt; its cells won't send mouseleave
    const box = el("div", { style: { display: "flex", flexDirection: "column", gap: "4px" } });
    const list = history ?? [];
    box.append(el("div", { style: css.head, textContent: `History (${history ? list.length : "…"})` }));
    if (!list.length) {
      box.append(el("div", { style: css.dim, textContent: history ? "Imported images show up here." : "loading…" }));
      return box;
    }
    box.append(el("div", { style: css.dim, textContent: `Click to load${isLocked(t) ? " (Lock on: picture only)" : " with its settings"}`
      + " · ✕ removes it from the history · Ctrl+wheel resizes" }));
    let size = 96;
    try { size = Math.min(320, Math.max(48, Number(localStorage.getItem(THUMB_KEY)) || 96)); } catch {}
    const grid = el("div", { style: { display: "flex", flexWrap: "wrap", gap: "4px", overflowY: "auto",
      maxHeight: `${size * 3 + 16}px` } });
    const cells = [];
    for (const e of list) {
      const cell = el("div", { style: { position: "relative", flex: "none", width: `${size}px`, height: `${size}px` } });
      const img = el("img", { src: viewURL(e.filename, "webp;70"), loading: "lazy", draggable: false,
        style: { width: "100%", height: "100%", objectFit: "cover", borderRadius: "4px", cursor: "pointer",
          boxSizing: "border-box", border: `2px solid ${e.filename === current ? "#4af" : "transparent"}`, background: "#111" },
        onclick: () => loadImageInto(t, e.filename, { snap: e.snapshot, reset: !e.snapshot }) });
      img.onerror = () => { img.style.opacity = "0.3"; cell.title = `${e.filename}\n(file not found in ComfyUI/input)`; hidePopup(); };
      const x = el("button", { textContent: "✕", title: "Remove from history (the file stays in ComfyUI/input)",
        style: { ...css.input, position: "absolute", top: "2px", right: "2px", padding: "0 4px", lineHeight: "16px",
          cursor: "pointer", display: "none", background: "rgba(0,0,0,0.7)" },
        onclick: (ev) => {
          ev.stopPropagation();
          const i = history.indexOf(e);
          if (i >= 0) history.splice(i, 1);
          saveHistory();
          render(last, lastError);
        } });
      cell.addEventListener("mouseenter", (ev) => { x.style.display = "block"; showPopup(e, ev); });
      cell.addEventListener("mousemove", (ev) => movePopup(ev));
      cell.addEventListener("mouseleave", () => { x.style.display = "none"; hidePopup(); });
      cell.append(img, x);
      cells.push(cell);
      grid.append(cell);
    }
    grid.addEventListener("wheel", (ev) => {
      if (!ev.ctrlKey) {
        if (grid.scrollHeight > grid.clientHeight) ev.stopPropagation(); // scroll the list, not the canvas
        return;
      }
      ev.preventDefault();
      ev.stopPropagation();
      size = Math.min(320, Math.max(48, size + (ev.deltaY < 0 ? 16 : -16)));
      try { localStorage.setItem(THUMB_KEY, String(size)); } catch {}
      for (const c of cells) { c.style.width = `${size}px`; c.style.height = `${size}px`; }
      grid.style.maxHeight = `${size * 3 + 16}px`;
      requestAnimationFrame(() => fitHost());
    }, { passive: false });
    box.append(grid);
    return box;
  }

  function fitHost() {
    const h = root.scrollHeight + 12;
    if (Math.abs((t.host.imageRecallPanelHeight ?? 0) - h) > 4) {
      t.host.imageRecallPanelHeight = h;
      const size = t.host.computeSize?.() ?? t.host.size;
      t.host.setSize?.([Math.max(t.host.size[0], 480), Math.max(size[1], h + 40)]);
      t.host.graph?.setDirtyCanvas?.(true, true);
    }
  }

  async function refreshPrompt() {
    const image = widget(t.recall, "image")?.value;
    if (!image) return;
    const before = last;
    try {
      const r = await fetchRecall(t, image);
      const cur = readSettings(t);
      if (before && cur.positive === before.outpaint_positive) {
        cur.positive = r.outpaint_positive;
        writeSettings(t, cur);
      }
      remember(t);
      render(r);
    } catch (err) {
      render(null, String(err));
    }
  }

  return { root, render, last: () => last };
}

function attachPanel(node) {
  if (node.imageRecallPanel) return;
  const t = targetOf(node);
  if (!t) return;
  const panel = makePanel(t);
  node.imageRecallPanel = panel;
  node.imageRecallTarget = t;
  const w = node.addDOMWidget?.("image_recall_settings", "image_recall_panel", panel.root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => node.imageRecallPanelHeight ?? 200,
    getMaxHeight: () => node.imageRecallPanelHeight ?? 200,
  });
  if (w) w.serializeValue = () => undefined;
  if (node.size[0] < 480) node.setSize?.([480, node.size[1]]);
  const image = widget(t.recall, "image")?.value;
  if (image && image !== "example.png") {
    loadImageInto(t, image, { reset: !widget(t.recall, "settings")?.value, force: true, bump: false });
  } else {
    panel.render(null);
    loadHistory().then(() => panel.render(null));
  }
}

// ---------------------------------------------------------------- text height (3x default)
function tallTextWidgets(node) {
  for (const w of node.widgets ?? []) {
    const elx = w.inputEl ?? w.element;
    if (elx?.tagName === "TEXTAREA") {
      elx.style.minHeight = `${TEXT_MIN_H}px`;
      const prev = w.options?.getMinHeight;
      if (w.options) w.options.getMinHeight = () => Math.max(TEXT_MIN_H, prev?.() ?? 0);
    }
  }
}

// ---------------------------------------------------------------- drop / paste anywhere
// Returns the target and image file when this event should go to Image Recall, else null.
function claim(files) {
  const file = [...(files ?? [])].find((f) => f.type?.startsWith("image/"));
  if (!file) return null;
  const t = findTarget();
  if (!t) return null;
  const over = app.canvas?.dragOverNode;
  if (over && over !== t.host && over !== t.recall) return null; // dropped on another node: leave it alone
  return { t, file };
}

async function take({ t, file }) {
  try {
    const name = await uploadImage(file);
    await loadImageInto(t, name, { reset: true });
  } catch (err) {
    t.host.imageRecallPanel?.render(null, String(err));
  }
}

window.addEventListener("drop", (e) => {
  if (e.shiftKey) return; // Shift+drop: ComfyUI's normal behaviour (load the image's workflow)
  const c = claim(e.dataTransfer?.files);
  if (!c) return;
  e.preventDefault(); // ComfyUI's own drop handler skips events already handled
  e.stopPropagation();
  take(c);
}, true);

window.addEventListener("paste", (e) => {
  const tag = document.activeElement?.tagName;
  if (tag === "TEXTAREA" || tag === "INPUT") return;
  const c = claim(e.clipboardData?.files);
  if (!c) return;
  e.preventDefault();
  e.stopPropagation();
  take(c);
}, true);

// ---------------------------------------------------------------- timing (for the estimate)
let lastTick = null;
api.addEventListener("progress", ({ detail }) => {
  const t = findTarget();
  if (!t || !detail) return;
  const now = performance.now();
  const id = String(detail.node ?? "").split(":").pop();
  const ks = [step1(t), step2(t)].filter(Boolean);
  const k = ks.find((n) => String(n.id) === id);
  if (!k || !lastTick || lastTick.id !== id || detail.value <= lastTick.value) {
    lastTick = { id, value: detail.value, at: now };
    return;
  }
  const s = readSettings(t);
  let mp;
  if (k === step1(t)) {
    const [W, H] = BUCKETS[widget(byType(t, "OutpaintCanvas"), "aspect")?.value] ?? [1344, 768];
    mp = (W * H) / 2 ** 20;
  } else {
    const up = byType(t, "UpscaleSize");
    const z = upscaleSize(1344, 768, widget(up, "aspect")?.value ?? FROM_IMAGE, Number(widget(up, "long_side")?.value ?? 1920));
    mp = (z.rw * z.rh) / 2 ** 20;
  }
  const perStep = (now - lastTick.at) / 1000 / (detail.value - lastTick.value);
  const sec = perStep / (mp * callsPerStep(s.sampler));
  const old = Number(localStorage.getItem(SPEED_KEY)) || sec;
  localStorage.setItem(SPEED_KEY, String(old * 0.7 + sec * 0.3));
  lastTick = { id, value: detail.value, at: now };
});

// ---------------------------------------------------------------- registration
app.registerExtension({
  name: "ImageRecall.Panel",
  nodeCreated(node) {
    setTimeout(() => {
      attachPanel(node);
      if (node.type === RECALL) tallTextWidgets(node);
    }, 0);
  },
  loadedGraphNode(node) {
    setTimeout(() => {
      attachPanel(node);
      if (node.type === RECALL) tallTextWidgets(node);
    }, 0);
  },
});
