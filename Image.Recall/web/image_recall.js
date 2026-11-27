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
function upscaleSize(srcW, srcH, aspect, longSide) {
  let [ra, rb] = aspect in BUCKETS ? aspect.split(":").map(Number) : [srcW, srcH];
  const [w, h] = ra >= rb ? [longSide, round8((longSide * rb) / ra)] : [round8((longSide * ra) / rb), longSide];
  return { rw: ceil64(w), rh: ceil64(h), w, h };
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

function applyDefaultsForImage(t, r) {
  const s2 = step2(t);
  if (s2) setWidget(s2, "steps", r.steps);
  syncUpscaleAspect(t);
}

function syncUpscaleAspect(t) {
  const mode = widget(t.recall, "mode")?.value;
  const canvas = byType(t, "OutpaintCanvas");
  const up = byType(t, "UpscaleSize");
  if (up) setWidget(up, "aspect", mode === "upscale" ? FROM_IMAGE : widget(canvas, "aspect")?.value ?? "16:9");
}

async function uploadImage(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("overwrite", "true");
  const res = await api.fetchApi("/upload/image", { method: "POST", body });
  if (!res.ok) throw new Error(`upload failed: ${res.status}`);
  const j = await res.json();
  return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
}

async function loadImageInto(t, filename, { reset }) {
  setWidget(t.recall, "image", filename);
  const panel = t.host.imageRecallPanel;
  try {
    const r = await fetchRecall(t, filename);
    if (reset) {
      writeSettings(t, settingsFromRecall(t, r));
      applyDefaultsForImage(t, r);
    }
    panel?.render(r);
  } catch (err) {
    panel?.render(null, String(err));
  }
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
  };
  const edit = (key, value) => {
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
      ? el("img", { src: api.apiURL(`/view?filename=${encodeURIComponent(image)}&type=input`),
          style: { maxWidth: "96px", maxHeight: "96px", borderRadius: "4px" } })
      : null;
    root.append(el("div", { style: css.row }, thumb,
      el("div", { style: css.dim, textContent: image
        ? `${image}\n${last?.size ?? ""}\nRead from: ${(last?.sources ?? []).join(", ") || "nothing"}`
        : "Drop an image anywhere in the window.\n(Shift+drop loads its workflow instead.)" })));
    if (lastError) root.append(el("div", { style: css.note, textContent: lastError }));
    for (const n of last?.notes ?? []) root.append(el("div", { style: css.note, textContent: n }));

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
    root.append(modeRow);
    if (!image) return;

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
    root.append(row("Add LoRA", select(["(pick)", ...(choices.loras ?? [])], "(pick)", (v) => {
      if (v === "(pick)") return;
      edit("loras", [...(readSettings(t).loras ?? []), { name: v, on: true, model: 1, clip: 1 }]);
    }, "100%")));

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
    root.append(row("Seed", num(s.seed, 1, (v) => edit("seed", v ?? 0), "170px")));
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
      root.append(row("Feather", num(widget(canvas, "feather")?.value, 8, (v) => {
        setWidget(canvas, "feather", v ?? 48); changed(); })));
    }

    // upscale
    const up = byType(t, "UpscaleSize");
    const s2 = step2(t);
    if (mode !== "outpaint" && up && s2) {
      root.append(el("div", { style: css.head, textContent: "Step 2 - Upscale (one SDXL pass)" }));
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
      onclick: () => loadImageInto(t, image, { reset: true }) }));

    requestAnimationFrame(() => {
      const h = root.scrollHeight + 12;
      if (Math.abs((t.host.imageRecallPanelHeight ?? 0) - h) > 4) {
        t.host.imageRecallPanelHeight = h;
        const size = t.host.computeSize?.() ?? t.host.size;
        t.host.setSize?.([Math.max(t.host.size[0], 480), Math.max(size[1], h + 40)]);
        t.host.graph?.setDirtyCanvas?.(true, true);
      }
    });
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
      render(r);
    } catch (err) {
      render(null, String(err));
    }
  }

  return { root, render };
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
  if (image && image !== "example.png") loadImageInto(t, image, { reset: !widget(t.recall, "settings")?.value });
  else panel.render(null);
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
