// LoRA Tag Picker - frontend half.
// Finds LoRA loaders upstream of the node, fetches each file's ranked training
// tags from the backend, and renders them as clickable chips. The chip state is
// saved with the workflow; the resolved tag list is sent along when you queue.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPE = "LORA_TAG_SELECTION";
const MODE_NEVER = 2;
const MODE_BYPASS = 4;
const MARGIN = 4;       // gap the frontend leaves around a DOM widget, each side
const MIN_HEIGHT = 44;
const MAX_HEIGHT = 340;
const POLL_MS = 1000;
const DEFAULT_MAX_TAGS = 8;
const AUTO_ON = 3;      // most tags of one LoRA that are on without being clicked

const STYLE = `
.ltp-root { box-sizing: border-box; width: 100%; height: 100%; overflow-y: auto;
  padding: 2px 4px; font: 12px/1.35 Inter, Arial, sans-serif;
  color: var(--input-text, #ddd); }
.ltp-root * { box-sizing: border-box; }
.ltp-lora { padding: 4px 0 6px; }
.ltp-lora + .ltp-lora { border-top: 1px solid var(--border-color, #4e4e4e); }
.ltp-head { display: flex; align-items: baseline; gap: 6px; margin-bottom: 4px; }
.ltp-name { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ltp-meta { flex: none; margin-left: auto; color: var(--descrip-text, #999); font-size: 11px; }
.ltp-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.ltp-chip { max-width: 100%; margin: 0; padding: 2px 8px; border-radius: 10px; cursor: pointer;
  font: inherit; text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  color: var(--descrip-text, #999); background: transparent;
  border: 1px dashed var(--border-color, #4e4e4e); }
.ltp-chip:hover { color: var(--input-text, #ddd); border-style: solid; }
.ltp-chip.is-on { color: var(--p-primary-contrast-color, #fff);
  background: var(--p-primary-color, #3b82f6); border: 1px solid transparent; }
.ltp-chip small { margin-left: 5px; font-size: 10px; opacity: 0.75; }
.ltp-more { margin: 4px 0 0; padding: 0; border: 0; background: none; cursor: pointer;
  font: inherit; font-size: 11px; color: var(--descrip-text, #999); text-decoration: underline; }
.ltp-more:hover { color: var(--input-text, #ddd); }
.ltp-note, .ltp-empty { color: var(--descrip-text, #999); font-size: 11px; }
.ltp-empty { padding: 6px 0; }
`;

function ensureStyle() {
  if (document.getElementById("ltp-style")) return;
  const style = document.createElement("style");
  style.id = "ltp-style";
  style.textContent = STYLE;
  document.head.appendChild(style);
}

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

// ------------------------------------------------------------ tag data

const infoCache = new Map(); // lora name -> Promise<info>

function fetchInfo(name) {
  if (!infoCache.has(name)) {
    const request = api
      .fetchApi(`/Lora.Tag.Picker/tags?name=${encodeURIComponent(name)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .catch((err) => {
        infoCache.delete(name); // retry on the next refresh
        return { name, found: false, tags: [], failed: true,
                 note: `Could not load tags (${err.message}).` };
      });
    infoCache.set(name, request);
  }
  return infoCache.get(name);
}

// ---------------------------------------------------------- graph walk

function lorasOnNode(node) {
  if (!node.outputs?.some((o) => o.type === "MODEL" || o.type === "CLIP")) return [];
  const names = [];
  for (const widget of node.widgets ?? []) {
    if (!String(widget.name ?? "").toLowerCase().includes("lora")) continue;
    let value = widget.value;
    if (value && typeof value === "object") value = value.on === false ? null : value.lora;
    if (typeof value === "string" && value && value !== "None") names.push(value);
  }
  return names;
}

// LoRA files loaded upstream of `node`, in load order (furthest upstream first).
function findUpstreamLoras(node) {
  const depthOf = new Map(); // upstream node -> its longest distance from `node`
  const limit = (node.graph?._nodes?.length ?? node.graph?.nodes?.length ?? 500) + 1;
  let frontier = [node];
  for (let depth = 1; frontier.length && depth <= limit; depth++) {
    const next = new Set();
    for (const current of frontier) {
      for (let slot = 0; slot < (current.inputs?.length ?? 0); slot++) {
        let source = null;
        try { source = current.getInputNode?.(slot) ?? null; } catch { source = null; }
        if (source && source !== node) next.add(source);
      }
    }
    for (const source of next) depthOf.set(source, depth);
    frontier = [...next];
  }
  const found = []; // [depth, order, name]
  for (const [source, depth] of depthOf) {
    // A bypassed or muted loader applies nothing, but the chain continues through it.
    if (source.mode === MODE_NEVER || source.mode === MODE_BYPASS) continue;
    for (const name of lorasOnNode(source)) found.push([depth, found.length, name]);
  }
  found.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  return [...new Set(found.map((f) => f[2]))];
}

// -------------------------------------------------------------- widget

function createPicker(node, inputName) {
  ensureStyle();
  const root = make("div", "ltp-root");
  const inner = make("div", "ltp-inner");
  root.appendChild(inner);

  const state = {
    overrides: {},        // { loraName: { tag: bool } } - only where it differs from the default
    expanded: new Set(),  // lora names showing every tag, not just the best ones
    loras: [],            // [{ name, info }]
    signature: null,
    height: MIN_HEIGHT,
    fitted: false,        // true once the node has been sized around real content
  };

  // The node's max_tags setting: how many tags one LoRA shows up front.
  const maxTags = () => {
    const value = Math.floor(Number(node.widgets?.find((w) => w.name === "max_tags")?.value));
    return Number.isFinite(value) && value >= 1 ? value : DEFAULT_MAX_TAGS;
  };

  // A LoRA's best tags: the ones shown up front. The first few triggers among
  // them (front.auto) are the only tags that are on without being clicked.
  const frontOf = (info) => {
    const front = new Set((info.tags ?? []).filter((t) => t.tier !== "rare").slice(0, maxTags()));
    front.auto = new Set([...front].filter((t) => t.tier === "trigger").slice(0, AUTO_ON));
    return front;
  };

  const isDefault = (entry, front) => front.auto.has(entry);

  const isActive = (name, entry, front) =>
    state.overrides[name]?.[entry.tag] ?? isDefault(entry, front);

  function toggle(name, entry, front) {
    const next = !isActive(name, entry, front);
    const forLora = (state.overrides[name] ??= {});
    if (next === isDefault(entry, front)) delete forLora[entry.tag];
    else forLora[entry.tag] = next;
    if (!Object.keys(forLora).length) delete state.overrides[name];
    render();
  }

  function describe(entry, info, pct) {
    if (entry.coverage == null) return "From the file's declared trigger or dataset folder name";
    const of = info.exact ? ` of ${info.images}` : "";
    if (entry.kind === "phrase") return `About ${entry.count}${of} captions open with this (${pct}%)`;
    if (entry.kind === "word") return `In about ${entry.count}${of} captions (${pct}%)`;
    return info.exact
      ? `In ${entry.count} of ${info.images} training images (${pct}%)`
      : `Used ${entry.count} times, ${pct}% of the most used tag (file records no image count)`;
  }

  function fitHeight() {
    if (!inner.offsetHeight) return; // not laid out yet
    const wanted = Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, inner.offsetHeight + 4 + 2 * MARGIN));
    if (wanted === state.height) return;
    const delta = wanted - state.height;
    state.height = wanted;
    if (node.size) {
      const needed = node.computeSize()[1];
      // First fit: a reloaded workflow's saved size already has room for the chips,
      // so only grow if it doesn't. After that, grow or shrink by the change so the
      // prompt box keeps whatever size you gave it.
      const height = state.fitted ? Math.max(node.size[1] + delta, needed)
                                  : Math.max(node.size[1], needed);
      if (height !== node.size[1]) node.setSize([node.size[0], height]);
    }
    state.fitted = state.loras.length > 0;
    node.graph?.setDirtyCanvas(true, true);
  }

  function renderLora({ name, info }) {
    const box = make("div", "ltp-lora");
    const head = make("div", "ltp-head");
    const label = make("span", "ltp-name", name.split(/[\\/]/).pop().replace(/\.safetensors$/i, ""));
    label.title = name;
    head.appendChild(label);
    if (info.images) head.appendChild(make("span", "ltp-meta", `${info.images} images`));
    box.appendChild(head);

    const tags = info.tags ?? [];
    const front = frontOf(info);
    const open = state.expanded.has(name);
    const shown = tags.filter((t) => open || front.has(t) || isActive(name, t, front));
    if (shown.length) {
      const chips = make("div", "ltp-chips");
      for (const entry of shown) {
        const on = isActive(name, entry, front);
        const chip = make("button", `ltp-chip${on ? " is-on" : ""}`, `${on ? "✓ " : ""}${entry.tag}`);
        chip.type = "button";
        chip.setAttribute("aria-pressed", String(on));
        const pct = entry.coverage == null ? null : Math.round(entry.coverage * 100);
        if (pct != null) chip.appendChild(make("small", "", `${pct}%`));
        chip.title = `${entry.tag}\n${describe(entry, info, pct)}`;
        chip.addEventListener("click", () => toggle(name, entry, front));
        chips.appendChild(chip);
      }
      box.appendChild(chips);
    }
    const hidden = tags.length - shown.length;
    if (hidden > 0 || (open && tags.length > front.size)) {
      const more = make("button", "ltp-more", open ? "show fewer" : `+${hidden} more`);
      more.type = "button";
      more.addEventListener("click", () => {
        if (open) state.expanded.delete(name);
        else state.expanded.add(name);
        render();
      });
      box.appendChild(more);
    }
    if (info.note) box.appendChild(make("div", "ltp-note", info.note));
    return box;
  }

  function render() {
    inner.replaceChildren();
    if (!state.loras.length) {
      inner.appendChild(make("div", "ltp-empty",
        "No LoRA loaders found upstream. Connect model or clip from your LoRA chain."));
    } else {
      for (const lora of state.loras) inner.appendChild(renderLora(lora));
    }
    fitHeight();
  }

  // Re-read the graph; re-render only when the upstream LoRAs or the limit changed.
  async function sync(force = false) {
    const names = findUpstreamLoras(node);
    const signature = `${maxTags()}\n${names.join("\n")}`;
    if (!force && signature === state.signature && !state.loras.some((l) => l.info.failed)) {
      fitHeight(); // the node may have just become visible
      return;
    }
    state.signature = signature;
    const infos = await Promise.all(names.map(fetchInfo));
    if (state.signature !== signature) return; // superseded by a newer sync
    state.loras = names.map((name, i) => ({ name, info: infos[i] }));
    render();
  }

  const widget = node.addDOMWidget(inputName, TYPE, root, {
    getValue: () => JSON.stringify({ v: 1, overrides: state.overrides }),
    setValue: (value) => {
      let parsed = null;
      try { parsed = typeof value === "string" ? JSON.parse(value) : value; } catch { parsed = null; }
      const overrides = parsed?.overrides;
      state.overrides = overrides && typeof overrides === "object" ? overrides : {};
      render();
    },
    getMinHeight: () => state.height,
    getMaxHeight: () => state.height,
    margin: MARGIN,
  });

  // At queue time, send exactly what the chips show.
  widget.serializeValue = async () => {
    try { await sync(true); } catch (err) { console.warn("[LoRA Tag Picker]", err); }
    const resolved = state.loras.map(({ name, info }) => {
      const front = frontOf(info);
      return {
        lora: name,
        tags: (info.tags ?? []).filter((t) => isActive(name, t, front)).map((t) => t.tag),
      };
    });
    return JSON.stringify({ v: 1, overrides: state.overrides, resolved });
  };

  const safeSync = () => sync().catch((err) => console.warn("[LoRA Tag Picker]", err));
  const timer = setInterval(() => { if (node.graph) safeSync(); }, POLL_MS);
  const onRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    clearInterval(timer);
    return onRemoved?.apply(this, args);
  };

  render();
  setTimeout(safeSync, 0);
  return widget;
}

app.registerExtension({
  name: "Lora.Tag.Picker.LoraTagPicker",
  getCustomWidgets() {
    return {
      [TYPE](node, inputName) {
        return { widget: createPicker(node, inputName) };
      },
    };
  },
});
