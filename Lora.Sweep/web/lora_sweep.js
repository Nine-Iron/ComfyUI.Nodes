// LoRA Sweep: lists every LoRA in the pipeline and sets each one's strength
// on its loader right before each queued run (on / off / sweep / random, with
// a max number of LoRAs per run). ComfyUI calls beforeQueued() on every widget
// before it builds each prompt of a batch, so run N of "Run x N" gets
// combination N.
import { app } from "../../scripts/app.js";

// Shared sweep engine. Sampler Sweep, LoRA Strength Sweep and Checkpoint Sweep
// each ship a copy; whichever loads first creates it and the others register
// into it, so all sweep nodes in a workflow share one counter (a full grid of
// runs). Keep the copies identical and bump ENGINE_VERSION in all of them.
const ENGINE_VERSION = 4;

function createEngine(app, previous) {
  const KINDS = { ...(previous?.KINDS ?? {}) };
  const MAX_VALUES = 1000;
  const FOLLOW = "same as model";

  const widget = (node, name) => node.widgets?.find((w) => w.name === name);
  const decimals = (s) => (s.includes(".") ? s.length - s.indexOf(".") - 1 : 0);
  const classOf = (n) => n.comfyClass ?? n.type;
  const kindOf = (n) => KINDS[classOf(n)];
  const graphNodes = (g) => g?.nodes ?? g?._nodes ?? [];

  // "20,30,40" / "20-40:5" / "0.4-1:0.2" / "0-1:.25" / "-1..1:0.5" -> numbers,
  // in order, at the precision written (no rounding beyond that).
  const NUM = String.raw`(?:\d+(?:\.\d*)?|\.\d+)`;
  const RANGE = new RegExp(String.raw`^(-?${NUM})\s*(?:\.\.|to|-)\s*(-?${NUM})(?:\s*:\s*(${NUM}))?$`, "i");
  function expandNumbers(spec, kind) {
    const out = [], errors = [];
    let tooMany = 0;
    for (const raw of String(spec).split(",")) {
      const part = raw.trim();
      if (!part) continue;
      const m = part.match(RANGE);
      if (m) {
        const step = m[3] ?? (kind === "int" ? "1" : "0.5");
        const d = Math.max(decimals(m[1]), decimals(m[2]), decimals(step));
        const scale = 10 ** d;
        const a = Math.round(parseFloat(m[1]) * scale), b = Math.round(parseFloat(m[2]) * scale);
        const s = Math.round(parseFloat(step) * scale);
        if (s <= 0) { errors.push(`step 0 in "${part}"`); continue; }
        const count = Math.floor(Math.abs(b - a) / s) + 1;
        if (out.length + count > MAX_VALUES) { tooMany += count; continue; }
        for (let k = 0, dir = a <= b ? 1 : -1; k < count; k++) out.push(Number(((a + dir * k * s) / scale).toFixed(d)));
      } else if (/^-?(?:\d+(?:\.\d*)?|\.\d+)$/.test(part)) {
        out.push(Number(part));
      } else {
        errors.push(`"${part}" is not a number or range`);
      }
    }
    if (tooMany) errors.push(`${tooMany} values is too many (the most is ${MAX_VALUES}): use a bigger step`);
    let vals = kind === "int" ? out.map((v) => Math.round(v)).filter((v) => v >= 1) : out;
    vals = [...new Set(vals)];
    return { values: vals, errors };
  }

  // "euler, dpmpp_*" -> option names; "*" or empty -> all of them.
  function expandCombo(spec, options) {
    const s = String(spec).trim();
    if (!s || s === "*") return { values: [...options], errors: [] };
    const out = [], errors = [];
    for (const raw of s.split(",")) {
      const part = raw.trim();
      if (!part) continue;
      if (part.includes("*")) {
        const re = new RegExp("^" + part.split("*").map((x) => x.replace(/[.+?^${}()|[\]\\]/g, "\\$&")).join(".*") + "$", "i");
        const hits = options.filter((o) => re.test(o));
        if (hits.length) out.push(...hits);
        else errors.push(`nothing matches "${part}"`);
      } else {
        const hit = options.find((o) => o.toLowerCase() === part.toLowerCase());
        if (hit) out.push(hit);
        else errors.push(`unknown "${part}"`);
      }
    }
    return { values: [...new Set(out)], errors };
  }

  function comboOptions(w) {
    const v = w?.options?.values;
    return typeof v === "function" ? v() : Array.isArray(v) ? v : [];
  }

  // The node feeding input slot `i`, looking through old-style Reroute nodes.
  function upstream(node, i) {
    const graph = node.graph ?? app.graph;
    let n = node, slot = i;
    for (let hops = 0; hops < 50; hops++) {
      const id = n.inputs?.[slot]?.link;
      if (id == null) return null;
      const link = graph.links?.get?.(id) ?? graph.links?.[id];
      const from = link && graph.getNodeById(link.origin_id);
      if (!from) return null;
      if (classOf(from) !== "Reroute") return from;
      n = from; slot = 0;
    }
    return null;
  }

  const isConnected = (node) => (node.inputs ?? []).some((inp) => inp.link != null && !inp.widget);

  // The nodes a sweep node drives: the ones wired into its inputs, or, with
  // nothing wired in, every node in the workflow that has its kind's widgets.
  function targets(node) {
    const names = kindOf(node).params.map((p) => p.target);
    const fits = (n) => n && !kindOf(n) && names.some((name) => widget(n, name));
    if (isConnected(node)) {
      const found = [];
      (node.inputs ?? []).forEach((inp, i) => {
        if (inp.widget) return;
        const n = upstream(node, i);
        if (fits(n) && !found.includes(n)) found.push(n);
      });
      return found;
    }
    return graphNodes(node.graph ?? app.graph).filter(fits);
  }

  const nameOf = (n) => `${n.title || classOf(n)} #${n.id}`;

  const keyOf = (p) => p.key ?? p.target;

  // Helpers a kind with its own plan()/apply()/describe() can use.
  const H = { widget, classOf, kindOf, graphNodes, expandNumbers, isConnected, nameOf: (n) => nameOf(n), FOLLOW };

  // One node's settings: the value list of every parameter that is not off.
  // A kind may build its own (dynamic) parameter list with plan(node, H).
  function buildPlan(node) {
    const kind = kindOf(node);
    if (kind.plan) {
      const r = kind.plan(node, H);
      const pl = { tgs: [], params: [], errors: [], warnings: [], ...r, node, kind };
      pl.ok ??= pl.tgs.length > 0 && pl.errors.length === 0 && pl.params.length > 0;
      return pl;
    }
    const tgs = targets(node);
    const params = [], errors = [];
    for (const p of kind.params) {
      const mode = widget(node, p.mode)?.value ?? "off";
      if (mode === "off") continue;
      if (mode === FOLLOW) { params.push({ ...p, mode }); continue; }
      const spec = widget(node, p.name)?.value ?? "";
      if (p.kind === "combo" && tgs.length && !tgs.some((t) => widget(t, p.target))) {
        errors.push(`${p.name}: no ${kind.noun} here has ${p.target} (set ${p.mode} to off)`);
        continue;
      }
      const res = p.kind === "combo"
        ? expandCombo(spec, comboOptions(tgs.map((t) => widget(t, p.target)).find(Boolean)))
        : expandNumbers(spec, p.kind);
      errors.push(...res.errors.map((e) => `${p.name}: ${e}`));
      if (res.values.length) params.push({ ...p, mode, values: res.values });
      else errors.push(`${p.name}: no values`);
    }
    if (!tgs.length) errors.unshift(isConnected(node) ? `what is wired in is not a ${kind.noun}` : `no ${kind.noun} found`);
    // A target widget that rounds (ComfyUI rounds float widgets to their step
    // by default) would turn finer values into duplicates: say so.
    const warnings = [];
    for (const p of params) {
      if (p.kind !== "float" || !p.values) continue;
      const r = tgs.map((t) => widget(t, p.target)?.options?.round).find((x) => typeof x === "number" && x > 0);
      if (r && p.values.some((v) => Math.abs(Math.round(v / r) * r - v) > 1e-9)) {
        warnings.push(`${p.name} is rounded to ${r} by the ${kind.noun}; for finer values set Settings › Lite Graph › ` +
          `Node Widget › Float widget rounding decimal places (then reload)`);
      }
    }
    return { node, kind, tgs, params, errors, warnings, ok: tgs.length > 0 && errors.length === 0 && params.length > 0 };
  }

  const active = (n) => widget(n, "enabled")?.value && n.mode !== 2 && n.mode !== 4; // 2 muted, 4 bypassed

  // Every active sweep node in the graph, top of the canvas first, with the
  // combined list of swept settings.
  function globalPlan(graph) {
    const plans = graphNodes(graph).filter((n) => kindOf(n) && active(n))
      // A kind's `order` puts it first (changing slowest): a model changes once
      // per sweep of everything else, not every run. Then top of the canvas first.
      .sort((a, b) => (kindOf(a).order ?? 0) - (kindOf(b).order ?? 0)
        || (a.pos?.[1] ?? 0) - (b.pos?.[1] ?? 0) || (a.pos?.[0] ?? 0) - (b.pos?.[0] ?? 0))
      .map(buildPlan);
    const sweep = plans.filter((pl) => pl.ok).flatMap((pl) => pl.params.filter((p) => p.mode === "sweep").map((p) => ({ p, pl })));
    const total = sweep.reduce((n, s) => n * s.p.values.length, 1);
    return { plans, sweep, total };
  }

  const getIndex = (gp) => (gp.plans.find((pl) => pl.ok)?.node.properties?.sweepIndex ?? 0) % gp.total;
  function setIndex(graph, i) {
    for (const n of graphNodes(graph)) if (kindOf(n)) (n.properties ??= {}).sweepIndex = i;
  }

  // Values for combination `index`, per sweep node (Map node -> {widget: value}).
  function valuesAt(gp, index, rollRandom = true) {
    const out = new Map(gp.plans.filter((pl) => pl.ok).map((pl) => [pl.node, {}]));
    let i = index;
    for (let k = gp.sweep.length - 1; k >= 0; k--) {
      const { p, pl } = gp.sweep[k];
      out.get(pl.node)[keyOf(p)] = p.values[i % p.values.length];
      i = Math.floor(i / p.values.length);
    }
    for (const pl of gp.plans) {
      if (!pl.ok) continue;
      const vals = out.get(pl.node);
      for (const p of pl.params) {
        if (p.mode === "random") vals[keyOf(p)] = rollRandom ? p.values[Math.floor(Math.random() * p.values.length)] : "random";
        if (p.mode === "fixed") vals[keyOf(p)] = p.values[0];
        if (p.mode === FOLLOW) vals[keyOf(p)] = FOLLOW;
      }
    }
    return out;
  }

  function describe(pl, vals, ran = false) {
    if (pl.kind.describe) return pl.kind.describe(pl, vals, ran, H);
    const kind = pl.kind;
    const bits = [];
    for (const p of kind.params) {
      const v = vals[keyOf(p)];
      if (v === undefined) continue;
      if (p.target === "steps") bits.push(`${v} st`);
      else if (p.target === "cfg") bits.push(`cfg ${v}`);
      else if (p.label) bits.push(`${p.label} ${v === FOLLOW ? "= model" : v}`);
      else bits.push(String(v));
    }
    return bits.join(" · ") || "nothing set";
  }

  // Writes the values into the target nodes' widgets.
  function apply(pl, vals) {
    if (pl.kind.apply) return pl.kind.apply(pl, vals, H) ?? [];
    const skipped = [];
    for (const t of pl.tgs) {
      for (const p of pl.kind.params) {
        if (!(p.target in vals)) continue;
        const w = widget(t, p.target);
        if (!w) continue; // e.g. strength_clip on a model-only LoRA loader
        if (t.inputs?.some((inp) => inp.widget?.name === p.target && inp.link != null)) {
          skipped.push(`${p.target} on #${t.id} is linked`);
          continue;
        }
        const v = vals[p.target] === FOLLOW ? widget(t, p.follows)?.value : vals[p.target];
        if (v === undefined) continue;
        w.value = v;
        w.callback?.(v);
      }
    }
    return skipped;
  }

  function setStatus(node, text) {
    node.__sweepStatus = text;
    const w = widget(node, "status");
    if (w) w.value = text;
    node.setDirtyCanvas?.(true, true);
  }

  // Shows on every sweep node what the next run will use.
  function previewAll(graph) {
    const gp = globalPlan(graph);
    const idx = getIndex(gp);
    const vals = valuesAt(gp, idx, false);
    const shared = gp.plans.filter((pl) => pl.ok).length > 1 ? " across all sweep nodes" : "";
    for (const n of graphNodes(graph)) {
      if (!kindOf(n)) continue;
      const pl = gp.plans.find((x) => x.node === n);
      if (!pl) { setStatus(n, n.mode === 4 ? "bypassed" : n.mode === 2 ? "muted" : "off"); continue; }
      if (!pl.ok) { setStatus(n, pl.errors.join("; ") || "every setting is off"); continue; }
      const head = gp.sweep.length ? `${gp.total} combinations${shared} (set Run count to ${gp.total}). Next #${idx + 1}` : "Next";
      const on = pl.targetText ?? (isConnected(n) ? ` → ${pl.tgs.map(nameOf).join(", ")}` : ` → all ${pl.tgs.length} ${pl.kind.noun}(s)`);
      const warn = pl.warnings.length ? ` ⚠ ${pl.warnings.join("; ")}` : "";
      setStatus(n, `${head}: ${describe(pl, vals.get(n))}${on}${warn}`);
    }
  }

  // Once per queued prompt (the first sweep node's hook does the work for all).
  let ranThisPrompt = false;
  function beforeQueued(graph) {
    if (ranThisPrompt) return;
    ranThisPrompt = true;
    queueMicrotask(() => { ranThisPrompt = false; });
    const gp = globalPlan(graph);
    if (!gp.plans.some((pl) => pl.ok)) return previewAll(graph);
    const idx = getIndex(gp);
    const vals = valuesAt(gp, idx, true);
    const next = gp.sweep.length ? (idx + 1) % gp.total : 0;
    setIndex(graph, next);
    for (const pl of gp.plans) {
      if (!pl.ok) { setStatus(pl.node, pl.errors.join("; ") || "every setting is off"); continue; }
      const skipped = apply(pl, vals.get(pl.node));
      const tag = gp.sweep.length ? `#${idx + 1}/${gp.total} ` : "";
      const done = gp.sweep.length && next === 0 ? " (sweep done, starts over)" : "";
      setStatus(pl.node, `Ran ${tag}${describe(pl, vals.get(pl.node), true)}${done}${skipped.length ? ` [skipped: ${skipped.join(", ")}]` : ""}`);
    }
  }

  const later = (graph) => setTimeout(() => previewAll(graph ?? app.graph), 0);

  function setup(nodeType) {
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const node = this;
      node.properties ??= {};
      node.properties.sweepIndex ??= 0;

      // Read-only in effect: an edit is put back to the last status text.
      const status = node.addWidget("text", "status", "", () => {
        setTimeout(() => { status.value = node.__sweepStatus ?? ""; node.setDirtyCanvas?.(true, true); }, 0);
      }, { serialize: false });
      status.serializeValue = () => undefined;
      node.addWidget("button", "restart sweep", null, () => {
        setIndex(node.graph, 0);
        previewAll(node.graph);
      }, { serialize: false });

      for (const w of node.widgets) {
        if (w === status || w.type === "button") continue;
        const cb = w.callback;
        w.callback = function () {
          const res = cb?.apply(this, arguments);
          setIndex(node.graph, 0); // a changed sweep starts over
          later(node.graph);
          return res;
        };
      }

      widget(node, "enabled").beforeQueued = () => {
        if (node.graph) beforeQueued(node.graph);
      };

      node.setSize?.([Math.max(node.size?.[0] ?? 0, 440), node.size?.[1] ?? 0]);
      later(node.graph);
      return r;
    };

    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configure?.apply(this, arguments);
      later(this.graph);
      return r;
    };

    const connections = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type) {
      const r = connections?.apply(this, arguments);
      if (type === 1 /* input */ && this.graph) {
        setIndex(this.graph, 0); // different targets: the sweep starts over
        later(this.graph);
      }
      return r;
    };

    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      const r = removed?.apply(this, arguments);
      later(app.graph);
      return r;
    };
  }

  // For a kind's own UI: a changed setting starts the sweep over / refreshes the preview.
  const restart = (graph) => { if (graph) { setIndex(graph, 0); later(graph); } };

  return { version: ENGINE_VERSION, KINDS, setup, restart, refresh: later, H };
}

function sweepEngine(app) {
  const cur = window.__comfySweepEngine;
  if (!cur || (cur.version ?? 0) < ENGINE_VERSION) window.__comfySweepEngine = createEngine(app, cur);
  return window.__comfySweepEngine;
}

// ------------------------------------------------------------------ LoRA kind
// One row per LoRA in the pipeline, each with a mode (on / off / sweep /
// random) and its own strength or min/max. The rows live in
// node.properties.loraSettings, keyed by loader node id (or "id:lora_N" for an
// rgthree Power Lora Loader entry), so they are saved with the workflow.

const MODES = ["on", "off", "sweep", "random"];
const LIST_ROW = 24;
const fmt = (v) => (typeof v === "number" ? String(Number(v.toFixed(3))) : String(v));
const shortName = (s) => {
  const b = String(s).split(/[\\/]/).pop().replace(/\.(safetensors|ckpt|pt|pth|bin)$/i, "");
  return b.length > 30 ? b.slice(0, 29) + "…" : b;
};
const isActive = (n) => n.mode !== 2 && n.mode !== 4; // 2 muted, 4 bypassed
const linkOf = (graph, id) => graph.links?.get?.(id) ?? graph.links?.[id];

// The LoRAs one node loads: [{key, node, name, power, w}].
function loraSources(n, H) {
  if (H.kindOf(n)) return [];
  const out = [];
  const nameW = H.widget(n, "lora_name");
  if (nameW && H.widget(n, "strength_model")) {
    out.push({ key: String(n.id), node: n, name: String(nameW.value ?? ""), power: false });
  }
  for (const w of n.widgets ?? []) {
    const v = w.value;
    if (/^lora_\d+$/i.test(w.name ?? "") && v && typeof v === "object" && v.lora && v.lora !== "None") {
      out.push({ key: `${n.id}:${w.name}`, node: n, name: String(v.lora), power: true, w });
    }
  }
  return out;
}

// The pipeline: with something wired in, every node upstream of it (the whole
// chain, furthest upstream first); otherwise every node, top of the canvas first.
function pipelineNodes(node, H) {
  const graph = node.graph ?? app.graph;
  if (!graph) return [];
  if (!H.isConnected(node)) {
    return [...H.graphNodes(graph)].sort((a, b) =>
      (a.pos?.[1] ?? 0) - (b.pos?.[1] ?? 0) || (a.pos?.[0] ?? 0) - (b.pos?.[0] ?? 0));
  }
  const seen = new Set([node.id]), queue = [node], out = [];
  while (queue.length) {
    const n = queue.shift();
    for (const inp of n.inputs ?? []) {
      if (inp.link == null || (n === node && inp.widget)) continue;
      const link = linkOf(graph, inp.link);
      const from = link && graph.getNodeById(link.origin_id);
      if (!from || seen.has(from.id)) continue;
      seen.add(from.id);
      out.push(from);
      queue.push(from);
    }
  }
  return out.reverse();
}

const findRows = (node, H) => pipelineNodes(node, H).flatMap((n) => loraSources(n, H));
const rowsSig = (rows) => rows.map((r) => `${r.key}|${r.name}|${isActive(r.node)}`).join("\n");

function settingsOf(node) {
  const p = (node.properties ??= {});
  if (!p.loraSettings || typeof p.loraSettings !== "object") p.loraSettings = {};
  return p.loraSettings;
}

// A row's settings; a LoRA seen for the first time starts "on" at the
// loader's current strength (or "off" if a Power Lora entry is switched off).
function rowSettings(node, src, H) {
  const all = settingsOf(node);
  let s = all[src.key];
  if (!s || typeof s !== "object") {
    const cur = src.power ? src.w.value.strength : H.widget(src.node, "strength_model")?.value;
    s = all[src.key] = {
      mode: src.power && src.w.value.on === false ? "off" : "on",
      strength: typeof cur === "number" && cur !== 0 ? cur : 1,
      min: 0.4,
      max: 1,
    };
  }
  s.name = src.name;
  return s;
}

const maxLoras = (node, H) => Math.max(0, Math.floor(Number(H.widget(node, "max_loras")?.value) || 0));

sweepEngine(app).KINDS.LoraSweep = {
  noun: "LoRA loader",
  params: [],

  plan(node, H) {
    const rows = findRows(node, H);
    const step = Number(H.widget(node, "step")?.value);
    const stepS = fmt(step > 0 ? step : 0.1);
    const params = [], errors = [];
    for (const src of rows) {
      const s = rowSettings(node, src, H);
      src.s = s;
      src.active = isActive(src.node);
      if (!src.active || s.mode === "off") continue;
      const label = shortName(src.name);
      if (s.mode === "on") {
        const v = Number(s.strength);
        if (Number.isFinite(v)) params.push({ key: src.key, mode: "fixed", values: [v], label, src });
        else errors.push(`${label}: strength is not a number`);
        continue;
      }
      const res = H.expandNumbers(`${fmt(Number(s.min))}..${fmt(Number(s.max))}:${stepS}`, "float");
      errors.push(...res.errors.map((e) => `${label}: ${e}`));
      if (res.values.length) params.push({ key: src.key, mode: s.mode, values: res.values, label, src });
      else errors.push(`${label}: no values`);
    }
    if (!rows.length) {
      errors.push(H.isConnected(node) ? "no LoRA loader upstream of what is wired in" : "no LoRA loader found");
    }
    node.__loraRows = rows;
    const sig = rowsSig(rows);
    if (sig !== node.__loraRendered) {
      node.__loraRendered = sig;
      setTimeout(() => renderList(node), 0);
    }
    return { tgs: [...new Set(rows.map((r) => r.node))], rows, params, errors, ok: rows.length > 0 && !errors.length, targetText: "" };
  },

  // Each run: "on" LoRAs always, the remaining max_loras slots filled at random
  // from the sweep/random ones; everything else (and "off") at strength 0.
  apply(pl, vals, H) {
    const max = maxLoras(pl.node, H);
    const picked = new Set(pl.params.filter((p) => p.src.s.mode === "on").map((p) => p.key));
    const rest = pl.params.filter((p) => !picked.has(p.key));
    for (let i = rest.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [rest[i], rest[j]] = [rest[j], rest[i]];
    }
    for (const p of rest) {
      if (max && picked.size >= max) break;
      picked.add(p.key);
    }
    vals.__picked = picked;

    const skipped = [];
    const set = (n, name, v) => {
      const w = H.widget(n, name);
      if (!w) return; // e.g. strength_clip on LoraLoaderModelOnly
      if (n.inputs?.some((inp) => inp.widget?.name === name && inp.link != null)) {
        skipped.push(`${name} on #${n.id} is linked`);
        return;
      }
      w.value = v;
      w.callback?.(v);
    };
    for (const src of pl.rows) {
      if (!src.active) continue;
      const use = picked.has(src.key);
      const v = use ? vals[src.key] : 0;
      if (src.power) {
        const cur = src.w.value;
        cur.on = use;
        if (use) {
          cur.strength = v;
          if (cur.strengthTwo != null) cur.strengthTwo = v;
        }
      } else {
        set(src.node, "strength_model", v);
        set(src.node, "strength_clip", v);
      }
      src.node.setDirtyCanvas?.(true, true);
    }
    return skipped;
  },

  describe(pl, vals, ran, H) {
    const max = maxLoras(pl.node, H);
    const picked = ran ? vals.__picked : null;
    const bits = [];
    for (const p of pl.params) {
      if (picked && !picked.has(p.key)) continue;
      const v = vals[p.key];
      bits.push(`${p.label} ${v === "random" ? "random" : fmt(v)}`);
    }
    let text = bits.join(" · ") || (pl.rows.length ? "every LoRA off" : "no LoRA");
    const always = pl.params.filter((p) => p.src.s.mode === "on").length;
    const pool = pl.params.length - always;
    const slots = Math.max(0, max - always);
    const limited = max && pool > 0 && slots < pool;
    if (limited && !ran) {
      text += always
        ? ` (the ${always} "on" + ${slots} of the other ${pool} each run, picked at random)`
        : ` (${slots} of these ${pool} each run, picked at random)`;
    }
    if (limited && ran) text += ` (${picked.size} of ${pl.params.length})`;
    return text;
  },
};

// ------------------------------------------------------------------ LoRA list UI

function listHeight(node) {
  return 6 + Math.max(1, node.__loraRows?.length ?? 0) * LIST_ROW;
}

function changed(node, rerender) {
  if (rerender) renderList(node);
  sweepEngine(app).restart(node.graph);
}

function renderList(node) {
  const el = node.__loraEl;
  if (!el) return;
  const rows = node.__loraRows ?? [];
  el.replaceChildren();
  if (!rows.length) {
    el.textContent = "No LoRA loaders found.";
  }
  const input = (value, disabled, onChange) => {
    const i = document.createElement("input");
    i.type = "number";
    i.step = "0.05";
    i.value = disabled ? "" : fmt(Number(value));
    i.disabled = disabled;
    Object.assign(i.style, {
      width: "100%", boxSizing: "border-box", background: "var(--comfy-input-bg, #222)",
      color: "inherit", border: "1px solid var(--border-color, #444)", borderRadius: "3px", padding: "1px 3px",
      opacity: disabled ? "0.35" : "1",
    });
    i.addEventListener("change", () => onChange(parseFloat(i.value)));
    return i;
  };
  for (const src of rows) {
    const s = src.s ?? rowSettings(node, src, sweepEngine(app).H);
    const line = document.createElement("div");
    Object.assign(line.style, {
      display: "grid", gridTemplateColumns: "minmax(0,1fr) 72px 54px 54px", gap: "4px",
      alignItems: "center", height: `${LIST_ROW - 2}px`, opacity: src.active === false ? "0.45" : "1",
    });

    const label = document.createElement("span");
    label.textContent = `${shortName(src.name)} #${src.node.id}` + (src.active === false ? " (bypassed)" : "");
    label.title = `${src.name}\non ${src.node.title || src.node.type} #${src.node.id}`;
    Object.assign(label.style, { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" });

    const sel = document.createElement("select");
    for (const m of MODES) sel.add(new Option(m, m, false, m === s.mode));
    Object.assign(sel.style, {
      width: "100%", background: "var(--comfy-input-bg, #222)", color: "inherit",
      border: "1px solid var(--border-color, #444)", borderRadius: "3px",
    });
    sel.addEventListener("change", () => { s.mode = sel.value; changed(node, true); });

    line.append(label, sel);
    if (s.mode === "on") {
      const i = input(s.strength, false, (v) => { s.strength = v; changed(node); });
      i.title = "strength";
      i.style.gridColumn = "span 2";
      line.append(i);
    } else {
      const off = s.mode === "off";
      const lo = input(s.min, off, (v) => { s.min = v; changed(node); });
      const hi = input(s.max, off, (v) => { s.max = v; changed(node); });
      lo.title = "min";
      hi.title = "max";
      line.append(lo, hi);
    }
    el.append(line);
  }
  const h = listHeight(node);
  el.style.height = `${h}px`;
  const size = node.computeSize?.();
  if (size) node.setSize([Math.max(node.size?.[0] ?? 0, 460), size[1]]);
  node.setDirtyCanvas?.(true, true);
}

function setupList(nodeType) {
  const created = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = created?.apply(this, arguments);
    const node = this;
    settingsOf(node);
    const el = document.createElement("div");
    Object.assign(el.style, {
      display: "flex", flexDirection: "column", gap: "2px", font: "12px sans-serif",
      color: "var(--input-text, #ddd)", padding: "3px 4px", boxSizing: "border-box", overflow: "hidden",
    });
    // Keep canvas shortcuts (delete, ctrl+c...) from firing while typing.
    el.addEventListener("keydown", (e) => e.stopPropagation());
    node.__loraEl = el;
    const w = node.addDOMWidget?.("loras", "lora_sweep_list", el, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => listHeight(node),
      getMaxHeight: () => listHeight(node),
    });
    if (w) {
      w.serializeValue = () => undefined;
      w.computeSize = (width) => [width, listHeight(node)];
    }
    return r;
  };

  // Notice LoRA loaders being added, removed, swapped or bypassed.
  const draw = nodeType.prototype.onDrawForeground;
  nodeType.prototype.onDrawForeground = function () {
    const r = draw?.apply(this, arguments);
    const now = performance.now();
    if (this.graph && now - (this.__loraChecked ?? 0) > 700) {
      this.__loraChecked = now;
      if (rowsSig(findRows(this, sweepEngine(app).H)) !== this.__loraRendered) sweepEngine(app).refresh(this.graph);
    }
    return r;
  };

  // Workflows saved with the old LoRA Strength Sweep (model_mode, strength_model,
  // clip_mode, strength_clip) load their values into the wrong widgets: reset.
  const configure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function (info) {
    const r = configure?.apply(this, arguments);
    const H = sweepEngine(app).H;
    const maxW = H.widget(this, "max_loras"), stepW = H.widget(this, "step");
    if (maxW && typeof maxW.value !== "number") maxW.value = 0;
    if (stepW && !(typeof stepW.value === "number" && stepW.value > 0)) stepW.value = 0.1;
    settingsOf(this);
    return r;
  };
}

app.registerExtension({
  name: "LoraSweep",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LoraSweep") return;
    sweepEngine(app).setup(nodeType);
    setupList(nodeType);
  },
});
