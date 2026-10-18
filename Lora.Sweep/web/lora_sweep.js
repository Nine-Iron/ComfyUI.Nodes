// LoRA Strength Sweep: sets strength_model / strength_clip on LoRA loaders
// right before each queued run. ComfyUI calls beforeQueued() on every widget
// before it builds each prompt of a batch, so run N of "Run x N" gets
// combination N. Within the node, clip changes fastest.
import { app } from "../../scripts/app.js";

// Shared sweep engine. Sampler Sweep and LoRA Strength Sweep each ship a copy;
// whichever loads first creates it and the other registers into it, so with
// both installed all sweep nodes share one counter (a full grid of runs).
const ENGINE_VERSION = 2;

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

  // One node's settings: the value list of every parameter that is not off.
  function buildPlan(node) {
    const kind = kindOf(node);
    const tgs = targets(node);
    const params = [], errors = [];
    for (const p of kind.params) {
      const mode = widget(node, p.mode)?.value ?? "off";
      if (mode === "off") continue;
      if (mode === FOLLOW) { params.push({ ...p, mode }); continue; }
      const spec = widget(node, p.name)?.value ?? "";
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
      .sort((a, b) => (a.pos?.[1] ?? 0) - (b.pos?.[1] ?? 0) || (a.pos?.[0] ?? 0) - (b.pos?.[0] ?? 0))
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
      out.get(pl.node)[p.target] = p.values[i % p.values.length];
      i = Math.floor(i / p.values.length);
    }
    for (const pl of gp.plans) {
      if (!pl.ok) continue;
      const vals = out.get(pl.node);
      for (const p of pl.params) {
        if (p.mode === "random") vals[p.target] = rollRandom ? p.values[Math.floor(Math.random() * p.values.length)] : "random";
        if (p.mode === FOLLOW) vals[p.target] = FOLLOW;
      }
    }
    return out;
  }

  function describe(kind, vals) {
    const bits = [];
    for (const p of kind.params) {
      const v = vals[p.target];
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
      const on = isConnected(n) ? ` → ${pl.tgs.map(nameOf).join(", ")}` : ` → all ${pl.tgs.length} ${pl.kind.noun}(s)`;
      const warn = pl.warnings.length ? ` ⚠ ${pl.warnings.join("; ")}` : "";
      setStatus(n, `${head}: ${describe(pl.kind, vals.get(n))}${on}${warn}`);
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
      setStatus(pl.node, `Ran ${tag}${describe(pl.kind, vals.get(pl.node))}${done}${skipped.length ? ` [skipped: ${skipped.join(", ")}]` : ""}`);
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

  return { version: ENGINE_VERSION, KINDS, setup };
}

function sweepEngine(app) {
  const cur = window.__comfySweepEngine;
  if (!cur || (cur.version ?? 0) < ENGINE_VERSION) window.__comfySweepEngine = createEngine(app, cur);
  return window.__comfySweepEngine;
}

sweepEngine(app).KINDS.LoraSweep = {
  noun: "LoRA loader",
  params: [
    { name: "strength_model", mode: "model_mode", target: "strength_model", kind: "float", label: "model" },
    { name: "strength_clip", mode: "clip_mode", target: "strength_clip", kind: "float", label: "clip",
      follows: "strength_model" },
  ],
};

app.registerExtension({
  name: "LoraSweep",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "LoraSweep") sweepEngine(app).setup(nodeType);
  },
});
