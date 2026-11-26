"""Reads how an image was made from its embedded metadata.

Pure Python plus Pillow (no ComfyUI imports), so it can be tested on its own.

Sources, highest priority first; each fills only what the ones before it left empty:
  1. Tagged Image Saver essentials (Prompt, Negative, Checkpoint, Lora, Seed, ...)
  2. The ComfyUI API graph ("prompt" chunk), read at the first sampler of the
     graph that feeds the save node
  3. A1111-style "parameters" text (Civitai, Forge, moFlo previews)
"""
from __future__ import annotations

import json
import re

try:
    from . import graphinfo as g
except ImportError:  # run as a plain module in tests
    import graphinfo as g  # type: ignore

FIELDS = ("checkpoint", "positive", "negative", "seed", "steps", "cfg", "sampler", "scheduler", "denoise")

# A1111 sampler label -> (comfy sampler, scheduler implied by the label or None)
A1111_SAMPLERS = {
    "euler a": "euler_ancestral", "euler": "euler", "lms": "lms", "heun": "heun",
    "dpm2": "dpm_2", "dpm2 a": "dpm_2_ancestral", "dpm++ 2s a": "dpmpp_2s_ancestral",
    "dpm++ 2m": "dpmpp_2m", "dpm++ sde": "dpmpp_sde", "dpm++ 2m sde": "dpmpp_2m_sde",
    "dpm++ 2m sde gpu": "dpmpp_2m_sde_gpu", "dpm++ 3m sde": "dpmpp_3m_sde",
    "dpm fast": "dpm_fast", "dpm adaptive": "dpm_adaptive", "ddim": "ddim",
    "ddpm": "ddpm", "lcm": "lcm", "unipc": "uni_pc", "uni_pc": "uni_pc",
    "deis": "deis", "restart": "restart",
}
A1111_SCHEDULERS = {"karras": "karras", "exponential": "exponential", "sgm uniform": "sgm_uniform",
                    "simple": "simple", "normal": "normal", "ddim": "ddim_uniform",
                    "beta": "beta", "kl optimal": "kl_optimal", "align your steps": "normal"}

_LORA_TAG = re.compile(r"<lora:([^:>]+):([-\d.]+)(?::([-\d.]+))?>", re.I)
_SETTING = re.compile(r'\s*([\w][\w \-/().]*?):\s*("(?:\\.|[^\\"])*"|[^,]*)(?:,|$)')


# --- reading the file ----------------------------------------------------
def read_text_chunks(path) -> dict:
    """PNG tEXt/iTXt (and the few other text fields Pillow exposes) as {key: str}."""
    from PIL import Image

    with Image.open(path) as im:
        info = dict(im.info)
        size = im.size
        try:
            exif = im.getexif()
            # 0x9286 UserComment, 0x010e ImageDescription: where JPEG/WebP tools put A1111 text
            for tag in (0x9286, 0x010e):
                v = exif.get(tag)
                if isinstance(v, bytes):
                    v = v.split(b"\x00", 8)[-1].decode("utf-8", "ignore") if v.startswith(b"UNICODE") \
                        else v.decode("utf-8", "ignore")
                if isinstance(v, str) and "Steps:" in v and "parameters" not in info:
                    info["parameters"] = v
        except Exception:
            pass
    out = {k: v for k, v in info.items() if isinstance(k, str) and isinstance(v, str)}
    out["__size__"] = f"{size[0]}x{size[1]}"
    return out


# --- source 1: Tagged Image Saver essentials -----------------------------
def parse_lora_field(text: str) -> list[dict]:
    """'name:0.8, other:0.65/0.5' -> [{name, model, clip}]"""
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, strength = part.rpartition(":")
        if not name:
            name, strength = strength, "1"
        model, _, clip = strength.partition("/")
        m, c = g.as_float(model), g.as_float(clip) if clip else None
        out.append({"name": name.strip(), "model": 1.0 if m is None else m, "clip": c})
    return out


def from_essentials(info: dict) -> dict:
    if not any(k in info for k in ("Checkpoint", "Prompt", "Seed", "Lora")):
        return {}
    num = lambda k, f: f(info[k]) if info.get(k) else None
    r = {
        "checkpoint": info.get("Checkpoint") or info.get("Model"),
        "positive": info.get("Prompt"),
        "negative": info.get("Negative"),
        "sampler": info.get("Sampler"),
        "scheduler": info.get("Scheduler"),
        "seed": _safe(num, "Seed", int),
        "steps": _safe(num, "Steps", int),
        "cfg": _safe(num, "CFG", float),
        "denoise": _safe(num, "Denoise", float),
    }
    if "Lora" in info:
        r["loras"] = parse_lora_field(info["Lora"])
    return r


def _safe(fn, *args):
    try:
        return fn(*args)
    except (TypeError, ValueError):
        return None


# --- source 2: the ComfyUI API graph ---------------------------------------
def _is_save(node: dict) -> bool:
    ct = str(node.get("class_type", "")).lower()
    return "save" in ct and "images" in (node.get("inputs") or {})


def from_graph(prompt) -> dict:
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except ValueError:
            return {}
    if not isinstance(prompt, dict) or not prompt:
        return {}
    prompt = {str(k): v for k, v in prompt.items() if isinstance(v, dict)}
    never = lambda _nid: False
    saves = sorted((n for n, v in prompt.items() if _is_save(v)), key=g.natural_id)
    # Image Recall nodes record sampler settings on themselves; they are a source of
    # values, not the sampler, so never pick one as the anchor.
    not_sampler = lambda nid: prompt[nid].get("class_type") == "ImageRecall"
    if saves:
        save = saves[0]
        dist = g.upstream_distances(prompt, save)
        anchor = g.pick_anchor(prompt, dist, save, not_sampler, "farthest")
    else:
        samplers = sorted((n for n, v in prompt.items()
                           if not not_sampler(n) and g._is_sampler(v.get("inputs") or {})),
                          key=g.natural_id)
        if not samplers:
            return {}
        save = anchor = samplers[0]
        dist = g.upstream_distances(prompt, save)
    if anchor == save and saves:
        return {}  # no sampler upstream of the save node
    f = g.harvest_fields(prompt, anchor, never)
    pos, neg = g.prompt_texts(prompt, anchor)
    loras = g.collect_loras(prompt, dist, save, never, None)
    return {
        "checkpoint": f.get("model"),
        "positive": pos or None,
        "negative": neg or None,
        "sampler": f.get("sampler"),
        "scheduler": f.get("scheduler"),
        "seed": _safe(int, f["seed"]) if f.get("seed") is not None else None,
        "steps": _safe(int, f["steps"]) if f.get("steps") is not None else None,
        "cfg": g.as_float(f.get("cfg")),
        "denoise": g.as_float(f.get("denoise")),
        "loras": [{"name": l["name"], "model": l["model"], "clip": l["clip"]} for l in loras],
    }


# --- source 3: A1111 "parameters" ----------------------------------------
def parse_parameters(text: str) -> dict:
    if not text or not text.strip():
        return {}
    lines = text.strip().split("\n")
    settings_at = None
    for i in range(len(lines) - 1, -1, -1):
        if re.match(r"\s*Steps:\s*\d", lines[i]):
            settings_at = i
            break
    settings = {}
    if settings_at is not None:
        for m in _SETTING.finditer(lines[settings_at]):
            key, val = m.group(1).strip(), m.group(2).strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            settings[key] = val
        body = lines[:settings_at]
    else:
        body = lines
    pos_lines, neg_lines, into = [], [], None
    for line in body:
        if line.startswith("Negative prompt:"):
            into = neg_lines
            line = line[len("Negative prompt:"):]
        (into if into is not None else pos_lines).append(line)
    positive = "\n".join(pos_lines).strip()
    loras = []
    for m in _LORA_TAG.finditer(positive):
        model = g.as_float(m.group(2))
        loras.append({"name": m.group(1).strip(),
                      "model": 1.0 if model is None else model,
                      "clip": g.as_float(m.group(3)) if m.group(3) else None})
    positive = _LORA_TAG.sub("", positive)
    positive = re.sub(r"\s+,", ",", re.sub(r"[ \t]{2,}", " ", positive))
    positive = re.sub(r",(\s*,)+", ",", positive).strip(" ,\n")

    sampler, scheduler = a1111_sampler(settings.get("Sampler"), settings.get("Schedule type"))
    r = {
        "positive": positive or None,
        "negative": "\n".join(neg_lines).strip() or None,
        "checkpoint": settings.get("Model"),
        "sampler": sampler,
        "scheduler": scheduler,
        "seed": _safe(int, settings.get("Seed")),
        "steps": _safe(int, settings.get("Steps")),
        "cfg": _safe(float, settings.get("CFG scale")),
        "denoise": _safe(float, settings.get("Denoising strength")),
    }
    if loras:
        r["loras"] = loras
    return r


def a1111_sampler(label, schedule=None):
    """('DPM++ 2M Karras', None) -> ('dpmpp_2m', 'karras'). ComfyUI names pass through."""
    if not label:
        return None, _a1111_schedule(schedule)
    text = label.strip()
    low = text.lower()
    scheduler = _a1111_schedule(schedule)
    for suffix, sched in (("karras", "karras"), ("exponential", "exponential"),
                          ("sgm uniform", "sgm_uniform"), ("beta", "beta")):
        if low.endswith(" " + suffix):
            low = low[: -len(suffix) - 1].strip()
            scheduler = scheduler or sched
    sampler = A1111_SAMPLERS.get(low, text if re.fullmatch(r"[a-z0-9_]+", text) else None)
    return sampler, scheduler


def _a1111_schedule(schedule):
    if not schedule or schedule.strip().lower() == "automatic":
        return None
    low = schedule.strip().lower()
    return A1111_SCHEDULERS.get(low, low.replace(" ", "_"))


# --- combining -------------------------------------------------------------
def recall(info: dict) -> dict:
    """Everything the metadata says about how the image was made.

    Keys: checkpoint, positive, negative, seed, steps, cfg, sampler, scheduler,
    denoise (None when unknown), loras (list), size, sources (list of names)."""
    out = {k: None for k in FIELDS}
    out.update(loras=None, size=info.get("__size__"), sources=[])
    layers = (
        ("Tagged essentials", lambda: from_essentials(info)),
        ("ComfyUI graph", lambda: from_graph(info.get("prompt"))),
        ("A1111 parameters", lambda: parse_parameters(info.get("parameters", ""))),
    )
    for name, read in layers:
        try:
            got = read()
        except Exception:  # one bad chunk should not hide the others
            got = {}
        used = False
        for k in FIELDS:
            if out[k] in (None, "") and got.get(k) not in (None, ""):
                out[k] = got[k]
                used = True
        if out["loras"] is None and got.get("loras") is not None:
            out["loras"] = got["loras"]
            used = used or bool(got["loras"])
        if used:
            out["sources"].append(name)
    out["loras"] = out["loras"] or []
    return out


def recall_file(path) -> dict:
    return recall(read_text_chunks(path))


def describe(r: dict, notes=()) -> str:
    """Human-readable summary for the node's info box."""
    if not r.get("sources"):
        return "No generation metadata found in this image.\n" + "\n".join(notes)
    fmt = g.fmt_num
    lines = [f"Checkpoint: {r.get('checkpoint') or '?'}"]
    if r["loras"]:
        parts = []
        for l in r["loras"]:
            s = fmt(l["model"])
            if l.get("clip") is not None and l["clip"] != l["model"]:
                s += "/" + fmt(l["clip"])
            parts.append(f"{g.model_stem(l['name'])} {s}")
        lines.append("LoRAs: " + ", ".join(parts))
    else:
        lines.append("LoRAs: none")
    settings = [f"{label} {fmt(r[k])}" for k, label in
                (("seed", "seed"), ("steps", "steps"), ("cfg", "cfg"), ("denoise", "denoise"))
                if r.get(k) is not None]
    if r.get("sampler"):
        settings.append(f"{r['sampler']}/{r.get('scheduler') or '?'}")
    if r.get("size"):
        settings.append(r["size"])
    lines.append(" · ".join(settings))
    lines.append(f"Prompt: {r.get('positive') or '(none)'}")
    lines.append(f"Negative: {r.get('negative') or '(none)'}")
    lines.extend(notes)
    lines.append(f"Read from: {', '.join(r['sources'])}")
    return "\n".join(lines)
