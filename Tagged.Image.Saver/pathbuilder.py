"""Builds a save path from the executed ComfyUI API prompt.

Pure Python, no ComfyUI imports, so it can be tested on its own.
"""
from __future__ import annotations

import datetime
import hashlib
import re
from collections import deque

DEFAULT_TEMPLATE = "{workflow}/{date}/{model}/{sampler}-{scheduler}/{time}-{lora_names}-{steps}-{cfg}-{denoise}-{seed}"

SEGMENT_BYTE_LIMIT = 200
MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft")

# Field name(s) to look for on the sampler (or upstream of it) for each token.
FIELD_KEYS = {
    "sampler": ("sampler_name",),
    "scheduler": ("scheduler",),
    "steps": ("steps",),
    "cfg": ("cfg",),
    "denoise": ("denoise",),
    "seed": ("seed", "noise_seed"),
    "guidance": ("guidance",),
    "model": ("ckpt_name", "unet_name"),
}

# Keys tried on a source node when a sampler/LoRA field is fed by a link.
PASSTHROUGH_KEYS = (
    "value", "seed", "noise_seed", "int", "float", "number",
    "string", "text", "sampler_name", "scheduler", "lora_name",
)

LORA_OUTPUT_TYPES = ("MODEL", "CLIP", "HOOKS")

# --- indexed stack fields: (pattern, role) --------------------------------
_INDEXED_PATTERNS = [
    (r"^lora_name_0*(\d+)$", "name"),            # Comfyroll, Efficiency
    (r"^lora_0*(\d+)_name$", "name"),            # Easy-Use
    (r"^lora_0*(\d+)$", "name"),                 # rgthree stack, WanVideo multi
    (r"^lora_wt_0*(\d+)$", "simple"),            # Efficiency simple mode
    (r"^lora_0*(\d+)_strength$", "simple"),      # Easy-Use simple mode
    (r"^strength_0*(\d+)$", "simple"),           # rgthree stack, WanVideo multi
    (r"^lora_strength_0*(\d+)$", "simple"),
    (r"^weight_0*(\d+)$", "simple"),
    (r"^model_str_0*(\d+)$", "model"),           # Efficiency advanced mode
    (r"^lora_0*(\d+)_model_strength$", "model"), # Easy-Use advanced mode
    (r"^model_strength_0*(\d+)$", "model"),
    (r"^model_weight_0*(\d+)$", "model"),        # Comfyroll
    (r"^clip_str_0*(\d+)$", "clip"),
    (r"^lora_0*(\d+)_clip_strength$", "clip"),
    (r"^clip_strength_0*(\d+)$", "clip"),
    (r"^clip_weight_0*(\d+)$", "clip"),
    (r"^switch_0*(\d+)$", "switch"),             # Comfyroll "On"/"Off"
    (r"^(?:enabled|on|toggle)_0*(\d+)$", "switch"),
    (r"^lora_0*(\d+)_(?:on|enabled|toggle)$", "switch"),
]
INDEXED_PATTERNS = [(re.compile(p), role) for p, role in _INDEXED_PATTERNS]

SINGLE_NAME_KEYS = ("lora_name", "lora", "lora_file")
SINGLE_MODEL_KEYS = ("strength_model", "lora_model_strength", "model_strength",
                     "strength", "lora_strength", "weight", "lora_wt")
SINGLE_CLIP_KEYS = ("strength_clip", "lora_clip_strength", "clip_strength")

_BAD_CHARS = re.compile(r"[\x00-\x1f\x7f/\\:*?\"<>|%\s'`$;&!#{}]+")
_TOKEN = re.compile(r"\{([a-zA-Z_]+)(?::([^}]*))?\}")


# --- small helpers -------------------------------------------------------
def is_link(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def unwrap(value):
    if isinstance(value, dict) and "__value__" in value and len(value) <= 2:
        return value["__value__"]
    return value


def as_float(value):
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def flag_on(value) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("on", "true", "yes", "1", "enabled", "enable")
    return True


def name_is_off(value) -> bool:
    return not isinstance(value, str) or value.strip().lower() in ("", "none", "null")


def fmt_num(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "nan"
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return "0" if text in ("", "-0") else text
    return str(value)


def sanitize(text: str, max_len: int | None = None) -> str:
    text = _BAD_CHARS.sub("_", str(text))
    text = re.sub(r"_+", "_", text).strip("._- ")
    if max_len and len(text) > max_len:
        text = text[:max_len].rstrip("._- ")
    return text


def model_stem(name: str) -> str:
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    lower = base.lower()
    for ext in MODEL_EXTENSIONS:
        if lower.endswith(ext):
            return base[: -len(ext)]
    return base


def fit_bytes(segment: str, limit: int = SEGMENT_BYTE_LIMIT) -> str:
    raw = segment.encode("utf-8")
    if len(raw) <= limit:
        return segment
    tag = "~" + hashlib.sha1(raw).hexdigest()[:6]
    cut = raw[: limit - len(tag)].decode("utf-8", "ignore").rstrip("._- ")
    return cut + tag


def comfy_date(fmt: str, now: datetime.datetime) -> str:
    """ComfyUI-style date tokens (yyyy, MM, dd, hh, mm, ss) -> strftime."""
    out = fmt
    for token, code in (("yyyy", "%Y"), ("yy", "%y"), ("MM", "%m"), ("dd", "%d"),
                        ("HH", "%H"), ("hh", "%H"), ("mm", "%M"), ("ss", "%S")):
        out = out.replace(token, code)
    return now.strftime(out)


def natural_id(node_id: str):
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p)
                 for p in re.split(r"[:.]", str(node_id)))


# --- graph walking -------------------------------------------------------
def upstream_distances(prompt: dict, start_id: str) -> dict[str, int]:
    """BFS over input links. Bypassed/muted nodes never appear in the prompt."""
    start_id = str(start_id)
    dist = {start_id: 0}
    queue = deque([start_id])
    while queue:
        nid = queue.popleft()
        inputs = (prompt.get(nid) or {}).get("inputs") or {}
        for value in inputs.values():
            if is_link(value):
                src = str(value[0])
                if src in prompt and src not in dist:
                    dist[src] = dist[nid] + 1
                    queue.append(src)
    return dist


def resolve(prompt: dict, value, field: str, depth: int = 0):
    """Return a literal value, following links into primitive/selector nodes."""
    if not is_link(value):
        return unwrap(value)
    if depth > 8:
        return None
    node = prompt.get(str(value[0]))
    if not node:
        return None
    inputs = node.get("inputs") or {}
    for key in (field,) + PASSTHROUGH_KEYS:
        if key in inputs:
            got = resolve(prompt, inputs[key], field, depth + 1)
            if got is not None:
                return got
    scalars = [v for v in inputs.values() if isinstance(v, (str, int, float, bool))]
    if len(scalars) == 1:
        return scalars[0]
    return None


# --- sampler detection ---------------------------------------------------
def _is_sampler(inputs: dict) -> bool:
    return "sampler_name" in inputs and any(
        k in inputs for k in ("steps", "scheduler", "cfg", "seed", "noise_seed"))


def _is_custom_sampler(inputs: dict) -> bool:
    return is_link(inputs.get("sampler")) and any(
        is_link(inputs.get(k)) for k in ("sigmas", "guider", "noise"))


def pick_anchor(prompt, dist, save_id, ignored, pick):
    def choose(pred):
        found = [n for n in dist if n != save_id and not ignored(n)
                 and pred((prompt[n].get("inputs") or {}))]
        if not found:
            return None
        if pick == "farthest":
            return max(found, key=lambda n: (dist[n], natural_id(n)))
        return min(found, key=lambda n: (dist[n], natural_id(n)))

    return choose(_is_sampler) or choose(_is_custom_sampler) or save_id


def harvest_fields(prompt, anchor, ignored) -> dict:
    """Each field from the anchor first, then the nearest node upstream of it."""
    order = sorted(upstream_distances(prompt, anchor).items(),
                   key=lambda kv: (kv[1], natural_id(kv[0])))
    values = {}
    for token, keys in FIELD_KEYS.items():
        for nid, _ in order:
            if nid != anchor and ignored(nid):
                continue
            inputs = prompt[nid].get("inputs") or {}
            hit = next((k for k in keys if k in inputs), None)
            if hit is None:
                continue
            got = resolve(prompt, inputs[hit], hit)
            if got is not None and not isinstance(got, (list, dict)):
                values[token] = got
                break
    return values


# --- LoRA extraction -----------------------------------------------------
def _entry(prompt, name, model, clip):
    name = resolve(prompt, name, "lora_name")
    if name_is_off(name):
        return None
    model_f = as_float(resolve(prompt, model, "strength")) if model is not None else 1.0
    clip_f = as_float(resolve(prompt, clip, "strength")) if clip is not None else None
    if model_f is None:
        model_f = 1.0
    if model_f == 0 and (clip_f is None or clip_f == 0):
        return None
    return {"name": name, "model": model_f, "clip": clip_f}


def _from_dict_entries(prompt, inputs):
    out = []

    def take(item):
        if not isinstance(item, dict):
            return
        name = item.get("lora", item.get("name", item.get("lora_name")))
        if not isinstance(name, str):
            return
        if not flag_on(item.get("on", item.get("active", item.get("enabled")))):
            return
        model = item.get("strength", item.get("model_strength", item.get("strength_model", 1.0)))
        clip = item.get("strengthTwo", item.get("clipStrength",
                        item.get("clip_strength", item.get("strength_clip"))))
        entry = _entry(prompt, name, model, clip)
        if entry:
            out.append(entry)

    for value in inputs.values():
        value = unwrap(value)
        if isinstance(value, dict):
            take(value)
        elif isinstance(value, list) and not is_link(value):
            for item in value:
                take(item)
    return out


def _from_indexed(prompt, inputs):
    slots: dict[int, dict] = {}
    for key, value in inputs.items():
        for pattern, role in INDEXED_PATTERNS:
            m = pattern.match(key)
            if not m:
                continue
            if role == "name" and isinstance(unwrap(value), (dict, list)):
                break
            slots.setdefault(int(m.group(1)), {})[role] = value
            break
    slots = {i: s for i, s in slots.items() if "name" in s}
    if not slots:
        return []

    count = resolve(prompt, inputs.get("lora_count", inputs.get("num_loras")), "int")
    if isinstance(count, (int, float)) and not isinstance(count, bool):
        base = 0 if min(slots) == 0 else 1
        slots = {i: s for i, s in slots.items() if i - base < int(count)}

    mode = str(resolve(prompt, inputs.get("input_mode", inputs.get("mode")), "mode") or "").lower()
    simple = mode == "simple"

    out = []
    for i in sorted(slots):
        s = slots[i]
        if not flag_on(resolve(prompt, s.get("switch"), "switch")):
            continue
        if simple:
            model, clip = s.get("simple", s.get("model")), None
        else:
            model, clip = s.get("model", s.get("simple")), s.get("clip")
        entry = _entry(prompt, s["name"], model, clip)
        if entry:
            out.append(entry)
    return out


def _from_single(prompt, inputs):
    name_key = next((k for k in SINGLE_NAME_KEYS if k in inputs
                     and not isinstance(unwrap(inputs[k]), (dict, list))), None)
    if name_key is None:
        return []
    model = next((inputs[k] for k in SINGLE_MODEL_KEYS if k in inputs), None)
    clip = next((inputs[k] for k in SINGLE_CLIP_KEYS if k in inputs), None)
    entry = _entry(prompt, inputs[name_key], model, clip)
    return [entry] if entry else []


def extract_loras(prompt: dict, inputs: dict) -> list[dict]:
    if not flag_on(resolve(prompt, inputs.get("toggle", inputs.get("enabled")), "toggle")):
        return []
    return (_from_dict_entries(prompt, inputs)
            or _from_indexed(prompt, inputs)
            or _from_single(prompt, inputs))


def _could_load_lora(class_type, return_types) -> bool:
    if return_types is None:
        return True
    types = return_types(class_type)
    if types is None:
        return True
    return any(t in LORA_OUTPUT_TYPES or "LORA" in t.upper() for t in types)


def collect_loras(prompt, dist, save_id, ignored, return_types) -> list[dict]:
    loras, seen = [], set()
    nodes = sorted((n for n in dist if n != save_id),
                   key=lambda n: (-dist[n], natural_id(n)))
    for nid in nodes:
        node = prompt[nid]
        if ignored(nid) or not _could_load_lora(node.get("class_type", ""), return_types):
            continue
        for entry in extract_loras(prompt, node.get("inputs") or {}):
            key = (model_stem(entry["name"]), entry["model"], entry["clip"])
            if key not in seen:
                seen.add(key)
                loras.append(dict(entry, node=nid))
    return loras


def format_loras(loras, name_max, empty_text) -> str:
    parts = []
    for lora in loras:
        strength = fmt_num(lora["model"])
        if lora["clip"] is not None and lora["clip"] != lora["model"]:
            strength += "c" + fmt_num(lora["clip"])
        parts.append(f"{sanitize(model_stem(lora['name']), name_max)}_{strength}")
    return "_".join(parts) if parts else empty_text


# --- template rendering --------------------------------------------------
def clean_segment(segment: str) -> str:
    segment = re.sub(r"([-_])[-_]+", r"\1", segment)
    return segment.strip("._- ")


def build_prefix(prompt, node_id, template=DEFAULT_TEMPLATE, workflow_name="",
                 sampler_pick="nearest", lora_name_max=40, no_lora_text="nolora",
                 ignore_classes="", width=0, height=0, return_types=None, now=None):
    """Return (filename_prefix, info) for ComfyUI's get_save_image_path."""
    prompt = prompt or {}
    save_id = str(node_id)
    now = now or datetime.datetime.now()

    ignore = [s.strip().lower() for s in str(ignore_classes or "").split(",") if s.strip()]

    def ignored(nid):
        cls = str((prompt.get(nid) or {}).get("class_type", "")).lower()
        return any(token in cls for token in ignore)

    dist = upstream_distances(prompt, save_id) if save_id in prompt else {}
    anchor = pick_anchor(prompt, dist, save_id, ignored, sampler_pick) if dist else save_id
    fields = harvest_fields(prompt, anchor, ignored) if anchor in prompt else {}
    loras = collect_loras(prompt, dist, save_id, ignored, return_types) if dist else []

    name = str(workflow_name or "").strip()
    if not name or name.lower() == "auto":
        name = "unsaved"

    def render(match):
        token, arg = match.group(1).lower(), match.group(2)
        if token == "workflow":
            stem = name.replace("\\", "/").rsplit("/", 1)[-1]
            if stem.lower().endswith(".json"):
                stem = stem[:-5]
            return sanitize(stem)
        if token == "date":
            return sanitize(comfy_date(arg or "yyyy-MM-dd", now))
        if token == "time":
            return sanitize(comfy_date(arg or "hhmmss", now))
        if token == "loras":
            return sanitize(format_loras(loras, lora_name_max, no_lora_text))
        if token == "lora_names":
            names = [sanitize(model_stem(l["name"]), lora_name_max) for l in loras]
            return "_".join(names) if names else sanitize(no_lora_text)
        if token == "lora_count":
            return str(len(loras))
        if token == "width":
            return str(width)
        if token == "height":
            return str(height)
        if token == "model":
            value = fields.get("model")
            return sanitize(model_stem(value)) if value is not None else ""
        if token in FIELD_KEYS:
            return sanitize(fmt_num(fields.get(token)))
        return match.group(0)

    rendered = _TOKEN.sub(render, template or DEFAULT_TEMPLATE)
    segments = []
    for seg in rendered.replace("\\", "/").split("/"):
        seg = clean_segment(seg)
        if seg in ("", ".", ".."):
            continue
        segments.append(fit_bytes(seg))
    prefix = "/".join(segments) or "ComfyUI"

    info = {"anchor": anchor if anchor != save_id else None,
            "anchor_class": (prompt.get(anchor) or {}).get("class_type") if anchor != save_id else None,
            "fields": fields, "loras": loras}
    return prefix, info


# --- image metadata ------------------------------------------------------
# Text inputs on an encoder, and inputs a conditioning pass-through node
# (FluxGuidance, ConditioningCombine, ...) is followed through.
ENCODER_TEXT_KEYS = ("text", "text_g", "text_l", "t5xxl", "clip_l", "prompt",
                     "wildcard_text", "populated_text")
COND_KEYS = ("conditioning", "conditioning_1", "conditioning_2", "conditioning_to",
             "conditioning_from", "cond", "positive", "negative")
# Text that generator nodes in this repo record on their own node entry.
META_TEXT_KEYS = ("final_prompt", "generated_prompt")


def _linked_text(prompt, value, depth=0):
    """A text input's value: literal, a generator's recorded output, or a primitive."""
    if not is_link(value):
        value = unwrap(value)
        return value if isinstance(value, str) else ""
    if depth > 8:
        return ""
    node = prompt.get(str(value[0])) or {}
    meta = node.get("_meta") if isinstance(node.get("_meta"), dict) else {}
    # Nodes with several text outputs (Image Recall) record each one by output slot.
    per_output = meta.get("output_texts")
    if isinstance(per_output, dict) and isinstance(per_output.get(str(value[1])), str):
        return per_output[str(value[1])]
    for key in META_TEXT_KEYS:
        if isinstance(meta.get(key), str):
            return meta[key]
    inputs = node.get("inputs") or {}
    for key in ("text", "string", "value", "prompt", "populated_text", "wildcard_text"):
        if key in inputs:
            got = _linked_text(prompt, inputs[key], depth + 1)
            if got:
                return got
    return ""


def _cond_text(prompt, value, depth=0, seen=None):
    """The prompt text a conditioning input was encoded from."""
    seen = set() if seen is None else seen
    if not is_link(value) or depth > 12 or str(value[0]) in seen:
        return ""
    nid = str(value[0])
    seen.add(nid)
    inputs = (prompt.get(nid) or {}).get("inputs") or {}
    parts = []
    for key in ENCODER_TEXT_KEYS:
        if key in inputs:
            text = _linked_text(prompt, inputs[key]).strip()
            if text and text not in parts:
                parts.append(text)
    if parts:
        return "\n".join(parts)
    for key in COND_KEYS:
        if is_link(inputs.get(key)):
            text = _cond_text(prompt, inputs[key], depth + 1, seen)
            if text and text not in parts:
                parts.append(text)
    return "\n".join(parts)


def prompt_texts(prompt, anchor) -> tuple[str, str]:
    """(positive, negative) text used by the anchor sampler."""
    inputs = ((prompt or {}).get(str(anchor)) or {}).get("inputs") or {}
    pos, neg = inputs.get("positive"), inputs.get("negative")
    if pos is None and is_link(inputs.get("guider")):
        g = (prompt.get(str(inputs["guider"][0])) or {}).get("inputs") or {}
        pos, neg = g.get("positive", g.get("conditioning")), g.get("negative")
    return _cond_text(prompt, pos), _cond_text(prompt, neg)


def format_lora_field(loras) -> str:
    """'name:0.8, other:0.65/0.5' (clip strength after '/' only when it differs)."""
    parts = []
    for lora in loras:
        strength = fmt_num(lora["model"])
        if lora["clip"] is not None and lora["clip"] != lora["model"]:
            strength += "/" + fmt_num(lora["clip"])
        parts.append(f"{model_stem(lora['name'])}:{strength}")
    return ", ".join(parts)


def find_vae(prompt, save_id):
    """vae_name of the nearest separate VAE loader upstream of the save node, or None."""
    dist = upstream_distances(prompt, save_id) if save_id in prompt else {}
    for nid in sorted(dist, key=lambda n: (dist[n], natural_id(n))):
        inputs = prompt[nid].get("inputs") or {}
        if "vae_name" in inputs:
            got = resolve(prompt, inputs["vae_name"], "vae_name")
            if isinstance(got, str) and not name_is_off(got):
                return got
    return None


def build_metadata(info, width, height, positive="", negative="", vae=None) -> dict:
    """Ordered {keyword: text} of what produced the image; empty values left out.
    'parameters' is the same data in A1111 form for Civitai and other readers."""
    f = info.get("fields") or {}
    loras = info.get("loras") or []
    model = f.get("model")
    num = lambda k: fmt_num(f.get(k)) if f.get(k) is not None else ""
    out = {
        "Prompt": positive,
        "Negative": negative,
        "Model": model_stem(model) if model else "",
        "Checkpoint": str(model) if model else "",
        "Sampler": str(f.get("sampler") or ""),
        "Scheduler": str(f.get("scheduler") or ""),
        "Steps": num("steps"),
        "CFG": num("cfg"),
        "Seed": str(f["seed"]) if f.get("seed") is not None else "",
        "Denoise": num("denoise"),
        "Guidance": num("guidance"),
        "Size": f"{width}x{height}" if width and height else "",
        "Lora": format_lora_field(loras),
        "VAE": model_stem(vae) if vae else "",
    }
    out = {k: v for k, v in out.items() if v}

    pairs = [("Steps", out.get("Steps")), ("Sampler", out.get("Sampler")),
             ("Schedule type", out.get("Scheduler")), ("CFG scale", out.get("CFG")),
             ("Seed", out.get("Seed")), ("Size", out.get("Size")), ("Model", out.get("Model")),
             ("Denoising strength", out.get("Denoise")), ("Distilled CFG Scale", out.get("Guidance")),
             ("VAE", out.get("VAE"))]
    settings = ", ".join(f"{k}: {v}" for k, v in pairs if v)
    lora_tags = " ".join(f"<lora:{model_stem(l['name'])}:{fmt_num(l['model'])}>" for l in loras)
    lines = [" ".join(t for t in (positive, lora_tags) if t)]
    if negative:
        lines.append("Negative prompt: " + negative)
    if settings:
        lines.append(settings)
    params = "\n".join(l for l in lines if l)
    if params:
        out["parameters"] = params
    return out
