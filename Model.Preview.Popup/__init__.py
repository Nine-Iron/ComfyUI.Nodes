"""Model Preview Popup - ComfyUI extension (no nodes).

Hovering a model in the Model Library sidebar shows a large popup with the
preview images saved beside the model file:

    <name>.preview.<ext>      the main preview (what ComfyUI shows as the icon)
    <name>.<ext>              also accepted as the main preview
    <name>.preview-2.<ext>    extras, as moflo -previews N saves them
    ...

Click an extra in the popup to view it large. Beside the picture, a column
shows what it was made with (prompt, negative, sampler settings, LoRAs), read
from <name>.preview.json (moFlo saves these) or from metadata inside the
image (A1111/Forge parameters, ComfyUI prompt graph). The popup also shows the
model's base model, and its "Base models" button picks which base models are
shown in the Model Library and in loader dropdowns. When a model has preview files
beside it, the thumbnail some authors embed inside the model file is ignored,
so the sidebar icon stays on your preview instead of switching on hover.

Install: copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
"""

import json
import logging
import os
import re
import struct
import zlib

import folder_paths

WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

IMAGE_EXTS = ("png", "jpg", "jpeg", "webp", "gif")
_EXTRA = re.compile(r"^preview-(\d+)\.(%s)$" % "|".join(IMAGE_EXTS), re.IGNORECASE)


def _resolve(folder, filename):
    """Full path of a model, only if ComfyUI itself lists it in that folder."""
    if not folder or not filename or folder not in folder_paths.folder_names_and_paths:
        return None
    if filename not in folder_paths.get_filename_list(folder):
        return None
    return folder_paths.get_full_path(folder, filename)


def preview_files(model_path):
    """Preview images beside a model: main first, then extras in number order."""
    directory = os.path.dirname(model_path)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    main, plain, extras = [], [], []
    for name in names:
        if not name.startswith(stem + "."):
            continue
        suffix = name[len(stem) + 1:]
        low = suffix.lower()
        if low in IMAGE_EXTS:
            plain.append(name)
        elif low.startswith("preview.") and low[len("preview."):] in IMAGE_EXTS:
            main.append(name)
        else:
            m = _EXTRA.match(suffix)
            if m:
                extras.append((int(m.group(1)), name))
    extras.sort()
    return sorted(main) + sorted(plain) + [n for _, n in extras]


# ------------------------------------------------------- generation settings
# What each preview picture was made with, from (first match wins):
#   1. a sidecar <name>.preview.json / <name>.preview-N.json, which moFlo
#      writes with the image's data from Civitai;
#   2. metadata embedded in the image: A1111/Forge "parameters" text (PNG text
#      chunk or EXIF UserComment) or a ComfyUI prompt graph (PNG "prompt" chunk,
#      or "prompt:" in the EXIF of a ComfyUI WebP).
# Everything is normalised to {prompt, negative, settings: [[label, value]],
# loras: [[name, weight]], source}.

MAX_TEXT = 20000  # longest prompt passed on; longer ones are cut
_info_cache = {}  # image path -> (stamp, info)


def _clean(v):
    if v is None or isinstance(v, (dict, list)):
        return ""
    if isinstance(v, float):
        v = round(v, 4)
        if v == int(v):
            v = int(v)
    return str(v).strip()


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Settings shown, in this order: (label, keys tried in A1111 / Civitai / ComfyUI spelling)
SETTING_KEYS = [
    ("Model", ("Model", "model", "ckpt_name", "unet_name")),
    ("Sampler", ("Sampler", "sampler", "sampler_name")),
    ("Scheduler", ("Schedule type", "scheduler", "Scheduler")),
    ("Steps", ("Steps", "steps")),
    ("CFG", ("CFG scale", "cfgScale", "cfg")),
    ("Guidance", ("Distilled CFG Scale", "guidance", "Guidance")),
    ("Seed", ("Seed", "seed", "noise_seed")),
    ("Size", ("Size", "size")),
    ("Clip skip", ("Clip skip", "clipSkip")),
    ("Denoise", ("Denoising strength", "denoise", "denoisingStrength")),
    ("Hires upscale", ("Hires upscale",)),
    ("Hires upscaler", ("Hires upscaler",)),
    ("Hires steps", ("Hires steps",)),
    ("VAE", ("VAE", "vae", "vae_name")),
]
_KNOWN = {k.lower() for _, keys in SETTING_KEYS for k in keys}
# Keys never shown as "other" settings: noise, hashes, or shown elsewhere.
_SKIP = {"prompt", "negativeprompt", "negative prompt", "resources", "civitairesources", "hashes",
         "comfy", "workflow", "additionalresources", "model hash", "vae hash", "lora hashes",
         "ti hashes", "version", "width", "height", "draft", "nsfw", "external", "software",
         "process", "type", "baseModel".lower(), "modelids", "versionids", "engine"}


def _settings(d):
    """Ordered [label, value] pairs from a flat dict of settings."""
    low = {str(k).lower(): v for k, v in d.items()}
    out = []
    for label, keys in SETTING_KEYS:
        for k in keys:
            v = _clean(low.get(k.lower()))
            if v:
                out.append([label, v])
                break
    if not any(l == "Size" for l, _ in out):
        w, h = _clean(low.get("width")), _clean(low.get("height"))
        if w and h:
            out.append(["Size", f"{w}x{h}"])
    for k, v in d.items():  # anything else short and readable, after the known ones
        kl = str(k).lower()
        if kl in _KNOWN or kl in _SKIP or "hash" in kl or len(out) >= 24:
            continue
        v = _clean(v)
        if v and len(v) <= 120:
            out.append([str(k), v])
    return out


def _info(prompt, negative, settings, loras, source):
    prompt, negative = (prompt or "").strip(), (negative or "").strip()
    if not (prompt or negative or settings):
        return None
    return {"prompt": prompt[:MAX_TEXT], "negative": negative[:MAX_TEXT], "settings": settings,
            "loras": loras, "source": source}


_A1111_PAIR = re.compile(r'\s*([^:,]+?):\s*("(?:\\.|[^\\"])*"|[^,]*)(?:,|$)')
_LORA_TAG = re.compile(r"<(?:lora|lyco):([^:>]+)(?::([^:>]+))?[^>]*>", re.IGNORECASE)


def parse_a1111(text):
    """A1111/Forge "parameters": prompt, "Negative prompt: ...", then "Steps: ..., ..."."""
    lines = text.strip().split("\n")
    at = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].lstrip().startswith("Steps:")), None)
    params = {}
    if at is not None:
        for k, v in _A1111_PAIR.findall(" ".join(lines[at:])):
            v = v.strip()
            if v.startswith('"') and v.endswith('"'):
                try:
                    v = json.loads(v)
                except ValueError:
                    v = v[1:-1]
            params[k.strip()] = v
        lines = lines[:at]
    prompt, negative, in_neg = [], [], False
    for line in lines:
        if line.startswith("Negative prompt:"):
            in_neg, line = True, line[len("Negative prompt:"):].lstrip()
        (negative if in_neg else prompt).append(line)
    prompt = "\n".join(prompt)
    loras = [[m.group(1), _clean(m.group(2)) or "1"] for m in _LORA_TAG.finditer(prompt)]
    return _info(prompt, "\n".join(negative), _settings(params), loras, "a1111")


def _comfy_graph(prompt):
    """ComfyUI API-format prompt: {id: {class_type, inputs}}."""
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except ValueError:
            return None
    if not isinstance(prompt, dict):
        return None
    nodes = {str(k): v for k, v in prompt.items()
             if isinstance(v, dict) and isinstance(v.get("class_type"), str) and isinstance(v.get("inputs", {}), dict)}
    for v in nodes.values():
        v.setdefault("inputs", {})
    return nodes or None


def parse_comfy(prompt):
    nodes = _comfy_graph(prompt)
    if not nodes:
        return None

    def link(v):
        return isinstance(v, list) and len(v) == 2 and str(v[0]) in nodes

    def scalar(v, depth=0):
        """A widget value, following links into primitive / constant nodes."""
        if link(v):
            if depth > 6:
                return None
            ins = nodes[str(v[0])].get("inputs", {})
            for k in ("value", "seed", "noise_seed", "int", "float", "number", "string", "text"):
                if k in ins:
                    return scalar(ins[k], depth + 1)
            return None
        return v if isinstance(v, (int, float, str)) and not isinstance(v, bool) else None

    TEXT_KEYS = ("text", "text_g", "text_l", "t5xxl", "clip_l", "prompt", "string", "value", "wildcard_text",
                 "populated_text", "positive", "text_a", "text_b")
    COND_KEYS = ("conditioning", "conditioning_1", "conditioning_2", "conditioning_to", "conditioning_from",
                 "positive", "negative", "cond", "guider")

    def text_of(v, depth=0, seen=None):
        """The prompt text a conditioning input comes from."""
        seen = seen if seen is not None else set()
        if isinstance(v, str):
            return v
        if not link(v) or depth > 12 or str(v[0]) in seen:
            return ""
        seen.add(str(v[0]))
        ins = nodes[str(v[0])].get("inputs", {})
        parts = []
        for k in TEXT_KEYS:  # an encoder: its own text widgets
            if k in ins and k not in ("positive",):
                t = ins[k] if isinstance(ins[k], str) else text_of(ins[k], depth + 1, seen) if link(ins[k]) else ""
                if t and t.strip() and t not in parts:
                    parts.append(t)
        if parts:
            return "\n".join(parts)
        for k in COND_KEYS:  # a pass-through (guidance, combine, area...): follow it
            if k in ins and link(ins[k]):
                t = text_of(ins[k], depth + 1, seen)
                if t:
                    parts.append(t)
        return "\n".join(dict.fromkeys(parts))

    def order(nid):
        return (0, int(nid)) if nid.isdigit() else (1, nid)

    samplers = [n for n in sorted(nodes, key=order) if any(
        s in nodes[n]["class_type"] for s in ("KSampler", "SamplerCustom", "Sampler"))]
    if not samplers:
        return None
    s_in = nodes[samplers[0]].get("inputs", {})

    # Positive and negative: on the sampler, or on the guider it is wired to.
    pos_src, neg_src = s_in.get("positive"), s_in.get("negative")
    if pos_src is None and link(s_in.get("guider")):
        g_in = nodes[str(s_in["guider"][0])].get("inputs", {})
        pos_src = g_in.get("positive", g_in.get("conditioning"))
        neg_src = g_in.get("negative")
    prompt_text = text_of(pos_src) if pos_src is not None else ""
    negative_text = text_of(neg_src) if neg_src is not None else ""

    # Settings: the sampler's own widgets first, then the first value any node has.
    flat = {}
    want = ("seed", "noise_seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "guidance",
            "ckpt_name", "unet_name", "vae_name")
    for nid in samplers[:1] + sorted(nodes, key=order):
        node = nodes[nid]
        ct = node["class_type"]
        if "Lora" in ct or "LoRA" in ct:
            continue
        for k in want:
            if k in flat or k not in node.get("inputs", {}):
                continue
            v = scalar(node["inputs"][k])
            if v is not None and v != "":
                flat[k] = v
    if "noise_seed" in flat and "seed" not in flat:
        flat["seed"] = flat.pop("noise_seed")
    for nid in sorted(nodes, key=order):
        node = nodes[nid]
        if "Latent" in node["class_type"] and "Empty" in node["class_type"]:
            w, h = scalar(node["inputs"].get("width")), scalar(node["inputs"].get("height"))
            if w and h:
                flat["Size"] = f"{w}x{h}"
                break
    for k in ("ckpt_name", "unet_name"):
        if k in flat:
            flat["model"] = os.path.splitext(os.path.basename(str(flat.pop(k)).replace("\\", "/")))[0]
            break
    flat.pop("unet_name", None)

    loras = []
    for nid in sorted(nodes, key=order):
        node = nodes[nid]
        ins = node.get("inputs", {})
        if "lora_name" in ins:
            name, w = scalar(ins["lora_name"]), scalar(ins.get("strength_model", ins.get("strength", 1)))
            if name and name != "None":
                loras.append([os.path.splitext(os.path.basename(str(name).replace("\\", "/")))[0], _clean(w) or "1"])
        for v in ins.values():  # multi-LoRA loaders: {"on": true, "lora": "x", "strength": 1}
            if isinstance(v, dict) and v.get("lora") and v.get("on", True) and v["lora"] != "None":
                loras.append([os.path.splitext(os.path.basename(str(v["lora"]).replace("\\", "/")))[0],
                              _clean(v.get("strength", 1)) or "1"])
    return _info(prompt_text, negative_text, _settings(flat), loras, "comfyui")


def parse_civitai(meta):
    """The "meta" Civitai keeps for an image (A1111-style keys, sometimes a ComfyUI graph)."""
    if not isinstance(meta, dict):
        return None
    loras = []
    for r in meta.get("resources") or []:
        if isinstance(r, dict) and str(r.get("type", "")).lower() in ("lora", "locon", "lycoris", "dora") and r.get("name"):
            loras.append([str(r["name"]), _clean(r.get("weight")) or "1"])
    prompt, negative = meta.get("prompt") or "", meta.get("negativePrompt") or ""
    settings = _settings(meta)
    comfy = None
    if not prompt and meta.get("comfy"):  # generated on Civitai with ComfyUI: only the graph
        c = meta["comfy"]
        if isinstance(c, str):
            try:
                c = json.loads(c)
            except ValueError:
                c = None
        if isinstance(c, dict):
            comfy = parse_comfy(c.get("prompt", c))
    if comfy:
        have = {l for l, _ in comfy["settings"]}
        comfy["settings"] += [s for s in settings if s[0] not in have]
        comfy["loras"] = comfy["loras"] or loras
        comfy["source"] = "civitai"
        return comfy
    if not loras:
        loras = [[m.group(1), _clean(m.group(2)) or "1"] for m in _LORA_TAG.finditer(prompt)]
    return _info(prompt, negative, settings, loras, "civitai")


# --- reading metadata out of image files (header chunks only, stdlib) --------

def _png_texts(f):
    out = {}
    if f.read(8) != b"\x89PNG\r\n\x1a\n":
        return out
    while True:
        head = f.read(8)
        if len(head) < 8:
            break
        n, kind = struct.unpack(">I4s", head)
        if kind == b"IEND":
            break
        if kind in (b"tEXt", b"iTXt", b"zTXt") and n <= 16 * 1024 * 1024:
            data = f.read(n)
            try:
                key, rest = data.split(b"\0", 1)
                key = key.decode("latin-1")
                if kind == b"tEXt":
                    out[key] = rest.decode("utf-8", "replace")
                elif kind == b"zTXt":
                    out[key] = zlib.decompress(rest[1:]).decode("utf-8", "replace")
                else:
                    compressed = rest[0] == 1
                    rest = rest[2:]
                    _lang, rest = rest.split(b"\0", 1)
                    _tkey, rest = rest.split(b"\0", 1)
                    out[key] = (zlib.decompress(rest) if compressed else rest).decode("utf-8", "replace")
            except (ValueError, IndexError, zlib.error):
                pass
            f.seek(4, 1)  # CRC
        else:
            f.seek(n + 4, 1)
    return out


def _exif_texts(tiff):
    """Text tags from an EXIF (TIFF) block: IFD0 strings and the UserComment."""
    out = {}
    if tiff.startswith(b"Exif\0\0"):
        tiff = tiff[6:]
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        return out
    e = "<" if tiff[:2] == b"II" else ">"

    def ifd(off):
        tags = {}
        if off <= 0 or off + 2 > len(tiff):
            return tags
        count = struct.unpack(e + "H", tiff[off:off + 2])[0]
        for i in range(min(count, 512)):
            p = off + 2 + i * 12
            if p + 12 > len(tiff):
                break
            tag, typ, cnt = struct.unpack(e + "HHI", tiff[p:p + 8])
            size = {1: 1, 2: 1, 3: 2, 4: 4, 7: 1}.get(typ)
            if size is None:
                continue
            ln = size * cnt
            if ln <= 4:
                raw = tiff[p + 8:p + 8 + ln]
            else:
                vo = struct.unpack(e + "I", tiff[p + 8:p + 12])[0]
                raw = tiff[vo:vo + ln]
            tags[tag] = (typ, raw)
        return tags

    def as_int(raw):
        return struct.unpack(e + "I", raw[:4])[0] if len(raw) >= 4 else 0

    ifd0 = ifd(struct.unpack(e + "I", tiff[4:8])[0])
    for tag, name in ((0x010E, "ImageDescription"), (0x010F, "Make"), (0x0110, "Model")):
        if tag in ifd0 and ifd0[tag][0] == 2:
            out[name] = ifd0[tag][1].rstrip(b"\0").decode("utf-8", "replace")
    if 0x8769 in ifd0:
        sub = ifd(as_int(ifd0[0x8769][1]))
        if 0x9286 in sub:
            raw = sub[0x9286][1]
            code, body = raw[:8], raw[8:]
            if code.startswith(b"UNICODE"):
                # piexif writes big-endian; some tools write little-endian.
                cands = [body.decode("utf-16-be", "replace"), body.decode("utf-16-le", "replace")]
                text = max(cands, key=lambda s: sum(c.isascii() and c.isprintable() or c in "\n\t" for c in s))
            else:
                text = body.decode("utf-8", "replace")
            out["UserComment"] = text.rstrip("\0")
    return out


def _jpeg_texts(f):
    if f.read(2) != b"\xff\xd8":
        return {}
    out = {}
    while True:
        b = f.read(2)
        if len(b) < 2 or b[0] != 0xFF:
            break
        marker = b[1]
        if marker in (0xD9, 0xDA):  # end, or start of the picture data
            break
        n = struct.unpack(">H", f.read(2))[0]
        data = f.read(n - 2)
        if marker == 0xE1 and data.startswith(b"Exif\0\0"):
            out.update(_exif_texts(data))
        elif marker == 0xFE:
            out.setdefault("Comment", data.decode("utf-8", "replace"))
    return out


def _webp_texts(f):
    head = f.read(12)
    if head[:4] != b"RIFF" or head[8:12] != b"WEBP":
        return {}
    out = {}
    while True:
        h = f.read(8)
        if len(h) < 8:
            break
        kind, n = h[:4], struct.unpack("<I", h[4:])[0]
        if kind == b"EXIF" and n <= 16 * 1024 * 1024:
            out.update(_exif_texts(f.read(n)))
            if n & 1:
                f.seek(1, 1)
        else:
            f.seek(n + (n & 1), 1)
    return out


def embedded_info(path):
    """Generation settings embedded in an image file, or None."""
    try:
        with open(path, "rb") as f:
            sig = f.read(12)
            f.seek(0)
            if sig.startswith(b"\x89PNG"):
                texts = _png_texts(f)
            elif sig.startswith(b"\xff\xd8"):
                texts = _jpeg_texts(f)
            elif sig[:4] == b"RIFF" and sig[8:12] == b"WEBP":
                texts = _webp_texts(f)
            else:
                return None
    except (OSError, struct.error):
        return None
    if texts.get("prompt"):  # ComfyUI PNG
        info = parse_comfy(texts["prompt"])
        if info:
            return info
    for k in ("Model", "Make", "ImageDescription"):  # ComfyUI WebP: "prompt:{...}" in an EXIF string
        v = texts.get(k, "")
        if v.startswith("prompt:"):
            info = parse_comfy(v[len("prompt:"):])
            if info:
                return info
    for k in ("parameters", "UserComment", "Comment", "ImageDescription"):
        v = texts.get(k, "")
        if v.strip().startswith("{"):
            try:  # some tools store Civitai-style JSON
                info = parse_civitai(json.loads(v))
            except ValueError:
                info = None
            if info:
                return info
        elif "Steps:" in v or "Negative prompt:" in v:
            return parse_a1111(v)
    return None


def _sidecar(image_path):
    """<name>.preview.json beside <name>.preview.png (only for preview-named pictures,
    since <name>.json beside a model is that model's own info file)."""
    stem = os.path.splitext(image_path)[0]
    if not re.search(r"\.preview(-\d+)?$", stem, re.IGNORECASE):
        return None
    return stem + ".json"


def sidecar_info(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    meta = data.get("meta", data)
    if _comfy_graph(meta):  # a bare ComfyUI prompt graph
        return parse_comfy(meta)
    info = parse_civitai(meta)
    if info and data.get("url"):
        info["url"] = str(data["url"])
    return info


def generation_info(image_path):
    """Normalised generation settings for one preview picture, or None (cached by mtime)."""
    side = _sidecar(image_path)
    stamp = []
    for p in (image_path, side):
        try:
            st = os.stat(p) if p else None
            stamp.append((st.st_mtime, st.st_size) if st else None)
        except OSError:
            stamp.append(None)
    stamp = tuple(stamp)
    hit = _info_cache.get(image_path)
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        info = (sidecar_info(side) if stamp[1] else None) or embedded_info(image_path)
    except Exception as e:  # odd metadata must never break the popup
        logging.warning("[Model Preview Popup] could not read settings from %s: %s", image_path, e)
        info = None
    if len(_info_cache) > 2000:
        _info_cache.clear()
    _info_cache[image_path] = (stamp, info)
    return info


# ------------------------------------------------------------- base models
# Which base model each model file is for, worked out in this order:
#   1. the moFlo state file (<models>/.moflo.json)
#   2. a tag in the name: flux1_dev_x, x_flux1_dev, x.flux1_dev, or a flux1_dev/ folder
#   3. the base model the file itself records (kohya / modelspec metadata)
#   4. a guess from the tensor names in the file header
# Folders where filtering applies; text encoders, VAEs and upscalers are never hidden.
FILTERED_FOLDERS = ["checkpoints", "loras", "diffusion_models", "unet", "embeddings",
                    "controlnet", "hypernetworks"]

LABELS = {
    "flux1_dev": "Flux.1 D", "flux1_schnell": "Flux.1 S", "flux1_krea": "Flux.1 Krea",
    "flux1_kontext": "Flux.1 Kontext", "flux2_dev": "Flux.2 D", "sdxl1": "SDXL 1.0", "sd1": "SD 1.5",
    "sd21": "SD 2.1", "sd3": "SD 3", "sd35": "SD 3.5", "qwen": "Qwen", "krea2": "Krea 2", "pony": "Pony",
    "pony_v7": "Pony V7", "illustrious": "Illustrious", "noobai": "NoobAI", "chroma": "Chroma",
    "hidream": "HiDream", "zimageturbo": "Z-Image Turbo", "zimagebase": "Z-Image Base",
    "unknown": "Unknown",
}

_state_cache = {"mtime": None, "map": {}}
_base_cache = {}  # path -> (mtime, size, (tag, label, source))


def _state_bases():
    """{absolute path: (tag, label)} from the moFlo state file."""
    path = os.path.join(folder_paths.models_dir, ".moflo.json")
    if not os.path.exists(path):  # not yet moved over from moFlo's old name
        path = os.path.join(folder_paths.models_dir, ".civitai-likes.json")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    if _state_cache["mtime"] != mtime:
        mapping = {}
        try:
            with open(path, encoding="utf-8") as f:
                files = json.load(f).get("files", {})
            for rel, e in files.items():
                if isinstance(e, dict) and e.get("base"):
                    full = os.path.normpath(os.path.join(folder_paths.models_dir, rel))
                    mapping[full] = (e["base"], e.get("baseName") or "")
        except (OSError, ValueError, AttributeError):
            pass
        _state_cache.update(mtime=mtime, map=mapping)
    return _state_cache["map"]


def _known_tags():
    return set(LABELS) | {t for t, _ in _state_bases().values()}


def _tag_from_name(path, tags):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    parent = os.path.basename(os.path.dirname(path)).lower()
    for tag in sorted(tags, key=len, reverse=True):  # longest first: sdxl1 before sd1
        if tag == "unknown":
            continue
        if stem.startswith(tag + "_") or stem.endswith("_" + tag) or stem.endswith("." + tag) or parent == tag:
            return tag
    return None


def _tag_from_header(path):
    if not path.lower().endswith((".safetensors", ".sft")):
        return None, None
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 64 * 1024 * 1024:
                return None, None
            header = json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return None, None
    meta = header.get("__metadata__") or {}
    ver = str(meta.get("ss_base_model_version", "")).lower()
    arch = str(meta.get("modelspec.architecture", "")).lower()
    for text in (arch, ver):
        if not text:
            continue
        if "krea" in text and "2" in text:
            return "krea2", "metadata"
        if "schnell" in text:
            return "flux1_schnell", "metadata"
        if "flux" in text:
            return ("flux2_dev" if "flux-2" in text or "flux2" in text else "flux1_dev"), "metadata"
        if "xl" in text:
            return "sdxl1", "metadata"
        if "qwen" in text:
            return "qwen", "metadata"
        if "v1" in text or "sd_v1" in text:
            return "sd1", "metadata"
        if "v2" in text:
            return "sd21", "metadata"
    keys = [k.lower() for k in header if k != "__metadata__"]
    has = lambda part: any(part in k for k in keys)
    if has("txtfusion.projector"):
        return "krea2", "guess"
    if has("double_blocks") or has("single_transformer_blocks"):
        return ("flux1_schnell" if not has("guidance") and has("double_blocks.0.img_attn") else "flux1_dev"), "guess"
    if has("conditioner.embedders.1") or has("lora_te2_") or has("add_embedding"):
        return "sdxl1", "guess"
    if has("cond_stage_model.transformer") or (has("lora_te_text_model") and has("lora_unet")):
        return "sd1", "guess"
    return None, None


def base_of(path):
    """(tag, label, source) for one model file."""
    path = os.path.normpath(path)
    try:
        st = os.stat(path)
    except OSError:
        return ("unknown", LABELS["unknown"], "none")
    from_state = _state_bases().get(path)
    if from_state:
        tag, name = from_state
        return (tag, name or LABELS.get(tag, tag), "civitai")
    hit = _base_cache.get(path)
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    tag = _tag_from_name(path, _known_tags())
    source = "name"
    if not tag:
        tag, source = _tag_from_header(path)
    if not tag:
        tag, source = "unknown", "none"
    result = (tag, LABELS.get(tag, tag), source)
    _base_cache[path] = (st.st_mtime, st.st_size, result)
    return result


def all_bases():
    """{folder: {filename: tag}} for the filtered folders, plus labels."""
    folders, labels = {}, {}
    for folder in FILTERED_FOLDERS:
        if folder not in folder_paths.folder_names_and_paths:
            continue
        entries = {}
        for filename in folder_paths.get_filename_list(folder):
            full = folder_paths.get_full_path(folder, filename)
            if not full:
                continue
            tag, label, _ = base_of(full)
            entries[filename] = tag
            labels.setdefault(tag, label)
        folders[folder] = entries
    return {"folders": folders, "labels": labels}


try:
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    @routes.get("/Model.Preview.Popup/list")
    async def _list(request):
        folder = request.query.get("folder", "")
        filename = request.query.get("filename", "")
        path = _resolve(folder, filename)
        if not path:
            return web.json_response({"found": False, "images": []})
        tag, label, source = base_of(path)
        images = preview_files(path)
        directory = os.path.dirname(path)
        info = {}
        for name in images:  # generation settings of each picture, where known
            gi = generation_info(os.path.join(directory, name))
            if gi:
                info[name] = gi
        return web.json_response({"found": True, "images": images, "info": info,
                                  "base": tag, "baseLabel": label, "baseSource": source})

    @routes.get("/Model.Preview.Popup/bases")
    async def _bases(request):
        return web.json_response(all_bases())

    @routes.get("/Model.Preview.Popup/image")
    async def _image(request):
        folder = request.query.get("folder", "")
        filename = request.query.get("filename", "")
        name = request.query.get("name", "")
        path = _resolve(folder, filename)
        if not path or name not in preview_files(path):  # only listed previews, never other files
            return web.Response(status=404)
        resp = web.FileResponse(os.path.join(os.path.dirname(path), name))
        resp.headers["Cache-Control"] = "no-cache"
        return resp
except Exception as e:  # keep ComfyUI starting even if the routes can't register
    logging.warning("[Model Preview Popup] could not register routes: %s", e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
