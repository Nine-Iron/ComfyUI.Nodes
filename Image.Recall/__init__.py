"""Image Recall for ComfyUI.

Image Recall: drop an image on the node and it loads the checkpoint and LoRAs the
image was made with, and outputs its prompts and sampler settings.
Outpaint Canvas: places the image on an SDXL-sized canvas of a chosen aspect
ratio and returns the mask of the area to fill.
Upscale Size: the render and final size for the one-pass SDXL upscale.

The subgraph in subgraphs/ chains them into outpaint / outpaint + upscale / upscale.
"""
import hashlib
import json
import logging
import math
import os
import re

import torch

import comfy.samplers
import comfy.utils
import folder_paths
import nodes

from . import recall as rc
from .graphinfo import model_stem

log = logging.getLogger("ImageRecall")

FROM_IMAGE = "(from image)"
MODES = ["outpaint", "outpaint + upscale", "upscale"]
DEFAULTS = {"seed": 0, "steps": 25, "cfg": 6.0, "sampler": "euler", "scheduler": "normal"}


# --- finding files the metadata names -------------------------------------
def _top_dir(name):
    name = str(name or "").replace("\\", "/")
    return name.split("/", 1)[0] if "/" in name else ""


def find_model(kind, name, prefer_dir=""):
    """The entry in folder_paths' list for `name`: exact path first, then by file stem
    (case-insensitive), preferring one in `prefer_dir` (e.g. sdxl1/ next to the checkpoint)."""
    if not name:
        return None
    available = folder_paths.get_filename_list(kind)
    norm = str(name).replace("\\", "/").strip()
    by_path = {a.replace("\\", "/"): a for a in available}
    if norm in by_path:
        return by_path[norm]
    stem = model_stem(norm).lower()
    hits = [a for a in available if model_stem(a).lower() == stem]
    if not hits:
        return None
    if prefer_dir:
        same = [a for a in hits if _top_dir(a) == prefer_dir]
        if same:
            return same[0]
    return hits[0]


def plan(r, checkpoint="(from image)", apply_loras=True, lora_scale=1.0):
    """What would be loaded for a recall result: (checkpoint, [(lora, model, clip)], notes)."""
    notes = []
    if checkpoint and checkpoint != FROM_IMAGE:
        ckpt = checkpoint
    else:
        ckpt = find_model("checkpoints", r.get("checkpoint"))
        if r.get("checkpoint") and not ckpt:
            notes.append(f"! Checkpoint not found: {r['checkpoint']}. Pick one in 'checkpoint'.")
        elif not r.get("checkpoint"):
            notes.append("! The image does not name a checkpoint. Pick one in 'checkpoint'.")
    loras = []
    if apply_loras:
        for l in r.get("loras") or []:
            found = find_model("loras", l["name"], _top_dir(ckpt))
            if not found:
                notes.append(f"! LoRA not found, skipped: {l['name']}")
                continue
            clip = l["clip"] if l.get("clip") is not None else l["model"]
            loras.append((found, l["model"] * lora_scale, clip * lora_scale))
    for key in ("seed", "steps", "cfg", "sampler", "scheduler"):
        if r.get(key) in (None, ""):
            notes.append(f"! No {key} in the image; using {DEFAULTS[key]}.")
    return ckpt, loras, notes


def settings(r):
    """seed, steps, cfg, sampler, scheduler with defaults and ComfyUI's valid names."""
    sampler = r.get("sampler") if r.get("sampler") in comfy.samplers.KSampler.SAMPLERS else None
    scheduler = r.get("scheduler") if r.get("scheduler") in comfy.samplers.KSampler.SCHEDULERS else None
    return (int(r["seed"]) if r.get("seed") is not None else DEFAULTS["seed"],
            int(r["steps"]) if r.get("steps") else DEFAULTS["steps"],
            float(r["cfg"]) if r.get("cfg") is not None else DEFAULTS["cfg"],
            sampler or DEFAULTS["sampler"],
            scheduler or DEFAULTS["scheduler"])


def join_prompt(prefix, text):
    prefix, text = (p.strip().strip(",").strip() for p in (prefix or "", text or ""))
    if prefix and text.lower().startswith(prefix.lower()):
        return text  # recalled from an image this node already prefixed
    return ", ".join(p for p in (prefix, text) if p)


def texts(r, prompt_prefix, prompt_override, negative_override):
    base = prompt_override if (prompt_override or "").strip() else (r.get("positive") or "")
    neg = negative_override if (negative_override or "").strip() else (r.get("negative") or "")
    return join_prompt(prompt_prefix, base), neg.strip()


def outpaint_text(r, prompt_prefix, prompt_override, subject, subject_weight):
    """Prompt for the outpainted area: the recalled prompt with the main subject weighted down."""
    base = prompt_override if (prompt_override or "").strip() else (r.get("positive") or "")
    return join_prompt(prompt_prefix, deweight(base, subject, subject_weight))


def parse_settings(text):
    """The panel's edits: a JSON object, or {} when empty or invalid."""
    try:
        s = json.loads(text) if isinstance(text, str) and text.strip() else {}
    except ValueError:
        s = {}
    return s if isinstance(s, dict) else {}


def effective(r, w):
    """What will actually run: the recalled values, then the node's widgets, then the
    panel's edits (the `settings` JSON) on top. `w` is the node's widget values."""
    s = parse_settings(w.get("settings"))
    ckpt_choice = s.get("checkpoint") or w.get("checkpoint") or FROM_IMAGE
    ckpt, loras, notes = plan(r, ckpt_choice, bool(w.get("apply_loras", True)), float(w.get("lora_scale", 1.0)))
    if isinstance(s.get("loras"), list):
        loras = []
        for l in s["loras"]:
            if not isinstance(l, dict) or not l.get("on", True):
                continue
            found = find_model("loras", l.get("name"), _top_dir(ckpt))
            if not found:
                notes.append(f"! LoRA not found, skipped: {l.get('name')}")
                continue
            m = float(l.get("model", 1.0))
            c = l.get("clip")
            loras.append((found, m, m if c is None else float(c)))
    seed, steps, cfg, sampler, scheduler = settings(r)
    try:
        seed = int(s.get("seed", seed))
        steps = max(1, int(s.get("steps", steps)))
        cfg = float(s.get("cfg", cfg))
    except (TypeError, ValueError):
        pass
    if s.get("sampler") in comfy.samplers.KSampler.SAMPLERS:
        sampler = s["sampler"]
    if s.get("scheduler") in comfy.samplers.KSampler.SCHEDULERS:
        scheduler = s["scheduler"]
    pp, po, no = w.get("prompt_prefix", ""), w.get("prompt_override", ""), w.get("negative_override", "")
    positive, negative = texts(r, pp, po, no)
    outpaint = outpaint_text(r, pp, po, w.get("subject", ""), float(w.get("subject_weight", 0.4)))
    if isinstance(s.get("positive"), str) and s["positive"].strip():
        positive = outpaint = s["positive"].strip()
    if isinstance(s.get("negative"), str):
        negative = s["negative"].strip()
    mode = w.get("mode") if w.get("mode") in MODES else MODES[1]
    if mode == "upscale":
        outpaint = positive  # no new area: the subject is not weighted down
    return {"ckpt": ckpt, "loras": loras, "notes": notes, "positive": positive, "negative": negative,
            "outpaint": outpaint, "seed": seed, "steps": steps, "cfg": cfg, "sampler": sampler,
            "scheduler": scheduler, "mode": mode,
            "do_outpaint": mode != "upscale", "do_upscale": mode != "outpaint"}


# --- main subject, weighted down for the outpainted area ------------------
QUALITY_TAGS = re.compile(r"^(score_\w+|masterpiece|best quality|(high|highly|ultra|very|extremely)[ -]?"
                          r"(quality|detailed|res|resolution)|\d+k|uhd|hdr|absurdres|photo ?realistic|"
                          r"realistic|raw photo|award[ -]winning.*|official art|newest|amazing quality)$", re.I)
WEIGHTED = re.compile(r"^\((.*):\s*[-+]?\d*\.?\d+\s*\)$", re.S)


def split_prompt(prompt):
    """Comma-separated parts, ignoring commas inside (), [] and <>."""
    parts, depth, cur = [], 0, ""
    for ch in prompt or "":
        if ch in "([<{":
            depth += 1
        elif ch in ")]>}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def unweighted(part):
    """'(cat:1.2)' / '((cat))' -> 'cat'."""
    p = part.strip()
    while True:
        m = WEIGHTED.match(p)
        if m:
            p = m.group(1).strip()
        elif len(p) > 1 and p[0] + p[-1] in ("()", "[]"):
            p = p[1:-1].strip()
        else:
            return p


def auto_subject(prompt):
    """The first part of the prompt that isn't a quality tag or a LoRA/embedding tag, minus a/an/the."""
    for part in split_prompt(prompt):
        core = unweighted(part)
        if not core or core.startswith("<") or core.lower().startswith("embedding:") or QUALITY_TAGS.match(core):
            continue
        return re.sub(r"^(a|an|the|one)\s+", "", core, flags=re.I)
    return ""


def deweight(prompt, subject, weight):
    """The prompt with every part mentioning the subject re-weighted to `weight` (0 drops it).
    subject: comma-separated terms; empty = auto (first real part of the prompt); 'none' = off."""
    subject = (subject or "").strip()
    if subject.lower() == "none":
        return prompt or ""
    terms = [t for t in (unweighted(s) for s in split_prompt(subject)) if t] if subject else \
        [t for t in [auto_subject(prompt)] if t]
    if not terms or weight == 1.0:
        return prompt or ""
    pats = [re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.I) for t in terms]
    out = []
    for part in split_prompt(prompt):
        core = unweighted(part)
        if any(p.search(core) for p in pats):
            if weight > 0:
                out.append(f"({core}:{round(weight, 2):g})")
            continue
        out.append(part)
    return ", ".join(out)


def _image_files():
    input_dir = folder_paths.get_input_directory()
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    return sorted(folder_paths.filter_files_content_types(files, ["image"]))


# --- Image Recall ---------------------------------------------------------
class ImageRecall:
    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "IMAGE", "STRING", "STRING", "INT", "INT", "FLOAT",
                    comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "STRING", "STRING",
                    "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("model", "clip", "vae", "image", "positive", "negative", "seed", "steps", "cfg",
                    "sampler_name", "scheduler", "info", "outpaint_positive", "do_outpaint", "do_upscale")
    FUNCTION = "load"
    DESCRIPTION = ("Drop an image on this node: it loads the checkpoint and LoRAs the image was made "
                   "with and outputs its prompts and sampler settings. Reads Tagged Image Saver, "
                   "ComfyUI and A1111/Civitai metadata.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_image_files(), {"image_upload": True,
                                           "tooltip": "Drop a PNG on the node (not on the empty canvas, "
                                                      "which would load the image's workflow instead)."}),
                "checkpoint": ([FROM_IMAGE] + folder_paths.get_filename_list("checkpoints"),
                               {"default": FROM_IMAGE,
                                "tooltip": "(from image) uses the one recorded in the image."}),
                "apply_loras": ("BOOLEAN", {"default": True,
                                            "tooltip": "Load the LoRAs recorded in the image."}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                         "tooltip": "Multiplies every recalled LoRA strength."}),
                "prompt_prefix": ("STRING", {"default": "wide shot", "multiline": True,
                                             "tooltip": "Put in front of the recalled prompt, e.g. what "
                                                        "the outpainted area should show."}),
                "prompt_override": ("STRING", {"default": "", "multiline": True,
                                               "tooltip": "Replaces the recalled prompt when not empty."}),
                "negative_override": ("STRING", {"default": "", "multiline": True,
                                                 "tooltip": "Replaces the recalled negative when not empty."}),
                "subject": ("STRING", {"default": "",
                                       "tooltip": "Main object of the picture (e.g. 'cat', or 'cat, kitten'). "
                                                  "Weighted down in outpaint_positive so the new area doesn't "
                                                  "repeat it. Empty = first part of the prompt; 'none' = off."}),
                "subject_weight": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.5, "step": 0.05,
                                             "tooltip": "Subject weight in outpaint_positive. 0 removes it, "
                                                        "1 leaves the prompt as is."}),
                "mode": (MODES, {"default": MODES[1],
                                 "tooltip": "outpaint, outpaint + upscale, or upscale only. Sets do_outpaint "
                                            "/ do_upscale, which pick the path in the subgraph."}),
                "settings": ("STRING", {"default": "", "multiline": True,
                                        "tooltip": "Edits made on the settings panel (JSON). Empty = use "
                                                   "what the image recorded."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, image, **_):
        m = hashlib.sha256()
        with open(folder_paths.get_annotated_filepath(image), "rb") as f:
            m.update(f.read())
        return m.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image):  # no **kwargs: ComfyUI still checks the other inputs itself
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

    def load(self, image, **w):
        path = folder_paths.get_annotated_filepath(image)
        r = rc.recall_file(path)
        e = effective(r, w)
        ckpt, notes = e["ckpt"], e["notes"]
        if not ckpt:
            raise ValueError("Image Recall: " + " ".join(n.lstrip("! ") for n in notes if "heckpoint" in n))
        model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(ckpt)[:3]
        lora_loader = nodes.LoraLoader()
        for name, sm, sc in e["loras"]:
            model, clip = lora_loader.load_lora(model, clip, name, sm, sc)
        img = nodes.LoadImage().load_image(image)[0]
        info = rc.describe(r, notes) + f"\nPrompt used: {e['outpaint'] or '(none)'}"
        log.info("Image Recall %s (%s): %s + %d LoRA(s)", image, e["mode"], ckpt, len(e["loras"]))
        return (model, clip, vae, img, e["positive"], e["negative"], e["seed"], e["steps"], e["cfg"],
                e["sampler"], e["scheduler"], info, e["outpaint"], e["do_outpaint"], e["do_upscale"])


# --- Outpaint Canvas ------------------------------------------------------
BUCKETS = {
    "16:9": (1344, 768), "9:16": (768, 1344), "21:9": (1536, 640), "9:21": (640, 1536),
    "4:3": (1152, 896), "3:4": (896, 1152), "3:2": (1216, 832), "2:3": (832, 1216),
    "1:1": (1024, 1024),
}
ANCHORS = ["center", "left", "right", "top", "bottom"]


def round8(v):
    return max(8, int(round(v / 8.0)) * 8)


def layout(src_w, src_h, aspect, anchor, upscale_by, rw=0, rh=0):
    """Where the source goes on the SDXL canvas and the exact-ratio final size.
    rw/rh: resize the source to this first (0 = fit the canvas; one 0 = keep the ratio)."""
    W, H = BUCKETS[aspect]
    rw, rh = int(rw or 0), int(rh or 0)
    if rw > 0 and rh > 0:
        tw, th = rw, rh
    elif rw > 0:
        tw, th = rw, rw * src_h / src_w
    elif rh > 0:
        tw, th = rh * src_w / src_h, rh
    else:
        tw, th = src_w * 1e9, src_h * 1e9  # no size asked: fit the canvas
    s = min(1.0, W / tw, H / th)  # never larger than the canvas
    # snapped to the 8 px latent grid so the seam falls on latent cell edges
    w, h = min(W, round8(tw * s)), min(H, round8(th * s))
    fx, fy = W - w, H - h
    x = 0 if anchor == "left" else fx if anchor == "right" else fx // 16 * 8
    y = 0 if anchor == "top" else fy if anchor == "bottom" else fy // 16 * 8
    a, b = (int(n) for n in aspect.split(":"))
    if W >= H:
        final_w = round8(W * upscale_by)
        final_h = round8(final_w * b / a)
    else:
        final_h = round8(H * upscale_by)
        final_w = round8(final_h * a / b)
    return {"W": W, "H": H, "x": x, "y": y, "w": w, "h": h, "final_w": final_w, "final_h": final_h}


def outpaint_mask(L, feather, repaint=0.0):
    """1 where new content is generated, 0 over the source; ramps into the source
    along every edge that borders new area. repaint lifts the source's floor from 0
    (kept) toward 1 (fully redrawn), so the new scene can take over the source."""
    W, H, x, y, w, h = L["W"], L["H"], L["x"], L["y"], L["w"], L["h"]
    mask = torch.ones((H, W), dtype=torch.float32)
    inner = torch.zeros((h, w), dtype=torch.float32)
    if feather > 0:
        xs = torch.arange(w, dtype=torch.float32).view(1, w)
        ys = torch.arange(h, dtype=torch.float32).view(h, 1)
        ramp = lambda d: ((feather - d).clamp(min=0) / feather) ** 2
        if x > 0:
            inner = torch.maximum(inner, ramp(xs).expand(h, w))
        if x + w < W:
            inner = torch.maximum(inner, ramp(w - 1 - xs).expand(h, w))
        if y > 0:
            inner = torch.maximum(inner, ramp(ys).expand(h, w))
        if y + h < H:
            inner = torch.maximum(inner, ramp(h - 1 - ys).expand(h, w))
    repaint = min(1.0, max(0.0, float(repaint)))
    if repaint > 0:
        inner = repaint + (1.0 - repaint) * inner
    mask[y:y + h, x:x + w] = inner
    return mask


class OutpaintCanvas:
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("canvas", "mask", "width", "height", "final_width", "final_height")
    FUNCTION = "build"
    DESCRIPTION = ("Puts the image on an SDXL-sized canvas of the chosen aspect ratio (about 1 megapixel) "
                   "and returns the mask of the area to fill, plus the exact-ratio size after upscaling.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "aspect": (list(BUCKETS), {"default": "16:9"}),
            "anchor": (ANCHORS, {"default": "center",
                                 "tooltip": "Where the original sits. left = extend to the right, "
                                            "center = extend both sides."}),
            "feather": ("INT", {"default": 48, "min": 0, "max": 1024, "step": 8,
                                "tooltip": "Pixels of the original blended into the new area."}),
            "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25,
                                     "tooltip": "Final size = canvas x this, trimmed to the exact ratio."}),
        }, "optional": {
            "resize_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                                     "tooltip": "Resize the image to this width before outpainting. 0 = fit the "
                                                "canvas (or keep the ratio if only height is set)."}),
            "resize_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                                      "tooltip": "Resize the image to this height before outpainting. 0 = fit the "
                                                 "canvas (or keep the ratio if only width is set)."}),
            "repaint_original": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "How much of the original itself is redrawn: 0 = kept as is, "
                                                      "0.5 = like img2img at 0.5, 1 = replaced by the new scene."}),
        }}

    def build(self, image, aspect, anchor, feather, upscale_by, resize_width=0, resize_height=0, repaint_original=0.0):
        L = layout(image.shape[2], image.shape[1], aspect, anchor, upscale_by, resize_width, resize_height)
        src = comfy.utils.common_upscale(image.movedim(-1, 1)[:, :3], L["w"], L["h"], "lanczos",
                                         "disabled").movedim(1, -1).clamp(0, 1)
        canvas = torch.full((image.shape[0], L["H"], L["W"], 3), 0.5, dtype=src.dtype, device=src.device)
        canvas[:, L["y"]:L["y"] + L["h"], L["x"]:L["x"] + L["w"], :] = src
        mask = outpaint_mask(L, feather, repaint_original).unsqueeze(0).to(src.device)
        return (canvas, mask, L["W"], L["H"], L["final_w"], L["final_h"])


# --- Upscale Size ---------------------------------------------------------
def ceil64(v):
    return int(math.ceil(v / 64.0)) * 64


def upscale_size(src_w, src_h, aspect, long_side):
    """(render_w, render_h, width, height): the exact-ratio final size with its long side
    at `long_side`, and the render size rounded up to 64 (cropped back after decoding)."""
    if aspect in BUCKETS:
        ra, rb = (int(n) for n in aspect.split(":"))
    else:
        ra, rb = src_w, src_h
    if ra >= rb:
        w, h = long_side, round8(long_side * rb / ra)
    else:
        w, h = round8(long_side * ra / rb), long_side
    return ceil64(w), ceil64(h), w, h


class UpscaleSize:
    CATEGORY = "image"
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("render_width", "render_height", "width", "height")
    FUNCTION = "size"
    DESCRIPTION = ("Final size for the one-pass SDXL upscale: the exact ratio with the long side at "
                   "long_side (16:9 at 1920 = 1920x1080), and the render size rounded up to 64 (1920x1088).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "aspect": ([FROM_IMAGE] + list(BUCKETS), {"default": FROM_IMAGE,
                       "tooltip": "(from image) keeps the image's own ratio."}),
            "long_side": ("INT", {"default": 1920, "min": 512, "max": 8192, "step": 8,
                                  "tooltip": "Long side of the result. The upscale pass renders about "
                                             "(long_side x short side) pixels: 1920x1088 = about 2 normal renders."}),
        }}

    def size(self, image, aspect, long_side):
        return upscale_size(image.shape[2], image.shape[1], aspect, long_side)


# --- queue-time metadata --------------------------------------------------
def _on_prompt(json_data):
    """Record what Image Recall will load on its own entry in the queued graph, so
    Tagged Image Saver (and a later Image Recall of the result) can read it: the
    checkpoint, LoRAs, sampler settings, and each prompt output's text."""
    try:
        prompt = json_data.get("prompt") if isinstance(json_data, dict) else None
        for node in (prompt or {}).values():
            if not isinstance(node, dict) or node.get("class_type") != "ImageRecall":
                continue
            inputs = node.get("inputs") or {}
            image = inputs.get("image")
            if not isinstance(image, str) or not folder_paths.exists_annotated_filepath(image):
                continue
            r = rc.recall_file(folder_paths.get_annotated_filepath(image))
            w = {k: v for k, v in inputs.items() if not isinstance(v, list)}  # literal widget values only
            e = effective(r, w)
            if e["ckpt"]:
                inputs["ckpt_name"] = e["ckpt"]
            for i, (name, sm, sc) in enumerate(e["loras"], 1):
                inputs[f"lora_{i:02d}"] = name
                inputs[f"model_strength_{i:02d}"] = round(sm, 4)
                inputs[f"clip_strength_{i:02d}"] = round(sc, 4)
            inputs.update(seed=e["seed"], steps=e["steps"], cfg=e["cfg"], sampler_name=e["sampler"],
                          scheduler=e["scheduler"])
            node.setdefault("_meta", {})["output_texts"] = {"4": e["positive"], "5": e["negative"],
                                                            "12": e["outpaint"]}
    except Exception as e:  # never block a queue over metadata
        log.warning("Image Recall: could not record metadata: %s", e)
    return json_data


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/image_recall/info")
    async def _image_recall_info(request):
        name = request.query.get("filename", "")
        try:
            path = folder_paths.get_annotated_filepath(name)
            if not os.path.isfile(path):
                return web.json_response({"text": f"File not found: {name}"}, status=404)
            r = rc.recall_file(path)
            _, _, notes = plan(r, request.query.get("checkpoint", FROM_IMAGE),
                               request.query.get("apply_loras", "true") != "false")
            text = rc.describe(r, notes)
            q = request.query
            if r["sources"]:
                try:
                    sw = float(q.get("subject_weight", 0.4))
                except ValueError:
                    sw = 0.4
                text += "\nOutpaint area prompt: " + (outpaint_text(
                    r, q.get("prompt_prefix", ""), q.get("prompt_override", ""), q.get("subject", ""), sw)
                    or "(none)")
            return web.json_response({"text": text, "found": bool(r["sources"])})
        except Exception as e:
            return web.json_response({"text": f"Could not read {name}: {e}"}, status=400)

    @PromptServer.instance.routes.get("/image_recall/recall")
    async def _image_recall_recall(request):
        """Everything the settings panel shows for an image, plus the choices for its selects."""
        q = request.query
        name = q.get("filename", "")
        try:
            path = folder_paths.get_annotated_filepath(name)
            if not os.path.isfile(path):
                return web.json_response({"error": f"File not found: {name}"}, status=404)
            r = rc.recall_file(path)
            try:
                sw = float(q.get("subject_weight", 0.4))
            except ValueError:
                sw = 0.4
            ckpt, _, notes = plan(r)
            seed, steps, cfg, sampler, scheduler = settings(r)
            loras = []
            for l in r.get("loras") or []:
                found = find_model("loras", l["name"], _top_dir(ckpt))
                loras.append({"name": found or l["name"], "found": bool(found), "on": bool(found),
                              "model": l["model"], "clip": l["model"] if l.get("clip") is None else l["clip"]})
            return web.json_response({
                "found": bool(r["sources"]), "sources": r["sources"], "notes": notes, "size": r.get("size"),
                "checkpoint": ckpt or "", "checkpoint_recorded": r.get("checkpoint") or "",
                "loras": loras, "positive": r.get("positive") or "", "negative": r.get("negative") or "",
                "lora_tags": r.get("lora_tags"),
                "outpaint_positive": outpaint_text(r, q.get("prompt_prefix", ""), "", q.get("subject", ""), sw),
                "seed": seed, "steps": steps, "cfg": cfg, "sampler": sampler, "scheduler": scheduler,
                "choices": {"checkpoints": folder_paths.get_filename_list("checkpoints"),
                            "loras": folder_paths.get_filename_list("loras"),
                            "samplers": list(comfy.samplers.KSampler.SAMPLERS),
                            "schedulers": list(comfy.samplers.KSampler.SCHEDULERS)},
            })
        except Exception as e:
            return web.json_response({"error": f"Could not read {name}: {e}"}, status=400)

    PromptServer.instance.add_on_prompt_handler(_on_prompt)
except Exception as e:
    log.warning("Image Recall: could not register route/handler: %s", e)


NODE_CLASS_MAPPINGS = {"ImageRecall": ImageRecall, "OutpaintCanvas": OutpaintCanvas,
                       "UpscaleSize": UpscaleSize}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageRecall": "Image Recall", "OutpaintCanvas": "Outpaint Canvas",
                              "UpscaleSize": "Upscale Size"}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
