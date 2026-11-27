"""Image Recall for ComfyUI.

Image Recall: drop an image on the node and it loads the checkpoint and LoRAs the
image was made with, and outputs its prompts and sampler settings.
Outpaint Canvas: places the image on an SDXL-sized canvas of a chosen aspect
ratio and returns the mask of the area to fill.
Tiled Refine: a low-denoise img2img pass over a large image in overlapping tiles,
so SDXL can add detail to an upscale without running out of memory.

The example workflow (example_workflows/) chains them into outpaint -> upscale.
"""
import hashlib
import logging
import math
import os
import re

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import nodes

from . import recall as rc
from .graphinfo import model_stem

log = logging.getLogger("ImageRecall")

FROM_IMAGE = "(from image)"
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
                    comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "image", "positive", "negative", "seed", "steps", "cfg",
                    "sampler_name", "scheduler", "info", "outpaint_positive")
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

    def load(self, image, checkpoint, apply_loras, lora_scale, prompt_prefix, prompt_override,
             negative_override, subject="", subject_weight=0.4):
        path = folder_paths.get_annotated_filepath(image)
        r = rc.recall_file(path)
        ckpt, loras, notes = plan(r, checkpoint, apply_loras, lora_scale)
        if not ckpt:
            raise ValueError("Image Recall: " + " ".join(n.lstrip("! ") for n in notes if "heckpoint" in n))
        model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(ckpt)[:3]
        lora_loader = nodes.LoraLoader()
        for name, sm, sc in loras:
            model, clip = lora_loader.load_lora(model, clip, name, sm, sc)
        img = nodes.LoadImage().load_image(image)[0]
        positive, negative = texts(r, prompt_prefix, prompt_override, negative_override)
        seed, steps, cfg, sampler, scheduler = settings(r)
        outpaint = outpaint_text(r, prompt_prefix, prompt_override, subject, subject_weight)
        info = rc.describe(r, notes) + f"\nOutpaint area prompt: {outpaint or '(none)'}"
        log.info("Image Recall %s: %s + %d LoRA(s)", image, ckpt, len(loras))
        return (model, clip, vae, img, positive, negative, seed, steps, cfg, sampler, scheduler, info, outpaint)


# --- Outpaint Canvas ------------------------------------------------------
BUCKETS = {
    "16:9": (1344, 768), "9:16": (768, 1344), "21:9": (1536, 640), "9:21": (640, 1536),
    "4:3": (1152, 896), "3:4": (896, 1152), "3:2": (1216, 832), "2:3": (832, 1216),
    "1:1": (1024, 1024),
}
ANCHORS = ["center", "left", "right", "top", "bottom"]


def round8(v):
    return max(8, int(round(v / 8.0)) * 8)


def layout(src_w, src_h, aspect, anchor, upscale_by):
    """Where the source goes on the SDXL canvas and the exact-ratio final size."""
    W, H = BUCKETS[aspect]
    s = min(W / src_w, H / src_h)
    # snapped to the 8 px latent grid so the seam falls on latent cell edges
    w, h = min(W, round8(src_w * s)), min(H, round8(src_h * s))
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


def outpaint_mask(L, feather):
    """1 where new content is generated, 0 over the source; ramps into the source
    along every edge that borders new area."""
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
            "feather": ("INT", {"default": 48, "min": 0, "max": 256, "step": 8,
                                "tooltip": "Pixels of the original blended into the new area."}),
            "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25,
                                     "tooltip": "Final size = canvas x this, trimmed to the exact ratio."}),
        }}

    def build(self, image, aspect, anchor, feather, upscale_by):
        L = layout(image.shape[2], image.shape[1], aspect, anchor, upscale_by)
        src = comfy.utils.common_upscale(image.movedim(-1, 1)[:, :3], L["w"], L["h"], "lanczos",
                                         "disabled").movedim(1, -1).clamp(0, 1)
        canvas = torch.full((image.shape[0], L["H"], L["W"], 3), 0.5, dtype=src.dtype, device=src.device)
        canvas[:, L["y"]:L["y"] + L["h"], L["x"]:L["x"] + L["w"], :] = src
        mask = outpaint_mask(L, feather).unsqueeze(0).to(src.device)
        return (canvas, mask, L["W"], L["H"], L["final_w"], L["final_h"])


# --- Tiled Refine ---------------------------------------------------------
def tile_spans(length, tile, overlap):
    """[(start, size)] covering 0..length with tiles of `tile` px (multiples of 8)."""
    if length <= tile:
        size = max(8, length // 8 * 8)
        return [(0, size)] if size == length else [(0, size), (length - size, size)]
    n = math.ceil((length - overlap) / (tile - overlap))
    n = max(n, 2)
    return [(int(round(i * (length - tile) / (n - 1))), tile) for i in range(n)]


def _ramp(size, start, length, overlap):
    w = torch.ones(size, dtype=torch.float32)
    if overlap <= 0:
        return w
    ramp = (torch.arange(size, dtype=torch.float32) + 0.5) / overlap
    if start > 0:
        w = torch.minimum(w, ramp.clamp(max=1))
    if start + size < length:
        w = torch.minimum(w, ramp.flip(0).clamp(max=1))
    return w.clamp(min=1e-3)


class TiledRefine:
    CATEGORY = "sampling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "refine"
    DESCRIPTION = ("Low-denoise img2img over a large image in overlapping tiles, blended together. "
                   "Lets SDXL add detail to an upscale on a small GPU.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "image": ("IMAGE",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "steps": ("INT", {"default": 10, "min": 1, "max": 200,
                              "tooltip": "Steps per tile. Every step runs even at low denoise, so the "
                                         "total cost is tiles x steps (6 tiles at 2688x1512)."}),
            "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m",
                             "tooltip": "Prefer a 1-call-per-step sampler; *_2s_* and sde samplers cost double."}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
            "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "0.2 keeps it closer to the upscale, 0.4 adds more detail."}),
            "tile_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
            "overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 8}),
        }}

    def refine(self, model, positive, negative, vae, image, seed, steps, cfg, sampler_name, scheduler,
               denoise, tile_size, overlap):
        overlap = min(overlap, tile_size // 2)
        B, H, W, _ = image.shape
        xs, ys = tile_spans(W, tile_size, overlap), tile_spans(H, tile_size, overlap)
        acc = torch.zeros((B, H, W, 3), dtype=torch.float32)
        wsum = torch.zeros((1, H, W, 1), dtype=torch.float32)
        n_tiles = len(xs) * len(ys)
        total = n_tiles * steps
        log.info("Tiled Refine: %d tiles (%dx%d) x %d steps = %d sampler steps", n_tiles, len(xs), len(ys),
                 steps, total)
        pbar = comfy.utils.ProgressBar(total)  # one bar for the whole node, not one per tile
        index = 0
        for y, th in ys:
            for x, tw in xs:
                comfy.model_management.throw_exception_if_processing_interrupted()
                tile = image[:, y:y + th, x:x + tw, :3]
                log.info("Tiled Refine: tile %d/%d", index + 1, n_tiles)
                latent = comfy.sample.fix_empty_latent_channels(model, vae.encode(tile))
                noise = comfy.sample.prepare_noise(latent, seed + index)
                done = index * steps
                cb = lambda step, x0, x, total_steps, done=done: pbar.update_absolute(done + step + 1, total)
                samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                                              latent, denoise=denoise, callback=cb, disable_pbar=True,
                                              seed=seed + index)
                decoded = vae.decode(samples)
                if decoded.dim() == 5:
                    decoded = decoded.reshape(-1, *decoded.shape[-3:])
                decoded = decoded[:, :th, :tw, :3].float().cpu()
                weight = (_ramp(th, y, H, overlap).view(th, 1) * _ramp(tw, x, W, overlap).view(1, tw))
                weight = weight.view(1, th, tw, 1)
                acc[:, y:y + th, x:x + tw] += decoded * weight
                wsum[:, y:y + th, x:x + tw] += weight
                index += 1
        return ((acc / wsum).clamp(0, 1).to(image.dtype),)


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
            literal = lambda k, d: inputs.get(k, d) if not isinstance(inputs.get(k), list) else d
            ckpt, loras, _ = plan(r, literal("checkpoint", FROM_IMAGE), bool(literal("apply_loras", True)),
                                  float(literal("lora_scale", 1.0)))
            if ckpt:
                inputs["ckpt_name"] = ckpt
            for i, (name, sm, sc) in enumerate(loras, 1):
                inputs[f"lora_{i:02d}"] = name
                inputs[f"model_strength_{i:02d}"] = round(sm, 4)
                inputs[f"clip_strength_{i:02d}"] = round(sc, 4)
            seed, steps, cfg, sampler, scheduler = settings(r)
            inputs.update(seed=seed, steps=steps, cfg=cfg, sampler_name=sampler, scheduler=scheduler)
            strings = [literal(k, None) for k in ("prompt_prefix", "prompt_override", "negative_override")]
            if all(isinstance(s, str) for s in strings):
                pos, neg = texts(r, *strings)
                meta = node.setdefault("_meta", {})
                meta["output_texts"] = {"4": pos, "5": neg}
                subj, sw = literal("subject", ""), literal("subject_weight", 0.4)
                if isinstance(subj, str) and isinstance(sw, (int, float)):
                    meta["output_texts"]["12"] = outpaint_text(r, strings[0], strings[1], subj, float(sw))
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

    PromptServer.instance.add_on_prompt_handler(_on_prompt)
except Exception as e:
    log.warning("Image Recall: could not register route/handler: %s", e)


NODE_CLASS_MAPPINGS = {"ImageRecall": ImageRecall, "OutpaintCanvas": OutpaintCanvas,
                       "TiledRefine": TiledRefine}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageRecall": "Image Recall", "OutpaintCanvas": "Outpaint Canvas",
                              "TiledRefine": "Tiled Refine"}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
