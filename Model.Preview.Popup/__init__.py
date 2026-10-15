"""Model Preview Popup - ComfyUI extension (no nodes).

Hovering a model in the Model Library sidebar shows a large popup with the
preview images saved beside the model file:

    <name>.preview.<ext>      the main preview (what ComfyUI shows as the icon)
    <name>.<ext>              also accepted as the main preview
    <name>.preview-2.<ext>    extras, as civitai-likes -previews N saves them
    ...

Click an extra in the popup to view it large. The popup also shows the
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


# ------------------------------------------------------------- base models
# Which base model each model file is for, worked out in this order:
#   1. the civitai-likes state file (<models>/.civitai-likes.json)
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
    """{absolute path: (tag, label)} from the civitai-likes state file."""
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
        return web.json_response({"found": True, "images": preview_files(path),
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
