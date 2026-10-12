"""LoRA Tag Picker - ComfyUI custom node.

Sits after your LoRA loaders. It finds every LoRA loaded upstream, reads the
training tags embedded in each file, shows them as clickable chips, and
appends the active ones to your prompt when you queue.

Install: copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
No dependencies beyond ComfyUI itself.

How tags are ranked: by coverage, the share of training images whose caption
contained the tag (count / image count for that dataset folder). When a file
records no image count, the most used tag's count stands in for it.

    trigger  coverage >= 90%, or the tag matches the dataset folder's token
             (kohya "<repeats>_<token> <class>") or a declared trigger phrase.
             Active by default.
    common   coverage >= 25%. Shown, off by default.
    rare     everything else. Hidden behind "+N more".
"""

import json
import logging
import os
import re
import struct

import folder_paths

WEB_DIRECTORY = "./web"

TRIGGER_COVERAGE = 0.90
COMMON_COVERAGE = 0.25
MAX_TAGS = 80  # per LoRA, sent to the UI
MAX_HEADER_BYTES = 100 * 1024 * 1024  # sanity limit; real headers are far smaller
SELECTION_TYPE = "LORA_TAG_SELECTION"

_cache = {}  # path -> (mtime, size, info)


# ---------------------------------------------------------------- metadata

def read_safetensors_metadata(path):
    """Return the __metadata__ dict from a .safetensors file (empty if none)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("file too short to be safetensors")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or header_len > MAX_HEADER_BYTES:
            raise ValueError("implausible header length; not a safetensors file?")
        header = json.loads(f.read(header_len))
    meta = header.get("__metadata__") if isinstance(header, dict) else None
    return meta if isinstance(meta, dict) else {}


def _as_dict(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _folder_tokens(folder):
    """kohya folders are "<repeats>_<token> <class>"; return the trigger candidates."""
    m = re.match(r"^\d+_(.+)$", str(folder).strip())
    if not m:
        return []
    rest = m.group(1).strip()
    if not rest:
        return []
    first = rest.split()[0]
    return [first] if first == rest else [first, rest]


def analyze(meta):
    """Rank a LoRA's training tags into trigger / common / rare tiers."""
    freq = _as_dict(meta.get("ss_tag_frequency"))
    dirs = _as_dict(meta.get("ss_dataset_dirs"))

    folders = []  # (denominator, {tag: count}, has_image_count)
    for folder, tags in freq.items():
        if not isinstance(tags, dict):
            continue
        counts = {}
        for tag, count in tags.items():
            tag = str(tag).strip()
            try:
                count = int(count)
            except (TypeError, ValueError):
                continue
            if tag and count > 0:
                counts[tag] = counts.get(tag, 0) + count
        if not counts:
            continue
        dir_info = dirs.get(folder)
        try:
            images = int(dir_info.get("img_count")) if isinstance(dir_info, dict) else 0
        except (TypeError, ValueError):
            images = 0
        images = max(images, 0)
        folders.append((max(images, max(counts.values())), counts, images > 0))

    # A stray one-image folder next to the real dataset would make every one
    # of its tags look like a 100% trigger; ignore folders that small.
    if len(folders) > 1:
        biggest = max(f[0] for f in folders)
        folders = [f for f in folders if f[0] >= 0.1 * biggest]

    forced = set()
    for folder in list(freq.keys()) + list(dirs.keys()):
        forced.update(t.lower() for t in _folder_tokens(folder))
    declared = []
    phrase = meta.get("modelspec.trigger_phrase")
    if isinstance(phrase, str):
        declared = [p.strip() for p in phrase.split(",") if p.strip()]
        forced.update(p.lower() for p in declared)

    stats = {}  # tag -> [coverage, count, first-seen order]
    for denom, counts, _ in folders:
        for tag, count in counts.items():
            entry = stats.setdefault(tag, [0.0, 0, len(stats)])
            entry[0] = max(entry[0], min(1.0, count / denom))
            entry[1] += count

    tags = []
    for tag, (coverage, count, order) in stats.items():
        if coverage >= TRIGGER_COVERAGE or tag.lower() in forced:
            tier = "trigger"
        elif coverage >= COMMON_COVERAGE:
            tier = "common"
        else:
            tier = "rare"
        tags.append({"tag": tag, "count": count, "coverage": round(coverage, 4),
                     "tier": tier, "_order": order, "_named": tag.lower() in forced})

    present = {t["tag"].lower() for t in tags}
    for p in declared:  # the author's own declared trigger, even if never tagged
        if p.lower() not in present:
            present.add(p.lower())
            tags.append({"tag": p, "count": 0, "coverage": None,
                         "tier": "trigger", "_order": -1})
    if not tags:  # no caption data at all: offer the folder token as a guess
        for folder in dirs.keys():
            for tok in _folder_tokens(folder)[:1]:
                if tok.lower() not in present:
                    present.add(tok.lower())
                    tags.append({"tag": tok, "count": 0, "coverage": None,
                                 "tier": "common", "_order": len(tags)})

    rank = {"trigger": 0, "common": 1, "rare": 2}
    # Within a tier: tags named by the folder or the author first, then by coverage.
    tags.sort(key=lambda t: (rank[t["tier"]], not t.get("_named", True),
                             -(t["coverage"] or 0), t["_order"]))
    total = len(tags)
    tags = tags[:MAX_TAGS]
    for t in tags:
        t.pop("_order", None)
        t.pop("_named", None)

    images = sum(f[0] for f in folders if f[2]) or None
    note = ""
    if not meta:
        note = "No metadata in this file (stripped by the author)."
    elif not tags:
        note = "Metadata present, but no training tags were recorded."
    return {
        "tags": tags,
        "total_tags": total,
        "images": images,
        # False when coverage is measured against the top tag, not an image count
        "exact": bool(folders) and all(f[2] for f in folders),
        "base": meta.get("ss_base_model_version") or meta.get("modelspec.architecture") or "",
        "note": note,
    }


def get_lora_info(name):
    """Tag info for a LoRA by its name in the loras folder. Never raises."""
    info = {"name": name, "found": False, "tags": [], "total_tags": 0,
            "images": None, "exact": False, "base": "", "note": ""}
    path = folder_paths.get_full_path("loras", name) if isinstance(name, str) and name else None
    if not path or name not in folder_paths.get_filename_list("loras"):
        info["note"] = "Not found in the loras folder."
        return info
    info["found"] = True
    if not path.lower().endswith(".safetensors"):
        info["note"] = "Not a .safetensors file, so there is no metadata to read."
        return info
    try:
        st = os.stat(path)
        cached = _cache.get(path)
        if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
            return cached[2]
        info.update(analyze(read_safetensors_metadata(path)))
        _cache[path] = (st.st_mtime, st.st_size, info)
    except (OSError, ValueError) as e:
        info["note"] = f"Could not read the file header ({e})."
    return info


def default_active(tag_entry):
    return tag_entry.get("tier") == "trigger"


# ------------------------------------------------------------ graph walk

def find_upstream_loras(prompt, unique_id):
    """LoRA files loaded upstream of a node, in load order (furthest upstream first).

    Only used when the workflow is queued without the web UI (API use); in the
    UI the frontend resolves the list itself and sends it along.
    """
    if not isinstance(prompt, dict) or unique_id is None:
        return []
    known = set(folder_paths.get_filename_list("loras"))
    start = str(unique_id)

    def inputs_of(nid):
        node = prompt.get(nid)
        inputs = node.get("inputs") if isinstance(node, dict) else None
        return inputs if isinstance(inputs, dict) else {}

    depth_of = {}  # upstream node id -> its longest distance from the start node
    frontier, depth = [start], 0
    while frontier and depth <= len(prompt):
        depth += 1
        nxt = []
        for nid in frontier:
            for value in inputs_of(nid).values():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in prompt:
                    src = str(value[0])
                    if src != start and src not in nxt:
                        nxt.append(src)
        for src in nxt:
            depth_of[src] = depth
        frontier = nxt

    found = []
    for nid, node_depth in depth_of.items():
        for key, value in inputs_of(nid).items():
            if "lora" not in str(key).lower():
                continue
            if isinstance(value, dict):
                value = value.get("lora") if value.get("on", True) else None
            if isinstance(value, str) and value in known:
                found.append((node_depth, len(found), value))
    ordered = []
    for _, _, name in sorted(found, key=lambda f: (-f[0], f[1])):
        if name not in ordered:
            ordered.append(name)
    return ordered


# ------------------------------------------------------------------ node

def _split_tokens(text):
    return [t.strip() for t in re.split(r"[,\n]", text or "") if t.strip()]


class LoraTagPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "",
                                    "tooltip": "Your prompt. Active LoRA tags are added to it."}),
                "tag_position": (["after prompt", "before prompt"],
                                 {"tooltip": "Where the active tags go relative to your prompt."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Passed through. LoRAs loaded upstream of it are detected."}),
                "clip": ("CLIP", {"tooltip": "Passed through. LoRAs loaded upstream of it are detected."}),
                "selection": (SELECTION_TYPE, {"socketless": True}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "text", "tags")
    OUTPUT_TOOLTIPS = (
        "The model input, unchanged.",
        "The clip input, unchanged.",
        "Your prompt with the active LoRA tags added. Wire this to a text encoder.",
        "Only the active tags, comma-separated.",
    )
    FUNCTION = "run"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Finds the LoRAs loaded upstream, lists their training tags as clickable "
        "chips, and adds the active ones to your prompt."
    )

    def run(self, text, tag_position, model=None, clip=None, selection=None,
            prompt=None, unique_id=None):
        state = _as_dict(selection)
        resolved = state.get("resolved")

        groups = []  # one list of active tags per LoRA, in chain order
        if isinstance(resolved, list):
            for group in resolved:
                tags = group.get("tags") if isinstance(group, dict) else None
                if isinstance(tags, list):
                    groups.append([str(t) for t in tags if isinstance(t, str)])
        else:
            overrides = _as_dict(state.get("overrides"))
            for name in find_upstream_loras(prompt, unique_id):
                chosen = _as_dict(overrides.get(name))
                groups.append([t["tag"] for t in get_lora_info(name)["tags"]
                               if chosen.get(t["tag"], default_active(t))])

        seen = {t.lower() for t in _split_tokens(text)}  # don't repeat the prompt
        tags = []
        for group in groups:
            for tag in group:
                tag = tag.strip()
                if tag and tag.lower() not in seen:
                    seen.add(tag.lower())
                    tags.append(tag)

        tag_text = ", ".join(tags)
        body = (text or "").strip().rstrip(",").rstrip()
        if not tag_text:
            out = text or ""
        elif not body:
            out = tag_text
        elif tag_position == "before prompt":
            out = f"{tag_text}, {body}"
        else:
            out = f"{body}, {tag_text}"
        return (model, clip, out, tag_text)


# ----------------------------------------------------------------- route

try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/lora_tag_picker/tags")
    async def _lora_tag_picker_tags(request):
        return web.json_response(get_lora_info(request.query.get("name", "")))
except Exception as e:  # keep the node usable even if the route can't register
    logging.warning("[LoRA Tag Picker] could not register API route: %s", e)


NODE_CLASS_MAPPINGS = {"LoraTagPicker": LoraTagPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraTagPicker": "LoRA Tag Picker"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
