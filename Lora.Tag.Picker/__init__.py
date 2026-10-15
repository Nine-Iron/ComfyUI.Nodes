"""LoRA Tag Picker - ComfyUI custom node.

Sits after your LoRA loaders. It finds every LoRA loaded upstream, reads the
training tags embedded in each file, shows them as clickable chips, and
appends the active ones to your prompt when you queue.

Install: copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
No dependencies beyond ComfyUI itself.

How tags are ranked: by coverage, the share of training images whose caption
contained the tag (count / image count for that dataset folder). When a file
records no image count, the most used tag's count stands in for it.

    trigger  the tag matching the dataset folder's token (kohya
             "<repeats>_<token> <class>") or a declared trigger phrase; when
             the file names neither, every tag with coverage >= 90%.
             Active by default.
    common   coverage >= 25%. Shown, off by default.
    rare     everything else. Hidden behind "+N more".

Sentence captions: when a LoRA was trained on prose, its recorded "tags" are
just comma-split pieces of sentences, dozens of them and none useful. Those
are condensed instead: the phrase the captions open with becomes the trigger,
and the pieces are boiled down to their most common words.

max_tags limits how many tags one LoRA shows up front; the rest stay behind
"+N more". However many triggers a file has, at most three are on by default.
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
MAX_TAGS = 40          # per LoRA, sent to the UI
DEFAULT_MAX_TAGS = 8   # per LoRA, shown up front
AUTO_ON = 3            # per LoRA, most tags that are on without being clicked
CAPTION_WORDS = 4      # a "tag" this long is a piece of a sentence
CAPTION_SHARE = 0.25   # share of such pieces that marks a sentence-captioned dataset
LEAD_WORDS = 4         # longest opening phrase considered as a trigger
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


_WORD = re.compile(r"[^\W_](?:[\w'\-]*[^\W_])?")
_SENTENCE_END = re.compile(r"[.!?:;](?:\s|$)")
# Function words, plus the filler caption models write around the actual content.
_STOPWORDS = frozenset("""
a an the and or but nor so yet of in on at to for from by with without into onto over under above below
between among through during before after about against around near behind beside beyond along across off
out up down is are was were be been being am it its it's this that these those there here their his her hers
him he she they them we us our you your my me as if then than while when where which who whom whose what why
how not no all any both each few more most other some such only own same too very can will just has have had
having do does did doing also there's that's
image picture depicts depicting features featuring shows showing appears giving creating adding suggesting
overall set left right side center top bottom one two
""".split())


def _words(text):
    return _WORD.findall(str(text).lower())


def _norm(tag):
    return " ".join(str(tag).lower().replace("_", " ").split())


def _is_caption_style(counts):
    """True when the "tags" are mostly pieces of sentences rather than tags."""
    total = sum(counts.values())
    prose = sum(c for tag, c in counts.items() if len(_words(tag)) >= CAPTION_WORDS)
    return total > 0 and prose / total >= CAPTION_SHARE


def _leading_words(piece):
    """The words a caption piece opens with, up to its first function word."""
    out = []
    for w in _words(_SENTENCE_END.split(piece, maxsplit=1)[0]):
        if w in _STOPWORDS:
            break
        if out and out[-1] == w:  # "mytrigger mytrigger. a drawing of..."
            continue
        out.append(w)
        if len(out) == LEAD_WORDS:
            break
    return out


def _tag_entries(counts, denom):
    """Tag-captioned folder: one entry per tag, spelling variants merged."""
    merged = {}
    for tag, count in counts.items():
        e = merged.setdefault(_norm(tag), {"tag": tag, "best": 0, "count": 0})
        if count > e["best"]:  # show the most used spelling
            e["tag"], e["best"] = tag, count
        e["count"] += count
    out = {}
    for key, e in merged.items():
        coverage = min(1.0, e["count"] / denom)
        out[key] = {"tag": e["tag"], "count": e["count"], "coverage": coverage,
                    "trigger": coverage >= TRIGGER_COVERAGE, "kind": "tag"}
    return out


def _caption_entries(counts, denom, exact):
    """Sentence-captioned folder: condense the pieces of prose.

    The trigger is the phrase that at least 90% of captions open a piece with;
    everything else is reduced to how many captions mention each word.
    """
    word_n, lead_n = {}, {}
    for piece, count in counts.items():
        for w in dict.fromkeys(_words(piece)):
            if len(w) >= 3 and w not in _STOPWORDS and not w.isdigit():
                word_n[w] = word_n.get(w, 0) + count
        lead = _leading_words(piece)
        for i in range(1, len(lead) + 1):
            lead_n[tuple(lead[:i])] = lead_n.get(tuple(lead[:i]), 0) + count
    if not exact:
        # No image count recorded, so estimate the number of captions. A word can
        # sit in several pieces of one caption and overshoot; an opener cannot, so
        # trust the commonest opener when it is in the same range.
        top_word = max(word_n.values(), default=0)
        top_lead = max(lead_n.values(), default=0)
        denom = max(denom, top_lead if top_lead >= 0.6 * top_word else top_word)

    # Plenty of openers can be common ("wearing ...", "natural light"). The trigger
    # is the one the very first caption starts with: trainers record pieces in
    # caption order, so that is the first piece in the file.
    first = tuple(_leading_words(next(iter(counts))))
    trigger = None
    for size in range(len(first), 0, -1):
        if lead_n.get(first[:size], 0) / denom >= TRIGGER_COVERAGE:
            trigger = first[:size]
            break

    out = {}
    if trigger:
        text = " ".join(trigger)
        out[text] = {"tag": text, "count": lead_n[trigger], "trigger": True, "kind": "phrase",
                     "coverage": min(1.0, lead_n[trigger] / denom)}
    for w, n in word_n.items():
        if trigger and w in trigger:
            continue
        out[w] = {"tag": w, "count": n, "coverage": min(1.0, n / denom),
                  "trigger": False, "kind": "word"}
    return out


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
        forced.update(_norm(t) for t in _folder_tokens(folder))
    declared = []
    phrase = meta.get("modelspec.trigger_phrase")
    if isinstance(phrase, str):
        declared = [p.strip() for p in phrase.split(",") if p.strip()]
        forced.update(_norm(p) for p in declared)

    stats = {}  # normalised tag -> entry
    captions = False
    for denom, counts, exact in folders:
        prose = _is_caption_style(counts)
        captions = captions or prose
        entries = _caption_entries(counts, denom, exact) if prose else _tag_entries(counts, denom)
        for key, e in entries.items():
            have = stats.get(_norm(key))
            if have is None:
                stats[_norm(key)] = dict(e, order=len(stats))
            else:
                have["coverage"] = max(have["coverage"], e["coverage"])
                have["count"] += e["count"]
                have["trigger"] = have["trigger"] or e["trigger"]

    # When the file names its own trigger (dataset folder or declared phrase), that
    # is the trigger; other tags that happen to be in every caption are not.
    named = bool(declared) or any(key in forced for key in stats)
    tags = []
    for key, e in stats.items():
        if key in forced or (e["trigger"] and not named):
            tier = "trigger"
        elif e["coverage"] >= COMMON_COVERAGE:
            tier = "common"
        else:
            tier = "rare"
        tags.append({"tag": e["tag"], "count": e["count"], "coverage": round(e["coverage"], 4),
                     "tier": tier, "kind": e["kind"], "_order": e["order"], "_named": key in forced})

    present = {_norm(t["tag"]) for t in tags}
    for p in declared:  # the author's own declared trigger, even if never tagged
        if _norm(p) not in present:
            present.add(_norm(p))
            tags.append({"tag": p, "count": 0, "coverage": None, "kind": "declared",
                         "tier": "trigger", "_order": -1})
    if not tags:  # no caption data at all: offer the folder token as a guess
        for folder in dirs.keys():
            for tok in _folder_tokens(folder)[:1]:
                if _norm(tok) not in present:
                    present.add(_norm(tok))
                    tags.append({"tag": tok, "count": 0, "coverage": None, "kind": "declared",
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
    elif captions:
        note = "Trained on sentence captions, so these are its most common words, not tags."
    return {
        "tags": tags,
        "total_tags": total,
        "images": images,
        # False when coverage is measured against the top tag, not an image count
        "exact": bool(folders) and all(f[2] for f in folders),
        "captions": captions,
        "base": meta.get("ss_base_model_version") or meta.get("modelspec.architecture") or "",
        "note": note,
    }


def get_lora_info(name):
    """Tag info for a LoRA by its name in the loras folder. Never raises."""
    info = {"name": name, "found": False, "tags": [], "total_tags": 0,
            "images": None, "exact": False, "captions": False, "base": "", "note": ""}
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


def front_tags(tags, max_tags):
    """The best tags of a LoRA, in order: the ones shown up front."""
    try:
        limit = max(1, int(max_tags))
    except (TypeError, ValueError):
        limit = DEFAULT_MAX_TAGS
    return [t["tag"] for t in tags if t.get("tier") != "rare"][:limit]


def auto_tags(tags, max_tags):
    """The tags that are on without being clicked: the first few triggers up front."""
    front = set(front_tags(tags, max_tags))
    return [t["tag"] for t in tags if t.get("tier") == "trigger" and t["tag"] in front][:AUTO_ON]


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
                # Listed last so workflows saved before it existed still load.
                "max_tags": ("INT", {"default": DEFAULT_MAX_TAGS, "min": 1, "max": MAX_TAGS,
                                     "tooltip": "Most tags one LoRA shows up front. The rest stay "
                                                "behind '+N more'."}),
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
            max_tags=DEFAULT_MAX_TAGS, prompt=None, unique_id=None):
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
                tags = get_lora_info(name)["tags"]
                auto = auto_tags(tags, max_tags)
                groups.append([t["tag"] for t in tags
                               if chosen.get(t["tag"], t["tag"] in auto)])

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

    @PromptServer.instance.routes.get("/Lora.Tag.Picker/tags")
    async def _lora_tag_picker_tags(request):
        return web.json_response(get_lora_info(request.query.get("name", "")))
except Exception as e:  # keep the node usable even if the route can't register
    logging.warning("[LoRA Tag Picker] could not register API route: %s", e)


NODE_CLASS_MAPPINGS = {"LoraTagPicker": LoraTagPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraTagPicker": "LoRA Tag Picker"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
