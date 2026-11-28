"""Civitai-compatible model hashes, cached, plus a CLI that adds them to old PNGs.

Civitai links an image to a checkpoint/LoRA page only by hash, in the A1111
'parameters' text:
    Model hash: <AutoV2>                     sha256 of the whole file, first 10 hex
    Lora hashes: "name: <AutoV3>, ..."       sha256 of the file after the safetensors
                                             header, first 12 hex (A1111's "addnet" hash)

No ComfyUI imports here, so the CLI runs with plain Python:
    python hashes.py --models /path/to/ComfyUI/models image.png [more.png ...]
"""
import hashlib
import json
import os
import struct
import threading

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hash_cache.json")
_lock = threading.Lock()
_cache = None


def _sha256(path, skip_safetensors_header=False):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if skip_safetensors_header:
            n = struct.unpack("<Q", fh.read(8))[0]
            fh.seek(8 + n)
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load():
    global _cache
    if _cache is None:
        try:
            with open(CACHE_PATH, encoding="utf-8") as fh:
                _cache = json.load(fh)
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _cached(path, kind):
    """kind 'v2' (whole file) or 'v3' (after safetensors header); full hex digest."""
    path = os.path.abspath(path)
    st = os.stat(path)
    with _lock:
        cache = _load()
        entry = cache.get(path)
        # Cache key policy: size + mtime. Cheap, and catches a re-downloaded or
        # replaced file. A same-size in-place edit that keeps mtime would slip by,
        # which never happens to downloaded model files in practice.
        if not entry or entry.get("size") != st.st_size or entry.get("mtime") != st.st_mtime:
            entry = {"size": st.st_size, "mtime": st.st_mtime}
            cache[path] = entry
        if kind not in entry:
            entry[kind] = _sha256(path, skip_safetensors_header=(kind == "v3"))
            tmp = CACHE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, indent=1)
            os.replace(tmp, CACHE_PATH)
        return entry[kind]


def model_hash(path):
    """AutoV2, what A1111 writes as 'Model hash'."""
    return _cached(path, "v2")[:10]


def lora_hash(path):
    """AutoV3 for .safetensors (A1111 'Lora hashes'), AutoV2-length-12 otherwise."""
    if path.lower().endswith(".safetensors"):
        return _cached(path, "v3")[:12]
    return _cached(path, "v2")[:12]


def hash_fields(model_path, lora_paths):
    """{'Model hash': ..., 'Lora hashes': ...} for whatever paths resolve.
    lora_paths: list of (stem, path-or-None). Missing/unreadable files are skipped."""
    out = {}
    if model_path:
        try:
            out["Model hash"] = model_hash(model_path)
        except OSError:
            pass
    parts = []
    for stem, path in lora_paths:
        if not path:
            continue
        try:
            parts.append(f"{stem}: {lora_hash(path)}")
        except (OSError, struct.error):
            pass
    if parts:
        out["Lora hashes"] = '"' + ", ".join(parts) + '"'
    return out


# --- CLI: add hashes to PNGs saved before this existed -------------------
def _index_models(root):
    """stem -> path, for every model file under root."""
    found = {}
    for dirpath, _, files in os.walk(root, followlinks=True):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in (".safetensors", ".ckpt", ".pt", ".pth", ".bin"):
                found.setdefault(stem, os.path.join(dirpath, f))
    return found


def _chunks(data):
    i = 8
    while i < len(data):
        n = struct.unpack(">I", data[i:i + 4])[0]
        yield i, data[i + 4:i + 8], data[i + 8:i + 8 + n], 12 + n
        i += 12 + n


def _text_chunk(key, value):
    """tEXt when the value is Latin-1, iTXt (UTF-8) otherwise, like the saver."""
    import zlib
    try:
        ctype, body = b"tEXt", key.encode("latin-1") + b"\0" + value.encode("latin-1")
    except UnicodeEncodeError:
        ctype, body = b"iTXt", key.encode("latin-1") + b"\0\0\0\0\0" + value.encode("utf-8")
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body))


def _read_text(ctype, body):
    """(key, value) from a tEXt / iTXt / zTXt chunk, or None."""
    import zlib
    if b"\0" not in body:
        return None
    key, rest = body.split(b"\0", 1)
    key = key.decode("latin-1")
    if ctype == b"tEXt":
        return key, rest.decode("latin-1")
    if ctype == b"zTXt":
        return key, zlib.decompress(rest[1:]).decode("latin-1")
    if ctype == b"iTXt":
        compressed = rest[0] == 1
        rest = rest[2:]                       # flag, method
        _lang, rest = rest.split(b"\0", 1)
        _tkey, text = rest.split(b"\0", 1)
        return key, (zlib.decompress(text) if compressed else text).decode("utf-8")
    return None


def params_from_fields(fields):
    """Rebuild an A1111 block from the saver's separate fields (Prompt, Model, Lora...)."""
    loras = [l.strip() for l in fields.get("Lora", "").split(",") if l.strip()]
    tags = " ".join(f"<lora:{l.split(':')[0]}:{l.split(':')[1].split('/')[0]}>" if ":" in l
                    else f"<lora:{l}:1>" for l in loras)
    pairs = [("Steps", "Steps"), ("Sampler", "Sampler"), ("Schedule type", "Scheduler"),
             ("CFG scale", "CFG"), ("Seed", "Seed"), ("Size", "Size"), ("Model", "Model"),
             ("Denoising strength", "Denoise"), ("Distilled CFG Scale", "Guidance"), ("VAE", "VAE")]
    settings = ", ".join(f"{a}: {fields[b]}" for a, b in pairs if fields.get(b))
    if not settings:
        return None
    prompt = fields.get("Prompt") or fields.get("Positive") or ""
    lines = [" ".join(t for t in (prompt, tags) if t)]
    if fields.get("Negative"):
        lines.append("Negative prompt: " + fields["Negative"])
    lines.append(settings)
    return "\n".join(l for l in lines if l)


def add_hashes_to_params(params, fields, index):
    """Insert Model hash / Lora hashes into an A1111 settings line."""
    lines = params.split("\n")
    settings = lines[-1]
    kv = dict(p.split(": ", 1) for p in settings.split(", ") if ": " in p)
    if "Model hash" in settings or "Lora hashes" in params:
        return None
    extra = []
    model = kv.get("Model") or fields.get("Model")
    if model and model in index:
        extra.append(f"Model hash: {model_hash(index[model])}")
    loras = [l.split(":")[0].strip() for l in fields.get("Lora", "").split(",") if l.strip()]
    lh = hash_fields(None, [(s, index.get(s)) for s in loras]).get("Lora hashes")
    if lh:
        extra.append(f"Lora hashes: {lh}")
    missing = [s for s in ([model] if model else []) + loras if s not in index]
    if missing:
        print("  not found under --models:", ", ".join(missing))
    if not extra:
        return None
    lines[-1] = settings + ", " + ", ".join(extra)
    return "\n".join(lines)


def params_from_graph(graph_json, prompt_override=None):
    """Saver-style fields from ComfyUI's API graph (the 'prompt' chunk stock SaveImage
    writes), using the same graph reader as the saver. Base sampler pass is used."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pathbuilder as pb
    graph = json.loads(graph_json)
    saves = [k for k, n in graph.items() if "save" in str(n.get("class_type", "")).lower()]
    if not saves:
        return None
    _, info = pb.build_prefix(graph, saves[0], "{model}", sampler_pick="farthest")
    pos, neg = pb.prompt_texts(graph, info.get("anchor"))
    if prompt_override is not None:
        pos = prompt_override
    return pos, neg, info


def fix_png(path, index, prompt_override=None):
    data = open(path, "rb").read()
    fields, params_at, iend_at = {}, None, None
    for at, ctype, body, size in _chunks(data):
        if ctype == b"IEND":
            iend_at = at
        kv = _read_text(ctype, body) if ctype in (b"tEXt", b"iTXt", b"zTXt") else None
        if not kv:
            continue
        fields[kv[0]] = kv[1]
        if kv[0] == "parameters":
            params_at = (at, size)
    params = fields.get("parameters")
    if params is None:
        params = params_from_fields(fields)
        if params is None and "prompt" in fields:
            got = params_from_graph(fields["prompt"], prompt_override)
            if got:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import pathbuilder as pb
                pos, neg, info = got
                w, h = struct.unpack(">II", data[16:24])
                built = pb.build_metadata(info, w, h, pos, neg)
                fields.update(built)
                params = built.get("parameters")
                if not pos:
                    print(f"{path}: positive prompt not in the graph; pass --prompt to add it")
        if params is None:
            print(f"{path}: no 'parameters' and no saver fields; text chunks found: "
                  f"{', '.join(fields) or 'none'} (saved by a node other than Tagged Image Saver?)")
            return
    new = add_hashes_to_params(params, fields, index)
    if new is None:
        print(f"{path}: nothing to add")
        return
    at, size = params_at if params_at else (iend_at, 0)   # no chunk yet: insert before IEND
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data[:at] + _text_chunk("parameters", new) + data[at + size:])
    os.replace(tmp, path)
    print(f"{path}:\n  {new.splitlines()[-1]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Add Civitai hashes to Tagged Image Saver PNGs.")
    ap.add_argument("--models", required=True, action="append",
                    help="ComfyUI models dir (repeatable); searched recursively")
    ap.add_argument("--prompt", help="positive prompt to use when the PNG has only ComfyUI's graph "
                                     "and the text can't be traced (applies to every PNG given)")
    ap.add_argument("pngs", nargs="+")
    a = ap.parse_args()
    idx = {}
    for root in a.models:
        idx.update({k: v for k, v in _index_models(root).items() if k not in idx})
    for p in a.pngs:
        fix_png(p, idx, a.prompt)
