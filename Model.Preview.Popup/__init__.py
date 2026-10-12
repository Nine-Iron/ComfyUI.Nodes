"""Model Preview Popup - ComfyUI extension (no nodes).

Hovering a model in the Model Library sidebar shows a large popup with the
preview images saved beside the model file:

    <name>.preview.<ext>      the main preview (what ComfyUI shows as the icon)
    <name>.<ext>              also accepted as the main preview
    <name>.preview-2.<ext>    extras, as civitai-likes -previews N saves them
    ...

Click an extra in the popup to view it large. When a model has preview files
beside it, the thumbnail some authors embed inside the model file is ignored,
so the sidebar icon stays on your preview instead of switching on hover.

Install: copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
"""

import logging
import os
import re

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


try:
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    @routes.get("/model_preview_popup/list")
    async def _list(request):
        folder = request.query.get("folder", "")
        filename = request.query.get("filename", "")
        path = _resolve(folder, filename)
        if not path:
            return web.json_response({"found": False, "images": []})
        return web.json_response({"found": True, "images": preview_files(path)})

    @routes.get("/model_preview_popup/image")
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
