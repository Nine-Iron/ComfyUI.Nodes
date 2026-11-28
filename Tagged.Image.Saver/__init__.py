"""Tagged Image Saver: saves images under a path built from the executed graph.

Default layout:
    {workflow}/{date}/{sampler}-{scheduler}/{loras}-{steps}-{cfg}-{denoise}-{seed}
"""
import json
import logging
import os

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args
from nodes import SaveImage

from .pathbuilder import DEFAULT_TEMPLATE, build_metadata, build_prefix, find_vae, prompt_texts_near

log = logging.getLogger("TaggedImageSaver")


def _return_types(class_type):
    """Output types for a registered node class, or None if unknown."""
    try:
        import nodes

        cls = nodes.NODE_CLASS_MAPPINGS.get(class_type)
        if cls is None:
            return None
        types = getattr(cls, "RETURN_TYPES", None)
        if types is None:
            return None
        return tuple(str(t) for t in types)
    except Exception:
        return None


METADATA_MODES = ["essentials + workflow", "essentials", "full (ComfyUI default)"]


def _add_text(png, key, value):
    """tEXt for Latin-1 text, iTXt (UTF-8) otherwise."""
    try:
        value.encode("latin-1")
        png.add_text(key, value)
    except UnicodeEncodeError:
        png.add_itxt(key, value)


TEMPLATE_HELP = (
    "Tokens: {workflow} {date} {date:yyyy-MM-dd_hhmm} {time} {sampler} {scheduler} "
    "{steps} {cfg} {denoise} {seed} {guidance} {model} {loras} {lora_names} "
    "{lora_count} {width} {height}. '/' makes folders. Missing values drop out."
)


class TaggedImageSaver(SaveImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "template": ("STRING", {"default": DEFAULT_TEMPLATE, "tooltip": TEMPLATE_HELP}),
                "workflow_name": ("STRING", {
                    "default": "auto",
                    "tooltip": "'auto' = the open workflow's file name, filled in at queue time. "
                               "Type anything else to override.",
                }),
                "sampler_pick": (["nearest", "farthest"], {
                    "default": "nearest",
                    "tooltip": "Which sampler supplies sampler/steps/cfg/seed when there are several "
                               "(e.g. hires fix or detailer): nearest to this node or farthest upstream.",
                }),
            },
            "optional": {
                "lora_name_max": ("INT", {
                    "default": 40, "min": 8, "max": 120,
                    "tooltip": "Truncate each LoRA name to this many characters.",
                }),
                "no_lora_text": ("STRING", {
                    "default": "nolora",
                    "tooltip": "Used for {loras} when no LoRA is active.",
                }),
                "ignore_classes": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated class-name fragments to skip when scanning, "
                               "e.g. TagPicker,Sweep",
                }),
                "metadata": (METADATA_MODES, {
                    "default": METADATA_MODES[0],
                    "tooltip": "essentials: only what made the image (Prompt, Negative, Model, Checkpoint, "
                               "Sampler, Scheduler, Steps, CFG, Seed, Lora, ...) as separate PNG fields "
                               "readable with exiftool. + workflow: also the workflow, so dragging the "
                               "image into ComfyUI reloads it. full: ComfyUI's complete graph as well.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "path")
    FUNCTION = "save_tagged"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = ("Saves images to a folder/filename built from the sampler, scheduler, active LoRAs "
                   "(any loader type), steps, cfg, denoise and seed actually used in the run.")

    def save_tagged(self, images, template, workflow_name, sampler_pick,
                    lora_name_max=40, no_lora_text="nolora", ignore_classes="",
                    metadata=METADATA_MODES[0],
                    prompt=None, extra_pnginfo=None, unique_id=None):
        height, width = int(images[0].shape[0]), int(images[0].shape[1])
        info = None
        try:
            prefix, info = build_prefix(
                prompt or {}, unique_id, template, workflow_name, sampler_pick,
                lora_name_max, no_lora_text, ignore_classes,
                width=width, height=height, return_types=_return_types,
            )
            log.debug("TaggedImageSaver %s -> %s (%s)", unique_id, prefix, info)
        except Exception:
            log.exception("TaggedImageSaver: path build failed, falling back to 'ComfyUI'")
            prefix = "ComfyUI"

        fields = None
        if info is not None:
            try:
                positive, negative = prompt_texts_near(prompt or {}, unique_id, info.get("anchor"))
                vae = find_vae(prompt or {}, str(unique_id))
                fields = build_metadata(info, width, height, positive, negative, vae)
            except Exception:
                log.exception("TaggedImageSaver: metadata build failed, saving full ComfyUI metadata")

        if fields is None or metadata == METADATA_MODES[2]:
            out = self.save_png(images, prefix, prompt, extra_pnginfo, fields, full=True)
        else:
            workflow = extra_pnginfo if metadata == METADATA_MODES[0] else None
            out = self.save_png(images, prefix, None, workflow, fields, full=False)
        ui = dict(out.get("ui", {}))
        ui["text"] = [prefix]
        return {"ui": ui, "result": (images, prefix)}


    def save_png(self, images, prefix, prompt, extra_pnginfo, fields, full):
        """SaveImage.save_images with our own text chunks. In full mode ComfyUI's
        'prompt' graph is kept, so the positive text goes under 'Positive' instead
        (exiftool shows the 'prompt' chunk as Prompt too)."""
        prefix += self.prefix_append
        folder, filename, counter, subfolder, prefix = folder_paths.get_save_image_path(
            prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = []
        for batch_number, image in enumerate(images):
            img = Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8))
            png = None
            if not args.disable_metadata:
                png = PngInfo()
                for key, value in (fields or {}).items():
                    if full and key == "Prompt":
                        key = "Positive"
                    _add_text(png, key, value)
                if full and prompt is not None:
                    png.add_text("prompt", json.dumps(prompt))
                for key, value in (extra_pnginfo or {}).items():
                    if full or key == "workflow":
                        png.add_text(key, json.dumps(value))
            file = f"{filename.replace('%batch_num%', str(batch_number))}_{counter:05}_.png"
            img.save(os.path.join(folder, file), pnginfo=png, compress_level=self.compress_level)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1
        return {"ui": {"images": results}, "result": (images,)}


NODE_CLASS_MAPPINGS = {"TaggedImageSaver": TaggedImageSaver}
NODE_DISPLAY_NAME_MAPPINGS = {"TaggedImageSaver": "Save Image (Tagged Path)"}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
