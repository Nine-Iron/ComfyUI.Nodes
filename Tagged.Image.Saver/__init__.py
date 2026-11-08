"""Tagged Image Saver: saves images under a path built from the executed graph.

Default layout:
    {workflow}/{date}/{sampler}-{scheduler}/{loras}-{steps}-{cfg}-{denoise}-{seed}
"""
import logging

from nodes import SaveImage

from .pathbuilder import DEFAULT_TEMPLATE, build_prefix

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
                    prompt=None, extra_pnginfo=None, unique_id=None):
        height, width = int(images[0].shape[0]), int(images[0].shape[1])
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

        out = self.save_images(images, prefix, prompt, extra_pnginfo)
        ui = dict(out.get("ui", {}))
        ui["text"] = [prefix]
        return {"ui": ui, "result": (images, prefix)}


NODE_CLASS_MAPPINGS = {"TaggedImageSaver": TaggedImageSaver}
NODE_DISPLAY_NAME_MAPPINGS = {"TaggedImageSaver": "Save Image (Tagged Path)"}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
