"""LoRA Strength Sweep for ComfyUI.

A controller node: right before each queued run it sets strength_model /
strength_clip on the LoRA loaders, either swept through every value (one per
run) or picked at random. The values go into the loaders' own widgets, so the
workflow saved in each image shows what was really used. With Sampler Sweep
also installed, all sweep nodes in a workflow share one counter and together
cover every combination.

Wire it in after a LoRA loader (MODEL/CLIP in, MODEL/CLIP out) to sweep that
loader only; with nothing wired in it drives every LoRA loader. The sweep
itself happens in the browser (web/lora_sweep.js); on the server the node just
passes the model and clip through.
"""

MODES = ["off", "sweep", "random"]


class LoraSweep:
    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "passthrough"
    DESCRIPTION = ("Sets strength_model / strength_clip on LoRA loaders before each run: 'sweep' steps "
                   "through the values (one per run), 'random' picks one. Several sweep nodes in one "
                   "workflow share one counter, so together they cover every combination. Wire a "
                   "loader's MODEL/CLIP through it to sweep that loader only; with nothing wired in it "
                   "drives every LoRA loader.")

    @classmethod
    def INPUT_TYPES(cls):
        nums = ("Comma list of values and ranges: 0.6,0.8,1 or 0.4-1:0.2 (from-to:step). "
                "Negative: -1..1:0.5")
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "model_mode": (MODES, {"default": "sweep", "tooltip": "strength_model: off, sweep, or random."}),
                "strength_model": ("STRING", {"default": "0.4-1:0.2", "tooltip": nums}),
                "clip_mode": (MODES + ["same as model"], {"default": "same as model",
                              "tooltip": "strength_clip: off, sweep, random, or the same value as strength_model."}),
                "strength_clip": ("STRING", {"default": "1", "tooltip": nums}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Wire a LoRA loader's MODEL output in here (and on) to sweep "
                                    "that loader only. Nothing wired in: every LoRA loader."}),
                "clip": ("CLIP", {"tooltip": "The same loader's CLIP output, passed through."}),
            },
        }

    def passthrough(self, model=None, clip=None, **_):
        return (model, clip)


NODE_CLASS_MAPPINGS = {"LoraSweep": LoraSweep}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraSweep": "LoRA Strength Sweep"}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
