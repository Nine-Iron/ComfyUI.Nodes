"""LoRA Sweep for ComfyUI.

A controller node that lists every LoRA in the pipeline and, right before each
queued run, sets their strengths on the loaders themselves. Each LoRA has a
mode:

    on      always used, at its own strength
    off     never used (strength 0, which ComfyUI skips)
    sweep   steps from min to max (one value per run)
    random  a random value between min and max

max_loras limits how many LoRAs one run uses: the ones set to "on" always go
in, the remaining slots are filled at random from the sweep/random ones, and
the rest run at strength 0. The values go into the loaders' own widgets, so the
workflow saved in each image shows what was really used. With Sampler Sweep or
Checkpoint Sweep also installed, all sweep nodes share one counter and together
cover every combination.

Wire it in after a LoRA loader chain (MODEL/CLIP in, MODEL/CLIP out) to list
the LoRAs upstream of it only; with nothing wired in it lists every LoRA loader
in the workflow (LoraLoader, LoraLoaderModelOnly, rgthree Power Lora Loader).
The work happens in the browser (web/lora_sweep.js); on the server the node
just passes the model and clip through. The per-LoRA settings are kept in the
node's properties, which are saved with the workflow.
"""


class LoraSweep:
    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "passthrough"
    DESCRIPTION = ("Lists every LoRA in the pipeline with a mode each: on (its own strength), off, "
                   "sweep (min to max, one value per run) or random (between min and max). max_loras "
                   "limits how many LoRAs one run uses: 'on' ones always, the rest picked at random. "
                   "Several sweep nodes in one workflow share one counter. Wire a loader chain's "
                   "MODEL/CLIP through it to list only the LoRAs upstream; with nothing wired in it "
                   "lists every LoRA loader.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "max_loras": ("INT", {"default": 0, "min": 0, "max": 100,
                              "tooltip": "Most LoRAs used in one run (0 = no limit). LoRAs set to 'on' are "
                                         "always used; the remaining slots are filled at random from the "
                                         "ones set to sweep or random."}),
                "step": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                         "tooltip": "Spacing of the values between min and max, for sweep and random."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Wire the MODEL output of a LoRA loader chain in here (and on) "
                                    "to list only the LoRAs upstream. Nothing wired in: every LoRA loader."}),
                "clip": ("CLIP", {"tooltip": "The same chain's CLIP output, passed through."}),
            },
        }

    def passthrough(self, model=None, clip=None, **_):
        return (model, clip)


NODE_CLASS_MAPPINGS = {"LoraSweep": LoraSweep}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraSweep": "LoRA Sweep"}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
