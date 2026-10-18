"""Sampler Sweep for ComfyUI.

A controller node: right before each queued run it sets sampler / scheduler /
steps / cfg on the KSampler nodes, either swept through every combination (one
per run) or picked at random. The values go into the KSampler's own widgets,
so filenames such as %KSampler.steps% and the workflow saved in each image
show what was really used. With LoRA Strength Sweep also installed, all sweep
nodes in a workflow share one counter and together cover every combination.

Wire it in after a KSampler (LATENT in, LATENT out) to sweep that sampler
only; with nothing wired in it drives every KSampler. The sweep itself happens
in the browser (web/sampler_sweep.js); on the server the node just passes the
latent through.
"""
MODES = ["off", "sweep", "random"]


class SamplerSweep:
    CATEGORY = "sampling"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "passthrough"
    DESCRIPTION = ("Sets the KSampler's sampler, scheduler, steps and cfg before each run: "
                   "'sweep' steps through every combination (one per run), 'random' picks one. "
                   "Set the Run count to the total it shows. Wire a KSampler's LATENT through it to "
                   "sweep that sampler only; with nothing wired in it drives every KSampler.")

    @classmethod
    def INPUT_TYPES(cls):
        def mode(tip):
            return (MODES, {"default": "off", "tooltip": tip})

        def text(default, tip):
            return ("STRING", {"default": default, "multiline": False, "tooltip": tip})

        lists = ("Comma list. '*' or empty = all; wildcards like dpmpp_* work.")
        nums = ("Comma list of values and ranges: 20,30,40 or 20-40:5 (from-to:step).")
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "sampler_mode": mode("Sampler: off (leave it), sweep, or random."),
                "sampler": text("euler, dpmpp_2m, res_multistep", lists),
                "scheduler_mode": mode("Scheduler: off, sweep, or random."),
                "scheduler": text("simple, normal, karras", lists),
                "steps_mode": mode("Steps: off, sweep, or random."),
                "steps": text("20-40:10", nums),
                "cfg_mode": mode("CFG: off, sweep, or random."),
                "cfg": text("3-7:1", nums),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "Wire the KSampler's LATENT output in here (and on to VAE Decode) "
                                      "to sweep that KSampler only. Nothing wired in: every KSampler."}),
            },
        }

    def passthrough(self, latent=None, **_):
        return (latent,)


NODE_CLASS_MAPPINGS = {"SamplerSweep": SamplerSweep}
NODE_DISPLAY_NAME_MAPPINGS = {"SamplerSweep": "Sampler Sweep"}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
