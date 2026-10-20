"""Checkpoint Sweep for ComfyUI.

A controller node: right before each queued run it sets ckpt_name on the
checkpoint loaders and/or unet_name on the diffusion model loaders (Load
Diffusion Model, Unet Loader (GGUF)), either swept through a list (one per
run) or picked at random. The names go into the loaders' own widgets, so the
workflow saved in each image shows what was really used, and a filename prefix
like %CheckpointLoaderSimple.ckpt_name% names the images after it.

With Sampler Sweep and LoRA Strength Sweep also installed, all sweep nodes in
a workflow share one counter and together cover every combination. The model
always changes slowest, so the other settings are swept with each model
loaded once rather than reloading it every run.

Wire it in after a loader (MODEL/CLIP/VAE in, the same out) to sweep that
loader only; with nothing wired in it drives every loader. The sweep itself
happens in the browser (web/checkpoint_sweep.js); on the server the node just
passes everything through.
"""

MODES = ["off", "sweep", "random"]


class CheckpointSweep:
    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "passthrough"
    DESCRIPTION = ("Sets the checkpoint (ckpt_name) and/or diffusion model (unet_name) on the loaders "
                   "before each run: 'sweep' steps through the list (one per run), 'random' picks one. "
                   "Several sweep nodes in one workflow share one counter, and the model changes slowest, "
                   "so each model is loaded once. Wire a loader's outputs through it to sweep that loader "
                   "only; with nothing wired in it drives every loader.")

    @classmethod
    def INPUT_TYPES(cls):
        lists = ("Comma list of names as the loader lists them, or patterns: sdxl1/*, *juggernaut*, "
                 "*copax*. '*' or empty = every one the loader lists.")
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "checkpoint_mode": (MODES, {"default": "sweep",
                                    "tooltip": "ckpt_name on checkpoint loaders: off (leave it), sweep, or random."}),
                "checkpoints": ("STRING", {"default": "*", "multiline": False, "tooltip": lists}),
                "diffusion_mode": (MODES, {"default": "off",
                                   "tooltip": "unet_name on diffusion model loaders (Load Diffusion Model, "
                                              "GGUF): off, sweep, or random."}),
                "diffusion_models": ("STRING", {"default": "*", "multiline": False, "tooltip": lists}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Wire a loader's MODEL output in here (and on) to sweep that "
                                    "loader only. Nothing wired in: every loader."}),
                "clip": ("CLIP", {"tooltip": "The same loader's CLIP output, passed through."}),
                "vae": ("VAE", {"tooltip": "The same loader's VAE output, passed through."}),
            },
        }

    def passthrough(self, model=None, clip=None, vae=None, **_):
        return (model, clip, vae)


NODE_CLASS_MAPPINGS = {"CheckpointSweep": CheckpointSweep}
NODE_DISPLAY_NAME_MAPPINGS = {"CheckpointSweep": "Checkpoint Sweep"}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
