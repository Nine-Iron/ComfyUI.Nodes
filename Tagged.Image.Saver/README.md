# Tagged Image Saver

A drop-in replacement for **Save Image** that builds the folder/filename from what actually ran:

```
{workflow}/{date}/{sampler}-{scheduler}/{loras}-{steps}-{cfg}-{denoise}-{seed}
```

e.g. `krea2_turbo/2026-11-05/euler-simple/Wackadoodle_Krea2_0.8_inkwash_0.65-8-1-1-123456789_00001_.png`

## Install

Put the `Tagged.Image.Saver` folder in `ComfyUI/custom_nodes/` (or symlink it there) and restart ComfyUI. Hard-refresh the browser so the small frontend script loads. The node is **Save Image (Tagged Path)** under `image`.

## Widgets

| Widget | Default | Notes |
|---|---|---|
| `template` | see above | `/` makes folders. Missing values drop out cleanly. |
| `workflow_name` | `auto` | `auto` = the open workflow's file name, read at queue time. Type anything to override. |
| `sampler_pick` | `nearest` | With several samplers (hires fix, detailer): `nearest` to this node, or `farthest` upstream (the base pass). |
| `lora_name_max` | 40 | Characters kept per LoRA name. |
| `no_lora_text` | `nolora` | What `{loras}` becomes when nothing is active. |
| `ignore_classes` | empty | Comma-separated class-name fragments to skip, e.g. `TagPicker,Sweep`. |

Outputs: `images` (passthrough) and `path` (the resolved prefix). The resolved prefix also shows in the node's `last_saved` field after each run.

## Tokens

`{workflow}` `{date}` `{date:yyyy-MM-dd_hhmm}` `{time}` `{sampler}` `{scheduler}` `{steps}` `{cfg}` `{denoise}` `{seed}` `{guidance}` `{model}` `{loras}` `{lora_names}` `{lora_count}` `{width}` `{height}`

ComfyUI's own `%width%`, `%year%` etc. and `%batch_num%` also still work.

## How it finds things

It reads the executed API prompt (the hidden `PROMPT` input) and walks upstream from this node.

- **Bypassed and muted nodes are never counted**: ComfyUI leaves them out of the prompt.
- **Sampler**: the nearest/farthest node with `sampler_name` plus steps/cfg/seed (KSampler, KSamplerAdvanced, Efficiency, detailers). If there is none, it falls back to `SamplerCustomAdvanced` and takes each value from the nearest upstream piece (`RandomNoise`, `KSamplerSelect`, `BasicScheduler`, `CFGGuider`, `FluxGuidance`).
- **Linked values** (seed from a Seed node, sampler from a selector, strength from a primitive) are followed back to their source.
- **LoRAs**: any upstream node whose outputs include MODEL, CLIP, HOOKS or a `*LORA*` type. Entries are read in chain order (closest to the checkpoint first) from:
  - core `LoraLoader` / `LoraLoaderModelOnly` / hook LoRAs, pysssss, Nunchaku, WanVideo single
  - rgthree Power Lora Loader (`on` respected, separate clip strength shown as `1c0.5`) and Lora Loader Stack
  - Comfyroll `CR LoRA Stack` (`switch_N`), Efficiency `LoRA Stacker` (`lora_count`, simple/advanced) and Efficient Loader
  - Easy-Use `loraStack` (`toggle`, `num_loras`, mode), LoRA Manager (`active` flag), WanVideo multi-select
- LoRAs with `None` selected, toggled off, or at 0 strength are skipped. Folder paths and `.safetensors` are stripped from names.

## Caveats

- A node that only *reads* LoRA names but passes MODEL/CLIP through would be counted. Put its class name in `ignore_classes`.
- Sweep/batch nodes that vary LoRA or strength per image: the path shows the widget values, not each image's value.
- Each path segment is capped at 200 bytes. Over that, it's truncated with a short hash so names stay unique.
- Calls through the HTTP API skip the frontend script, so `auto` becomes `unsaved`. Set `workflow_name` explicitly there.
