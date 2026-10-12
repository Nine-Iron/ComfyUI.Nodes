# ComfyUI.Nodes

Three small tools for working with models in [ComfyUI](https://github.com/comfyanonymous/ComfyUI):

| Folder | What it is | What it does |
| --- | --- | --- |
| [`Lora.Tag.Picker/`](#lora-tag-picker) | Custom node | Reads the training tags out of the LoRAs you have loaded and adds the ones you pick to your prompt. |
| [`Model.Preview.Popup/`](#model-preview-popup) | Frontend extension (no nodes) | Shows a large preview popup when you hover a model in the Model Library sidebar. |
| [`CivitAI.Model.Plus.Preview.Picker/`](#civitai-likes) | Command-line tool (Go) | Downloads the models you have liked on Civitai, with preview images, skipping anything you already have. |

Each one works on its own. The two ComfyUI pieces need nothing beyond ComfyUI itself, and the Go tool uses only the standard library.

## Install

The repository holds several independent folders, so do not clone it straight into `custom_nodes/`. Clone it anywhere, then copy or symlink the folders you want:

```sh
git clone https://github.com/Nine-Iron/ComfyUI.Nodes.git
cd ComfyUI.Nodes

cp -r Lora.Tag.Picker     ~/ComfyUI/custom_nodes/
cp -r Model.Preview.Popup ~/ComfyUI/custom_nodes/
```

Restart ComfyUI and reload the browser tab.

## LoRA Tag Picker

A node that sits after your LoRA loaders. It finds every LoRA loaded upstream, reads the training tags embedded in each `.safetensors` file, shows them as clickable chips, and appends or prepends the active ones to your prompt when you queue.

### Usage

1. Add **LoRA Tag Picker** (category `loaders`).
2. Connect `model` and/or `clip` from the end of your LoRA chain. Both are passed through unchanged, so the node can sit inline.
3. Type your prompt in the node's text box and wire the `text` output to your text encoder.
4. Click chips to turn tags on or off. Your choices are saved with the workflow.

| Input | Notes |
| --- | --- |
| `text` | Your prompt. Active tags are added to it. |
| `tag_position` | `after prompt` or `before prompt`. |
| `model`, `clip` | Optional. Passed through; used to find the LoRAs upstream. |

| Output | Notes |
| --- | --- |
| `model`, `clip` | The inputs, unchanged. |
| `text` | Your prompt with the active tags added. |
| `tags` | Only the active tags, comma-separated. |

Tags already present in your prompt are not added a second time. Bypassed or muted loaders are skipped. Any upstream node with a `MODEL` or `CLIP` output and a widget whose name contains `lora` counts as a loader, which covers the built-in loaders and most multi-LoRA loader nodes.

### How tags are ranked

Tags are ranked by coverage: the share of training images whose caption contained the tag.

| Tier | Rule | In the UI |
| --- | --- | --- |
| trigger | Coverage of 90% or more, or the tag matches the kohya dataset folder token (`<repeats>_<token> <class>`) or the file's declared trigger phrase | On by default |
| common | Coverage of 25% or more | Shown, off by default |
| rare | Everything else | Hidden behind "+N more" |

When a file records no image count, coverage is measured against the most used tag instead, and the chip tooltip says so. At most 80 tags per LoRA are shown.

A LoRA shows no tags if its author stripped the metadata, if it was trained without captions, or if it is not a `.safetensors` file. The node says which.

### API use

When a workflow is queued without the web UI, the node walks the prompt graph itself and applies the trigger-tier tags plus any saved overrides. The tag data is also available at `GET /lora_tag_picker/tags?name=<lora name>`.

## Model Preview Popup

Hover a model in the Model Library sidebar to see a large popup with the preview images saved beside the model file, plus the title, trigger phrase, author, and description from its metadata when present.

Preview files are matched by name, next to the model:

| File | Role |
| --- | --- |
| `<name>.preview.<ext>` | Main preview (what ComfyUI shows as the icon) |
| `<name>.<ext>` | Also accepted as the main preview |
| `<name>.preview-2.<ext>`, `-3`, ... | Extras, shown as thumbnails |

Accepted extensions are `png`, `jpg`, `jpeg`, `webp`, and `gif`. Hover or click a thumbnail to view it large.

Two things to know:

- When a model has preview files beside it, the thumbnail some authors embed inside the model file is ignored, so the sidebar icon stays on your preview.
- Models with no preview files are left alone, and ComfyUI's own hover popup shows as usual.

New preview files are picked up within about 30 seconds, without a restart.

## civitai-likes

A command-line tool that downloads the models you have liked on Civitai into your ComfyUI models folder, without ever fetching one you already have. It also saves preview images in the layout Model Preview Popup reads.

### Setup

Build it (Go, standard library only):

```sh
cd "CivitAI.Model.Plus.Preview.Picker"
go build -o civitai-likes civitai-likes.go
```

A prebuilt `civitai-likes-linux-arm64` binary is included for 64-bit ARM Linux.

Create an API key at <https://civitai.com/user/account>, then either export it or save it to a file:

```sh
export CIVITAI_TOKEN=...
# or
mkdir -p ~/.config/civitai && echo '...' > ~/.config/civitai/token
```

### Usage

```sh
civitai-likes -n                 # show what would happen, change nothing
civitai-likes                    # show the plan, ask, then download
civitai-likes -yes               # no question asked
civitai-likes -base +pony        # the default base models plus Pony
civitai-likes -base sdxl1,sd1    # only these base models
civitai-likes -base all          # every base model
civitai-likes -previews 4        # keep up to 4 preview images per model
civitai-likes -previews-only     # only fetch missing previews for models you have
civitai-likes -rename            # also rename files you already have to the tagged form
civitai-likes -latest            # also fetch the newest version when you hold an older one
civitai-likes -checksum          # checksum every model file now (slow, once)
```

Start with `-n` to see the plan before anything is written.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-models` | `~/ComfyUI/models` | ComfyUI models folder |
| `-base` | `flux1_dev,sdxl1,qwen,sd1,krea2` | Base models to fetch, as tags or Civitai names (`"Flux.1 D"`). Prefix with `+` to add to the defaults; `all` takes every base. |
| `-previews` | `1` | Preview images to keep per model file (`0` for none) |
| `-max` | `1000` | Refuse to act on a likes list longer than this |
| `-state` | `<models>/.civitai-likes.json` | State file |
| `-token-file` | `~/.config/civitai/token` | File holding the API token |
| `-api` | `https://civitai.com` | Civitai base URL |

Every run lists the other base models found in your likes, with the tag to pass to `-base`.

### What it does

- **Naming.** Files are saved as `<base model tag>_<Civitai's file name>`, for example `flux1_dev_some_lora.safetensors`, so loader lists group by base model. A liked model with versions for several base models gets one file per selected base model, the newest version for each.
- **Placement.** Each Civitai model type goes to its ComfyUI folder: checkpoints, `loras` (LORA, LoCon, DoRA), `embeddings`, `vae`, `controlnet`, `upscale_models`, `hypernetworks`, `animatediff_models`.
- **Duplicate checks.** Before downloading, each file is checked against the whole models tree, whatever folder it was filed under: first by its tagged and untagged name for every version, then by size and SHA-256 to catch a copy saved under another name. Only what is missing is downloaded, and nothing is ever overwritten.
- **Previews.** Each model file gets the first still image from its Civitai page saved beside it as `<name>.preview.<ext>`. With `-previews N`, extras are saved as `<name>.preview-2.<ext>` and so on; rename the one you want to `<name>.preview.<ext>` to make it the icon. Files you already have get previews too, unless one is already there.
- **Checksums.** The state file records what has been seen. Each run checksums only the model files that are new or changed since the last run. `-checksum` does every file that has none on record.
- **Safety margins.** Downloads keep 2 GB free on the disk, retry up to four times, and abort if no data arrives for two minutes.

## License

Copyright (C) 2026 Nine-Iron

Licensed under the [GNU General Public License v3.0](LICENSE) or later. You may use, modify, and redistribute this software freely, provided that anything you distribute that is built on it stays under the same license with its source available.
