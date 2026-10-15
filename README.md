# ComfyUI.Nodes

Four small tools for working with models and prompts in [ComfyUI](https://github.com/comfyanonymous/ComfyUI):

| Folder | What it is | What it does |
| --- | --- | --- |
| [`Lora.Tag.Picker/`](#lora-tag-picker) | Custom node | Reads the training tags out of the LoRAs you have loaded and adds the ones you pick to your prompt. |
| [`Prompt.Generator/`](#prompt-generator) | Custom nodes | Makes random prompts, from wildcard templates or by asking a language model (Claude, ChatGPT, or a local server). |
| [`Model.Preview.Popup/`](#model-preview-popup) | Frontend extension (no nodes) | Shows a large preview popup when you hover a model in the Model Library sidebar, and lets you hide base models you don't use. |
| [`CivitAI.Model.Plus.Preview.Picker/`](#civitai-likes) | Command-line tool (Go) | Downloads the models you have liked on Civitai, with preview images, skipping anything you already have. |

Each one works on its own, and they work well together: civitai-likes saves previews and base-model tags in the layout Model Preview Popup reads, and LoRA Tag Picker reads the tags inside the LoRAs it downloads.

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [LoRA Tag Picker](#lora-tag-picker)
- [Prompt Generator](#prompt-generator)
- [Model Preview Popup](#model-preview-popup)
- [civitai-likes](#civitai-likes)
- [HTTP routes](#http-routes)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Requirements

- **ComfyUI pieces:** a recent ComfyUI with the new frontend (the Model Library sidebar and DOM widgets). Nothing else to install: they use only the Python standard library and what ComfyUI already ships.
- **civitai-likes:** Go 1.21 or later to build it (standard library only), and a Civitai API key.

## Install

The repository holds several independent folders, so do not clone it straight into `custom_nodes/`. Clone it anywhere, then copy or symlink the folders you want:

```sh
git clone https://github.com/Nine-Iron/ComfyUI.Nodes.git
cd ComfyUI.Nodes

cp -r Lora.Tag.Picker     ~/ComfyUI/custom_nodes/
cp -r Prompt.Generator    ~/ComfyUI/custom_nodes/
cp -r Model.Preview.Popup ~/ComfyUI/custom_nodes/
```

To pick up updates with a `git pull`, symlink instead of copying:

```sh
ln -s "$PWD/Lora.Tag.Picker" ~/ComfyUI/custom_nodes/Lora.Tag.Picker
```

Restart ComfyUI and reload the browser tab. The Python side is only read at startup; the JavaScript side is only read when the page loads.

## LoRA Tag Picker

A node that sits after your LoRA loaders. It finds every LoRA loaded upstream, reads the training tags embedded in each `.safetensors` file, shows them as clickable chips, and appends or prepends the active ones to your prompt when you queue.

### Usage

1. Add **LoRA Tag Picker** (category `loaders`).
2. Connect `model` and/or `clip` from the end of your LoRA chain. Both are passed through unchanged, so the node can sit inline.
3. Type your prompt in the node's text box and wire the `text` output to your text encoder.
4. Click chips to turn tags on or off. Your choices are saved with the workflow.

| Input | Default | Notes |
| --- | --- | --- |
| `text` | | Your prompt. Active tags are added to it. |
| `tag_position` | `after prompt` | `after prompt` or `before prompt`. |
| `model`, `clip` | | Optional. Passed through; used to find the LoRAs upstream. |
| `max_tags` | `8` | Most tags one LoRA shows up front (1 to 40). The rest stay behind "+N more". |

| Output | Notes |
| --- | --- |
| `model`, `clip` | The inputs, unchanged. |
| `text` | Your prompt with the active tags added. |
| `tags` | Only the active tags, comma-separated. |

Each LoRA gets its own block of chips, in the order the LoRAs are loaded, with the number of training images when the file records it. Each chip shows its coverage; hover it for the details. Up to three trigger tags per LoRA are on by default; everything else is off until you click it. The node only stores the chips you changed, so a LoRA you swap in later starts from its own defaults.

Tags already present in your prompt are not added a second time, and a tag shared by two LoRAs is added once. Bypassed or muted loaders are skipped. Any upstream node with a `MODEL` or `CLIP` output and a widget whose name contains `lora` counts as a loader, which covers the built-in loaders and most multi-LoRA loader nodes (including ones whose entries can be switched off). The node re-checks the graph every second, so rewiring or changing a LoRA updates the chips without a refresh.

### How tags are ranked

Tags are ranked by coverage: the share of training images whose caption contained the tag. Spelling variants (`long_hair`, `Long Hair`) are merged.

| Tier | Rule | In the UI |
| --- | --- | --- |
| trigger | The tag matching the kohya dataset folder token (`<repeats>_<token> <class>`) or the file's declared trigger phrase (`modelspec.trigger_phrase`). When the file names neither, every tag with coverage of 90% or more. | Shown first; up to three on by default |
| common | Coverage of 25% or more | Shown, off by default |
| rare | Everything else | Hidden behind "+N more" |

When a file records no image count, coverage is measured against the most used tag instead, and the chip tooltip says so. A stray dataset folder with under a tenth of the images of the biggest one is ignored, so a single extra image does not turn all its tags into triggers. At most 40 tags per LoRA are sent to the UI.

**Sentence captions.** LoRAs trained on prose captions (as written by captioning models) record dozens of comma-split sentence fragments instead of tags. The node detects this and condenses them: the phrase the first caption opens with becomes the trigger if at least 90% of captions open with it, and the rest is reduced to the most common content words. The LoRA's block says when this happened.

A LoRA shows no tags if its author stripped the metadata, if it was trained without captions, or if it is not a `.safetensors` file. The node says which. If it has no tags but its dataset folder name is recorded, the folder token is offered as a guess.

### API use

When a workflow is queued without the web UI, the node walks the prompt graph itself and applies the same defaults (the first triggers, limited by `max_tags`) plus any saved overrides. The tag data is also available over HTTP; see [HTTP routes](#http-routes).

## Prompt Generator

Two nodes in the `prompt` category that output a prompt string to wire into your text encoder. Chain them to have a template supply the idea and a language model polish it.

### Random Prompt (Template)

Fills a template from word lists. Instant, uses no GPU memory, and the same seed always gives the same prompt.

| Input | Notes |
| --- | --- |
| `theme` | A built-in template (`scene`, `portrait`, `landscape`, `fantasy`, `sci-fi`, `photo`), `random` to pick one, or `custom (use template)`. |
| `template` | Your own template. Used alone with `custom`; with another theme it is added after the themed prompt. |
| `seed` | Same seed, same prompt. |
| `prefix`, `suffix` | Optional. Expanded the same way and put before and after. |

Template syntax:

| Syntax | Result |
| --- | --- |
| `{a\|b\|c}` | Picks one |
| `{2$$a\|b\|c}` | Picks two, joined with `, ` |
| `__subject__` | A random line from `wildcards/subject.txt` |
| `__folder/name__` | A wildcard file in a subfolder |

Choices and wildcards can be nested. Wildcard files are plain text, one option per line, with `#` for comments. They are looked up in `ComfyUI/wildcards/`, then `ComfyUI/models/wildcards/`, then the node's own `wildcards/` folder, first match wins, so you can override a bundled list by putting a file of the same name in either of the first two. The bundled lists are `camera`, `color`, `lighting`, `mood`, `quality`, `setting`, `style`, `subject`, and `weather`. A missing wildcard is replaced by its name and reported in the console. Edited files are picked up without a restart.

### LLM Prompt

Has a language model write a prompt, or rewrite one wired into `seed_prompt` (for example from Random Prompt).

| Input | Default | Notes |
| --- | --- | --- |
| `provider` | `anthropic` | `anthropic` (Claude), `openai` (ChatGPT), or `openai_compatible` (llama.cpp server, Ollama, LM Studio, OpenRouter, ...). |
| `model` | blank | Blank uses `claude-haiku-5-5` for anthropic, `gpt-4o-mini` for openai, and the server's own model for openai_compatible (Ollama needs a name). |
| `base_url` | blank | Blank uses the provider's address; for openai_compatible the default is `http://127.0.0.1:8080/v1`. For Ollama use `http://127.0.0.1:11434/v1`. |
| `idea` | blank | What the image should be about. Blank lets the model choose. |
| `style` | | `natural sentences` (Flux, Qwen, Krea, SD3) or `comma tags` (SD 1.5, SDXL, Pony). |
| `temperature` | `1.0` | 0 to 2. Capped at 1.0 for Anthropic, whose range is 0 to 1. |
| `max_tokens` | `300` | 32 to 4000. |
| `timeout` | `120` | Seconds to wait for an answer, 5 to 900. |
| `seed` | | A new seed asks for a new prompt. ComfyUI reuses the last answer while every input stays the same. |
| `seed_prompt` | | Optional input. A prompt to rewrite and enrich. |

The reply is cleaned of preambles, quotes, code fences, and `<think>` blocks before it is output.

API keys are never typed into the node, because workflows are saved inside every image. Set them in the environment ComfyUI runs in, or copy `keys.example.json` to `keys.json` in the node folder and fill it in:

| Provider | Environment variable | `keys.json` field |
| --- | --- | --- |
| anthropic | `ANTHROPIC_API_KEY` | `anthropic` |
| openai | `OPENAI_API_KEY` | `openai` |
| openai_compatible | `OPENAI_COMPATIBLE_API_KEY` (optional) | `openai_compatible` |

The environment variable wins when both are set. `keys.json` is read at every run, so editing it needs no restart. It is listed in `.gitignore`.

A key is only sent to its provider's own host: `api.anthropic.com`, `api.openai.com`, or `127.0.0.1` for openai_compatible. Without this, a shared workflow could set `base_url` to someone else's server and collect your key when you queue it. To use another host, such as a proxy, OpenRouter, or a server elsewhere on your network, list it outside the workflow:

| Provider | Environment variable | `keys.json` |
| --- | --- | --- |
| anthropic | `ANTHROPIC_KEY_HOSTS` | `"key_hosts": {"anthropic": [...]}` |
| openai | `OPENAI_KEY_HOSTS` | `"key_hosts": {"openai": [...]}` |
| openai_compatible | `OPENAI_COMPATIBLE_KEY_HOSTS` | `"key_hosts": {"openai_compatible": [...]}` |

Hosts are plain names (`openrouter.ai`), comma or space separated in the variable. Listed hosts get the key over `https` only, except `localhost` and `127.0.0.1`. For anthropic and openai, an unlisted `base_url` is refused with an error. For openai_compatible, the request still goes out but without the key, so local servers that need none keep working. Redirects are never followed, so a listed host cannot pass the key on.

## Model Preview Popup

Hover a model in the Model Library sidebar to see a large popup with the preview images saved beside the model file, its base model, and the title, trigger phrase, author, and description from its metadata when present.

### Previews

Preview files are matched by name, next to the model:

| File | Role |
| --- | --- |
| `<name>.preview.<ext>` | Main preview (what ComfyUI shows as the icon) |
| `<name>.<ext>` | Also accepted as the main preview |
| `<name>.preview-2.<ext>`, `-3`, ... | Extras, shown as thumbnails |

Accepted extensions are `png`, `jpg`, `jpeg`, `webp`, and `gif`. Extras appear as a strip of thumbnails above the main picture; hover or click one to view it large. The popup stays put while you move into it, so the strip stays under the pointer.

Pictures load one at a time, main preview first, and moving to another model cancels whatever was still loading. Pictures for the last 8 models hovered are kept, so going back is instant.

Two things to know:

- When a model has preview files beside it, the thumbnail some authors embed inside the model file is ignored, so the sidebar icon stays on your preview.
- A model with no preview files shows its embedded thumbnail in the popup when it has one.

New preview files are picked up within about 30 seconds, without a restart.

### Base models

The top of the popup names the model's base model (for example **Flux.1 D**). It is worked out in this order:

1. The civitai-likes state file (`<models>/.civitai-likes.json`), for files civitai-likes downloaded or identified.
2. A tag in the file or folder name, in any of the forms civitai-likes writes: `flux1_dev_x`, `x_flux1_dev`, `x.flux1_dev`, or a `flux1_dev/` folder.
3. The base model the file itself records (`ss_base_model_version` or `modelspec.architecture`).
4. A guess from the tensor names in the file header. The label's tooltip says when it was guessed.

Anything left over is **Unknown**.

The **Base models** button in the popup opens a list of every base model found, with a file count for each. Untick a base model to hide its files from the Model Library and from loader dropdowns in the graph. A model already chosen in a node stays visible there. The filter applies to `checkpoints`, `loras`, `diffusion_models`, `unet`, `embeddings`, `controlnet`, and `hypernetworks`; text encoders, VAEs, and upscalers are always shown. The choice is saved in your ComfyUI user settings, so it follows you across browsers.

## civitai-likes

A command-line tool that downloads the models you have liked on Civitai into your ComfyUI models folder, without ever fetching one you already have. It also saves preview images in the layout Model Preview Popup reads, can tag file names with their base model, and can identify model files you got elsewhere.

### Setup

Build it (Go 1.21+, standard library only):

```sh
cd "CivitAI.Model.Plus.Preview.Picker"
go build -o civitai-likes civitai.likes.go
```

A prebuilt `civitai.likes.linux.arm64` binary is included for 64-bit ARM Linux. To build for another platform, set `GOOS` and `GOARCH`, for example `GOOS=windows GOARCH=amd64 go build -o civitai-likes.exe civitai.likes.go`.

Create an API key at <https://civitai.com/user/account>, then either export it or save it to a file:

```sh
export CIVITAI_TOKEN=...
# or
mkdir -p ~/.config/civitai && echo '...' > ~/.config/civitai/token
```

The environment variable wins when both are set.

### Usage

```sh
civitai-likes -n                 # show what would happen, change nothing
civitai-likes                    # show the plan, ask, then download (every base model)
civitai-likes -yes               # no question asked
civitai-likes -base pony         # only Pony models
civitai-likes -base sdxl1,sd1    # only these base models
civitai-likes -previews 4        # keep up to 4 preview images per model
civitai-likes -previews-only     # only fetch missing previews for models you have
civitai-likes -rename            # also rename files you already have to the tagged form
civitai-likes -rename-only       # only rename, no downloads or previews
civitai-likes -rename-only -local          # also rename files that are not liked models, looked up by checksum
civitai-likes -rename-only -local -guess   # and tag the ones Civitai does not know from their metadata
civitai-likes -tag folder -rename-only     # switch to loras/flux1_dev/x.safetensors style
civitai-likes -latest            # also fetch the newest version when you hold an older one
civitai-likes -checksum          # checksum every model file now (slow, once)
civitai-likes -ea                # only retry early access versions refused before
civitai-likes -ea -all           # retry those, and do everything a normal run does
```

Start with `-n` to see the plan before anything is written. Ctrl+C stops cleanly at any point; checksums done so far are kept.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-models` | `~/ComfyUI/models` | ComfyUI models folder |
| `-base` | `all` | Base models to handle (downloads, renames, previews), comma separated, as tags (`flux1_dev`) or Civitai names (`"Flux.1 D"`). `all` handles every base. |
| `-tag` | `front` | Where the base model tag goes in file names; see [File names](#file-names). |
| `-previews` | `1` | Preview images to keep per model file (`0` for none) |
| `-previews-only` | off | Only fetch missing previews for liked models already on disk |
| `-rename` | off | Also rename files you already have to the tagged form |
| `-rename-only` | off | Only rename; no downloads, no previews |
| `-local` | off | Also look up model files that are not liked models on Civitai by checksum, so they get tagged and previewed too |
| `-guess` | off | Tag model files Civitai does not know from the base model their own metadata names |
| `-latest` | off | Fetch the newest version even when an older one is on disk |
| `-checksum` | off | Checksum every model file with none on record |
| `-ea` | off | Retry early access versions refused before; with `-all`, also do a normal run |
| `-yes` | off | Go ahead without asking |
| `-n` | off | Show the plan; change nothing |
| `-max` | `1000` | Refuse to act on a likes list longer than this |
| `-state` | `<models>/.civitai-likes.json` | State file |
| `-token-file` | `~/.config/civitai/token` | File holding the API token |
| `-api` | `https://civitai.com` | Civitai base URL |
| `-config` | `~/.config/civitai-likes/config` | Config file with default flags |
| `-print-config` | | Print the settings in effect, in config file form, and exit |

`-rename-only` cannot be combined with `-previews-only` or `-ea`, and `-previews-only` cannot be combined with `-ea`.

A run with `-base` lists the other base models found in your likes with the tag to pass, and says per selected base model how many files are on disk and how many are to download. `civitai-likes -base all -n` prints every tag in your likes.

### Config file

Flags you always use can go in `~/.config/civitai-likes/config` (on macOS, `~/Library/Application Support/civitai-likes/config`), one `flag = value` per line, with `#` comments. A bare flag name means `true`. Flags given on the command line win. To start one with every flag and its help text:

```sh
mkdir -p ~/.config/civitai-likes
civitai-likes -print-config > ~/.config/civitai-likes/config
```

```ini
# example
models   = /mnt/ai/ComfyUI/models
base     = flux1_dev,sdxl1
tag      = folder
previews = 3
```

### File names

Files are saved with their base model tag in the name, so loader lists group by base model. `-tag` picks the form:

| `-tag` | Result |
| --- | --- |
| `front` | `loras/flux1_dev_some_lora.safetensors` |
| `end` | `loras/some_lora_flux1_dev.safetensors` |
| `dot` | `loras/some_lora.flux1_dev.safetensors` |
| `folder` | `loras/flux1_dev/some_lora.safetensors` |
| `none` | `loras/some_lora.safetensors` (Civitai's name, untouched) |

Files saved under any of these forms are recognised, so you can change `-tag` later and run `-rename-only` to move existing files to the new form. Previews and info files beside a model (`.preview.*`, `.preview-N.*`, `.civitai.info`, `.json`, `.txt`) move with it. A file in one tag folder is moved to its own, never nested (`loras/sdxl1/x` for a Pony file becomes `loras/pony/x`). A `.gguf` diffusion model filed under `checkpoints` is moved to `diffusion_models`, where ComfyUI's GGUF loader looks for it.

Tags are short forms of Civitai's base model names: `Flux.1 D` is `flux1_dev`, `Flux.1 S` is `flux1_schnell`, `SDXL 1.0` is `sdxl1`, `SD 1.5` is `sd1`, `Krea 2` is `krea2`, `Pony V7` is `pony_v7`, and so on. Upscalers work with any base model and are never tagged.

### What it does

- **Versions.** A liked model with versions for several base models gets one file per selected base model, the newest version for each.
- **Placement.** Each Civitai model type goes to its ComfyUI folder: checkpoints, `loras` (LORA, LoCon, DoRA), `embeddings`, `vae`, `controlnet`, `upscale_models`, `hypernetworks`, `animatediff_models`.
- **Duplicate checks.** Before downloading, each file is checked against the whole models tree, whatever folder it was filed under and whichever `-tag` form it has: first by name for every version, then by size and SHA-256 to catch a copy saved under another name. Only what is missing is downloaded, and nothing is ever overwritten. Files that share a name and size but can't be told apart without checksums are left alone and listed, never guessed.
- **Previews.** Each model file gets the first still image from its Civitai page saved beside it as `<name>.preview.<ext>`, safe-for-work images first. With `-previews N`, extras are saved as `<name>.preview-2.<ext>` and so on; rename the one you want to `<name>.preview.<ext>` to make it the icon. Files you already have get previews too, unless one is already there.
- **Other model files.** With `-local`, model files no liked model accounts for (in `checkpoints`, `loras`, `diffusion_models`, `unet`, `embeddings`, `controlnet`, `hypernetworks`, `vae`, `animatediff_models`) are checksummed once and looked up on Civitai, so they get tagged and previewed like the rest. With `-guess`, files Civitai does not know, such as your own trainings, are tagged from the base model their metadata names. Whatever is left without a base model is listed with the reason.
- **Checksums.** The state file records what has been seen. Downloads are always checksummed to verify them; a download that fails is kept aside as `<name>.unverified`, hidden from ComfyUI, and listed on every run. Other files are checksummed only with `-checksum` or `-local`, each once.
- **Early access.** A version Civitai refuses to hand over (401, 402, or 403, typically early access you have not paid for) is recorded in the state file rather than reported as a failure. Later runs hold it back until its early access ends, or indefinitely if Civitai gives no end date, and list it under "Refused before, held back". Run with `-ea` once you have paid for it to try again right away; on its own, `-ea` retries only those versions, and `-ea -all` also does a full run. The plan marks versions that are currently in early access with their end date.
- **Mangled names.** Files with a model extension in the middle of their name (`x.safetensors_1`), which ComfyUI does not list, are reported so you can fix them by hand.
- **Safety margins.** Downloads keep 2 GB free on the disk, retry up to four times, and abort if no data arrives for two minutes.

### State file

`<models>/.civitai-likes.json` records each model file's size, date, checksum, Civitai version, and base model, plus the early access refusals. Model Preview Popup reads it to label base models. It is safe to delete; the next run rebuilds what it can, but checksums are lost and have to be redone.

## HTTP routes

The ComfyUI pieces add these routes to the ComfyUI server, handy for scripts:

| Route | Returns |
| --- | --- |
| `GET /Lora.Tag.Picker/tags?name=<lora name>` | A LoRA's ranked tags, image count, and notes, as JSON |
| `GET /Model.Preview.Popup/list?folder=<folder>&filename=<file>` | A model's preview file names and base model |
| `GET /Model.Preview.Popup/image?folder=<folder>&filename=<file>&name=<preview>` | One preview image (only files the list route returns) |
| `GET /Model.Preview.Popup/bases` | The base model of every file in the filtered folders, plus labels |

Names are as ComfyUI lists them, relative to the model folder (for example `flux1_dev/x.safetensors`).

## Troubleshooting

- **A node or the popup does not appear.** Check that the folder is directly inside `custom_nodes/` (`custom_nodes/Lora.Tag.Picker/__init__.py`, not one level deeper), restart ComfyUI, and hard-reload the browser tab. The console log names any import error.
- **LoRA Tag Picker says "No LoRA loaders found upstream".** Wire `model` or `clip` from the last loader in the chain; the node only looks upstream of its own inputs.
- **A LoRA shows no tags.** Read the note under its name: no metadata, no captions, or not a `.safetensors` file. Nothing can be done for those.
- **LLM Prompt fails.** The error names the cause: a missing key (see the key table), a `base_url` whose host is not listed for the key (see the host table), a wrong `base_url`, or a server that did not answer within `timeout`.
- **A model is missing from the Model Library.** Its base model may be unticked under **Base models** in the popup.
- **civitai-likes says "Civitai sign-in failed".** The token is wrong or expired; make a new one.

## License

Copyright (C) 2026 Nine-Iron

Licensed under the [GNU General Public License v3.0](LICENSE) or later. You may use, modify, and redistribute this software freely, provided that anything you distribute that is built on it stays under the same license with its source available.
