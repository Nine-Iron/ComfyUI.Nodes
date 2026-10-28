# ComfyUI.Nodes

Small tools for working with models and prompts in [ComfyUI](https://github.com/comfyanonymous/ComfyUI):

| Folder | What it is | What it does |
| --- | --- | --- |
| [`Lora.Tag.Picker/`](#lora-tag-picker) | Custom node | Reads the training tags out of the LoRAs you have loaded and adds the ones you pick to your prompt. |
| [`Prompt.Generator/`](#prompt-generator) | Custom nodes | Makes random prompts, from wildcard templates or by asking a language model (Claude, ChatGPT, or a local server). |
| [`Model.Preview.Popup/`](#model-preview-popup) | Frontend extension (no nodes) | Shows a large preview popup when you hover a model in the Model Library sidebar, with the prompt and settings each preview was made with, and lets you hide base models you don't use. |
| [`Checkpoint.Sweep/`, `Sampler.Sweep/`, `Lora.Sweep/`](#sweeps) | Custom nodes | Change the checkpoint, sampler settings or LoRA strengths on every run of a batch, through every combination or at random. |
| [`moFlo/`](#moflo) | Command-line tool (Go) | Downloads the models you have liked on Civitai, with preview images, skipping anything you already have. |

Each one works on its own, and they work well together: moFlo saves previews, their generation settings, and base-model tags in the layout Model Preview Popup reads, and LoRA Tag Picker reads the tags inside the LoRAs it downloads.

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [LoRA Tag Picker](#lora-tag-picker)
- [Prompt Generator](#prompt-generator)
- [Model Preview Popup](#model-preview-popup)
- [Sweeps](#sweeps)
- [moFlo](#moflo)
- [HTTP routes](#http-routes)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Requirements

- **ComfyUI pieces:** a recent ComfyUI with the new frontend (the Model Library sidebar and DOM widgets). Nothing else to install: they use only the Python standard library and what ComfyUI already ships.
- **moFlo:** Go 1.21 or later to build it (standard library only), and a Civitai API key.

## Install

The repository holds several independent folders, so do not clone it straight into `custom_nodes/`. Clone it anywhere, then copy or symlink the folders you want:

```sh
git clone https://github.com/Nine-Iron/ComfyUI.Nodes.git
cd ComfyUI.Nodes

cp -r Lora.Tag.Picker     ~/ComfyUI/custom_nodes/
cp -r Prompt.Generator    ~/ComfyUI/custom_nodes/
cp -r Model.Preview.Popup ~/ComfyUI/custom_nodes/
cp -r Checkpoint.Sweep Sampler.Sweep Lora.Sweep ~/ComfyUI/custom_nodes/
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
| `theme` | A built-in template (`scene`, `portrait`, `landscape`, `fantasy`, `dark fantasy`, `sci-fi`, `cyberpunk`, `photo`), `random` to pick one, or `custom (use template)`. |
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

Choices and wildcards can be nested. Wildcard files are plain text, one option per line, with `#` for comments. They are looked up in `ComfyUI/wildcards/`, then `ComfyUI/models/wildcards/`, then the node's own `wildcards/` folder, first match wins, so you can override a bundled list by putting a file of the same name in either of the first two. The bundled shared lists are `camera`, `color`, `lighting`, `mood`, `quality`, `setting`, `style`, `subject`, and `weather`. Each theme except `scene` also has its own folder of lists (`portrait/`, `landscape/`, `fantasy/`, `dark-fantasy/`, `sci-fi/`, `cyberpunk/`, `photo/`), for example `__sci-fi/cosmic__` for black holes, twin suns and extra moons; override one with a file at the same path, such as `ComfyUI/wildcards/sci-fi/cosmic.txt`. A missing wildcard is replaced by its name and reported in the console. Edited files are picked up without a restart.

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

Hover a model in the Model Library sidebar to see a large popup with the preview images saved beside the model file, its base model, its name, and the trigger phrase, author, and description from its metadata when present.

### Model names

ComfyUI labels a model with the title its trainer wrote into the file, which is often `ComfyUI1`, `last`, or `tst_01`. For files moFlo knows (liked models, and others it found on Civitai by checksum), the sidebar and the popup show the model's name on Civitai instead, without the base model list authors put at its end: `1950's (Technicolor) style XL + F1D + Illu + Pony + zit + zib` shows as `1950's (Technicolor) style`, `Crystals [FLUX+SDXL+PONY+1.5]` as `Crystals`. A lone base word that may be part of the name (`Juggernaut XL`) stays. The popup adds the version and the trainer's title in small print. For other files with a title that says nothing, the file name is shown. Nothing in the file changes.

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

### Generation settings

When the settings a picture was made with are known, a column beside it shows them: prompt, negative prompt, model, sampler, scheduler, steps, CFG, seed, size, and LoRAs with their weights. It follows the picture you are looking at. **Copy** copies the prompt or negative prompt; **Copy all** copies everything as one A1111-style block. Models with no settings for any of their pictures keep the narrow popup.

The settings come from, first match wins:

| Source | Where it comes from |
| --- | --- |
| `<name>.preview.json`, `<name>.preview-N.json` | Saved by moFlo beside each preview it downloads: what Civitai records for that image. For previews downloaded before, run `moflo -previews-only -preview-info` once. |
| Metadata inside the picture | A1111 / Forge `parameters` (PNG text, or JPEG/WebP EXIF), or the ComfyUI prompt graph in a PNG or WebP saved by ComfyUI. Your own renders dropped beside a model show their settings this way. |

A plain `<name>.<ext>` preview only uses metadata inside the picture, since `<name>.json` beside a model is the model's own info file. Civitai's resized images do not keep embedded metadata, which is why moFlo saves the settings separately.

### Base models

The top of the popup names the model's base model (for example **Flux.1 D**). It is worked out in this order:

1. The moFlo state file (`<models>/.moflo.json`), for files moFlo downloaded or identified.
2. A tag in the file or folder name, in any of the forms moFlo writes: `flux1_dev_x`, `x_flux1_dev`, `x.flux1_dev`, or a `flux1_dev/` folder.
3. The base model the file itself records (`ss_base_model_version` or `modelspec.architecture`).
4. A guess from the tensor names in the file header. The label's tooltip says when it was guessed.

Anything left over is **Unknown**.

The **Base models** button in the popup opens a list of every base model found, with a file count for each. Untick a base model to hide its files from the Model Library and from loader dropdowns in the graph. A model already chosen in a node stays visible there. The filter applies to `checkpoints`, `loras`, `diffusion_models`, `unet`, `embeddings`, `controlnet`, and `hypernetworks`; text encoders, VAEs, and upscalers are always shown. The choice is saved in your ComfyUI user settings, so it follows you across browsers.

## Sweeps

Three controller nodes that change settings on other nodes right before each queued run, so one **Run × N** tries N different setups:

| Node | Sets | On |
| --- | --- | --- |
| **Checkpoint Sweep** (loaders) | `ckpt_name`, `unet_name` | Load Checkpoint, Load Diffusion Model, Unet Loader (GGUF) |
| **Sampler Sweep** (sampling) | `sampler_name`, `scheduler`, `steps`, `cfg` | KSampler |
| **LoRA Strength Sweep** (loaders) | `strength_model`, `strength_clip` | LoRA loaders |

Each setting has a mode: **off** (leave it alone), **sweep** (step through the list, one value per run) or **random** (pick one each run). The values are written into the target node's own widgets, so the workflow saved in each image shows what was really used, and a filename prefix like `%CheckpointLoaderSimple.ckpt_name%_%KSampler.steps%` labels the images.

- **One counter for all of them.** Every sweep node in a workflow shares one counter and together they cover every combination. The status line shows the total ("12 combinations, set Run count to 12") and what the next run will use. **restart sweep** starts over from the first combination.
- **The model changes slowest.** Checkpoint Sweep always comes first, so each model is loaded once and everything else is swept with it, rather than reloading a model every run. Among the other sweep nodes, the one highest on the canvas changes slowest.
- **Which nodes it drives.** With nothing wired in, a sweep node drives every matching node in the workflow. To drive just one (say, the base checkpoint and not a refiner), wire that node's outputs through the sweep node: Checkpoint Sweep passes MODEL / CLIP / VAE through, Sampler Sweep passes LATENT through.
- **Lists.** Names: a comma list, or patterns such as `sdxl1/*`, `*juggernaut*`, `dpmpp_*`. `*` or empty means every one the loader lists. Numbers: `20,30,40` or ranges `20-40:5` (from-to:step).

### Checkpoint Sweep

`checkpoint_mode` / `checkpoints` drive `ckpt_name`; `diffusion_mode` / `diffusion_models` drive `unet_name` (off by default). For a workflow with only Load Diffusion Model, set `checkpoint_mode` to off and `diffusion_mode` to sweep.

`*` respects Model Preview Popup's base filter: base models you have hidden there are not swept.

Every new model is a full load, which takes a while on small machines such as a Jetson, and **random** may load a different model every run. Prefer **sweep** for checkpoints, and keep the list short, for example a few patterns rather than `*` across a large folder.

## moFlo

A command-line tool that downloads the models you have liked on Civitai into your ComfyUI models folder, without ever fetching one you already have. It also saves preview images in the layout Model Preview Popup reads, can tag file names with their base model, and can identify model files you got elsewhere.

moFlo was called civitai-likes. Its first run carries everything over: the config (`~/.config/civitai-likes/config.json`, or the older `config` file beside it) is imported into `~/.config/moflo/config.json`, and the state file `<models>/.civitai-likes.json` is moved to `<models>/.moflo.json`. Model Preview Popup reads either state file, so nothing breaks in between. The API token stays where it was (`CIVITAI_TOKEN` or `~/.config/civitai/token`).

### Setup

Build it (Go 1.21+, standard library only):

```sh
cd "moFlo"
go build -o moflo moflo.go
```

A prebuilt `moflo.linux.arm64` binary is included for 64-bit ARM Linux. To build for another platform, set `GOOS` and `GOARCH`, for example `GOOS=windows GOARCH=amd64 go build -o moflo.exe moflo.go`.

Create an API key at <https://civitai.com/user/account>, then either export it or save it to a file:

```sh
export CIVITAI_TOKEN=...
# or
mkdir -p ~/.config/civitai && echo '...' > ~/.config/civitai/token
```

The environment variable wins when both are set.

### Usage

```sh
moflo -n                 # show what would happen, change nothing
moflo                    # first run: pick base models; then show the plan, ask, download
moflo -yes               # no question asked
moflo -reconfig          # pick again which base models are downloaded
moflo -base pony         # only Pony models, this run
moflo -base sdxl1,sd1    # only these base models, this run
moflo -set-conf -previews 3 -tag folder   # save these as the defaults in the config file
moflo -previews 4        # keep up to 4 preview images per model
moflo -previews-only     # only fetch missing previews for models you have
moflo -previews-only -preview-info   # also save generation settings for previews you already have
moflo -rename            # also rename files you already have to the tagged form
moflo -rename-only       # only rename, no downloads or previews
moflo -rename-only -local          # also rename files that are not liked models, looked up by checksum
moflo -rename-only -local -guess   # and tag the ones Civitai does not know from their metadata
moflo -tag folder -rename-only     # switch to loras/flux1_dev/x.safetensors style
moflo -latest            # also fetch the newest version when you hold an older one
moflo -checksum          # checksum every model file now (slow, once)
moflo -ea                # only retry early access versions refused before
moflo -ea -all           # retry those, and do everything a normal run does
```

Start with `-n` to see the plan before anything is written. Ctrl+C stops cleanly at any point; checksums done so far are kept. Only one run works on a models folder at a time: a second one stops and names the run that is going (a `-n` run goes ahead with a warning).

| Flag | Default | Meaning |
| --- | --- | --- |
| `-models` | `~/ComfyUI/models` | ComfyUI models folder |
| `-base` | from the config | Base models to handle this run (downloads, renames, previews), comma separated, as tags (`flux1_dev`) or Civitai names (`"Flux.1 D"`); `all` handles every base. Without it, the config's choice applies. |
| `-tag` | `front` | Where the base model tag goes in file names; see [File names](#file-names). |
| `-name` | `auto` | What downloaded files are called: `auto`, `civitai` or `model`; see [File names](#file-names). |
| `-previews` | `1` | Preview images to keep per model file (`0` for none) |
| `-previews-only` | off | Only fetch missing previews for liked models already on disk |
| `-preview-info` | off | Also save generation settings for preview pictures already on disk that have none; see [Previews](#what-it-does) |
| `-rename` | off | Also rename files you already have to the tagged form |
| `-rename-only` | off | Only rename; no downloads, no previews |
| `-local` | off | Also look up model files that are not liked models on Civitai by checksum, so they get tagged and previewed too |
| `-guess` | off | Tag model files Civitai does not know from the base model their own metadata names |
| `-latest` | off | Fetch the newest version even when an older one is on disk |
| `-checksum` | off | Checksum every model file with none on record (files that could be a liked file, or a copy of another, are checksummed anyway) |
| `-ea` | off | Retry early access versions refused before; with `-all`, also do a normal run |
| `-yes` | off | Go ahead without asking |
| `-n` | off | Show the plan; change nothing |
| `-max` | `1000` | Refuse to act on a likes list longer than this |
| `-state` | `<models>/.moflo.json` | State file |
| `-token-file` | `~/.config/civitai/token` | File holding the API token |
| `-api` | `https://civitai.com` | Civitai base URL |
| `-config` | `~/.config/moflo/config.json` | The [config file](#config-file) |
| `-set-conf` | | Save the flags given on the command line into the config file as the new defaults (`-base` sets which base models are downloaded). `-n` is never saved. |
| `-reconfig` | | List the base models in your likes again and ask which to download |
| `-print-config` | | Print the config as it applies to this run, and exit |
| `-version` | | Print the version and exit |

`-rename-only` cannot be combined with `-previews-only` or `-ea`, and `-previews-only` cannot be combined with `-ea`.

Every run lists the other base models found in your likes that are not downloaded, and says per base model how many files are on disk and how many are to download.

### Config file

`~/.config/moflo/config.json` (on macOS, `~/Library/Application Support/moflo/config.json`) holds the default of every flag, which base models are downloaded, and per base model the liked models with what became of each. It is created on the first run.

**First run.** The base models in your likes are listed by number; answer with the ones to download, `1 2 3` or `1,2,3` (ranges like `2-5`, `all` and `none` work too; Enter means all). The answer is saved; change it in the file, or run `-reconfig` to be asked again. With `-yes` on the first run nothing can be asked, so every base model is set to download.

**New base models.** When a liked model brings a base model the file has no choice for, `newBaseModels` decides: `ask` (the default) lists them, marked new, with the others; `download` or `skip` decides without asking. A run with `-yes` cannot ask, so it leaves new ones off and says so.

**Flags.** A flag on the command line applies to that run only. Add `-set-conf` to save the flags given into the file as new defaults: `moflo -set-conf -previews 3 -tag folder`. Add `-n` to save without downloading anything.

The file looks like this (the `help` section, written by the tool, explains every setting):

```json
{
  "newBaseModels": "ask",
  "settings": {
    "models": "/home/jetson/ComfyUI/models",
    "tag": "front",
    "previews": 1,
    "latest": false,
    "yes": false
  },
  "bases": {
    "flux1_dev": {
      "name": "Flux.1 D",
      "download": true,
      "previews": 3,
      "latest": null,
      "models": {
        "10": { "name": "Amanita", "type": "LORA", "download": true, "status": "on disk",
                "version": "v1-flux", "file": "loras/flux1_dev_Amanita.safetensors", "updated": "2026-10-16" },
        "40": { "name": "Early Bird", "type": "LORA", "download": true,
                "status": "held back: refused Oct 16 (403 ...); early access ends Jan 1 2099" }
      }
    },
    "sdxl1": {
      "name": "SDXL 1.0",
      "download": true,
      "previews": null,
      "latest": null,
      "models": {
        "10": { "name": "Amanita", "type": "LORA", "download": false,
                "status": "skipped: not downloaded: \"download\" is false for it in the config" }
      }
    },
    "pony": { "name": "Pony", "download": false, "previews": null, "latest": null, "models": {
        "10": { "name": "Amanita", "type": "LORA", "download": true, "status": "base model not downloaded" } } }
  }
}
```

| Part | Yours to set | Written by the tool |
| --- | --- | --- |
| `settings` | Every flag's default | Settings missing from the file, at their built-in default |
| `newBaseModels` | `ask`, `download` or `skip` | |
| `bases.<base>.download` | `true` / `false` (`null`: not chosen yet, asked about) | Your answer to the numbered list |
| `bases.<base>.previews`, `.latest` | A number / `true` or `false` for this base only; `null` uses the setting. A `-previews` or `-latest` flag on the command line wins. | |
| `bases.<base>.models.<id>.download` | `false` keeps this liked model from being downloaded for this base | `true` for a model met for the first time |
| `bases.<base>.models.<id>` other fields | | Name, type, status, version, file, date of the last change |

Models are keyed by their Civitai model id, and listed under every base model they have a version for, including base models you do not download, so you can switch one off before turning its base on. To get one LoRA for one base model only, set `"download": false` on it under the others. Upscalers and other models with no base model are under `none`. A model you unlike is kept and marked `no longer liked`.

The tool only ever rewrites its own parts. If you edit the file while a run is going, your edits are kept; they take effect on the next run. A mistake in the file stops the run with its line and column. A config from civitai-likes (see above) is imported the first time and no longer read.

### File names

**The name.** `-name` picks it:

| `-name` | Result |
| --- | --- |
| `auto` | Civitai's file name, unless it says nothing or another liked model uses the same one. Then the model's name and version: `World of Filaments (V2 "Glass")` becomes `World_of_Filaments_V2_Glass.safetensors`. A name says nothing when it is a trainer's or ComfyUI's default (`ComfyUI1`, `ComfyUI 1`, `lora`, `last`, `pytorch_lora_weights`), a UUID, a number, a base model's name (`sdxl`), or when what is left once counters, versions and base model words are taken off is five letters or fewer (`tst_01`, `kids4`, `trip4`). |
| `civitai` | Civitai's file name, always. |
| `model` | The model's name and version, always. |

Built names start from the model's name without the base model list at its end (see [Model names](#model-names)), then keep letters of any script, digits, `-` and `.`; spaces and other characters become single underscores, apostrophes, brackets, quotes and emoji are dropped, and names are cut at 120 characters. Files you already have are found under either name. With `-rename` they are renamed to the new one, previews included; the plan says why. Model files that are not liked models but that Civitai knows by their checksum (`-local`, or found by an earlier `-local` run) are named the same way; for those found before this version, the name is built from the model name alone. Saved ComfyUI workflows refer to LoRAs and checkpoints by file name, so `-name model -rename` on a big collection breaks those workflows; `auto` only touches names that were useless anyway.

**The base model tag.**

Files are saved with their base model tag in the name, so loader lists group by base model. `-tag` picks the form:

| `-tag` | Result |
| --- | --- |
| `front` | `loras/flux1_dev_some_lora.safetensors` |
| `end` | `loras/some_lora_flux1_dev.safetensors` |
| `dot` | `loras/some_lora.flux1_dev.safetensors` |
| `folder` | `loras/flux1_dev/some_lora.safetensors` |
| `none` | `loras/some_lora.safetensors` (Civitai's name, untouched) |

Files saved under any of these forms are recognised, so you can change `-tag` later and run `-rename-only` to move existing files to the new form. Previews, their settings files, and info files beside a model (`.preview.*`, `.preview-N.*`, `.civitai.info`, `.json`, `.txt`) move with it. A file in one tag folder is moved to its own, never nested (`loras/sdxl1/x` for a Pony file becomes `loras/pony/x`). A `.gguf` diffusion model filed under `checkpoints` is moved to `diffusion_models`, where ComfyUI's GGUF loader looks for it.

Tags are short forms of Civitai's base model names: `Flux.1 D` is `flux1_dev`, `Flux.1 S` is `flux1_schnell`, `SDXL 1.0` is `sdxl1`, `SD 1.5` is `sd1`, `Krea 2` is `krea2`, `Pony V7` is `pony_v7`, and so on. Upscalers work with any base model and are never tagged.

### What it does

- **Versions.** A liked model with versions for several base models gets one file per selected base model, the newest version for each.
- **Placement.** Each Civitai model type goes to its ComfyUI folder: checkpoints, `loras` (LORA, LoCon, DoRA), `embeddings`, `vae`, `controlnet`, `upscale_models`, `hypernetworks`, `animatediff_models`.
- **Duplicate checks.** Before downloading, each file is checked against the whole models tree, whatever folder it was filed under and whichever `-tag` form it has: first by name for every version, then by size and SHA-256 to catch a copy saved under another name. Only what is missing is downloaded, and nothing is ever overwritten. A file that could be a liked file under another name (the same size, or the same name for two liked files) needs a checksum to settle it. moFlo asks first: it lists each file it wants to checksum with the reason, the total size, and what happens without it, and checksums only on yes (`-yes` says yes; no answer counts as no). Each file is checksummed once and the result kept, so the question does not come back for it. Said no, such a file is left alone and so is the liked file it may be, so nothing is downloaded twice; the next run asks again. `-checksum` and `-local` are themselves the request, so they do not ask.
- **Copies and leftovers.** The same file stored twice in one models folder (as two runs with different `-tag` leave behind, `loras/sdxl1/x` and `loras/sdxl1_x`, say) is found by checksum (asked first, as above) and cut down to one: the copy named the way `-tag` names files now is kept (else the one with the most previews, else the oldest), and its twin's previews and settings files move over when it lacks them. A partial download (`.part`) whose finished file is already there is removed, and so is a failed download (`.unverified`) of a file now on disk and checksummed, under any `-tag` form. Both are listed in the plan with the space they free, and happen only when you say yes. A file a symbolic link points to is never removed or renamed, and copies in different folders (`loras` and `checkpoints`) are left alone.
- **Previews.** Each model file gets the first still image from its Civitai page saved beside it as `<name>.preview.<ext>`, safe-for-work images first. With `-previews N`, extras are saved as `<name>.preview-2.<ext>` and so on; rename the one you want to `<name>.preview.<ext>` to make it the icon (rename its `.json` with it, or the settings stay with the old name). Files you already have get previews too, unless one is already there. Beside each preview goes `<name>.preview.json` (`.preview-N.json` for extras) with what Civitai records about how that picture was made, which Model Preview Popup shows; when the model listing carries none, the version's image gallery is checked. Previews saved before this existed get theirs with `-preview-info`: which Civitai image a picture is cannot be told from its name, so up to 16 of the model's images are downloaded again and matched byte for byte. Pictures that match none (your own, say) are counted and left alone.
- **Other model files.** With `-local`, model files no liked model accounts for (in `checkpoints`, `loras`, `diffusion_models`, `unet`, `embeddings`, `controlnet`, `hypernetworks`, `vae`, `animatediff_models`) are checksummed once and looked up on Civitai, so they get tagged and previewed like the rest. With `-guess`, files Civitai does not know, such as your own trainings, are tagged from the base model their metadata names. Whatever is left without a base model is listed with the reason.
- **Checksums.** The state file records what has been seen. Downloads are always checksummed to verify them; a download that fails is kept aside as `<name>.unverified`, hidden from ComfyUI, and downloaded again on the next run; the bad copy goes once a good one checks out. Other files are checksummed when they might be a liked file or a copy (see above), with `-checksum`, or with `-local`, each once.
- **Early access.** A version Civitai refuses to hand over (401, 402, or 403, typically early access you have not paid for) is recorded in the state file rather than reported as a failure. Later runs hold it back until its early access ends, or indefinitely if Civitai gives no end date, and list it under "Refused before, held back". Run with `-ea` once you have paid for it to try again right away; on its own, `-ea` retries only those versions, and `-ea -all` also does a full run. The plan marks versions that are currently in early access with their end date.
- **Mangled names.** Files with a model extension in the middle of their name (`x.safetensors_1`), which ComfyUI does not list, are reported so you can fix them by hand.
- **Safety margins.** Downloads keep 2 GB free on the disk, retry up to four times, and abort if no data arrives for two minutes.

### State file

`<models>/.moflo.json` records each model file's size, date, checksum, Civitai version, base model, and title (the model's name on Civitai without the base model list at its end), plus the early access refusals. Model Preview Popup reads it to label base models and name models. It is safe to delete; the next run rebuilds what it can, but checksums are lost and have to be redone.

## HTTP routes

The ComfyUI pieces add these routes to the ComfyUI server, handy for scripts:

| Route | Returns |
| --- | --- |
| `GET /Lora.Tag.Picker/tags?name=<lora name>` | A LoRA's ranked tags, image count, and notes, as JSON |
| `GET /Model.Preview.Popup/list?folder=<folder>&filename=<file>` | A model's preview file names, the generation settings of each picture that has them, and its base model |
| `GET /Model.Preview.Popup/image?folder=<folder>&filename=<file>&name=<preview>` | One preview image (only files the list route returns) |
| `GET /Model.Preview.Popup/bases` | The base model of every file in the filtered folders, plus labels |

Names are as ComfyUI lists them, relative to the model folder (for example `flux1_dev/x.safetensors`).

## Troubleshooting

- **A node or the popup does not appear.** Check that the folder is directly inside `custom_nodes/` (`custom_nodes/Lora.Tag.Picker/__init__.py`, not one level deeper), restart ComfyUI, and hard-reload the browser tab. The console log names any import error.
- **LoRA Tag Picker says "No LoRA loaders found upstream".** Wire `model` or `clip` from the last loader in the chain; the node only looks upstream of its own inputs.
- **A LoRA shows no tags.** Read the note under its name: no metadata, no captions, or not a `.safetensors` file. Nothing can be done for those.
- **LLM Prompt fails.** The error names the cause: a missing key (see the key table), a `base_url` whose host is not listed for the key (see the host table), a wrong `base_url`, or a server that did not answer within `timeout`.
- **A model is missing from the Model Library.** Its base model may be unticked under **Base models** in the popup.
- **moFlo says "Civitai sign-in failed".** The token is wrong or expired; make a new one.

## License

Copyright (C) 2026 Nine-Iron

Licensed under the [GNU General Public License v3.0](LICENSE) or later. You may use, modify, and redistribute this software freely, provided that anything you distribute that is built on it stays under the same license with its source available.
