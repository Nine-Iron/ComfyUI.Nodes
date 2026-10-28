"""Prompt Generator - two ComfyUI nodes that make random prompts.

Random Prompt (Template)
    Fills a template from word lists. Instant, no GPU memory, and the same
    seed always gives the same prompt.
        {a|b|c}          pick one
        {2$$a|b|c}       pick two, joined with ", "
        __subject__      a random line from a wildcard file
        __folder/name__  a wildcard file in a subfolder
    Wildcard files are plain text, one option per line, # for comments, read
    from (first match wins):
        ComfyUI/wildcards/  ComfyUI/models/wildcards/  <this folder>/wildcards/

LLM Prompt
    Has a language model write (or rewrite) a prompt. Providers:
        anthropic          Claude, through the Anthropic API
        openai             ChatGPT models, through the OpenAI API
        openai_compatible  anything that speaks the OpenAI chat API: local
                           llama.cpp server, Ollama, LM Studio, OpenRouter, ...
    API keys are never typed into the node (workflows are saved inside every
    image). They come from the environment or a keys file:
        ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENAI_COMPATIBLE_API_KEY
        or <this folder>/keys.json  {"anthropic": "...", "openai": "...", "openai_compatible": "..."}
    A key is only sent to its provider's own host (api.anthropic.com,
    api.openai.com, 127.0.0.1 for openai_compatible). Other hosts, such as a
    proxy or OpenRouter, must be listed outside the workflow:
        ANTHROPIC_KEY_HOSTS / OPENAI_KEY_HOSTS / OPENAI_COMPATIBLE_KEY_HOSTS
        or keys.json  {"key_hosts": {"openai_compatible": ["openrouter.ai"]}}

Install: copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
No dependencies beyond ComfyUI itself.
"""

import json
import os
import random
import re
import urllib.error
import urllib.parse
import urllib.request

import folder_paths

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = getattr(folder_paths, "base_path", os.path.dirname(os.path.dirname(HERE)))
WILDCARD_DIRS = [
    os.path.join(BASE, "wildcards"),
    os.path.join(folder_paths.models_dir, "wildcards"),
    os.path.join(HERE, "wildcards"),
]

# ------------------------------------------------------------------ templates

# Each theme mixes its own lists (wildcards/<theme>/) with the shared ones.
# {x|} clauses are optional, so prompts vary in length as well as content.
THEMES = {
    "scene": "{__style__ of __subject__ __setting__|__subject__ __setting__, __style__"
             "|__mood__ __style__ of __subject__ __setting__}, "
             "{__lighting__|__weather__, __lighting__}, __camera__, __color__{, __quality__|}",
    "portrait": "{__portrait/style__ of __portrait/subject__|__portrait/subject__, __portrait/style__}, "
                "__portrait/expression__, __portrait/framing__, __lighting__{, __color__|}{, __mood__ atmosphere|}",
    "landscape": "{__landscape/style__ of __landscape/terrain__ with __landscape/feature__ __landscape/time__"
                 "|__landscape/terrain__ __landscape/time__, __landscape/feature__ in the distance, __landscape/style__"
                 "|{a vast|a quiet|a dramatic|a sweeping} view of __landscape/terrain__, __landscape/feature__, __landscape/style__}, "
                 "{__weather__|__lighting__|__weather__, __lighting__}{, __color__|}",
    "fantasy": "{__fantasy/flavor__ __fantasy/style__ of __fantasy/subject__ __fantasy/setting__"
               "|__fantasy/subject__ __fantasy/setting__, __fantasy/flavor__ __fantasy/style__}, "
               "__fantasy/magic__{, __lighting__|}{, __mood__|}{, __color__|}",
    "dark fantasy": "{__dark-fantasy/style__ of __dark-fantasy/subject__ __dark-fantasy/setting__"
                    "|__dark-fantasy/subject__ __dark-fantasy/setting__, __dark-fantasy/style__}, "
                    "__dark-fantasy/omen__, {ominous|haunting|foreboding|grim|desolate} atmosphere, "
                    "{dramatic chiaroscuro|cold moonlit shadows|flickering torchlight|sickly green light|__lighting__}"
                    "{, desaturated colors|, crimson and black|}{, __weather__|}",
    "sci-fi": "{__sci-fi/style__ of __sci-fi/subject__ __sci-fi/setting__|__sci-fi/subject__ __sci-fi/setting__, __sci-fi/style__}, "
              "{__sci-fi/cosmic__|__sci-fi/cosmic__|__sci-fi/cosmic__, __sci-fi/tech__|__sci-fi/tech__}, "
              "{cold starlight|rim lighting|harsh unfiltered sunlight|glowing console light|volumetric light|__lighting__}"
              "{, __color__|}{, __camera__|}",
    "cyberpunk": "{__cyberpunk/style__ of __cyberpunk/subject__ __cyberpunk/setting__"
                 "|__cyberpunk/subject__ __cyberpunk/setting__, __cyberpunk/style__}, "
                 "__cyberpunk/tech__, __cyberpunk/atmosphere__, "
                 "{neon glow|neon pink and cyan lighting|red emergency lighting|harsh fluorescent light|holographic glow}"
                 "{, __camera__|}{, {tense|gritty|lonely|chaotic|melancholic} mood|}",
    "photo": "{__photo/style__ of __photo/subject__ __photo/setting__|__photo/subject__ __photo/setting__, __photo/style__}, "
             "{__weather__, |}__lighting__, __photo/film__{, __camera__|}",
}
THEME_NAMES = ["random"] + list(THEMES) + ["custom (use template)"]

_CHOICE = re.compile(r"\{([^{}]*)\}")
_WILD = re.compile(r"__([A-Za-z0-9_\-/\. ]+?)__")
_wild_cache = {}


def wildcard_names():
    names = set()
    for d in WILDCARD_DIRS:
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".txt"):
                    rel = os.path.relpath(os.path.join(root, f), d)
                    names.add(rel[:-4].replace(os.sep, "/"))
    return sorted(names)


def load_wildcard(name):
    """Lines of a wildcard file, or None. Names never leave the wildcard folders."""
    name = name.strip().strip("/")
    if not name or ".." in name.split("/"):
        return None
    for d in WILDCARD_DIRS:
        path = os.path.realpath(os.path.join(d, name + ".txt"))
        if not path.startswith(os.path.realpath(d) + os.sep) or not os.path.isfile(path):
            continue
        st = os.stat(path)
        hit = _wild_cache.get(path)
        if hit and hit[0] == st.st_mtime:
            return hit[1]
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        _wild_cache[path] = (st.st_mtime, lines)
        return lines
    return None


def expand(template, rng, missing, depth=0):
    """Resolve {a|b} choices and __wildcards__, innermost first, nested allowed."""
    text = template
    for _ in range(200):  # guards against templates that expand forever
        changed = False

        def choice(m):
            nonlocal changed
            changed = True
            body = m.group(1)
            count = 1
            if "$$" in body:
                head, body = body.split("$$", 1)
                try:
                    count = max(1, int(head.strip()))
                except ValueError:
                    count = 1
            opts = body.split("|")
            if count == 1:
                return rng.choice(opts)
            return ", ".join(rng.sample(opts, min(count, len(opts))))

        def wild(m):
            nonlocal changed
            name = m.group(1)
            lines = load_wildcard(name)
            if not lines:
                missing.add(name)
                return name.split("/")[-1].replace("_", " ")
            changed = True
            return rng.choice(lines)

        text = _CHOICE.sub(choice, text)
        text = _WILD.sub(wild, text)
        if not changed:
            break
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*(,\s*)+", ", ", text)
    return text.strip(" ,")


class RandomPromptTemplate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "theme": (THEME_NAMES, {"default": "random",
                                        "tooltip": "A built-in template. 'custom' uses the template box below."}),
                "template": ("STRING", {
                    "multiline": True, "default": "",
                    # The node resolves {a|b} itself, from the seed; stop the
                    # frontend from rolling them first.
                    "dynamicPrompts": False,
                    "tooltip": "{a|b|c} picks one, {2$$a|b|c} picks two, __name__ picks a line from "
                               "wildcards/name.txt. Used when theme is 'custom'; with another theme it "
                               "is added after the themed prompt."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": "randomize",
                                 "tooltip": "Same seed, same prompt."}),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "dynamicPrompts": False}),
                "suffix": ("STRING", {"default": "", "dynamicPrompts": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "prompt"
    DESCRIPTION = "Builds a random prompt from a template and wildcard word lists."

    def run(self, theme, template, seed, prefix="", suffix=""):
        rng = random.Random(seed)
        parts = []
        if theme == "random":
            theme = rng.choice(list(THEMES))
        if theme in THEMES:
            parts.append(THEMES[theme])
        if template.strip():
            parts.append(template.strip())
        if not parts:
            raise ValueError("Theme is 'custom' but the template box is empty.")
        missing = set()
        body = expand(", ".join(parts), rng, missing)
        if missing:
            print(f"[Prompt Generator] wildcard(s) not found, used the name instead: {', '.join(sorted(missing))}")
        out = ", ".join(p for p in (expand(prefix, rng, missing), body, expand(suffix, rng, missing)) if p)
        return (out,)


# ------------------------------------------------------------------------ LLM

PROVIDERS = ["anthropic", "openai", "openai_compatible"]
DEFAULTS = {
    "anthropic": ("https://api.anthropic.com", "claude-haiku-5-5"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "openai_compatible": ("http://127.0.0.1:8080/v1", ""),
}
ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
            "openai_compatible": "OPENAI_COMPATIBLE_API_KEY"}
# Extra hosts a provider's key may be sent to, beyond its default one. Only from
# the environment or keys.json, never from the workflow: workflows travel inside
# images, and a base_url in one could otherwise send your key anywhere.
ENV_HOSTS = {"anthropic": "ANTHROPIC_KEY_HOSTS", "openai": "OPENAI_KEY_HOSTS",
             "openai_compatible": "OPENAI_COMPATIBLE_KEY_HOSTS"}
LOOPBACK = {"localhost", "127.0.0.1", "::1"}

STYLES = {
    "natural sentences (Flux, Qwen, Krea, SD3)":
        "Write one image-generation prompt as 2 to 4 vivid, concrete natural-language sentences: "
        "the subject, what it is doing, the setting, lighting, composition and visual style.",
    "comma tags (SD 1.5, SDXL, Pony)":
        "Write one image-generation prompt as a single line of comma-separated tags and short phrases, "
        "most important first: subject, details, setting, lighting, style, quality tags. 20 to 40 tags.",
}
SYSTEM = ("You write prompts for text-to-image models. {style} Reply with the prompt only: no preamble, "
          "no quotes, no title, no explanation, no markdown.")


def _keys_file():
    try:
        with open(os.path.join(HERE, "keys.json"), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def api_key(provider):
    key = os.environ.get(ENV_KEYS[provider], "").strip()
    if key:
        return key
    return str(_keys_file().get(provider) or "").strip()


def _host(url):
    try:
        return (urllib.parse.urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return ""


def key_hosts(provider):
    """Hosts this provider's key may go to: its default host plus the configured extras."""
    hosts = {_host(DEFAULTS[provider][0])}
    hosts.update(h for h in os.environ.get(ENV_HOSTS[provider], "").replace(",", " ").split())
    extra = (_keys_file().get("key_hosts") or {})
    extra = extra.get(provider) if isinstance(extra, dict) else None
    if isinstance(extra, str):
        extra = [extra]
    if isinstance(extra, list):
        hosts.update(str(h) for h in extra)
    return {h.strip().rstrip(".").lower() for h in hosts if str(h).strip()}


def key_allowed(provider, url):
    """True when the key may be sent to url: an allowed host, over https unless it is this machine."""
    try:
        scheme = urllib.parse.urlsplit(url).scheme.lower()
    except ValueError:
        return False
    host = _host(url)
    return host in key_hosts(provider) and (scheme == "https" or host in LOOPBACK)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # urllib copies headers to the redirected request, the key among them.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} not followed", headers, fp)


_opener = urllib.request.build_opener(_NoRedirect)


def post_json(url, headers, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with _opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        try:
            err = json.loads(detail).get("error")
            detail = err.get("message", detail) if isinstance(err, dict) else (err or detail)
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"{url} answered HTTP {e.code}: {detail or e.reason}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach {url}: {e.reason}") from None
    except TimeoutError:
        raise RuntimeError(f"{url} did not answer within {timeout}s") from None


def clean(text):
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # reasoning models
    text = re.sub(r"^```\w*\n?|\n?```$", "", text).strip()
    text = re.sub(r"^(here(?:'s| is) (?:a|the|your) [^:\n]*prompt[^:\n]*:|prompt:)\s*", "", text, flags=re.I)
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text.strip()


def ask(provider, model, base_url, system, user, temperature, max_tokens, seed, timeout):
    default_url, default_model = DEFAULTS[provider]
    base = (base_url.strip() or default_url).rstrip("/")
    model = model.strip() or default_model
    key = api_key(provider)
    if key and not key_allowed(provider, base):
        if provider != "openai_compatible":
            raise RuntimeError(
                f"Refusing to send your {provider} API key to {base}: only {DEFAULTS[provider][0]} is "
                f"trusted. To use another host, add it to {ENV_HOSTS[provider]} or to \"key_hosts\" in "
                f"keys.json (https only).")
        key = ""  # a local server needs no key; one elsewhere must be listed to get it
    if provider == "anthropic":
        if not key:
            raise RuntimeError("No Anthropic API key: set ANTHROPIC_API_KEY or put it in keys.json.")
        data = post_json(base + "/v1/messages",
                         {"x-api-key": key, "anthropic-version": "2023-06-01"},
                         {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                          "system": system, "messages": [{"role": "user", "content": user}]}, timeout)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    if provider == "openai" and not key:
        raise RuntimeError("No OpenAI API key: set OPENAI_API_KEY or put it in keys.json.")
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens, "seed": seed % (2 ** 31)}
    if model:
        body["model"] = model
    try:
        data = post_json(base + "/chat/completions", {"Authorization": f"Bearer {key}"} if key else {},
                         body, timeout)
    except RuntimeError as e:
        if api_key(provider) and not key and any(f"HTTP {c}" in str(e) for c in (401, 403)):
            raise RuntimeError(f"{e}\nYour {ENV_KEYS[provider]} was not sent, because {_host(base)} is not "
                               f"listed in {ENV_HOSTS[provider]} or \"key_hosts\" in keys.json.") from None
        raise
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"no answer in the reply: {str(data)[:300]}")
    return (choices[0].get("message") or {}).get("content") or ""


class LLMPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (PROVIDERS, {"tooltip": "openai_compatible covers local servers (llama.cpp, "
                                                    "Ollama, LM Studio) and services like OpenRouter."}),
                "model": ("STRING", {"default": "", "tooltip": "Blank: claude-haiku-5-5 for anthropic, "
                                                               "gpt-4o-mini for openai, the server's own model "
                                                               "for openai_compatible (Ollama needs a name)."}),
                "base_url": ("STRING", {"default": "", "tooltip": "Blank uses the provider's own address; "
                                                                  "for openai_compatible the default is "
                                                                  "http://127.0.0.1:8080/v1 (llama.cpp server). "
                                                                  "Ollama: http://127.0.0.1:11434/v1"}),
                "idea": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False,
                                    "tooltip": "What the image should be about. Blank lets the model choose."}),
                "style": (list(STYLES), {"tooltip": "Prompt format that suits your image model."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 300, "min": 32, "max": 4000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": "randomize",
                                 "tooltip": "A new seed asks for a new prompt. ComfyUI reuses the last "
                                            "answer while every input stays the same."}),
                "timeout": ("INT", {"default": 120, "min": 5, "max": 900}),
            },
            "optional": {
                "seed_prompt": ("STRING", {"forceInput": True,
                                           "tooltip": "A prompt to rewrite and enrich, e.g. from "
                                                      "Random Prompt (Template)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "prompt"
    DESCRIPTION = "Has a language model (Claude, ChatGPT, or a local/compatible server) write an image prompt."

    def run(self, provider, model, base_url, idea, style, temperature, max_tokens, seed, timeout, seed_prompt=None):
        system = SYSTEM.format(style=STYLES[style])
        asks = []
        if seed_prompt and seed_prompt.strip():
            asks.append(f"Rewrite and enrich this rough prompt, keeping its subject and intent:\n{seed_prompt.strip()}")
        if idea.strip():
            asks.append(f"The image should be about: {idea.strip()}")
        if not asks:
            asks.append("Invent an original, visually striking image of your own choosing.")
        asks.append(f"(variation {seed % 100000})")  # nudges models that ignore the seed parameter
        if provider == "anthropic":
            temperature = min(temperature, 1.0)  # Anthropic's range is 0 to 1
        text = clean(ask(provider, model, base_url, system, "\n\n".join(asks),
                         temperature, max_tokens, seed, timeout))
        if not text:
            raise RuntimeError("The model returned an empty prompt.")
        return (text,)


NODE_CLASS_MAPPINGS = {"RandomPromptTemplate": RandomPromptTemplate, "LLMPrompt": LLMPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"RandomPromptTemplate": "Random Prompt (Template)", "LLMPrompt": "LLM Prompt"}
