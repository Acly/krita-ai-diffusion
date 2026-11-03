from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PyQt5.QtCore import QByteArray, QUrl
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from ..settings import settings
from ..util import client_logger as log
from ..util import plugin_dir, user_data_dir
from .network import NetworkError, RequestManager

# Ollama runs as its own service, independent of the ComfyUI connection, so it gets
# its own request manager rather than borrowing the one owned by the client.
_requests = RequestManager()


class EnhanceTask(Enum):
    enhance = "enhance"
    rewrite = "rewrite"
    detail = "detail"
    variations = "variations"
    instruct = "instruct"
    describe = "describe"
    structure = "structure"


_task_instructions = {
    EnhanceTask.enhance: (
        "Expand and improve the following prompt. Keep the subject, composition and"
        " intent unchanged, but make it more specific and visually detailed."
    ),
    EnhanceTask.rewrite: (
        "Write a new prompt for the following idea. You may restructure it completely,"
        " but stay true to the subject the user described."
    ),
    EnhanceTask.detail: (
        "Add missing detail to the following prompt. Do not change or remove anything"
        " that is already there - only append what is missing (lighting, materials,"
        " camera, background)."
    ),
    EnhanceTask.instruct: (
        "Apply the following change to the prompt below. Keep everything the change does"
        " not concern exactly as it is, and answer with the complete modified prompt."
        " Delete whatever contradicts the change instead of keeping both versions, and"
        " never state what was removed - a removed thing is simply absent from the"
        " answer.\n\nChange to apply: {instruction}"
    ),
    EnhanceTask.describe: (
        "Look at the attached image and write an image generation prompt which would"
        " recreate it. Describe only what is actually visible - subject, appearance,"
        " clothing, pose, expression, setting, props, lighting, colours and camera angle."
        " Do not invent anything that is not in the image, do not mention that this is an"
        " image or a description, and answer with the prompt only.{instruction}"
    ),
    EnhanceTask.structure: (
        "Sort the following prompt into labelled sections. Keep every detail of the prompt,"
        " add nothing and drop nothing."
    ),
    EnhanceTask.variations: (
        "Write {count} different variations of the following prompt. Vary pose, setting,"
        " lighting and camera, but keep the same subject. Output one variation per line,"
        " nothing else - no numbering, no blank lines, no commentary."
    ),
}


def url() -> str:
    value = settings.ollama_url.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value


async def list_models() -> list[str]:
    result = await _requests.get(f"{url()}/api/tags", timeout=10)
    models = result.get("models") or []
    return sorted(m["name"] for m in models if m.get("name"))


# Tasks which modify an existing prompt rather than inventing one. A high temperature
# makes the model drift off and keep contradicting tags around, so it gets capped.
_conservative_tasks = {
    EnhanceTask.instruct,
    EnhanceTask.detail,
    EnhanceTask.describe,
    EnhanceTask.structure,
}
_conservative_temperature = 0.6


def temperature_for(task: EnhanceTask) -> float:
    if task in _conservative_tasks:
        return min(settings.ollama_temperature, _conservative_temperature)
    return settings.ollama_temperature


def _request_body(
    prompt: str,
    system: str,
    model: str,
    temperature: float | None,
    stream: bool,
    seed: int | None = None,
    keep_alive: int | None = None,
    images: list[str] | None = None,
):
    options: dict = {
        "temperature": settings.ollama_temperature if temperature is None else temperature,
        "num_predict": settings.ollama_max_tokens,
    }
    if seed is not None:
        options["seed"] = seed
    data = {
        "model": model or settings.ollama_model,
        "prompt": prompt,
        "stream": stream,
        "think": False,  # ignored by models without a thinking mode
        "keep_alive": settings.ollama_keep_alive if keep_alive is None else keep_alive,
        "options": options,
    }
    if system:  # without it the model keeps whatever SYSTEM its Modelfile defines
        data["system"] = system
    if images:  # base64 PNG, only understood by vision models
        data["images"] = images
    return data


async def generate(
    prompt: str,
    *,
    system: str = "",
    model: str = "",
    temperature: float | None = None,
    timeout: float | None = None,
    images: list[str] | None = None,
) -> str:
    # The reply is consumed as a single JSON document - streaming would produce
    # newline-delimited JSON which the request manager cannot parse.
    data = _request_body(prompt, system, model, temperature, stream=False, images=images)
    timeout = timeout or settings.ollama_timeout
    result = await _requests.http("POST", f"{url()}/api/generate", data, timeout=timeout)
    return clean_response(result.get("response", ""))


class Generation:
    """A streaming /api/generate call. Ollama sends one JSON document per token, so
    progress can be reported while the model is still writing."""

    def __init__(self, on_progress: Callable[[str], None] | None = None):
        self._net = QNetworkAccessManager()
        self._reply: QNetworkReply | None = None
        self._buffer = b""
        self._text = ""
        self._on_progress = on_progress
        self._future: asyncio.Future[str] | None = None

    @property
    def text(self):
        return self._text

    async def run(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str = "",
        temperature: float | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        keep_alive: int | None = None,
        images: list[str] | None = None,
    ) -> str:
        self._text = ""
        self._buffer = b""
        data = _request_body(prompt, system, model, temperature, True, seed, keep_alive, images)
        request = QNetworkRequest(QUrl(f"{url()}/api/generate"))
        request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
        # applies to inactivity, not to the total duration - a slow model is fine as long
        # as it keeps emitting tokens
        request.setTransferTimeout(int((timeout or settings.ollama_timeout) * 1000))

        self._future = asyncio.get_running_loop().create_future()
        reply = self._net.post(request, QByteArray(json.dumps(data).encode("utf-8")))
        assert reply is not None, "Failed to start Ollama request"
        self._reply = reply
        reply.readyRead.connect(self._read)
        reply.finished.connect(self._finish)
        return await self._future

    def cancel(self):
        if self._reply is not None and self._reply.isRunning():
            self._reply.abort()

    def _read(self):
        if self._reply is None:
            return
        self._buffer += self._reply.readAll().data()
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial or unexpected line, the next read may complete it
            if error := chunk.get("error"):
                # Ollama reports model-level failures (unknown model, out of memory) in
                # the stream itself rather than as an HTTP error
                if self._future is not None and not self._future.done():
                    self._future.set_exception(NetworkError(0, str(error), url()))
                self.cancel()
                return
            self._text += chunk.get("response", "")
        if self._on_progress is not None:
            self._on_progress(self._text)

    def _finish(self):
        reply, self._reply = self._reply, None
        if reply is None or self._future is None or self._future.done():
            return
        code = reply.error()  # type: ignore (bug in PyQt5-stubs)
        if code == QNetworkReply.NetworkError.OperationCanceledError:
            self._future.cancel()
        elif code != QNetworkReply.NetworkError.NoError:
            self._future.set_exception(NetworkError.from_reply(reply))
        else:
            self._future.set_result(clean_response(self._text))
        reply.deleteLater()


async def unload():
    """Ask Ollama to drop the model from VRAM right now."""
    data = {"model": settings.ollama_model, "prompt": "", "stream": False, "keep_alive": 0}
    try:
        await _requests.http("POST", f"{url()}/api/generate", data, timeout=30)
    except NetworkError as e:
        log.warning(f"Failed to unload Ollama model: {e}")


async def free_comfy_vram(client):
    """Unload checkpoints on the ComfyUI side to make room for the language model.
    Only useful when both run on the same GPU."""
    try:
        await client._requests.http(
            "POST",
            f"{client.url}/free",
            {"unload_models": True, "free_memory": True},
            timeout=30,
        )
    except NetworkError as e:
        log.warning(f"Failed to free ComfyUI memory: {e}")


# Response clean-up ####################################################################

_think_re = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_fence_re = re.compile(r"^```[a-z]*\n?|```$", re.MULTILINE)
_prefix_re = re.compile(r"^\s*(?:prompt|positive prompt|output)\s*:\s*", re.IGNORECASE)


def clean_response(text: str) -> str:
    text = _think_re.sub("", text)
    text = _fence_re.sub("", text)
    text = _prefix_re.sub("", text.strip())
    text = text.strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text.strip()


# Term lists #########################################################################

_list_marker_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def normalize_term(text: str) -> str:
    """Key used to recognize duplicates: 'Long Wavy Hair.' and 'long wavy hair' are the
    same option as far as a wildcard list is concerned."""
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def parse_terms(text: str) -> list[str]:
    terms = []
    for line in text.splitlines():
        line = _list_marker_re.sub("", line).strip().strip("\"'").strip()
        # a line with a colon is usually a heading ("Hairstyles:") or an explanation
        if not line or ":" in line or len(line.split()) > 8:
            continue
        terms.append(line.rstrip(".,;").lower())
    return terms


# Protecting syntax the language model must not touch ##################################

# LoRA tags, file wildcards, random and sequential wildcard groups. These carry meaning
# for the plugin and get mangled if the model is allowed to rewrite them.
# File wildcards are matched like text.pattern_file_wildcard: names may contain '_', '/'.
# DOTALL: inserted groups put brackets and pipes on their own lines (text.format_option_group).
_protected_re = re.compile(r"<lora:[^>]*>|__[\w\-./]+?__|\[\[.*?\]\]|\{[^{}]*\|[^{}]*\}", re.DOTALL)


@dataclass
class ProtectedPrompt:
    text: str  # prompt with protected parts removed, safe to send to the model
    tokens: list[str]  # protected parts, in the order they appeared

    def restore(self, result: str) -> str:
        if not self.tokens:
            return result
        return ", ".join([result.rstrip(" ,")] + self.tokens)


def protect(prompt: str) -> ProtectedPrompt:
    tokens = _protected_re.findall(prompt)
    text = _protected_re.sub("", prompt)
    text = re.sub(r"[^\S\n]+", " ", text)  # collapse gaps left by removed tokens
    text = re.sub(r"(?:\s*,)+", ",", text)  # ... and the dangling commas around them
    text = re.sub(r",(?=\S)", ", ", text)
    return ProtectedPrompt(text.strip().strip(",").strip(), tokens)


# Prompt profiles ######################################################################


@dataclass
class Profile:
    id: str
    name: str
    system: str = ""  # empty: keep the system prompt baked into the Ollama model
    model: str = ""  # empty: use the model selected in the settings
    describe: str = ""  # system prompt for image description, empty: reuse `system`
    describe_model: str = ""  # vision model, empty: fall back to `model`

    def system_for(self, task: EnhanceTask) -> str:
        if task is EnhanceTask.describe and self.describe:
            return self.describe
        return self.system

    def model_for(self, task: EnhanceTask) -> str:
        if task is EnhanceTask.describe:
            return self.describe_model or self.model
        return self.model


class Profiles:
    """Maps a base model family (SD XL, Illustrious, Krea 2, ...) to the system prompt
    which teaches the language model how prompts for that family are written."""

    _instance: Profiles | None = None

    default_path = plugin_dir / "presets" / "prompt_enhance.json"
    user_path = user_data_dir / "prompt_enhance.json"

    def __init__(self, data: dict):
        self._profiles = {
            key: Profile(
                key,
                value.get("name", key),
                value.get("system", ""),
                value.get("model", ""),
                value.get("describe", ""),
                value.get("describe_model", ""),
            )
            for key, value in (data.get("profiles") or {}).items()
        }
        self._families: dict[str, str] = data.get("families") or {}
        self._default = data.get("default", next(iter(self._profiles), ""))

    @classmethod
    def instance(cls) -> Profiles:
        if cls._instance is None:
            cls._instance = cls.load()
        return cls._instance

    @classmethod
    def load(cls) -> Profiles:
        data = _read_json(cls.default_path) or {}
        if user := _read_json(cls.user_path):
            data.setdefault("profiles", {}).update(user.get("profiles") or {})
            data.setdefault("families", {}).update(user.get("families") or {})
            data["default"] = user.get("default", data.get("default"))
        return Profiles(data)

    @classmethod
    def reload(cls):
        cls._instance = cls.load()
        return cls._instance

    def __iter__(self):
        return iter(self._profiles.values())

    def get(self, profile_id: str) -> Profile | None:
        return self._profiles.get(profile_id)

    def for_family(self, family: str) -> Profile | None:
        profile_id = self._families.get(family, self._default)
        return self._profiles.get(profile_id) or self._profiles.get(self._default)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"Failed to read prompt profiles from {path}: {e}")
        return None


class PoolMode(Enum):
    """How a batch of prompts is produced before generating images."""

    variation = "variation"  # variations of the prompt that is already there
    random = "random"  # fresh scenes for a theme


_pool_instructions = {
    PoolMode.variation: (
        "Write ONE variation of the prompt below. Vary pose, setting, lighting and camera,"
        " but keep the same subject and style. Answer with the finished prompt only."
    ),
    PoolMode.random: (
        "Write ONE image prompt for the theme below. Invent the scene freely - subject"
        " details, setting, lighting and camera are yours to choose. Answer with the"
        " finished prompt only."
    ),
}


_terms_system = (
    "You build wildcard lists for image generation prompts. A wildcard list holds"
    " interchangeable options of one single category - one option per line, and every"
    " line must work as a drop-in replacement for the others inside a prompt.\n\n"
    "Output ONLY the list. No numbering, no bullets, no headings, no explanation, no"
    " markdown, no quotes, no blank lines.\n\n"
    "Rules:\n"
    "- One line is one option: an English fragment of one to five words, lowercase.\n"
    "- Never write a full prompt, a sentence, or several categories on one line.\n"
    "- Options must be genuinely different from each other, not rewordings.\n"
    "- Stay strictly inside the requested category.\n"
    "- Explicit and erotic options are allowed where the category calls for them."
    " Everyone depicted is an adult; never use age markers, youth or childhood terms.\n"
    "- No real, named people."
)


def terms_system_prompt() -> str:
    """A term list is not a prompt, so the family profile (tags vs prose) does not
    apply - the model needs its own instructions here."""
    return _terms_system


def build_terms_prompt(category: str, count: int, existing: list[str] | None = None) -> str:
    text = f"Write {count} different options for this category:\n{category}"
    if existing:
        # models happily repeat their own output, so the list so far is quoted back
        previous = "\n".join(existing)
        text += (
            f"\n\nThese {len(existing)} options already exist. Write {count} more that are"
            f" not in this list and not variations of its entries:\n{previous}"
        )
    return text


def build_pool_prompt(mode: PoolMode, base: str, avoid: list[str] | None = None) -> str:
    label = "Prompt" if mode is PoolMode.variation else "Theme"
    text = f"{_pool_instructions[mode]}\n\n{label}:\n{base}"
    if avoid:
        # Without this the model repeats itself almost verbatim across a batch, even
        # with a different seed each time.
        previous = "\n".join(f"- {a}" for a in avoid)
        text += (
            "\n\nYou already wrote the following. Write something clearly different"
            f" this time:\n{previous}"
        )
    return text


_negative_instruction = (
    "\n\nAnything the change asks to remove, avoid or exclude must NOT stay in the prompt"
    " as a negation. Answer in two parts: first the complete modified prompt, then a new"
    " line starting with 'NEGATIVE:' followed by a comma-separated list of the unwanted"
    " things themselves (for example 'hat, text'), without words like 'no' or 'without'."
    " If the change excludes nothing, write 'NEGATIVE:' with nothing after it."
)

_negative_re = re.compile(
    r"^[ \t>*-]*\**negative(?: prompt)?\**\s*:\**[ \t]*", re.IGNORECASE | re.MULTILINE
)


def split_negative(response: str) -> tuple[str, str]:
    """Separates the 'NEGATIVE:' part the model was asked to append from the prompt."""
    matches = list(_negative_re.finditer(response))
    if not matches:
        return response.strip(), ""
    last = matches[-1]
    negative = response[last.end() :].replace("\n", ", ")
    return response[: last.start()].strip(), re.sub(r"(?:\s*,)+", ",", negative).strip(" ,")


def merge_negative(existing: str, addition: str) -> str:
    """Appends terms to a negative prompt, skipping those that are already in there."""
    known = {normalize_term(t) for t in existing.split(",")}
    new = []
    for term in addition.split(","):
        key = normalize_term(term)
        if key and key not in known:
            known.add(key)
            new.append(term.strip())
    if not new:
        return existing
    return ", ".join([existing.rstrip(" ,")] * bool(existing.strip()) + new)


_structure_system = (
    "You organize image generation prompts. You receive one prompt and sort its content"
    " into labelled sections, so the user can edit each part on its own.\n\n"
    "Output format - repeat for every section that has content:\n"
    "# Section name\n"
    "the part of the prompt that belongs there\n\n"
    "Use these sections, in this order, and only those which have content:\n"
    "- Quality & Style: quality tags, medium, art style, photo style, artist.\n"
    "- Character: who the person is - subject count, age, ethnicity, face, hair, eyes, skin,"
    " body shape, breasts, height. NEVER garments or accessories.\n"
    "- Clothing: every garment, shoe, accessory and piece of jewelry (skirt, dress, top,"
    " stockings, necklace, glasses, hat) together with its colour, pattern, material and"
    " state (lifted, open, torn, wet, removed).\n"
    "- Pose & Expression: pose, gesture, gaze, facial expression, action.\n"
    "- Environment: location, background, props, weather, time of day.\n"
    "- Lighting: light sources, mood, colour palette.\n"
    "- Camera: shot type, angle, lens, depth of field, framing.\n"
    "- Other: anything that fits nowhere else.\n\n"
    "Rules:\n"
    "- Keep the wording, the language, the syntax and the weights of the original prompt"
    " ((word:1.2), <lora:...>, __wildcards__, [[a|b]]). Move text, do not rewrite it.\n"
    "- Keep the original style: a comma separated tag list stays a tag list, prose stays"
    " prose.\n"
    "- Every part of the prompt goes into exactly one section. Never add, remove or"
    " translate anything.\n"
    "- Content goes on a single line below its heading. Never put a '#' anywhere else.\n"
    "- Answer with the sections only. No explanation, no markdown, no quotes.\n\n"
    "Example input: masterpiece, 1girl, red hair, forest, fog, red dress, wide shot\n"
    "Example output:\n"
    "# Quality & Style\nmasterpiece\n\n"
    "# Character\n1girl, red hair\n\n"
    "# Clothing\nred dress\n\n"
    "# Environment\nforest, fog\n\n"
    "# Camera\nwide shot"
)


def structure_system_prompt() -> str:
    """Sorting a prompt is not writing one, so the family profile does not apply."""
    return _structure_system


_section_order = [
    "Quality & Style",
    "Character",
    "Clothing",
    "Pose & Expression",
    "Environment",
    "Lighting",
    "Camera",
    "Other",
    "Loras",  # filled by the plugin, not the model: LoRA tags never reach it
]
_section_names = {name.lower() for name in _section_order}


def _heading_of(line: str) -> str:
    """The section name if `line` is a heading, even one the model wrote without '#'."""
    bare = line.strip().lstrip("#*- ").rstrip("*:, ").strip()
    if line.lstrip().startswith("#") or bare.lower() in _section_names:
        return bare
    return ""


def format_sections(response: str, source: str = "", protected: list[str] | None = None) -> str:
    """Normalizes the sectioned reply into '# Heading' comment lines, each followed by its
    content. Models repeat tags across sections (a skirt under Character and again under
    Clothing), so every tag is kept only once, in the last section that lists it - the
    later sections are the more specific ones. Content lines get a trailing comma so the
    sections still run together as one prompt once the comments are stripped.

    Models also drop tags they don't know where to put (score_7, no humans) and mix up the
    section order, however clearly they are told not to. So the sections are put back in
    the documented order, and every tag of `source` missing from the reply goes under
    'Other' - sorting must never lose part of the prompt.

    `protected` are the parts taken out before the model saw the prompt (see `protect`):
    LoRA tags go under a last '# Loras' heading, one per line; wildcards and wildcard
    groups under 'Other'."""
    sections: list[tuple[str, list[str]]] = []
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        if heading := _heading_of(line):
            sections.append((heading, []))
            continue
        if not sections:
            sections.append(("Other", []))
        sections[-1][1].extend(t.strip() for t in line.split(",") if t.strip())

    # a tag the model only re-spelled ('score\_7' - a markdown escape - for 'score_7') is
    # written back exactly as the user typed it; one it made up ('side angle' next to
    # 'from side') is dropped - sorting adds nothing
    original = {normalize_term(t): t.strip() for t in source.split(",") if t.strip()}
    seen: set[str] = set()
    for _name, terms in reversed(sections):
        kept = []
        for term in reversed(terms):
            key = normalize_term(term)
            if original and key not in original:
                continue
            if key not in seen:
                seen.add(key)
                kept.append(original.get(key, term))
        terms[:] = reversed(kept)

    missing = []
    for term in source.split(","):
        term = term.strip()
        if term and (key := normalize_term(term)) not in seen:
            seen.add(key)
            missing.append(term)
    if missing:
        sections.append(("Other", missing))
    loras = [t for t in protected or [] if t.startswith("<lora:")]
    if rest := [t for t in protected or [] if not t.startswith("<lora:")]:
        sections.append(("Other", rest))

    merged: dict[str, tuple[str, list[str]]] = {}
    for name, terms in sections:
        if name.lower() in merged:
            merged[name.lower()][1].extend(terms)
        else:
            merged[name.lower()] = (name, list(terms))
    rank = {name.lower(): float(i) for i, name in enumerate(_section_order)}
    # headings the model made up go just before Other, in the order it wrote them
    ordered = sorted(merged.values(), key=lambda s: rank.get(s[0].lower(), rank["other"] - 0.5))
    blocks = [f"# {name}\n" + ", ".join(terms) for name, terms in ordered if terms]
    if loras:
        blocks.append("# Loras\n" + ",\n".join(loras))
    return ",\n\n".join(blocks).strip()


def build_prompt(
    task: EnhanceTask,
    prompt: str,
    count: int = 4,
    instruction: str = "",
    want_negative: bool = False,
) -> str:
    if task is EnhanceTask.describe:
        # the image carries the content, an existing prompt is not part of the request
        extra = f"\n\nAdditional instruction: {instruction}" if instruction.strip() else ""
        return _task_instructions[task].format(count=count, instruction=extra)
    task_text = _task_instructions[task].format(count=count, instruction=instruction)
    if want_negative and task is EnhanceTask.instruct:
        task_text += _negative_instruction
    if not prompt.strip():
        if task is EnhanceTask.instruct:
            # nothing to modify - treat the instruction itself as the idea to write about
            rewrite = _task_instructions[EnhanceTask.rewrite]
            return f"{rewrite}\n\nPrompt:\n{instruction}"
        return "The user did not provide a prompt. Invent an interesting image prompt from scratch."
    return f"{task_text}\n\nPrompt:\n{prompt}"
