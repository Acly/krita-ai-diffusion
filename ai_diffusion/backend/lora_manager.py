from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from .. import eventloop, util
from ..util import client_logger as log

if TYPE_CHECKING:
    from ..model.jobs import JobParams
    from .network import RequestManager

_STRIP_SUFFIXES = (".safetensors", ".pt", ".ckpt", ".bin")
_CACHE_MAX_AGE = 6 * 3600  # seconds
# Bump when the cached data shape/semantics change, to invalidate old caches
# written before a fix (e.g. the favorite flag used to always be False, or the
# commercial-use flag counting a "Sell"-only license as permission to sell images).
_CACHE_VERSION = 6

# In-memory copy of the disk cache, so opening the browser doesn't re-parse ~10k entries
_memory_cache: dict[str, tuple[float, list[LoraInfo]]] = {}
_preload_tasks: dict[str, asyncio.Task] = {}


def _clean_name(file_name: str) -> str:
    for suffix in _STRIP_SUFFIXES:
        if file_name.endswith(suffix):
            return file_name[: -len(suffix)]
    return file_name


@dataclass
class LoraInfo:
    name: str  # file name without extension - this is what ComfyUI expects in <lora:name:weight>
    display_name: str = ""  # human-readable title shown in the UI
    base_model: str = ""
    tags: list[str] = field(default_factory=list)
    preview_url: str = ""
    # Each entry is one alternative trigger phrase / word group, as reported by
    # CivitAI - they are alternatives, not meant to be concatenated together.
    trigger_words: list[str] = field(default_factory=list)
    sha256: str = ""
    favorite: bool = False
    modified: float = 0.0  # unix timestamp, file mtime - used as "date added" proxy
    version: str = ""  # CivitAI model version name (e.g. "v1.0")
    file_path: str = ""  # full path on the server, needed to query per-model metadata
    commercial: str = ""  # "", "yes" or "no" - filled in lazily (see fetch_commercial_use)
    civitai_model_id: int = 0  # 0 if not from CivitAI
    # CivitAI content-rating bit flag for the preview image specifically (not
    # necessarily the whole model): 1=G 2=PG 4=PG-13 8=R 16=X 32=XXX, 0=unrated
    nsfw_level: int = 0

    @staticmethod
    def from_api(data: dict, base_url: str) -> LoraInfo:
        # ComfyUI-Lora-Manager format: GET /api/lm/loras/list
        file_name = data.get("file_name") or data.get("model_name", "")
        name = _clean_name(file_name)
        display_name = data.get("model_name") or name
        sha256 = data.get("sha256", "")
        preview = data.get("preview_url", "")
        if preview and preview.startswith("/"):
            preview = base_url + preview
        tags = data.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        trigger_words = []
        version = ""
        civitai_model_id = 0
        civitai = data.get("civitai") or {}
        if isinstance(civitai, dict):
            trigger_words = civitai.get("trainedWords") or []
            version = civitai.get("name", "") or ""
            civitai_model_id = civitai.get("modelId") or 0
        return LoraInfo(
            name=name,
            display_name=display_name,
            base_model=data.get("base_model", ""),
            tags=tags,
            preview_url=preview,
            trigger_words=trigger_words,
            version=version,
            civitai_model_id=civitai_model_id,
            nsfw_level=int(data.get("preview_nsfw_level") or 0),
            favorite=bool(data.get("favorite")),
            sha256=sha256,
            modified=float(data.get("modified") or 0.0),
            file_path=data.get("file_path", "") or "",
        )


# base_model strings (as stored by ComfyUI-Lora-Manager metadata) → Arch enum value name
# Longer/more specific keys must come before shorter ones they overlap with
# (e.g. "illustrious" before "sd xl", "flux kontext" before "flux").
_BASE_MODEL_MAP = [
    ("illustrious", "illu"),
    ("pony", "sdxl"),
    ("sdxl", "sdxl"),
    ("sd xl", "sdxl"),
    ("sd 1", "sd15"),
    ("sd1", "sd15"),
    ("v1", "sd15"),
    ("sd 3", "sd3"),
    ("sd3", "sd3"),
    ("flux kontext", "flux_k"),
    ("flux", "flux"),
    ("chroma", "chroma"),
    ("qwen", "qwen"),
    ("anima", "anima"),
    ("z-image", "zimage"),
    ("zimage", "zimage"),
    ("ernie", "ernie"),
    ("krea", "krea2"),
    ("moss", "moss"),
]


def arch_for_base_model(base_model: str) -> str:
    """Return Arch enum name (e.g. 'sdxl') for a base_model string, or '' if unknown."""
    lower = base_model.lower()
    for key, arch in _BASE_MODEL_MAP:
        if key in lower:
            return arch
    return ""


def _cache_path(base_url: str) -> Path:
    url_hash = hashlib.md5(base_url.encode()).hexdigest()[:8]
    return util.user_data_dir / f"lora_manager_cache_{url_hash}.json"


def load_cached_loras(base_url: str) -> list[LoraInfo] | None:
    """Return cached LoRA list if present and not expired, else None."""
    if base_url in _memory_cache:
        timestamp, loras = _memory_cache[base_url]
        if time.time() - timestamp <= _CACHE_MAX_AGE:
            return list(loras)
        del _memory_cache[base_url]
    path = _cache_path(base_url)
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != _CACHE_VERSION:
            return None
        timestamp = data.get("timestamp", 0)
        if time.time() - timestamp > _CACHE_MAX_AGE:
            return None
        loras = [LoraInfo(**item) for item in data.get("loras", [])]
        _memory_cache[base_url] = (timestamp, loras)
        return list(loras)
    except Exception as e:
        log.warning(f"Could not load LoRA cache: {e}")
        return None


def save_lora_cache(base_url: str, loras: list[LoraInfo]):
    timestamp = time.time()
    _memory_cache[base_url] = (timestamp, list(loras))
    path = _cache_path(base_url)
    try:
        data = {
            "version": _CACHE_VERSION,
            "timestamp": timestamp,
            "loras": [asdict(l) for l in loras],
        }
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception as e:
        log.warning(f"Could not save LoRA cache: {e}")


async def _fetch_favorite_hashes(requests: RequestManager, base: str) -> set[str]:
    """The regular /list endpoint does not reliably report `favorite`, so query
    the dedicated favorites_only filter and collect sha256 hashes from that."""
    hashes: set[str] = set()
    try:
        page = 1
        page_size = 200
        while True:
            data = await requests.get(
                f"{base}/api/lm/loras/list?page={page}&page_size={page_size}&favorites_only=true",
                timeout=15.0,
            )
            if isinstance(data, (bytes, bytearray)):
                data = json.loads(data)
            if not isinstance(data, dict):
                break
            items = data.get("items") or data.get("loras") or []
            if not items:
                break
            hashes.update(item.get("sha256", "") for item in items)
            total = data.get("total", page * page_size)
            actual_page_size = data.get("page_size", page_size)
            if page * actual_page_size >= total or len(items) < actual_page_size:
                break
            page += 1
    except Exception as e:
        log.warning(f"Could not fetch LoRA favorites: {e}")
    return hashes


async def fetch_loras_pages(requests: RequestManager, base_url: str):
    """Yield LoRA list incrementally, one server page at a time.

    Falls back to a single yield from /models/loras (filename list only) if
    ComfyUI-Lora-Manager is not installed.
    """
    base = base_url.rstrip("/")

    # ComfyUI-Lora-Manager (rich metadata: tags, base_model, preview, trigger words)
    # Server caps page_size regardless of what we request, so page through all results.
    try:
        favorite_hashes = await _fetch_favorite_hashes(requests, base)
        page = 1
        page_size = 200
        got_any = False
        while True:
            data = await requests.get(
                f"{base}/api/lm/loras/list?page={page}&page_size={page_size}", timeout=15.0
            )
            if isinstance(data, (bytes, bytearray)):
                data = json.loads(data)
            if not isinstance(data, dict):
                break
            items = data.get("items") or data.get("loras") or []
            if not items:
                break
            got_any = True
            batch = [LoraInfo.from_api(item, base) for item in items]
            for lora in batch:
                if lora.sha256 in favorite_hashes:
                    lora.favorite = True
            yield batch
            total = data.get("total", page * page_size)
            actual_page_size = data.get("page_size", page_size)
            loaded = page * actual_page_size
            if loaded >= total or len(items) < actual_page_size:
                break
            page += 1
        if got_any:
            return
    except Exception as e:
        log.warning(f"Lora Manager API not available: {e}")

    # Fallback: standard ComfyUI /models/loras (filename list only)
    try:
        data = await requests.get(f"{base}/models/loras", timeout=10.0)
        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data)
        if isinstance(data, list):
            result = []
            for entry in data:
                if isinstance(entry, str):
                    result.append(
                        LoraInfo(name=_clean_name(entry), display_name=_clean_name(entry))
                    )
                elif isinstance(entry, dict):
                    result.append(LoraInfo.from_api(entry, base))
            if result:
                log.info(f"Loaded {len(result)} LoRAs from /models/loras (no metadata)")
                yield result
    except Exception as e:
        log.warning(f"Could not fetch LoRA list: {e}")


async def _preload_loras(requests: RequestManager, base_url: str):
    loras: list[LoraInfo] = []
    async for batch in fetch_loras_pages(requests, base_url):
        loras.extend(batch)
    if loras:
        save_lora_cache(base_url, loras)
        log.info(f"Preloaded {len(loras)} LoRAs")


def start_lora_preload(requests: RequestManager, base_url: str):
    """Fill the LoRA cache in the background (after connecting), so the browser opens
    without waiting for the server - unless the cache is still fresh anyway."""
    if lora_preload_task(base_url) is not None or load_cached_loras(base_url):
        return
    _preload_tasks[base_url] = eventloop.run(_preload_loras(requests, base_url))


def lora_preload_task(base_url: str) -> asyncio.Task | None:
    """The preload still running for this server, if any - await it instead of fetching twice."""
    task = _preload_tasks.get(base_url)
    if task is not None and task.done():
        del _preload_tasks[base_url]
        return None
    return task


async def fetch_new_loras(
    requests: RequestManager, base_url: str, known: list[LoraInfo]
) -> list[LoraInfo] | None:
    """The LoRAs Lora Manager lists that `known` doesn't have yet.

    Pages through the list newest first and stops at the first page with nothing
    new once the counts add up, instead of reloading all ~10k entries. The date
    alone can't be trusted - a copied file keeps its old modification time - so
    the server's total has to match too. Returns None when an incremental update
    isn't reliable (files were removed, unexpected answer): reload fully then."""
    base = base_url.rstrip("/")
    known_keys = {lora.file_path or lora.name for lora in known}
    new: list[LoraInfo] = []
    total = -1
    try:
        page, page_size = 1, 200
        while True:
            data = await requests.get(
                f"{base}/api/lm/loras/list?page={page}&page_size={page_size}&sort_by=date:desc",
                timeout=15.0,
            )
            if isinstance(data, (bytes, bytearray)):
                data = json.loads(data)
            if not isinstance(data, dict) or not isinstance(data.get("total"), int):
                return None
            total = data["total"]
            items = data.get("items") or []
            batch = [LoraInfo.from_api(item, base) for item in items]
            fresh = [lora for lora in batch if (lora.file_path or lora.name) not in known_keys]
            new.extend(fresh)
            if not fresh and len(known) + len(new) == total:
                return new
            if not fresh and len(known) + len(new) > total:
                return None  # files were removed - paging further can't fix the count
            if not items or len(items) < data.get("page_size", page_size):
                break
            page += 1
    except Exception as e:
        log.warning(f"Could not fetch new LoRAs: {e}")
        return None
    return new if len(known) + len(new) == total else None


# Lora Manager base_model string -> Krita Style "base_model_family" label.
# Longer/more specific keys before shorter overlapping ones.
_STYLE_FAMILY_MAP = [
    ("illustrious", "Illustrious"),
    ("pony", "Pony"),
    ("sd xl", "SD XL"),
    ("sdxl", "SD XL"),
    ("sd 1", "SD 1.5"),
    ("sd1", "SD 1.5"),
    ("v1", "SD 1.5"),
    ("sd 3", "SD 3"),
    ("sd3", "SD 3"),
    ("flux kontext", "Flux Kontext"),
    ("flux", "Flux"),
    ("chroma", "Chroma"),
    ("qwen", "Qwen"),
    ("anima", "Anima"),
    ("z-image", "Z-Image"),
    ("zimage", "Z-Image"),
    ("ernie", "ERNIE Image"),
    ("krea", "Krea 2"),
]


def style_family_for_base_model(base_model: str) -> str:
    """Map a checkpoint's base_model string to a Krita Style base_model_family
    label, or 'Auto' if unknown (lets the style guess from the file name)."""
    lower = base_model.lower()
    for key, family in _STYLE_FAMILY_MAP:
        if key in lower:
            return family
    return "Auto"


async def fetch_checkpoints_pages(requests: RequestManager, base_url: str):
    """Yield the checkpoint list from Lora Manager incrementally, one page at a
    time (reuses LoraInfo - same fields). Empty if Lora Manager isn't installed."""
    base = base_url.rstrip("/")
    try:
        page = 1
        page_size = 200
        while True:
            data = await requests.get(
                f"{base}/api/lm/checkpoints/list?page={page}&page_size={page_size}", timeout=15.0
            )
            if isinstance(data, (bytes, bytearray)):
                data = json.loads(data)
            if not isinstance(data, dict):
                break
            items = data.get("items") or []
            if not items:
                break
            yield [LoraInfo.from_api(item, base) for item in items]
            total = data.get("total", page * page_size)
            actual_page_size = data.get("page_size", page_size)
            if page * actual_page_size >= total or len(items) < actual_page_size:
                break
            page += 1
    except Exception as e:
        log.warning(f"Could not fetch checkpoints from Lora Manager: {e}")


@dataclass
class RecipeLora:
    name: str  # lora file name without extension, usable in <lora:...> tags
    strength: float
    available: bool  # present in the local library and not excluded


@dataclass
class RecipeInfo:
    id: str
    title: str
    base_model: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    checkpoint: str = ""
    preview_url: str = ""
    favorite: bool = False
    loras: list[RecipeLora] = field(default_factory=list)
    created_date: float = 0.0  # unix timestamp

    @staticmethod
    def from_api(data: dict, base_url: str) -> RecipeInfo:
        gen = data.get("gen_params") or {}
        checkpoint = (data.get("checkpoint") or {}).get("file_name", "") or ""
        preview = data.get("file_url", "")
        if preview.startswith("/"):
            preview = base_url + preview
        loras = []
        for entry in data.get("loras") or []:
            name = _clean_name(entry.get("file_name", ""))
            if not name:
                continue
            available = bool(
                entry.get("inLibrary", False)
                and not entry.get("isDeleted", False)
                and not entry.get("exclude", False)
            )
            loras.append(RecipeLora(name, float(entry.get("strength", 1.0)), available))
        return RecipeInfo(
            id=data.get("id", ""),
            title=data.get("title", ""),
            base_model=data.get("base_model", ""),
            prompt=gen.get("prompt", ""),
            negative_prompt=gen.get("negative_prompt", ""),
            checkpoint=checkpoint,
            preview_url=preview,
            favorite=bool(data.get("favorite")),
            loras=loras,
            created_date=float(data.get("created_date") or data.get("modified") or 0.0),
        )


async def fetch_recipes(requests: RequestManager, base_url: str) -> list[RecipeInfo]:
    """Fetch all recipes from ComfyUI-Lora-Manager, paging through server results."""
    base = base_url.rstrip("/")
    result: list[RecipeInfo] = []
    try:
        page = 1
        page_size = 100
        while True:
            data = await requests.get(
                f"{base}/api/lm/recipes?page={page}&page_size={page_size}", timeout=15.0
            )
            if isinstance(data, (bytes, bytearray)):
                data = json.loads(data)
            if not isinstance(data, dict):
                break
            items = data.get("items") or []
            if not items:
                break
            result.extend(RecipeInfo.from_api(item, base) for item in items)
            total = data.get("total", len(result))
            actual_page_size = data.get("page_size", page_size)
            if page * actual_page_size >= total or len(items) < actual_page_size:
                break
            page += 1
        log.info(f"Loaded {len(result)} recipes from Lora Manager")
    except Exception as e:
        log.warning(f"Could not fetch recipes: {e}")
    return result


async def fetch_preview_bytes(requests: RequestManager, preview_url: str) -> bytes | None:
    """Fetch preview image bytes. Returns None on error."""
    if not preview_url:
        return None
    try:
        result = await requests.download(preview_url, timeout=8.0)
        return bytes(result) if result else None
    except Exception as e:
        log.warning(f"Could not fetch LoRA preview {preview_url}: {e}")
        return None


async def _lookup_lora_hash(requests: RequestManager, base: str, name: str) -> dict:
    """Look up sha256/display-name for a LoRA by file name, so Lora Manager can match
    it to its local library entry instead of showing "Not in library". We don't have
    the hash at generation time (workflow.py only computes it for LoRAs that need
    uploading, not ones already present on the server)."""
    try:
        data = await requests.get(
            f"{base}/api/lm/loras/list?page=1&page_size=5&search={quote(name)}", timeout=8.0
        )
        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data)
        items = data.get("items", []) if isinstance(data, dict) else []
        for item in items:
            if Path(str(item.get("file_name", ""))).stem == name:
                return {
                    "hash": (item.get("sha256") or "").lower(),
                    "name": item.get("model_name", name),
                }
    except Exception as e:
        log.warning(f"Could not look up LoRA info for '{name}': {e}")
    return {}


async def set_favorite(
    requests: RequestManager, base_url: str, file_path: str, favorite: bool
) -> bool:
    """Toggle a LoRA's favorite flag in Lora Manager. Returns True on success."""
    if not file_path:
        return False
    base = base_url.rstrip("/")
    try:
        result = await requests.post(
            f"{base}/api/lm/loras/save-metadata", {"file_path": file_path, "favorite": favorite}
        )
        return isinstance(result, dict) and bool(result.get("success"))
    except Exception as e:
        log.warning(f"Could not set favorite for '{file_path}': {e}")
        return False


def commercial_use_from_license(allow: list[str] | str | None) -> str:
    """Map CivitAI's allowCommercialUse to "yes"/"no"/"" (unstated) for the only
    question that matters here: may generated images be sold?

    CivitAI values are Image / Rent / RentCivit / Sell. Only "Image" grants that
    right. "Sell" is about reselling the model itself, "RentCivit"/"Rent" only allow
    running it on a generation service - none of those permit selling output."""
    if allow is None:
        return ""
    if isinstance(allow, str):
        allow = [allow]
    if not allow:
        return "no"  # explicitly empty list = no commercial use at all
    return "yes" if "Image" in allow else "no"


async def fetch_commercial_use(requests: RequestManager, base_url: str, file_path: str) -> str:
    """Return "yes"/"no"/"" (unknown) for whether the model's CivitAI license allows
    selling generated images. Reads per-model metadata (not in the list)."""
    if not file_path:
        return ""
    base = base_url.rstrip("/")
    try:
        data = await requests.get(
            f"{base}/api/lm/loras/metadata?file_path={quote(file_path)}", timeout=8.0
        )
        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data)
        model = ((data or {}).get("metadata") or {}).get("model") or {}
        return commercial_use_from_license(model.get("allowCommercialUse"))
    except Exception as e:
        log.warning(f"Could not fetch commercial-use info for '{file_path}': {e}")
        return ""


async def save_recipe(
    requests: RequestManager,
    base_url: str,
    name: str,
    tags: list[str],
    image_bytes: bytes,
    params: JobParams,
) -> str | None:
    """Save a generated image as a Lora Manager recipe (POST /api/lm/recipes/save).

    Returns an error message on failure, None on success.
    """
    base = base_url.rstrip("/")
    meta = params.metadata

    loras = []
    for lora in meta.get("loras", []):
        if isinstance(lora, dict) and lora.get("name"):
            clean_name = Path(str(lora["name"])).stem
            entry = {"file_name": clean_name, "weight": float(lora.get("weight", 1.0))}
            entry.update(await _lookup_lora_hash(requests, base, clean_name))
            loras.append(entry)

    gen_params = {
        "prompt": meta.get("prompt_final", meta.get("prompt", "")),
        "negative_prompt": meta.get("negative_prompt_final", meta.get("negative_prompt", "")),
        "steps": meta.get("steps", 0),
        "sampler": meta.get("sampler", ""),
        "cfg_scale": meta.get("guidance", 0.0),
        "seed": params.seed,
        "size": f"{params.bounds.width}x{params.bounds.height}",
    }
    strength = meta.get("strength")
    if strength is not None and strength != 1.0:
        gen_params["denoising_strength"] = strength

    metadata_payload: dict = {"loras": loras, "gen_params": gen_params}
    if checkpoint := meta.get("checkpoint"):
        metadata_payload["checkpoint"] = {"file_name": checkpoint, "name": checkpoint}

    fields = {
        "name": name[:100] or "Krita AI recipe",
        "tags": json.dumps(tags),
        "metadata": json.dumps(metadata_payload),
        "extension": ".png",
    }
    try:
        result = await requests.post_multipart(
            f"{base}/api/lm/recipes/save",
            fields,
            file_field="image",
            file_bytes=image_bytes,
            file_name="preview.png",
            timeout=20.0,
        )
        if isinstance(result, dict) and result.get("success"):
            return None
        return f"Lora Manager returned an error: {result}"
    except Exception as e:
        log.warning(f"Could not save recipe: {e}")
        return f"Could not reach Lora Manager: {e}"


# ── downloading models through Lora Manager ──
# Lora Manager does the actual fetching from CivitAI (it has the API key, knows the
# folder layout and writes metadata + preview next to the file), so the plugin only
# schedules a download and watches its progress.


def new_download_id() -> str:
    """Lora Manager only returns its own download id when the transfer has finished,
    which is far too late to show progress. It accepts one we generate instead."""
    return uuid.uuid4().hex


async def start_download(
    requests: RequestManager,
    base_url: str,
    model_id: int,
    version_id: int = 0,
    download_id: str = "",
    model_root: str = "",
    relative_path: str = "",
) -> dict:
    """Download a model from CivitAI into the library. Resolves only once the
    transfer is done (or failed), so poll fetch_download_progress() alongside it.

    Returns the Lora Manager result dict; on failure it carries an `error` message,
    e.g. "Model version already exists in lora library" for a model already present.
    """
    base = base_url.rstrip("/")
    payload: dict = {
        "model_id": int(model_id),
        "download_id": download_id or new_download_id(),
        "use_default_paths": not model_root,
    }
    if version_id:
        payload["model_version_id"] = int(version_id)
    if model_root:
        payload["model_root"] = model_root
        payload["relative_path"] = relative_path
    try:
        # no timeout: the request stays open for the whole (potentially long) transfer
        result = await requests.post(f"{base}/api/lm/download-model", payload)
        if isinstance(result, (bytes, bytearray)):
            result = json.loads(result)
        if isinstance(result, dict):
            return result
        return {"success": False, "error": f"Unexpected response: {result}"}
    except Exception as e:
        log.warning(f"Model download failed: {e}")
        return {"success": False, "error": str(e)}


async def fetch_download_progress(
    requests: RequestManager, base_url: str, download_id: str
) -> dict:
    """Progress of a running download: percent, bytes and speed. Empty dict while
    the download has not reported anything yet (the endpoint 404s until then)."""
    base = base_url.rstrip("/")
    try:
        data = await requests.get(
            f"{base}/api/lm/download-progress/{quote(download_id)}", timeout=8.0
        )
        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data)
        return data if isinstance(data, dict) and data.get("success") else {}
    except Exception:
        return {}  # 404 until the first progress event - not worth logging


async def cancel_download(requests: RequestManager, base_url: str, download_id: str) -> bool:
    base = base_url.rstrip("/")
    try:
        result = await requests.get(
            f"{base}/api/lm/cancel-download-get?download_id={quote(download_id)}", timeout=10.0
        )
        if isinstance(result, (bytes, bytearray)):
            result = json.loads(result)
        return isinstance(result, dict) and bool(result.get("success"))
    except Exception as e:
        log.warning(f"Could not cancel download {download_id}: {e}")
        return False


def clear_lora_cache(base_url: str):
    """Drop the cached LoRA list, so the next browser open reflects new downloads."""
    _memory_cache.pop(base_url, None)
    try:
        _cache_path(base_url).unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"Could not clear LoRA cache: {e}")


async def fetch_model_roots(requests: RequestManager, base_url: str, kind: str = "loras"):
    """Model root folders configured in Lora Manager, e.g. the loras directory.
    First entry is the default. Empty if Lora Manager is not installed."""
    base = base_url.rstrip("/")
    try:
        data = await requests.get(f"{base}/api/lm/{kind}/roots", timeout=10.0)
        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data)
        if isinstance(data, dict) and data.get("success"):
            return [str(r) for r in data.get("roots") or []]
    except Exception as e:
        log.warning(f"Could not fetch {kind} roots: {e}")
    return []


async def fetch_folders(requests: RequestManager, base_url: str, kind: str = "loras"):
    """Existing subfolders below the model roots, as relative paths ("" = root)."""
    base = base_url.rstrip("/")
    try:
        data = await requests.get(f"{base}/api/lm/{kind}/folders", timeout=15.0)
        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data)
        if isinstance(data, dict):
            return [str(f) for f in data.get("folders") or []]
    except Exception as e:
        log.warning(f"Could not fetch {kind} folders: {e}")
    return []


async def fetch_installed_hashes(
    requests: RequestManager, base_url: str, hashes: list[str], kind: str = "loras"
) -> set[str]:
    """Which of the given sha256 hashes are present in the local library, upper case.

    Lora Manager's list endpoint takes a `hashes` filter, so this is one request per
    batch instead of pulling the whole library - and unlike the browser's cached list
    it is authoritative even when nothing has been cached yet.
    """
    found: set[str] = set()
    base = base_url.rstrip("/")
    unique = [h for h in {h.upper() for h in hashes if h}]
    for start in range(0, len(unique), 100):  # keep the query string manageable
        batch = unique[start : start + 100]
        try:
            query = quote(json.dumps(batch))
            data = await requests.get(
                f"{base}/api/lm/{kind}/list?page_size=200&hashes={query}", timeout=15.0
            )
            if isinstance(data, (bytes, bytearray)):
                data = json.loads(data)
            if not isinstance(data, dict):
                continue
            for item in data.get("items") or []:
                if sha := item.get("sha256"):
                    found.add(str(sha).upper())
        except Exception as e:
            log.warning(f"Could not check installed hashes: {e}")
            break
    return found
