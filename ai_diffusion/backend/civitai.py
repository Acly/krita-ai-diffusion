from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib.parse import quote

from ..settings import settings
from ..util import client_logger as log
from .lora_manager import commercial_use_from_license
from .network import RequestManager


def api_url() -> str:
    """API root of the configured CivitAI front end (civitai.red or civitai.com -
    same API, same models)."""
    host = (settings.civitai_host or "civitai.red").strip().rstrip("/")
    host = host.split("//")[-1]
    return f"https://{host}/api/v1"


# Arch enum name -> the `baseModels` values CivitAI filters on. Verified against the
# live API - unlisted or misspelled values are accepted but silently match nothing,
# so only labels that actually return results belong here. Several architectures map
# to more than one label: Pony/Illustrious/NoobAI are separate labels even though
# they are SDXL underneath (same problem as arch_for_base_model solves locally).
# An empty list means "CivitAI has no usable label for this" - the filter then falls
# back to matching client-side on the version's base model string.
civitai_base_models = {
    "sd15": ["SD 1.5", "SD 1.5 LCM"],
    "sdxl": ["SDXL 1.0", "SDXL 1.0 LCM", "Pony", "Illustrious", "NoobAI"],
    "illu": ["Illustrious", "NoobAI"],
    "sd3": [],
    "flux": ["Flux.1 D", "Flux.1 S"],
    "flux_k": ["Flux.1 Kontext"],
    "chroma": ["Chroma"],
    "qwen": ["Qwen"],
    "anima": ["Anima"],
    "zimage": ["ZImageTurbo"],
    "ernie": ["Ernie"],
    "krea2": ["Krea 2"],
    "moss": [],
}

# CivitAI model types which can hold a LoRA-style file. LoCon/DoRA load through the
# same <lora:...> tag, so they belong in the same search.
lora_types = ["LORA", "LoCon", "DoRA"]
checkpoint_types = ["Checkpoint"]

sort_options = ["Highest Rated", "Most Downloaded", "Most Liked", "Newest"]
period_options = ["AllTime", "Year", "Month", "Week", "Day"]

_video_extensions = (".mp4", ".webm", ".mov")


@dataclass
class CivitaiFile:
    name: str = ""
    size_kb: float = 0.0
    sha256: str = ""  # upper case, as CivitAI reports it
    primary: bool = False


@dataclass
class CivitaiVersion:
    id: int
    name: str = ""
    base_model: str = ""
    trained_words: list[str] = field(default_factory=list)
    published_at: str = ""
    availability: str = ""  # "Public", "EarlyAccess", ...
    early_access: bool = False
    preview_url: str = ""
    preview_nsfw_level: int = 0
    files: list[CivitaiFile] = field(default_factory=list)

    @property
    def primary_file(self) -> CivitaiFile | None:
        for f in self.files:
            if f.primary:
                return f
        return self.files[0] if self.files else None

    @property
    def size_mb(self) -> float:
        f = self.primary_file
        return f.size_kb / 1024 if f else 0.0

    @staticmethod
    def from_api(data: dict) -> CivitaiVersion:
        files = []
        for entry in data.get("files") or []:
            if entry.get("type") not in (None, "Model", "Pruned Model"):
                continue  # config/vae/training data - not the weights we download
            hashes = entry.get("hashes") or {}
            files.append(
                CivitaiFile(
                    name=entry.get("name", ""),
                    size_kb=float(entry.get("sizeKB") or 0.0),
                    sha256=str(hashes.get("SHA256", "")).upper(),
                    primary=bool(entry.get("primary", False)),
                )
            )
        preview_url = ""
        preview_level = 0
        for image in data.get("images") or []:
            if url := image.get("url"):
                preview_url = url
                preview_level = int(image.get("nsfwLevel") or 0)
                break
        availability = data.get("availability", "") or ""
        return CivitaiVersion(
            id=int(data.get("id") or 0),
            name=data.get("name", "") or "",
            base_model=data.get("baseModel", "") or "",
            trained_words=data.get("trainedWords") or [],
            published_at=(data.get("publishedAt") or "")[:10],
            availability=availability,
            early_access=availability == "EarlyAccess" or bool(data.get("paidAccess")),
            preview_url=preview_url,
            preview_nsfw_level=preview_level,
            files=files,
        )


@dataclass
class CivitaiModel:
    id: int
    name: str = ""
    type: str = ""  # LORA / LoCon / DoRA / Checkpoint
    creator: str = ""
    tags: list[str] = field(default_factory=list)
    nsfw: bool = False
    nsfw_level: int = 0
    poi: bool = False  # depicts a real person
    # raw CivitAI license values, e.g. ["Image", "RentCivit", "Rent"] - kept as-is so
    # the UI can show what the license actually says, not just the derived verdict
    allow_commercial: list[str] = field(default_factory=list)
    downloads: int = 0
    versions: list[CivitaiVersion] = field(default_factory=list)

    @property
    def commercial(self) -> str:
        """ "yes" if generated images may be sold, "no" if not, "" if unstated."""
        return commercial_use_from_license(self.allow_commercial)

    @property
    def license_summary(self) -> str:
        return ", ".join(self.allow_commercial) if self.allow_commercial else "-"

    @staticmethod
    def from_api(data: dict) -> CivitaiModel:
        allow = data.get("allowCommercialUse")
        if isinstance(allow, str):
            allow = [allow]
        stats = data.get("stats") or {}
        creator = (data.get("creator") or {}).get("username", "") or ""
        return CivitaiModel(
            id=int(data.get("id") or 0),
            name=data.get("name", "") or "",
            type=data.get("type", "") or "",
            creator=creator,
            tags=[t for t in (data.get("tags") or []) if isinstance(t, str)],
            nsfw=bool(data.get("nsfw")),
            nsfw_level=int(data.get("nsfwLevel") or 0),
            poi=bool(data.get("poi")),
            allow_commercial=list(allow or []),
            downloads=int(stats.get("downloadCount") or 0),
            versions=[CivitaiVersion.from_api(v) for v in (data.get("modelVersions") or [])],
        )


def is_video_url(url: str) -> bool:
    lower = url.lower().split("?")[0]
    return any(lower.endswith(ext) for ext in _video_extensions)


def preview_thumbnail_url(url: str, width: int = 320) -> str:
    """Rewrite a CivitAI image URL to fetch a thumbnail instead of the original.

    The second-to-last path segment carries the CDN transform. `anim=false` is what
    does the heavy lifting: it makes the CDN return a single JPEG frame, which turns
    a 2.9 MB mp4 preview into a 105 KB still - no local video decoding needed - and
    a width alone is sometimes ignored (1.4 MB original vs 50 KB with anim+quality).
    """
    if not url:
        return url
    transform = f"anim=false,width={width},quality=75"
    parts = url.split("/")
    if len(parts) < 2:
        return url
    for i, part in enumerate(parts):
        if part.startswith(("original=", "width=", "anim=")):
            parts[i] = transform
            return "/".join(parts)
    parts.insert(len(parts) - 1, transform)  # no transform segment present
    return "/".join(parts)


def model_page_url(model_id: int, version_id: int = 0) -> str:
    host = api_url().split("//")[-1].split("/")[0]
    url = f"https://{host}/models/{model_id}"
    return f"{url}?modelVersionId={version_id}" if version_id else url


_manager: RequestManager | None = None


def requests() -> RequestManager:
    """A RequestManager of our own for civitai.com. Deliberately not the ComfyUI
    client's one: that may carry an auth bearer and custom headers for the cloud
    service, which must not be sent to a third party."""
    global _manager
    if _manager is None:
        _manager = RequestManager()
    return _manager


def _query(params: list[tuple[str, str]]) -> str:
    return "&".join(f"{k}={quote(str(v))}" for k, v in params if str(v) != "")


async def search_models(
    query: str = "",
    types: list[str] | None = None,
    base_models: list[str] | None = None,
    tag: str = "",
    sort: str = "Highest Rated",
    period: str = "AllTime",
    nsfw: bool | None = None,
    limit: int = 50,
    cursor: str = "",
    api_key: str = "",
) -> tuple[list[CivitaiModel], str]:
    """Search civitai.com. Returns (models, next_cursor); an empty cursor means the
    end of the result set. CivitAI ignores `page` for most sort orders, so paging
    has to go through the cursor it hands back."""
    params: list[tuple[str, str]] = [("limit", str(limit)), ("sort", sort), ("period", period)]
    if query:
        params.append(("query", query))
    if tag:
        params.append(("tag", tag))
    for t in types or lora_types:
        params.append(("types", t))
    for b in base_models or []:
        params.append(("baseModels", b))
    if nsfw is not None:
        params.append(("nsfw", "true" if nsfw else "false"))
    if cursor:
        params.append(("cursor", cursor))

    url = f"{api_url()}/models?{_query(params)}"
    data = await requests().get(url, timeout=30.0, bearer=api_key or None)
    if isinstance(data, (bytes, bytearray)):
        data = json.loads(data)
    if not isinstance(data, dict):
        return [], ""
    models = [CivitaiModel.from_api(item) for item in data.get("items") or []]
    next_cursor = str((data.get("metadata") or {}).get("nextCursor") or "")
    return models, next_cursor


async def fetch_model_preview(model_id: int, api_key: str = "") -> tuple[str, int]:
    """Preview image url and its rating for a single model, via the detail endpoint.

    The list endpoint returns no images at all for models rated R and above (~a third
    of results), while /models/{id} does hand them out - so a tile without a preview
    can still get one, at the cost of one request per model.
    """
    try:
        data = await requests().get(
            f"{api_url()}/models/{int(model_id)}", timeout=20.0, bearer=api_key or None
        )
        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data)
        if not isinstance(data, dict):
            return "", 0
        for version in data.get("modelVersions") or []:
            for image in version.get("images") or []:
                if url := image.get("url"):
                    return url, int(image.get("nsfwLevel") or 0)
        return "", 0
    except Exception as e:
        log.warning(f"Could not fetch CivitAI model {model_id}: {e}")
        return "", 0


async def fetch_tags(query: str = "", limit: int = 100, api_key: str = "") -> list[str]:
    """Tag vocabulary, most used first (or prefix matches for `query`).

    Worth having because CivitAI silently ignores a `tag` it does not know - the
    search then returns unfiltered results that look like a match. Offering the real
    vocabulary keeps the filter honest."""
    params = [("limit", str(limit))]
    if query:
        params.append(("query", query))
    try:
        data = await requests().get(
            f"{api_url()}/tags?{_query(params)}", timeout=20.0, bearer=api_key or None
        )
        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data)
        if not isinstance(data, dict):
            return []
        return [str(t.get("name", "")) for t in data.get("items") or [] if t.get("name")]
    except Exception as e:
        log.warning(f"Could not fetch CivitAI tags: {e}")
        return []


async def fetch_image(url: str) -> bytes | None:
    if not url:
        return None
    try:
        result = await requests().download(url, timeout=15.0)
        return bytes(result) if result else None
    except Exception as e:
        log.warning(f"Could not fetch CivitAI image {url}: {e}")
        return None
