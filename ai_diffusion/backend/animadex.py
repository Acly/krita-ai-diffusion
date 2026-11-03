from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import quote, urlsplit

from .. import __version__ as plugin_version
from .. import util
from ..localization import translate as _
from ..util import client_logger as log
from .network import NetworkError, RequestManager

# AnimaDex catalogues characters and artists known to the ANIMA model, each with the
# danbooru-style trigger phrase that reproduces them. Endpoints follow the Flask app
# at github.com/zetaneko/AnimaDex (routes/api_gallery.py): /api/<mode>/{search,facets,
# facet/<name>}. There is no API version or stability promise, so treat every answer
# as untrusted and handle failures gracefully.

_DEFAULT_HOST = "animadex.net"
_PAGE_SIZE = 36  # server-side setting (gallery.page_size), reported back as page_size
_FACET_CACHE_MAX_AGE = 24 * 3600  # seconds - facet lists barely move
_FACET_CACHE_VERSION = 1

sort_options = ["count", "az", "random", "liked", "favourited", "recent"]
artist_sort_options = sort_options + ["score"]

# Facet keys, in the order they should appear in the UI. The server reports a
# human-readable label for each of them.
character_facets = ["character", "copyright", "eye_color", "gender", "hair_color", "hair_length"]
artist_facets = ["artist", "category", "score"]


def api_url(host: str = "") -> str:
    """API root of the configured AnimaDex instance. The software is open source
    (github.com/zetaneko/AnimaDex), so `host` may well be a local Flask server -
    keep an explicit http:// scheme instead of forcing TLS on it."""
    host = (host or _DEFAULT_HOST).strip().rstrip("/")
    if "//" not in host:
        host = ("http://" if _is_local(host) else "https://") + host
    return f"{host}/api"


def _is_local(host: str) -> bool:
    name = host.split("/", maxsplit=1)[0].split(":", maxsplit=1)[0].lower()
    return name in ("localhost", "127.0.0.1", "::1") or name.endswith(".local")


_unescaped_paren = re.compile(r"(?<!\\)([()])")


def prompt_tag(tag: str, artist: bool = False) -> str:
    """Write a catalogue tag the way the Anima model card asks for it.

    Parentheses are danbooru disambiguators ("sabo (one piece)"), but ComfyUI reads
    them as attention weights - it would drop them and weight "one piece" instead.
    Escaped, they reach the text encoder literally. Artists need an "@" prefix,
    without it the style barely registers. Idempotent, so already escaped or
    prefixed tags stay as they are."""
    tag = _unescaped_paren.sub(r"\\\1", tag.strip())
    if artist and tag and not tag.startswith("@"):
        tag = "@" + tag
    return tag


def user_agent() -> str:
    return f"krita-ai-diffusion/{plugin_version}"


_manager: RequestManager | None = None


def requests() -> RequestManager:
    """A RequestManager of our own for animadex.net. Deliberately not the ComfyUI
    client's one: that may carry an auth bearer and custom headers which must not
    be sent to a third party. Cloudflare sits in front of the site, so identify
    ourselves properly - requests without a User-Agent get challenged."""
    global _manager
    if _manager is None:
        _manager = RequestManager()
        _manager.add_header("User-Agent", user_agent())
        _manager.add_header("Accept", "application/json")
    return _manager


@dataclass
class Lora:
    name: str = ""
    url: str = ""  # CivitAI model page
    thumb_url: str = ""

    @property
    def civitai_model_id(self) -> int:
        # https://civitai.com/models/2672413 (may carry a slug or query after the id)
        part = self.url.rstrip("/").split("/models/")[-1].split("/")[0].split("?")[0]
        return int(part) if part.isdigit() else 0

    @staticmethod
    def from_api(data: dict) -> Lora:
        return Lora(
            name=str(data.get("name") or ""),
            url=str(data.get("url") or ""),
            thumb_url=str(data.get("thumb") or ""),
        )


@dataclass
class Entry:
    """One character or artist. Both endpoints return the same shape, except that
    characters carry copyright/tags/loras and artists carry a quality score."""

    slug: str = ""
    name: str = ""
    trigger: str = ""  # the prompt text that invokes this character/artist
    tags: list[str] = field(default_factory=list)
    copyright: str = ""
    copyright_name: str = ""
    count: int = 0  # number of danbooru posts the model saw for this tag
    score: float = 0.0  # artists only: image quality classifier, 0..1
    source_url: str = ""  # danbooru search for the trigger tag
    thumb_url: str = ""
    image_url: str = ""
    has_image: bool = False
    is_hidden: bool = False
    loras: list[Lora] = field(default_factory=list)
    up_votes: int = 0
    down_votes: int = 0
    fav_count: int = 0
    is_artist: bool = False

    @property
    def prompt(self) -> str:
        """The trigger as it belongs in an Anima prompt (see prompt_tag)."""
        return prompt_tag(self.trigger, artist=self.is_artist)

    @property
    def prompt_with_tags(self) -> str:
        parts = [self.prompt] + [prompt_tag(t) for t in self.tags if t]
        return ", ".join(p for p in parts if p)

    @staticmethod
    def from_api(data: dict, is_artist: bool = False) -> Entry:
        tags = data.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        rating = data.get("rating") or {}
        loras = data.get("loras") or []
        return Entry(
            slug=str(data.get("slug") or ""),
            name=str(data.get("name") or data.get("slug") or ""),
            trigger=str(data.get("trigger") or ""),
            tags=[str(t) for t in tags],
            copyright=str(data.get("copyright") or ""),
            copyright_name=str(data.get("copyright_name") or ""),
            count=int(data.get("count") or 0),
            score=float(data.get("score") or 0.0),
            source_url=str(data.get("url") or ""),
            thumb_url=str(data.get("thumb_url") or ""),
            image_url=str(data.get("img_url") or ""),
            has_image=bool(data.get("has_image")),
            is_hidden=bool(data.get("is_hidden")),
            loras=[Lora.from_api(l) for l in loras if isinstance(l, dict)],
            up_votes=int(rating.get("up") or 0),
            down_votes=int(rating.get("down") or 0),
            fav_count=int(data.get("fav_count") or 0),
            is_artist=is_artist,
        )


@dataclass
class SearchResult:
    entries: list[Entry] = field(default_factory=list)
    total: int = 0
    page: int = 1
    pages: int = 1
    page_size: int = _PAGE_SIZE

    @property
    def has_more(self) -> bool:
        return self.page < self.pages


@dataclass
class FacetValue:
    value: str
    label: str
    count: int = 0


@dataclass
class Facet:
    key: str
    label: str
    values: list[FacetValue] = field(default_factory=list)


def _query(params: list[tuple[str, str]]) -> str:
    return "&".join(f"{quote(k)}={quote(str(v))}" for k, v in params if str(v) != "")


async def _get_json(url: str, timeout: float = 15.0) -> dict:
    data = await requests().get(url, timeout=timeout)
    if isinstance(data, (bytes, bytearray)):
        data = json.loads(data)
    return data if isinstance(data, dict) else {}


async def search(
    mode: str = "characters",
    query: str = "",
    filters: dict[str, list[str]] | None = None,
    sort: str = "count",
    page: int = 1,
    loras_only: bool = False,
    seed: int = 0,
    host: str = "",
) -> SearchResult:
    """Search the catalogue. `mode` is "characters" or "artists"; `filters` maps a
    facet key (see character_facets / artist_facets) to the values to match."""
    assert mode in ("characters", "artists"), f"Unknown animadex mode: {mode}"
    params: list[tuple[str, str]] = [("sort", sort), ("page", str(max(1, page)))]
    if query:
        params.append(("q", query))
    if sort == "random" and seed > 0:
        params.append(("seed", str(seed)))
    if loras_only:
        params.append(("loras", "1"))
    for key, values in (filters or {}).items():
        params.extend((key, v) for v in values)

    url = f"{api_url(host)}/{mode}/search?{_query(params)}"
    try:
        data = await _get_json(url)
    except Exception as e:
        log.warning(f"Animadex search failed: {e}")
        return SearchResult()
    results = data.get("results") or []
    return SearchResult(
        entries=[Entry.from_api(r, mode == "artists") for r in results if isinstance(r, dict)],
        total=int(data.get("total") or 0),
        page=int(data.get("page") or page),
        pages=int(data.get("pages") or 1),
        page_size=int(data.get("page_size") or _PAGE_SIZE),
    )


def _parse_facet(key: str, data: dict) -> Facet:
    values = [
        FacetValue(
            value=str(v.get("value") or ""),
            label=str(v.get("label") or v.get("value") or ""),
            count=int(v.get("count") or 0),
        )
        for v in (data.get("values") or [])
        if isinstance(v, dict)
    ]
    return Facet(key=key, label=str(data.get("label") or key), values=values)


async def fetch_facets(mode: str = "characters", host: str = "") -> list[Facet]:
    """All filter categories with their most common values. The server hard-caps
    this at 30 values per facet - use fetch_facet_values to search the full list."""
    try:
        data = await _get_json(f"{api_url(host)}/{mode}/facets")
    except Exception as e:
        log.warning(f"Could not fetch animadex facets: {e}")
        return []
    facets = data.get("facets") or {}
    order = character_facets if mode == "characters" else artist_facets
    keys = [k for k in order if k in facets] + [k for k in facets if k not in order]
    return [_parse_facet(k, facets[k]) for k in keys if isinstance(facets[k], dict)]


async def fetch_facet_values(
    facet: str, query: str = "", mode: str = "characters", host: str = ""
) -> Facet:
    """Values of a single facet, optionally narrowed by a search term. Needed for
    facets with thousands of entries (character, copyright, artist)."""
    params = _query([("q", query)])
    url = f"{api_url(host)}/{mode}/facet/{quote(facet)}" + (f"?{params}" if params else "")
    try:
        return _parse_facet(facet, await _get_json(url))
    except Exception as e:
        log.warning(f"Could not fetch animadex facet '{facet}': {e}")
        return Facet(key=facet, label=facet)


# Hosts where connections died before any HTTP response. Images and the export
# live on a separate host (blobs.animadex.net), which antivirus web filters block
# by its TLS server name while letting animadex.net itself through - browsers still
# get there because they hide the name with Encrypted Client Hello, Qt can't.
unreachable_hosts: set[str] = set()


def host_of(url: str) -> str:
    return urlsplit(url).hostname or ""


def is_connection_failure(e: Exception) -> bool:
    """True if no HTTP response came back at all (refused, reset, TLS failure)."""
    return isinstance(e, NetworkError) and not e.status


def connection_hint(url: str) -> str:
    host = host_of(url)
    return _(
        "{host} closed the connection before answering. This usually means a firewall or"
        " antivirus web filter (e.g. Bitdefender Online Threat Prevention) blocks it for"
        " Krita while the browser gets through - add {host} to its exceptions."
    ).format(host=host)


async def fetch_thumbnail(url: str) -> bytes | None:
    """Preview image bytes (served from a separate blob host). None on error."""
    if not url:
        return None
    try:
        result = await requests().download(url, timeout=8.0)
        unreachable_hosts.discard(host_of(url))
        return bytes(result) if result else None
    except Exception as e:
        if is_connection_failure(e):
            host = host_of(url)
            if host not in unreachable_hosts:  # once, not for every tile
                unreachable_hosts.add(host)
                log.warning(f"Could not fetch animadex image {url}: {e} - {connection_hint(url)}")
        else:
            log.warning(f"Could not fetch animadex image {url}: {e}")
        return None


def _facet_cache_path(mode: str, host: str) -> Path:
    url_hash = hashlib.md5(api_url(host).encode()).hexdigest()[:8]
    return util.user_data_dir / f"animadex_facets_{mode}_{url_hash}.json"


def load_cached_facets(mode: str = "characters", host: str = "") -> list[Facet] | None:
    """Cached facet lists if present and fresh, else None. Search results are not
    cached: 36k entries reshuffle with every filter, and the server pages them."""
    path = _facet_cache_path(mode, host)
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != _FACET_CACHE_VERSION:
            return None
        if time.time() - data.get("timestamp", 0) > _FACET_CACHE_MAX_AGE:
            return None
        return [
            Facet(
                key=f["key"],
                label=f["label"],
                values=[FacetValue(**v) for v in f.get("values", [])],
            )
            for f in data.get("facets", [])
        ]
    except Exception as e:
        log.warning(f"Could not load animadex facet cache: {e}")
        return None


def save_facet_cache(facets: list[Facet], mode: str = "characters", host: str = ""):
    path = _facet_cache_path(mode, host)
    try:
        data = {
            "version": _FACET_CACHE_VERSION,
            "timestamp": time.time(),
            "facets": [asdict(f) for f in facets],
        }
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception as e:
        log.warning(f"Could not save animadex facet cache: {e}")


def clear_facet_cache(host: str = ""):
    for mode in ("characters", "artists"):
        try:
            _facet_cache_path(mode, host).unlink(missing_ok=True)
        except Exception as e:
            log.warning(f"Could not clear animadex facet cache: {e}")
