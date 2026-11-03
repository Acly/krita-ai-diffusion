"""Offline copy of the AnimaDex catalogue (see animadex.py for the live API).

The official "offline dataset export" hands out a manifest in exchange for a
personal token (animadex.net -> Account -> Offline dataset export). Everything
the manifest points to - the two catalogue CSVs, a per-row version index and
the thumbnails - is served publicly from R2 without the token. So we import
only the metadata (a few MB), search it locally, and download thumbnails lazily
as they are shown instead of pulling the multi-GB image set up front.

Row parsing mirrors animadex/db.py (parse_character_row / parse_artist_row) of
github.com/zetaneko/AnimaDex so facets and file names match the site."""

from __future__ import annotations

import csv
import io
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from .. import util
from ..util import client_logger as log
from . import animadex
from .animadex import Entry, Facet, FacetValue, SearchResult
from .network import NetworkError, RequestManager

_ILLEGAL_FS_CHARS = '<>:"/\\|?*'

HAIR_COLOR_TAGS = {
    "aqua hair", "black hair", "blonde hair", "blue hair", "brown hair",
    "dark blue hair", "gradient hair", "green hair", "grey hair",
    "light blue hair", "light brown hair", "light green hair",
    "light purple hair", "multicolored hair", "orange hair", "pink hair",
    "purple hair", "red hair", "silver hair", "split-color hair",
    "streaked hair", "two-tone hair", "white hair",
}  # fmt: skip
HAIR_LENGTH_ORDER = (
    "very short hair", "short hair", "medium hair",
    "long hair", "very long hair", "absurdly long hair",
)  # fmt: skip
EYE_COLOR_TAGS = {
    "aqua eyes", "black eyes", "blue eyes", "brown eyes", "gradient eyes",
    "green eyes", "grey eyes", "multicolored eyes", "orange eyes",
    "pink eyes", "purple eyes", "red eyes", "two-tone eyes", "yellow eyes",
}  # fmt: skip
GENDER_TAGS = {"1boy": "Male", "1girl": "Female", "1other": "Ambiguous", "no humans": "Non-Human"}

# (key, label, lower bound inclusive, upper bound exclusive)
SCORE_BUCKETS = (
    ("5", "50% and up", 0.50, 1.01),
    ("4", "40 - 50%", 0.40, 0.50),
    ("3", "30 - 40%", 0.30, 0.40),
    ("2", "20 - 30%", 0.20, 0.30),
    ("1", "Under 20%", 0.00, 0.20),
)

facet_labels = {
    "character": "Character",
    "copyright": "Copyright",
    "hair_color": "Hair Color",
    "hair_length": "Hair Length",
    "eye_color": "Eye Color",
    "gender": "Gender",
    "artist": "Artist",
    "score": "Score",
}
# Artist categories are hand-curated on the site and not part of the export, and
# neither are LoRA links, votes or favourites - those stay live-API only.
offline_facets = {
    "characters": ["character", "copyright", "eye_color", "gender", "hair_color", "hair_length"],
    "artists": ["artist", "score"],
}
_trait_facets = ("hair_color", "hair_length", "eye_color", "gender")
_hair_length_rank = {t: i for i, t in enumerate(HAIR_LENGTH_ORDER)}

_STATE_VERSION = 1
_modes = {
    # mode -> (csv file, index key, manifest thumbnail prefix key)
    "characters": ("characters.csv", "chars", "char_thumb"),
    "artists": ("artists.csv", "artists", "artist_thumb"),
}


def sanitize_filename(name: str) -> str:
    cleaned = "".join("_" if c in _ILLEGAL_FS_CHARS else c for c in name)
    return cleaned.rstrip(" .") or "unnamed"


def titlecase(text: str) -> str:
    def cap(word: str):
        for i, ch in enumerate(word):
            if ch.isalpha():
                return word[:i] + ch.upper() + word[i + 1 :]
        return word

    return " ".join(cap(w) for w in text.split(" "))


def _trait_label(tag: str) -> str:
    return titlecase(tag.rsplit(" ", 1)[0])


def _int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _float(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _bool(value) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes")


@dataclass
class Record:
    entry: Entry
    stem: str  # file name of the images, without extension
    search_blob: str
    name_lower: str
    trait_sets: dict[str, set[str]] = field(default_factory=dict)  # facet -> tags


def parse_character_row(row: dict) -> Record | None:
    character = (row.get("character") or "").strip()
    if not character:
        return None
    copyright_ = (row.get("copyright") or "").strip()
    trigger = (row.get("trigger") or "").strip()
    core = (row.get("core_tags") or "").strip()
    if ", " in trigger:
        nm, cp = trigger.split(", ", 1)
    else:
        nm, cp = trigger, ""
    name = titlecase(nm) if nm else titlecase(character.replace("_", " "))
    copyright_name = titlecase(cp) if cp else titlecase(copyright_.replace("_", " "))
    tags = [t.strip() for t in core.split(",") if t.strip()]

    trait_sets: dict[str, set[str]] = {}
    for t in tags:
        if t in HAIR_COLOR_TAGS:
            trait_sets.setdefault("hair_color", set()).add(t)
        elif t in _hair_length_rank:
            trait_sets.setdefault("hair_length", set()).add(t)
        elif t in EYE_COLOR_TAGS:
            trait_sets.setdefault("eye_color", set()).add(t)
        elif t in GENDER_TAGS:
            trait_sets.setdefault("gender", set()).add(t)

    entry = Entry(
        slug=character,
        name=name,
        trigger=trigger or character.replace("_", " "),
        tags=tags,
        copyright=copyright_,
        copyright_name=copyright_name,
        count=_int(row.get("count")),
        source_url=(row.get("url") or "").strip(),
        is_hidden=_bool(row.get("is_hidden") or row.get("hidden")),
    )
    return Record(
        entry=entry,
        stem=sanitize_filename(trigger or character),  # as scripts/import_from_site.py
        search_blob=" ".join((character, copyright_, trigger, core)).lower(),
        name_lower=name.lower(),
        trait_sets=trait_sets,
    )


def parse_artist_row(row: dict) -> Record | None:
    artist = (row.get("artist") or "").strip()
    if not artist:
        return None
    trigger = (row.get("trigger") or "").strip() or artist.replace("_", " ")
    name = titlecase(trigger)
    entry = Entry(
        slug=artist,
        name=name,
        trigger=trigger,
        count=_int(row.get("count")),
        score=_float(row.get("score")),
        is_artist=True,
        source_url=(row.get("url") or "").strip(),
        is_hidden=_bool(row.get("is_hidden") or row.get("hidden")),
    )
    return Record(
        entry=entry,
        stem=sanitize_filename(trigger),
        search_blob=" ".join((artist, trigger)).lower(),
        name_lower=name.lower(),
    )


def parse_csv(text: str, mode: str) -> list[Record]:
    parse = parse_character_row if mode == "characters" else parse_artist_row
    records = []
    for row in csv.DictReader(io.StringIO(text)):
        record = parse(row)
        if record is not None:
            records.append(record)
    return records


def parse_query(q: str) -> list[str]:
    """Comma-separated terms, all of which must match (same as the site)."""
    return [t.strip().lower() for t in q.split(",") if t.strip()]


def _score_bucket(score: float) -> str:
    for key, _, lo, hi in SCORE_BUCKETS:
        if lo <= score < hi:
            return key
    return ""


def wildcard_lines(entries: list[Entry], with_tags: bool = False) -> list[str]:
    """One prompt line per entry, for writing a __wildcard__ file."""
    lines = [e.prompt_with_tags if with_tags else e.prompt for e in entries]
    return list(dict.fromkeys(l for l in lines if l))


@dataclass
class CatalogState:
    version: str = ""  # catalogue version reported by the manifest
    imported_at: float = 0.0
    host: str = ""
    manifest: dict = field(default_factory=dict)  # public R2 URLs only, never the token
    versions: dict[str, dict[str, int]] = field(default_factory=dict)  # index key -> slug -> ver


@dataclass
class ImportSummary:
    characters: int = 0
    artists: int = 0
    changed: int = 0  # rows whose version differs from the previous import
    csv_updated: bool = False


class ImportFailed(Exception):
    """kind: "token" (missing/rejected), "unpublished", "network" or "format"."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class Catalog:
    default_folder = util.user_data_dir / "animadex"

    def __init__(self, folder: Path | None = None):
        self.folder = folder or self.default_folder
        self.state = self._load_state()
        self._records: dict[str, list[Record]] = {}
        self._by_slug: dict[str, dict[str, Record]] = {}

    # --- state / files -----------------------------------------------------

    @property
    def state_path(self):
        return self.folder / "state.json"

    def csv_path(self, mode: str):
        return self.folder / _modes[mode][0]

    def thumb_dir(self, mode: str):
        return self.folder / "thumbs" / mode

    @property
    def available(self) -> bool:
        return self.csv_path("characters").exists() or self.csv_path("artists").exists()

    def _load_state(self) -> CatalogState:
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                if data.pop("state_version", None) == _STATE_VERSION:
                    return CatalogState(**data)
        except Exception as e:
            log.warning(f"Could not read AnimaDex catalogue state: {e}")
        return CatalogState()

    def _save_state(self):
        data = {"state_version": _STATE_VERSION, **self.state.__dict__}
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".part")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.state_path)

    def _write_atomic(self, path: Path, data: bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)

    def clear(self):
        """Forget the imported catalogue, including cached thumbnails."""
        for mode in _modes:
            self.csv_path(mode).unlink(missing_ok=True)
            for path in self.thumb_dir(mode).glob("*.webp"):
                path.unlink(missing_ok=True)
        self.state_path.unlink(missing_ok=True)
        self.state = CatalogState()
        self.invalidate()

    def invalidate(self):
        self._records.clear()
        self._by_slug.clear()

    # --- loading -------------------------------------------------------------

    def records(self, mode: str) -> list[Record]:
        if mode not in self._records:
            path = self.csv_path(mode)
            records = []
            if path.exists():
                try:
                    records = parse_csv(path.read_text(encoding="utf-8"), mode)
                except Exception as e:
                    log.warning(f"Could not read AnimaDex {mode} catalogue: {e}")
            self._records[mode] = records
            self._by_slug[mode] = {r.entry.slug: r for r in records}
            for r in records:
                r.entry.thumb_url = self.thumb_url(mode, r)
                r.entry.has_image = bool(r.entry.thumb_url)
        return self._records[mode]

    def record(self, mode: str, slug: str) -> Record | None:
        self.records(mode)
        return self._by_slug.get(mode, {}).get(slug)

    def count(self, mode: str) -> int:
        return len(self.records(mode))

    # --- search --------------------------------------------------------------

    def _matches(self, r: Record, terms: list[str], filters: dict[str, list[str]]):
        if any(t not in r.search_blob for t in terms):
            return False
        for key, values in filters.items():
            if not values:
                continue
            wanted = set(values)
            if key in ("character", "artist"):
                if r.entry.slug not in wanted:
                    return False
            elif key == "copyright":
                if r.entry.copyright not in wanted:
                    return False
            elif key in _trait_facets:
                if not (r.trait_sets.get(key, set()) & wanted):
                    return False
            elif key == "score":
                if _score_bucket(r.entry.score) not in wanted:
                    return False
        return True

    def search(
        self,
        mode: str = "characters",
        query: str = "",
        filters: dict[str, list[str]] | None = None,
        sort: str = "count",
        page: int = 1,
        page_size: int = 48,
        seed: int = 0,
        show_hidden: bool = False,
    ) -> SearchResult:
        terms = parse_query(query)
        filters = filters or {}
        hits = [
            r
            for r in self.records(mode)
            if (show_hidden or not r.entry.is_hidden) and self._matches(r, terms, filters)
        ]
        if sort == "random":
            hits.sort(key=lambda r: r.entry.slug)
            random.Random(seed or 1).shuffle(hits)
        elif sort == "az":
            hits.sort(key=lambda r: r.name_lower)
        elif sort == "score":
            hits.sort(key=lambda r: (-r.entry.score, r.name_lower))
        else:  # "count", and site-only sorts (liked, favourited, recent) fall back to it
            hits.sort(key=lambda r: (-r.entry.count, r.name_lower))

        total = len(hits)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, page), pages)
        start = (page - 1) * page_size
        return SearchResult(
            entries=[r.entry for r in hits[start : start + page_size]],
            total=total,
            page=page,
            pages=pages,
            page_size=page_size,
        )

    def all_matching(
        self, mode: str, query: str = "", filters: dict[str, list[str]] | None = None
    ) -> list[Entry]:
        """Every hit of a search, unpaged - e.g. to export it as a wildcard file."""
        result = self.search(mode, query, filters, page_size=max(1, self.count(mode)))
        return result.entries

    # --- facets --------------------------------------------------------------

    def _facet_counts(self, mode: str, facet: str) -> dict[str, tuple[str, int]]:
        counts: dict[str, tuple[str, int]] = {}  # value -> (label, count)

        def add(value: str, label: str, n: int = 1):
            prev = counts.get(value)
            counts[value] = (label, (prev[1] if prev else 0) + n)

        for r in self.records(mode):
            e = r.entry
            if e.is_hidden:
                continue
            if facet in ("character", "artist"):
                add(e.slug, e.name, e.count)  # the site ranks these by popularity
            elif facet == "copyright":
                if e.copyright:
                    add(e.copyright, e.copyright_name)
            elif facet in _trait_facets:
                for tag in r.trait_sets.get(facet, ()):
                    label = GENDER_TAGS[tag] if facet == "gender" else _trait_label(tag)
                    add(tag, label)
            elif facet == "score":
                bucket = _score_bucket(e.score)
                if bucket and e.score > 0:
                    add(bucket, next(b[1] for b in SCORE_BUCKETS if b[0] == bucket))
        return counts

    def facet_values(
        self, facet: str, query: str = "", mode: str = "characters", limit: int = 30
    ) -> Facet:
        q = query.strip().lower()
        items = [
            FacetValue(value=v, label=label, count=n)
            for v, (label, n) in self._facet_counts(mode, facet).items()
            if not q or q in v.lower() or q in label.lower()
        ]
        if facet == "hair_length":
            items.sort(key=lambda fv: _hair_length_rank.get(fv.value, 99))
        elif facet == "score":
            items.sort(key=lambda fv: fv.value, reverse=True)
        else:
            items.sort(key=lambda fv: (-fv.count, fv.label.lower()))
        if limit > 0:
            items = items[:limit]
        return Facet(key=facet, label=facet_labels.get(facet, facet), values=items)

    def facets(self, mode: str = "characters", limit: int = 30) -> list[Facet]:
        return [self.facet_values(f, "", mode, limit) for f in offline_facets[mode]]

    # --- thumbnails ------------------------------------------------------------

    def thumb_url(self, mode: str, record: Record) -> str:
        m = self.state.manifest
        base = str(m.get("r2_base") or "").rstrip("/")
        prefix = str((m.get("prefixes") or {}).get(_modes[mode][2]) or "").strip("/")
        if not base or not prefix:
            return ""
        # Encoding must match the site: spaces become %20, parentheses stay literal.
        return f"{base}/{prefix}/{quote(record.stem + '.webp', safe='()')}"

    def thumb_path(self, mode: str, record: Record) -> Path:
        return self.thumb_dir(mode) / (record.stem + ".webp")

    async def thumbnail(self, mode: str, entry: Entry) -> bytes | None:
        """Thumbnail bytes from the local cache, downloading them on first use."""
        record = self.record(mode, entry.slug)
        if record is None:
            return None
        path = self.thumb_path(mode, record)
        if path.exists():
            try:
                return path.read_bytes()
            except Exception as e:
                log.warning(f"Could not read cached thumbnail {path}: {e}")
        data = await animadex.fetch_thumbnail(record.entry.thumb_url)
        if data:
            try:
                self._write_atomic(path, data)
            except Exception as e:
                log.warning(f"Could not cache thumbnail {path}: {e}")
        return data

    def cached_thumbnail_count(self) -> int:
        return sum(len(list(self.thumb_dir(m).glob("*.webp"))) for m in _modes)

    # --- import --------------------------------------------------------------

    async def update(
        self, token: str = "", host: str = "", progress: Callable[[str], None] | None = None
    ) -> ImportSummary:
        """Import the catalogue, or bring an existing import up to date.

        With a token a fresh manifest is requested. Without one the manifest saved
        by the last import is reused - its R2 URLs are public, so that works for as
        long as the site keeps them stable; if it stops working, ImportFailed with
        kind "token" asks for a new token. Only the delta manifest is requested:
        its URLs point to the complete CSVs as well, and unlike ?full=1 it is not
        limited to once every 48 hours. Thumbnails are not downloaded here."""

        def report(text: str):
            log.info(f"AnimaDex import: {text}")
            if progress:
                progress(text)

        host = host or self.state.host
        if token:
            report("Requesting manifest")
            manifest = await fetch_manifest(token, host)
        elif self.state.manifest:
            manifest = self.state.manifest
        else:
            raise ImportFailed("token", "An export token is needed for the first import.")

        csv_urls = manifest.get("csv") or {}
        index_url = str(manifest.get("index_url") or "")
        if not isinstance(csv_urls, dict) or not csv_urls.get("characters") or not index_url:
            raise ImportFailed("format", "The export manifest has an unexpected format.")

        report("Checking for changes")
        try:
            index = await _download_json(index_url)
        except NetworkError as e:
            if not token and e.status in (401, 403, 404, 410):
                raise ImportFailed("token", "Saved export links expired, a new token is needed.")
            raise ImportFailed("network", _network_message(e, index_url, "the catalogue index"))

        summary = ImportSummary()
        new_versions: dict[str, dict[str, int]] = {}
        stale: dict[str, set[str]] = {}
        for mode, (_, key, _) in _modes.items():
            current = {str(k): _int(v) for k, v in (index.get(key) or {}).items()}
            previous = self.state.versions.get(key, {})
            new_versions[key] = current
            stale[mode] = {s for s, v in current.items() if previous.get(s) not in (None, v)}
            summary.changed += sum(1 for s, v in current.items() if previous.get(s) != v)
            summary.changed += sum(1 for s in previous if s not in current)

        version = str(manifest.get("version") or "")
        needs_csv = (
            not all(self.csv_path(m).exists() for m in _modes if csv_urls.get(m))
            or summary.changed > 0
            or (bool(version) and version != self.state.version)
        )
        if needs_csv:
            for mode in _modes:
                url = csv_urls.get(mode)
                if not url:
                    continue
                report(f"Downloading {mode}")
                try:
                    data = await animadex.requests().download(str(url), timeout=180.0)
                except NetworkError as e:
                    if not token and e.status in (401, 403, 404, 410):
                        raise ImportFailed(
                            "token", "Saved export links expired, a new token is needed."
                        )
                    raise ImportFailed("network", _network_message(e, str(url), mode))
                text = bytes(data).decode("utf-8-sig")
                if not parse_csv(text, mode):
                    raise ImportFailed("format", f"The {mode} catalogue is empty or unreadable.")
                self._write_atomic(self.csv_path(mode), text.encode("utf-8"))
            summary.csv_updated = True

        self.invalidate()
        # Changed rows may come with a regenerated image: drop the cached thumbnail
        # so the next view fetches the new one. File names derive from the trigger.
        for mode, slugs in stale.items():
            for slug in slugs:
                record = self.record(mode, slug)
                if record is not None:
                    self.thumb_path(mode, record).unlink(missing_ok=True)

        keep = ("r2_base", "prefixes", "csv", "index_url", "version")
        self.state = CatalogState(
            version=version,
            imported_at=time.time(),
            host=host,
            manifest={k: manifest[k] for k in keep if manifest.get(k)},
            versions=new_versions,
        )
        self._save_state()
        self.invalidate()
        summary.characters = self.count("characters")
        summary.artists = self.count("artists")
        report(f"{summary.characters} characters, {summary.artists} artists")
        return summary

    _instance: Catalog | None = None

    @classmethod
    def instance(cls) -> Catalog:
        if cls._instance is None:
            cls._instance = Catalog()
        return cls._instance


def _network_message(e: Exception, url: str, what: str) -> str:
    log.warning(f"AnimaDex import: could not download {url}: {e}")
    message = f"Could not download {what}: {e}"
    if animadex.is_connection_failure(e):
        message += "\n\n" + animadex.connection_hint(url)
    return message


async def _download_json(url: str) -> dict:
    data = await animadex.requests().download(url, timeout=60.0)
    result = json.loads(bytes(data).decode("utf-8-sig"))
    return result if isinstance(result, dict) else {}


async def fetch_manifest(token: str, host: str = "") -> dict:
    # A short-lived manager: the token must only ever go to this one endpoint,
    # never ride along on thumbnail or R2 requests.
    requests = RequestManager()
    requests.add_header("User-Agent", animadex.user_agent())
    requests.add_header("X-Export-Token", token.strip())
    url = f"{animadex.api_url(host)}/export/manifest"
    try:
        data = await requests.get(url, timeout=30.0)
    except NetworkError as e:
        if e.status == 401:
            raise ImportFailed("token", "The export token was rejected - generate a new one.")
        if e.status == 503:
            raise ImportFailed("unpublished", "The site has not published an export yet.")
        raise ImportFailed("network", f"Could not reach AnimaDex: {e}")
    if isinstance(data, (bytes, bytearray)):
        try:
            data = json.loads(data)
        except ValueError:
            data = None
    if not isinstance(data, dict) or not data.get("r2_base"):
        raise ImportFailed("format", "The export manifest has an unexpected format.")
    return data
