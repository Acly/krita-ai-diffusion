import asyncio
import hashlib
import json
from base64 import b64decode, b64encode
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum, Flag
from pathlib import Path
from typing import Any, NamedTuple, cast

from PyQt6.QtCore import QAbstractListModel, QModelIndex, QSortFilterProxyModel, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QImage

from . import eventloop
from .backend.network import NetworkError, RequestManager
from .backend.resources import Arch
from .settings import settings
from .util import client_logger as log
from .util import encode_json, read_json_with_comments, user_data_dir


class FileSource(Flag):
    unavailable = 0
    local = 1
    remote = 2


class FileFormat(Enum):
    unknown = 0
    checkpoint = 1  # TE + VAE + Diffusion model
    diffusion = 2  # Diffusion model only
    lora = 3


@dataclass
class File:
    id: str
    name: str
    source: FileSource = FileSource.unavailable
    format: FileFormat = FileFormat.unknown
    icon: QIcon | None = None
    hash: str | None = None
    path: Path | None = None
    size: int | None = None
    metadata: dict[str, Any] | None = None
    civitai: int | str | None = None
    arch: Arch | None = None
    version: str | None = None
    image: str | None = None

    @property
    def display_name(self):
        if self.format is FileFormat.lora:
            return File.remote(self.id).name
        return self.name

    @property
    def full_name(self):
        if self.civitai and self.civitai != "not-found":
            return f"{self.name} {self.version or ''}".strip()
        return self.id

    @staticmethod
    def remote(id: str, format=FileFormat.unknown):
        name = id.replace("\\", "/")
        dot = name.rfind(".")
        name = name if dot == -1 else name[:dot]
        return File(id, name, FileSource.remote, format)

    @staticmethod
    def local(path: Path, format=FileFormat.unknown, compute_hash=False):
        file = File(path.name, path.stem, FileSource.local, format, path=path)
        file.size = path.stat().st_size
        if compute_hash:
            file.compute_hash()
        return file

    @staticmethod
    def from_dict(data: dict):
        data["source"] = FileSource(data["source"])
        if isinstance(data.get("format"), str):
            data["format"] = FileFormat[data["format"]]
        data["path"] = Path(data["path"]) if data.get("path") else None
        if data.get("arch"):
            data["arch"] = Arch[data["arch"]]
        return File(**data)

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k != "icon" and v is not None}

    def update(self, other: "File"):
        assert self.id == other.id, "Cannot update file with different id"
        has_changes = self != other
        if has_changes:
            self.source = self.source | other.source
            self.icon = other.icon or self.icon
            self.hash = other.hash or self.hash
            self.path = other.path or self.path
            self.size = other.size or self.size
        return has_changes

    def compute_hash(self):
        if self.hash:
            return self.hash
        assert self.path is not None, "Local filepath must be set to compute hash"
        sha = hashlib.sha256()
        with open(self.path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                sha.update(chunk)
        self.hash = b64encode(sha.digest()).decode()
        return self.hash

    def meta(self, key: str, default=None) -> Any:
        return self.metadata.get(key, default) if self.metadata else default


class FileCollection(QAbstractListModel):
    source_role = Qt.ItemDataRole.UserRole + 1
    file_role = Qt.ItemDataRole.UserRole + 2
    search_role = Qt.ItemDataRole.UserRole + 3
    collected = pyqtSignal()

    def __init__(self, database: Path | None = None, parent=None):
        super().__init__(parent)

        self._database = database
        self._files: list[File] = []
        self.load()

    def rowCount(self, parent: QModelIndex | None = None):
        return len(self._files)

    def data(self, index: QModelIndex, role: int = 0):
        if 0 <= index.row() < len(self._files):
            item = self._files[index.row()]

            match role:
                case Qt.ItemDataRole.DisplayRole | Qt.ItemDataRole.EditRole:
                    return item.display_name
                case Qt.ItemDataRole.DecorationRole:
                    return item.icon
                case Qt.ItemDataRole.UserRole | Qt.ItemDataRole.ToolTipRole:
                    return item.id
                case FileCollection.source_role:
                    return item.source.value
                case FileCollection.file_role:
                    return item
                case FileCollection.search_role:
                    return f"{item.display_name}\n{item.id}"
                case _:
                    return None

    def extend(self, input_files: Sequence[File]):
        if len(input_files) == 0:
            return

        existing = {f.id: (i, f) for i, f in enumerate(self._files)}
        new_files = []
        for f in input_files:
            if lookup := existing.get(f.id):
                idx, existing_file = lookup
                if existing_file.update(f):
                    self.dataChanged.emit(self.index(idx), self.index(idx))
            else:
                new_files.append(f)

        if len(new_files) > 0:
            end = len(self._files)
            self.beginInsertRows(QModelIndex(), end, end + len(new_files) - 1)
            self._files.extend(new_files)
            self.endInsertRows()

        self.save()
        self.collected.emit()

    def remove(self, index: int):
        self.beginRemoveRows(QModelIndex(), index, index)
        del self._files[index]
        self.endRemoveRows()
        self.save()

    def update(self, new_files: Sequence[File], source: FileSource):
        new_ids = {f.id for f in new_files}
        for i, f in enumerate(self._files):
            if source in f.source and f.id not in new_ids:
                f.source = f.source & (~source)
                self.dataChanged.emit(self.index(i), self.index(i))

        self.extend(new_files)

    def add(self, file: File):
        self.extend([file])
        return file

    def find(self, id: str):
        return next((f for f in self if f.id == id), None)

    def find_local(self, id: str):
        return next((f for f in self if f.id == id and FileSource.local in f.source), None)

    def find_index(self, id: str):
        return next((i for i, f in enumerate(self) if f.id == id), -1)

    def set_meta(self, file: File, key: str, value: Any):
        if not file.metadata:
            file.metadata = {}
        file.metadata[key] = value
        self.notify(file)

    def notify(self, file: File):
        self.save()
        idx = self.find_index(file.id)
        if idx >= 0:
            self.dataChanged.emit(self.index(idx), self.index(idx))

    def image_path(self, file: File):
        if self._database and file.image:
            return self._database.parent / "images" / file.image
        return None

    def set_image(self, file: File, image: QImage):
        if image.isNull() or not self._database:
            raise ValueError("Invalid preview image or missing database")
        side = min(image.width(), image.height())
        image = image.copy((image.width() - side) // 2, (image.height() - side) // 2, side, side)
        image = image.scaled(
            256,
            256,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        image = image.convertToFormat(QImage.Format.Format_RGB32)
        # Preserve subfolders, but never allow a remote identifier to escape the image directory.
        parts = file.id.replace("\\", "/").split("/")
        parts = [p.replace(":", "_") for p in parts if p not in {"", ".", ".."}]
        filename = "/".join(parts) + ".jpg"
        path = self._database.parent / "images" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not image.save(str(path), "JPG", 90):
            raise OSError(f"Failed to save preview image: {path}")
        file.image = filename
        self.notify(file)

    def load(self):
        if not self._database or not self._database.exists():
            return
        try:
            data = read_json_with_comments(self._database)
            self.extend([File.from_dict(f) for f in data])
            log.info(f"Loaded {len(self)} model files from {self._database}")
            self._remove_missing_files()
        except Exception as e:
            log.error(f"Failed to read {self._database}: {e}")

    def save(self):
        if self._database:
            db = [f.to_dict() for f in self._files]
            self._database.parent.mkdir(parents=True, exist_ok=True)
            self._database.write_text(json.dumps(db, indent=2, default=encode_json))

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return super().flags(index) | Qt.ItemFlag.ItemIsDropEnabled
        return super().flags(index) | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled

    def _remove_missing_files(self):
        i = 0
        while i < len(self._files):
            f = self._files[i]
            is_valid = True
            if f.source is FileSource.local and f.path:
                try:
                    is_valid = f.path.exists()
                except OSError:
                    is_valid = False
            if not is_valid:
                log.warning(f"Local file {f.path} inaccessible, removing from collection")
                self.remove(i)
            else:
                i += 1

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index: int):
        return self._files[index]


class LoraInfoWorker:
    def __init__(self, files: FileCollection, remote_hash: Callable[[str], Awaitable[str | None]]):
        self.files = files
        self.remote_hash = remote_hash
        self.last_error = ""
        self._connection_errors = 0
        self._task: asyncio.Task | None = None
        self._pending = False
        self._requests = RequestManager()
        files.collected.connect(self.start)
        settings.changed.connect(self._settings_changed)

    def _settings_changed(self, name: str, value):
        if name == "civitai_collection" and value:
            self.start()

    def start(self):
        if self._task and not self._task.done():
            self._pending = True
        else:
            self._task = eventloop.run(self.run())

    async def _handle_file(self, file: File):
        if not file.hash:
            if FileSource.local in file.source and file.path:
                # Hash a copy: only the main thread mutates the model.
                copy = File.local(file.path)
                file.hash = await asyncio.to_thread(copy.compute_hash)
            elif FileSource.remote in file.source:
                file.hash = await self.remote_hash(file.id)
            if file.hash:
                self.files.notify(file)
        if file.hash and settings.civitai_collection and self._connection_errors < 3:
            await self.collect(file)

    async def run(self):
        try:
            while True:
                self._pending = False
                for file in list(self.files):
                    if file.civitai is not None or file.source is FileSource.unavailable:
                        continue
                    try:
                        await self._handle_file(file)
                    except Exception as e:
                        log.warning(f"Failed to collect LoRA information for {file.id}: {e}")
                if not self._pending:
                    break
        except Exception as e:
            log.warning(f"LoRA information worker failed: {e}")

    async def collect(self, file: File):
        assert file.hash
        if not settings.civitai_collection or self._connection_errors >= 3:
            return
        try:
            digest = b64decode(file.hash).hex()
            info = await self._requests.get(
                f"https://civitai.com/api/v1/model-versions/by-hash/{digest}", timeout=30
            )
            name, version, model_id = info["model"]["name"], info["name"], info["modelId"]
            file.name = name
            file.version = version
            file.civitai = model_id
            file.arch = civitai_arch(info.get("baseModel", ""))
            if not file.meta("lora_triggers") and info.get("trainedWords"):
                self.files.set_meta(file, "lora_triggers", ", ".join(info["trainedWords"]))
            self.files.notify(file)
            log.info(f"Civitai LoRA collected: {file.name} {file.version}, id {file.civitai}")
            images = info.get("images", [])
            if images and not file.image and settings.civitai_collection:
                data = await self._requests.download(images[0]["url"], timeout=30)
                # A custom preview set while the download was pending takes precedence.
                if not file.image:
                    self.files.set_image(file, QImage.fromData(data))
        except NetworkError as e:
            if e.status == 404 and file.civitai is None:
                file.civitai = "not-found"
                self.files.notify(file)
            else:
                self.last_error = str(e)
                self.files.set_meta(file, "civitai_error", self.last_error)
                if e.status is None or e.status == 429 or e.status >= 500:
                    self._connection_errors += 1
                log.warning(f"Failed to retrieve Civitai information for {file.id}: {e}")
        except Exception as e:
            self.last_error = str(e)
            self.files.set_meta(file, "civitai_error", self.last_error)
            log.warning(f"Failed to retrieve Civitai information for {file.id}: {e}")


def civitai_arch(base_model: str):
    name = base_model.lower()
    if name.startswith(("zimage", "z-image")):
        return Arch.zimage
    if name.startswith("sd 1"):
        return Arch.sd15
    if "illustrious" in name or "noobai" in name:
        return Arch.illu_v if "v-pred" in name else Arch.illu
    if name.startswith(("sdxl", "pony")) or name == "odora":
        return Arch.sdxl
    if name.startswith("sd 3"):
        return Arch.sd3
    if "flux" in name:
        if "klein" in name:
            return Arch.flux2_9b if "9b" in name else Arch.flux2_4b
        return Arch.flux_k if "kontext" in name else Arch.flux
    if "qwen" in name:
        if "edit" in name:
            return Arch.qwen_e_p if "2509" in name or "2511" in name else Arch.qwen_e
        return Arch.qwen2 if "2512" in name else Arch.qwen
    return next((arch for arch in Arch if arch.value.lower() == name), None)


_instance = None


class FileLibrary(NamedTuple):
    checkpoints: FileCollection
    loras: FileCollection

    @staticmethod
    def load(database_dir: Path | None = None):
        global _instance
        database_dir = database_dir or user_data_dir / "database"
        _instance = FileLibrary(
            checkpoints=FileCollection(),
            loras=FileCollection(database_dir / "loras.json"),
        )
        return _instance

    @staticmethod
    def instance():
        if not _instance:
            return FileLibrary.load()
        return _instance


class FileFilter(QSortFilterProxyModel):
    def __init__(self, source: FileCollection, parent=None):
        super().__init__(parent)
        self._available_only = False
        self._name_prefix = ""
        self._arch = Arch.auto
        self._excluded_files: set[str] = set()
        self.setSourceModel(source)
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.sort(0)

    @property
    def available_only(self):
        return self._available_only

    @available_only.setter
    def available_only(self, value):
        self._available_only = value
        self.invalidateFilter()

    @property
    def name_prefix(self):
        return self._name_prefix

    @name_prefix.setter
    def name_prefix(self, value):
        self._name_prefix = value
        self.invalidateFilter()

    @property
    def arch(self):
        return self._arch

    @arch.setter
    def arch(self, value: Arch):
        self._arch = value
        self.invalidateFilter()

    @property
    def excluded_files(self):
        return self._excluded_files

    @excluded_files.setter
    def excluded_files(self, value: set[str]):
        self._excluded_files = value.copy()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex):
        if src := self.sourceModel():
            index = src.index(source_row, 0, source_parent)
            if self._available_only:
                source = FileSource(src.data(index, FileCollection.source_role))
                if source is FileSource.unavailable:
                    return False
            if self._name_prefix:
                name = src.data(index)
                if not name.startswith(self._name_prefix):
                    return False
            if self._excluded_files or self._arch not in (Arch.auto, Arch.all):
                file: File = src.data(index, FileCollection.file_role)
                if file.id in self._excluded_files:
                    return False
                if (
                    file.arch not in (None, Arch.auto, Arch.all)
                    and self._arch not in (Arch.auto, Arch.all)
                    and not Arch.is_compatible(file.arch, self._arch)
                ):
                    return False
        return True

    def __getitem__(self, index: int):
        src = cast(FileCollection, self.sourceModel())
        idx = self.mapToSource(self.index(index, 0)).row()
        return src[idx]
