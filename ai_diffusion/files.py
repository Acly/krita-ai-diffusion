import json
import hashlib
from base64 import b64encode
from enum import Enum, Flag
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, NamedTuple, Sequence, cast
from PyQt5.QtCore import QAbstractListModel, QSortFilterProxyModel, QModelIndex, Qt
from PyQt5.QtGui import QIcon

from .util import encode_json, read_json_with_comments, user_data_dir, client_logger as log


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
        data["path"] = Path(data["path"]) if data.get("path") else None
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
            while chunk := f.read(4096):
                sha.update(chunk)
        self.hash = b64encode(sha.digest()).decode()
        return self.hash

    def meta(self, key: str, default=None) -> Any:
        return self.metadata.get(key, default) if self.metadata else default


class FileCollection(QAbstractListModel):
    source_role = Qt.ItemDataRole.UserRole + 1

    def __init__(self, database: Path | None = None, parent=None):
        super().__init__(parent)

        self._database = database
        self._files: list[File] = []
        self.load()

    def rowCount(self, parent=QModelIndex()):
        return len(self._files)

    def data(self, index: QModelIndex, role: int = 0):
        if 0 <= index.row() < len(self._files):
            item = self._files[index.row()]

            match role:
                case Qt.ItemDataRole.DisplayRole | Qt.ItemDataRole.EditRole:
                    return item.name
                case Qt.ItemDataRole.DecorationRole:
                    return item.icon
                case Qt.ItemDataRole.UserRole:
                    return item.id
                case FileCollection.source_role:
                    return item.source.value
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
        self.save()

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
            if f.source is FileSource.local and f.path and not f.path.exists():
                log.warning(f"Local file {f.path} not found, removing from collection")
                self.remove(i)
            else:
                i += 1

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index: int):
        return self._files[index]


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
        return True

    def __getitem__(self, index: int):
        src = cast(FileCollection, self.sourceModel())
        idx = self.mapToSource(self.index(index, 0)).row()
        return src[idx]
