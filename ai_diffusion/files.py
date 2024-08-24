import json
import hashlib
from base64 import b64encode
from enum import Flag
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence
from PyQt5.QtCore import QAbstractListModel, QModelIndex, Qt
from PyQt5.QtGui import QIcon

from .util import encode_json, read_json_with_comments, user_data_dir, client_logger as log


class FileSource(Flag):
    local = 1
    remote = 2


@dataclass
class File:
    id: str
    name: str
    source: FileSource
    icon: QIcon | None = None
    hash: str | None = None
    path: Path | None = None
    size: int | None = None

    @staticmethod
    def remote(id: str):
        id = id.replace("\\", "/")
        dot = id.rfind(".")
        name = id if dot == -1 else id[:dot]
        return File(id, name, FileSource.remote)

    @staticmethod
    def local(path: Path, compute_hash=False):
        file = File(path.name, path.stem, FileSource.local, path=path)
        file.size = path.stat().st_size
        if compute_hash:
            file.compute_hash()
        return file

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


class FileCollection(QAbstractListModel):

    def __init__(self, database: Path | None = None, parent=None):
        super().__init__(parent)

        self._database = database
        self._files: list[File] = []

    def rowCount(self, parent=QModelIndex()):
        return len(self._files)

    def data(self, index: QModelIndex, role: int = 0):
        if 0 <= index.row() < len(self._files):
            item = self._files[index.row()]

            match role:
                case Qt.ItemDataRole.DisplayRole | Qt.ItemDataRole.EditRole:
                    return item.name
                case Qt.ItemDataRole.UserRole:
                    return item.id
                case Qt.ItemDataRole.DecorationRole:
                    return item.icon
                case _:
                    return None

    def extend(self, files: list[File]):
        self.beginInsertRows(QModelIndex(), len(self._files), len(self._files) + len(files) - 1)
        self._files.extend(files)
        self.endInsertRows()

        if any(FileSource.local in f.source for f in files):
            self.save()

    def remove(self, index: int):
        self.beginRemoveRows(QModelIndex(), index, index)
        del self._files[index]
        self.endRemoveRows()

    def update(self, new_files: Sequence[File], source: FileSource):
        existing_ids = {f.id for f in self._files if f.source is source}
        new_ids = {f.id for f in new_files}
        for id in existing_ids:
            if id not in new_ids:
                self.remove(self.find_index(id))
        self.extend([f for f in new_files if f.id not in existing_ids])

    def add(self, file: File):
        self.extend([file])
        return file

    def find(self, id: str):
        return next((f for f in self if f.id == id), None)

    def find_local(self, id: str):
        return next((f for f in self if f.id == id and FileSource.local in f.source), None)

    def find_index(self, id: str):
        return next((i for i, f in enumerate(self) if f.id == id), -1)

    def load(self):
        if not self._database or not self._database.exists():
            return
        try:
            data = read_json_with_comments(self._database)
            self.extend([File(**f) for f in data])
            log.info(f"Loaded {len(self)} model files from {self._database}")
        except Exception as e:
            log.error(f"Failed to read {self._database}: {e}")

    def save(self):
        if self._database:
            self._database.parent.mkdir(parents=True, exist_ok=True)
            self._database.write_text(json.dumps(self._files, indent=4, default=encode_json))

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return super().flags(index) | Qt.ItemFlag.ItemIsDropEnabled
        return super().flags(index) | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        return self._files[index]


_instance = None


class FileLibrary(NamedTuple):
    checkpoints: FileCollection
    loras: FileCollection

    @staticmethod
    def load(database_dir: Path | None = None):
        global _instance
        database_dir = database_dir or user_data_dir / "data"
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
