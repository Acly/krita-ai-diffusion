import asyncio
from pathlib import Path

import pytest
from PyQt6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QColor, QIcon, QImage

from ai_diffusion.backend.network import NetworkError, RequestManager
from ai_diffusion.backend.resources import Arch
from ai_diffusion.files import (
    File,
    FileCollection,
    FileFilter,
    FileFormat,
    FileSource,
    LoraInfoWorker,
)
from ai_diffusion.settings import settings

from .conftest import qtapp


class EventHandler:
    def __init__(self, files: FileCollection):
        self.begin_insert = []
        self.end_insert = []
        self.begin_remove = []
        self.end_remove = []
        self.data_changed = []
        self.connect(files)

    def on_begin_insert(self, parent, start, end):
        self.begin_insert.append((parent, start, end))

    def on_end_insert(self):
        self.end_insert.append(())

    def on_begin_remove(self, parent, start, end):
        self.begin_remove.append((parent, start, end))

    def on_end_remove(self):
        self.end_remove.append(())

    def on_data_changed(self, start, end):
        self.data_changed.append((start, end))

    def connect(self, collection: FileCollection):
        collection.rowsAboutToBeInserted.connect(self.on_begin_insert)
        collection.rowsInserted.connect(self.on_end_insert)
        collection.rowsAboutToBeRemoved.connect(self.on_begin_remove)
        collection.rowsRemoved.connect(self.on_end_remove)
        collection.dataChanged.connect(self.on_data_changed)


def write_test_file(folder: Path):
    test_file_path = folder / "data.bin"
    test_file_path.write_bytes(b"test data")
    return test_file_path


def test_data_roles():
    files = FileCollection()
    file1 = files.add(File.remote("path\\to\\file1.txt"))
    file1.source = file1.source | FileSource.local
    file2 = files.add(File.remote("/path/to/file2.txt"))
    file2.icon = QIcon()

    assert files.data(files.index(0), Qt.ItemDataRole.DisplayRole) == "path/to/file1"
    assert files.data(files.index(1), Qt.ItemDataRole.DisplayRole) == "/path/to/file2"
    assert files.data(files.index(0), Qt.ItemDataRole.EditRole) == "path/to/file1"
    assert files.data(files.index(1), Qt.ItemDataRole.EditRole) == "/path/to/file2"
    assert files.data(files.index(0), Qt.ItemDataRole.DecorationRole) is None
    assert files.data(files.index(1), Qt.ItemDataRole.DecorationRole) == file2.icon
    assert files.data(files.index(0), Qt.ItemDataRole.UserRole) == file1.id
    assert files.data(files.index(1), Qt.ItemDataRole.UserRole) == file2.id
    assert files.data(files.index(0), FileCollection.source_role) == file1.source.value
    assert files.data(files.index(1), FileCollection.source_role) == file2.source.value


def test_extend(tmp_path: Path):
    files = FileCollection()
    events = EventHandler(files)
    test_file_path = write_test_file(tmp_path)

    file1 = files.add(File.remote(test_file_path.name))
    input = [File.remote("/path/to/file2.txt"), File.local(test_file_path)]
    files.extend(input)

    assert len(files) == 2
    assert file1.source == FileSource.local | FileSource.remote
    assert files[1].source == FileSource.remote
    assert files[1].id == "/path/to/file2.txt"

    assert events.begin_insert == [(QModelIndex(), 0, 0), (QModelIndex(), 1, 1)]
    assert len(events.end_insert) == 2


def test_remove():
    files = FileCollection()
    events = EventHandler(files)

    files.add(File.remote("/path/to/file1.txt"))
    file2 = files.add(File.remote("/path/to/file2.txt"))
    files.remove(0)

    assert len(files) == 1
    assert files[0].id == file2.id
    assert events.begin_remove == [(QModelIndex(), 0, 0)]
    assert len(events.end_remove) == 1


def test_update(tmp_path: Path):
    test_file_path = write_test_file(tmp_path)

    files = FileCollection()
    file0 = files.add(File.remote("/path/to/file0.txt"))
    file1 = files.add(File.local(test_file_path))
    file2 = files.add(File("file2.txt", "file2", FileSource.unavailable))
    file3 = files.add(File.remote("path/to/file3.txt"))

    events = EventHandler(files)
    input = [
        File.remote("/path/to/file0.txt"),
        File.remote(test_file_path.name),
        File.remote("path/to/file4.txt"),
    ]
    files.update(input, FileSource.remote)

    assert len(files) == 5
    assert files[0] is file0 and files[0].source == FileSource.remote
    assert files[1] is file1 and files[1].source == FileSource.remote | FileSource.local
    assert files[2] is file2 and files[2].source == FileSource.unavailable
    assert files[3] is file3 and files[3].source == FileSource.unavailable
    assert files[4].id == "path/to/file4.txt" and files[4].source == FileSource.remote

    assert events.begin_insert == [(QModelIndex(), 4, 4)]
    assert len(events.end_insert) == 1
    assert events.data_changed == [
        (files.index(3), files.index(3)),
        (files.index(1), files.index(1)),
    ]


def test_find(tmp_path: Path):
    test_file_path = write_test_file(tmp_path)

    files = FileCollection()
    file1 = files.add(File.remote("path/to/file1.txt"))
    file2 = files.add(File.local(test_file_path))

    assert files.find("path/to/file1.txt") is file1
    assert files.find(test_file_path.name) is file2
    assert files.find("path/to/file5.txt") is None

    assert files.find_local("path/to/file1.txt") is None
    assert files.find_local(test_file_path.name) is file2

    assert files.find_index("path/to/file1.txt") == 0
    assert files.find_index(test_file_path.name) == 1


def test_serialization(tmp_path: Path):
    test_file_path = write_test_file(tmp_path)

    files = FileCollection(tmp_path / "db.json")
    file1 = files.add(File.remote("path/to/file1.txt"))
    file2 = files.add(File.local(test_file_path, FileFormat.lora, compute_hash=True))
    files.set_meta(file2, "key", "value")
    file2.name = "Example model"
    file2.civitai = 123
    file2.version = "v1"
    file2.arch = Arch.sd15
    image = QImage(512, 256, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    files.set_image(file2, image)

    files2 = FileCollection(tmp_path / "db.json")
    assert len(files2) == 2
    assert files2[0].id == file1.id and files2[0].source == file1.source
    assert files2[1].id == file2.id and files2[1].source == file2.source
    assert files2[1].path == file2.path and files2[1].hash == file2.hash
    assert files2[1].meta("key") == "value"
    assert files2[1].format is FileFormat.lora
    assert files2[1].name == "Example model"
    assert files2.data(files2.index(1)) == test_file_path.stem
    assert files2[1].civitai == 123
    assert files2[1].version == "v1"
    assert files2[1].arch is Arch.sd15
    path = files2.image_path(files2[1])
    assert path is not None and path.suffix == ".jpg"
    preview = QImage(str(path))
    assert preview.width() == preview.height() == 256
    assert preview.pixelColor(128, 128).red() > 250


@qtapp
async def test_lora_hash_worker(tmp_path: Path):
    previous_setting = settings.civitai_collection
    settings.civitai_collection = False
    files = FileCollection(tmp_path / "loras.json")
    file = files.add(File.local(write_test_file(tmp_path), FileFormat.lora))
    known = files.add(File.remote("known.safetensors", FileFormat.lora))
    known.civitai = "not-found"

    async def remote_hash(filename: str):
        raise AssertionError("Known LoRAs must not be hashed again")

    worker = LoraInfoWorker(files, remote_hash)
    try:
        await worker.run()
        assert file.hash is not None
        assert known.hash is None
        assert FileCollection(tmp_path / "loras.json")[0].hash == file.hash
        await worker.run()
    finally:
        settings.changed.disconnect(worker._settings_changed)
        settings.civitai_collection = previous_setting


@qtapp
async def test_civitai_connection_errors(tmp_path: Path):
    class OfflineNetwork(RequestManager):
        calls = 0

        def get(self, url: str, timeout=None, bearer=None):
            self.calls += 1
            future = asyncio.get_running_loop().create_future()
            future.set_exception(NetworkError(1, "Offline", url))
            return future

    files = FileCollection(tmp_path / "loras.json")
    for i in range(5):
        file = files.add(File.remote(f"{i}.safetensors", FileFormat.lora))
        file.hash = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

    async def remote_hash(filename: str):
        raise AssertionError("Files already have hashes")

    worker = LoraInfoWorker(files, remote_hash)
    network = OfflineNetwork()
    worker._requests = network
    try:
        await worker.run()
        assert network.calls == 3
        assert worker.last_error == "Offline"
        assert FileCollection(tmp_path / "loras.json")[2].meta("civitai_error") == "Offline"
        await worker.run()
        assert network.calls == 3
    finally:
        settings.changed.disconnect(worker._settings_changed)


@qtapp
async def test_civitai_retrieval(tmp_path: Path, pytestconfig):
    if pytestconfig.getoption("ci"):
        pytest.skip("Live Civitai requests are disabled on CI")
    files = FileCollection(tmp_path / "loras.json")
    file = files.add(File.local(Path(__file__).parent / "data/LowRA.safetensors", FileFormat.lora))
    missing = files.add(File.remote("missing.safetensors", FileFormat.lora))
    missing.hash = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

    async def remote_hash(filename: str):
        raise AssertionError("Only local or already hashed files are used")

    worker = LoraInfoWorker(files, remote_hash)
    try:
        await worker.run()
        assert isinstance(file.civitai, int), worker.last_error
        assert file.name and file.version and file.arch is Arch.sd15
        assert missing.civitai == "not-found", worker.last_error
        assert file.hash
        # Running again must skip both entries, including the missing model.
        await worker.run()
    finally:
        settings.changed.disconnect(worker._settings_changed)


def test_sort_filter():
    files = FileCollection()
    files.add(File.remote("b.txt"))
    files.add(File.remote("a.txt"))
    files.add(File.remote("piong/c.txt"))

    filtered = FileFilter(files)
    assert filtered[0].id == "a.txt"
    assert filtered[1].id == "b.txt"
    assert filtered[2].id == "piong/c.txt"

    files.add(File.remote("piong/a.txt"))
    assert filtered[2].id == "piong/a.txt"
    assert filtered[3].id == "piong/c.txt"

    files[0].source = FileSource.unavailable
    filtered.available_only = True
    assert filtered[0].id == "a.txt"
    assert filtered[1].id == "piong/a.txt"

    filtered.name_prefix = "piong"
    assert filtered[0].id == "piong/a.txt"
    assert filtered[1].id == "piong/c.txt"

    files[2].arch = Arch.sdxl
    filtered.arch = Arch.sd15
    assert filtered.rowCount() == 1
    assert filtered[0].id == "piong/a.txt"
    filtered.arch = Arch.illu
    assert filtered.rowCount() == 2
    filtered.arch = Arch.all
    assert filtered.rowCount() == 2

    filtered.excluded_files = {"piong/c.txt"}
    assert filtered.rowCount() == 1
    assert filtered[0].id == "piong/a.txt"
    filtered.excluded_files = set()
    assert filtered.rowCount() == 2

    search = QSortFilterProxyModel()
    search.setSourceModel(filtered)
    search.setFilterRole(FileCollection.search_role)
    search.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    search.setFilterFixedString("C.TXT")
    assert search.rowCount() == 1
    index = search.index(0, 0)
    assert index.data(Qt.ItemDataRole.UserRole) == "piong/c.txt"
    assert index.data(FileCollection.file_role) is files[2]
    search.setFilterFixedString("missing")
    assert search.rowCount() == 0
    search.setFilterFixedString("")
    assert search.rowCount() == 2

    files[2].arch = Arch.sd15
    filtered.arch = Arch.sd15
    assert search.rowCount() == 2
    files[2].arch = Arch.sdxl
    files.notify(files[2])
    assert search.rowCount() == 1

    windows_path = File.remote(r"piong\windows.safetensors", FileFormat.lora)
    files.add(windows_path)
    search.setFilterFixedString("piong/windows")
    assert search.rowCount() == 1
    search.setFilterFixedString(r"piong\windows")
    assert search.rowCount() == 1
