from __future__ import annotations

import atexit
import hashlib
import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from PyQt5.QtCore import QLockFile

from . import util


class MediaSession:
    def __init__(self, root: Path, metadata_cache: Path | None = None):
        self.metadata_cache = metadata_cache
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.sweep()
        self.path = self.root / f"session-{uuid.uuid4().hex}"
        self.path.mkdir()
        self._lock = QLockFile(str(self.path / ".lock"))
        self._lock.setStaleLockTime(0)
        if not self._lock.tryLock():
            raise RuntimeError("Cannot lock temporary media session")
        if self.metadata_cache is not None and self.metadata_cache.exists():
            self._copy_metadata(self.metadata_cache, self.folder("model-sidecars"))

    def sweep(self):
        for path in self.root.iterdir():
            if not re.fullmatch(r"session-[0-9a-f]{32}", path.name):
                continue
            if _is_link(path) or not path.is_dir():
                continue
            lock = QLockFile(str(path / ".lock"))
            lock.setStaleLockTime(0)
            if lock.tryLock():
                lock.unlock()
                self._remove(path)

    def folder(self, name: str):
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            raise ValueError("Invalid temporary media category")
        path = self.path / name
        path.mkdir(exist_ok=True)
        return path

    def close(self):
        self._lock.unlock()
        self._remove(self.path)

    def _remove(self, path: Path):
        if path.parent != self.root or not re.fullmatch(r"session-[0-9a-f]{32}", path.name):
            raise ValueError("Refusing cleanup outside a media session")
        if not path.exists() or _is_link(path):
            return
        errors = []

        def on_error(_func, _path, info):
            errors.append(info[1])

        if self.metadata_cache is not None:
            self._copy_metadata(path / "model-sidecars", self.metadata_cache)
        for attempt in range(5):
            errors.clear()
            shutil.rmtree(path, onerror=on_error)
            if not path.exists():
                return
            if attempt < 4:
                time.sleep(0.1)
        util.client_logger.warning(f"Temporary media cleanup deferred until next startup: {errors}")

    def _copy_metadata(self, source: Path, target: Path):
        if not source.exists() or _is_link(source):
            return
        for path in source.rglob("*.json"):
            if not (path.name.endswith(".metadata.json") or path.name == ".lm-sidecar-roots.json"):
                continue
            if _is_link(path) or any(
                _is_link(p) for p in path.parents if p != source and p.is_relative_to(source)
            ):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                data.pop("preview_url", None)
                destination = target / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp = destination.with_suffix(".json.tmp")
                temp.write_text(json.dumps(data), encoding="utf-8")
                temp.replace(destination)
            except (OSError, ValueError) as e:
                util.client_logger.warning(f"Could not preserve model metadata {path}: {e}")


_session: MediaSession | None = None


def session() -> MediaSession:
    global _session
    if _session is None:
        _session = MediaSession(
            util.user_data_dir / "temporary-media", util.user_data_dir / "model-metadata"
        )
        atexit.register(shutdown)
    return _session


def shutdown():
    global _session
    _external_accepted.clear()
    _workflow_accepted.clear()
    if _session is not None:
        _session.close()
        _session = None


def temporary_directory(prefix: str):
    return tempfile.TemporaryDirectory(prefix=prefix, dir=session().folder("transfers"))


def _is_link(path: Path):
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)


def initialize():
    session()
    # Only known, plugin-owned staging areas; never scan user model/output folders.
    for path in [
        util.user_data_dir / "eagle_upload",
        util.user_data_dir / "animadex" / "thumbs",
        util.user_data_dir / "prompt_previews",
    ]:
        if path.is_dir() and not _is_link(path):
            try:
                shutil.rmtree(path)
            except OSError as e:
                util.client_logger.warning(f"Could not clean legacy media staging {path}: {e}")

    for path in util.log_dir.iterdir():
        if path.name == "workflow.json" or re.fullmatch(r"[0-9]+\.png", path.name):
            if path.is_file() and not _is_link(path):
                try:
                    path.unlink()
                except OSError as e:
                    util.client_logger.warning(f"Could not clean diagnostic media {path}: {e}")


def managed_server_environment(server_path: Path):
    media = session()
    manager = media.folder("lora-manager")
    # Import settings only, never old thumbnail/index databases with stale local paths.
    config = util.user_data_dir.parent / "ComfyUI-LoRA-Manager" / "settings.json"
    try:
        data = json.loads(config.read_text(encoding="utf-8")) if config.exists() else {}
    except (OSError, ValueError):
        data = {}
    for key in ["libraries", "active_library", "folder_paths", "extra_folder_paths"]:
        data.pop(key, None)
    data.update({
        "use_portable_settings": False,
        "sidecar_storage_mode": "centralized",
        "sidecar_storage_path": str(media.folder("model-sidecars")),
        "recipes_path": data.get("recipes_path")
        or str(server_path / "models" / "loras" / "recipes"),
        "example_images_path": str(media.folder("example-images")),
    })
    (manager / "settings.json").write_text(json.dumps(data), encoding="utf-8")
    temp = str(media.folder("server-temp"))
    return {"LORA_MANAGER_SETTINGS_DIR": str(manager), "TEMP": temp, "TMP": temp, "TMPDIR": temp}


def managed_server_arguments():
    media = session()
    return [
        "--temp-directory",
        str(media.folder("comfy-temp")),
        "--output-directory",
        str(media.folder("comfy-output")),
        "--input-directory",
        str(media.folder("comfy-input")),
        "--user-directory",
        str(media.folder("comfy-user")),
    ]


_external_accepted: set[str] = set()


def require_external_acceptance(url: str):
    if url in _external_accepted:
        return
    from PyQt5.QtWidgets import QApplication, QMessageBox

    from .localization import translate as _

    allowed = (
        isinstance(QApplication.instance(), QApplication)
        and QMessageBox.question(
            None,
            _("External server media storage"),
            _(
                "This server is not managed by the plugin. It may retain uploaded/generated media outside temporary storage. Allow sending media to this server for this session?"
            )
            + f"\n{url}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        == QMessageBox.Yes
    )
    if not allowed:
        raise util.PluginError(_("External media storage was not approved."))
    _external_accepted.add(url)


_workflow_accepted: set[str] = set()


def require_workflow_acceptance(graph: dict):
    from PyQt5.QtWidgets import QApplication, QMessageBox

    from .localization import translate as _

    key = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()
    if key in _workflow_accepted:
        return
    allowed = (
        isinstance(QApplication.instance(), QApplication)
        and QMessageBox.question(
            None,
            _("Custom workflow media access"),
            _(
                "Custom nodes can write files outside temporary storage. Allow this workflow to save media files for this session?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        == QMessageBox.Yes
    )
    if not allowed:
        raise util.PluginError(_("Custom workflow media storage was not approved."))
    _workflow_accepted.add(key)
