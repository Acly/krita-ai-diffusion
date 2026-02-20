from __future__ import annotations

import asyncio
import itertools
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal

from . import platform_tools, util
from .client import ClientMessage
from .connection import Connection, ConnectionState
from .custom_workflow import WorkflowCollection
from .document import Document, KritaDocument
from .files import File, FileFormat, FileLibrary, FileSource
from .model import Model
from .persistence import ModelSync, RecentlyUsedSync, import_prompt_from_file
from .server import Server, ServerState
from .settings import ServerMode, settings
from .ui.theme import checkpoint_icon
from .updates import AutoUpdate
from .util import client_logger as log


class Root(QObject):
    """Root object, exists once, maintains all other instances. Keeps track of documents
    openend in Krita and creates a corresponding Model for each."""

    @dataclass
    class PerDocument:
        model: Model
        sync: ModelSync | None = None

    model_created = pyqtSignal(Model)

    def __init__(self):
        super().__init__()

    def init(self):
        self._server = Server(settings.server_path)
        self._connection = Connection()
        self._files = FileLibrary.load()
        self._workflows = WorkflowCollection(self._connection)
        self._models: list[Root.PerDocument] = []
        self._null_model = Model(Document(), self._connection, self._workflows)
        self._recent = RecentlyUsedSync.from_settings()
        self._auto_update = AutoUpdate()
        if settings.auto_update:
            self._auto_update.check()
        self._connection.message_received.connect(self._handle_message)
        self._connection.models_changed.connect(self._update_files)

    def prune_models(self):
        # Remove models for documents that have been closed
        self._models = [m for m in self._models if m.model.document.is_valid]

    def create_model(self, doc: KritaDocument):
        model = Model(doc, self._connection, self._workflows)
        model_entry = Root.PerDocument(model)
        self._models.append(model_entry)
        self._recent.track(model)
        model_entry.sync = ModelSync(model)
        import_prompt_from_file(model)
        self.model_created.emit(model)
        return model

    def model_for_active_document(self) -> Model | None:
        self.prune_models()
        if doc := KritaDocument.active():
            model = next((m.model for m in self._models if m.model.document == doc), None)
            if model is None:
                model = self.create_model(doc)
            else:
                model.document = doc
            return model
        return None

    @property
    def models(self) -> list[Model]:
        return [m.model for m in self._models]

    @property
    def connection(self) -> Connection:
        return self._connection

    @property
    def server(self):
        return self._server

    @property
    def files(self) -> FileLibrary:
        return self._files

    @property
    def workflows(self) -> WorkflowCollection:
        return self._workflows

    @property
    def auto_update(self) -> AutoUpdate:
        return self._auto_update

    @property
    def active_model(self):
        if model := self.model_for_active_document():
            return model
        return self._null_model

    async def autostart(self, signal_server_change: Callable):
        connection = self._connection
        try:
            if (
                settings.server_mode is ServerMode.managed
                and self._server.state is ServerState.stopped
            ):
                url = await self._server.start()
                signal_server_change()
                await connection._connect(url, ServerMode.managed)
                signal_server_change()
            elif settings.server_mode is ServerMode.cloud:
                await connection._connect(
                    settings.server_url, ServerMode.cloud, settings.access_token
                )
            elif settings.server_mode in [ServerMode.undefined, ServerMode.external]:
                urls = [settings.server_url]
                if settings.server_mode is ServerMode.undefined:
                    urls.append("127.0.0.1:8000")  # ComfyUI Desktop default port
                    retries = 1
                else:
                    retries = 5
                for url, retry in itertools.product(urls, range(retries)):
                    await connection._connect(url, ServerMode.external)
                    if connection.state is ConnectionState.connected:
                        settings.server_url = url
                        break
                    elif (
                        connection.error_kind != "network"
                        or urls[0] != settings.server_url
                        or settings.server_mode not in [ServerMode.undefined, ServerMode.external]
                    ):
                        break
                    await asyncio.sleep(5 * (retry + 1))
                if settings.server_mode is ServerMode.undefined:
                    if connection.state is ConnectionState.connected:
                        settings.server_mode = ServerMode.external
                        settings.save()
                    else:
                        connection.state = ConnectionState.disconnected
                        connection.error = ""
                        settings.server_mode = ServerMode.cloud
        except Exception as e:
            log.warning(f"Failed to launch/connect server at startup: {e}")

    def get_active_model_used_storage(self):  # in bytes
        doc = KritaDocument.active()
        if doc is None or not doc.is_valid:
            return 0
        if persist := next((m.sync for m in self._models if m.model.document == doc), None):
            return persist.memory_used
        return 0

    def _find_model(self, job_id: str) -> Model | None:
        return next((m.model for m in self._models if m.model.jobs.find(job_id)), None)

    def _handle_message(self, msg: ClientMessage):
        model = self._find_model(msg.job_id)
        if model is not None:
            model.handle_message(msg)

    def _update_files(self):
        if client := self._connection.client_if_connected:
            checkpoints = [
                File(
                    cp.filename,
                    cp.name,
                    FileSource.remote,
                    cp.format,
                    icon=checkpoint_icon(cp.arch, cp.format, client),
                )
                for cp in client.models.checkpoints.values()
            ]
            self._files.checkpoints.update(checkpoints, FileSource.remote)

            loras = [File.remote(lora, FileFormat.lora) for lora in client.models.loras]
            self._files.loras.update(loras, FileSource.remote)


root = Root()


def _read_log(log_path: Path, last_n: int = 1000):
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-last_n:]
            return "".join(reversed(lines))
    except Exception as e:
        return f"Failed to read log file {log_path}: {e}"


def collect_diagnostics(redact_user=True):
    import platform

    from . import __version__

    try:
        from krita import Krita

        krita_version = Krita.instance().version()
    except Exception:
        krita_version = "Unknown"

    out = "Krita AI Diffusion Plugin Diagnostics\n"
    out += "-------------------------------------\n"
    out += f"Plugin Version: {__version__}\n"
    out += f"Krita Version: {krita_version}\n"
    out += f"Python Version: {sys.version}\n"
    out += "-------------------------------------\n"
    out += "System Information:\n"
    out += f"  Platform: {platform.system()} {platform.release()}\n"
    out += f"  Architecture: {platform.machine()}\n"
    out += f"  Processor: {platform.processor()}\n"
    out += f"  CUDA Capability: {', '.join(f'{a}.{b}' for a, b in platform_tools.get_cuda_devices())}\n"
    out += "-------------------------------------\n"
    out += "Path Configuration:\n"
    out += f"  Plugin Directory: {util.plugin_dir}\n"
    out += f"  User Data Directory: {util.user_data_dir}\n"
    out += "-------------------------------------\n"
    out += "Settings:\n"
    for name, value in settings:
        if name in ("access_token", "server_authorization") or value == settings.access_token:
            value = "<redacted>"
        out += f"  {name}: {value}\n"
    out += "-------------------------------------\n"
    out += "Client Log:\n"
    out += _read_log(util.log_dir / "client.log", last_n=300)
    if settings.server_mode is ServerMode.managed:
        out += "\n-------------------------------------\n"
        out += "Server Log:\n"
        out += _read_log(util.log_dir / "server.log", last_n=300)

    if redact_user:
        user_name = None
        if platform_tools.is_windows:
            pattern = r"[A-Z]:\\Users\\(.*)\\AppData\\"
            if match := re.search(pattern, str(util.user_data_dir.absolute())):
                user_name = match.group(1)
        else:
            pattern = r"/home/(.*)/\.local/share"
            if match := re.search(pattern, str(util.user_data_dir.absolute())):
                user_name = match.group(1)

        if user_name:
            out = out.replace(user_name, "<redacted>")

    return out[: 65536 - 5000]  # GitHub issue body limit is 65536 characters
