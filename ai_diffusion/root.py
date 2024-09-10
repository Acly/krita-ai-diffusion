from __future__ import annotations
from typing import Callable, NamedTuple
from PyQt5.QtCore import QObject, pyqtSignal

from .connection import Connection, ConnectionState
from .client import ClientMessage
from .server import Server, ServerState
from .document import Document, KritaDocument
from .model import Model
from .files import FileLibrary, File, FileSource
from .persistence import ModelSync, RecentlyUsedSync, import_prompt_from_file
from .ui.theme import checkpoint_icon
from .settings import ServerMode, settings
from .util import client_logger as log


class Root(QObject):
    """Root object, exists once, maintains all other instances. Keeps track of documents
    openend in Krita and creates a corresponding Model for each."""

    class PerDocument(NamedTuple):
        model: Model
        sync: ModelSync

    _server: Server
    _connection: Connection
    _models: list[PerDocument]
    _recent: RecentlyUsedSync

    model_created = pyqtSignal(Model)

    def __init__(self):
        super().__init__()

    def init(self):
        self._server = Server(settings.server_path)
        self._connection = Connection()
        self._files = FileLibrary.load()
        self._models = []
        self._recent = RecentlyUsedSync.from_settings()
        self._connection.message_received.connect(self._handle_message)
        self._connection.models_changed.connect(self._update_files)

    def prune_models(self):
        # Remove models for documents that have been closed
        self._models = [m for m in self._models if m.model.document.is_valid]

    def create_model(self, doc: KritaDocument):
        model = Model(doc, self._connection)
        self._recent.track(model)
        persistence_sync = ModelSync(model)
        import_prompt_from_file(model)
        self._models.append(Root.PerDocument(model, persistence_sync))
        self.model_created.emit(model)
        self.prune_models()
        return model

    def model_for_active_document(self) -> Model | None:
        doc = KritaDocument.active()
        if doc is None or not doc.is_valid:
            return None
        model = next((m.model for m in self._models if m.model.document == doc), None)
        if model is None:
            model = self.create_model(doc)
        else:
            model.document = doc
        return model

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
    def active_model(self):
        if model := self.model_for_active_document():
            return model
        return Model(Document(), self._connection)

    async def autostart(self, signal_server_change: Callable):
        connection = self._connection
        try:
            if (
                settings.server_mode is ServerMode.managed
                and self._server.state is ServerState.stopped
                and not self._server.upgrade_required
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
                await connection._connect(settings.server_url, ServerMode.external)
                if settings.server_mode is ServerMode.undefined:
                    if connection.state is ConnectionState.connected:
                        settings.server_mode = ServerMode.external
                    else:
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

            loras = [File.remote(lora) for lora in client.models.loras]
            self._files.loras.update(loras, FileSource.remote)


root = Root()
