from __future__ import annotations
from typing import Callable
from PyQt5.QtCore import QObject, pyqtSignal

from .connection import Connection, ConnectionState
from .client import ClientMessage
from .server import Server, ServerState
from .document import Document, KritaDocument
from .model import Model
from .settings import ServerMode, settings
from .util import client_logger as log


class Root(QObject):
    """Root object, exists once, maintains all other instances. Keeps track of documents
    openend in Krita and creates a corresponding Model for each."""

    _server: Server
    _connection: Connection
    _models: list[Model]

    model_created = pyqtSignal(Model)

    def __init__(self):
        super().__init__()

    def init(self):
        self._server = Server(settings.server_path)
        self._connection = Connection()
        self._models = []
        self._connection.message_received.connect(self._handle_message)

    def model_for_active_document(self):
        # Remove models for documents that have been closed
        self._models = [m for m in self._models if m.is_valid]

        # Find or create model for active document
        if doc := KritaDocument.active():
            model = next((m for m in self._models if m.is_active), None)
            if model is None:
                model = Model(doc, self._connection)
                self._models.append(model)
                self.model_created.emit(model)
            return model

        return None

    @property
    def connection(self):
        return self._connection

    @property
    def server(self):
        return self._server

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
                await connection._connect(url)
                signal_server_change()
            elif settings.server_mode in [ServerMode.undefined, ServerMode.external]:
                await connection._connect(settings.server_url)
                if settings.server_mode is ServerMode.undefined:
                    if connection.state is ConnectionState.connected:
                        settings.server_mode = ServerMode.external
                    else:
                        settings.server_mode = ServerMode.managed
        except Exception as e:
            log.warning(f"Failed to launch/connect server at startup: {e}")

    def _find_model(self, job_id: str) -> Model | None:
        return next((m for m in self._models if m.jobs.find(job_id)), None)

    def _handle_message(self, msg: ClientMessage):
        model = self._find_model(msg.job_id)
        if model is not None:
            model.handle_message(msg)


root = Root()
