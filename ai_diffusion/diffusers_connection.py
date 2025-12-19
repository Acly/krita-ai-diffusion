"""
Diffusers Connection Manager

Manages the connection to the diffusers server for Qwen Image Layered generation.
This runs alongside the main ComfyUI connection.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Callable

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from .client import ClientMessage, ClientEvent
from .diffusers_client import DiffusersClient
from .diffusers_server import DiffusersServer, DiffusersServerState
from .settings import settings
from .properties import Property, ObservableProperties
from .localization import translate as _
from . import util, eventloop


class DiffusersConnectionState(Enum):
    disconnected = 0
    connecting = 1
    connected = 2
    error = 3
    server_not_installed = 4
    server_starting = 5
    loading_model = 6


class DiffusersConnection(QObject, ObservableProperties):
    """Manages connection to the diffusers server for layered generation."""

    state = Property(DiffusersConnectionState.disconnected)
    error = Property("")
    model_loading_message = Property("")
    model_loading_progress = Property(0.0)

    state_changed = pyqtSignal(DiffusersConnectionState)
    error_changed = pyqtSignal(str)
    model_loading_changed = pyqtSignal(str, float)  # message, progress
    message_received = pyqtSignal(ClientMessage)

    def __init__(self):
        super().__init__()

        self._client: DiffusersClient | None = None
        self._server: DiffusersServer | None = None
        self._task: asyncio.Task | None = None
        self._model_poll_timer: QTimer | None = None
        self._server_managed = False

    def __del__(self):
        if self._task is not None:
            self._task.cancel()

    @property
    def client(self) -> DiffusersClient:
        """Get the connected client. Raises if not connected."""
        assert self.state is DiffusersConnectionState.connected and self._client is not None
        return self._client

    @property
    def client_if_connected(self) -> DiffusersClient | None:
        """Get client if connected, otherwise None."""
        return self._client if self.state is DiffusersConnectionState.connected else None

    @property
    def is_available(self) -> bool:
        """Check if diffusers backend is available (enabled and server installed)."""
        if not settings.diffusers_enabled:
            return False
        if self._server is None:
            self._server = DiffusersServer()
        return self._server.is_installed

    def check_server(self) -> DiffusersServer:
        """Check server installation status and return server object."""
        if self._server is None:
            self._server = DiffusersServer()
        self._server.check_install()
        if not self._server.is_installed:
            self.state = DiffusersConnectionState.server_not_installed
        return self._server

    async def _connect(self, url: str):
        """Connect to diffusers server."""
        if self.state is DiffusersConnectionState.connecting:
            return
        if self.state is DiffusersConnectionState.connected:
            await self.disconnect()

        self.error = ""
        self.state = DiffusersConnectionState.connecting

        try:
            self._client = await DiffusersClient.connect(url)

            if self._task is None:
                self._task = eventloop._loop.create_task(self._handle_messages())

            # Check if model needs to be loaded
            model_status = await self._client.get_model_status()
            if model_status.is_loaded:
                self.state = DiffusersConnectionState.connected
                util.client_logger.info(f"Connected to diffusers server at {url}")
            else:
                # Start model loading and monitor progress
                self.state = DiffusersConnectionState.loading_model
                self.model_loading_message = model_status.message or _("Starting model load...")
                self.model_loading_progress = model_status.progress
                util.client_logger.info(f"Connected, waiting for model to load")

                # Request model load if not already loading
                if not model_status.is_loading:
                    await self._client.request_model_load()

                # Start QTimer to poll model loading status
                self._start_model_poll_timer()

        except Exception as e:
            self.error = util.log_error(e)
            self.state = DiffusersConnectionState.error

    def _start_model_poll_timer(self):
        """Start a QTimer to poll model loading status."""
        if self._model_poll_timer is not None:
            self._model_poll_timer.stop()

        self._model_poll_timer = QTimer(self)
        self._model_poll_timer.setInterval(1000)  # 1 second
        self._model_poll_timer.timeout.connect(self._poll_model_status)
        self._model_poll_timer.start()
        util.client_logger.info("Started model loading poll timer")

    def _stop_model_poll_timer(self):
        """Stop the model polling timer."""
        if self._model_poll_timer is not None:
            self._model_poll_timer.stop()
            self._model_poll_timer = None

    def _poll_model_status(self):
        """Poll model status (called by QTimer)."""
        if self._client is None:
            self._stop_model_poll_timer()
            return

        async def do_poll():
            try:
                model_status = await self._client.get_model_status()

                self.model_loading_message = model_status.message
                self.model_loading_progress = model_status.progress
                self.model_loading_changed.emit(model_status.message, model_status.progress)

                util.client_logger.info(
                    f"Model poll: status={model_status.status}, progress={model_status.progress:.2f}"
                )

                if model_status.is_loaded:
                    self._stop_model_poll_timer()
                    self.state = DiffusersConnectionState.connected
                    util.client_logger.info("Model loaded, connection ready")
                elif model_status.is_error:
                    self._stop_model_poll_timer()
                    self.error = model_status.error or _("Model loading failed")
                    self.state = DiffusersConnectionState.error
                    util.client_logger.error(f"Model loading error: {self.error}")

            except Exception as e:
                util.client_logger.warning(f"Failed to get model status: {e}")

        eventloop.run(do_poll())

    def connect(self):
        """Connect to diffusers server (async wrapper)."""
        url = f"http://{settings.diffusers_server_url}"
        eventloop.run(self._connect(url))

    async def _start_server_and_connect(self, progress_callback: Callable | None = None):
        """Start managed server and connect to it."""
        self._server = self.check_server()

        if not self._server.is_installed:
            self.error = _("Diffusers server is not installed")
            self.state = DiffusersConnectionState.server_not_installed
            return

        self.state = DiffusersConnectionState.server_starting

        try:
            url = await self._server.start()
            self._server_managed = True
            await self._connect(url)
        except Exception as e:
            self.error = util.log_error(e)
            self.state = DiffusersConnectionState.error

    def start_server_and_connect(self, progress_callback: Callable | None = None):
        """Start managed server and connect (async wrapper)."""
        eventloop.run(self._start_server_and_connect(progress_callback))

    async def disconnect(self):
        """Disconnect from server."""
        util.client_logger.info("Disconnecting from diffusers server...")

        # Stop model polling timer
        self._stop_model_poll_timer()

        # Clear client queue and signal disconnect
        if self._client is not None:
            self._client.clear_queue()

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._server_managed and self._server and self._server.is_running:
            await self._server.stop()
            self._server_managed = False

        self._client = None
        self.error = ""
        self.model_loading_message = ""
        self.model_loading_progress = 0.0
        self.state = DiffusersConnectionState.disconnected
        util.client_logger.info("Disconnected from diffusers server")

    def interrupt(self):
        """Interrupt current job."""
        if self._client:
            eventloop.run(self._client.interrupt())

    async def _handle_messages(self):
        """Handle messages from the client."""
        client = self._client
        assert client is not None

        try:
            async for msg in client.listen():
                try:
                    self._handle_message(msg)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    util.client_logger.exception(e)
                    self.error = _("Error handling diffusers message: ") + str(e)
        except asyncio.CancelledError:
            pass

    def _handle_message(self, msg: ClientMessage):
        """Process a single message."""
        if msg.event is ClientEvent.error and msg.job_id == "":
            self.error = _("Error communicating with diffusers server: ") + str(msg.error)
        else:
            self.message_received.emit(msg)


# Global instance
_diffusers_connection: DiffusersConnection | None = None


def get_diffusers_connection() -> DiffusersConnection:
    """Get the global diffusers connection instance."""
    global _diffusers_connection
    if _diffusers_connection is None:
        _diffusers_connection = DiffusersConnection()
    return _diffusers_connection
