from __future__ import annotations
from enum import Enum
from typing import Optional
from PyQt5.QtCore import QObject, pyqtSignal
import asyncio

from .client import Client, ClientMessage, ClientEvent, DeviceInfo, MissingResource
from .network import NetworkError
from .settings import Settings, PerformancePreset, settings
from .properties import Property, PropertyMeta
from . import util, eventloop


class ConnectionState(Enum):
    disconnected = 0
    connecting = 1
    connected = 2
    error = 3


class Connection(QObject, metaclass=PropertyMeta):
    state = Property(ConnectionState.disconnected)
    error = Property("")
    missing_resource: MissingResource | None = None

    state_changed = pyqtSignal(ConnectionState)
    error_changed = pyqtSignal(str)
    models_changed = pyqtSignal()
    message_received = pyqtSignal(ClientMessage)

    _client: Client | None = None
    _task: asyncio.Task | None = None

    def __init__(self):
        super().__init__()

    def __del__(self):
        if self._task is not None:
            self._task.cancel()

    async def _connect(self, url: str):
        if self.state is ConnectionState.connected:
            await self.disconnect()
        self.error = None
        self.missing_resource = None
        self.state = ConnectionState.connecting
        try:
            self._client = await Client.connect(url)
            apply_performance_preset(settings, self._client.device_info)
            if self._task is None:
                self._task = eventloop._loop.create_task(self._handle_messages())
            self.state = ConnectionState.connected
            self.models_changed.emit()
        except MissingResource as e:
            self.error = (
                f"Connection established, but the server is missing one or more {e.kind.value}s."
            )
            self.missing_resource = e
            self.state = ConnectionState.error
        except NetworkError as e:
            self.error = e.message
            self.state = ConnectionState.error
        except Exception as e:
            self.error = util.log_error(e)
            self.state = ConnectionState.error

    def connect(self, url: str = settings.server_url):
        eventloop.run(self._connect(url))

    async def disconnect(self):
        if self._task is not None:
            self._task.cancel()
            await self._task
            self._task = None

        self._client = None
        self.error = None
        self.missing_resource = None
        self.state = ConnectionState.disconnected

    def interrupt(self):
        eventloop.run(self.client.interrupt())

    def clear_queue(self):
        eventloop.run(self.client.clear_queue())

    def refresh(self):
        async def _refresh():
            await self.client.refresh()
            self.models_changed.emit()

        if self.state is ConnectionState.connected:
            eventloop.run(_refresh())

    @property
    def client(self):
        assert self.state is ConnectionState.connected and self._client is not None
        return self._client

    @property
    def client_if_connected(self):
        return self._client

    async def _handle_messages(self):
        client = self._client
        temporary_disconnect = False
        assert client is not None

        try:
            async for msg in client.listen():
                try:
                    if msg.event is ClientEvent.error and not msg.job_id:
                        self.error = f"Error communicating with server: {msg.error}"
                    elif msg.event is ClientEvent.disconnected:
                        temporary_disconnect = True
                        self.error = "Disconnected from server, trying to reconnect..."
                    elif msg.event is ClientEvent.connected:
                        if temporary_disconnect:
                            temporary_disconnect = False
                            self.error = ""
                    else:
                        self.message_received.emit(msg)
                        # model = self._find_model(msg.job_id)
                        # if model is not None:
                        #     model.handle_message(msg)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    util.client_logger.exception(e)
                    self.error = f"Error handling server message: {str(e)}"
        except asyncio.CancelledError:
            pass  # shutdown

    # def _report_error(self, message: str):
    #     self.error = message
    #     self.error_reported.emit(message)

    # def _clear_error(self):
    #     self.error = None
    #     self.error_reported.emit("")


def apply_performance_preset(settings: Settings, device: DeviceInfo):
    if settings.performance_preset is PerformancePreset.auto:
        if device.type == "cpu":
            settings.apply_performance_preset(PerformancePreset.cpu)
        elif device.vram <= 6:
            settings.apply_performance_preset(PerformancePreset.low)
        elif device.vram <= 12:
            settings.apply_performance_preset(PerformancePreset.medium)
        else:
            settings.apply_performance_preset(PerformancePreset.high)
