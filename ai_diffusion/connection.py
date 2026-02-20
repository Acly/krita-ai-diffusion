from __future__ import annotations

import asyncio
from collections.abc import Iterable
from enum import Enum

from PyQt5.QtCore import QObject, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices

from . import eventloop, util
from .client import Client, ClientEvent, ClientMessage, DeviceInfo, MissingResources, SharedWorkflow
from .cloud_client import CloudClient
from .comfy_client import ComfyClient
from .localization import translate as _
from .network import NetworkError
from .properties import ObservableProperties, Property
from .settings import PerformancePreset, ServerMode, Settings, settings


class ConnectionState(Enum):
    disconnected = 0
    connecting = 1
    connected = 2
    error = 3

    discover_models = 10

    auth_missing = 20
    auth_requesting = 21
    auth_pending = 22
    auth_error = 23


class Connection(QObject, ObservableProperties):
    state = Property(ConnectionState.disconnected)
    error = Property("")
    progress = Property((1, 1))

    state_changed = pyqtSignal(ConnectionState)
    error_changed = pyqtSignal(str)
    progress_changed = pyqtSignal(tuple)
    models_changed = pyqtSignal()
    message_received = pyqtSignal(ClientMessage)
    workflow_published = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self._client: Client | None = None
        self._task: asyncio.Task | None = None
        self._workflows: dict[str, dict] = {}
        self._temporary_disconnect = False
        self.error_kind = ""
        self.missing_resources: MissingResources | None = None

        settings.changed.connect(self._handle_settings_changed)
        self._update_state()

    def __del__(self):
        if self._task is not None:
            self._task.cancel()

    async def _sign_in(self, url: str):
        self._client = CloudClient(url)
        self.state = ConnectionState.auth_requesting
        try:
            sign_in = self._client.sign_in()
            url = await anext(sign_in)
            self.state = ConnectionState.auth_pending
            QDesktopServices.openUrl(QUrl(url))

            settings.access_token = await anext(sign_in)
            self.state = ConnectionState.disconnected
            await self._connect(self._client.url, ServerMode.cloud, settings.access_token)
            settings.save()

        except Exception as e:
            self.error = util.log_error(e)
            self.state = ConnectionState.auth_error

    def sign_in(self):
        eventloop.run(self._sign_in(CloudClient.default_api_url))

    async def _connect(self, url: str, mode: ServerMode, access_token=""):
        if self.state is ConnectionState.connecting:
            return
        if self.state is ConnectionState.connected:
            await self.disconnect()
        self.error = ""
        self.error_kind = ""
        self.state = ConnectionState.connecting
        try:
            if mode is ServerMode.cloud:
                if access_token == "":
                    self.state = ConnectionState.auth_missing
                    return
                self._client = await CloudClient.connect(CloudClient.default_api_url, access_token)
            else:
                self._client = await ComfyClient.connect(url)
                self.state = ConnectionState.discover_models
                async for status in self._client.discover_models(refresh=False):
                    self.progress = (status.current, status.total)
                self.missing_resources = self._client.missing_resources

            apply_performance_preset(settings, self._client.device_info)
            if self._task is None:
                self._task = eventloop._loop.create_task(self._handle_messages())
            self.state = ConnectionState.connected
            self.models_changed.emit()
        except NetworkError as e:
            self.error = e.message
            self.error_kind = "network"
            self.state = ConnectionState.error
            if e.status == 401:  # Unauthorized
                settings.access_token = ""
                self._update_state()
        except MissingResources as e:
            self.error = _(
                "Connection established, but the server is missing required custom nodes or models."
            )
            self.error_kind = "missing_resources"
            self.missing_resources = e
            self.state = ConnectionState.error
        except Exception as e:
            self.error = util.log_error(e)
            self.error_kind = "unknown"
            self.state = ConnectionState.error

    def connect(self):
        eventloop.run(
            self._connect(settings.server_url, settings.server_mode, settings.access_token)
        )

    async def disconnect(self):  # type: ignore (hides QObject.disconnect)
        if self._task is not None:
            self._task.cancel()
            await self._task
            self._task = None

        self._client = None
        self.error = ""
        self.missing_resources = None
        self.state = ConnectionState.disconnected
        self._update_state()

    def interrupt(self):
        eventloop.run(self.client.interrupt())

    def cancel(self, job_ids: Iterable[str]):
        eventloop.run(self.client.cancel(job_ids))

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

    @property
    def user(self):
        if client := self.client_if_connected:
            return client.user

    @property
    def workflows(self):
        return self._workflows

    async def _handle_messages(self):
        client = self._client
        self._temporary_disconnect = False
        assert client is not None

        try:
            async with client:
                async for msg in client.listen():
                    try:
                        self._handle_message(msg)
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        util.client_logger.exception(e)
                        self.error = _("Error handling server message: ") + str(e)
        except asyncio.CancelledError:
            pass  # shutdown

    def _handle_message(self, msg: ClientMessage):
        match msg:
            case (ClientEvent.error, "", *_):
                self.error = _("Error communicating with server: ") + str(msg.error)
            case (ClientEvent.disconnected, *_):
                self._temporary_disconnect = True
                self.error = _("Disconnected from server, trying to reconnect...")
            case (ClientEvent.connected, *_):
                if self._temporary_disconnect:
                    self._temporary_disconnect = False
                    self.error = ""
            case (ClientEvent.published, *_):
                assert isinstance(msg.result, SharedWorkflow)
                self._workflows[msg.result.publisher] = msg.result.workflow
                self.workflow_published.emit(msg.result.publisher)
            case _:
                self.message_received.emit(msg)

    def _update_state(self):
        if (
            self.state in [ConnectionState.disconnected, ConnectionState.error]
            and settings.server_mode is ServerMode.cloud
            and settings.access_token == ""
        ):
            self.state = ConnectionState.auth_missing
        elif (
            self.state in [ConnectionState.auth_missing, ConnectionState.auth_error]
            and settings.server_mode is not ServerMode.cloud
        ):
            self.state = ConnectionState.disconnected

    def _handle_settings_changed(self, key: str, value: object):
        if key == "server_mode":
            client_is_cloud = isinstance(self._client, CloudClient)
            mode_is_cloud = settings.server_mode is ServerMode.cloud
            if client_is_cloud != mode_is_cloud:
                self.error = ""
                eventloop.run(self.disconnect())
            self._update_state()

        elif key == "access_token":
            self._update_state()


def apply_performance_preset(settings: Settings, device: DeviceInfo):
    if settings.performance_preset is PerformancePreset.auto:
        if device.type == "cpu":
            settings.apply_performance_preset(PerformancePreset.cpu)
        elif device.type.lower() == "cloud":
            settings.apply_performance_preset(PerformancePreset.cloud)
        elif device.vram <= 6:
            settings.apply_performance_preset(PerformancePreset.low)
        elif device.vram <= 12:
            settings.apply_performance_preset(PerformancePreset.medium)
        else:
            settings.apply_performance_preset(PerformancePreset.high)
