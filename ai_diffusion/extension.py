import sys
from typing import Callable, Optional
from PyQt5.QtWidgets import QAction
from krita import Extension, Krita, DockWidgetFactory, DockWidgetFactoryBase, Window  # type: ignore

from . import eventloop, settings, __version__, Server, ServerMode, ServerState
from .ui import actions, ImageDiffusionWidget, SettingsDialog, Workspace
from .ui.connection import Connection, ConnectionState
from .util import client_logger as log


class AIToolsExtension(Extension):
    _actions: dict[str, QAction] = {}
    _server: Server

    def __init__(self, parent):
        super().__init__(parent)
        log.info(f"Extension initialized, Version: {__version__}, Python: {sys.version}")

        eventloop.setup()
        settings.load()
        self._server = Server(settings.server_path)
        ImageDiffusionWidget._server = self._server

        notifier = Krita.instance().notifier()
        notifier.setActive(True)
        notifier.applicationClosing.connect(self.shutdown)

    def setup(self):
        eventloop.run(self.autostart())

    def shutdown(self):
        self._server.terminate()
        eventloop.stop()

    async def autostart(self):
        connection = Connection.instance()
        try:
            if (
                settings.server_mode is ServerMode.managed
                and self._server.state is ServerState.stopped
                and not self._server.upgrade_required
            ):
                url = await self._server.start()
                self._settings_dialog.connection.update()
                await connection._connect(url)
                self._settings_dialog.connection.update()
            elif settings.server_mode in [ServerMode.undefined, ServerMode.external]:
                await connection._connect(settings.server_url)
                if settings.server_mode is ServerMode.undefined:
                    if connection.state is ConnectionState.connected:
                        settings.server_mode = ServerMode.external
                    else:
                        settings.server_mode = ServerMode.managed
        except Exception as e:
            log.warning(f"Failed to launch/connect server at startup: {e}")

    def _create_action(self, window: Window, name: str, func: Callable[[], None]):
        action = window.createAction(f"ai_diffusion_{name}", "", "")
        action.triggered.connect(func)
        self._actions[name] = action

    def createActions(self, window):
        self._settings_dialog = SettingsDialog(window.qwindow(), self._server)
        self._create_action(window, "settings", self._settings_dialog.show)
        self._create_action(window, "generate", actions.generate)
        self._create_action(window, "cancel", actions.cancel_active)
        self._create_action(window, "cancel_queued", actions.cancel_queued)
        self._create_action(window, "cancel_all", actions.cancel_all)
        self._create_action(window, "apply", actions.apply)
        self._create_action(
            window, "set_workspace_generation", actions.set_workspace(Workspace.generation)
        )
        self._create_action(
            window, "set_workspace_upscaling", actions.set_workspace(Workspace.upscaling)
        )
        self._create_action(window, "toggle_workspace", actions.toggle_workspace)


Krita.instance().addExtension(AIToolsExtension(Krita.instance()))
Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("imageDiffusion", DockWidgetFactoryBase.DockRight, ImageDiffusionWidget)  # type: ignore
)
