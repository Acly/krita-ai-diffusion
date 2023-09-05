from typing import Callable, Optional
from krita import Extension, Krita, DockWidgetFactory, DockWidgetFactoryBase

from . import eventloop, settings, Server, ServerMode, ServerState
from .ui import actions, ImageDiffusionWidget, SettingsDialog, Connection, ConnectionState
from .util import client_logger as log


class AIToolsExtension(Extension):
    _actions = {}
    _server: Server

    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        eventloop.setup()
        settings.load()
        self._server = Server(settings.server_path)
        eventloop.run(self.autostart())

        notifier = Krita.instance().notifier()
        notifier.setActive(True)
        notifier.applicationClosing.connect(self._server.terminate)

    async def autostart(self):
        connection = Connection.instance()
        try:
            if (
                settings.server_mode is ServerMode.managed
                and self._server.state is ServerState.stopped
            ):
                url = await self._server.start()
                await connection._connect(url)
            elif settings.server_mode in [ServerMode.undefined, ServerMode.external]:
                await connection._connect(settings.server_url)
                if settings.server_mode is ServerMode.undefined:
                    if connection.state is ConnectionState.connected:
                        settings.server_mode = ServerMode.external
                    else:
                        settings.server_mode = ServerMode.managed
        except Exception as e:
            log.warning(f"Failed to launch/connect server at startup: {e}")

    def _create_action(self, window, name: str, func: Callable[[], None]):
        action = window.createAction(f"ai_diffusion_{name}", "", "")
        action.triggered.connect(func)
        self._actions[name] = action

    def createActions(self, window):
        self._settings_dialog = SettingsDialog(window.qwindow(), self._server)
        self._create_action(window, "settings", self._settings_dialog.show)
        self._create_action(window, "generate", actions.generate)
        self._create_action(window, "cancel", actions.cancel)
        self._create_action(window, "apply", actions.apply)


Krita.instance().addExtension(AIToolsExtension(Krita.instance()))
Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("imageDiffusion", DockWidgetFactoryBase.DockTop, ImageDiffusionWidget)
)
