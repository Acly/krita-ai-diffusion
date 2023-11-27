import sys
from typing import Callable, Optional
from PyQt5.QtWidgets import QAction
from krita import Extension, Krita, DockWidgetFactory, DockWidgetFactoryBase, Window  # type: ignore

from . import eventloop, settings, __version__, Server, ServerMode, ServerState, Workspace
from .ui import actions, ImageDiffusionWidget, SettingsDialog
from .connection import Connection, ConnectionState
from .root import Root
from .util import client_logger as log


class AIToolsExtension(Extension):
    _actions: dict[str, QAction] = {}
    _root: Root

    def __init__(self, parent):
        super().__init__(parent)
        log.info(f"Extension initialized, Version: {__version__}, Python: {sys.version}")

        eventloop.setup()
        settings.load()
        self._root = Root()
        ImageDiffusionWidget._root = self._root

        notifier = Krita.instance().notifier()
        notifier.setActive(True)
        notifier.applicationClosing.connect(self.shutdown)

    def setup(self):
        eventloop.run(self._root.autostart())

    def shutdown(self):
        self._root.server.terminate()
        eventloop.stop()

    def _create_action(self, window: Window, name: str, func: Callable[[], None]):
        action = window.createAction(f"ai_diffusion_{name}", "", "")
        action.triggered.connect(func)
        self._actions[name] = action

    def createActions(self, window):
        self._settings_dialog = SettingsDialog(window.qwindow(), self._root.server)
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
