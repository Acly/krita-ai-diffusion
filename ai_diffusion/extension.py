import sys
from typing import Callable
from PyQt5.QtWidgets import QAction
from krita import Extension, Krita, DockWidgetFactory, DockWidgetFactoryBase, Window  # type: ignore

from . import eventloop, __version__
from .settings import settings
from .model import Workspace
from .ui import actions
from .ui.diffusion import ImageDiffusionWidget
from .ui.settings import SettingsDialog
from .root import root
from .util import client_logger as log


class AIToolsExtension(Extension):
    _actions: dict[str, QAction] = {}
    _settings_dialog: SettingsDialog

    def __init__(self, parent):
        super().__init__(parent)
        log.info(f"Extension initialized, Version: {__version__}, Python: {sys.version}")

        eventloop.setup()
        settings.load()
        root.init()
        self._settings_dialog = SettingsDialog(root.server)

        notifier = Krita.instance().notifier()
        notifier.setActive(True)
        notifier.applicationClosing.connect(self.shutdown)

    def setup(self):
        eventloop.run(root.autostart(self._settings_dialog.connection.update))

    def shutdown(self):
        root.server.terminate()
        eventloop.stop()

    def _create_action(self, window: Window, name: str, func: Callable[[], None]):
        action = window.createAction(f"ai_diffusion_{name}", "", "")
        action.triggered.connect(func)
        self._actions[name] = action

    def createActions(self, window):
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
