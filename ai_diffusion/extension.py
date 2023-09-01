from typing import Callable, Optional
from krita import Extension, Krita, DockWidgetFactory, DockWidgetFactoryBase

from . import eventloop, settings
from .ui import actions, ImageDiffusionWidget, Model, SettingsDialog, Connection


class AIToolsExtension(Extension):
    _actions = {}

    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        eventloop.setup()
        settings.load()
        Connection.instance().connect()

    def _create_action(self, window, name: str, func: Callable[[], None]):
        action = window.createAction(f"ai_diffusion_{name}", "", "")
        action.triggered.connect(func)
        self._actions[name] = action

    def createActions(self, window):
        self._settings_dialog = SettingsDialog(window.qwindow())
        self._create_action(window, "settings", self._settings_dialog.show)
        self._create_action(window, "generate", actions.generate)
        self._create_action(window, "cancel", actions.cancel)
        self._create_action(window, "apply", actions.apply)


Krita.instance().addExtension(AIToolsExtension(Krita.instance()))
Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("imageDiffusion", DockWidgetFactoryBase.DockTop, ImageDiffusionWidget)
)
