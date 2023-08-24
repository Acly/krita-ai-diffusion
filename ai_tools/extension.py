from typing import Callable, Optional
from krita import Extension, Krita, DockWidgetFactory, DockWidgetFactoryBase

from . import eventloop, settings
from .ui import ImageDiffusionWidget, Model, SettingsDialog, Connection


class AIToolsExtension(Extension):
    _actions = {}

    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        eventloop.setup()
        settings.load()
        Connection.instance().connect()

    @staticmethod
    def generate_action():
        model = Model.active()
        model.generate()

    @staticmethod
    def cancel_action():
        model = Model.active()
        if model.jobs.any_executing():
            model.cancel()

    @staticmethod
    def apply_action():
        model = Model.active()
        if model.can_apply_result:
            model.apply_current_result()

    def _create_action(self, window, name: str, func: Callable[[], None]):
        action = window.createAction(f"ai_tools_{name}", "", "")
        action.triggered.connect(func)
        self._actions[name] = action

    def createActions(self, window):
        pass
        self._settings_dialog = SettingsDialog()
        self._create_action(window, "settings", self._settings_dialog.show)
        self._create_action(window, "generate", self.generate_action)
        self._create_action(window, "cancel", self.cancel_action)
        self._create_action(window, "apply", self.apply_action)


Krita.instance().addExtension(AIToolsExtension(Krita.instance()))
Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("imageDiffusion", DockWidgetFactoryBase.DockTop, ImageDiffusionWidget)
)
