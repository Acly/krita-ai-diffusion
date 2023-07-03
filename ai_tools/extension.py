from typing import Callable, Optional
from krita import Extension, Krita
from . import eventloop, settings
from .ui import Model, State, SettingsDialog, DiffusionServer


class AIToolsExtension(Extension):
    _actions = {}

    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        eventloop.setup()
        settings.load()
        DiffusionServer.instance().connect()

    @staticmethod
    def generate_action():
        model = Model.active()
        if model.state == State.setup:
            model.setup()
            model.generate()
        elif model.state == State.preview:
            model.generate()

    @staticmethod
    def cancel_action():
        model = Model.active()
        if State.generating in model.state:
            model.cancel()

    @staticmethod
    def apply_action():
        model = Model.active()
        if State.preview in model.state:
            model.apply_current_result()
            model.reset()

    @staticmethod
    def apply_multiple_action():
        model = Model.active()
        if State.preview in model.state:
            model.apply_current_result()

    @staticmethod
    def discard_action():
        model = Model.active()
        if State.preview in model.state:
            model.reset()

    def _create_action(self, window, name: str, func: Callable[[], None]):
        action = window.createAction(f"ai_tools_{name}", "", "")
        action.triggered.connect(func)
        self._actions[name] = action

    def createActions(self, window):
        self._settings_dialog = SettingsDialog()
        self._create_action(window, "settings", self._settings_dialog.show)
        self._create_action(window, "generate", self.generate_action)
        self._create_action(window, "cancel", self.cancel_action)
        self._create_action(window, "apply", self.apply_action)
        self._create_action(window, "apply_multiple", self.apply_multiple_action)
        self._create_action(window, "discard", self.discard_action)


Krita.instance().addExtension(AIToolsExtension(Krita.instance()))
