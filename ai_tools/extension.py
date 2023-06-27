from krita import Extension, Krita
from . import eventloop
from .settings import SettingsDialog


class AIToolsExtension(Extension):
    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        eventloop.setup()

    def createActions(self, window):
        self._settings_dialog = SettingsDialog()
        self._settings_action = window.createAction(
            "ai_tools_configure", "Configure AI Tools", "tools/scripts"
        )
        self._settings_action.triggered.connect(self._settings_dialog.show)


Krita.instance().addExtension(AIToolsExtension(Krita.instance()))
