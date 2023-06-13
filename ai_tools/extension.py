from krita import Extension, Krita
from . import eventloop

class AIToolsExtension(Extension):

    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        eventloop.setup()

    def createActions(self, window):
        pass

Krita.instance().addExtension(AIToolsExtension(Krita.instance()))
