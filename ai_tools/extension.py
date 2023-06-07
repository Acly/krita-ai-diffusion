from krita import Extension, Krita

class AIToolsExtension(Extension):

    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        pass

    def createActions(self, window):
        pass

Krita.instance().addExtension(AIToolsExtension(Krita.instance()))
