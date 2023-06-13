import sys
import pytest
from pathlib import Path
from PyQt5.QtCore import QCoreApplication

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from ai_tools import eventloop

class QtTestApp:
    def __init__(self):
        self._app = QCoreApplication([])
        eventloop.setup()
    
    def run(self, coro):
        task = eventloop.run(coro)
        while not task.done():
            self._app.processEvents()
        return task.result()

@pytest.fixture(scope='session')
def qtapp():
    return QtTestApp()
