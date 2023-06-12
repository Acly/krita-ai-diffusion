import sys
import pytest
import time
from pathlib import Path
from PyQt5.QtCore import QCoreApplication

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

class QtTestApp:
    def __init__(self):
        self._app = QCoreApplication([])
    
    def wait(self, cond):
        while not cond():
            self._app.processEvents()

@pytest.fixture(scope='session')
def qtapp():
    return QtTestApp()
