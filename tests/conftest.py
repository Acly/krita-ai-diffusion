import sys
import pytest
import subprocess
from pathlib import Path
from PyQt5.QtCore import QCoreApplication

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from ai_diffusion import eventloop, network
from ai_diffusion.settings import settings


def pytest_addoption(parser):
    parser.addoption("--test-install", action="store_true")
    parser.addoption("--ci", action="store_true")
    parser.addoption("--benchmark", action="store_true")


class QtTestApp:
    def __init__(self):
        self._app = QCoreApplication([])
        eventloop.setup()

    def run(self, coro):
        task = eventloop.run(coro)
        while not task.done():
            self._app.processEvents()
        return task.result()


@pytest.fixture(scope="session")
def qtapp():
    return QtTestApp()


@pytest.fixture()
def temp_settings():
    yield settings
    settings.restore()


@pytest.fixture()
def local_download_server():
    script = root_dir / "scripts" / "file_server.py"
    port = 51222

    with subprocess.Popen([sys.executable, str(script), str(port)]) as proc:
        assert proc.poll() is None
        network.HOSTMAP = network.HOSTMAP_LOCAL
        yield f"http://localhost:{port}"
        network.HOSTMAP = {}

        proc.terminate()
        proc.wait()
