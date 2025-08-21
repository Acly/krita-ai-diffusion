import sys
import pytest
import shutil
import subprocess
import dotenv
from pathlib import Path
from PyQt5.QtCore import QCoreApplication

sys.path.append(str(Path(__file__).parent.parent))
from ai_diffusion import eventloop, network, util
from .config import result_dir

root_dir = Path(__file__).parent.parent


def pytest_addoption(parser):
    parser.addoption("--test-install", action="store_true")
    parser.addoption("--pod-process", action="store_true")
    parser.addoption("--ci", action="store_true")
    parser.addoption("--benchmark", action="store_true")


def pytest_collection_modifyitems(session, config, items: list[pytest.Item]):
    def order(item: pytest.Item):
        if not item.parent or "test_workflow" not in item.parent.name:
            return 0
        if "cloud" in item.name and "sdxl" in item.name:
            return 11
        elif "cloud" in item.name:
            return 10
        elif "flux" in item.name:
            return 3
        elif "sdxl" in item.name:
            return 2
        else:
            return 1

    items.sort(key=order)


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


@pytest.fixture(scope="session", autouse=True)
def clear_appdata():
    user_dir = util.user_data_dir
    assert user_dir.name == ".appdata", "expected local test appdata dir"

    data_dir = user_dir / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)


@pytest.fixture(scope="session", autouse=True)
def clear_results():
    if result_dir.exists():
        for file in result_dir.iterdir():
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()
    result_dir.mkdir(exist_ok=True)


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


has_local_cloud = (root_dir / "service").exists()

if has_local_cloud:
    dotenv.load_dotenv(root_dir / "service" / "web" / ".env.local")
