import asyncio
import aiohttp
import sys
import psutil
import pytest
import os
import shutil
import subprocess
import dotenv
import json
from pathlib import Path
from typing import Any
from PyQt5.QtCore import QCoreApplication

sys.path.append(str(Path(__file__).parent.parent))
from ai_diffusion import eventloop, network, util
from .config import result_dir

root_dir = Path(__file__).parent.parent


def pytest_addoption(parser):
    parser.addoption("--test-install", action="store_true")
    parser.addoption("--cloud", action="store_true")
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
        elif "flux2" in item.name:
            return 4
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


class CloudService:
    def __init__(self, loop: QtTestApp, enabled=True):
        self.loop = loop
        self.dir = root_dir / "service"
        self.log_dir = result_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.url = os.environ.get("TEST_SERVICE_URL", "http://localhost:8787")
        self.coord_proc: asyncio.subprocess.Process | None = None
        self.coord_log = None
        self.worker_proc: asyncio.subprocess.Process | None = None
        self.worker_task: asyncio.Task | None = None
        self.worker_log = None
        self.worker_url = ""
        self.worker_secret = ""
        self.enabled = has_local_cloud and enabled

    async def serve(self, process: asyncio.subprocess.Process, log_file):
        try:
            async for line in util.ensure(process.stdout):
                print(line.decode("utf-8"), end="", file=log_file, flush=True)
        except asyncio.CancelledError:
            pass

    async def check(self, url: str, token: str | None = None) -> bool:
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url) as response:
                    return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def launch_coordinator(self):
        assert self.coord_proc is None, "Coordinator already running"
        self.coord_log = open(self.log_dir / "api.log", "w", encoding="utf-8")
        if await self.check(f"{self.url}/health"):
            print(f"Coordinator running in external process at {self.url}", file=self.coord_log)
            return

        npm = shutil.which("npm")
        assert npm is not None, "npm not found in PATH"
        args = [npm, "run", "dev"]
        self.coord_proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=self.dir / "api",
            stdout=self.coord_log,
            stderr=asyncio.subprocess.STDOUT,
        )

    async def launch_worker(self, job_timeout=480):
        if self.worker_proc and self.worker_proc.returncode is None:
            return
        config = self.dir / "pod" / "_var" / "worker.json"
        assert config.exists(), "Worker config not found"
        config_dict = json.loads(config.read_text(encoding="utf-8"))
        self.worker_url = config_dict["public_url"]
        self.worker_secret = config_dict["admin_secret"]
        if config_dict["job_timeout"] != job_timeout:
            config_dict["job_timeout"] = job_timeout
            config.write_text(json.dumps(config_dict), encoding="utf-8")

        if self.worker_log is None:
            self.worker_log = open(self.log_dir / "worker.log", "w", encoding="utf-8")

        if await self.check(f"{self.worker_url}/health", token=self.worker_secret):
            print(f"Worker running in external process at {self.worker_url}", file=self.worker_log)
            return

        workerpy = str(self.dir / "pod" / "worker.py")
        args = ["-u", "-Xutf8", workerpy, str(config)]
        self.worker_proc = await asyncio.create_subprocess_exec(
            sys.executable,
            *args,
            cwd=self.dir / "pod",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert self.worker_proc.stdout is not None
        async for line in self.worker_proc.stdout:
            text = line.decode("utf-8")
            print(text[:80], end="", file=self.worker_log, flush=True)
            if "Uvicorn running" in text:
                break

        self.worker_task = asyncio.create_task(self.serve(self.worker_proc, self.worker_log))

    async def start(self):
        if not self.enabled or not has_local_cloud:
            return
        try:
            await self.launch_coordinator()
            await self.launch_worker()
        except Exception as e:
            await self.stop()
            raise e

    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
            await self.worker_task
        if self.worker_proc:
            self.worker_proc.terminate()
            await self.worker_proc.wait()
        if self.coord_proc and self.coord_proc.pid:
            children = psutil.Process(self.coord_proc.pid).children(recursive=True)
            for child in children:
                child.terminate()
            self.coord_proc.terminate()
            await self.coord_proc.wait()

    async def create_user(self, username: str) -> dict[str, Any]:
        assert self.enabled, "Cloud service is not enabled"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.url}/admin/user/create",
                json={"name": username},
            ) as response:
                response.raise_for_status()
                result = await response.json()
                if "error" in result:
                    raise Exception(result["error"])
                return result

    async def set_worker_job_timeout(self, timeout: int):
        if self.worker_proc is not None:
            self.worker_proc.terminate()
            await self.worker_proc.wait()
            await self.launch_worker(job_timeout=timeout)
        else:  # running in external process which will restart itself automatically
            headers = {
                "Authorization": f"Bearer {self.worker_secret}",
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(
                    f"{self.worker_url}/configure", json={"job_timeout": timeout}
                ) as response:
                    response.raise_for_status()

    def __enter__(self):
        self.loop.run(self.start())
        return self

    def __exit__(self, exc_type, exc, tb):
        self.loop.run(self.stop())


@pytest.fixture(scope="session")
def cloud_service(qtapp, pytestconfig):
    with CloudService(qtapp, pytestconfig.getoption("--cloud")) as service:
        yield service
