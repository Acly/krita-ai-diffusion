from __future__ import annotations
import asyncio
import locale
from enum import Enum
from itertools import chain
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, NamedTuple, Optional, Union
from zipfile import ZipFile
from PyQt5.QtNetwork import QNetworkAccessManager

from .settings import settings, ServerBackend
from . import resources, __version__
from .resources import CustomNode, ModelResource
from .network import download, DownloadProgress
from .util import is_windows, client_logger as log, server_logger as server_log


_exe = ".exe" if is_windows else ""
_process_flags = subprocess.CREATE_NO_WINDOW if is_windows else 0


class ServerState(Enum):
    not_installed = 0
    missing_resources = 2
    installing = 3
    stopped = 4
    starting = 5
    running = 6


class InstallationProgress(NamedTuple):
    stage: str
    progress: Optional[DownloadProgress] = None
    message: str = ""


Callback = Callable[[InstallationProgress], None]
InternalCB = Callable[[str, Union[str, DownloadProgress]], None]


class Server:
    path: Path
    url: Optional[str] = None
    backend = ServerBackend.cuda
    state = ServerState.stopped
    missing_resources: list[str]
    comfy_dir: Optional[Path] = None
    version: Optional[str] = None

    _python_cmd: Optional[Path] = None
    _pip_cmd: Optional[Path] = None
    _cache_dir: Path
    _version_file: Path
    _process: Optional[asyncio.subprocess.Process] = None
    _task: Optional[asyncio.Task] = None

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or settings.server_path)
        if not self.path.is_absolute():
            self.path = Path(__file__).parent / self.path
        self.backend = settings.server_backend
        self.check_install()

    def check_install(self):
        self.missing_resources = []
        self._cache_dir = self.path / ".cache"

        self._version_file = self.path / ".version"
        if self._version_file.exists():
            self.version = self._version_file.read_text().strip()
            log.info(f"Found server installation v{self.version} at {self.path}")
        else:
            self.version = None

        comfy_pkg = ["main.py", "nodes.py", "custom_nodes"]
        self.comfy_dir = _find_component(comfy_pkg, [self.path, self.path / "ComfyUI"])

        python_pkg = ["python3.dll", "python.exe"] if is_windows else ["python3", "pip3"]
        python_search_paths = [
            self.path / "python",
            self.path / "venv" / "bin",
            self.path / ".venv" / "bin",
        ]
        if self.comfy_dir:
            python_search_paths += [
                self.comfy_dir / "python",
                self.comfy_dir / "venv" / "bin",
                self.comfy_dir / ".venv" / "bin",
                self.comfy_dir.parent / "python",
                self.comfy_dir.parent / "python_embeded",
            ]
        python_path = _find_component(python_pkg, python_search_paths)
        if python_path is None:
            self._python_cmd = _find_program("python", "python3")
            self._pip_cmd = _find_program("pip", "pip3")
        else:
            self._python_cmd = python_path / f"python{_exe}"
            self._pip_cmd = python_path / "pip"
            if is_windows:
                self._pip_cmd = python_path / "Scripts" / "pip.exe"

        if not (self.has_comfy and self.has_python):
            self.state = ServerState.not_installed
            self.missing_resources = resources.all
            return

        assert self.comfy_dir is not None
        missing_nodes = [
            package.name
            for package in resources.required_custom_nodes
            if not Path(self.comfy_dir / "custom_nodes" / package.folder).exists()
        ]
        self.missing_resources += missing_nodes

        def find_missing(folder: Path, resources: list[ModelResource]):
            return [
                resource.name
                for resource in resources
                if not (folder / resource.folder / resource.filename).exists()
            ]

        self.missing_resources += find_missing(self.comfy_dir, resources.required_models)
        if len(self.missing_resources) > 0:
            self.state = ServerState.missing_resources
        else:
            self.state = ServerState.stopped

        # Optional resources
        self.missing_resources += find_missing(self.comfy_dir, resources.default_checkpoints)
        self.missing_resources += find_missing(self.comfy_dir, resources.optional_models)

    async def _install(self, cb: InternalCB):
        self.state = ServerState.installing
        cb("Installing", f"Installation started in {self.path}")

        network = QNetworkAccessManager()
        self._cache_dir = self.path / ".cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        no_python = self._python_cmd is None or self._pip_cmd is None
        if is_windows and (self.comfy_dir is None or no_python):
            # On Windows install an embedded version of Python
            python_dir = self.path / "python"
            self._python_cmd = python_dir / f"python{_exe}"
            self._pip_cmd = python_dir / "Scripts" / f"pip{_exe}"
            await _install_if_missing(python_dir, self._install_python, network, cb)
        elif not is_windows and (self.comfy_dir is None or self._pip_cmd is None):
            # On Linux a system Python is required to create a virtual environment
            python_dir = self.path / "venv"
            await _install_if_missing(python_dir, self._create_venv, cb)
            self._python_cmd = python_dir / "bin" / "python3"
            self._pip_cmd = python_dir / "bin" / "pip3"
        assert self._python_cmd is not None and self._pip_cmd is not None
        log.info(f"Using Python: {await get_python_version(self._python_cmd)}, {self._python_cmd}")
        log.info(f"Using pip: {await get_python_version(self._pip_cmd)}, {self._pip_cmd}")

        self.comfy_dir = self.comfy_dir or self.path / "ComfyUI"
        await _install_if_missing(self.comfy_dir, self._install_comfy, network, cb)

        for pkg in resources.required_custom_nodes:
            dir = self.comfy_dir / "custom_nodes" / pkg.folder
            await _install_if_missing(dir, self._install_custom_node, pkg, network, cb)

        for resource in resources.required_models:
            target_folder = self.comfy_dir / resource.folder
            target_file = self.comfy_dir / resource.folder / resource.filename
            if not target_file.exists():
                target_folder.mkdir(parents=True, exist_ok=True)
                await _download_cached(resource.name, network, resource.url, target_file, cb)

        self._version_file.write_text(__version__)

        self.state = ServerState.stopped
        cb("Finished", f"Installation finished in {self.path}")
        self.check_install()

    async def _install_python(self, network: QNetworkAccessManager, cb: InternalCB):
        url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
        archive_path = self._cache_dir / "python-3.10.11-embed-amd64.zip"
        dir = self.path / "python"

        await _download_cached("Python", network, url, archive_path, cb)
        await _extract_archive("Python", archive_path, dir, cb)

        python_pth = dir / "python310._pth"
        cb("Installing Python", f"Patching {python_pth}")
        with open(python_pth, "a") as file:
            file.write("import site\n")

        git_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        get_pip_file = dir / "get-pip.py"
        await _download_cached("Python", network, git_pip_url, get_pip_file, cb)
        await _execute_process("Python", [self._python_cmd, get_pip_file], dir, cb)

        cb("Installing Python", f"Patching {python_pth}")
        _prepend_file(python_pth, "../ComfyUI\n")
        cb("Installing Python", "Finished installing Python")

    async def _create_venv(self, cb: InternalCB):
        cb("Creating Python virtual environment", f"Creating venv in {self.path / 'venv'}")
        venv_cmd = [self._python_cmd, "-m", "venv", "venv"]
        await _execute_process("Python", venv_cmd, self.path, cb)

    async def _install_comfy(self, network: QNetworkAccessManager, cb: InternalCB):
        assert self.comfy_dir is not None
        url = "https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip"
        archive_path = self._cache_dir / "ComfyUI.zip"
        await _download_cached("ComfyUI", network, url, archive_path, cb)
        await _extract_archive("ComfyUI", archive_path, self.comfy_dir.parent, cb)
        _rename_extracted_folder("ComfyUI", self.comfy_dir, "-master")

        torch_args = ["install", "torch", "torchvision", "torchaudio"]
        if self.backend is ServerBackend.cpu:
            torch_args += ["--index-url", "https://download.pytorch.org/whl/cpu"]
        elif self.backend is ServerBackend.cuda:
            torch_args += ["--index-url", "https://download.pytorch.org/whl/cu121"]
        await _execute_process("PyTorch", [self._pip_cmd, *torch_args], self.comfy_dir, cb)

        requirements_txt = self.comfy_dir / "requirements.txt"
        requirements_cmd = [self._pip_cmd, "install", "-r", requirements_txt]
        await _execute_process("ComfyUI", requirements_cmd, self.comfy_dir, cb)

        if self.backend is ServerBackend.directml:
            await _execute_process(  # for some reason this must come AFTER ComfyUI requirements
                "PyTorch", [self._pip_cmd, "install", "torch-directml"], self.comfy_dir, cb
            )
        cb("Installing ComfyUI", "Finished installing ComfyUI")

    async def _install_custom_node(
        self, pkg: CustomNode, network: QNetworkAccessManager, cb: InternalCB
    ):
        assert self.comfy_dir is not None
        folder = self.comfy_dir / "custom_nodes" / pkg.folder
        resource_url = f"{pkg.url}/archive/refs/heads/main.zip"
        resource_zip_path = self._cache_dir / f"{pkg.folder}.zip"
        await _download_cached(pkg.name, network, resource_url, resource_zip_path, cb)
        await _extract_archive(pkg.name, resource_zip_path, folder.parent, cb)
        _rename_extracted_folder(pkg.name, folder, "-main")

        requirements_txt = folder / "requirements.txt"
        requirements_cmd = [self._pip_cmd, "install", "-r", requirements_txt]
        if requirements_txt.exists():
            await _execute_process(pkg.name, requirements_cmd, folder, cb)
        cb(f"Installing {pkg.name}", f"Finished installing {pkg.name}")

    async def install(self, callback: Callback):
        assert self.state in [ServerState.not_installed, ServerState.missing_resources]
        if not is_windows and self._python_cmd is None:
            raise Exception(
                "Python not found. Please install python3, python3-venv via your package manager"
                " and restart."
            )

        def cb(stage: str, message: str | DownloadProgress):
            out_message = ""
            progress = None
            filters = ["Downloading", "Installing", "Collecting", "Using"]
            if isinstance(message, str):
                log.info(message)
                if any(s in message[:16] for s in filters):
                    out_message = message
            elif isinstance(message, DownloadProgress):
                progress = message
            callback(InstallationProgress(stage, progress, out_message))

        try:
            await self._install(cb)
        except Exception as e:
            log.exception(str(e))
            log.error("Installation failed")
            self.state = ServerState.stopped
            self.check_install()
            raise e

    async def install_optional(self, packages: list[str], callback: Callback):
        assert self.comfy_dir, "Must install ComfyUI before downloading checkpoints"
        network = QNetworkAccessManager()
        prev_state = self.state
        self.state = ServerState.installing

        def cb(stage: str, message: str | DownloadProgress):
            if isinstance(message, str):
                log.info(message)
            progress = message if isinstance(message, DownloadProgress) else None
            callback(InstallationProgress(stage, progress))

        try:
            all_optional = chain(resources.default_checkpoints, resources.optional_models)
            to_install = (r for r in all_optional if r.name in packages)
            for resource in to_install:
                target_file = self.comfy_dir / resource.folder / resource.filename
                if not target_file.exists():
                    await _download_cached(resource.name, network, resource.url, target_file, cb)
        except Exception as e:
            log.exception(str(e))
            raise e
        finally:
            self.state = prev_state
            self.check_install()

    async def start(self):
        assert self.state in [ServerState.stopped, ServerState.missing_resources]
        assert self._python_cmd

        self.state = ServerState.starting
        args = ["-u", "-X", "utf8", "main.py"]
        if self.backend is ServerBackend.cpu:
            args.append("--cpu")
        elif self.backend is ServerBackend.directml:
            args.append("--directml")
        if settings.server_arguments:
            args += settings.server_arguments.split(" ")
        self._process = await asyncio.create_subprocess_exec(
            self._python_cmd,
            *args,
            cwd=self.comfy_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=_process_flags,
        )

        assert self._process.stdout is not None
        async for line in self._process.stdout:
            text = line.decode("utf-8").strip()
            server_log.info(text)
            if text.startswith("To see the GUI go to:"):
                self.state = ServerState.running
                self.url = text.split("http://")[-1]
                break

        if self.state != ServerState.running:
            error = "Process exited unexpectedly"
            try:
                out, err = await asyncio.wait_for(self._process.communicate(), timeout=10)
                server_log.info(out.decode("utf-8").strip())
                error = err.decode("utf-8")
                server_log.error(error)
            except asyncio.TimeoutError:
                self._process.kill()

            self.state = ServerState.stopped
            ret = self._process.returncode
            self._process = None
            raise Exception(f"Error during server startup: {error} [{ret}]")

        self._task = asyncio.create_task(self.run())
        assert self.url is not None
        return self.url

    async def run(self):
        assert self.state is ServerState.running
        assert self._process and self._process.stdout and self._process.stderr

        async def forward(stream: asyncio.StreamReader):
            async for line in stream:
                server_log.info(line.decode().strip())

        try:
            await asyncio.gather(
                forward(self._process.stdout),
                forward(self._process.stderr),
            )
        except asyncio.CancelledError:
            pass

    async def stop(self):
        assert self.state is ServerState.running
        try:
            if self._process and self._task:
                log.info("Stopping server")
                self._process.terminate()
                self._task.cancel()
                await asyncio.wait_for(self._process.communicate(), timeout=5)
                log.info(f"Server terminated with code {self._process.returncode}")
        except asyncio.TimeoutError:
            log.warning("Server did not terminate in time")
            pass
        finally:
            self.state = ServerState.stopped
            self._process = None
            self._task = None

    def terminate(self):
        try:
            if self._process is not None:
                self._process.terminate()
        except Exception as e:
            print(e)
            pass

    @property
    def has_python(self):
        return self._python_cmd is not None and self._pip_cmd is not None

    @property
    def has_comfy(self):
        return self.comfy_dir is not None


def _find_component(files: list[str], search_paths: list[Path]):
    return next(
        (
            path
            for path in search_paths
            if all(p.exists() for p in [path] + [path / file for file in files])
        ),
        None,
    )


def _find_program(*commands: str):
    for command in commands:
        p = shutil.which(command)
        if p is not None:
            return Path(p)
    return None


async def _download_cached(
    name: str, network: QNetworkAccessManager, url: str, file: Path, cb: InternalCB
):
    if file.exists():
        cb(f"Found existing {name}", f"Using {file}")
    else:
        cb(f"Downloading {name}", f"Downloading {url} to {file}")
        async for progress in download(network, url, file):
            cb(f"Downloading {name}", progress)


async def _extract_archive(name: str, archive: Path, target: Path, cb: InternalCB):
    cb(f"Installing {name}", f"Extracting {archive} to {target}")
    with ZipFile(archive) as zip_file:
        zip_file.extractall(target)


async def _execute_process(name: str, cmd: list, cwd: Path, cb: InternalCB):
    PIPE = asyncio.subprocess.PIPE
    enc = locale.getpreferredencoding(False)
    errlog = ""

    cmd = [str(c) for c in cmd]
    cb(f"Installing {name}", f"Executing {' '.join(cmd)}")
    process = await asyncio.create_subprocess_exec(
        cmd[0], *cmd[1:], cwd=cwd, stdout=PIPE, stderr=PIPE, creationflags=_process_flags
    )

    async def forward(stream: asyncio.StreamReader):
        async for line in stream:
            cb(f"Installing {name}", line.decode(enc, errors="surrogateescape").strip())

    async def collect(stream: asyncio.StreamReader):
        nonlocal errlog
        async for line in stream:
            errlog += line.decode(enc, errors="surrogateescape")

    assert process.stdout and process.stderr
    await asyncio.gather(forward(process.stdout), collect(process.stderr))

    if process.returncode != 0:
        if errlog == "":
            errlog = f"Process exited with code {process.returncode}"
        raise Exception(f"Error during installation: {errlog}")


async def _install_if_missing(path: Path, installer, *args):
    if not path.exists():
        try:
            await installer(*args)
        except Exception as e:
            shutil.rmtree(path, ignore_errors=True)
            raise e


def _prepend_file(path: Path, line: str):
    with open(path, "r+") as file:
        lines = file.readlines()
        lines.insert(0, line)
        file.seek(0)
        file.writelines(lines)
        file.truncate()


def _rename_extracted_folder(name: str, path: Path, suffix: str):
    extracted_folder = path.parent / f"{path.name}{suffix}"
    if not extracted_folder.exists():
        raise Exception(
            f"Error during {name} installation: folder {extracted_folder} does not exist"
        )
    extracted_folder.rename(path)


async def get_python_version(python_cmd: Path):
    enc = locale.getpreferredencoding(False)
    proc = await asyncio.create_subprocess_exec(
        python_cmd, "--version", stdout=asyncio.subprocess.PIPE
    )
    out, _ = await proc.communicate()
    return out.decode(enc).strip()
