import asyncio
from enum import Enum
from pathlib import Path
import shutil
import sys
from typing import Callable, List, NamedTuple, Optional, Sequence
from zipfile import ZipFile
from PyQt5.QtNetwork import QNetworkAccessManager

from .settings import settings
from .network import download


class CustomNode(NamedTuple):
    name: str
    folder: str
    url: str
    nodes: Sequence[str]


class ResourceKind(Enum):
    checkpoint = "SD Checkpoint"
    controlnet = "ControlNet model"
    clip_vision = "CLIP Vision model"
    ip_adapter = "IP-Adapter model"
    node = "custom node"


class ModelResource(NamedTuple):
    name: str
    kind: ResourceKind
    folder: Path
    filename: str
    url: str


required_custom_nodes = [
    CustomNode(
        "ControlNet Preprocessors",
        "comfyui_controlnet_aux",
        "https://github.com/Fannovel16/comfyui_controlnet_aux",
        ["InpaintPreprocessor"],
    ),
    CustomNode(
        "IP-Adapter",
        "IPAdapter-ComfyUI",
        "https://github.com/laksjdjf/IPAdapter-ComfyUI",
        ["IPAdapter"],
    ),
    CustomNode(
        "External Tooling Nodes",
        "comfyui-tooling-nodes",
        "https://github.com/Acly/comfyui-tooling-nodes",
        [
            "ETN_LoadImageBase64",
            "ETN_LoadMaskBase64",
            "ETN_SendImageWebSocket",
            "ETN_CropImage",
            "ETN_ApplyMaskToImage",
        ],
    ),
]


required_models = [
    ModelResource(
        "ControlNet Inpaint",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_v11p_sd15_inpaint.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors",
    ),
    ModelResource(
        "CLIP Vision",
        ResourceKind.clip_vision,
        Path("models/clip_vision/SD1.5"),
        "pytorch_model.bin",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin",
    ),
    ModelResource(
        "IP-Adapter",
        ResourceKind.ip_adapter,
        Path("custom_nodes/IPAdapter-ComfyUI/models"),
        "ip-adapter_sd15.bin",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin",
    ),
]

default_checkpoints = [
    ModelResource(
        "Realistic Vision",
        ResourceKind.checkpoint,
        Path("models/checkpoints"),
        "realisticVisionV51_v51VAE.safetensors",
        "https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    ),
    ModelResource(
        "DreamShaper",
        ResourceKind.checkpoint,
        Path("models/checkpoints"),
        "dreamshaper_7.safetensors",
        "https://civitai.com/api/download/models/109123?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    ),
]


class MissingResource(Exception):
    kind: ResourceKind
    names: Optional[Sequence[str]]

    def __init__(self, kind: ResourceKind, names: Optional[Sequence[str]] = None):
        self.kind = kind
        self.names = names

    def __str__(self):
        return f"Missing {self.kind.value}: {', '.join(self.names)}"


_all_resources = [
    MissingResource(ResourceKind.node, [n.name for n in required_custom_nodes]),
    MissingResource(ResourceKind.controlnet),
    MissingResource(ResourceKind.clip_vision),
    MissingResource(ResourceKind.ip_adapter),
    MissingResource(ResourceKind.checkpoint, [n.name for n in default_checkpoints]),
]

_is_windows = "win" in sys.platform
_exe = ".exe" if _is_windows else ""


class ServerBackend(Enum):
    cpu = "cpu"
    cuda = "cuda"


class ServerState(Enum):
    missing_comfy = 0
    missing_python = 1
    missing_resources = 2
    installing = 3
    stopped = 4
    starting = 5
    running = 6
    error = 7


class InstallationProgress(NamedTuple):
    stage: str
    message: str = ""
    progress: float = -1


Callback = Callable[[InstallationProgress], None]


class Server:
    path: Path
    url: str
    backend = ServerBackend.cuda
    state = ServerState.stopped
    missing_resources: List[MissingResource]

    _python_cmd: Optional[str]
    _pip_cmd: Optional[str]
    _comfy_dir: Optional[Path]
    _cache_dir: Optional[Path]
    _process: Optional[asyncio.subprocess.Process]
    _task: Optional[asyncio.Task]

    def __init__(self, path: str = None):
        self.path = Path(path or settings.server_path)
        if not self.path.is_absolute():
            self.path = Path(__file__).parent / self.path
        self.check_install()

    def check_install(self):
        self.missing_resources = []

        self._cache_dir = self.path / ".cache"
        comfy_pkg = ["main.py", "nodes.py", "custom_nodes"]
        self._comfy_dir = _find_component(comfy_pkg, [self.path, self.path / "ComfyUI"])
        if self._comfy_dir is None:
            self.state = ServerState.missing_comfy
            self.missing_resources = _all_resources
            return

        python_path = _find_component(
            ["python3.dll", "python.exe"] if _is_windows else ["python.so", "python"],
            [self.path / "python", self._comfy_dir / "python", self._comfy_dir / "../python"],
        )
        if python_path is None:
            self._python_cmd = _find_program("python")
            self._pip_cmd = _find_program("pip")
            if self._python_cmd is None or self._pip_cmd is None:
                self.state = ServerState.missing_python
                self.missing_resources = _all_resources
                return
        else:
            self._python_cmd = python_path / f"python{_exe}"
            self._pip_cmd = python_path / "Scripts" / f"pip{_exe}"

        missing_nodes = [
            package.name
            for package in required_custom_nodes
            if not Path(self._comfy_dir / "custom_nodes" / package.folder).exists()
        ]
        if len(missing_nodes) > 0:
            self.missing_resources.append(MissingResource(ResourceKind.node, missing_nodes))

        self.missing_resources += [
            MissingResource(r.kind)
            for r in required_models
            if not (self._comfy_dir / r.folder / r.filename).exists()
        ]
        if len(self.missing_resources) > 0:
            self.state = ServerState.missing_resources

        # Optional resources
        self.missing_resources += [
            MissingResource(r.kind, [r.name])
            for r in default_checkpoints
            if not (self._comfy_dir / r.folder / r.filename).exists()
        ]

    async def install(self, cb: Callback):
        assert self.state in [
            ServerState.missing_comfy,
            ServerState.missing_python,
            ServerState.missing_resources,
        ]
        self.state = ServerState.installing

        network = QNetworkAccessManager()
        self._cache_dir = self.path / ".cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        python_dir = self.path / "python"
        self._python_cmd = python_dir / f"python{_exe}"
        self._pip_cmd = python_dir / "Scripts" / f"pip{_exe}"
        await _install_if_missing(python_dir, self._install_python, network, cb)

        self._comfy_dir = self.path / "ComfyUI"
        await _install_if_missing(self._comfy_dir, self._install_comfy, network, cb)

        for pkg in required_custom_nodes:
            dir = self._comfy_dir / "custom_nodes" / pkg.folder
            await _install_if_missing(dir, self._install_custom_node, pkg, network, cb)

        for resource in required_models:
            target_folder = self._comfy_dir / resource.folder
            target_file = self._comfy_dir / resource.folder / resource.filename
            if not target_file.exists():
                target_folder.mkdir(parents=True, exist_ok=True)
                await _download_cached(resource.name, network, resource.url, target_file, cb)

        self.state = ServerState.stopped
        self.check_install()

    async def install_optional(self, cb: Callback):
        network = QNetworkAccessManager()
        prev_state = self.state
        self.state = ServerState.installing

        for resource in default_checkpoints:
            target_file = self._comfy_dir / resource.folder / resource.filename
            if not target_file.exists():
                await _download_cached(resource.name, network, resource.url, target_file, cb)

        self.state = prev_state
        self.check_install()

    async def _install_python(self, network: QNetworkAccessManager, cb: Callback):
        url = "https://www.python.org/ftp/python/3.10.9/python-3.10.9-embed-amd64.zip"
        archive_path = self._cache_dir / "python-3.10.9-embed-amd64.zip"
        dir = self.path / "python"

        await _download_cached("Python", network, url, archive_path, cb)
        await _extract_archive("Python", archive_path, dir, cb)

        python_pth = dir / "python310._pth"
        cb(InstallationProgress("Installing Python", f"Patching {python_pth}"))
        with open(python_pth, "a") as file:
            file.write("import site\n")

        git_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        get_pip_file = dir / "get-pip.py"
        await _download_cached("Python", network, git_pip_url, get_pip_file, cb)
        await _execute_process("Python", [self._python_cmd, get_pip_file], dir, cb)

        torch_args = ["install", "torch", "torchvision", "torchaudio", "--index-url"]
        torch_index = {
            ServerBackend.cuda: "https://download.pytorch.org/whl/cu118",
            ServerBackend.cpu: "https://download.pytorch.org/whl/cpu",
        }
        torch_cmd = [self._pip_cmd, *torch_args, torch_index[self.backend]]
        await _execute_process("PyTorch", torch_cmd, dir, cb)

        cb(InstallationProgress("Installing Python", f"Patching {python_pth}"))
        _prepend_file(python_pth, "../ComfyUI\n")
        cb(InstallationProgress("Installing Python", "Finished installing Python"))

    async def _install_comfy(self, network: QNetworkAccessManager, cb: Callback):
        url = "https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip"
        archive_path = self._cache_dir / "ComfyUI.zip"
        await _download_cached("ComfyUI", network, url, archive_path, cb)
        await _extract_archive("ComfyUI", archive_path, self._comfy_dir.parent, cb)
        _rename_extracted_folder("ComfyUI", self._comfy_dir, "-master")

        requirements_txt = self._comfy_dir / "requirements.txt"
        requirements_cmd = [self._pip_cmd, "install", "-r", requirements_txt]
        await _execute_process("ComfyUI", requirements_cmd, self._comfy_dir, cb)
        cb(InstallationProgress("Installing ComfyUI", "Finished installing ComfyUI"))

    async def _install_custom_node(
        self, pkg: CustomNode, network: QNetworkAccessManager, cb: Callback
    ):
        folder = self._comfy_dir / "custom_nodes" / pkg.folder
        resource_url = f"{pkg.url}/archive/refs/heads/main.zip"
        resource_zip_path = self._cache_dir / f"{pkg.folder}.zip"
        await _download_cached(pkg.name, network, resource_url, resource_zip_path, cb)
        await _extract_archive(pkg.name, resource_zip_path, folder.parent, cb)
        _rename_extracted_folder(pkg.name, folder, "-main")

        requirements_txt = folder / "requirements.txt"
        requirements_cmd = [self._pip_cmd, "install", "-r", requirements_txt]
        if requirements_txt.exists():
            await _execute_process(pkg.name, requirements_cmd, folder, cb)
        cb(InstallationProgress(f"Installing {pkg.name}", f"Finished installing {pkg.name}"))

    async def start(self, log_cb: Callable[[str], None]):
        assert self.state in [ServerState.stopped, ServerState.error, ServerState.missing_resources]

        self.state = ServerState.starting
        PIPE = asyncio.subprocess.PIPE
        args = ["-u", "main.py"] + (["--cpu"] if self.backend is ServerBackend.cpu else [])
        self._process = await asyncio.create_subprocess_exec(
            self._python_cmd, *args, cwd=self._comfy_dir, stdout=PIPE, stderr=PIPE
        )

        async for line in self._process.stdout:
            text = line.decode().strip()
            log_cb(text)
            if text.startswith("To see the GUI go to:"):
                self.state = ServerState.running
                self.url = text.split("http://")[-1]
                break

        if self.state != ServerState.running:
            self.error = "Process exited unexpectedly"
            try:
                out, err = await asyncio.wait_for(self._process.communicate(), timeout=10)
                log_cb(out.decode().strip())
                self.error = err.decode()
            except asyncio.TimeoutError:
                self._process.kill()

            self.state = ServerState.error
            ret = self._process.returncode
            self._process = None
            raise Exception(f"Error during server startup: {self.error} [{ret}]")

        self._task = asyncio.create_task(self.run(log_cb))
        return self.url

    async def run(self, log_cb: Callable[[str], None]):
        assert self.state is ServerState.running

        async def forward(stream: asyncio.StreamReader):
            async for line in stream:
                log_cb(line.decode().strip())

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
            self._process.terminate()
            self._task.cancel()
            await asyncio.wait_for(self._process.communicate(), timeout=5)
        except asyncio.TimeoutError:
            pass
        finally:
            self.state = ServerState.stopped
            self._process = None
            self._task = None


def _find_component(files: List[str], search_paths: List[Path]):
    return next(
        (
            path
            for path in search_paths
            if all(p.exists() for p in [path] + [path / file for file in files])
        ),
        None,
    )


def _find_program(command: str):
    p = shutil.which(command)
    return Path(p) if p is not None else None


async def _download_cached(
    name: str,
    network: QNetworkAccessManager,
    url: str,
    archive: Path,
    cb: Callback,
):
    if not archive.exists():
        cb(InstallationProgress(f"Downloading {name}", 0, f"Downloading {url} to {archive}"))
        async for progress in download(network, url, archive):
            cb(InstallationProgress(f"Downloading {name}", progress=progress))


async def _extract_archive(name: str, archive: Path, target: Path, cb: Callback):
    cb(InstallationProgress(f"Installing {name}", f"Extracting {archive} to {target}"))
    with ZipFile(archive) as zip_file:
        zip_file.extractall(target)


async def _execute_process(name: str, cmd: list, cwd: Path, cb: Callback):
    PIPE = asyncio.subprocess.PIPE
    cmd = [str(c) for c in cmd]
    cb(InstallationProgress(f"Installing {name}", f"Executing {' '.join(cmd)}"))
    process = await asyncio.create_subprocess_exec(
        cmd[0], *cmd[1:], cwd=cwd, stdout=PIPE, stderr=PIPE
    )
    async for line in process.stdout:
        cb(InstallationProgress(f"Installing {name}", line.decode().strip()))
    if process.returncode != 0:
        err = (await process.stderr.read()).decode()
        raise Exception(f"Error during PyTorch installation: {err}")


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
