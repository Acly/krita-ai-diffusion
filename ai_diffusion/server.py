import asyncio
import locale
from enum import Enum
from itertools import chain
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, List, NamedTuple, Optional, Sequence
from zipfile import ZipFile
from PyQt5.QtNetwork import QNetworkAccessManager

from .settings import settings, ServerBackend
from .style import SDVersion
from .network import download, DownloadProgress
from .util import client_logger as log, server_logger as server_log


class CustomNode(NamedTuple):
    name: str
    folder: str
    url: str
    nodes: Sequence[str]


class ResourceKind(Enum):
    checkpoint = "Stable Diffusion Checkpoint"
    controlnet = "ControlNet model"
    clip_vision = "CLIP Vision model"
    ip_adapter = "IP-Adapter model"
    upscaler = "Upscale model"
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


class ControlType(Enum):
    inpaint = 0
    scribble = 1
    line_art = 2
    soft_edge = 3
    canny_edge = 4
    depth = 5
    normal = 6
    pose = 7
    segmentation = 8

    @property
    def is_lines(self):
        return self in [
            ControlType.scribble,
            ControlType.line_art,
            ControlType.soft_edge,
            ControlType.canny_edge,
        ]


control_filename = {
    ControlType.inpaint: {
        SDVersion.sd1_5: "control_v11p_sd15_inpaint",
        SDVersion.sdxl: None,
    },
    ControlType.scribble: {
        SDVersion.sd1_5: "control_v11p_sd15_scribble",
        SDVersion.sdxl: None,
    },
    ControlType.line_art: {
        SDVersion.sd1_5: "control_v11p_sd15_lineart",
        SDVersion.sdxl: "control-lora-sketch-rank256",
    },
    ControlType.soft_edge: {
        SDVersion.sd1_5: "control_v11p_sd15_softedge",
        SDVersion.sdxl: None,
    },
    ControlType.canny_edge: {
        SDVersion.sd1_5: "control_v11p_sd15_canny",
        SDVersion.sdxl: "control-lora-canny-rank256",
    },
    ControlType.depth: {
        SDVersion.sd1_5: "control_lora_rank128_v11f1p_sd15_depth",
        SDVersion.sdxl: "control-lora-depth-rank256",
    },
    ControlType.normal: {
        SDVersion.sd1_5: "control_lora_rank128_v11p_sd15_normalbae",
        SDVersion.sdxl: None,
    },
    ControlType.pose: {
        SDVersion.sd1_5: "control_v11p_sd15_openpose",
        SDVersion.sdxl: "control-lora-openposeXL2-rank256",
    },
    ControlType.segmentation: {
        SDVersion.sd1_5: "control_lora_rank128_v11p_sd15_seg",
        SDVersion.sdxl: None,
    },
}

required_models = [
    ModelResource(
        "ControlNet Inpaint",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_v11p_sd15_inpaint_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors",
    ),
    ModelResource(
        "CLIP Vision model",
        ResourceKind.clip_vision,
        Path("models/clip_vision/SD1.5"),
        "pytorch_model.bin",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin",
    ),
    ModelResource(
        "IP-Adapter model",
        ResourceKind.ip_adapter,
        Path("custom_nodes/IPAdapter-ComfyUI/models"),
        "ip-adapter_sd15.bin",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin",
    ),
    ModelResource(
        "NMKD Superscale model",
        ResourceKind.upscaler,
        Path("models/upscale_models"),
        "4x_NMKD-Superscale-SP_178000_G.pth",
        "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth",
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
        "dreamshaper_8.safetensors",
        "https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    ),
]

optional_models = [
    ModelResource(
        "ControlNet Scribble",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_v11p_sd15_scribble_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_scribble_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Line Art",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_v11p_sd15_lineart_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Soft Edge",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_v11p_sd15_softedge_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Canny Edge",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_v11p_sd15_canny_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Depth",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Normal",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Pose",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_v11p_sd15_openpose_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Segmentation",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_lora_rank128_v11p_sd15_seg_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_seg_fp16.safetensors",
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


_all_resources = (
    [n.name for n in required_custom_nodes]
    + [m.name for m in required_models]
    + [c.name for c in default_checkpoints]
    + [m.name for m in optional_models]
)

_is_windows = "win" in sys.platform
_exe = ".exe" if _is_windows else ""
_process_flags = subprocess.CREATE_NO_WINDOW if _is_windows else 0


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
InternalCB = Callable[[str, str, Optional[DownloadProgress]], None]


class Server:
    path: Path
    url: Optional[str] = None
    backend = ServerBackend.cuda
    state = ServerState.stopped
    missing_resources: List[str]

    _python_cmd: Optional[str] = None
    _pip_cmd: Optional[str] = None
    _comfy_dir: Optional[Path] = None
    _cache_dir: Optional[Path] = None
    _process: Optional[asyncio.subprocess.Process] = None
    _task: Optional[asyncio.Task] = None

    def __init__(self, path: str = None):
        self.path = Path(path or settings.server_path)
        if not self.path.is_absolute():
            self.path = Path(__file__).parent / self.path
        self.backend = settings.server_backend
        self.check_install()

    def check_install(self):
        self.missing_resources = []
        self._cache_dir = self.path / ".cache"

        comfy_pkg = ["main.py", "nodes.py", "custom_nodes"]
        self._comfy_dir = _find_component(comfy_pkg, [self.path, self.path / "ComfyUI"])

        python_pkg = ["python3.dll", "python.exe"] if _is_windows else ["python3", "pip3"]
        python_search_paths = [
            self.path / "python",
            self.path / "venv" / "bin",
            self.path / ".venv" / "bin",
        ]
        if self._comfy_dir:
            python_search_paths += [
                self._comfy_dir / "python",
                self._comfy_dir / "venv" / "bin",
                self._comfy_dir / ".venv" / "bin",
                self._comfy_dir.parent / "python",
                self._comfy_dir.parent / "python_embeded",
            ]
        python_path = _find_component(python_pkg, python_search_paths)
        if python_path is None:
            self._python_cmd = _find_program("python", "python3")
            self._pip_cmd = _find_program("pip", "pip3")
        else:
            self._python_cmd = python_path / f"python{_exe}"
            self._pip_cmd = python_path / "pip"
            if _is_windows:
                self._pip_cmd = python_path / "Scripts" / "pip.exe"

        if not (self.has_comfy and self.has_python):
            self.state = ServerState.not_installed
            self.missing_resources = _all_resources
            return

        missing_nodes = [
            package.name
            for package in required_custom_nodes
            if not Path(self._comfy_dir / "custom_nodes" / package.folder).exists()
        ]
        self.missing_resources += missing_nodes

        def find_missing(folder: Path, resources: List[ModelResource]):
            return [
                resource.name
                for resource in resources
                if not (folder / resource.folder / resource.filename).exists()
            ]

        self.missing_resources += find_missing(self._comfy_dir, required_models)
        if len(self.missing_resources) > 0:
            self.state = ServerState.missing_resources
        else:
            self.state = ServerState.stopped

        # Optional resources
        self.missing_resources += find_missing(self._comfy_dir, default_checkpoints)
        self.missing_resources += find_missing(self._comfy_dir, optional_models)

    async def _install(self, cb: InternalCB):
        self.state = ServerState.installing
        cb("Installing", f"Installation started in {self.path}")

        network = QNetworkAccessManager()
        self._cache_dir = self.path / ".cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        no_python = self._python_cmd is None or self._pip_cmd is None
        if _is_windows and (self._comfy_dir is None or no_python):
            # On Windows install an embedded version of Python
            python_dir = self.path / "python"
            self._python_cmd = python_dir / f"python{_exe}"
            self._pip_cmd = python_dir / "Scripts" / f"pip{_exe}"
            await _install_if_missing(python_dir, self._install_python, network, cb)
        elif not _is_windows and (self._comfy_dir is None or self._pip_cmd is None):
            # On Linux a system Python is required to create a virtual environment
            python_dir = self.path / "venv"
            await _install_if_missing(python_dir, self._create_venv, cb)
            self._python_cmd = python_dir / "bin" / "python3"
            self._pip_cmd = python_dir / "bin" / "pip3"

        self._comfy_dir = self._comfy_dir or self.path / "ComfyUI"
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
        cb("Finished", f"Installation finished in {self.path}")
        self.check_install()

    async def _install_python(self, network: QNetworkAccessManager, cb: InternalCB):
        url = "https://www.python.org/ftp/python/3.10.9/python-3.10.9-embed-amd64.zip"
        archive_path = self._cache_dir / "python-3.10.9-embed-amd64.zip"
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
        url = "https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip"
        archive_path = self._cache_dir / "ComfyUI.zip"
        await _download_cached("ComfyUI", network, url, archive_path, cb)
        await _extract_archive("ComfyUI", archive_path, self._comfy_dir.parent, cb)
        _rename_extracted_folder("ComfyUI", self._comfy_dir, "-master")

        torch_args = ["install", "torch", "torchvision", "torchaudio", "--index-url"]
        torch_index = {
            ServerBackend.cuda: "https://download.pytorch.org/whl/cu118",
            ServerBackend.cpu: "https://download.pytorch.org/whl/cpu",
        }
        torch_cmd = [self._pip_cmd, *torch_args, torch_index[self.backend]]
        await _execute_process("PyTorch", torch_cmd, self._comfy_dir, cb)

        requirements_txt = self._comfy_dir / "requirements.txt"
        requirements_cmd = [self._pip_cmd, "install", "-r", requirements_txt]
        await _execute_process("ComfyUI", requirements_cmd, self._comfy_dir, cb)
        cb("Installing ComfyUI", "Finished installing ComfyUI")

    async def _install_custom_node(
        self, pkg: CustomNode, network: QNetworkAccessManager, cb: InternalCB
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
        cb(f"Installing {pkg.name}", f"Finished installing {pkg.name}")

    async def install(self, callback: Callback):
        assert self.state in [ServerState.not_installed, ServerState.missing_resources]
        if not _is_windows and self._python_cmd is None:
            raise Exception(
                "Python not found. Please install python3, python3-venv via your package manager"
                " and restart."
            )

        def cb(stage: str, message: str, progress: Optional[DownloadProgress] = None):
            if message:
                log.info(message)
            filters = ["Downloading", "Installing", "Collecting", "Using"]
            if not (message and any(s in message[:16] for s in filters)):
                message = ""
            callback(InstallationProgress(stage, progress, message))

        try:
            await self._install(cb)
        except Exception as e:
            log.exception(str(e))
            log.error("Installation failed")
            self.state = ServerState.stopped
            self.check_install()
            raise e

    async def install_optional(self, packages: List[str], callback: Callback):
        assert self.has_comfy, "Must install ComfyUI before downloading checkpoints"
        network = QNetworkAccessManager()
        prev_state = self.state
        self.state = ServerState.installing

        def cb(stage: str, message: str, progress: Optional[DownloadProgress] = None):
            if message:
                log.info(message)
            callback(InstallationProgress(stage, progress))

        try:
            all_optional = chain(default_checkpoints, optional_models)
            to_install = (r for r in all_optional if r.name in packages)
            for resource in to_install:
                target_file = self._comfy_dir / resource.folder / resource.filename
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

        self.state = ServerState.starting
        args = ["-u", "-X", "utf8", "main.py"]
        if self.backend is ServerBackend.cpu:
            args.append("--cpu")
        self._process = await asyncio.create_subprocess_exec(
            self._python_cmd,
            *args,
            cwd=self._comfy_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=_process_flags,
        )

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
        return self.url

    async def run(self):
        assert self.state is ServerState.running

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
        return self._comfy_dir is not None


def _find_component(files: List[str], search_paths: List[Path]):
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
    name: str,
    network: QNetworkAccessManager,
    url: str,
    archive: Path,
    cb: InternalCB,
):
    if not archive.exists():
        cb(f"Downloading {name}", f"Downloading {url} to {archive}")
        async for progress in download(network, url, archive):
            cb(f"Downloading {name}", None, progress)


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
