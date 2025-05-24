from __future__ import annotations
import asyncio
from enum import Enum
from itertools import chain
from pathlib import Path
import shutil
import re
import os
import time
from typing import Callable, NamedTuple, Optional, Union
from PyQt5.QtNetwork import QNetworkAccessManager

from .settings import settings, ServerBackend
from . import eventloop, resources
from .resources import CustomNode, ModelResource, ModelRequirements, Arch
from .resources import VerificationStatus, VerificationState
from .network import download, DownloadProgress
from .localization import translate as _
from .util import ZipFile, is_windows, create_process, decode_pipe_bytes, determine_system_encoding
from .util import client_logger as log, server_logger as server_log


_exe = ".exe" if is_windows else ""


class ServerState(Enum):
    not_installed = 0
    missing_resources = 2
    installing = 3
    stopped = 4
    starting = 5
    running = 6
    verifying = 7
    uninstalling = 8


class InstallationProgress(NamedTuple):
    stage: str
    progress: DownloadProgress | tuple[int, int] | None = None
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

    _uv_cmd: Optional[Path] = None
    _python_cmd: Optional[Path] = None
    _cache_dir: Path
    _version_file: Path
    _process: Optional[asyncio.subprocess.Process] = None
    _task: Optional[asyncio.Task] = None
    _installed_backend: Optional[ServerBackend] = None

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
        self.version = None
        if self._version_file.exists():
            content = self._version_file.read_text().strip().split()
            if len(content) > 0:
                self.version = content[0]
            if len(content) > 1 and content[1] in ServerBackend.__members__:
                self._installed_backend = ServerBackend[content[1]]
        if self.version is not None:
            backend = f" [{self._installed_backend.name}]" if self._installed_backend else ""
            log.info(f"Found server installation v{self.version}{backend} at {self.path}")

        comfy_pkg = ["main.py", "nodes.py", "custom_nodes"]
        self.comfy_dir = _find_component(comfy_pkg, [self.path / "ComfyUI"])

        uv_exe = "uv.exe" if is_windows else "uv"
        self._uv_cmd = self.path / "uv" / uv_exe
        if not self._uv_cmd.exists():
            self._uv_cmd = None
            self._uv_cmd = _find_program(uv_exe)

        python_pkg = ["python.exe"] if is_windows else ["python3"]
        python_search_paths = [
            self.path / "python",
            self.path / "venv" / "bin",
            self.path / "venv" / "Scripts",
        ]
        python_path = _find_component(python_pkg, python_search_paths)
        if python_path is None:
            if not is_windows:
                self._python_cmd = _find_program(
                    "python3.12", "python3.11", "python3.10", "python3", "python"
                )
        else:
            self._python_cmd = python_path / f"python{_exe}"

        if not (self.has_comfy and self.has_python):
            self.state = ServerState.not_installed
            self.missing_resources = resources.all_resources
            return

        eventloop.run(determine_system_encoding(str(self._python_cmd)))

        assert self.comfy_dir is not None
        missing_nodes = [
            package.name
            for package in resources.required_custom_nodes
            if not Path(self.comfy_dir / "custom_nodes" / package.folder).exists()
        ]
        self.missing_resources += missing_nodes

        model_folders = [self.path, self.comfy_dir]
        self.missing_resources += find_missing(model_folders, resources.required_models, Arch.all)
        missing_sd15 = find_missing(model_folders, resources.required_models, Arch.sd15)
        missing_sdxl = find_missing(model_folders, resources.required_models, Arch.sdxl)
        if len(self.missing_resources) > 0 or (len(missing_sd15) > 0 and len(missing_sdxl) > 0):
            self.state = ServerState.missing_resources
        else:
            self.state = ServerState.stopped
        self.missing_resources += missing_sd15 + missing_sdxl

        # Optional resources
        self.missing_resources += find_missing(model_folders, resources.default_checkpoints)
        self.missing_resources += find_missing(model_folders, resources.upscale_models)
        self.missing_resources += find_missing(model_folders, resources.optional_models)

    async def _install(self, cb: InternalCB):
        self.state = ServerState.installing
        cb("Installing", f"Installation started in {self.path}")

        network = QNetworkAccessManager()
        self._cache_dir = self.path / ".cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._version_file.write_text("incomplete")

        has_venv = (self.path / "venv").exists()
        has_embedded_python = (self.path / "python").exists()
        has_uv = self._uv_cmd is not None
        if not any((has_venv, has_embedded_python, has_uv)):
            await try_install(self.path / "uv", self._install_uv, network, cb)

        if self.comfy_dir is None or not (has_venv or has_embedded_python):
            python_dir = self.path / "venv"
            await install_if_missing(python_dir, self._create_venv, cb)
        assert self._python_cmd is not None
        await self._log_python_version()
        await determine_system_encoding(str(self._python_cmd))

        comfy_dir = self.comfy_dir or self.path / "ComfyUI"
        if not self.has_comfy:
            await try_install(comfy_dir, self._install_comfy, comfy_dir, network, cb)

        for pkg in chain(resources.required_custom_nodes, resources.optional_custom_nodes):
            dir = comfy_dir / "custom_nodes" / pkg.folder
            await install_if_missing(dir, self._install_custom_node, pkg, network, cb)

        self._version_file.write_text(f"{resources.version} {self.backend.name}")
        self.state = ServerState.stopped
        cb("Finished", f"Installation finished in {self.path}")
        self.check_install()

    async def _log_python_version(self):
        if self._uv_cmd is not None:
            uv_ver = await get_python_version_string(self._uv_cmd)
            log.info(f"Using uv: {uv_ver}")
        if self._python_cmd is not None:
            python_ver = await get_python_version_string(self._python_cmd)
            log.info(f"Using Python: {python_ver}, {self._python_cmd}")
            if self._uv_cmd is None:
                pip_ver = await get_python_version_string(self._python_cmd, "-m", "pip")
                log.info(f"Using pip: {pip_ver}")

    def _pip_install(self, name: str, args: list[str], cb: InternalCB):
        env = None
        if self._uv_cmd is not None:
            env = {"VIRTUAL_ENV": str(self.path / "venv")}
            cmd = [self._uv_cmd, "pip", "install", *args]
        else:
            cmd = [self._python_cmd, "-su", "-m", "pip", "install", *args]
        return _execute_process(name, cmd, self.path, cb, env=env)

    async def _install_uv(self, network: QNetworkAccessManager, cb: InternalCB):
        script_ext = ".ps1" if is_windows else ".sh"
        url = f"https://astral.sh/uv/0.6.10/install{script_ext}"
        script_path = self._cache_dir / f"install_uv{script_ext}"
        await _download_cached("Python", network, url, script_path, cb)

        env = {"UV_INSTALL_DIR": str(self.path / "uv")}
        if is_windows:
            if "PSModulePath" in os.environ:
                del os.environ["PSModulePath"]  # Don't inherit this from parent process
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
            try:
                await _execute_process("Python", cmd, self.path, cb, env=env)
            except FileNotFoundError:
                sysroot = os.environ.get("SYSTEMROOT", "C:\\Windows")
                cmd[0] = f"{sysroot}\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
                log.warning(f"powershell command not found, trying to find it at {cmd[0]}")
                await _execute_process("Python", cmd, self.path, cb, env=env)
        else:
            cmd = ["/bin/sh", str(script_path)]
            await _execute_process("Python", cmd, self.path, cb, env=env)

        self._uv_cmd = self.path / "uv" / ("uv" + _exe)
        cb("Installing Python", f"Installed uv at {self._uv_cmd}")

    async def _create_venv(self, cb: InternalCB):
        cb("Creating Python virtual environment", f"Creating venv in {self.path / 'venv'}")
        assert self._uv_cmd is not None
        venv_cmd = [self._uv_cmd, "venv", "--python", "3.12", str(self.path / "venv")]
        await _execute_process("Python", venv_cmd, self.path, cb)

        if is_windows:
            self._python_cmd = self.path / "venv" / "Scripts" / "python.exe"
        else:
            self._python_cmd = self.path / "venv" / "bin" / "python3"

    async def _install_comfy(self, comfy_dir: Path, network: QNetworkAccessManager, cb: InternalCB):
        url = f"{resources.comfy_url}/archive/{resources.comfy_version}.zip"
        archive_path = self._cache_dir / f"ComfyUI-{resources.comfy_version}.zip"
        await _download_cached("ComfyUI", network, url, archive_path, cb)
        await _extract_archive("ComfyUI", archive_path, comfy_dir.parent, cb)
        temp_comfy_dir = comfy_dir.parent / f"ComfyUI-{resources.comfy_version}"

        torch_args = ["torch~=2.7.0", "torchvision~=0.22.0", "torchaudio~=2.7.0"]
        if self.backend is ServerBackend.cpu:
            torch_args += ["--index-url", "https://download.pytorch.org/whl/cpu"]
        elif self.backend is ServerBackend.cuda:
            torch_args += ["--index-url", "https://download.pytorch.org/whl/cu128"]
        elif self.backend is ServerBackend.directml:
            torch_args = ["numpy<2", "torch-directml"]
        elif self.backend is ServerBackend.xpu:
            torch_args = ["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"]
            torch_args += ["--index-url", "https://download.pytorch.org/whl/xpu"]
        await self._pip_install("PyTorch", torch_args, cb)

        requirements_txt = Path(__file__).parent / "server_requirements.txt"
        await self._pip_install("ComfyUI", ["-r", str(requirements_txt)], cb)

        if self.backend is ServerBackend.xpu:
            idx_url = "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
            cmd = ["intel-extension-for-pytorch==2.6.10+xpu", "--extra-index-url", idx_url]
            await self._pip_install("Ipex", cmd, cb)

        requirements_txt = temp_comfy_dir / "requirements.txt"
        await self._pip_install("ComfyUI", ["-r", str(requirements_txt)], cb)

        _configure_extra_model_paths(temp_comfy_dir)
        await rename_extracted_folder("ComfyUI", comfy_dir, resources.comfy_version)
        self.comfy_dir = comfy_dir
        cb("Installing ComfyUI", "Finished installing ComfyUI")

    async def _install_custom_node(
        self, pkg: CustomNode, network: QNetworkAccessManager, cb: InternalCB
    ):
        assert self.comfy_dir is not None
        folder = self.comfy_dir / "custom_nodes" / pkg.folder
        resource_url = pkg.url
        if not resource_url.endswith(".zip"):  # git repo URL
            resource_url = f"{pkg.url}/archive/{pkg.version}.zip"
        resource_zip_path = self._cache_dir / f"{pkg.folder}-{pkg.version}.zip"
        await _download_cached(pkg.name, network, resource_url, resource_zip_path, cb)
        await _extract_archive(pkg.name, resource_zip_path, folder.parent, cb)
        await rename_extracted_folder(pkg.name, folder, pkg.version)
        cb(f"Installing {pkg.name}", f"Finished installing {pkg.name}")

    async def _install_insightface(self, network: QNetworkAccessManager, cb: InternalCB):
        assert self.comfy_dir is not None and self._python_cmd is not None

        dependencies = ["onnx==1.16.1", "onnxruntime"]  # onnx version pinned due to #1033
        await self._pip_install("FaceID", dependencies, cb)

        pyver = await get_python_version_string(self._python_cmd)
        if is_windows and ("3.11" in pyver or "3.12" in pyver):
            whl_file = self._cache_dir / "insightface-0.7.3-cp311-cp311-win_amd64.whl"
            whl_url = "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl"
            if "3.12" in pyver:
                whl_file = self._cache_dir / "insightface-0.7.3-cp312-cp312-win_amd64.whl"
                whl_url = "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl"
            await _download_cached("FaceID", network, whl_url, whl_file, cb)
            await self._pip_install("FaceID", [str(whl_file)], cb)
        else:
            await self._pip_install("FaceID", ["insightface"], cb)

    async def _install_requirements(
        self, requirements: ModelRequirements, network: QNetworkAccessManager, cb: InternalCB
    ):
        if requirements is ModelRequirements.insightface:
            await self._install_insightface(network, cb)

    async def install(self, callback: Callback):
        assert self.state in [ServerState.not_installed, ServerState.missing_resources] or (
            self.state is ServerState.stopped and self.upgrade_required
        )

        if not is_windows and self._python_cmd is None:
            raise Exception(
                _(
                    "Python not found. Please install python3, python3-venv via your package manager and restart."
                )
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
            raise Exception(parse_common_errors(str(e)))

    async def download_required(self, callback: Callback):
        models = [m.name for m in resources.required_models if m.arch is Arch.all]
        await self.download(models, callback)

    async def download(self, packages: list[str], callback: Callback):
        assert self.comfy_dir, "Must install ComfyUI before downloading models"
        network = QNetworkAccessManager()
        prev_state = self.state
        self.state = ServerState.installing

        def cb(stage: str, message: str | DownloadProgress):
            if isinstance(message, str):
                log.info(message)
            progress = message if isinstance(message, DownloadProgress) else None
            callback(InstallationProgress(stage, progress))

        try:
            all_models = chain(
                resources.required_models,
                resources.default_checkpoints,
                resources.upscale_models,
                resources.optional_models,
            )
            to_install = (r for r in all_models if r.name in packages)
            for resource in to_install:
                if not resource.exists_in(self.path) and not resource.exists_in(self.comfy_dir):
                    await self._install_requirements(resource.requirements, network, cb)
                    for file in resource.files:
                        target_file = self.path / file.path
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        await _download_cached(resource.name, network, file.url, target_file, cb)
        except Exception as e:
            log.exception(str(e))
            raise e
        finally:
            self.state = prev_state
            self.check_install()

    async def upgrade(self, callback: Callback):
        assert self.upgrade_required and self.comfy_dir is not None

        def info(message: str):
            log.info(message)
            callback(InstallationProgress("Upgrading", message=message))

        info(f"Starting upgrade from {self.version} to {resources.version}")
        comfy_dir = self.comfy_dir
        upgrade_dir = self.path / f"upgrade-{resources.version}"
        upgrade_comfy_dir = upgrade_dir / "ComfyUI"
        keep_paths = [
            Path("models"),
            Path("custom_nodes", "comfyui_controlnet_aux", "ckpts"),
        ]
        info(f"Backing up {comfy_dir} to {upgrade_comfy_dir}")
        if upgrade_comfy_dir.exists():
            raise Exception(
                _(
                    "Backup folder {dir} already exists! Please make sure it does not contain any valuable data such as checkpoints or other models you downloaded. Then delete the folder and try again.",
                    dir=upgrade_comfy_dir,
                )
            )
        upgrade_comfy_dir.parent.mkdir(exist_ok=True)
        shutil.move(comfy_dir, upgrade_comfy_dir)
        self.comfy_dir = None
        try:
            await self.install(callback)
        except Exception as e:
            if upgrade_comfy_dir.exists():
                log.warning(f"Error during upgrade: {str(e)} - Restoring {upgrade_comfy_dir}")
                safe_remove_dir(comfy_dir)
                shutil.move(upgrade_comfy_dir, comfy_dir)
            raise e

        try:
            _upgrade_models_dir(upgrade_comfy_dir / "models", self.path / "models")
            for path in keep_paths:
                src = upgrade_comfy_dir / path
                dst = comfy_dir / path
                if src.exists():
                    info(f"Migrating {dst}")
                    safe_remove_dir(dst)  # Remove placeholder
                    shutil.move(src, dst)
            _upgrade_extra_model_paths(upgrade_comfy_dir, comfy_dir)
            self.check_install()

            # Clean up temporary directory
            safe_remove_dir(upgrade_dir)
            info(message=f"Finished upgrade to {resources.version}")
        except Exception as e:
            log.error(f"Error during upgrade: {str(e)}")
            raise Exception(
                _("Error during model migration")
                + f": {str(e)}\n"
                + _("Some models remain in")
                + f" {upgrade_comfy_dir}"
            )

    async def start(self, port: int | None = None):
        assert self.state in [ServerState.stopped, ServerState.missing_resources]
        assert self._python_cmd
        await self._log_python_version()

        self.state = ServerState.starting
        last_line = ""
        try:
            args = ["-su", "main.py"]
            env = {}
            if self.backend is ServerBackend.cpu:
                args.append("--cpu")
            elif self.backend is ServerBackend.directml:
                args.append("--directml")
            if settings.server_arguments:
                args += settings.server_arguments.split(" ")
            if port is not None:
                args += ["--port", str(port)]
            if self.backend is not ServerBackend.cpu:
                env["ONEDNN_MAX_CPU_ISA"] = "AVX2"  # workaround for #401

            log.info(f"Starting server with python {' '.join(args)}")
            self._process = await create_process(
                self._python_cmd, *args, cwd=self.comfy_dir, additional_env=env
            )

            assert self._process.stdout is not None
            async for line in self._process.stdout:
                text = decode_pipe_bytes(line).strip()
                last_line = text
                server_log.info(text)
                if text.startswith("To see the GUI go to:"):
                    self.state = ServerState.running
                    self.url = text.split("http://")[-1]
                    break
        except Exception as e:
            log.exception(f"Error during server start: {str(e)}")
            if self._process is None:
                self.state = ServerState.stopped
                raise e

        if self.state != ServerState.running:
            error = "Process exited unexpectedly"
            try:
                out, err = await asyncio.wait_for(self._process.communicate(), timeout=10)
                server_log.error(decode_pipe_bytes(out).strip())
                error = last_line + decode_pipe_bytes(err or out)
            except asyncio.TimeoutError:
                self._process.kill()
            except Exception as e:
                log.exception(f"Error while waiting for process: {str(e)}")
                error = str(e)

            self.state = ServerState.stopped
            ret = self._process.returncode
            self._process = None
            error_msg = parse_common_errors(error, ret)
            raise Exception(_("Error during server startup") + f": {error_msg}")

        self._task = asyncio.create_task(self.run())
        assert self.url is not None
        return self.url

    async def run(self):
        assert self.state is ServerState.running
        assert self._process and self._process.stdout

        try:
            async for line in self._process.stdout:
                server_log.info(decode_pipe_bytes(line).strip())

            code = await asyncio.wait_for(self._process.wait(), timeout=1)
            if code != 0:
                log.error(f"Server process terminated with code {self._process.returncode}")
            elif code is not None:
                log.info("Server process was shut down sucessfully")

        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            log.warning("Server process did not terminate after the pipe was closed")

        self.state = ServerState.stopped
        self._process = None

    async def stop(self):
        assert self.state is ServerState.running
        try:
            if self._process and self._task:
                log.info("Stopping server")
                self._process.terminate()
                await asyncio.wait_for(self._task, timeout=5)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            log.warning("Server did not terminate in time")

    def terminate(self):
        try:
            if self._process is not None:
                self._process.terminate()
        except Exception as e:
            print(e)
            pass

    async def verify(self, callback: Callback):
        assert self.state in [ServerState.stopped, ServerState.missing_resources]

        self.state = ServerState.verifying
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.to_thread(self._verify, callback, loop)
            for status in result:
                log.warning(f"File verification failed for {status.file.path}: {status.state.name}")
                if status.state is VerificationState.mismatch:
                    log.info(f"-- expected sha256: {status.file.sha256} but got: {status.info}")
            return result
        except Exception as e:
            log.exception(f"Error during server verification: {str(e)}")
            raise e
        finally:
            self.state = ServerState.stopped

    def _verify(self, callback: Callback, loop: asyncio.AbstractEventLoop):
        assert self.comfy_dir is not None
        errors: list[VerificationStatus] = []
        total = sum(
            len(m.files)
            for m in resources.all_models()
            if m.exists_in(self.path) or m.exists_in(self.comfy_dir)
        )
        verified = 0

        for status in resources.verify_model_integrity(self.path):
            if status.state is VerificationState.in_progress:
                loop.call_soon_threadsafe(
                    callback,
                    InstallationProgress(
                        "Verifying model files",
                        message=status.file.name,
                        progress=(verified, total),
                    ),
                )
                verified += 1
            elif status.state in [VerificationState.mismatch, VerificationState.error]:
                errors.append(status)

        loop.call_soon_threadsafe(
            callback,
            InstallationProgress("Verification finished", message="", progress=(total, total)),
        )
        return errors

    async def fix_models(self, bad_models: list[VerificationStatus], callback: Callback):
        network = QNetworkAccessManager()
        prev_state = self.state
        self.state = ServerState.installing

        def cb(stage: str, message: str | DownloadProgress):
            if isinstance(message, str):
                log.info(message)
            progress = message if isinstance(message, DownloadProgress) else None
            callback(InstallationProgress(stage, progress))

        cb("Replacing files", "Trying to remove and re-download corrupted model files")
        try:
            for status in bad_models:
                if status.state is VerificationState.mismatch:
                    filepath = self.path / status.file.path
                    log.info(f"Removing {filepath}")
                    remove_file(filepath)
                    await _download_cached(status.file.name, network, status.file.url, filepath, cb)
        except Exception as e:
            log.exception(str(e))
            raise e
        finally:
            self.state = prev_state
            self.check_install()

    async def uninstall(self, callback: Callback, delete_models=False):
        log.info(f"Uninstalling server at {self.path}")
        assert self.state in [ServerState.stopped, ServerState.missing_resources]

        self.state = ServerState.uninstalling
        try:
            loop = asyncio.get_running_loop()
            await asyncio.to_thread(self._uninstall, callback, delete_models, loop)
        except Exception as e:
            log.exception(f"Error during server uninstall: {str(e)}")
            raise e
        finally:
            self.state = ServerState.stopped
            self.check_install()

    def _uninstall(self, callback: Callback, delete_models: bool, loop: asyncio.AbstractEventLoop):
        assert self.state is ServerState.uninstalling

        def cb(msg: str):
            loop.call_soon_threadsafe(callback, InstallationProgress("Uninstalling", message=msg))

        try:
            if self.comfy_dir and self.comfy_dir.exists():
                cb("Removing ComfyUI")
                if not delete_models:
                    try:
                        safe_remove_dir(self.comfy_dir / "models")
                    except Exception as e:
                        raise Exception(
                            str(e)
                            + f"\n\nPlease move model files located in\n{self.comfy_dir / 'models'}"
                            + f"\nto\n{self.path / 'models'}\nand try again."
                        )
                remove_subdir(self.comfy_dir, origin=self.path)

            venv_dir = self.path / "venv"
            if venv_dir.exists():
                cb("Removing Python venv")
                remove_subdir(venv_dir, origin=self.path)

            if self._cache_dir.exists():
                cb("Removing cache")
                remove_subdir(self._cache_dir, origin=self.path)

            uv_dir = self.path / "uv"
            if uv_dir.exists():
                cb("Removing uv")
                remove_subdir(uv_dir, origin=self.path)

            self._version_file.write_text("incomplete")

            if delete_models:
                model_dir = self.path / "models"
                if model_dir.exists():
                    cb("Removing models")
                    remove_subdir(model_dir, origin=self.path)

                remove_subdir(self.path, origin=self.path)

            cb("Finished uninstalling")
        except Exception as e:
            log.exception(f"Error during server uninstall: {str(e)}")
            raise e

    @property
    def has_python(self):
        return self._python_cmd is not None

    @property
    def has_comfy(self):
        return self.comfy_dir is not None

    def is_installed(self, package: str | ModelResource | CustomNode):
        name = package if isinstance(package, str) else package.name
        return name not in self.missing_resources

    def all_installed(self, packages: list[str] | list[ModelResource] | list[CustomNode]):
        return all(self.is_installed(p) for p in packages)

    @property
    def can_install(self):
        if not self.path.exists():
            return True
        if self.path.is_dir():
            return self.version == "incomplete" or not any(self.path.iterdir())
        return False

    @property
    def upgrade_required(self):
        gpu_backends = [ServerBackend.cuda, ServerBackend.directml, ServerBackend.xpu]
        backend_mismatch = (
            self._installed_backend is not None
            and self._installed_backend != self.backend
            and self.backend in gpu_backends
            and self._installed_backend in (gpu_backends + [ServerBackend.cpu])
        )
        return (
            self.state is not ServerState.not_installed
            and self.version is not None
            and self.version != "incomplete"
            and (self.version != resources.version or backend_mismatch)
        )


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


async def _execute_process(
    name: str, cmd: list, cwd: Path, cb: InternalCB, env: dict | None = None
):
    errlog = ""

    cmd = [str(c) for c in cmd]
    cb(f"Installing {name}", f"Executing {' '.join(cmd)}")
    process = await create_process(cmd[0], *cmd[1:], cwd=cwd, additional_env=env, pipe_stderr=True)

    async def forward(stream: asyncio.StreamReader):
        async for line in stream:
            cb(f"Installing {name}", decode_pipe_bytes(line).strip())

    async def collect(stream: asyncio.StreamReader):
        nonlocal errlog
        async for line in stream:
            errlog += decode_pipe_bytes(line)

    assert process.stdout and process.stderr
    await asyncio.gather(forward(process.stdout), collect(process.stderr))

    if process.returncode != 0:
        if errlog == "":
            errlog = f"Process exited with code {process.returncode}"
        raise Exception(_("Error during installation") + f": {errlog}")


async def try_install(path: Path, installer, *args):
    already_exists = path.exists()
    try:
        await installer(*args)
    except Exception as e:
        # Revert installation so it may be attempted again
        if not already_exists:
            shutil.rmtree(path, ignore_errors=True)
        raise e


async def install_if_missing(path: Path, installer, *args):
    if not path.exists():
        await try_install(path, installer, *args)


def find_missing(folders: list[Path], resources: list[ModelResource], ver: Arch | None = None):
    return [
        res.name
        for res in resources
        if (not ver or res.arch is ver) and not any(res.exists_in(f) for f in folders)
    ]


async def rename_extracted_folder(name: str, path: Path, suffix: str):
    if path.exists() and path.is_dir() and not any(path.iterdir()):
        path.rmdir()
    elif path.exists():
        raise Exception(f"Error during {name} installation: target folder {path} already exists")

    extracted_folder = path.parent / f"{path.name}-{suffix}"
    if not extracted_folder.exists():
        raise Exception(
            f"Error during {name} installation: folder {extracted_folder} does not exist"
        )
    for tries in range(3):  # Because Windows, or virus scanners, or something #515
        try:
            extracted_folder.rename(path)
            return
        except Exception as e:
            log.warning(f"Rename failed during {name} installation: {str(e)} - retrying...")
            await asyncio.sleep(1)
    extracted_folder.rename(path)


def safe_remove_dir(path: Path, max_size=12 * 1024 * 1024):
    if path.is_dir():
        for p in path.rglob("*"):
            if p.is_file():
                if p.stat().st_size > max_size:
                    raise Exception(
                        f"Failed to remove {path}: found remaining large file {p.relative_to(path)}"
                    )
                if p.suffix == ".safetensors":
                    raise Exception(
                        f"Failed to remove {path}: found remaining model {p.relative_to(path)}"
                    )
        shutil.rmtree(path, ignore_errors=True)


def remove_subdir(path: Path, *, origin: Path):
    assert path.is_dir() and path.is_relative_to(origin)
    errors = []

    def handle_error(func, path, excinfo):
        type, value, traceback = excinfo
        if type is FileNotFoundError:
            return
        log.warning(f"Failed to remove {path}: [{type}] {value}")
        errors.append(value)

    for i in range(3):
        shutil.rmtree(path, onerror=handle_error)
        if len(errors) == 0:
            return
        elif i == 2:
            raise errors[0]
        time.sleep(0.1)
        errors.clear()


def remove_file(path: Path):
    if path.is_file():
        try:
            path.unlink()
            for i in range(3):
                if not path.exists():
                    return
                time.sleep(0.1)
            raise Exception(f"Failed to remove {path}: file still exists")
        except Exception as e:
            log.warning(f"Failed to remove {path}: {str(e)}")
            raise e


async def get_python_version_string(python_cmd: Path, *args: str):
    proc = await asyncio.create_subprocess_exec(
        python_cmd, *args, "--version", stdout=asyncio.subprocess.PIPE
    )
    out, _ = await proc.communicate()
    return decode_pipe_bytes(out).strip()


async def get_python_version(python_cmd: Path, *args: str):
    string = await get_python_version_string(python_cmd, *args)
    matches = re.match(r"Python (\d+)\.(\d+)", string)
    if not matches:
        log.warning(f"Could not determine Python version: {string}")
        return string, None, None
    else:
        return string, int(matches.group(1)), int(matches.group(2))


def parse_common_errors(output: str, return_code: int | None = None):
    if "error while attempting to bind on address" in output:
        message_part = output.split("bind on address")[-1].strip()
        return (
            _("Could not bind on address")
            + f" {message_part}. "
            + "<a href='https://docs.interstice.cloud/common-issues#bind-address'>More information...</a>"
        )

    nvidia_driver = "Found no NVIDIA driver on your system"
    nvidia_driver_translated = _("Found no NVIDIA driver on your system")
    if nvidia_driver in output:
        message_part = output.split(nvidia_driver)[-1]
        return f"{nvidia_driver_translated} {message_part}<br>" + _(
            "If you do not have an NVIDIA GPU, select a different backend below. Server reinstall may be required."
        )

    readtimeout_pattern = re.compile(
        r"ReadTimeoutError: HTTPSConnectionPool\(host='(.+)'.+\): Read timed out"
    )
    if match := readtimeout_pattern.search(output):
        return _(
            "Connection to {host} timed out during download. Please make sure you have a stable internet connection and try again.",
            host=match.group(1),
        )

    if return_code is not None:
        return f"{output} [{return_code}]"
    return output


_extra_model_paths_yaml = """

krita-managed:
    base_path: ../models
    checkpoints: checkpoints
    clip: clip
    clip_vision: clip_vision
    controlnet: controlnet
    diffusion_models: diffusion_models
    embeddings: embeddings
    inpaint: inpaint
    ipadapter: ipadapter
    loras: loras
    style_models: style_models
    text_encoders: text_encoders
    upscale_models: upscale_models
    unet: unet
    vae: vae
"""


def _configure_extra_model_paths(comfy_dir: Path):
    path = comfy_dir / "extra_model_paths.yaml"
    example = comfy_dir / "extra_model_paths.yaml.example"
    if not path.exists() and example.exists():
        example.rename(path)
    if not path.exists():
        raise Exception(f"Could not find or create extra_model_paths.yaml in {comfy_dir}")
    contents = path.read_text()
    if "krita-managed" not in contents:
        log.info(f"Extending {path}")
        path.write_text(contents + _extra_model_paths_yaml)


def _upgrade_extra_model_paths(src_dir: Path, dst_dir: Path):
    src = src_dir / "extra_model_paths.yaml"
    dst = dst_dir / "extra_model_paths.yaml"
    if src.exists():
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        _configure_extra_model_paths(dst_dir)


def _upgrade_models_dir(src_dir: Path, dst_dir: Path):
    if src_dir.exists() and not dst_dir.exists():
        log.info(f"Moving {src_dir} to {dst_dir}")
        try:
            shutil.move(src_dir, dst_dir)
        except Exception as e:
            log.error(f"Could not move model folder to new location: {str(e)}")
