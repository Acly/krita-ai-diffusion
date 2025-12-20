"""
Diffusers Server Manager

Manages the installation, startup, and lifecycle of the diffusers server
that runs the Qwen Image Layered pipeline.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Union

from PyQt5.QtNetwork import QNetworkAccessManager

from .network import DownloadProgress
from .platform_tools import create_process, decode_pipe_bytes, is_windows, is_macos
from .settings import settings, ServerBackend
from .util import client_logger as log, server_logger as server_log


_exe = ".exe" if is_windows else ""


class DiffusersServerState(Enum):
    not_installed = 0
    stopped = 1
    starting = 2
    running = 3
    installing = 4


class DiffusersHardware(Enum):
    """Hardware backend for diffusers server."""
    cuda = "NVIDIA (CUDA)"
    rocm = "AMD (ROCm)"
    mps = "Apple Silicon (MPS)"
    cpu = "CPU"

    @staticmethod
    def detect() -> "DiffusersHardware":
        """Auto-detect available hardware."""
        # Check for macOS / Apple Silicon
        if platform.system() == "Darwin":
            return DiffusersHardware.mps

        # Check for NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return DiffusersHardware.cuda
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        # Check for ROCm (AMD)
        try:
            result = subprocess.run(
                ["rocm-smi"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return DiffusersHardware.rocm
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        # Check for ROCm environment variables
        if any(env in os.environ for env in ["ROCM_PATH", "HIP_PATH", "ROCM_HOME"]):
            return DiffusersHardware.rocm

        # Check if ROCm tools exist
        if shutil.which("rocminfo") or shutil.which("rocm-smi"):
            return DiffusersHardware.rocm

        # Default to CPU
        return DiffusersHardware.cpu

    @staticmethod
    def available() -> list["DiffusersHardware"]:
        """Return list of available hardware options."""
        options = [DiffusersHardware.cpu]  # CPU always available

        if platform.system() == "Darwin":
            options.insert(0, DiffusersHardware.mps)
        else:
            # Check NVIDIA
            try:
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    options.insert(0, DiffusersHardware.cuda)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

            # Check ROCm
            rocm_available = False
            try:
                result = subprocess.run(
                    ["rocm-smi"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    rocm_available = True
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

            if not rocm_available:
                if any(env in os.environ for env in ["ROCM_PATH", "HIP_PATH", "ROCM_HOME"]):
                    rocm_available = True
                elif shutil.which("rocminfo") or shutil.which("rocm-smi"):
                    rocm_available = True

            if rocm_available:
                # Insert after CUDA if present, otherwise at start
                insert_pos = 1 if DiffusersHardware.cuda in options else 0
                options.insert(insert_pos, DiffusersHardware.rocm)

        return options


class InstallationProgress(NamedTuple):
    stage: str
    progress: DownloadProgress | tuple[int, int] | None = None
    message: str = ""


Callback = Callable[[InstallationProgress], None]
InternalCB = Callable[[str, Union[str, DownloadProgress]], None]


class DiffusersServer:
    """Manages the diffusers server for Qwen Image Layered pipeline."""

    path: Path
    url: Optional[str] = None
    backend = ServerBackend.cuda
    state = DiffusersServerState.not_installed

    _uv_cmd: Optional[Path] = None
    _python_cmd: Optional[Path] = None
    _process: Optional[asyncio.subprocess.Process] = None
    _task: Optional[asyncio.Task] = None
    _cache_dir: Path

    default_port = 8189

    def __init__(self, path: str | Path | None = None, backend: ServerBackend | None = None):
        base_path = Path(path or settings.server_path)
        if not base_path.is_absolute():
            base_path = Path(__file__).parent / base_path
        self.path = base_path / "diffusers"
        self.backend = backend or settings.server_backend
        self._cache_dir = base_path / ".cache"
        self.check_install()

    def check_install(self):
        """Check if diffusers server is installed."""
        # Check for venv
        python_search_paths = [
            self.path / "venv" / "bin",
            self.path / "venv" / "Scripts",
        ]
        for search_path in python_search_paths:
            python_path = search_path / f"python{_exe}"
            if python_path.exists():
                self._python_cmd = python_path
                break

        # Check for server script
        server_script = self.path / "diffusers_server.py"

        # Check for uv (may be shared with main server)
        uv_exe = "uv.exe" if is_windows else "uv"
        parent_uv = self.path.parent / "uv" / uv_exe
        if parent_uv.exists():
            self._uv_cmd = parent_uv
        else:
            self._uv_cmd = shutil.which(uv_exe)
            if self._uv_cmd:
                self._uv_cmd = Path(self._uv_cmd)

        if self._python_cmd and server_script.exists():
            self.state = DiffusersServerState.stopped
            log.info(f"Found diffusers server installation at {self.path}")
        else:
            self.state = DiffusersServerState.not_installed
            log.info(f"Diffusers server not installed at {self.path}")

    @property
    def is_installed(self) -> bool:
        return self.state != DiffusersServerState.not_installed

    @property
    def is_running(self) -> bool:
        return self.state == DiffusersServerState.running

    async def install(self, callback: Callback, hardware: DiffusersHardware | None = None):
        """Install the diffusers server.

        Args:
            callback: Progress callback function
            hardware: Hardware backend to install for. If None, auto-detects.
        """
        if self.state not in (DiffusersServerState.not_installed,):
            log.warning("Diffusers server already installed")
            return

        self.state = DiffusersServerState.installing
        hardware = hardware or DiffusersHardware.detect()

        def cb(stage: str, message: str | DownloadProgress):
            out_message = ""
            progress = None
            if isinstance(message, str):
                log.info(message)
                out_message = message
            elif isinstance(message, DownloadProgress):
                progress = message
            callback(InstallationProgress(stage, progress, out_message))

        try:
            await self._install(cb, hardware)
        except Exception as e:
            log.exception(f"Diffusers server installation failed: {e}")
            self.state = DiffusersServerState.not_installed
            raise
        finally:
            self.check_install()

    async def _install(self, cb: InternalCB, hardware: DiffusersHardware):
        """Internal installation logic."""
        cb("Installing Diffusers", f"Installing diffusers server in {self.path} for {hardware.value}")

        self.path.mkdir(parents=True, exist_ok=True)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure uv is available
        if not self._uv_cmd:
            raise Exception("uv not found. Please install the main server first.")

        # Create venv
        venv_path = self.path / "venv"
        if not venv_path.exists():
            cb("Creating Python venv", f"Creating virtual environment at {venv_path}")
            cmd = [str(self._uv_cmd), "venv", "--python", "3.12", str(venv_path)]
            await self._execute_process("Python", cmd, self.path, cb)

        # Update python path
        if is_windows:
            self._python_cmd = venv_path / "Scripts" / "python.exe"
        else:
            self._python_cmd = venv_path / "bin" / "python3"

        # Install PyTorch based on hardware
        cb("Installing PyTorch", f"Installing PyTorch for {hardware.value}...")
        torch_args = self._get_torch_install_args(hardware)
        await self._pip_install("PyTorch", torch_args, cb)

        # Install huggingface-hub first with correct version constraint (diffusers requires <1.0)
        cb("Installing HuggingFace Hub", "Installing huggingface-hub...")
        await self._pip_install("HuggingFace Hub", ["huggingface-hub>=0.34.0,<1.0"], cb)

        # Install diffusers from git (required for Qwen)
        cb("Installing Diffusers", "Installing diffusers from git...")
        diffusers_args = ["git+https://github.com/huggingface/diffusers", "--no-deps"]
        await self._pip_install("Diffusers", diffusers_args, cb)

        # Install diffusers dependencies (excluding huggingface-hub which we pinned)
        cb("Installing Diffusers deps", "Installing diffusers dependencies...")
        await self._pip_install("Diffusers deps", ["diffusers[torch]"], cb)

        # Install transformers (stable version compatible with diffusers)
        cb("Installing Transformers", "Installing transformers...")
        await self._pip_install("Transformers", ["transformers>=4.45.0,<5.0"], cb)

        # Re-pin huggingface-hub in case transformers upgraded it
        cb("Fixing HuggingFace Hub", "Ensuring huggingface-hub version...")
        await self._pip_install("HuggingFace Hub", ["huggingface-hub>=0.34.0,<1.0"], cb)

        # Install server dependencies
        cb("Installing Server", "Installing FastAPI and uvicorn...")
        server_deps = ["fastapi", "uvicorn", "pillow", "accelerate"]
        await self._pip_install("Server", server_deps, cb)

        # Install optimum-quanto for quantization support
        cb("Installing Quantization", "Installing optimum-quanto...")
        await self._pip_install("Quantization", ["optimum-quanto"], cb)

        # Install RamTorch for memory-efficient inference (optional but included)
        cb("Installing RamTorch", "Installing RamTorch for memory optimization...")
        await self._pip_install("RamTorch", ["ramtorch"], cb)

        # Copy server script
        cb("Copying Server Script", "Copying diffusers_server.py...")
        # Resolve symlinks to find the actual project root (needed when running as Krita plugin)
        real_path = Path(__file__).resolve().parent.parent / "scripts" / "diffusers_server.py"
        dest_script = self.path / "diffusers_server.py"
        if real_path.exists():
            shutil.copy2(real_path, dest_script)
            log.info(f"Copied server script to {dest_script}")
        else:
            raise Exception(f"Server script not found at {real_path}")

        # Save hardware config for start command
        config_file = self.path / "hardware.txt"
        config_file.write_text(hardware.name)

        cb("Installation Complete", f"Diffusers server installed successfully for {hardware.value}")
        self.state = DiffusersServerState.stopped

    async def update_dependencies(self, callback: Callback):
        """Update/reinstall server dependencies without full reinstall."""
        if not self.is_installed:
            raise Exception("Server is not installed")

        if self.state == DiffusersServerState.running:
            raise Exception("Stop the server before updating dependencies")

        def cb(stage: str, message: str | DownloadProgress):
            out_message = ""
            progress = None
            if isinstance(message, str):
                log.info(message)
                out_message = message
            elif isinstance(message, DownloadProgress):
                progress = message
            callback(InstallationProgress(stage, progress, out_message))

        try:
            cb("Updating Dependencies", "Updating server dependencies...")

            # Fix huggingface-hub version first (diffusers requires <1.0)
            cb("Fixing huggingface-hub", "Downgrading huggingface-hub to compatible version...")
            await self._pip_install("huggingface-hub", ["huggingface-hub>=0.34.0,<1.0", "--force-reinstall"], cb)

            # Reinstall diffusers from git (may have updates)
            cb("Updating Diffusers", "Updating diffusers from git...")
            diffusers_args = ["git+https://github.com/huggingface/diffusers", "--force-reinstall", "--no-deps"]
            await self._pip_install("Diffusers", diffusers_args, cb)

            # Reinstall diffusers deps with correct versions
            cb("Installing Diffusers deps", "Installing diffusers dependencies...")
            await self._pip_install("Diffusers deps", ["diffusers[torch]"], cb)

            # Update transformers (stable version compatible with diffusers)
            cb("Updating Transformers", "Updating transformers...")
            await self._pip_install("Transformers", ["transformers>=4.45.0,<5.0", "--force-reinstall"], cb)

            # Re-pin huggingface-hub in case transformers upgraded it
            cb("Fixing huggingface-hub", "Ensuring huggingface-hub version...")
            await self._pip_install("huggingface-hub", ["huggingface-hub>=0.34.0,<1.0"], cb)

            # Update server dependencies
            cb("Updating Server", "Updating FastAPI and uvicorn...")
            server_deps = ["fastapi", "uvicorn", "pillow", "accelerate", "--upgrade"]
            await self._pip_install("Server", server_deps, cb)

            # Install/update optimum-quanto for quantization
            cb("Updating Quantization", "Installing/updating optimum-quanto...")
            await self._pip_install("Quantization", ["optimum-quanto", "--upgrade"], cb)

            # Install/update RamTorch for memory optimization
            cb("Updating RamTorch", "Installing/updating RamTorch...")
            await self._pip_install("RamTorch", ["ramtorch", "--upgrade"], cb)

            # Update server script
            cb("Updating Server Script", "Copying latest diffusers_server.py...")
            real_path = Path(__file__).resolve().parent.parent / "scripts" / "diffusers_server.py"
            dest_script = self.path / "diffusers_server.py"
            if real_path.exists():
                shutil.copy2(real_path, dest_script)
                log.info(f"Updated server script at {dest_script}")
            else:
                log.warning(f"Server script not found at {real_path}")

            cb("Update Complete", "Dependencies updated successfully")

        except Exception as e:
            log.exception(f"Failed to update dependencies: {e}")
            raise

    def _get_torch_install_args(self, hardware: DiffusersHardware) -> list[str]:
        """Get PyTorch installation arguments for the specified hardware."""
        if hardware is DiffusersHardware.cpu:
            return ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"]

        elif hardware is DiffusersHardware.cuda:
            # CUDA 12.8 for latest NVIDIA GPUs
            return ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"]

        elif hardware is DiffusersHardware.rocm:
            # ROCm for AMD GPUs - use PyTorch ROCm 6.4 wheels
            return ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/rocm6.4"]

        elif hardware is DiffusersHardware.mps:
            # Apple Silicon uses standard PyTorch which includes MPS support
            return ["torch", "torchvision"]

        else:
            # Fallback to CPU
            return ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"]

    async def _pip_install(self, name: str, args: list[str], cb: InternalCB):
        """Install packages using pip via uv."""
        env = {"VIRTUAL_ENV": str(self.path / "venv")}
        cmd = [str(self._uv_cmd), "pip", "install", *args]
        return await self._execute_process(name, cmd, self.path, cb, env=env)

    async def _execute_process(
        self,
        name: str,
        cmd: list,
        cwd: Path,
        cb: InternalCB,
        env: dict | None = None,
    ):
        """Execute a subprocess and forward output."""
        errlog = ""
        cmd = [str(c) for c in cmd]
        cb(f"Installing {name}", f"Executing {' '.join(cmd)}")

        process = await create_process(
            cmd[0], *cmd[1:], cwd=cwd, additional_env=env, pipe_stderr=True
        )

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
            raise Exception(f"Error during {name} installation: {errlog}")

    def _get_installed_hardware(self) -> DiffusersHardware:
        """Read saved hardware configuration from installation."""
        config_file = self.path / "hardware.txt"
        if config_file.exists():
            try:
                hw_name = config_file.read_text().strip()
                return DiffusersHardware[hw_name]
            except (KeyError, ValueError):
                pass
        # Fallback to detection
        return DiffusersHardware.detect()

    def _get_device_arg(self, hardware: DiffusersHardware) -> str:
        """Get the device argument for the server based on hardware."""
        if hardware is DiffusersHardware.cuda:
            return "cuda"
        elif hardware is DiffusersHardware.rocm:
            return "cuda"  # ROCm uses CUDA compatibility layer via HIP
        elif hardware is DiffusersHardware.mps:
            return "mps"
        else:
            return "cpu"

    def _update_server_script(self):
        """Copy the latest server script to the installation directory."""
        real_path = Path(__file__).resolve().parent.parent / "scripts" / "diffusers_server.py"
        dest_script = self.path / "diffusers_server.py"
        if real_path.exists():
            try:
                shutil.copy2(real_path, dest_script)
                log.debug(f"Updated server script from {real_path}")
            except Exception as e:
                log.warning(f"Could not update server script: {e}")

    async def start(self, port: int | None = None):
        """Start the diffusers server."""
        if self.state != DiffusersServerState.stopped:
            if self.state == DiffusersServerState.running:
                log.info("Diffusers server already running")
                return self.url
            raise Exception(f"Cannot start server in state {self.state}")

        if not self._python_cmd:
            raise Exception("Python not found. Is the server installed?")

        port = port or self.default_port
        self.state = DiffusersServerState.starting

        # Always copy the latest server script before starting
        self._update_server_script()

        # Determine device from saved hardware config
        hardware = self._get_installed_hardware()
        device = self._get_device_arg(hardware)

        try:
            args = [
                str(self._python_cmd),
                "diffusers_server.py",
                "--host", "127.0.0.1",
                "--port", str(port),
                "--device", device,
            ]

            # Default model
            if settings.diffusers_default_model:
                args.extend(["--model", settings.diffusers_default_model])

            # CPU offload mode
            if settings.diffusers_cpu_offload:
                args.extend(["--offload", "model"])

            # VAE tiling
            if settings.diffusers_vae_tiling:
                args.append("--vae-tiling")
            else:
                args.append("--no-vae-tiling")

            # Quantization
            quant = settings.diffusers_quantization
            if quant and quant != "none":
                args.extend(["--quantization", quant])

            # RamTorch for memory-efficient inference
            if settings.diffusers_ramtorch:
                args.append("--ramtorch")

            log.info(f"Starting diffusers server for {hardware.value}: {' '.join(args)}")

            self._process = await create_process(
                args[0], *args[1:], cwd=self.path, is_job=True
            )

            # Wait for server to be ready
            assert self._process.stdout is not None
            async for line in self._process.stdout:
                text = decode_pipe_bytes(line).strip()
                server_log.info(f"[diffusers] {text}")

                # uvicorn outputs "Uvicorn running on http://..."
                if "Uvicorn running on" in text or "Application startup complete" in text:
                    self.state = DiffusersServerState.running
                    self.url = f"http://127.0.0.1:{port}"
                    log.info(f"Diffusers server started at {self.url}")
                    break

            if self.state != DiffusersServerState.running:
                raise Exception("Server failed to start")

            # Start background task to monitor output
            self._task = asyncio.create_task(self._run())
            return self.url

        except Exception as e:
            log.exception(f"Failed to start diffusers server: {e}")
            self.state = DiffusersServerState.stopped
            if self._process:
                self._process.kill()
                self._process = None
            raise

    async def _run(self):
        """Background task to monitor server output."""
        assert self._process and self._process.stdout

        try:
            async for line in self._process.stdout:
                server_log.info(f"[diffusers] {decode_pipe_bytes(line).strip()}")

            code = await asyncio.wait_for(self._process.wait(), timeout=1)
            if code != 0:
                log.error(f"Diffusers server terminated with code {code}")
            else:
                log.info("Diffusers server shut down successfully")

        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            log.warning("Diffusers server did not terminate cleanly")

        self.state = DiffusersServerState.stopped
        self._process = None

    async def stop(self):
        """Stop the diffusers server."""
        if self.state != DiffusersServerState.running:
            return

        try:
            if self._process and self._task:
                log.info("Stopping diffusers server")
                self._process.terminate()
                await asyncio.wait_for(self._task, timeout=5)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            log.warning("Diffusers server did not terminate in time")
            if self._process:
                self._process.kill()
        finally:
            self.state = DiffusersServerState.stopped
            self._process = None

    def terminate(self):
        """Forcefully terminate the server."""
        try:
            if self._process is not None:
                self._process.terminate()
        except Exception as e:
            log.warning(f"Error terminating diffusers server: {e}")

    async def uninstall(self, callback: Callback):
        """Uninstall the diffusers server."""
        if self.state == DiffusersServerState.running:
            await self.stop()

        def info(message: str):
            log.info(message)
            callback(InstallationProgress("Uninstalling", message=message))

        try:
            info(f"Removing diffusers server at {self.path}")
            if self.path.exists():
                shutil.rmtree(self.path)
            info("Diffusers server uninstalled")
        except Exception as e:
            log.error(f"Error uninstalling diffusers server: {e}")
            raise
        finally:
            self.check_install()
