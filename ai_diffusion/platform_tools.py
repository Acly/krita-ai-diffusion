import asyncio
import locale
import os
import subprocess
import sys
import zipfile
from logging import Logger
from pathlib import Path

is_windows = sys.platform.startswith("win")
is_macos = sys.platform == "darwin"
is_linux = not is_windows and not is_macos


if is_linux:
    import ctypes
    import signal

    libc = ctypes.CDLL("libc.so.6")

    def set_pdeathsig():
        return libc.prctl(1, signal.SIGTERM)


async def create_process(
    program: str | Path,
    *args: str,
    cwd: Path | None = None,
    additional_env: dict | None = None,
    pipe_stderr=False,
    is_job=False,
):
    platform_args = {}
    if is_windows:
        platform_args["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore
    if is_linux:
        platform_args["preexec_fn"] = set_pdeathsig

    env = os.environ.copy()
    if additional_env:
        env.update(additional_env)
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]  # Krita adds its own python path, which can cause conflicts

    out = asyncio.subprocess.PIPE
    err = asyncio.subprocess.PIPE if pipe_stderr else asyncio.subprocess.STDOUT

    p = await asyncio.create_subprocess_exec(
        program, *args, cwd=cwd, stdout=out, stderr=err, env=env, **platform_args
    )
    if is_windows and is_job:
        try:
            from . import win32

            win32.attach_process_to_job(p.pid)
        except Exception:  # noqa
            pass
    return p


_system_encoding = locale.getpreferredencoding(False)
_system_encoding_initialized = not is_windows


async def determine_system_encoding(python_cmd: str, log: Logger):
    """Windows: Krita's embedded Python always reports UTF-8, even if the system
    uses a different encoding (likely). To decode subprocess output correctly,
    the encoding used by an outside process must be determined."""
    global _system_encoding
    global _system_encoding_initialized
    if _system_encoding_initialized:
        return
    try:
        _system_encoding_initialized = True  # only try once
        result = await create_process(
            python_cmd, "-c", "import locale; print(locale.getpreferredencoding(False))"
        )
        out, err = await result.communicate()
        if out:
            enc = out.decode().strip()
            b"test".decode(enc)
            _system_encoding = enc
            log.info(f"System locale encoding determined: {_system_encoding}")
        else:
            log.warning(f"Failed to determine system locale: {err}")
    except Exception as e:
        log.warning(f"Failed to determine system locale: {e}")


def decode_pipe_bytes(data: bytes) -> str:
    return data.decode(_system_encoding, errors="replace")


class LongPathZipFile(zipfile.ZipFile):
    # zipfile.ZipFile does not support long paths (260+?) on Windows
    # for latest python, changing cwd and using relative paths helps, but not for python in Krita 5.2
    def _extract_member(self, member, targetpath, pwd):
        # Prepend \\?\ to targetpath to bypass MAX_PATH limit
        targetpath = os.path.abspath(targetpath)
        if targetpath.startswith("\\\\"):
            targetpath = "\\\\?\\UNC\\" + targetpath[2:]
        else:
            targetpath = "\\\\?\\" + targetpath
        return super()._extract_member(member, targetpath, pwd)  # type: ignore


ZipFile = LongPathZipFile if is_windows else zipfile.ZipFile


def _get_cuda_compute_capabilities() -> list[tuple[int, int]]:
    """
    Returns a list of (major, minor) compute capability tuples for each CUDA device
    if the CUDA driver is available and initializes successfully.
    """
    import ctypes
    import ctypes.util

    # candidate library names for Windows and Linux
    candidates = []
    if is_windows:
        candidates = ["nvcuda.dll"]
    else:
        # try common sonames; order matters (libcuda.so.1 is usually a stable symlink)
        candidates = [
            "libcuda.so.1",
            "libcuda.so",
            ctypes.util.find_library("cuda") or "libcuda.so",
        ]

    lib = None
    for name in candidates:
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return []

    CUDA_SUCCESS = 0

    try:
        cuInit = lib.cuInit
        cuInit.argtypes = [ctypes.c_uint]
        cuInit.restype = ctypes.c_int

        cuDeviceGetCount = lib.cuDeviceGetCount
        cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cuDeviceGetCount.restype = ctypes.c_int

        cuDeviceGet = lib.cuDeviceGet
        cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cuDeviceGet.restype = ctypes.c_int

        cuDeviceComputeCapability = lib.cuDeviceComputeCapability
        cuDeviceComputeCapability.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        cuDeviceComputeCapability.restype = ctypes.c_int
    except AttributeError:
        return []

    if cuInit(0) != CUDA_SUCCESS:
        return []

    count = ctypes.c_int(0)
    if cuDeviceGetCount(ctypes.byref(count)) != CUDA_SUCCESS:
        return []

    devices = []
    for i in range(count.value):
        dev = ctypes.c_int(0)
        if cuDeviceGet(ctypes.byref(dev), i) != CUDA_SUCCESS:
            continue
        major = ctypes.c_int(0)
        minor = ctypes.c_int(0)
        if (
            cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), dev.value)
            != CUDA_SUCCESS
        ):
            continue
        devices.append((int(major.value), int(minor.value)))

    return devices


_cuda_device_list = None


def get_cuda_devices() -> list[tuple[int, int]]:
    global _cuda_device_list
    if _cuda_device_list is None:
        _cuda_device_list = _get_cuda_compute_capabilities()
    return _cuda_device_list
