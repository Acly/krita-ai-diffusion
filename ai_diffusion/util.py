from enum import Enum, Flag
from dataclasses import asdict, is_dataclass
from itertools import islice
from .jobs import Job
from pathlib import Path
from typing import Generator
import asyncio
import importlib.util
import os
import subprocess
import sys
import json
import locale
import logging
import logging.handlers
import re
import statistics
import struct
import zipfile
import zlib
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar
from PyQt5 import sip
from PyQt5.QtCore import QObject, QStandardPaths

T = TypeVar("T")
R = TypeVar("R")
QOBJECT = TypeVar("QOBJECT", bound=QObject)

is_windows = sys.platform.startswith("win")
is_macos = sys.platform == "darwin"
is_linux = not is_windows and not is_macos

plugin_dir = dir = Path(__file__).parent


def _get_user_data_dir():
    if importlib.util.find_spec("krita") is None:
        dir = plugin_dir.parent / ".appdata"
        dir.mkdir(exist_ok=True)
        return dir
    try:
        dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
        if dir.exists() and "krita" in dir.name.lower():
            dir = dir / "ai_diffusion"
        else:
            dir = Path(QStandardPaths.writableLocation(QStandardPaths.GenericDataLocation))
            dir = dir / "krita-ai-diffusion"
        dir.mkdir(exist_ok=True)
        return dir
    except Exception:
        return Path(__file__).parent


user_data_dir = _get_user_data_dir()


def _get_log_dir():
    dir = user_data_dir / "logs"
    dir.mkdir(exist_ok=True)

    legacy_dir = plugin_dir / ".logs"
    try:  # Move logs from old location (v1.14 and earlier)
        if legacy_dir.exists():
            for file in legacy_dir.iterdir():
                file.rename(dir / file.name)
            legacy_dir.rmdir()
    except Exception:
        print(f"Failed to move logs from {legacy_dir} to {dir}")

    return dir


def create_logger(name: str, path: Path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if os.environ.get("AI_DIFFUSION_ENV") == "WORKER":
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.handlers.RotatingFileHandler(
            path, encoding="utf-8", maxBytes=10 * 1024 * 1024, backupCount=4
        )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


log_dir = _get_log_dir()
client_logger = create_logger("krita.ai_diffusion.client", log_dir / "client.log")
server_logger = create_logger("krita.ai_diffusion.server", log_dir / "server.log")


def log_error(error: Exception):
    message = str(error)
    if isinstance(error, AssertionError):
        message = f"Error: Internal assertion failed [{error}]"
    elif not message.startswith("Error:"):
        message = f"Error: {message}"
    client_logger.exception(message)
    return message


def ensure(value: Optional[T], msg="") -> T:
    assert value is not None, msg or "a value is required"
    return value


def maybe(func: Callable[[T], R], value: Optional[T]) -> Optional[R]:
    if value is not None:
        return func(value)
    return None


def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def clamp(value: int, min_value: int, max_value: int):
    return max(min(value, max_value), min_value)


def median_or_zero(values: Iterable[float]) -> float:
    try:
        return statistics.median(values)
    except statistics.StatisticsError:
        return 0


def isnumber(x):
    return isinstance(x, (int, float))


def base_type_match(a, b):
    return type(a) is type(b) or (isnumber(a) and isnumber(b))


def unique(seq: Sequence[T], key) -> list[T]:
    seen = set()
    return [x for x in seq if (k := key(x)) not in seen and not seen.add(k)]


def flatten(seq: Sequence[T | list[T]]) -> Generator[T, None, None]:
    for x in seq:
        if isinstance(x, list):
            yield from x
        else:
            yield x


def sequence_equal(a: Sequence[T], b: Sequence[T]) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def trim_text(text: str, max_length: int) -> str:
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def encode_json(obj: Any):
    if isinstance(obj, Flag):
        return obj.value
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, Path):
        return str(obj.as_posix())
    if is_dataclass(obj):
        assert not isinstance(obj, type)
        return asdict(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def read_json_with_comments(path: Path):
    lines = path.read_text().splitlines()
    return json.loads("\n".join("" if line.strip().startswith("//") else line for line in lines))


def sanitize_prompt(prompt: str):
    if prompt == "":
        return "no prompt"
    prompt = prompt[:40]
    return "".join(c for c in prompt if c.isalnum() or c in " _-")


def find_unused_path(path: Path):
    """Finds an unused path by appending a number to the filename"""
    if not path.exists():
        return path
    stem = path.stem
    ext = path.suffix
    i = 1
    while (new_path := path.with_name(f"{stem}-{i}{ext}")).exists():
        i += 1
    return new_path


if is_linux:
    import signal
    import ctypes

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
        except Exception as e:
            client_logger.warning(f"Failed to attach process to job: {e}")
    return p


_system_encoding = locale.getpreferredencoding(False)
_system_encoding_initialized = not is_windows


async def determine_system_encoding(python_cmd: str):
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
            client_logger.info(f"System locale encoding determined: {_system_encoding}")
        else:
            client_logger.warning(f"Failed to determine system locale: {err}")
    except Exception as e:
        client_logger.warning(f"Failed to determine system locale: {e}")


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


def acquire_elements(l: list[QOBJECT]) -> list[QOBJECT]:
    # Many Pykrita functions return a `QList<QObject*>` where the objects are
    # allocated for the caller. SIP does not handle this case and just leaks
    # the objects outright. Fix this by taking explicit ownership of the objects.
    # Note: ONLY call this if you are confident that the Pykrita function
    # allocates the list members!
    for obj in l:
        if obj is not None:
            sip.transferback(obj)
    return l


# Resave the image with PNG metadata
def resave_image_with_metadata(img_path: Path, job: Job):
    # All this method does is call the other two methods, one to format the metadata and the next to add it to the image
    params = job.params

    metadata_text = _format_like_automatic1111(params)

    _add_png_itxt(img_path, "parameters", metadata_text)


def _format_like_automatic1111(params):
    meta = params.metadata

    prompt = meta.get("prompt", "")
    neg_prompt = meta.get("negative_prompt", "")
    sampler_info = meta.get("sampler", "")
    model = meta.get("checkpoint", "Unknown")
    seed = params.seed
    width = params.bounds.width
    height = params.bounds.height
    strength = meta.get("strength", None)
    loras = meta.get("loras", [])

    # Try to extract sampler, steps, and cfg scale from "sampler"
    match = re.match(r".*?-\s*(.+?)\s*\((\d+)\s*/\s*([\d.]+)\)", sampler_info)
    if match:
        sampler, steps, cfg_scale = match.groups()
    else:
        sampler, steps, cfg_scale = sampler_info, "Unknown", "Unknown"

    # Embed LoRAs in the prompt
    lora_tags = ""
    for lora in loras:
        if isinstance(lora, dict):
            name = lora.get("name")
            weight = lora.get("weight", 1.0)
        elif isinstance(lora, (list, tuple)) and len(lora) >= 2:
            name, weight = lora[0], lora[1]
        else:
            continue
        lora_tags += f" <lora:{name}:{weight}>"

    full_prompt = f"{prompt.strip()}{lora_tags}"

    # Construct output
    lines = []
    lines.append(f"Prompt: {full_prompt}")
    lines.append(f"Negative prompt: {neg_prompt}")
    lines.append(
        f"Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg_scale}, Seed: {seed}, Size: {width}x{height}, Model hash: unknown, Model: {model}"
    )

    if strength is not None and strength != 1.0:
        lines[-1] += f", Denoising strength: {strength}"

    return "\n".join(lines)


def _add_png_itxt(img_path, keyword, text):
    with open(img_path, "rb") as f:
        png_data = f.read()

    if png_data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a valid PNG file")

    offset = 8
    chunks = []
    while offset < len(png_data):
        length = struct.unpack(">I", png_data[offset : offset + 4])[0]
        chunk_type = png_data[offset + 4 : offset + 8]
        chunk_data = png_data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        chunks.append((chunk_type, chunk_data))

    new_png = bytearray()
    new_png += b"\x89PNG\r\n\x1a\n"

    for chunk_type, chunk_data in chunks:
        new_png += struct.pack(">I", len(chunk_data))
        new_png += chunk_type
        new_png += chunk_data
        new_png += struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)

        if chunk_type == b"IHDR":
            # Insert iTXt chunk after IHDR
            keyword_bytes = keyword.encode("latin1")
            text_bytes = text.encode("utf-8")
            itxt_data = (
                keyword_bytes
                + b"\x00"
                + b"\x00"  # compression flag: 0 (not compressed)
                + b"\x00"  # compression method: 0
                + b"\x00"  # language tag: empty
                + b"\x00"  # translated keyword: empty
                + text_bytes
            )
            crc = zlib.crc32(b"iTXt" + itxt_data) & 0xFFFFFFFF
            new_png += struct.pack(">I", len(itxt_data))
            new_png += b"iTXt"
            new_png += itxt_data
            new_png += struct.pack(">I", crc)

    with open(img_path, "wb") as f:
        f.write(new_png)
