from enum import Enum, Flag
from dataclasses import asdict, is_dataclass
from itertools import islice
from pathlib import Path
from typing import Generator
import importlib.util
import os
import sys
import json
import logging
import logging.handlers
import statistics
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar
from PyQt5 import sip
from PyQt5.QtCore import QObject, QStandardPaths

T = TypeVar("T")
R = TypeVar("R")
QOBJECT = TypeVar("QOBJECT", bound=QObject)

plugin_dir = dir = Path(__file__).parent


def _get_user_data_dir():
    spec = importlib.util.find_spec("krita")
    origin = getattr(spec, "origin", "") if spec is not None else ""
    # Treat local helper modules named `krita.py` as "no Krita" so tests and
    # non-Krita environments use a local appdata directory.
    if spec is None or (origin and origin.endswith("krita.py")):
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
