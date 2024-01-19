from enum import Enum
from itertools import islice
from pathlib import Path
import os
import sys
import logging
import logging.handlers
import statistics
import zipfile
from typing import Iterable, Optional, Sequence, TypeVar

T = TypeVar("T")

is_windows = sys.platform.startswith("win")
is_macos = sys.platform == "darwin"


def create_logger(name: str, path: Path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.handlers.RotatingFileHandler(
        path, encoding="utf-8", maxBytes=10 * 1024 * 1024, backupCount=4
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    return logger


log_path = Path(__file__).parent / ".logs"
log_path.mkdir(exist_ok=True)
client_logger = create_logger("krita.ai_diffusion.client", log_path / "client.log")
server_logger = create_logger("krita.ai_diffusion.server", log_path / "server.log")


def log_error(error: Exception):
    if isinstance(error, AssertionError):
        message = f"Error: Internal assertion failed [{error}]"
    else:
        message = f"Error: {error}"
    client_logger.exception(message)
    return message


def ensure(value: Optional[T]) -> T:
    assert value is not None
    return value


def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def median_or_zero(values: Iterable[float]) -> float:
    try:
        return statistics.median(values)
    except statistics.StatisticsError:
        return 0


def encode_json(obj):
    if isinstance(obj, Enum):
        return obj.name
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


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


def get_path_dict(paths: Sequence[str | Path]) -> dict:
    """Builds a tree like structure out of a list of paths"""

    def _recurse(dic: dict, chain: tuple[str, ...] | list[str]):
        if len(chain) == 0:
            return
        if len(chain) == 1:
            dic[chain[0]] = None
            return
        key, *new_chain = chain
        if key not in dic:
            dic[key] = {}
        _recurse(dic[key], new_chain)
        return

    new_path_dict = {}
    for path in paths:
        _recurse(new_path_dict, Path(path).parts)
    return new_path_dict


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
