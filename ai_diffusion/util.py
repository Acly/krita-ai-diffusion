from enum import Enum
from itertools import islice
from pathlib import Path
import os
import sys
import logging
import logging.handlers
import zipfile
from typing import Optional, TypeVar

T = TypeVar("T")

is_windows = sys.platform.startswith("win")
is_macos = (sys.platform == 'darwin')

from .image import Extent
from .settings import settings

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


def encode_json(obj):
    if isinstance(obj, Enum):
        return obj.name
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def compute_batch_size(extent: Extent, min_size=512, max_batches: Optional[int] = None):
    max_batches = max_batches or settings.batch_size
    desired_pixels = min_size * min_size * max_batches
    requested_pixels = extent.width * extent.height
    return max(1, min(max_batches, desired_pixels // requested_pixels))


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
