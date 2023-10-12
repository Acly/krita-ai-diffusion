from enum import Enum
from itertools import islice
from pathlib import Path
import sys
import logging
import logging.handlers

from .image import Extent
from .settings import settings

is_windows = sys.platform.startswith("win")


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


def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def encode_json(obj):
    if isinstance(obj, Enum):
        return obj.name
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def compute_batch_size(extent: Extent, min_size=512, max_batches: int = None):
    max_batches = max_batches or settings.batch_size
    desired_pixels = min_size * min_size * max_batches
    requested_pixels = extent.width * extent.height
    return max(1, min(max_batches, desired_pixels // requested_pixels))
