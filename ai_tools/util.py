from .image import Extent
from .settings import settings


def compute_batch_size(extent: Extent, min_size: int = None, max_batches: int = None):
    min_size = min_size or settings.min_image_size
    max_batches = max_batches or settings.batch_size
    desired_pixels = min_size * min_size * max_batches
    requested_pixels = extent.width * extent.height
    return max(1, min(max_batches, desired_pixels // requested_pixels))
