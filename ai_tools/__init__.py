from . import image
from . import diffusion
from . import workflow
from .diffusion import Auto1111, Progress
from .image import Bounds, Extent, Mask, Image, ImageCollection

import importlib.util

if importlib.util.find_spec("krita"):
    from .extension import AIToolsExtension
    from .widget import ImageDiffusionWidget
