from . import image
from . import diffusion
from . import workflow
from .diffusion import Auto1111, Progress, Interrupted, NetworkError
from .image import Bounds, Extent, Mask, Image, ImageCollection
from .settings import Settings, Setting, settings

import importlib.util

if importlib.util.find_spec("krita"):
    from .document import Document
    from .ui import ImageDiffusionWidget, SettingsDialog
    from .extension import AIToolsExtension
