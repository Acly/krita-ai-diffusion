"""Generative AI plugin for Krita using Stable Diffusion"""

__version__ = "1.8.1"

import importlib.util

if not importlib.util.find_spec(".websockets.src", "ai_diffusion"):
    raise ImportError(
        "Could not find websockets module. This indicates that it was not installed with the"
        " plugin. Please make sure to download a plugin release package (NOT just the source!). You"
        " can find the latest release package here:"
        " https://github.com/Acly/krita-ai-diffusion/releases"
    )

from . import util
from .settings import Settings, Setting, settings, PerformancePreset, ServerBackend, ServerMode
from .style import SDVersion, Style, Styles, StyleSettings
from .resources import ControlMode, CustomNode, ResourceKind, MissingResource
from .network import NetworkError, Interrupted, OutOfMemoryError, RequestManager, DownloadProgress
from .client import Client, ClientEvent, ClientMessage, DeviceInfo
from .image import Bounds, Extent, Mask, Image, ImageCollection
from .comfyworkflow import ComfyWorkflow
from .server import Server, ServerState, InstallationProgress
from .workflow import Control, Conditioning
from . import network, workflow

# The following imports depend on the code running inside Krita, so the cannot be imported in tests.
if importlib.util.find_spec("krita"):
    from .document import Document
    from .ui import ImageDiffusionWidget, SettingsDialog
    from .extension import AIToolsExtension
