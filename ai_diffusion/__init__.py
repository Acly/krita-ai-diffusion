"""Generative AI plugin for Krita using Stable Diffusion"""

__version__ = "1.0.2"

from . import util
from .settings import Settings, Setting, settings, PerformancePreset, ServerBackend, ServerMode
from .style import SDVersion, Style, Styles, StyleSettings
from .network import NetworkError, Interrupted, OutOfMemoryError, RequestManager, DownloadProgress
from .client import Client, ClientEvent, ClientMessage, DeviceInfo, MissingResource, ResourceKind
from .image import Bounds, Extent, Mask, Image, ImageCollection
from .comfyworkflow import ComfyWorkflow
from .server import Server, ServerState, InstallationProgress
from .workflow import Control, ControlMode, Conditioning
from . import network, workflow

import importlib.util

if importlib.util.find_spec("krita"):
    from .document import Document
    from .ui import ImageDiffusionWidget, SettingsDialog
    from .extension import AIToolsExtension
