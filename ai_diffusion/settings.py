from __future__ import annotations
import os
import json
from enum import Enum
from pathlib import Path
from typing import Optional, Any
from PyQt5.QtCore import QObject, pyqtSignal

from . import util


class ServerMode(Enum):
    undefined = -1
    managed = 0
    external = 1


class ServerBackend(Enum):
    cpu = ("Run on CPU", True)
    cuda = ("Use CUDA (NVIDIA GPU)", not util.is_macos)
    mps = ("Use MPS (Metal Performance Shader)", util.is_macos)
    directml = ("Use DirectML (GPU)", util.is_windows)

    @staticmethod
    def supported():
        return [b for b in ServerBackend if b.value[1]]

    @staticmethod
    def default():
        if util.is_macos:
            return ServerBackend.mps
        else:
            return ServerBackend.cuda


class PerformancePreset(Enum):
    auto = "Automatic"
    cpu = "CPU"
    low = "GPU low (less than 6GB)"
    medium = "GPU medium (6GB to 12GB)"
    high = "GPU high (more than 12GB)"
    custom = "Custom"


class Setting:
    def __init__(self, name: str, default, desc="", help="", items=None):
        self.name = name
        self.desc = desc
        self.default = default
        self.help = help
        self.items = items

    def str_to_enum(self, s: str):
        assert isinstance(self.default, Enum)
        EnumType = type(self.default)
        try:
            return EnumType[s]
        except KeyError:
            return self.default


class Settings(QObject):
    default_path = Path(__file__).parent / "settings.json"

    server_mode: ServerMode
    _server_mode = Setting(
        "Server Management",
        ServerMode.undefined,
        "To generate images, the plugin connects to a ComfyUI server",
    )

    server_path: str
    _server_path = Setting(
        "Server Path",
        str(Path(__file__).parent / ".server"),
        "Directory where ComfyUI will be installed. At least 10GB of free disk space is required"
        " for a full installation.",
    )

    server_url: str
    _server_url = Setting(
        "Server URL",
        "127.0.0.1:8188",
        "URL used to connect to a running ComfyUI server. Default is 127.0.0.1:8188 (local).",
    )

    server_backend: ServerBackend
    _server_backend = Setting("Server Backend", ServerBackend.default())

    server_arguments: str
    _server_arguments = Setting(
        "Server Arguments", "", "Additional command line arguments passed to the server"
    )

    selection_grow: int
    _selection_grow = Setting(
        "Selection Grow", 7, "Selection area is expanded by a fraction of its size"
    )

    selection_feather: int
    _selection_feather = Setting(
        "Selection Feather", 7, "The border is blurred by a fraction of selection size"
    )

    selection_padding: int
    _selection_padding = Setting(
        "Selection Padding", 7, "Minimum additional padding around the selection area"
    )

    fixed_seed: bool
    _fixed_seed = Setting("Use Fixed Seed", False, "Fixes the random seed to a specific value")

    random_seed: str
    _random_seed = Setting(
        "Random Seed", "0", "Random number to produce different results with each generation"
    )

    prompt_line_count: int
    _prompt_line_count = Setting(
        "Prompt Line Count", 2, "Size of the text editor for image descriptions"
    )

    show_negative_prompt: bool
    _show_negative_prompt = Setting(
        "Negative Prompt", False, "Show text editor to describe things to avoid"
    )

    auto_preview: bool
    _auto_preview = Setting(
        "Auto Preview", True, "Automatically preview the first generated result on the canvas"
    )

    show_control_end: bool
    _show_control_end = Setting("Control ending step", False, "Show control ending step ratio")

    history_size: int
    _history_size = Setting(
        "History Size", 1000, "Main memory (RAM) used to keep the history of generated images"
    )

    performance_preset: PerformancePreset
    _performance_preset = Setting(
        "Performance Preset",
        PerformancePreset.auto,
        "Configures performance settings to match available hardware.",
    )

    batch_size: int
    _batch_size = Setting(
        "Maximum Batch Size",
        4,
        "Increase efficiency by generating multiple images at once",
    )

    diffusion_tile_size: int
    _diffusion_tile_size = Setting(
        "Diffusion Tile Size",
        2048,
        "Resolution threshold at which diffusion is split up into multiple tiles",
    )

    _performance_presets = {
        PerformancePreset.cpu: {
            "batch_size": 1,
            "diffusion_tile_size": 4096,
        },
        PerformancePreset.low: {
            "batch_size": 2,
            "diffusion_tile_size": 1024,
        },
        PerformancePreset.medium: {
            "batch_size": 4,
            "diffusion_tile_size": 2048,
        },
        PerformancePreset.high: {
            "batch_size": 8,
            "diffusion_tile_size": 4096,
        },
    }

    # Folder where intermediate images are stored for debug purposes (default: None)
    debug_image_folder = os.environ.get("KRITA_AI_DIFFUSION_DEBUG_IMAGE")

    changed = pyqtSignal(str, object)

    _values: dict[str, Any]

    def __init__(self):
        super().__init__()
        self.restore()

    def __getattr__(self, name: str):
        if name in self._values:
            return self._values[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        if name in self._values:
            self._values[name] = value
            self.changed.emit(name, value)
            if name == "performance_preset":
                self.apply_performance_preset(value)
        else:
            object.__setattr__(self, name, value)

    def restore(self):
        self.__dict__["_values"] = {
            k[1:]: v.default for k, v in Settings.__dict__.items() if isinstance(v, Setting)
        }

    def save(self, path: Optional[Path] = None):
        path = self.default_path if path is None else path
        with open(path, "w") as file:
            file.write(json.dumps(self._values, default=util.encode_json, indent=4))

    def load(self, path: Optional[Path] = None):
        path = self.default_path if path is None else path
        if not path.exists():
            self.save()  # create new file with defaults
            return
        with open(path, "r") as file:
            contents = json.loads(file.read())
            for k, v in contents.items():
                setting = getattr(Settings, f"_{k}", None)
                if setting is not None:
                    if isinstance(setting.default, Enum):
                        self._values[k] = setting.str_to_enum(v)
                    elif isinstance(setting.default, type(v)):
                        self._values[k] = v
                    else:
                        raise Exception(f"{v} is not a valid value for '{k}'")

    def apply_performance_preset(self, preset: PerformancePreset):
        if preset not in [PerformancePreset.custom, PerformancePreset.auto]:
            for k, v in self._performance_presets[preset].items():
                self._values[k] = v


settings = Settings()
