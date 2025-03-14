from __future__ import annotations
from dataclasses import dataclass
import os
import json
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional, Any
from PyQt5.QtCore import QObject, pyqtSignal

from .util import is_macos, is_windows, user_data_dir, client_logger as log
from .util import encode_json, read_json_with_comments
from .localization import translate as _


class ServerMode(Enum):
    undefined = -1
    managed = 0
    external = 1
    cloud = 2


class ServerBackend(Enum):
    cpu = (_("Run on CPU"), True)
    cuda = (_("Use CUDA (NVIDIA GPU)"), not is_macos)
    mps = (_("Use MPS (Metal Performance Shader)"), is_macos)
    directml = (_("Use DirectML (GPU)"), is_windows)

    @staticmethod
    def supported():
        return [b for b in ServerBackend if b.value[1]]

    @staticmethod
    def default():
        if is_macos:
            return ServerBackend.mps
        else:
            return ServerBackend.cuda


class GenerationFinishedAction(Enum):
    none = _("Do Nothing")
    preview = _("Preview")
    apply = _("Apply")


class ApplyBehavior(Enum):
    replace = _("Modify active layer")
    layer = _("New layer on top")
    layer_active = _("New layer above active")


class ApplyRegionBehavior(Enum):
    none = _("Do not update regions")
    replace = _("Modify region layers")
    layer_group = _("Layer group")
    transparency_mask = _("Layer group + mask")
    no_hide = _("Layer group (don't hide)")


class PerformancePreset(Enum):
    auto = _("Automatic")
    cpu = _("CPU")
    low = _("GPU low (up to 6GB)")
    medium = _("GPU medium (6GB to 12GB)")
    high = _("GPU high (more than 12GB)")
    cloud = _("Cloud")
    custom = _("Custom")


class PerformancePresetSettings(NamedTuple):
    batch_size: int = 4
    resolution_multiplier: float = 1.0
    max_pixel_count: int = 6
    tiled_vae: bool = False


@dataclass
class PerformanceSettings:
    batch_size: int = 4
    resolution_multiplier: float = 1.0
    max_pixel_count: int = 6
    dynamic_caching: bool = False
    tiled_vae: bool = False


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
    default_path = user_data_dir / "settings.json"

    language: str
    _language = Setting(
        _("Language"),
        "en",
        _("Interface language used by the plugin - requires restart!"),
    )

    auto_update: bool
    _auto_update = Setting(
        _("Enable Automatic Updates"), True, _("Check for new versions of the plugin on startup")
    )

    server_mode: ServerMode
    _server_mode = Setting(
        _("Server Management"),
        ServerMode.undefined,
        _("To generate images, the plugin connects to a ComfyUI server"),
    )

    access_token: str
    _access_token = Setting(_("Cloud Access Token"), "")

    server_path: str
    _server_path = Setting(
        _("Server Path"),
        str(user_data_dir / "server"),
        _(
            "Directory where ComfyUI will be installed. At least 10GB of free disk space is required for a minimal installation."
        ),
    )

    server_url: str
    _server_url = Setting(
        _("Server URL"),
        "127.0.0.1:8188",
        _("URL used to connect to a running ComfyUI server. Default is 127.0.0.1:8188 (local)."),
    )

    server_backend: ServerBackend
    _server_backend = Setting(_("Server Backend"), ServerBackend.default())

    server_arguments: str
    _server_arguments = Setting(
        _("Server Arguments"), "", _("Additional command line arguments passed to the server")
    )

    selection_grow: int
    _selection_grow = Setting(
        _("Selection Grow"), 5, _("Selection area is expanded by a fraction of its size")
    )

    selection_feather: int
    _selection_feather = Setting(
        _("Selection Feather"), 5, _("The border is blurred by a fraction of selection size")
    )

    selection_padding: int
    _selection_padding = Setting(
        _("Selection Padding"), 7, _("Minimum additional padding around the selection area")
    )

    nsfw_filter: float
    _nsfw_filter = Setting(
        _("NSFW Filter"), 0.0, _("Attempt to filter out images with explicit content")
    )

    new_seed_after_apply: bool
    _new_seed_after_apply = Setting(
        _("Live: New Seed after Apply"),
        False,
        _("Pick a new seed after copying the result to the canvas in Live mode"),
    )

    prompt_translation: str
    _prompt_translation = Setting(
        _("Prompt Translation"),
        "",
        _("Translate text prompts from the selected language to English"),
    )

    prompt_line_count: int
    _prompt_line_count = Setting(
        _("Prompt Line Count"), 2, _("Size of the text editor for image descriptions")
    )

    prompt_line_count_live: int
    _prompt_line_count_live = Setting("Prompt Line Count (Live)", 2)

    show_negative_prompt: bool
    _show_negative_prompt = Setting(
        _("Negative Prompt"), False, _("Show text editor to describe things to avoid")
    )

    generation_finished_action: GenerationFinishedAction
    _generation_finished_action = Setting(
        _("Finished Generation"),
        GenerationFinishedAction.preview,
        _("Action to take when an image generation job finishes"),
    )

    show_steps: bool
    _show_steps = Setting(
        _("Show Steps"), False, _("Display the number of steps to be evaluated in the weights box.")
    )

    tag_files: list[str]
    _tag_files = Setting(
        _("Tag Auto-Completion"),
        list(),
        _("Enable text completion for tags from the selected files"),
    )

    apply_behavior: ApplyBehavior
    _apply_behavior = Setting(
        _("Apply Behavior"),
        ApplyBehavior.layer,
        _("Choose how result images are applied to the canvas (generation workspaces)"),
    )

    apply_region_behavior: ApplyRegionBehavior
    _apply_region_behavior = Setting("Apply Region Behavior", ApplyRegionBehavior.layer_group)

    apply_behavior_live: ApplyBehavior
    _apply_behavior_live = Setting(
        "Apply Behavior (Live)",
        ApplyBehavior.replace,
        "Choose how result images are applied to the canvas in Live mode",
    )

    apply_region_behavior_live: ApplyRegionBehavior
    _apply_region_behavior_live = Setting(
        "Apply Region Behavior (Live)", ApplyRegionBehavior.replace
    )

    show_builtin_styles: bool
    _show_builtin_styles = Setting(_("Show pre-installed styles"), True)

    history_size: int
    _history_size = Setting(
        _("Active History Size"),
        1000,
        _("Main memory (RAM) used for the history of generated images"),
    )

    history_storage: int
    _history_storage = Setting(
        _("Stored History Size"),
        20,
        _("Memory used to store generated images in .kra files on disk"),
    )

    performance_preset: PerformancePreset
    _performance_preset = Setting(
        _("Performance Preset"),
        PerformancePreset.auto,
        _("Configures performance settings to match available hardware."),
    )

    batch_size: int
    _batch_size = Setting(
        _("Maximum Batch Size"),
        4,
        _("Increase efficiency by generating multiple images at once"),
    )

    resolution_multiplier: float
    _resolution_multiplier = Setting(
        _("Resolution Multiplier"),
        1.0,
        _(
            "Scaling factor for generation. Values below 1.0 improve performance for high resolution canvas."
        ),
    )

    max_pixel_count: int
    _max_pixel_count = Setting(
        _("Maximum Pixel Count"),
        6,
        _("Maximum resolution to generate images at, in megapixels (FullHD ~ 2MP, 4k ~ 8MP)."),
    )

    dynamic_caching: bool
    _dynamic_caching = Setting(
        _("Dynamic Caching"),
        False,
        _("Re-use outputs of previous steps (First Block Cache) to speed up generation."),
    )

    tiled_vae: bool
    _tiled_vae = Setting(
        _("Tiled VAE"),
        False,
        _("Conserve memory by processing output images in smaller tiles."),
    )

    _performance_presets = {
        PerformancePreset.cpu: PerformancePresetSettings(
            batch_size=1,
            resolution_multiplier=1.0,
            max_pixel_count=2,
        ),
        PerformancePreset.low: PerformancePresetSettings(
            batch_size=2,
            resolution_multiplier=1.0,
            max_pixel_count=2,
            tiled_vae=True,
        ),
        PerformancePreset.medium: PerformancePresetSettings(
            batch_size=4,
            resolution_multiplier=1.0,
            max_pixel_count=6,
        ),
        PerformancePreset.high: PerformancePresetSettings(
            batch_size=6,
            resolution_multiplier=1.0,
            max_pixel_count=8,
        ),
        PerformancePreset.cloud: PerformancePresetSettings(
            batch_size=8,
            resolution_multiplier=1.0,
            max_pixel_count=6,
        ),
    }

    debug_dump_workflow: bool
    _debug_dump_workflow = Setting(
        _("Dump Workflow"),
        False,
        _("Write latest ComfyUI prompt to the log folder for test & debug"),
    )

    document_defaults: dict[str, Any]
    _document_defaults = Setting(_("Document Defaults"), {}, _("Recently used document settings"))

    # Folder where intermediate images are stored for debug purposes (default: None)
    debug_image_folder = os.environ.get("KRITA_AI_DIFFUSION_DEBUG_IMAGE")

    changed = pyqtSignal(str, object)

    _values: dict[str, Any]

    def __init__(self):
        super().__init__()
        self.restore(init=True)

    def __getattr__(self, name: str):
        if name in self._values:
            return self._values[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        if name in self._values:
            if self._values[name] != value:
                self._values[name] = value
                if name != "document_defaults":
                    self.changed.emit(name, value)
                if name == "performance_preset":
                    self.apply_performance_preset(value)
        else:
            object.__setattr__(self, name, value)

    def restore(self, init=False):
        self.__dict__["_values"] = {
            k[1:]: v.default for k, v in Settings.__dict__.items() if isinstance(v, Setting)
        }
        if not init:
            self.server_mode = ServerMode.managed

    def save(self, path: Optional[Path] = None):
        path = self.default_path or path
        with open(path, "w") as file:
            file.write(json.dumps(self._values, default=encode_json, indent=4))

    def load(self, path: Optional[Path] = None):
        path = self.default_path or path
        self._migrate_legacy_settings(path)
        if not path.exists():
            self.save()  # create new file with defaults
            return

        log.info(f"Loading settings from {path}")
        try:
            contents = read_json_with_comments(path)
            for k, v in contents.items():
                setting: Setting | None = getattr(Settings, f"_{k}", None)
                if setting is not None:
                    if isinstance(setting.default, Enum):
                        self._values[k] = setting.str_to_enum(v)
                    elif isinstance(setting.default, type(v)):
                        self._values[k] = v
                    else:
                        log.error(f"{path}: {v} is not a valid value for '{k}'")
                        self._values[k] = setting.default
        except Exception as e:
            log.error(f"Failed to load settings: {e}")

    def apply_performance_preset(self, preset: PerformancePreset):
        if preset not in [PerformancePreset.custom, PerformancePreset.auto]:
            for k, v in self._performance_presets[preset]._asdict().items():
                self._values[k] = v

    def _migrate_legacy_settings(self, path: Path):
        if path == self.default_path:
            legacy_path = Path(__file__).parent / "settings.json"
            if legacy_path.exists() and not path.exists():
                try:
                    legacy_path.rename(path)
                    log.info(f"Migrated settings from {legacy_path} to {path}")
                except Exception as e:
                    log.warning(f"Failed to migrate settings from {legacy_path} to {path}: {e}")


settings = Settings()
