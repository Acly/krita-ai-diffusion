from __future__ import annotations
from copy import copy
from enum import Enum
from typing import Iterable, NamedTuple
import json
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal

from .api import CheckpointInput, LoraInput
from .settings import Setting, settings
from .resources import Arch
from .localization import translate as _
from .util import encode_json, find_unused_path, read_json_with_comments
from .util import plugin_dir, user_data_dir, client_logger as log


class StyleSettings:
    name = Setting(_("Name"), _("Default Style"))
    version = Setting("Version", 2)

    architecture = Setting(
        _("Diffusion Architecture"),
        Arch.auto,
        _("The base model ecosystem which the selected checkpoint belongs to."),
        items=[Arch.auto] + Arch.list(),
    )

    checkpoints = Setting(
        _("Model Checkpoint"),
        [],
        _("The Diffusion model checkpoint file"),
        _(
            "This has a large impact on which kind of content will be generated. To install additional checkpoints, place them into [ComfyUI]/models/checkpoints."
        ),
    )

    loras = Setting(
        _("LoRA"),
        [],
        _("Extensions to the checkpoint which expand its range based on additional training"),
    )

    style_prompt = Setting(
        _("Style Prompt"),
        "best quality, highres",
        _(
            "Text which is appended to all prompts. The {prompt} placeholder can be used to wrap prompts."
        ),
    )

    negative_prompt = Setting(
        _("Negative Prompt"),
        "bad quality, low resolution, blurry",
        _("Textual description of things to avoid in generated images."),
    )

    vae = Setting(
        _("VAE"),
        "Checkpoint Default",
        _("Model to encode and decode images. Commonly affects saturation and sharpness."),
    )

    clip_skip = Setting(
        _("Clip Skip"),
        0,
        _(
            "Clip layers to omit at the end. Some checkpoints prefer a different value than the default."
        ),
    )

    v_prediction_zsnr = Setting(
        _("V-Prediction / Zero Terminal SNR"),
        False,
        _(
            "Enable this if the checkpoint is a v-prediction model which requires zero terminal SNR noise schedule"
        ),
    )

    rescale_cfg = Setting("Rescale CFG", 0.7)

    self_attention_guidance = Setting(
        _("Enable SAG / Self-Attention Guidance"),
        False,
        _("Pay more attention to difficult parts of the image. Can improve fine details."),
    )

    preferred_resolution = Setting(
        _("Preferred Resolution"), 0, _("Image resolution the checkpoint was trained on")
    )

    sampler = Setting(_("Sampler"), "Default - DPM++ 2M", _("The sampling strategy and scheduler"))

    sampler_steps = Setting(
        _("Sampler Steps"),
        20,
        _("Higher values can produce more refined results but take longer"),
    )

    cfg_scale = Setting(
        _("Guidance Strength (CFG Scale)"),
        7.0,
        _("Value which indicates how closely image generation follows the text prompt"),
    )

    live_sampler = Setting(_("Sampler"), "Realtime - Hyper", sampler.desc)
    live_sampler_steps = Setting(_("Sampler Steps"), 6, sampler_steps.desc)
    live_cfg_scale = Setting(_("Guidance Strength (CFG Scale)"), 1.8, cfg_scale.desc)


class Style:
    filepath: Path
    version: int = StyleSettings.version.default
    name: str = StyleSettings.name.default
    architecture: Arch = StyleSettings.architecture.default
    checkpoints: list[str] = StyleSettings.checkpoints.default
    loras: list[dict[str, str | float | bool]]
    style_prompt: str = StyleSettings.style_prompt.default
    negative_prompt: str = StyleSettings.negative_prompt.default
    vae: str = StyleSettings.vae.default
    clip_skip: int = StyleSettings.clip_skip.default
    v_prediction_zsnr: bool = StyleSettings.v_prediction_zsnr.default
    rescale_cfg: float = StyleSettings.rescale_cfg.default
    self_attention_guidance: bool = StyleSettings.self_attention_guidance.default
    preferred_resolution: int = StyleSettings.preferred_resolution.default
    sampler: str = StyleSettings.sampler.default
    sampler_steps: int = StyleSettings.sampler_steps.default
    cfg_scale: float = StyleSettings.cfg_scale.default
    live_sampler: str = StyleSettings.live_sampler.default
    live_sampler_steps: int = StyleSettings.live_sampler_steps.default
    live_cfg_scale: float = StyleSettings.live_cfg_scale.default

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.loras = []

    @staticmethod
    def load(filepath: Path):
        numtype = (int, float)
        try:
            cfg = json.loads(filepath.read_text())
            style = Style(filepath)
            for name, setting in StyleSettings.__dict__.items():
                if isinstance(setting, Setting):
                    value = cfg.get(name, setting.default)
                    if isinstance(setting.default, Enum):
                        try:
                            value = type(setting.default)[value]
                        except KeyError:
                            pass  # handled below
                    if (
                        (setting.items is not None and value not in setting.items)
                        or (isinstance(setting.default, Enum) != isinstance(value, Enum))
                        or (isinstance(setting.default, str) != isinstance(value, str))
                        or (isinstance(setting.default, numtype) != isinstance(value, numtype))
                    ):
                        log.warning(f"Style {filepath} has invalid value for {name}: {value}")
                        value = setting.default
                    setattr(style, name, value)

            style.sampler = _map_sampler_preset(
                filepath, style.sampler, style.sampler_steps, style.cfg_scale
            )
            style.live_sampler = _map_sampler_preset(
                filepath, style.live_sampler, style.live_sampler_steps, style.live_cfg_scale
            )
            if "sd_checkpoint" in cfg:
                style.checkpoints = [cfg["sd_checkpoint"]]
            return style
        except json.JSONDecodeError as e:
            log.warning(f"Failed to load style {filepath}: {e}")
            return None

    def save(self):
        cfg = {
            name: getattr(self, name)
            for name, setting in StyleSettings.__dict__.items()
            if isinstance(setting, Setting)
        }
        cfg["version"] = StyleSettings.version.default
        self.filepath.write_text(json.dumps(cfg, indent=4, default=encode_json))

    @property
    def filename(self):
        if self.filepath.is_relative_to(Styles.default_user_folder):
            return str(self.filepath.relative_to(Styles.default_user_folder).as_posix())
        if self.filepath.is_relative_to(Styles.default_builtin_folder):
            return f"built-in/{self.filepath.name}"
        return self.filepath.name

    def preferred_checkpoint(self, available_checkpoints: Iterable[str]):
        return next((c for c in self.checkpoints if c in available_checkpoints), "not-found")

    def get_models(self, available_checkpoints: Iterable[str]):
        result = CheckpointInput(
            checkpoint=self.preferred_checkpoint(available_checkpoints),
            vae=self.vae,
            clip_skip=self.clip_skip,
            v_prediction_zsnr=self.v_prediction_zsnr,
            rescale_cfg=self.rescale_cfg,
            loras=[LoraInput.from_dict(l) for l in self.loras if l.get("enabled", True)],
            self_attention_guidance=self.self_attention_guidance,
        )
        return result

    def get_steps(self, is_live: bool) -> tuple[int, int]:
        sampler_name = self.live_sampler if is_live else self.sampler
        preset = SamplerPresets.instance()[sampler_name]
        max_steps = self.live_sampler_steps if is_live else self.sampler_steps
        max_steps = max_steps or preset.steps
        min_steps = min(preset.minimum_steps, max_steps)
        return min_steps, max_steps


def _map_sampler_preset(filepath: str | Path, name: str, steps: int, cfg: float):
    sampler_preset = SamplerPresets.instance().add_missing(name, steps, cfg)
    if sampler_preset is not None:
        return sampler_preset
    else:
        log.warning(f"Style {filepath} has invalid sampler preset {name}")
        return StyleSettings.sampler.default


class Styles(QObject):
    default_builtin_folder = Path(__file__).parent / "styles"
    default_user_folder = user_data_dir / "styles"

    _instance = None

    builtin_folder: Path
    user_folder: Path

    changed = pyqtSignal()
    name_changed = pyqtSignal()

    _list: list[Style]

    @classmethod
    def list(cls):
        if cls._instance is None:
            cls._instance = Styles(cls.default_builtin_folder, cls.default_user_folder)
        return cls._instance

    def __init__(self, builtin_folder: Path, user_folder: Path):
        super().__init__()
        self.builtin_folder = builtin_folder
        self.user_folder = user_folder
        self.user_folder.mkdir(exist_ok=True)
        self.reload()
        settings.changed.connect(self._handle_settings_change)

    @property
    def default(self):
        return self[0]

    def create(self, filename="style.json", checkpoint: str = "", copy_from: Style | None = None):
        filename = Path(filename).name
        path = find_unused_path(self.user_folder / filename)
        new_style = Style(path)
        new_style.name = _("New Style")
        if checkpoint:
            new_style.checkpoints = [checkpoint]
        if copy_from:
            for name, setting in StyleSettings.__dict__.items():
                if isinstance(setting, Setting):
                    setattr(new_style, name, copy(getattr(copy_from, name)))
            new_style.name = f"{copy_from.name} (Copy)"
        self._list.append(new_style)
        new_style.save()
        self.changed.emit()
        return new_style

    def delete(self, style: Style):
        self._list.remove(style)
        style.filepath.unlink()
        self.changed.emit()

    def find_style_files(self):
        for folder in (self.builtin_folder, self.user_folder):
            for file in folder.rglob("*.json"):
                yield file

    def reload(self):
        styles = (Style.load(f) for f in self.find_style_files())
        self._list = [s for s in styles if s is not None]
        self._list.sort(key=lambda s: s.name)
        if len(self._list) == 0:
            self.create("default")
        else:
            self.changed.emit()

    def find(self, filename: str):
        return next((style for style in self._list if style.filename == filename), None)

    def filtered(self, show_builtin: bool | None = None):
        if show_builtin is None:
            show_builtin = settings.show_builtin_styles
        return [s for s in self._list if show_builtin or not self.is_builtin(s)]

    def is_builtin(self, style: Style):
        return style.filepath.is_relative_to(self.builtin_folder)

    def _handle_settings_change(self, name: str, value):
        if name == "show_builtin_styles":
            self.changed.emit()

    def __getitem__(self, index) -> Style:
        return self._list[index]

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)


class SamplerPreset(NamedTuple):
    sampler: str
    scheduler: str
    steps: int
    cfg: float
    lora: str | None = None
    minimum_steps: int = 4
    hidden: bool = False


class SamplerPresets:
    default_preset_file = plugin_dir / "presets" / "samplers.json"
    default_user_preset_file = user_data_dir / "presets" / "samplers.json"

    _preset_file: Path
    _user_preset_file: Path
    _presets: dict[str, SamplerPreset]

    _instance: SamplerPresets | None = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = SamplerPresets()
        return cls._instance

    def __init__(self, preset_file: Path | None = None, user_preset_file: Path | None = None):
        self._preset_file = preset_file or self.default_preset_file
        self._user_preset_file = user_preset_file or self.default_user_preset_file
        self._presets = {}
        self.load(self._preset_file)
        if self._user_preset_file.exists():
            self.load(self._user_preset_file)

        if len(self._presets) == 0:
            log.warning(
                f"No sampler presets found in {self._preset_file} or {self._user_preset_file}"
            )
            self._presets["Default"] = SamplerPreset("dpmpp_2m", "karras", 20, 7.0)

    def load(self, file: Path):
        try:
            presets = read_json_with_comments(file)
            presets = {name: SamplerPreset(**preset) for name, preset in presets.items()}
            self._presets.update(presets)
            log.info(f"Loaded {len(presets)} sampler presets from {file}")
        except Exception as e:
            log.error(f"Failed to load sampler presets from {file}: {e}")

    def add_missing(self, name: str, steps: int, cfg_scale: float):
        if name in self._presets:
            return name
        if name in legacy_map:
            return legacy_map[name]
        if name in _sampler_map:
            self._presets[name] = SamplerPreset(
                sampler=_sampler_map[name],
                scheduler=_scheduler_map[name],
                steps=steps,
                cfg=cfg_scale,
            )
            return name
        return None

    def write_stub(self):
        if not self._user_preset_file.exists():
            self._user_preset_file.parent.mkdir(parents=True, exist_ok=True)
            self._user_preset_file.write_text(_sampler_presets_stub)
        return self._user_preset_file

    def __len__(self):
        return len(self._presets)

    def __getitem__(self, name: str) -> SamplerPreset:
        if result := self._presets.get(name, None):
            return result
        if name in legacy_map:
            return self[legacy_map[name]]
        raise KeyError(f"Sampler preset {name} not found")

    def items(self):
        return self._presets.items()

    def names(self):
        def is_visible(name: str, preset: SamplerPreset):
            return not preset.hidden or any(
                s.live_sampler == name or s.sampler == name for s in Styles.list()
            )

        return [name for name, preset in self._presets.items() if is_visible(name, preset)]


legacy_map = {
    "DPM++ 2M Karras": "Default - DPM++ 2M",
    "DPM++ 2M SDE Karras": "Creative - DPM++ 2M SDE",
    "Euler a": "Alternative - Euler A",
    "DPM++ SDE Karras": "Turbo/Lightning Merge - DPM++ SDE",
    "Lightning": "Lightning Merge - Euler A Uniform",
    "UniPC BH2": "Fast - UniPC BH2",
    "LCM": "Realtime - LCM",
    "Default": "Default - DPM++ 2M",
    "Creative": "Creative - DPM++ 2M SDE",
    "Turbo/Lightning Merge": "Turbo/Lightning Merge - DPM++ SDE",
    "Fast": "Fast - UniPC BH2",
    "Realtime LCM": "Realtime - LCM",
    "Realtime Lightning": "Realtime - Lightning",
}
_sampler_map = {
    "DDIM": "ddim",
    "DPM++ 2M": "dpmpp_2m",
    "DPM++ 2M Karras": "dpmpp_2m",
    "DPM++ 2M SDE": "dpmpp_2m_sde_gpu",
    "DPM++ 2M SDE Karras": "dpmpp_2m_sde_gpu",
    "DPM++ SDE Karras": "dpmpp_sde_gpu",
    "UniPC BH2": "uni_pc_bh2",
    "LCM": "lcm",
    "Lightning": "euler",
    "Euler": "euler",
    "Euler a": "euler_ancestral",
}
_scheduler_map = {
    "DDIM": "ddim_uniform",
    "DPM++ 2M": "normal",
    "DPM++ 2M Karras": "karras",
    "DPM++ 2M SDE": "normal",
    "DPM++ 2M SDE Karras": "karras",
    "DPM++ SDE Karras": "karras",
    "UniPC BH2": "ddim_uniform",
    "LCM": "sgm_uniform",
    "Lightning": "sgm_uniform",
    "Euler": "normal",
    "Euler a": "normal",
}
_sampler_presets_stub = """// Custom sampler presets - add your own sampler presets here!
// https://docs.interstice.cloud/samplers
//
// *** You have to restart Krita for the changes to take effect! ***
{
    "My Custom Sampler - DPM++ 3M": {
        "sampler": "dpmpp_3m_sde",
        "scheduler": "exponential",
        "steps": 20,
        "minimum_steps": 4,
        "cfg": 7.0
    }
}
"""
