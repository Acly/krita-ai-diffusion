from __future__ import annotations
from enum import Enum
from typing import NamedTuple
import json
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal

from .api import CheckpointInput, LoraInput
from .settings import Setting, settings
from .resources import SDVersion
from .util import encode_json, plugin_dir, user_data_dir, client_logger as log


class StyleSettings:
    name = Setting("Name", "Default Style")
    version = Setting("Version", 1)

    sd_version = Setting(
        "Stable Diffusion Version",
        SDVersion.auto,
        "The base architecture must match checkpoint and LoRA",
    )

    sd_checkpoint = Setting(
        "Model Checkpoint",
        "<no checkpoint set>",
        "The Stable Diffusion checkpoint file",
        "This has a large impact on which kind of content will"
        " be generated. To install additional checkpoints, place them into"
        " [ComfyUI]/models/checkpoints.",
    )

    loras = Setting(
        "LoRA",
        [],
        "Extensions to the checkpoint which expand its range based on additional training",
    )

    style_prompt = Setting(
        "Style Prompt",
        "best quality, highres",
        "Keywords which are appended to all prompts. Can be used to influence style and quality.",
    )

    negative_prompt = Setting(
        "Negative Prompt",
        "bad quality, low resolution, blurry",
        "Textual description of things to avoid in generated images",
    )

    vae = Setting(
        "VAE",
        "Checkpoint Default",
        "Model to encode and decode images. Commonly affects saturation and sharpness.",
    )

    clip_skip = Setting(
        "Clip Skip",
        0,
        "Clip layers to omit at the end. Some checkpoints prefer a different value than the"
        " default.",
    )

    v_prediction_zsnr = Setting(
        "V-Prediction / Zero Terminal SNR",
        False,
        "Enable this if the checkpoint is a v-prediction model which requires zero terminal SNR"
        " noise schedule",
    )

    self_attention_guidance = Setting(
        "Enable SAG / Self-Attention Guidance",
        False,
        'Pay more attention to "difficult" parts of the image. Can improve fine details.',
    )

    preferred_resolution = Setting(
        "Preferred Resolution", 0, "Image resolution the checkpoint was trained on"
    )

    sampler = Setting("Sampler", "Default", "The sampling strategy and scheduler")

    sampler_steps = Setting(
        "Sampler Steps",
        20,
        "Higher values can produce more refined results but take longer",
    )

    cfg_scale = Setting(
        "Guidance Strength (CFG Scale)",
        7.0,
        "Value which indicates how closely image generation follows the text prompt",
    )

    live_sampler = Setting("Sampler", "Realtime LCM", sampler.desc)
    live_sampler_steps = Setting("Sampler Steps", 6, sampler_steps.desc)
    live_cfg_scale = Setting("Guidance Strength (CFG Scale)", 1.8, cfg_scale.desc)


class Style:
    filepath: Path
    version: int = StyleSettings.version.default
    name: str = StyleSettings.name.default
    sd_version: SDVersion = StyleSettings.sd_version.default
    sd_checkpoint: str = StyleSettings.sd_checkpoint.default
    loras: list[dict[str, str | float]]
    style_prompt: str = StyleSettings.style_prompt.default
    negative_prompt: str = StyleSettings.negative_prompt.default
    vae: str = StyleSettings.vae.default
    clip_skip: int = StyleSettings.clip_skip.default
    v_prediction_zsnr: bool = StyleSettings.v_prediction_zsnr.default
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
        self.filepath.write_text(json.dumps(cfg, indent=4, default=encode_json))

    @property
    def filename(self):
        if self.filepath.is_relative_to(Styles.default_user_folder):
            return str(self.filepath.relative_to(Styles.default_user_folder).as_posix())
        return f"built-in/{self.filepath.name}"

    def get_models(self):
        result = CheckpointInput(
            checkpoint=self.sd_checkpoint,
            vae=self.vae,
            clip_skip=self.clip_skip,
            v_prediction_zsnr=self.v_prediction_zsnr,
            loras=[LoraInput.from_dict(l) for l in self.loras],
            self_attention_guidance=self.self_attention_guidance,
        )
        return result


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

    def create(self, name: str = "style", checkpoint: str = "") -> Style:
        if Path(self.user_folder / f"{name}.json").exists():
            i = 1
            basename = name
            while Path(self.user_folder / f"{basename}_{i}.json").exists():
                i += 1
            name = f"{basename}_{i}"

        new_style = Style(self.user_folder / f"{name}.json")
        new_style.name = "New Style"
        if checkpoint:
            new_style.sd_checkpoint = checkpoint
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
            presets = json.loads(file.read_text())
            presets = {name: SamplerPreset(**preset) for name, preset in presets.items()}
            self._presets.update(presets)
        except Exception as e:
            log.error(f"Failed to load sampler presets from {file}: {e}")

    def add_missing(self, name: str, steps: int, cfg_scale: float):
        if name in self._presets:
            return name
        if name in _legacy_map:
            return _legacy_map[name]
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
            self._user_preset_file.write_text(json.dumps(_sampler_presets_stub, indent=4))
        return self._user_preset_file

    def __len__(self):
        return len(self._presets)

    def __getitem__(self, name: str) -> SamplerPreset:
        if result := self._presets.get(name, None):
            return result
        if name in _legacy_map:
            return self[_legacy_map[name]]
        raise KeyError(f"Sampler preset {name} not found")

    def items(self):
        return self._presets.items()

    def names(self):
        return self._presets.keys()


_legacy_map = {
    "DPM++ 2M Karras": "Default",
    "DPM++ 2M SDE Karras": "Creative",
    "DPM++ SDE Karras": "Turbo/Lightning Merge",
    "UniPC BH2": "Fast",
    "LCM": "Realtime LCM",
    "Lightning": "Realtime Lightning",
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
_sampler_presets_stub = {
    "DPM++ 3M (Example custom sampler)": {
        "sampler": "dpmpp_3m_sde",
        "scheduler": "exponential",
        "steps": 30,
        "cfg": 7.0,
    }
}
