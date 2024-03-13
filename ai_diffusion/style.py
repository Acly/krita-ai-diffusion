from __future__ import annotations
from enum import Enum
import json
from pathlib import Path
from typing import NamedTuple
from PyQt5.QtCore import QObject, pyqtSignal

from .settings import Setting
from .resources import SDVersion
from .util import encode_json, user_data_dir, client_logger as log

sampler_options = [
    "Euler",
    "Euler ancestral",
    "Heun",
    "Heun++2",
    "DPM 2",
    "DPM 2 ancestral",
    "LMS",
    "DPM fast",
    "DPM adaptive",
    "DPM++ 2S ancestral",
    "DPM++ SDE",
    "DPM++ SDE GPU",
    "DPM++ 2M",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE GPU",
    "DPM++ 3M SDE",
    "DPM++ 3M SDE GPU",
    "DDPM",
    "LCM",
    "DDIM",
    "UniPC",
    "UniPC BH2",
]
scheduler_options = [
    "Normal",
    "Karras",
    "Exponential",
    "Sgm_uniform",
    "Simple",
    "Ddim_uniform",
]


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

    preferred_resolution = Setting(
        "Preferred Resolution", 0, "Image resolution the checkpoint was trained on"
    )

    sampler = Setting(
        "Sampler",
        "DPM++ 2M SDE",
        "The sampler",
        items=sampler_options,
    )
    scheduler = Setting(
        "Scheduler",
        "Normal",
        "The scheduler",
        items=scheduler_options,
    )

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

    live_sampler = Setting("Sampler", "LCM", sampler.desc, items=sampler_options)
    live_scheduler = Setting("Scheduler", "Sgm_uniform", scheduler.desc, items=scheduler_options)
    live_sampler_steps = Setting("Sampler Steps", 6, sampler_steps.desc)
    live_cfg_scale = Setting("Guidance Strength (CFG Scale)", 1.8, cfg_scale.desc)


class SamplerConfig(NamedTuple):
    sampler: str
    scheduler: str
    steps: int
    cfg: float


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
    preferred_resolution: int = StyleSettings.preferred_resolution.default
    sampler: str = StyleSettings.sampler.default
    scheduler: str = StyleSettings.scheduler.default
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
        return self.filepath.name

    def get_sampler_config(self, is_live=False):
        if is_live:
            return SamplerConfig(self.live_sampler,self.scheduler, self.live_sampler_steps, self.live_cfg_scale)
        return SamplerConfig(self.sampler, self.scheduler, self.sampler_steps, self.cfg_scale)


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
        return self._list

    def find(self, filename: str):
        return next(
            ((style, i) for i, style in enumerate(self._list) if style.filename == filename),
            (None, -1),
        )

    def __getitem__(self, index) -> Style:
        return self._list[index]

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)
