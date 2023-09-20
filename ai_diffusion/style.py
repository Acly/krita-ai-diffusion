from enum import Enum
import json
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal

from . import Setting, settings, util
from .util import client_logger as log


class SDVersion(Enum):
    auto = "Automatic"
    sd1_5 = "SD 1.5"
    sdxl = "SD XL"

    def matches(self, checkpoint: str):
        # Check is not very robust, but don't know better alternative
        xl_in_name = "xl" in checkpoint.lower()
        return self is SDVersion.auto or ((self is SDVersion.sdxl) == xl_in_name)

    def resolve(self, checkpoint: str):
        if self is SDVersion.auto:
            return SDVersion.sdxl if SDVersion.sdxl.matches(checkpoint) else SDVersion.sd1_5
        return self

    @property
    def has_controlnet_inpaint(self):
        return self is SDVersion.sd1_5

    @property
    def has_ip_adapter(self):
        return self is SDVersion.sd1_5


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
        (
            "This has a large impact on which kind of content will"
            " be generated. To install additional checkpoints, place them into"
            " [ComfyUI]/models/checkpoints."
        ),
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

    sampler = Setting(
        "Sampler",
        "DPM++ 2M Karras",
        "The sampling strategy and scheduler",
        items=["DDIM", "DPM++ 2M", "DPM++ 2M Karras", "DPM++ 2M SDE", "DPM++ 2M SDE Karras"],
    )

    sampler_steps = Setting(
        "Sampler Steps",
        20,
        "Higher values can produce more refined results but take longer",
    )

    sampler_steps_upscaling = Setting(
        "Sampler Steps (Upscaling)",
        15,
        "Additional sampling steps to run when automatically upscaling images",
    )

    cfg_scale = Setting(
        "Guidance Strength (CFG Scale)",
        7.0,
        "Value which indicates how closely image generation follows the text prompt",
    )


class Style:
    filepath: Path
    version = StyleSettings.version.default
    name = StyleSettings.name.default
    sd_version: SDVersion = StyleSettings.sd_version.default
    sd_checkpoint = StyleSettings.sd_checkpoint.default
    loras: list
    style_prompt = StyleSettings.style_prompt.default
    negative_prompt = StyleSettings.negative_prompt.default
    vae = StyleSettings.vae.default
    sampler = StyleSettings.sampler.default
    sampler_steps = StyleSettings.sampler_steps.default
    sampler_steps_upscaling = StyleSettings.sampler_steps_upscaling.default
    cfg_scale = StyleSettings.cfg_scale.default

    _list = []

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
        self.filepath.write_text(json.dumps(cfg, indent=4, default=util.encode_json))

    @property
    def filename(self):
        return self.filepath.name

    @property
    def sd_version_resolved(self):
        return self.sd_version.resolve(self.sd_checkpoint)


class Styles(QObject):
    default_folder = Path(__file__).parent / "styles"
    _instance = None

    folder: Path

    changed = pyqtSignal()
    name_changed = pyqtSignal()

    _list: list = None

    @classmethod
    def list(Class):
        if Class._instance is None:
            Class._instance = Styles()
        return Class._instance

    def __init__(self, folder: Path = default_folder):
        super().__init__()
        self.folder = folder
        self.reload()

    @property
    def default(self):
        return self[0]
    
    def create(self, name: str = "style") -> Style:
        if Path(self.folder / f"{name}.json").exists():
            i = 1
            basename = name
            while Path(self.folder / f"{basename}_{i}.json").exists():
                i += 1
            name = f"{basename}_{i}"
        
        new_style = Style(self.folder / f"{name}.json")
        new_style.name = name
        self._list.append(new_style)
        new_style.save()
        self.changed.emit()
        return new_style
    
    def delete(self, style: Style):
        self._list.remove(style)
        style.filepath.unlink()
        self.changed.emit()

    def reload(self):
        styles = (Style.load(f) for f in self.folder.iterdir() if f.suffix == ".json")
        self._list = [s for s in styles if s is not None]
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
