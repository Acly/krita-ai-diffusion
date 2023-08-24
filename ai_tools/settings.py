import os
import json
from enum import Enum
from pathlib import Path


class GPUMemoryPreset(Enum):
    custom = 0
    low = 1
    medium = 2
    high = 3

    @property
    def text(self):
        return ["Custom", "Low (less than 6GB)", "Medium (6GB to 12GB)", "High (more than 12GB)"][
            self.value
        ]


class Setting:
    def __init__(self, name: str, default, desc: str):
        self.name = name
        self.desc = desc
        self.default = default

    def str_to_enum(self, s: str):
        assert isinstance(self.default, Enum)
        EnumType = type(self.default)
        try:
            return EnumType[s]
        except KeyError:
            return self.default


def encode_json(obj):
    if isinstance(obj, Enum):
        return obj.name
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class Settings:
    default_path = Path(__file__).parent / "settings.json"

    _server_url = Setting(
        "Server URL",
        "127.0.0.1:8188",
        "URL used to connect to a running ComfyUI server. Default is 127.0.0.1:8188",
    )

    _negative_prompt = Setting(
        "Negative Prompt",
        "EasyNegative verybadimagenegative_v1.3",
        "Textual description of things to avoid in generated images.",
    )

    _upscale_prompt = Setting(
        "Upscale Prompt",
        "highres, 8k, uhd",
        "Additional text which is used to extend the prompt when upscaling images.",
    )

    _min_image_size = Setting(
        "Minimum Image Size",
        512,
        (
            "Generation will run at a resolution of at least the configured value, "
            "even if the selected input image content is smaller. "
            "Results are automatically downscaled to fit the target area."
        ),
    )

    _max_image_size = Setting(
        "Maximum Image Size",
        768,
        (
            "Initial image generation will run with a resolution no higher than the value "
            "configured here. If the resolution of the target area is higher, the results "
            "will be upscaled afterwards."
        ),
    )

    _gpu_memory_preset = Setting(
        "GPU Memory Preset",
        GPUMemoryPreset.medium,
        (
            "Controls how much GPU memory (VRAM) is used for image generation. If you encounter out"
            " of memory errors, switch to a lower setting. All functionality and resolutions should"
            " work even on the lowest setting. More memory will allow more efficient generation"
            " by using larger batches and tiles."
        ),
    )

    _batch_size = Setting(
        "Maximum Batch Size",
        4,
        (
            "Number of low resolution images which are generated at once. Improves"
            " generation efficiency but requires more GPU memory. Batch size is automatically"
            " adjusted for larger resolutions."
        ),
    )

    _vae_endoding_tile_size = Setting(
        "VAE Encoder Tile Size",
        1024,
        "Larger images are split up into tiles when passed to the VAE to allow large resolutions. ",
    )

    _diffusion_tile_size = Setting(
        "Diffusion Tile Size",
        2048,
        "Resolution threshold at which diffusion is split up into multiple tiles. ",
    )

    _gpu_memory_presets = {
        GPUMemoryPreset.low: {
            "batch_size": 2,
            "vae_endoding_tile_size": 512,
            "diffusion_tile_size": 1024,
        },
        GPUMemoryPreset.medium: {
            "batch_size": 4,
            "vae_endoding_tile_size": 1024,
            "diffusion_tile_size": 2048,
        },
        GPUMemoryPreset.high: {
            "batch_size": 8,
            "vae_endoding_tile_size": 2048,
            "diffusion_tile_size": 4096,
        },
    }

    _upscaler = Setting(
        "Upscaler",
        "Lanczos",
        "The algorithm to use whenever images need to be resized to a higher resolution.",
    )
    upscalers = ["Lanczos"]

    @property
    def upscaler_index(self):
        return self.upscalers.index(self.upscaler)

    # Folder where intermediate images are stored for debug purposes (default: None)
    debug_image_folder = os.environ.get("KRITA_AI_TOOLS_DEBUG_IMAGE")

    def __init__(self):
        self.restore()

    def __getattr__(self, name: str):
        if name in self._values:
            return self._values[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        if name in self._values:
            self._values[name] = value
            if name == "gpu_memory_preset":
                self._apply_gpu_memory_preset(value)
        else:
            object.__setattr__(self, name, value)

    def restore(self):
        self.__dict__["_values"] = {
            k[1:]: v.default for k, v in Settings.__dict__.items() if isinstance(v, Setting)
        }

    def save(self, path: Path = ...):
        path = self.default_path if path is ... else path
        with open(path, "w") as file:
            file.write(json.dumps(self._values, default=encode_json, indent=4))

    def load(self, path: Path = ...):
        path = self.default_path if path is ... else path
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

    def _apply_gpu_memory_preset(self, preset: GPUMemoryPreset):
        if preset is not GPUMemoryPreset.custom:
            for k, v in self._gpu_memory_presets[preset].items():
                self._values[k] = v


settings = Settings()
