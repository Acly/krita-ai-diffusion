import os
import json
from pathlib import Path


class Setting:
    def __init__(self, name: str, default, desc: str):
        self.name = name
        self.desc = desc
        self.default = default


class Settings:
    default_path = Path(__file__).parent / "settings.json"

    _server_url = Setting(
        "Server URL",
        "http://127.0.0.1:7860",
        "URL used to connect to a running Automatic1111 server.",
    )

    _negative_prompt = Setting(
        "Negative prompt",
        "EasyNegative verybadimagenegative_v1.3",
        "Textual description of things to avoid in generated images.",
    )

    _upscale_prompt = Setting(
        "Upscale prompt",
        "highres, 8k, uhd",
        "Additional text which is used to extend the prompt when upscaling images.",
    )

    _min_image_size = Setting(
        "Minimum image size",
        512,
        (
            "Generation will run at a resolution of at least the configured value, "
            "even if the selected input image content is smaller. "
            "Results are automatically downscaled to fit the target area if needed."
        ),
    )

    _max_image_size = Setting(
        "Maximum image size",
        768,
        (
            "Initial image generation will run with a resolution no higher than the value "
            "configured here. If the resolution of the target area is higher, the results "
            "will be upscaled afterwards."
        ),
    )

    _batch_size = Setting(
        "Batch size",
        2,
        (
            "Number of low resolution images which are generated at once. Improves generation "
            "speed but requires more GPU memory (VRAM)."
        ),
    )

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
        else:
            object.__setattr__(self, name, value)

    def restore(self):
        self.__dict__["_values"] = {
            k[1:]: v.default for k, v in Settings.__dict__.items() if isinstance(v, Setting)
        }

    def save(self, path: Path = ...):
        path = self.default_path if path is ... else path
        with open(path, "w") as file:
            file.write(json.dumps(self._values))

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
                    if isinstance(setting.default, type(v)):
                        self._values[k] = v
                    else:
                        raise Exception(f"{v} is not a valid value for '{k}'")


settings = Settings()
