from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .image import Bounds, Extent, Image
from .resources import ControlMode
from .util import ensure


class WorkflowKind(Enum):
    generate = 0
    inpaint = 1
    refine = 2
    refine_region = 3
    upscale_simple = 4
    upscale_tiled = 5
    control_image = 6


@dataclass
class ExtentInput:
    input: Extent  # resolution of input image and mask
    initial: Extent  # resolution for initial generation
    desired: Extent  # resolution for high res refinement pass
    target: Extent  # target resolution in canvas (may not be multiple of 8)


@dataclass
class ImageInput:
    extent: ExtentInput
    initial_image: Image | None = None
    initial_mask: Image | None = None
    hires_image: Image | None = None
    hires_mask: Image | None = None


@dataclass
class LoraInput:
    name: str
    strength: float

    @staticmethod
    def from_dict(data: dict[str, Any]):
        return LoraInput(data["name"], data["strength"])


@dataclass
class ModelInput:
    checkpoint: str
    vae: str = ""
    loras: list[LoraInput] = field(default_factory=list)
    clip_skip: int = 0
    v_prediction_zsnr: bool = False


@dataclass
class SamplingInput:
    sampler: str
    steps: int
    cfg_scale: float
    strength: float = 1.0
    seed: int = 0


@dataclass
class TextInput:
    positive: str
    negative: str = ""
    style: str = ""


@dataclass
class ControlInput:
    mode: ControlMode
    image: Image
    strength: float = 1.0
    range: tuple[float, float] = (0.0, 1.0)


class InpaintMode(Enum):
    automatic = 0
    fill = 1
    expand = 2
    add_object = 3
    remove_object = 4
    replace_background = 5
    custom = 6


class FillMode(Enum):
    none = 0
    neutral = 1
    blur = 2
    border = 3
    replace = 4
    inpaint = 5


@dataclass
class InpaintParams:
    mode: InpaintMode
    target_bounds: Bounds
    fill: FillMode = FillMode.neutral
    use_inpaint_model = False
    use_condition_mask = False
    use_reference = False


@dataclass
class WorkflowInput:
    kind: WorkflowKind
    images: ImageInput | None = None
    models: ModelInput | None = None
    sampling: SamplingInput | None = None
    text: TextInput | None = None
    control: list[ControlInput] = field(default_factory=list)
    inpaint: InpaintParams | None = None
    crop_upscale_extent: Extent | None = None
    upscale_model: str = ""
    control_mode: ControlMode = ControlMode.reference
    batch_count = 1

    @property
    def extent(self):
        return ensure(self.images).extent

    @property
    def image(self):
        return ensure(ensure(self.images).initial_image)

    @property
    def upscale_factor(self):
        return self.extent.target.width / self.extent.input.width
