from dataclasses import Field, dataclass, field, is_dataclass, fields, MISSING
from copy import copy
from enum import Enum
from types import GenericAlias, UnionType
from typing import Any, get_args, get_origin
import math

from .image import Bounds, Extent, Image, ImageCollection, ImageFileFormat
from .resources import ControlMode, Arch
from .util import ensure, clamp


class WorkflowKind(Enum):
    generate = 0
    inpaint = 1
    refine = 2
    refine_region = 3
    upscale_simple = 4
    upscale_tiled = 5
    control_image = 6
    custom = 7


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
    hires_image: Image | None = None
    hires_mask: Image | None = None

    @staticmethod
    def from_extent(e: Extent):
        return ImageInput(ExtentInput(e, e, e, e))


@dataclass
class LoraInput:
    name: str
    strength: float
    storage_id: str = ""  # Base64-encoded SHA256 hash

    @staticmethod
    def from_dict(data: dict[str, Any]):
        return LoraInput(data["name"], data["strength"])


@dataclass
class CheckpointInput:
    checkpoint: str
    version: Arch = Arch.sd15
    vae: str = ""
    loras: list[LoraInput] = field(default_factory=list)
    clip_skip: int = 0
    v_prediction_zsnr: bool = False
    rescale_cfg: float = 0.7
    self_attention_guidance: bool = False
    dynamic_caching: bool = False
    tiled_vae: bool = False


@dataclass
class SamplingInput:
    sampler: str
    scheduler: str
    cfg_scale: float
    total_steps: int
    start_step: int = 0
    seed: int = 0

    @property
    def actual_steps(self):
        return self.total_steps - self.start_step

    @property
    def denoise_strength(self):
        return self.actual_steps / self.total_steps


@dataclass
class ControlInput:
    mode: ControlMode
    image: Image | None = None
    strength: float = 1.0
    range: tuple[float, float] = (0.0, 1.0)


@dataclass
class RegionInput:
    mask: Image
    bounds: Bounds
    positive: str
    control: list[ControlInput] = field(default_factory=list)
    loras: list[LoraInput] = field(default_factory=list)


@dataclass
class ConditioningInput:
    positive: str
    negative: str = ""
    style: str = ""
    control: list[ControlInput] = field(default_factory=list)
    regions: list[RegionInput] = field(default_factory=list)
    language: str = ""


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
    grow: int = 0
    feather: int = 0
    use_inpaint_model: bool = False
    use_condition_mask: bool = False
    use_reference: bool = False

    def clamped(self):
        params = copy(self)
        params.grow = clamp(params.grow, 0, 499)
        params.feather = clamp(params.feather, 0, 499)
        return params


@dataclass
class UpscaleInput:
    model: str = ""  # if empty do tiled refine without upscale model
    tile_overlap: int = -1


@dataclass
class CustomWorkflowInput:
    workflow: dict
    params: dict[str, Any]


@dataclass
class WorkflowInput:
    kind: WorkflowKind
    images: ImageInput | None = None
    models: CheckpointInput | None = None
    sampling: SamplingInput | None = None
    conditioning: ConditioningInput | None = None
    inpaint: InpaintParams | None = None
    crop_upscale_extent: Extent | None = None
    upscale: UpscaleInput | None = None
    control_mode: ControlMode = ControlMode.reference
    batch_count: int = 1
    nsfw_filter: float = 0.0
    custom_workflow: CustomWorkflowInput | None = None

    @property
    def extent(self):
        return ensure(self.images).extent

    @property
    def image(self):
        return ensure(ensure(self.images).initial_image)

    @property
    def upscale_factor(self):
        return self.extent.target.width / self.extent.input.width

    @staticmethod
    def from_dict(data: dict[str, Any]):
        return Deserializer.run(data)

    def to_dict(
        self, image_format: ImageFileFormat | None = ImageFileFormat.webp, max_image_size: int = 0
    ):
        _check_image_size(self, max_image_size)
        return Serializer.run(self, image_format)

    @property
    def diffusion_extent(self):
        if self.crop_upscale_extent:
            return Extent.largest(self.extent.initial, self.crop_upscale_extent)
        return self.extent.desired

    @property
    def passes_count(self):
        if self.kind is WorkflowKind.upscale_tiled:
            tile_count_w = math.ceil(self.extent.target.width / self.extent.desired.width)
            tile_count_h = math.ceil(self.extent.target.height / self.extent.desired.height)
            return 2 * max(1, tile_count_w * tile_count_h)
        return self.batch_count

    @property
    def cost(self):
        if self.kind is WorkflowKind.control_image:
            return 1
        if self.kind is WorkflowKind.upscale_simple:
            return 2

        def cost_factor(batch: int, extent: Extent, steps: int):
            return batch * extent.pixel_count * math.sqrt(extent.pixel_count) * steps

        base = _base_cost(ensure(self.models).version)
        steps = max(8, ensure(self.sampling).actual_steps)
        unit = cost_factor(2, Extent(1024, 1024), 24)
        cost = cost_factor(self.passes_count, self.diffusion_extent, steps)
        return base + round((10 * cost) / unit)


def _base_cost(arch: Arch):
    if arch is Arch.sd15:
        return 1
    if arch.is_sdxl_like:
        return 2
    if arch is Arch.flux:
        return 4
    return 1


class Serializer:
    _images: ImageCollection

    @staticmethod
    def run(work: WorkflowInput, image_format: ImageFileFormat | None = ImageFileFormat.webp):
        serializer = Serializer()
        result = serializer._object(work)
        if image_format and len(serializer._images) > 0:
            blob, offsets = serializer._images.to_bytes(image_format)
            assert blob.size() > 0, "Image data is empty"
            result["image_data"] = {"bytes": blob.data(), "offsets": offsets}
        return result

    def __init__(self):
        self._images = ImageCollection()

    def _object(self, obj) -> dict[str, Any]:
        items = (
            (field.name, self._value(getattr(obj, field.name), field.default))
            for field in fields(obj)
        )
        return {k: v for k, v in items if v is not None}

    def _value(self, value, default=None):
        if value is None:
            return None
        if isinstance(value, Image):
            self._images.append(value)
            return len(self._images) - 1
        if isinstance(value, list):
            return [self._value(v) for v in value]
        if value == default:
            return None
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, tuple):
            return list(value)
        if is_dataclass(value):
            return self._object(value)
        return value


class Deserializer:
    _images: ImageCollection

    @staticmethod
    def run(data: dict[str, Any]) -> WorkflowInput:
        if image_data := data.get("image_data"):
            blob, offsets = image_data["bytes"], image_data["offsets"]
            images = ImageCollection.from_bytes(blob, offsets)
        else:
            images = ImageCollection()
        deserializer = Deserializer(images)
        return deserializer._object(WorkflowInput, data)

    def __init__(self, images: ImageCollection):
        self._images = images

    def _object(self, type, input: dict):
        values = (self._field(field, input.get(field.name)) for field in fields(type))
        return type(*values)

    def _field(self, field: Field, value):
        if value is None and field.default is not MISSING:
            return field.default
        elif value is None and field.default_factory is not MISSING:
            return field.default_factory()
        elif value is None:
            return None
        field_type = field.type
        if isinstance(field_type, UnionType):
            field_type = get_args(field_type)[0]
        return self._value(field_type, value)

    def _value(self, cls, value):
        if is_dataclass(cls):
            return self._object(cls, value)
        elif issubclass(cls, Enum):
            return cls[value]
        elif issubclass(cls, Image):
            return self._images[value]
        elif issubclass(cls, tuple):
            return cls(*value)
        elif isinstance(cls, GenericAlias) and issubclass(get_origin(cls), tuple):
            return tuple(value)
        elif isinstance(cls, GenericAlias) and issubclass(get_origin(cls), list):
            return [self._value(get_args(cls)[0], v) for v in value]
        else:
            return value


def _check_image_size(workflow: WorkflowInput, max_image_size: int):
    if max_image_size > 0:
        e = workflow.extent
        for extent in (e.input, e.initial, e.desired, e.target, workflow.crop_upscale_extent):
            if extent and extent.longest_side > max_image_size:
                raise ValueError(
                    f"Image size {extent.width}x{extent.height} exceeds maximum of {max_image_size}"
                )
