from __future__ import annotations
import math
from enum import Enum
from typing import NamedTuple, overload

from .api import ExtentInput, ImageInput
from .image import Bounds, Extent, Image, Point, multiple_of
from .resources import Arch
from .settings import PerformanceSettings
from .style import Style


def compute_bounds(extent: Extent, mask_bounds: Bounds | None, strength: float):
    """Compute the area of the image to use as input for diffusion (context area).

    Canvas extent: full size of the canvas
    Diffusion context: area of the image which is passed to diffusion initially
    Mask bounds: area of the mask and immediate vicinity, this is also the diffusion result size

    Mask bounds are contained within the diffusion context, which is contained within the canvas.
    In many cases diffusion context and mask bounds are the same.
    """
    if mask_bounds is not None:
        if strength == 1.0:
            # For 100% strength inpainting get additional surrounding image content for context
            context_padding = max(extent.longest_side // 16, mask_bounds.extent.average_side // 2)
            image_bounds = Bounds.pad(
                mask_bounds, context_padding, min_size=512, multiple=8, square=True
            )
            image_bounds = Bounds.clamp(image_bounds, extent)
            return image_bounds
        else:
            # For img2img inpainting (strength < 100%) only use the mask area as input
            return mask_bounds
    else:
        return Bounds(0, 0, *extent)


def compute_relative_bounds(image_bounds: Bounds, mask_bounds: Bounds):
    """Transforms bounds for mask to use offsets relative to the context area.

    image_bounds: Bounds of the context area passed to diffusion relative to the entire canvas
    mask_bounds: Bounds of the mask and immediate vicinity relative to the entire canvas
    return[0]: Bounds for inserting diffusion results into the canvas
    return[1]: Bounds for applying the mask to the diffusion context
    """
    return mask_bounds, mask_bounds.relative_to(image_bounds)


def compute_batch_size(extent: Extent, min_size: int, max_batches: int):
    desired_pixels = min_size * min_size * max_batches
    requested_pixels = extent.width * extent.height
    return max(1, min(max_batches, desired_pixels // requested_pixels))


class ScaleMode(Enum):
    none = 0
    resize = 1  # downscale, or tiny upscale, use simple scaling like bilinear
    upscale_small = 2  # upscale by small factor (<1.5)
    upscale_fast = 3  # upscale using a fast model
    upscale_quality = 4  # upscale using a quality model


class ScaledExtent(NamedTuple):
    input: Extent  # resolution of input image and mask
    initial: Extent  # resolution for initial generation
    desired: Extent  # resolution for high res refinement pass
    target: Extent  # target resolution in canvas (may not be multiple of 8)

    @staticmethod
    def no_scaling(extent: Extent):
        return ScaledExtent(extent, extent, extent, extent)

    @staticmethod
    def from_input(input: ExtentInput):
        return ScaledExtent(input.input, input.initial, input.desired, input.target)

    @property
    def as_input(self):
        return ExtentInput(*self)

    @overload
    def convert(self, extent: Extent, src: str, dst: str) -> Extent: ...

    @overload
    def convert(self, extent: Bounds, src: str, dst: str) -> Bounds: ...

    def convert(self, extent: Extent | Bounds, src: str, dst: str):
        """Converts an extent or bounds between two "resolution spaces"
        by scaling with the respective ratio."""
        src_extent: Extent = getattr(self, src)
        dst_extent: Extent = getattr(self, dst)
        scale_w = dst_extent.width / src_extent.width
        scale_h = dst_extent.height / src_extent.height
        if isinstance(extent, Extent):
            return Extent(round(extent.width * scale_w), round(extent.height * scale_h))
        else:
            return Bounds(
                round(extent.x * scale_w),
                round(extent.y * scale_h),
                round(extent.width * scale_w),
                round(extent.height * scale_h),
            )

    @property
    def initial_scaling(self):
        ratio = Extent.ratio(self.input, self.initial)
        if ratio != 1:
            return ScaleMode.resize
        else:
            return ScaleMode.none

    @property
    def refinement_scaling(self):
        ratio = Extent.ratio(self.initial, self.desired)
        if ratio < (1 / 1.5):
            return ScaleMode.upscale_quality
        elif ratio < 1:
            return ScaleMode.upscale_small
        elif ratio > 1:
            return ScaleMode.resize
        else:
            return ScaleMode.none

    @property
    def target_scaling(self):
        ratio = Extent.ratio(self.desired, self.target)
        if ratio == 1:
            return ScaleMode.none
        elif ratio < 0.9:
            return ScaleMode.upscale_fast
        else:
            return ScaleMode.resize


class CheckpointResolution(NamedTuple):
    """Preferred resolution for a SD checkpoint, typically the resolution it was trained on."""

    min_size: int
    max_size: int
    min_scale: float
    max_scale: float

    @staticmethod
    def compute(extent: Extent, arch: Arch, style: Style | None = None):
        arch = Arch.sdxl if arch.is_sdxl_like else arch
        if style is None or style.preferred_resolution == 0:
            min_size, max_size, min_pixel_count, max_pixel_count = {
                Arch.sd15: (512, 768, 512**2, 512 * 768),
                Arch.sdxl: (640, 1280, 800**2, 1024**2),
                Arch.sd3: (512, 1536, 512**2, 1536**2),
                Arch.flux: (256, 2048, 512**2, 2048**2),
            }[arch]
        else:
            range_offset = multiple_of(round(0.2 * style.preferred_resolution), 8)
            min_size = style.preferred_resolution - range_offset
            max_size = style.preferred_resolution + range_offset
            min_pixel_count = max_pixel_count = style.preferred_resolution**2
        min_scale = math.sqrt(min_pixel_count / extent.pixel_count)
        max_scale = math.sqrt(max_pixel_count / extent.pixel_count)
        return CheckpointResolution(min_size, max_size, min_scale, max_scale)


def apply_resolution_settings(extent: Extent, settings: PerformanceSettings):
    result = extent * settings.resolution_multiplier
    max_pixels = settings.max_pixel_count * 10**6
    if max_pixels > 0 and result.pixel_count > int(max_pixels * 1.05):
        result = result.scale_to_pixel_count(max_pixels)
    return result


def prepare_diffusion_input(
    extent: Extent,
    image: Image | None,
    arch: Arch,
    style: Style,
    perf: PerformanceSettings,
    downscale=True,
):
    # Take settings into account to compute the desired resolution for diffusion.
    desired = apply_resolution_settings(extent, perf)

    # The checkpoint may require a different resolution than what is requested.
    mult = 8
    if arch is Arch.flux:
        mult = 16
    if arch is Arch.sd3:
        mult = 64
    min_size, max_size, min_scale, max_scale = CheckpointResolution.compute(desired, arch, style)

    if downscale and max_scale < 1 and any(x > max_size for x in desired):
        # Desired resolution is larger than the maximum size. Do 2 passes:
        # first pass at checkpoint resolution, then upscale to desired resolution and refine.
        input = initial = (desired * max_scale).multiple_of(mult)
        desired = desired.multiple_of(mult)
        # Input images are scaled down here for the initial pass directly to avoid encoding
        # and processing large images in subsequent steps.
        image = Image.scale(image, initial) if image else None

    elif min_scale > 1 and all(x < min_size for x in desired):
        # Desired resolution is smaller than the minimum size. Do 1 pass at checkpoint resolution.
        input = extent
        scaled = desired * min_scale
        # Avoid unnecessary scaling if too small resolution is caused by resolution multiplier
        if all(x >= min_size and x <= max_size for x in extent):
            initial = desired = extent.multiple_of(mult)
        else:
            initial = desired = scaled.multiple_of(mult)

    else:  # Desired resolution is in acceptable range. Do 1 pass at desired resolution.
        input = extent
        initial = desired = desired.multiple_of(mult)

    # Scale down input images if needed due to resolution_multiplier or max_pixel_count
    if extent.pixel_count > desired.pixel_count:
        input = desired
        image = Image.scale(image, desired) if image else None

    batch = compute_batch_size(Extent.largest(initial, desired), 512, perf.batch_size)
    return ScaledExtent(input, initial, desired, extent), image, batch


def prepare_extent(
    extent: Extent, sd_ver: Arch, style: Style, perf: PerformanceSettings, downscale=True
):
    scaled, _, batch = prepare_diffusion_input(extent, None, sd_ver, style, perf, downscale)
    return ImageInput(scaled.as_input), batch


def prepare_image(
    image: Image, sd_ver: Arch, style: Style, perf: PerformanceSettings, downscale=True
):
    scaled, out_image, batch = prepare_diffusion_input(
        image.extent, image, sd_ver, style, perf, downscale
    )
    assert out_image is not None
    return ImageInput(scaled.as_input, out_image), batch


def prepare_control(image: Image, settings: PerformanceSettings):
    input = image.extent
    desired = apply_resolution_settings(input, settings)
    if input != desired:
        image = Image.scale(image, desired)
    return ImageInput(ExtentInput(desired, desired, desired, input), image)


def get_inpaint_reference(image: Image, area: Bounds):
    extent = image.extent
    area = Bounds.pad(area, 0, multiple=8)
    area = Bounds.clamp(area, extent)
    # Check for outpaint scenario where mask covers the entire left/top/bottom/right side
    # of the image. Crop away the masked area in that case.
    if area.height >= extent.height and extent.width - area.width > 224:
        offset = 0
        if area.x == 0:
            offset = area.width
        if area.x == 0 or area.x + area.width == extent.width:
            return Image.crop(image, Bounds(offset, 0, extent.width - area.width, extent.height))
    if area.width >= extent.width and extent.height - area.height > 224:
        offset = 0
        if area.y == 0:
            offset = area.height
        if area.y == 0 or area.y + area.height == extent.height:
            return Image.crop(image, Bounds(0, offset, extent.width, extent.height - area.height))
    return None


class TileLayout:
    image_extent: Extent
    tile_extent: Extent
    min_size: int
    padding: int
    blending: int
    tile_count: Extent

    def __init__(self, extent: Extent, min_tile_size: int, padding: int):
        self.image_extent = extent
        self.min_size = min_tile_size
        self.padding = padding
        self.blending = max(1, self.padding // 16) * 8 if padding > 0 else 0
        self.tile_count = (self.image_extent // (min_tile_size - 2 * self.padding)).at_least(1)

        padded = extent + (self.tile_count - Extent(1, 1)) * 2 * self.padding
        tile_extent = Extent(
            math.ceil(padded.width / self.tile_count.width),
            math.ceil(padded.height / self.tile_count.height),
        )
        self.tile_extent = tile_extent.multiple_of(8)

    @staticmethod
    def from_denoise_strength(extent: Extent, min_tile_size: int, strength: float):
        padding = round((16 + 64 * strength) / 8) * 8
        return TileLayout(extent, min_tile_size, padding)

    @property
    def total_tiles(self):
        return self.tile_count.width * self.tile_count.height

    def start(self, coord: Point):
        return Point(
            coord.x * (self.tile_extent.width - 2 * self.padding),
            coord.y * (self.tile_extent.height - 2 * self.padding),
        )

    def end(self, coord: Point):
        end = self.start(coord) + self.tile_extent
        return end.clamp(Bounds.from_extent(self.image_extent))

    def coord(self, index: int):
        # Note: this appears flipped compared to tile layout in tooling nodes, but
        # that's because torch tensor uses [H x W] layout
        return Point(index // self.tile_count.height, index % self.tile_count.height)

    def bounds(self, index: int):
        coord = self.coord(index)
        return Bounds.from_points(self.start(coord), self.end(coord))
