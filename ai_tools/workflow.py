from typing import NamedTuple, Tuple, Union, Optional
from .image import Bounds, Extent, Image, ImageCollection, Mask
from .diffusion import Progress, Auto1111
from .settings import settings

Inputs = Union[Extent, Image, Tuple[Image, Mask]]


class ScaledExtent(NamedTuple):
    initial: Extent
    target: Extent
    scale: float


class ScaledInputs(NamedTuple):
    image: Optional[Image]
    mask_image: Optional[Image]
    extent: ScaledExtent
    progress: Progress


def prepare(inputs: Inputs, progress: Progress, downscale=True) -> ScaledInputs:
    input_is_masked_image = isinstance(inputs, tuple) and isinstance(inputs[0], Image)
    image = inputs[0] if input_is_masked_image else None
    image = inputs if isinstance(inputs, Image) else image
    extent = inputs if isinstance(inputs, Extent) else image.extent
    mask = inputs[1] if input_is_masked_image else None
    mask_image = mask.to_image(extent) if mask else None

    min_size = settings.min_image_size
    max_size = settings.max_image_size

    if downscale and (extent.width > max_size or extent.height > max_size):
        # Image is larger than max size that diffusion can comfortably handle:
        # Scale it down so the largest side is equal to max size.
        scale = max_size / max(extent.width, extent.height)
        initial = extent * scale
        # Images are scaled here directly to avoid encoding and processing
        # very large images in subsequent steps.
        if image:
            image = Image.scale(image, initial)
        if mask_image:
            mask_image = Image.scale(mask_image, initial)
        # Adjust progress for upscaling steps required to bring the image back
        # to the requested resolution (in postprocess)
        progress = Progress.forward(progress, 1 / (1 + settings.batch_size))
        assert scale < 1
        return ScaledInputs(image, mask_image, ScaledExtent(initial, extent, scale), progress)

    if extent.width < min_size and extent.height < min_size:
        # Image is smaller than min size for which diffusion generates reasonable
        # results. Compute a resolution where the largest side is equal to min size.
        scale = min_size / min(extent.width, extent.height)
        initial = extent * scale
        # Images are not scaled here, but instead the requested target resolution
        # is passed along to defer scaling as long as possible in the pipeline.
        assert initial.width >= min_size and initial.height >= min_size
        assert scale > 1
        return ScaledInputs(image, mask_image, ScaledExtent(initial, extent, scale), progress)

    # Image is in acceptable range, don't do anything
    return ScaledInputs(image, mask_image, ScaledExtent(extent, extent, 1.0), progress)


async def postprocess(
    diffusion: Auto1111,
    result: ImageCollection,
    output_extent: Extent,
    prompt: str,
    progress: Progress,
):
    input_extent = result[0].extent
    if input_extent.width < output_extent.width or input_extent.height < output_extent.height:
        # Result image resolution is lower than requested -> upscale the results.
        return ImageCollection(
            [await diffusion.upscale(img, output_extent, prompt, progress) for img in result]
        )

    if input_extent.width > output_extent.width or input_extent.height > output_extent.height:
        # Result image resolution is too high to fit into the inpaint section -> downscale.
        return ImageCollection([Image.scale(img, output_extent) for img in result])

    assert input_extent == output_extent
    return result


async def generate(diffusion: Auto1111, extent: Extent, prompt: str, progress: Progress):
    _, _, extent, progress = prepare(extent, progress)
    result = await diffusion.txt2img(prompt, extent.initial, progress)
    result = await postprocess(diffusion, result, extent.target, prompt, progress)
    return result


async def inpaint(diffusion: Auto1111, image: Image, mask: Mask, prompt: str, progress: Progress):
    image, mask_image, extent, progress = prepare((image, mask), progress)
    result = await diffusion.txt2img_inpaint(image, mask_image, prompt, extent.initial, progress)

    # Result is the whole image, continue to work only with the inpainted region
    scaled_bounds = Bounds.scale(mask.bounds, extent.scale)
    result = result.map(lambda img: Image.sub_region(img, scaled_bounds))

    result = await postprocess(diffusion, result, mask.bounds.extent, prompt, progress)
    result.each(lambda img: Mask.apply(img, mask))
    return result


async def refine(
    diffusion: Auto1111, image: Image, prompt: str, strength: float, progress: Progress
):
    assert strength > 0 and strength < 1
    downscale_if_needed = strength >= 0.7
    image, _, extent, progress = prepare(image, progress, downscale_if_needed)

    result = await diffusion.img2img(image, prompt, strength, extent.initial, progress)
    result = await postprocess(diffusion, result, extent.target, prompt, progress)
    return result


async def refine_region(
    diffusion: Auto1111, image: Image, mask: Mask, prompt: str, strength: float, progress: Progress
):
    assert strength > 0 and strength < 1
    downscale_if_needed = strength >= 0.7
    image = Image.sub_region(image, mask.bounds)
    image, mask_image, extent, progress = prepare((image, mask), progress, downscale_if_needed)

    result = await diffusion.img2img_inpaint(
        image, mask_image, prompt, strength, extent.initial, progress
    )
    result = await postprocess(diffusion, result, extent.target, prompt, progress)
    result.each(lambda img: Mask.apply(img, mask))
    return result
