from typing import NamedTuple
from .image import Bounds, Extent, Image, ImageCollection, Mask
from .diffusion import Progress
from . import diffusion
from . import settings


class ScaledInputs(NamedTuple):
    image: Image
    mask: Mask
    scale: float
    target_extent: Extent
    progress: Progress


def prepare(image: Image, mask: Image, progress: Progress):
    assert image.extent == mask.extent
    min_size = settings.min_image_size
    max_size = settings.max_image_size

    if image.width > max_size or image.height > max_size:
        # Image is larger than max size that diffusion can comfortably handle:
        # Scale it so the largest side is equal to max size.
        # Images are scaled here directly to avoid encoding and processing
        # very large images in subsequent steps.
        original_width = image.width
        image = Image.scale(image, max_size)
        mask = Image.scale(mask, max_size)
        scale = image.width / original_width
        # Adjust progress for upscaling steps required to bring the image back
        # to the requested resolution
        progress = Progress.forward(progress, 1 / (1 + settings.batch_size))
        assert scale < 1
        return ScaledInputs(image, mask, scale, image.extent, progress)

    if image.width < min_size and image.height < min_size:
        # Image is smaller than min size for which diffusion generates reasonable
        # results. Compute a  resolution where the largest side is equal to min size.
        # Images are not scaled here, but instead the requested target resolution
        # is passed along to defer scaling as long as possible in the pipeline.
        scale = min_size / min(image.width, image.height)
        target = Extent(round(image.width * scale), round(image.height * scale))
        assert target.width >= min_size and target.height >= min_size
        assert scale > 1
        return ScaledInputs(image, mask, scale, target, progress)

    # Image is in acceptable range, don't do anything
    return ScaledInputs(image, mask, 1.0, image.extent, progress)


async def postprocess(
    result: ImageCollection, output_extent: Extent, prompt: str, progress: Progress
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


async def generate(image: Image, mask: Mask, prompt: str, progress: Progress):
    mask_image = mask.to_image(image.extent)
    input = prepare(image, mask_image, progress)

    result = await diffusion.txt2img_inpaint(
        input.image, input.mask, prompt, input.target_extent, progress
    )
    result.debug_save("diffusion_generate_result")

    # Result is the whole image, continue to work only with the inpainted region
    scaled_bounds = Bounds.scale(mask.bounds, input.scale)
    result = result.map(lambda img: Image.sub_region(img, scaled_bounds))

    result = await postprocess(result, mask.bounds.extent, prompt, progress)
    result.debug_save("diffusion_generate_post_result")

    result.each(lambda img: Mask.apply(img, mask))
    return result


async def refine(image: Image, mask: Mask, prompt: str, strength: float, progress: Progress):
    assert strength > 0 and strength < 1

    image = Image.sub_region(image, mask.bounds)
    input = prepare(image, mask.to_image(), progress)
    input.image.debug_save("diffusion_refine_input")
    input.mask.debug_save("diffusion_refine_input_mask")

    result = await diffusion.img2img_inpaint(
        input.image, input.mask, prompt, strength, input.target_extent, progress
    )
    result.debug_save("diffusion_refine_result")

    result = await postprocess(result, mask.bounds.extent, prompt, progress)
    result.debug_save("diffusion_refine_post_result")

    result.each(lambda img: Mask.apply(img, mask))
    return result
