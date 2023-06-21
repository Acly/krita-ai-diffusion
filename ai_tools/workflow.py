from .image import Bounds, Extent, Image, ImageCollection, Mask
from .diffusion import Progress
from . import diffusion
from . import settings

async def generate(image: Image, mask: Mask, prompt: str, progress: Progress):
    mask_image = mask.to_image(image.extent)

    max_res = settings.max_inpaint_resolution
    if image.width > max_res or image.height > max_res:
        original_width = image.width
        image = Image.downscale(image, max_res)
        mask_image = Image.downscale(mask_image, max_res)
        scale = image.width / original_width
        progress = Progress.forward(progress, 1 / (1 + settings.batch_size))
        assert scale < 1
    else:
        scale = 1

    result = await diffusion.txt2img_inpaint(image, mask_image, prompt, progress)
    result.debug_save('diffusion_generate_result')
    # Result is the whole image, but we really only need the inpainted region
    result = result.map(lambda img: Image.sub_region(img, Bounds.scale(mask.bounds, scale)))

    # If we downscaled the original image before, upscale now, but only
    # the inpainted region (rest of the image hasn't changed)
    if scale < 1:
        result = ImageCollection([
            await diffusion.upscale(img, mask.bounds.extent, prompt, progress)
            for img in result])

    result.debug_save('diffusion_generate_upscale_result')
    # Set alpha in the result image according to the inpaint mask
    result.each(lambda img: Mask.apply(img, mask))
    result.debug_save('diffusion_generate_masked_result')
    return result


async def refine(image: Image, mask: Mask, prompt: str, strength: float, progress: Progress):
    assert strength > 0 and strength < 1

    image = Image.sub_region(image, mask.bounds)
    mask_image = mask.to_image()
    max_res = settings.max_inpaint_resolution
    if image.width > max_res or image.height > max_res:
        original_width = image.width
        image = Image.downscale(image, max_res)
        mask_image = Image.downscale(mask_image, max_res)
        scale = image.width / original_width
        progress = Progress.forward(progress, 1 / (1 + settings.batch_size))
        assert scale < 1
    else:
        scale = 1

    image.debug_save('diffusion_refine_input')
    mask_image.debug_save('diffusion_refine_input_mask')

    result = await diffusion.img2img_inpaint(image, mask_image, prompt, strength, progress)
    result.debug_save('diffusion_refine_result')

    if scale < 1:
        result = ImageCollection([
            await diffusion.upscale(img, mask.bounds.extent, prompt, progress)
            for img in result])
    result.debug_save('diffusion_generate_upscale_result')

    result.each(lambda img: Mask.apply(img, mask))
    return result
