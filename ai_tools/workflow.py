from .image import Bounds, Extent, Image, Mask
from . import diffusion
from . import settings

def inpaint(image: Image, mask: Mask, prompt: str, cb):
    mask_image = mask.to_image(image.extent)

    max_res = settings.max_inpaint_resolution
    if image.width > max_res or image.height > max_res:
        original_width = image.width
        image = Image.downscale(image, max_res)
        mask_image = Image.downscale(mask_image, max_res)
        scale = image.width / original_width
        assert scale < 1
    else:
        scale = 1

    def inpaint_cb(result):
        result.debug_save('diffusion_inpaint_result')
        # Result is the whole image, but we really only need the inpainted region
        result = Image.sub_region(result, Bounds.scale(mask.bounds, scale))

        def upscale_cb(result):
            result.debug_save('diffusion_upscale_result')
            # Set alpha in the result image according to the inpaint mask
            Mask.apply(result, mask)
            result.debug_save('diffusion_masked_result')
            cb(result)
        
        # If we downscaled the original image before, upscale now, but only
        # the inpainted region (rest of the image hasn't changed)
        if scale < 1:
            diffusion.upscale(result, mask.bounds.extent, upscale_cb)
        else:
            upscale_cb(result)

    diffusion.inpaint(image, mask_image, prompt, inpaint_cb)
