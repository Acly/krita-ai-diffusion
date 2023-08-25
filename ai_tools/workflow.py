import json
import random
from pathlib import Path
from typing import NamedTuple, Tuple, Union, Optional

from .util import compute_batch_size
from .image import Bounds, Extent, Image, ImageCollection, Mask
from .client import Client
from .settings import settings


_workflows_path = Path(__file__).parent / "workflows"


def load_template(template: str):
    return json.loads((_workflows_path / template).read_text())


class Output(NamedTuple):
    node: int
    output: int


class ComfyWorkflow:
    _i = 0

    def __init__(self) -> None:
        self.root = {}

    def add(self, class_type: str, output_count: int, **inputs):
        normalize = lambda x: [str(x.node), x.output] if isinstance(x, Output) else x
        self._i += 1
        self.root[str(self._i)] = {
            "class_type": class_type,
            "inputs": {k: normalize(v) for k, v in inputs.items()},
        }
        return (
            Output(self._i, 0)
            if output_count == 1
            else tuple(Output(self._i, i) for i in range(output_count))
        )

    def ksampler(self, model, positive, negative, latent_image, steps=20, cfg=7, denoise=1):
        return self.add(
            "KSampler",
            1,
            seed=random.getrandbits(64),
            sampler_name="dpmpp_2m_sde_gpu",
            scheduler="karras",
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
        )

    def checkpoint_loader(self, checkpoint):
        return self.add("CheckpointLoaderSimple", 3, ckpt_name=checkpoint)

    def empty_latent_image(self, width, height):
        batch = compute_batch_size(
            Extent(width, height), settings.min_image_size, settings.batch_size
        )
        return self.add("EmptyLatentImage", 1, width=width, height=height, batch_size=batch)

    def clip_text_encode(self, clip, text):
        return self.add("CLIPTextEncode", 1, clip=clip, text=text)

    def vae_encode(self, vae, image):
        return self.add("VAEEncode", 1, vae=vae, pixels=image)

    def vae_decode(self, vae, latent_image):
        return self.add("VAEDecode", 1, vae=vae, samples=latent_image)

    def latent_upscale(self, latent, extent):
        return self.add(
            "LatentUpscale",
            1,
            samples=latent,
            width=extent.width,
            height=extent.height,
            upscale_method="nearest-exact",
            crop="disabled",
        )

    def scale_image(self, image, extent):
        return self.add(
            "ImageScale",
            1,
            image=image,
            width=extent.width,
            height=extent.height,
            upscale_method="bilinear",
            crop="disabled",
        )

    def crop_image(self, image, bounds: Bounds):
        return self.add(
            "ETN_CropImage",
            1,
            image=image,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def apply_mask(self, image, mask):
        return self.add("ETN_ApplyMaskToImage", 1, image=image, mask=mask)

    def load_image(self, image: Image):
        return self.add("ETN_LoadImageBase64", 1, image=image.to_base64())

    def load_mask(self, mask: Image):
        return self.add("ETN_LoadMaskBase64", 1, mask=mask.to_base64())

    def send_image(self, image):
        return self.add("ETN_SendImageWebSocket", 1, images=image)


Inputs = Union[Extent, Image, Tuple[Image, Mask]]


class ScaledExtent(NamedTuple):
    initial: Extent
    target: Extent
    scale: float


class ScaledInputs(NamedTuple):
    image: Optional[Image]
    mask_image: Optional[Image]
    extent: ScaledExtent


def prepare(inputs: Inputs, downscale=True) -> ScaledInputs:
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
        # Scale it down so the longer side is equal to max size.
        scale = max_size / max(extent.width, extent.height)
        initial = (extent * scale).multiple_of(8)
        # Images are scaled here directly to avoid encoding and processing
        # very large images in subsequent steps.
        if image:
            image = Image.scale(image, initial)
        if mask_image:
            mask_image = Image.scale(mask_image, initial)
        assert scale < 1
        return ScaledInputs(image, mask_image, ScaledExtent(initial, extent, scale))

    if extent.width < min_size and extent.height < min_size:
        # Image is smaller than min size for which diffusion generates reasonable
        # results. Compute a resolution where the shorter side is equal to min size.
        scale = min_size / min(extent.width, extent.height)
        initial = (extent * scale).multiple_of(8)

        assert initial.width >= min_size and initial.height >= min_size
        assert scale > 1
        return ScaledInputs(image, mask_image, ScaledExtent(initial, extent, scale))

    # Image is in acceptable range, only make sure it's a multiple of 8.
    return ScaledInputs(image, mask_image, ScaledExtent(extent.multiple_of(8), extent, 1.0))


# async def postprocess(
#     diffusion: Auto1111,
#     image: Image,
#     output_extent: Extent,
#     prompt: str,
#     progress: Progress,
# ):
#     if image.extent.width < output_extent.width or image.extent.height < output_extent.height:
#         # Result image resolution is lower than requested -> upscale the results.
#         return await diffusion.upscale(image, output_extent, prompt, progress)

#     if image.extent.width > output_extent.width or image.extent.height > output_extent.height:
#         # Result image resolution is too high to fit into the inpaint section -> downscale.
#         return Image.scale(image, output_extent)

#     assert image.extent == output_extent
#     return image


async def generate(comfy: Client, input_extent: Extent, prompt: str):
    _, _, extent = prepare(input_extent)

    w = ComfyWorkflow()
    model, clip, vae = w.checkpoint_loader("photon_v1.safetensors")
    latent = w.empty_latent_image(extent.initial.width, extent.initial.height)
    positive = w.clip_text_encode(clip, prompt)
    negative = w.clip_text_encode(clip, settings.negative_prompt)
    sampler = w.ksampler(model, positive, negative, latent)
    extent_sampled = extent.initial
    if extent.scale < 1:  # generated image is smaller than requested -> upscale
        extent_sampled = extent.target.multiple_of(8)
        upscale = w.latent_upscale(sampler, extent_sampled)
        positive_up = w.clip_text_encode(clip, f"{prompt}, {settings.upscale_prompt}")
        sampler = w.ksampler(model, positive_up, negative, upscale, denoise=0.5, steps=10)
    out_image = w.vae_decode(vae, sampler)
    if extent_sampled != extent.target:
        out_image = w.scale_image(out_image, extent.target)
    w.send_image(out_image)

    return await comfy.enqueue(w.root)


# async def inpaint(diffusion: Auto1111, image: Image, mask: Mask, prompt: str, progress: Progress):
#     image, mask_image, extent, progress = prepare((image, mask), progress)
#     result = await diffusion.txt2img_inpaint(image, mask_image, prompt, extent.initial, progress)

#     # Result is the whole image, continue to work only with the inpainted region
#     scaled_bounds = Bounds.scale(mask.bounds, extent.scale)
#     result = result.map(lambda img: Image.sub_region(img, scaled_bounds))

#     for img in result:
#         img = await postprocess(diffusion, img, mask.bounds.extent, prompt, progress)
#         Mask.apply(img, mask)
#         yield img


async def refine(comfy: Client, image: Image, prompt: str, strength: float):
    assert strength > 0 and strength < 1
    image, _, extent = prepare(image)

    w = ComfyWorkflow()
    model, clip, vae = w.checkpoint_loader("photon_v1.safetensors")
    in_image = w.load_image(image)
    if extent.initial != extent.target:
        in_image = w.scale_image(out_image, extent.initial)
    latent = w.vae_encode(vae, in_image)
    positive = w.clip_text_encode(clip, prompt)
    negative = w.clip_text_encode(clip, settings.negative_prompt)
    sampler = w.ksampler(model, positive, negative, latent, denoise=strength, steps=20)
    out_image = w.vae_decode(vae, sampler)
    if extent.initial != extent.target:
        out_image = w.scale_image(out_image, extent.target)
    w.send_image(out_image)

    return await comfy.enqueue(w.root)


# async def refine_region(
#     diffusion: Auto1111, image: Image, mask: Mask, prompt: str, strength: float, progress: Progress
# ):
#     assert strength > 0 and strength < 1
#     downscale_if_needed = strength >= 0.7
#     image = Image.sub_region(image, mask.bounds)
#     image, mask_image, extent, progress = prepare((image, mask), progress, downscale_if_needed)

#     result = await diffusion.img2img_inpaint(
#         image, mask_image, prompt, strength, extent.initial, progress
#     )
#     for img in result:
#         img = await postprocess(diffusion, img, extent.target, prompt, progress)
#         Mask.apply(img, mask)
#         yield img
