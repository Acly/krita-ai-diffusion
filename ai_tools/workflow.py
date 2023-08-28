from typing import NamedTuple, Tuple, Union, Optional

from .image import Bounds, Extent, Image, ImageCollection, Mask
from .client import Client
from .settings import settings
from .comfyworkflow import ComfyWorkflow, Output
from .util import compute_batch_size


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


def upscale_latent(
    w: ComfyWorkflow,
    latent: Output,
    target: Extent,
    prompt_pos: str,
    prompt_neg: Output,
    model: Output,
    clip: Output,
):
    assert target.is_multiple_of(8)
    upscale = w.scale_latent(latent, target)
    prompt = w.clip_text_encode(clip, f"{prompt_pos}, {settings.upscale_prompt}")
    return w.ksampler(model, prompt, prompt_neg, upscale, denoise=0.5, steps=10)


async def generate(comfy: Client, input_extent: Extent, prompt: str):
    _, _, extent = prepare(input_extent)

    w = ComfyWorkflow()
    model, clip, vae = w.load_checkpoint("photon_v1.safetensors")
    latent = w.empty_latent_image(extent.initial.width, extent.initial.height)
    positive = w.clip_text_encode(clip, prompt)
    negative = w.clip_text_encode(clip, settings.negative_prompt)
    out_latent = w.ksampler(model, positive, negative, latent)
    extent_sampled = extent.initial
    if extent.scale < 1:  # generated image is smaller than requested -> upscale
        extent_sampled = extent.target.multiple_of(8)
        out_latent = upscale_latent(w, out_latent, extent_sampled, prompt, negative, model, clip)
    out_image = w.vae_decode(vae, out_latent)
    if extent_sampled != extent.target:
        out_image = w.scale_image(out_image, extent.target)
    w.send_image(out_image)

    return await comfy.enqueue(w)


async def inpaint(comfy: Client, image: Image, mask: Mask, prompt: str):
    image, mask_image, extent = prepare((image, mask))
    batch_size = compute_batch_size(extent.target, settings.min_image_size, settings.batch_size)

    w = ComfyWorkflow()
    model, clip, vae = w.load_checkpoint("realisticVisionV20_v20.safetensors")
    controlnet = w.load_controlnet("control_v11p_sd15_inpaint.pth")
    clip_vision_model = w.load_clip_vision("SD1.5\\pytorch_model.bin")
    in_image = w.load_image(image)
    in_mask = w.load_mask(mask_image)
    cropped_mask = w.load_mask(mask.to_image())
    if extent.scale > 1:
        in_image = w.scale_image(in_image, extent.initial)
        in_mask = w.scale_mask(in_mask, extent.initial)
    clip_vision = w.clip_vision_encode(clip_vision_model, in_image)
    ip_adapter = w.ip_adapter("ip-adapter_sd15.bin", model, clip_vision, 0.5)
    control_image = w.inpaint_preprocessor(in_image, in_mask)
    positive = w.clip_text_encode(clip, prompt)
    positive = w.apply_controlnet(positive, controlnet, control_image)
    negative = w.clip_text_encode(clip, settings.negative_prompt)
    latent = w.vae_encode_inpaint(vae, in_image, in_mask)
    latent = w.batch_latent(latent, batch_size)
    out_latent = w.ksampler(ip_adapter, positive, negative, latent, sampler="ddim", steps=15, cfg=5)
    if extent.scale < 1:
        cropped_latent = w.crop_latent(out_latent, Bounds.scale(mask.bounds, extent.scale))
        scaled_latent = w.scale_latent(cropped_latent, mask.bounds.extent)
        no_mask = w.solid_mask(mask.bounds.extent, 1.0)
        masked_latent = w.set_latent_noise_mask(scaled_latent, no_mask)
        cropped_image = w.vae_decode(vae, cropped_latent)
        control_image_upscale = w.inpaint_preprocessor(cropped_image, cropped_mask)
        positive_upscale = w.clip_text_encode(clip, f"{prompt}, {settings.upscale_prompt}")
        positive_upscale = w.apply_controlnet(positive_upscale, controlnet, control_image_upscale)
        out_latent = w.ksampler(
            ip_adapter, positive_upscale, negative, masked_latent, denoise=0.5, steps=15, cfg=5
        )
    else:
        out_latent = w.crop_latent(out_latent, mask.bounds)
    out_image = w.vae_decode(vae, out_latent)
    if extent.scale > 1:
        out_image = w.scale_image(out_image, extent.target)
    out_masked = w.apply_mask(out_image, cropped_mask)
    w.send_image(out_masked)

    return await comfy.enqueue(w)


async def refine(comfy: Client, image: Image, prompt: str, strength: float):
    assert strength > 0 and strength < 1
    image, _, extent = prepare(image, downscale=False)

    w = ComfyWorkflow()
    model, clip, vae = w.load_checkpoint("photon_v1.safetensors")
    in_image = w.load_image(image)
    if extent.initial != extent.target:
        in_image = w.scale_image(in_image, extent.initial)
    latent = w.vae_encode(vae, in_image)
    positive = w.clip_text_encode(clip, prompt)
    negative = w.clip_text_encode(clip, settings.negative_prompt)
    sampler = w.ksampler(model, positive, negative, latent, denoise=strength, steps=20)
    out_image = w.vae_decode(vae, sampler)
    if extent.initial != extent.target:
        out_image = w.scale_image(out_image, extent.target)
    w.send_image(out_image)

    return await comfy.enqueue(w)


async def refine_region(comfy: Client, image: Image, mask: Mask, prompt: str, strength: float):
    assert strength > 0 and strength < 1
    assert mask.bounds.extent.is_multiple_of(8)
    downscale_if_needed = strength >= 0.7
    image = Image.sub_region(image, mask.bounds)
    image, mask_image, extent = prepare((image, mask), downscale_if_needed)

    w = ComfyWorkflow()
    model, clip, vae = w.load_checkpoint("photon_v1.safetensors")
    in_image = w.load_image(image)
    in_mask = w.load_mask(mask_image)
    if extent.scale > 1:
        in_image = w.scale_image(in_image, extent.initial)
        in_mask = w.scale_mask(in_mask, extent.initial)
    latent = w.vae_encode(vae, in_image)
    latent = w.set_latent_noise_mask(latent, in_mask)
    controlnet = w.load_controlnet("control_v11p_sd15_inpaint.pth")
    control_image = w.inpaint_preprocessor(in_image, in_mask)
    positive = w.clip_text_encode(clip, prompt)
    positive = w.apply_controlnet(positive, controlnet, control_image)
    negative = w.clip_text_encode(clip, settings.negative_prompt)
    out_latent = w.ksampler(model, positive, negative, latent, denoise=strength, steps=20)
    if extent.scale < 1:
        out_latent = upscale_latent(w, out_latent, extent.target, prompt, negative, model, clip)
    out_image = w.vae_decode(vae, out_latent)
    if extent.scale > 1:
        out_image = w.scale_image(out_image, extent.target)
    original_mask = w.load_mask(mask.to_image())
    out_masked = w.apply_mask(out_image, original_mask)
    w.send_image(out_masked)

    return await comfy.enqueue(w)
