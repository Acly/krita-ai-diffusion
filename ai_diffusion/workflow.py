import math
from typing import NamedTuple, Tuple, Union, Optional

from .image import Bounds, Extent, Image, ImageCollection, Mask
from .client import Client
from .settings import settings
from .style import SDVersion, Style
from .comfyworkflow import ComfyWorkflow, Output
from .util import compute_batch_size, client_logger as log


Inputs = Union[Extent, Image, Tuple[Image, Mask]]


class ScaledExtent(NamedTuple):
    initial: Extent  # resolution for initial generation
    expanded: Extent  # resolution for high res pass
    target: Extent  # target resolution (may not be multiple of 8)
    scale: float  # scale factor from target to initial

    @property
    def requires_upscale(self):
        assert self.scale == 1 or self.initial != self.expanded
        return self.scale < 1

    @property
    def requires_downscale(self):
        assert self.scale == 1 or self.initial != self.expanded
        return self.scale > 1

    @property
    def is_incompatible(self):
        assert self.target == self.expanded or not self.target.is_multiple_of(8)
        return self.target != self.expanded


class ScaledInputs(NamedTuple):
    image: Optional[Image]
    mask_image: Optional[Image]
    extent: ScaledExtent
    batch_size: int


def prepare(inputs: Inputs, sdver: SDVersion, downscale=True) -> ScaledInputs:
    input_is_masked_image = isinstance(inputs, tuple) and isinstance(inputs[0], Image)
    image = inputs[0] if input_is_masked_image else None
    image = inputs if isinstance(inputs, Image) else image
    extent = inputs if isinstance(inputs, Extent) else image.extent
    mask = inputs[1] if input_is_masked_image else None
    mask_image = mask.to_image(extent) if mask else None

    # Latent space uses an 8 times lower resolution, so results are always multiples of 8.
    # If the target image is not a multiple of 8, the result must be scaled to fit.
    expanded = extent.multiple_of(8)

    min_size, max_size, min_pixel_count, max_pixel_count = {
        SDVersion.sd1_5: (512, 768, 512**2, 512 * 768),
        SDVersion.sdxl: (896, 1280, 1024**2, 1024**2),
    }[sdver]
    min_scale = math.sqrt(min_pixel_count / extent.pixel_count)
    max_scale = math.sqrt(max_pixel_count / extent.pixel_count)

    if downscale and max_scale < 1 and any(x > max_size for x in extent):
        # Image is larger than the maximum size. Scale down to avoid repetition artifacts.
        scale = max_scale
        initial = (extent * scale).multiple_of(8)
        # Images are scaled here directly to avoid encoding and processing
        # very large images in subsequent steps.
        if image:
            image = Image.scale(image, initial)
        if mask_image:
            mask_image = Image.scale(mask_image, initial)

    elif min_scale > 1 and all(x < min_size for x in extent):
        # Image is smaller than the minimum size. Scale up to avoid clipping.
        scale = min_scale
        initial = (extent * scale).multiple_of(8)

    else:  # Image is in acceptable range.
        scale = 1.0
        initial = expanded

    batch = compute_batch_size(Extent.largest(initial, extent))
    return ScaledInputs(image, mask_image, ScaledExtent(initial, expanded, extent, scale), batch)


def _sampler_params(style: Style, clip_vision=False, upscale=False):
    sampler_name = {
        "DDIM": "ddim",
        "DPM++ 2M": "dpmpp_2m",
        "DPM++ 2M Karras": "dpmpp_2m",
        "DPM++ 2M SDE": "dpmpp_2m_sde_gpu",
        "DPM++ 2M SDE Karras": "dpmpp_2m_sde_gpu",
    }[style.sampler]
    sampler_scheduler = {
        "DDIM": "ddim_uniform",
        "DPM++ 2M": "normal",
        "DPM++ 2M Karras": "karras",
        "DPM++ 2M SDE": "normal",
        "DPM++ 2M SDE Karras": "karras",
    }[style.sampler]
    params = dict(
        sampler=sampler_name,
        scheduler=sampler_scheduler,
        steps=style.sampler_steps,
        cfg=style.cfg_scale,
    )
    if clip_vision:
        params["cfg"] = min(5, style.cfg_scale)
    if upscale:
        params["steps"] = style.sampler_steps_upscaling
    return params


def load_model_with_lora(w: ComfyWorkflow, comfy: Client, style: Style):
    checkpoint = style.sd_checkpoint
    if checkpoint not in comfy.checkpoints:
        checkpoint = comfy.checkpoints[0]
        log.warning(f"Style checkpoint {style.sd_checkpoint} not found, using default {checkpoint}")
    model, clip, vae = w.load_checkpoint(checkpoint)

    for lora in style.loras:
        if lora["name"] not in comfy.lora_models:
            log.warning(f"Style LoRA {lora['name']} not found, skipping")
            continue
        model, clip = w.load_lora(model, clip, lora["name"], lora["strength"], lora["strength"])
    return model, clip, vae


def upscale_latent(
    w: ComfyWorkflow,
    style: Style,
    latent: Output,
    target: Extent,
    prompt_pos: str,
    prompt_neg: Output,
    model: Output,
    clip: Output,
):
    assert target.is_multiple_of(8)
    upscale = w.scale_latent(latent, target)
    prompt = w.clip_text_encode(clip, f"{prompt_pos}, {style.style_prompt}")
    return w.ksampler(
        model, prompt, prompt_neg, upscale, denoise=0.5, **_sampler_params(style, upscale=True)
    )


def generate(comfy: Client, style: Style, input_extent: Extent, prompt: str):
    _, _, extent, batch = prepare(input_extent, style.sd_version_resolved)

    w = ComfyWorkflow()
    model, clip, vae = load_model_with_lora(w, comfy, style)
    latent = w.empty_latent_image(extent.initial.width, extent.initial.height, batch)
    positive = w.clip_text_encode(clip, f"{prompt}, {style.style_prompt}")
    negative = w.clip_text_encode(clip, style.negative_prompt)
    out_latent = w.ksampler(model, positive, negative, latent, **_sampler_params(style))
    if extent.requires_upscale:
        out_latent = upscale_latent(
            w, style, out_latent, extent.expanded, prompt, negative, model, clip
        )
    out_image = w.vae_decode(vae, out_latent)
    if extent.requires_downscale or extent.is_incompatible:
        out_image = w.scale_image(out_image, extent.target)
    w.send_image(out_image)
    return w


def inpaint(comfy: Client, style: Style, image: Image, mask: Mask, prompt: str):
    sd_ver = style.sd_version_resolved
    scaled_image, scaled_mask, extent, _ = prepare((image, mask), sd_ver)
    target_bounds = mask.bounds
    region_expanded = target_bounds.extent.multiple_of(8)
    expanded_bounds = Bounds(
        mask.bounds.x, mask.bounds.y, region_expanded.width, region_expanded.height
    )
    batch = compute_batch_size(Extent.largest(scaled_image.extent, region_expanded))

    w = ComfyWorkflow()
    model, clip, vae = load_model_with_lora(w, comfy, style)
    in_image = w.load_image(scaled_image)
    in_mask = w.load_mask(scaled_mask)
    cropped_mask = w.load_mask(mask.to_image())
    if extent.requires_downscale:
        in_image = w.scale_image(in_image, extent.initial)
        in_mask = w.scale_mask(in_mask, extent.initial)
    positive = w.clip_text_encode(clip, f"{prompt}, {style.style_prompt}")
    negative = w.clip_text_encode(clip, style.negative_prompt)
    if sd_ver.has_controlnet_inpaint:
        controlnet = w.load_controlnet(comfy.controlnet_model["inpaint"])
        clip_vision = w.load_clip_vision(comfy.clip_vision_model)
        model = w.ip_adapter(comfy.ip_adapter_model, model, clip_vision, in_image, 0.5)
        control_image = w.inpaint_preprocessor(in_image, in_mask)
        positive = w.apply_controlnet(positive, controlnet, control_image)
    latent = w.vae_encode_inpaint(vae, in_image, in_mask)
    latent = w.batch_latent(latent, batch)
    out_latent = w.ksampler(
        model, positive, negative, latent, **_sampler_params(style, clip_vision=True)
    )
    if extent.requires_upscale:
        latent = w.scale_latent(out_latent, extent.expanded)
        latent = w.crop_latent(latent, expanded_bounds)
        no_mask = w.solid_mask(expanded_bounds.extent, 1.0)
        latent = w.set_latent_noise_mask(latent, no_mask)
        positive_upscale = w.clip_text_encode(clip, f"{prompt}, {style.style_prompt}")
        if sd_ver.has_controlnet_inpaint:
            cropped_image = w.load_image(Image.crop(image, target_bounds))
            control_image_up = w.inpaint_preprocessor(cropped_image, cropped_mask)
            positive_upscale = w.apply_controlnet(positive_upscale, controlnet, control_image_up)
        params = _sampler_params(style, clip_vision=True, upscale=True)
        out_latent = w.ksampler(model, positive_upscale, negative, latent, denoise=0.5, **params)
    elif extent.requires_downscale:
        pass  # crop to target bounds after decode and downscale
    else:
        out_latent = w.crop_latent(out_latent, expanded_bounds)
    out_image = w.vae_decode(vae, out_latent)
    if expanded_bounds.extent != target_bounds.extent:
        out_image = w.scale_image(out_image, target_bounds.extent)
    if extent.requires_downscale:
        out_image = w.scale_image(out_image, extent.target)
        out_image = w.crop_image(out_image, target_bounds)
    out_masked = w.apply_mask(out_image, cropped_mask)
    w.send_image(out_masked)
    return w


def refine(comfy: Client, style: Style, image: Image, prompt: str, strength: float):
    assert strength > 0 and strength < 1
    image, _, extent, batch = prepare(image, style.sd_version_resolved, downscale=False)

    w = ComfyWorkflow()
    model, clip, vae = load_model_with_lora(w, comfy, style)
    in_image = w.load_image(image)
    if extent.is_incompatible:
        in_image = w.scale_image(in_image, extent.expanded)
    latent = w.vae_encode(vae, in_image)
    latent = w.batch_latent(latent, batch)
    positive = w.clip_text_encode(clip, f"{prompt}, {style.style_prompt}")
    negative = w.clip_text_encode(clip, style.negative_prompt)
    sampler = w.ksampler(
        model, positive, negative, latent, denoise=strength, **_sampler_params(style)
    )
    out_image = w.vae_decode(vae, sampler)
    if extent.is_incompatible:
        out_image = w.scale_image(out_image, extent.target)
    w.send_image(out_image)
    return w


def refine_region(
    comfy: Client, style: Style, image: Image, mask: Mask, prompt: str, strength: float
):
    assert strength > 0 and strength < 1

    downscale_if_needed = strength >= 0.7
    sd_ver = style.sd_version_resolved
    image = Image.crop(image, mask.bounds)
    image, mask_image, extent, batch = prepare((image, mask), sd_ver, downscale_if_needed)

    w = ComfyWorkflow()
    model, clip, vae = load_model_with_lora(w, comfy, style)
    in_image = w.load_image(image)
    in_mask = w.load_mask(mask_image)
    if extent.requires_downscale:
        in_image = w.scale_image(in_image, extent.initial)
        in_mask = w.scale_mask(in_mask, extent.initial)
    elif extent.is_incompatible:
        in_image = w.scale_image(in_image, extent.expanded)
        in_mask = w.scale_mask(in_mask, extent.expanded)
    latent = w.vae_encode(vae, in_image)
    latent = w.set_latent_noise_mask(latent, in_mask)
    latent = w.batch_latent(latent, batch)
    positive = w.clip_text_encode(clip, f"{prompt}, {style.style_prompt}")
    negative = w.clip_text_encode(clip, style.negative_prompt)
    if sd_ver.has_controlnet_inpaint:
        controlnet = w.load_controlnet(comfy.controlnet_model["inpaint"])
        control_image = w.inpaint_preprocessor(in_image, in_mask)
        positive = w.apply_controlnet(positive, controlnet, control_image)
    out_latent = w.ksampler(
        model, positive, negative, latent, denoise=strength, **_sampler_params(style)
    )
    if extent.requires_upscale:
        out_latent = upscale_latent(
            w, style, out_latent, extent.expanded, prompt, negative, model, clip
        )
    out_image = w.vae_decode(vae, out_latent)
    if extent.requires_downscale or extent.is_incompatible:
        out_image = w.scale_image(out_image, extent.target)
    original_mask = w.load_mask(mask.to_image())
    out_masked = w.apply_mask(out_image, original_mask)
    w.send_image(out_masked)
    return w
