import math
from typing import List, NamedTuple, Tuple, Union, Optional

from .image import Bounds, Extent, Image, ImageCollection, Mask
from .client import Client
from .style import SDVersion, Style, StyleSettings
from .resources import ControlMode
from .settings import settings
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


def compute_bounds(extent: Extent, mask_bounds: Optional[Bounds], strength: float):
    """Compute the area of the image to use as input for diffusion."""

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
    if settings.fixed_seed:
        try:
            params["seed"] = int(settings.random_seed)
        except ValueError:
            log.warning(f"Invalid random seed: {settings.random_seed}")
    return params


def load_model_with_lora(w: ComfyWorkflow, comfy: Client, style: Style):
    checkpoint = style.sd_checkpoint
    if checkpoint not in comfy.checkpoints:
        checkpoint = comfy.checkpoints[0]
        log.warning(f"Style checkpoint {style.sd_checkpoint} not found, using default {checkpoint}")
    model, clip, vae = w.load_checkpoint(checkpoint)

    if style.vae != StyleSettings.vae.default:
        if style.vae in comfy.vae_models:
            vae = w.load_vae(style.vae)
        else:
            log.warning(f"Style VAE {style.vae} not found, using default VAE from checkpoint")

    for lora in style.loras:
        if lora["name"] not in comfy.lora_models:
            log.warning(f"Style LoRA {lora['name']} not found, skipping")
            continue
        model, clip = w.load_lora(model, clip, lora["name"], lora["strength"], lora["strength"])
    return model, clip, vae


class Control:
    mode: ControlMode
    image: Union[Image, Output]
    mask: Union[None, Mask, Output] = None
    strength: float = 1.0

    def __init__(self, mode: ControlMode, image: Image, strength=1.0, mask: Optional[Mask] = None):
        self.mode = mode
        self.image = image
        self.strength = strength
        self.mask = mask

    def load_image(self, w: ComfyWorkflow):
        if isinstance(self.image, Image):
            self.image = w.load_image(self.image)
        return self.image

    def load_mask(self, w: ComfyWorkflow):
        if isinstance(self.mask, Mask):
            self.mask = w.load_mask(self.mask.to_image())
        return self.mask


class Conditioning:
    prompt: str
    control: List[Control]

    def __init__(self, prompt="", control: List[Control] = None):
        self.prompt = prompt
        self.control = control or []

    def add_control(
        self, type: ControlMode, image: Image, strength=1.0, mask: Optional[Mask] = None
    ):
        self.control.append(Control(type, image, strength, mask))

    def create(self, w: ComfyWorkflow, comfy: Client, clip: Output, style: Style):
        sd_ver = style.sd_version_resolved
        positive = w.clip_text_encode(clip, f"{self.prompt}, {style.style_prompt}")
        negative = w.clip_text_encode(clip, style.negative_prompt)

        for control in self.control:
            if control.mode is ControlMode.inpaint and not sd_ver.has_controlnet_inpaint:
                continue
            image = control.load_image(w)
            if control.mode is ControlMode.inpaint:
                image = w.inpaint_preprocessor(image, control.load_mask(w))
            if control.mode.is_lines:  # ControlNet expects white lines on black background
                image = w.invert_image(image)
            controlnet = w.load_controlnet(comfy.control_model[control.mode][sd_ver])
            positive = w.apply_controlnet(positive, controlnet, image, control.strength)

        return positive, negative

    def crop(self, w: ComfyWorkflow, bounds: Bounds):
        for control in self.control:
            control.image = w.crop_image(control.load_image(w), bounds)
            if control.mask:
                control.mask = w.crop_mask(control.load_mask(w), bounds)


def upscale(
    w: ComfyWorkflow,
    style: Style,
    latent: Output,
    extent: ScaledExtent,
    prompt_pos: Output,
    prompt_neg: Output,
    model: Output,
    vae: Output,
    comfy: Client,
):
    params = _sampler_params(style, upscale=True)
    if extent.scale > (1 / 1.5):
        # up to 1.5x scale: upscale latent
        upscale = w.scale_latent(latent, extent.expanded)
        params["denoise"] = 0.5
    else:
        # for larger upscaling factors use super-resolution model
        upscale_model = w.load_upscale_model(comfy.default_upscaler)
        decoded = w.vae_decode(vae, latent)
        upscale = w.upscale_image(upscale_model, decoded)
        upscale = w.scale_image(upscale, extent.expanded)
        upscale = w.vae_encode(vae, upscale)
        params["denoise"] = 0.4
        params["steps"] = max(1, int(params["steps"] * 0.8))

    return w.ksampler(model, prompt_pos, prompt_neg, upscale, **params)


def generate(comfy: Client, style: Style, input_extent: Extent, cond: Conditioning):
    _, _, extent, batch = prepare(input_extent, style.sd_version_resolved)

    w = ComfyWorkflow()
    model, clip, vae = load_model_with_lora(w, comfy, style)
    latent = w.empty_latent_image(extent.initial.width, extent.initial.height, batch)
    positive, negative = cond.create(w, comfy, clip, style)
    out_latent = w.ksampler(model, positive, negative, latent, **_sampler_params(style))
    if extent.requires_upscale:
        out_latent = upscale(w, style, out_latent, extent, positive, negative, model, vae, comfy)
    out_image = w.vae_decode(vae, out_latent)
    if extent.requires_downscale or extent.is_incompatible:
        out_image = w.scale_image(out_image, extent.target)
    w.send_image(out_image)
    return w


def inpaint(comfy: Client, style: Style, image: Image, mask: Mask, cond: Conditioning):
    sd_ver = style.sd_version_resolved
    scaled_image, scaled_mask, extent, _ = prepare((image, mask), sd_ver)
    target_bounds = mask.bounds
    region_expanded = target_bounds.extent.multiple_of(8)
    expanded_bounds = Bounds(*mask.bounds.offset, *region_expanded)
    batch = compute_batch_size(Extent.largest(scaled_image.extent, region_expanded))

    w = ComfyWorkflow()
    model, clip, vae = load_model_with_lora(w, comfy, style)
    in_image = w.load_image(scaled_image)
    in_mask = w.load_mask(scaled_mask)
    cropped_mask = w.load_mask(mask.to_image())
    if extent.requires_downscale:
        in_image = w.scale_image(in_image, extent.initial)
        in_mask = w.scale_mask(in_mask, extent.initial)
    cond.add_control(ControlMode.inpaint, in_image, mask=in_mask)
    positive, negative = cond.create(w, comfy, clip, style)
    if sd_ver.has_ip_adapter:
        clip_vision = w.load_clip_vision(comfy.clip_vision_model)
        ip_adapter = w.load_ip_adapter(comfy.ip_adapter_model)
        model = w.apply_ip_adapter(ip_adapter, clip_vision, in_image, model, 0.5)
    latent = w.vae_encode_inpaint(vae, in_image, in_mask)
    latent = w.batch_latent(latent, batch)
    out_latent = w.ksampler(
        model, positive, negative, latent, **_sampler_params(style, clip_vision=True)
    )
    if extent.requires_upscale:
        params = _sampler_params(style, clip_vision=True, upscale=True)
        if extent.scale > (1 / 1.5):
            # up to 1.5x scale: upscale latent
            latent = w.scale_latent(out_latent, extent.expanded)
            latent = w.crop_latent(latent, expanded_bounds)
            no_mask = w.solid_mask(expanded_bounds.extent, 1.0)
            latent = w.set_latent_noise_mask(latent, no_mask)
        else:
            # for larger upscaling factors use super-resolution model
            upscale_model = w.load_upscale_model(comfy.default_upscaler)
            upscale = w.vae_decode(vae, out_latent)
            upscale = w.crop_image(upscale, Bounds.scale(expanded_bounds, extent.scale))
            upscale = w.upscale_image(upscale_model, upscale)
            upscale = w.scale_image(upscale, expanded_bounds.extent)
            latent = w.vae_encode(vae, upscale)

        cond.control.pop()  # remove inpaint control
        cond.crop(w, expanded_bounds)
        cond.add_control(ControlMode.inpaint, Image.crop(image, target_bounds), mask=cropped_mask)
        positive_upscale, _ = cond.create(w, comfy, clip, style)
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


def refine(comfy: Client, style: Style, image: Image, cond: Conditioning, strength: float):
    assert strength > 0 and strength < 1
    image, _, extent, batch = prepare(image, style.sd_version_resolved, downscale=False)

    w = ComfyWorkflow()
    model, clip, vae = load_model_with_lora(w, comfy, style)
    in_image = w.load_image(image)
    if extent.is_incompatible:
        in_image = w.scale_image(in_image, extent.expanded)
    latent = w.vae_encode(vae, in_image)
    latent = w.batch_latent(latent, batch)
    positive, negative = cond.create(w, comfy, clip, style)
    sampler = w.ksampler(
        model, positive, negative, latent, denoise=strength, **_sampler_params(style)
    )
    out_image = w.vae_decode(vae, sampler)
    if extent.is_incompatible:
        out_image = w.scale_image(out_image, extent.target)
    w.send_image(out_image)
    return w


def refine_region(
    comfy: Client, style: Style, image: Image, mask: Mask, cond: Conditioning, strength: float
):
    assert strength > 0 and strength < 1

    downscale_if_needed = strength >= 0.7
    sd_ver = style.sd_version_resolved
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
    cond.add_control(ControlMode.inpaint, in_image, mask=in_mask)
    positive, negative = cond.create(w, comfy, clip, style)
    out_latent = w.ksampler(
        model, positive, negative, latent, denoise=strength, **_sampler_params(style)
    )
    if extent.requires_upscale:
        out_latent = upscale(w, style, out_latent, extent, positive, negative, model, vae, comfy)
    out_image = w.vae_decode(vae, out_latent)
    if extent.requires_downscale or extent.is_incompatible:
        out_image = w.scale_image(out_image, extent.target)
    original_mask = w.load_mask(mask.to_image())
    out_masked = w.apply_mask(out_image, original_mask)
    w.send_image(out_masked)
    return w


def create_control_image(image: Image, mode: ControlMode):
    w = ComfyWorkflow()
    input = w.load_image(image)
    if mode is ControlMode.canny_edge:
        result = w.add("Canny", 1, image=input, low_threshold=0.4, high_threshold=0.8)
    else:
        args = {
            "image": input,
            "resolution": image.extent.multiple_of(64).shortest_side,
        }
        if mode is ControlMode.scribble:
            result = w.add("FakeScribblePreprocessor", 1, **args, safe="enable")
        elif mode is ControlMode.line_art:
            result = w.add("LineArtPreprocessor", 1, **args, coarse="disable")
        elif mode is ControlMode.soft_edge:
            result = w.add("HEDPreprocessor", 1, **args, safe="enable")
        elif mode is ControlMode.depth:
            result = w.add("MiDaS-DepthMapPreprocessor", 1, **args, a=math.pi * 2, bg_threshold=0.1)
        elif mode is ControlMode.normal:
            result = w.add("BAE-NormalMapPreprocessor", 1, **args)
        elif mode is ControlMode.pose:
            feat = dict(detect_hand="enable", detect_body="enable", detect_face="enable")
            result = w.add("DWPreprocessor", 1, **args, **feat)
        elif mode is ControlMode.segmentation:
            result = w.add("OneFormer-COCO-SemSegPreprocessor", 1, **args)

        if args["resolution"] != image.extent.shortest_side:
            result = w.scale_image(result, image.extent)

    if mode.is_lines:
        result = w.invert_image(result)
    w.send_image(result)
    return w
