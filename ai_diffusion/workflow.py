from __future__ import annotations
import math
import random
from typing import Any, List, NamedTuple, Optional

from .image import Bounds, Extent, Image, Mask
from .client import Client, resolve_sd_version
from .style import SDVersion, Style, StyleSettings
from .resources import ControlMode
from .settings import settings
from .comfyworkflow import ComfyWorkflow, Output
from .util import compute_batch_size, client_logger as log


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


def create_inpaint_context(image: Image, area: Bounds, default: Output):
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
    return default


def prepare(
    extent: Extent, image: Image | None, mask: Mask | None, sdver: SDVersion, downscale=True
):
    mask_image = mask.to_image(extent) if mask else None

    # Latent space uses an 8 times lower resolution, so results are always multiples of 8.
    # If the target image is not a multiple of 8, the result must be scaled to fit.
    expanded = extent.multiple_of(8)

    min_size, max_size, min_pixel_count, max_pixel_count = {
        SDVersion.sd15: (512, 768, 512**2, 512 * 768),
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
    return ScaledExtent(initial, expanded, extent, scale), image, mask_image, batch


def prepare_extent(extent: Extent, sd_ver: SDVersion, downscale: bool = True):
    scaled, _, _, batch = prepare(extent, None, None, sd_ver, downscale)
    return scaled, batch


def prepare_image(image: Image, sd_ver: SDVersion, downscale: bool = True):
    scaled, out_image, _, batch = prepare(image.extent, image, None, sd_ver, downscale)
    assert out_image is not None
    return scaled, out_image, batch


def prepare_masked(image: Image, mask: Mask, sd_ver: SDVersion, downscale: bool = True):
    scaled, out_image, out_mask, batch = prepare(image.extent, image, mask, sd_ver, downscale)
    assert out_image and out_mask
    return scaled, out_image, out_mask, batch


class LiveParams:
    is_active = False
    strength = 0.3
    seed = random.randint(0, 2**31 - 1)


def _sampler_params(
    style: Style, clip_vision=False, upscale=False, live=LiveParams()
) -> dict[str, Any]:
    config = style.get_sampler_config(upscale, live.is_active)
    sampler_name = {
        "DDIM": "ddim",
        "DPM++ 2M": "dpmpp_2m",
        "DPM++ 2M Karras": "dpmpp_2m",
        "DPM++ 2M SDE": "dpmpp_2m_sde_gpu",
        "DPM++ 2M SDE Karras": "dpmpp_2m_sde_gpu",
        "LCM": "lcm",
    }[config.sampler]
    sampler_scheduler = {
        "DDIM": "ddim_uniform",
        "DPM++ 2M": "normal",
        "DPM++ 2M Karras": "karras",
        "DPM++ 2M SDE": "normal",
        "DPM++ 2M SDE Karras": "karras",
        "LCM": "sgm_uniform",
    }[config.sampler]
    params = dict(
        sampler=sampler_name, scheduler=sampler_scheduler, steps=config.steps, cfg=config.cfg
    )
    if clip_vision:
        params["cfg"] = min(5, config.cfg)
    if live.is_active:
        params["seed"] = live.seed
    elif settings.fixed_seed:
        try:
            params["seed"] = int(settings.random_seed)
        except ValueError:
            log.warning(f"Invalid random seed: {settings.random_seed}")
    return params


def load_model_with_lora(w: ComfyWorkflow, comfy: Client, style: Style, is_live=False):
    checkpoint = style.sd_checkpoint
    if checkpoint not in comfy.checkpoints:
        checkpoint = next(iter(comfy.checkpoints.keys()))
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

    if style.get_sampler_config(is_live=is_live).sampler == "LCM":
        sdver = resolve_sd_version(style, comfy)
        if comfy.lcm_model[sdver] is None:
            raise Exception(f"LCM LoRA model not found for {sdver.value}")
        model, _ = w.load_lora(model, clip, comfy.lcm_model[sdver], 1.0, 1.0)
        model = w.model_sampling_discrete(model, "lcm")

    return model, clip, vae


class Control:
    mode: ControlMode
    image: Image | Output
    mask: None | Mask | Output = None
    strength: float = 1.0
    end: float = 1.0

    def __init__(
        self,
        mode: ControlMode,
        image: Image | Output,
        strength=1.0,
        mask: None | Mask | Output = None,
        end: float=1.0,
    ):
        self.mode = mode
        self.image = image
        self.strength = strength
        self.mask = mask
        self.end = end

    def load_image(self, w: ComfyWorkflow):
        if isinstance(self.image, Image):
            self.image = w.load_image(self.image)
        return self.image

    def load_mask(self, w: ComfyWorkflow):
        assert self.mask is not None
        if isinstance(self.mask, Mask):
            self.mask = w.load_mask(self.mask.to_image())
        return self.mask

    def __eq__(self, other):
        if isinstance(other, Control):
            return self.__dict__ == other.__dict__
        return False


class Conditioning:
    prompt: str
    negative_prompt: str = ""
    area: Optional[Bounds] = None
    control: List[Control]

    def __init__(
        self,
        prompt="",
        negative_prompt="",
        control: list[Control] | None = None,
        area: Bounds | None = None,
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.control = control or []
        self.area = area

    def copy(self):
        return Conditioning(self.prompt, self.negative_prompt, [c for c in self.control], self.area)

    def crop(self, w: ComfyWorkflow, bounds: Bounds):
        for control in self.control:
            control.image = w.crop_image(control.load_image(w), bounds)
            if control.mask:
                control.mask = w.crop_mask(control.load_mask(w), bounds)


def merge_prompt(prompt: str, style_prompt: str):
    if style_prompt == "":
        return prompt
    elif "{prompt}" in style_prompt:
        return style_prompt.replace("{prompt}", prompt)
    elif prompt == "":
        return style_prompt
    return f"{prompt}, {style_prompt}"


def apply_conditioning(
    cond: Conditioning, w: ComfyWorkflow, comfy: Client, model: Output, clip: Output, style: Style
):
    prompt = merge_prompt(cond.prompt, style.style_prompt)
    if cond.area:
        prompt = merge_prompt("", style.style_prompt)
    positive = w.clip_text_encode(clip, prompt)
    negative = w.clip_text_encode(clip, merge_prompt(cond.negative_prompt, style.negative_prompt))
    model, positive, negative = apply_control(cond, w, comfy, model, positive, negative, style)
    if cond.area and cond.prompt != "":
        positive_area = w.clip_text_encode(clip, cond.prompt)
        positive_area = w.conditioning_area(positive_area, cond.area)
        positive = w.conditioning_combine(positive, positive_area)
    return model, positive, negative


def apply_control(
    cond: Conditioning,
    w: ComfyWorkflow,
    comfy: Client,
    model: Output,
    positive: Output,
    negative: Output,
    style: Style,
):
    sd_ver = resolve_sd_version(style, comfy)

    # Apply control net to the positive clip conditioning in a chain
    for control in (c for c in cond.control if c.mode is not ControlMode.image):
        model_file = comfy.control_model[control.mode][sd_ver]
        if model_file is None:
            continue
        image = control.load_image(w)
        if control.mode is ControlMode.inpaint:
            image = w.inpaint_preprocessor(image, control.load_mask(w))
        if control.mode.is_lines:  # ControlNet expects white lines on black background
            image = w.invert_image(image)
        controlnet = w.load_controlnet(model_file)
        positive, negative = w.apply_controlnet(positive, negative, controlnet, image, strength=control.strength, end_percent=control.end)

    # Merge all images into a single batch and apply IP-adapter to the model once
    ip_model_file = comfy.ip_adapter_model[sd_ver]
    if ip_model_file is not None:
        ip_image = None
        ip_strength = 0
        for control in (c for c in cond.control if c.mode is ControlMode.image):
            image = control.load_image(w)
            if ip_image is None:
                ip_image = image
                ip_strength = control.strength
            else:
                ip_image = w.batch_image(ip_image, image)
        if ip_image is not None:
            clip_vision = w.load_clip_vision(comfy.clip_vision_model)
            ip_adapter = w.load_ip_adapter(ip_model_file)
            weight_type = "original" if comfy.ip_adapter_has_weight_type else None
            model = w.apply_ip_adapter(
                ip_adapter, clip_vision, ip_image, model, ip_strength, weight_type=weight_type
            )

    return model, positive, negative


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


def generate(
    comfy: Client, style: Style, input_extent: Extent, cond: Conditioning, live=LiveParams()
):
    extent, batch = prepare_extent(
        input_extent, resolve_sd_version(style, comfy), downscale=not live.is_active
    )
    sampler_params = _sampler_params(style, live=live)
    batch = 1 if live.is_active else batch

    w = ComfyWorkflow()
    model, clip, vae = load_model_with_lora(w, comfy, style, is_live=live.is_active)
    latent = w.empty_latent_image(extent.initial.width, extent.initial.height, batch)
    model, positive, negative = apply_conditioning(cond, w, comfy, model, clip, style)
    out_latent = w.ksampler(model, positive, negative, latent, **sampler_params)
    if extent.requires_upscale:
        out_latent = upscale(w, style, out_latent, extent, positive, negative, model, vae, comfy)
    out_image = w.vae_decode(vae, out_latent)
    if extent.requires_downscale or extent.is_incompatible:
        out_image = w.scale_image(out_image, extent.target)
    w.send_image(out_image)
    return w


def inpaint(comfy: Client, style: Style, image: Image, mask: Mask, cond: Conditioning):
    sd_ver = resolve_sd_version(style, comfy)
    extent, scaled_image, scaled_mask, _ = prepare_masked(image, mask, sd_ver)
    target_bounds = mask.bounds
    region_expanded = target_bounds.extent.at_least(64).multiple_of(8)
    expanded_bounds = Bounds(*mask.bounds.offset, *region_expanded)

    w = ComfyWorkflow()
    model, clip, vae = load_model_with_lora(w, comfy, style)
    in_image = w.load_image(scaled_image)
    in_mask = w.load_mask(scaled_mask)
    cropped_mask = w.load_mask(mask.to_image())
    if extent.requires_downscale:
        in_image = w.scale_image(in_image, extent.initial)
        in_mask = w.scale_mask(in_mask, extent.initial)

    cond_base = cond.copy()
    cond_base.area = cond_base.area or mask.bounds
    cond_base.area = Bounds.scale(cond_base.area, extent.scale)
    image_strength = 0.5 if cond.prompt == "" else 0.3
    image_context = create_inpaint_context(scaled_image, cond_base.area, default=in_image)
    cond_base.control.append(Control(ControlMode.image, image_context, image_strength))
    cond_base.control.append(Control(ControlMode.inpaint, in_image, mask=in_mask))
    model, positive, negative = apply_conditioning(cond_base, w, comfy, model, clip, style)

    batch = compute_batch_size(Extent.largest(scaled_image.extent, region_expanded))
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

        cond_upscale = cond.copy()
        cond_upscale.area = None
        cond_upscale.crop(w, expanded_bounds)
        cond_upscale.control.append(
            Control(ControlMode.inpaint, Image.crop(image, target_bounds), mask=cropped_mask)
        )
        _, positive_upscale, _ = apply_conditioning(cond_upscale, w, comfy, model, clip, style)
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


def refine(
    comfy: Client,
    style: Style,
    image: Image,
    cond: Conditioning,
    strength: float,
    live=LiveParams(),
):
    assert strength > 0 and strength < 1
    extent, image, batch = prepare_image(image, resolve_sd_version(style, comfy), downscale=False)
    sampler_params = _sampler_params(style, live=live)

    w = ComfyWorkflow()
    model, clip, vae = load_model_with_lora(w, comfy, style, is_live=live.is_active)
    in_image = w.load_image(image)
    if extent.is_incompatible:
        in_image = w.scale_image(in_image, extent.expanded)
    latent = w.vae_encode(vae, in_image)
    if batch > 1 and not live.is_active:
        latent = w.batch_latent(latent, batch)
    model, positive, negative = apply_conditioning(cond, w, comfy, model, clip, style)
    sampler = w.ksampler(model, positive, negative, latent, denoise=strength, **sampler_params)
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
    sd_ver = resolve_sd_version(style, comfy)
    extent, image, mask_image, batch = prepare_masked(image, mask, sd_ver, downscale_if_needed)

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
    cond.control.append(Control(ControlMode.inpaint, in_image, mask=in_mask))
    model, positive, negative = apply_conditioning(cond, w, comfy, model, clip, style)
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
    assert mode not in [ControlMode.image, ControlMode.inpaint]

    w = ComfyWorkflow()
    input = w.load_image(image)
    result = None

    if mode is ControlMode.canny_edge:
        result = w.add("Canny", 1, image=input, low_threshold=0.4, high_threshold=0.8)
    else:
        args = {
            "image": input,
            "resolution": image.extent.multiple_of(64).shortest_side,
        }
        if mode is ControlMode.scribble:
            result = w.add("PiDiNetPreprocessor", 1, **args, safe="enable")
            result = w.add("ScribblePreprocessor", 1, image=result, resolution=args['resolution'])
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
        assert result is not None

        if args["resolution"] != image.extent.shortest_side:
            result = w.scale_image(result, image.extent)

    if mode.is_lines:
        result = w.invert_image(result)
    w.send_image(result)
    return w


def upscale_simple(comfy: Client, image: Image, model: str, factor: float):
    w = ComfyWorkflow()
    upscale_model = w.load_upscale_model(model)
    img = w.load_image(image)
    img = w.upscale_image(upscale_model, img)
    if factor != 4.0:
        img = w.scale_image(img, image.extent * factor)
    w.send_image(img)
    return w


def upscale_tiled(
    comfy: Client, image: Image, model: str, factor: float, style: Style, strength: float
):
    sd_ver = resolve_sd_version(style, comfy)
    cond = Conditioning("4k uhd")
    target_extent = image.extent * factor
    if sd_ver is SDVersion.sd15:
        tile_count = target_extent.longest_side / 768
        tile_extent = (target_extent * (1 / tile_count)).multiple_of(8)
    else:  # SDXL
        tile_extent = Extent(1024, 1024)

    w = ComfyWorkflow()
    img = w.load_image(image)
    checkpoint, clip, vae = load_model_with_lora(w, comfy, style)
    upscale_model = w.load_upscale_model(model)
    if sd_ver.has_controlnet_blur:
        cond.control.append(Control(ControlMode.blur, img))
    checkpoint, positive, negative = apply_conditioning(cond, w, comfy, checkpoint, clip, style)
    img = w.upscale_tiled(
        image=img,
        model=checkpoint,
        positive=positive,
        negative=negative,
        vae=vae,
        upscale_model=upscale_model,
        factor=factor,
        denoise=strength,
        original_extent=image.extent,
        tile_extent=tile_extent,
        **_sampler_params(style, upscale=True),
    )
    if not target_extent.is_multiple_of(8):
        img = w.scale_image(img, target_extent)
    w.send_image(img)
    return w
