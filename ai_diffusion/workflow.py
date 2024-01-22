from __future__ import annotations
from copy import copy
from enum import Enum
import math
import random
import re
from itertools import chain
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, overload

from .image import Bounds, Extent, Image, Mask, multiple_of
from .client import Client, resolve_sd_version
from .style import Style, StyleSettings
from .resources import ControlMode, SDVersion, UpscalerName
from .settings import settings
from .comfyworkflow import ComfyWorkflow, Output
from .util import ensure, median_or_zero, client_logger as log


_pattern_lora = re.compile(r"\s*<lora:([^:<>]+)(?::(-?[^:<>]*))?>\s*", re.IGNORECASE)


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


def compute_batch_size(extent: Extent, min_size=512, max_batches: Optional[int] = None):
    max_batches = max_batches or settings.batch_size
    desired_pixels = min_size * min_size * max_batches
    requested_pixels = extent.width * extent.height
    return max(1, min(max_batches, desired_pixels // requested_pixels))


class ScaleMode(Enum):
    none = 0
    resize = 1  # downscale, or tiny upscale, use simple scaling like bilinear
    upscale_latent = 2  # upscale by small factor, can be done in latent space
    upscale_fast = 3  # upscale using a fast model
    upscale_quality = 4  # upscale using a quality model


class ScaledExtent(NamedTuple):
    input: Extent  # resolution of input image and mask
    initial: Extent  # resolution for initial generation
    desired: Extent  # resolution for high res refinement pass
    target: Extent  # target resolution in canvas (may not be multiple of 8)

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
        if ratio < 1:
            return ScaleMode.resize
        else:
            return ScaleMode.none

    @property
    def refinement_scaling(self):
        ratio = Extent.ratio(self.initial, self.desired)
        if ratio < (1 / 1.5):
            return ScaleMode.upscale_quality
        elif ratio < 1:
            return ScaleMode.upscale_latent
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
    def compute(extent: Extent, sd_ver: SDVersion, style: Style | None = None):
        if style is None or style.preferred_resolution == 0:
            min_size, max_size, min_pixel_count, max_pixel_count = {
                SDVersion.sd15: (512, 768, 512**2, 512 * 768),
                SDVersion.sdxl: (896, 1280, 1024**2, 1024**2),
            }[sd_ver]
        else:
            range_offset = multiple_of(round(0.2 * style.preferred_resolution), 8)
            min_size = style.preferred_resolution - range_offset
            max_size = style.preferred_resolution + range_offset
            min_pixel_count = max_pixel_count = style.preferred_resolution**2
        min_scale = math.sqrt(min_pixel_count / extent.pixel_count)
        max_scale = math.sqrt(max_pixel_count / extent.pixel_count)
        return CheckpointResolution(min_size, max_size, min_scale, max_scale)


def apply_resolution_settings(extent: Extent):
    result = extent * settings.resolution_multiplier
    max_pixels = settings.max_pixel_count * 10**6
    if max_pixels > 0 and result.pixel_count > int(max_pixels * 1.1):
        result = result.scale_to_pixel_count(max_pixels)
    return result


def prepare(
    extent: Extent,
    image: Image | None,
    mask: Mask | None,
    sd_version: SDVersion,
    style: Style | None = None,
    downscale=True,
):
    mask_image = mask.to_image(extent) if mask else None

    # Take settings into account to compute the desired resolution for diffusion.
    desired = apply_resolution_settings(extent)
    # The checkpoint may require a different resolution than what is requested.
    min_size, max_size, min_scale, max_scale = CheckpointResolution.compute(
        desired, sd_version, style
    )

    if downscale and max_scale < 1 and any(x > max_size for x in desired):
        # Desired resolution is larger than the maximum size. Do 2 passes:
        # first pass at checkpoint resolution, then upscale to desired resolution and refine.
        input = initial = (desired * max_scale).multiple_of(8)
        desired = desired.multiple_of(8)
        # Input images are scaled down here for the initial pass directly to avoid encoding
        # and processing large images in subsequent steps.
        image, mask_image = _scale_images(image, mask_image, target=initial)

    elif min_scale > 1 and all(x < min_size for x in desired):
        # Desired resolution is smaller than the minimum size. Do 1 pass at checkpoint resolution.
        input = extent
        scaled = desired * min_scale
        # Avoid unnecessary scaling if too small resolution is caused by resolution multiplier
        if all(x >= min_size and x <= max_size for x in extent):
            initial = desired = extent.multiple_of(8)
        else:
            initial = desired = scaled.multiple_of(8)

    else:  # Desired resolution is in acceptable range. Do 1 pass at desired resolution.
        input = extent
        initial = desired = desired.multiple_of(8)
        # Scale down input images if needed due to resolution_multiplier or max_pixel_count
        if extent.pixel_count > desired.pixel_count:
            input = desired
            image, mask_image = _scale_images(image, mask_image, target=desired)

    batch = compute_batch_size(Extent.largest(initial, desired))
    return ScaledExtent(input, initial, desired, extent), image, mask_image, batch


def prepare_extent(extent: Extent, sd_ver: SDVersion, style: Style, downscale=True):
    scaled, _, _, batch = prepare(extent, None, None, sd_ver, style, downscale)
    return scaled, batch


def prepare_image(image: Image, sd_ver: SDVersion, style: Style, downscale=True):
    scaled, out_image, _, batch = prepare(image.extent, image, None, sd_ver, style, downscale)
    assert out_image is not None
    return scaled, out_image, batch


def prepare_masked(image: Image, mask: Mask, sd_ver: SDVersion, style: Style, downscale=True):
    scaled, out_image, out_mask, batch = prepare(
        image.extent, image, mask, sd_ver, style, downscale
    )
    assert out_image and out_mask
    return scaled, out_image, out_mask, batch


def _scale_images(*imgs: Image | None, target: Extent):
    return [Image.scale(img, target) if img else None for img in imgs]


def generate_seed():
    # Currently only using 32 bit because Qt widgets don't support int64
    return random.randint(0, 2**31 - 1)


def _sampler_params(
    style: Style, strength=1.0, seed=-1, clip_vision=False, advanced=True, is_live=False
) -> dict[str, Any]:
    config = style.get_sampler_config(is_live)
    sampler_name = {
        "DDIM": "ddim",
        "DPM++ 2M": "dpmpp_2m",
        "DPM++ 2M Karras": "dpmpp_2m",
        "DPM++ 2M SDE": "dpmpp_2m_sde_gpu",
        "DPM++ 2M SDE Karras": "dpmpp_2m_sde_gpu",
        "DPM++ SDE Karras": "dpmpp_sde_gpu",
        "UniPC BH2": "uni_pc_bh2",
        "LCM": "lcm",
        "Euler": "euler",
        "Euler a": "euler_ancestral",
    }[config.sampler]
    sampler_scheduler = {
        "DDIM": "ddim_uniform",
        "DPM++ 2M": "normal",
        "DPM++ 2M Karras": "karras",
        "DPM++ 2M SDE": "normal",
        "DPM++ 2M SDE Karras": "karras",
        "DPM++ SDE Karras": "karras",
        "UniPC BH2": "ddim_uniform",
        "LCM": "sgm_uniform",
        "Euler": "normal",
        "Euler a": "normal",
    }[config.sampler]
    params: dict[str, Any] = dict(
        sampler=sampler_name,
        scheduler=sampler_scheduler,
        steps=config.steps,
        cfg=config.cfg,
        seed=seed,
    )
    if advanced:
        if strength < 1.0:
            min_steps = config.steps if is_live else 1
            params["steps"], params["start_at_step"] = _apply_strength(
                strength, params["steps"], min_steps
            )
        else:
            params["start_at_step"] = 0
    if clip_vision:
        params["cfg"] = min(5, config.cfg)
    return params


def extract_loras(prompt: str, client_loras: list[str]):
    loras = []
    for match in _pattern_lora.findall(prompt):
        lora_name = ""

        for client_lora in client_loras:
            lora_filename = Path(client_lora).stem
            if match[0].lower() == lora_filename.lower():
                lora_name = client_lora

        if not lora_name:
            error = f"LoRA not found : {match[0]}"
            log.warning(error)
            raise Exception(error)

        lora_strength = match[1] if match[1] != "" else 1.0
        try:
            lora_strength = float(lora_strength)
        except ValueError:
            error = f"Invalid LoRA strength for {match[0]} : {lora_strength}"
            log.warning(error)
            raise Exception(error)

        loras.append(dict(name=lora_name, strength=lora_strength))
    return _pattern_lora.sub("", prompt), loras


def _apply_strength(strength: float, steps: int, min_steps: int = 0) -> tuple[int, int]:
    start_at_step = round(steps * (1 - strength))

    if min_steps and steps - start_at_step < min_steps:
        steps = math.floor(min_steps * 1 / strength)
        start_at_step = steps - min_steps

    return steps, start_at_step


def load_model_with_lora(
    w: ComfyWorkflow,
    comfy: Client,
    style: Style,
    cond: Conditioning,
    is_live=False,
):
    checkpoint = style.sd_checkpoint
    if checkpoint not in comfy.checkpoints:
        checkpoint = next(iter(comfy.checkpoints.keys()))
        log.warning(f"Style checkpoint {style.sd_checkpoint} not found, using default {checkpoint}")
    model, clip, vae = w.load_checkpoint(checkpoint)

    if style.clip_skip != StyleSettings.clip_skip.default:
        clip = w.clip_set_last_layer(clip, (style.clip_skip * -1))

    if style.vae != StyleSettings.vae.default:
        if style.vae in comfy.vae_models:
            vae = w.load_vae(style.vae)
        else:
            log.warning(f"Style VAE {style.vae} not found, using default VAE from checkpoint")

    for lora in chain(style.loras, cond.loras):
        if lora["name"] not in comfy.loras:
            log.warning(f"LoRA {lora['name']} not found, skipping")
            continue
        model, clip = w.load_lora(model, clip, lora["name"], lora["strength"], lora["strength"])

    sdver = resolve_sd_version(style, comfy)
    is_lcm = style.get_sampler_config(is_live=is_live).sampler == "LCM"
    if is_lcm:
        if lora := comfy.lora_models["lcm"][sdver]:
            model = w.load_lora_model(model, lora, 1.0)
        else:
            raise Exception(f"LCM LoRA model not found for {sdver.value}")

    face_weight = median_or_zero(c.strength for c in cond.control if c.mode is ControlMode.face)
    if face_weight > 0:
        if lora := comfy.lora_models["face"][sdver]:
            model = w.load_lora_model(model, lora, 0.65 * face_weight)
        else:
            raise Exception(f"IP-Adapter Face LoRA model not found for {sdver.value}")

    if style.v_prediction_zsnr:
        model = w.model_sampling_discrete(model, "v_prediction", zsnr=True)
        model = w.rescale_cfg(model, 0.7)
    elif is_lcm:
        model = w.model_sampling_discrete(model, "lcm")

    return model, clip, vae


class Control:
    mode: ControlMode
    image: Image | Output
    mask: None | Mask | Output = None
    strength: float = 1.0
    end: float = 1.0

    _original_extent: Extent | None = None

    def __init__(
        self,
        mode: ControlMode,
        image: Image | Output,
        strength=1.0,
        end=1.0,
        mask: None | Mask | Output = None,
    ):
        self.mode = mode
        self.image = image
        self.strength = strength
        self.mask = mask
        self.end = end

    def load_image(self, w: ComfyWorkflow, target_extent: Extent | None = None):
        if isinstance(self.image, Image):
            self._original_extent = self.image.extent
            self.image = w.load_image(self.image)
        if target_extent and self._original_extent != target_extent:
            return w.scale_control_image(self.image, target_extent)
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
    control: list[Control]
    area: Bounds | None = None
    loras: list[dict[str, Any]]

    def __init__(
        self,
        prompt="",
        negative_prompt="",
        control: list[Control] | None = None,
        area: Bounds | None = None,
        loras: list[dict[str, Any]] | None = None,
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.control = control or []
        self.area = area
        self.loras = loras or []

    def copy(self):
        return Conditioning(
            self.prompt,
            self.negative_prompt,
            [copy(c) for c in self.control],
            self.area,
            self.loras,
        )

    def downscale(self, original: Extent, target: Extent):
        # Meant to be called during preperation, when desired generation resolution is lower than canvas size:
        # no need to encode and send the full resolution images to the server.
        if original.width > target.width and original.height > target.height:
            for control in self.control:
                assert isinstance(control.image, Image)
                # Only scale if control image resolution matches canvas resolution
                if control.image.extent == original and control.mode is not ControlMode.canny_edge:
                    control.image = Image.scale(control.image, target)

    def crop(self, bounds: Bounds):
        # Meant to be called during preperation, before adding inpaint layer.
        for control in self.control:
            assert isinstance(control.image, Image) and control.mask is None
            control.image = Image.crop(control.image, bounds)


def merge_prompt(prompt: str, style_prompt: str):
    if style_prompt == "":
        return prompt
    elif "{prompt}" in style_prompt:
        return style_prompt.replace("{prompt}", prompt)
    elif prompt == "":
        return style_prompt
    return f"{prompt}, {style_prompt}"


def encode_text_prompt(w: ComfyWorkflow, cond: Conditioning, clip: Output, style: Style):
    prompt = merge_prompt(cond.prompt, style.style_prompt)
    if cond.area:
        prompt = merge_prompt("", style.style_prompt)
    positive = w.clip_text_encode(clip, prompt)
    negative = w.clip_text_encode(clip, merge_prompt(cond.negative_prompt, style.negative_prompt))
    return positive, negative


def apply_area(w: ComfyWorkflow, positive: Output, cond: Conditioning, clip: Output):
    if cond.area and cond.prompt != "":
        positive_area = w.clip_text_encode(clip, cond.prompt)
        positive_area = w.conditioning_area(positive_area, cond.area)
        positive = w.conditioning_combine(positive, positive_area)
    return positive


def apply_control(
    w: ComfyWorkflow,
    positive: Output,
    negative: Output,
    control_layers: list[Control],
    extent: Extent,
    comfy: Client,
    sd_version: SDVersion,
):
    for control in (c for c in control_layers if c.mode.is_control_net):
        model_file = comfy.control_model[control.mode][sd_version]
        if model_file is None:
            continue
        image = control.load_image(w, extent)
        if control.mode is ControlMode.inpaint:
            image = w.inpaint_preprocessor(image, control.load_mask(w))
        if control.mode.is_lines:  # ControlNet expects white lines on black background
            image = w.invert_image(image)
        controlnet = w.load_controlnet(model_file)
        positive, negative = w.apply_controlnet(
            positive,
            negative,
            controlnet,
            image,
            strength=control.strength,
            end_percent=control.end,
        )

    return positive, negative


def apply_ip_adapter(
    w: ComfyWorkflow,
    model: Output,
    control_layers: list[Control],
    comfy: Client,
    sd_version: SDVersion,
):
    # Create a separate embedding for each face ID (though more than 1 is questionable)
    if ipadapter_model_name := comfy.ip_adapter_model[ControlMode.face][sd_version]:
        face_layers = [c for c in control_layers if c.mode is ControlMode.face]
        if len(face_layers) > 0:
            clip_vision = w.load_clip_vision(comfy.clip_vision_model)
            ip_adapter = w.load_ip_adapter(ipadapter_model_name)
            insight_face = w.load_insight_face()
            for control in face_layers:
                model = w.apply_ip_adapter_face(
                    ip_adapter,
                    clip_vision,
                    insight_face,
                    model,
                    control.load_image(w),
                    control.strength,
                    end_at=control.end,
                    faceid_v2="v2" in ipadapter_model_name,
                )

    # Encode images with their weights into a batch and apply IP-adapter to the model once
    if ipadapter_model_name := comfy.ip_adapter_model[ControlMode.reference][sd_version]:
        ip_images = []
        ip_weights = []
        ip_end_at = 0.1

        for control in (c for c in control_layers if c.mode is ControlMode.reference):
            if len(ip_images) >= 4:
                raise Exception("Too many control layers of type 'reference image' (maximum is 4)")
            ip_images.append(control.load_image(w))
            ip_weights.append(control.strength)
            ip_end_at = max(ip_end_at, control.end)

        max_weight = max(ip_weights, default=0.0)
        if len(ip_images) > 0 and max_weight > 0:
            ip_weights = [w / max_weight for w in ip_weights]
            clip_vision = w.load_clip_vision(comfy.clip_vision_model)
            ip_adapter = w.load_ip_adapter(ipadapter_model_name)
            embeds = w.encode_ip_adapter(clip_vision, ip_images, ip_weights, noise=0.2)
            model = w.apply_ip_adapter(ip_adapter, embeds, model, max_weight, end_at=ip_end_at)

    return model


def scale(
    extent: Extent, target: Extent, mode: ScaleMode, w: ComfyWorkflow, image: Output, comfy: Client
):
    """Handles scaling images from `extent` to `target` resolution.
    Uses either simple bilinear scaling or a fast upscaling model."""

    if mode is ScaleMode.none:
        return image
    elif mode is ScaleMode.resize:
        return w.scale_image(image, target)
    else:
        assert mode is ScaleMode.upscale_fast
        ratio = target.pixel_count / extent.pixel_count
        factor = max(2, min(4, math.ceil(math.sqrt(ratio))))
        upscale_model_name = ensure(comfy.upscale_models[UpscalerName.fast_x(factor)])
        upscale_model = w.load_upscale_model(upscale_model_name)
        image = w.upscale_image(upscale_model, image)
        image = w.scale_image(image, target)
        return image


def scale_to_initial(
    extent: ScaledExtent, w: ComfyWorkflow, image: Output, comfy: Client, is_mask=False
):
    if is_mask and extent.initial_scaling is ScaleMode.resize:
        return w.scale_mask(image, extent.initial)
    elif not is_mask:
        return scale(extent.input, extent.initial, extent.initial_scaling, w, image, comfy)
    else:
        assert is_mask and extent.initial_scaling is ScaleMode.none
        return image


def scale_to_target(extent: ScaledExtent, w: ComfyWorkflow, image: Output, comfy: Client):
    return scale(extent.desired, extent.target, extent.target_scaling, w, image, comfy)


def scale_refine_and_decode(
    extent: ScaledExtent,
    w: ComfyWorkflow,
    style: Style,
    latent: Output,
    prompt_pos: Output,
    prompt_neg: Output,
    cond: Conditioning,
    seed: int,
    model: Output,
    vae: Output,
    comfy: Client,
):
    """Handles scaling images from `initial` to `desired` resolution.
    If it is a substantial upscale, runs a high-res SD refinement pass.
    Takes latent as input and returns a decoded image."""

    mode = extent.refinement_scaling
    if mode in [ScaleMode.none, ScaleMode.resize, ScaleMode.upscale_fast]:
        decoded = w.vae_decode(vae, latent)
        return scale(extent.initial, extent.desired, mode, w, decoded, comfy)

    if mode is ScaleMode.upscale_latent:
        upscale = w.scale_latent(latent, extent.desired)
        params = _sampler_params(style, strength=0.5, seed=seed)
    else:
        assert mode is ScaleMode.upscale_quality
        upscale_model = w.load_upscale_model(comfy.default_upscaler)
        decoded = w.vae_decode(vae, latent)
        upscale = w.upscale_image(upscale_model, decoded)
        upscale = w.scale_image(upscale, extent.desired)
        upscale = w.vae_encode(vae, upscale)
        params = _sampler_params(style, strength=0.4, seed=seed)

    sd_ver = resolve_sd_version(style, comfy)
    positive, negative = apply_control(
        w, prompt_pos, prompt_neg, cond.control, extent.desired, comfy, sd_ver
    )
    result = w.ksampler_advanced(model, positive, negative, upscale, **params)
    image = w.vae_decode(vae, result)
    return image


def generate(
    comfy: Client, style: Style, input_extent: Extent, cond: Conditioning, seed: int, is_live=False
):
    sd_ver = resolve_sd_version(style, comfy)
    extent, batch = prepare_extent(input_extent, sd_ver, style, downscale=not is_live)
    cond.downscale(input_extent, extent.desired)
    sampler_params = _sampler_params(style, seed=seed, is_live=is_live)
    batch = 1 if is_live else batch

    w = ComfyWorkflow(comfy.nodes_inputs)
    model, clip, vae = load_model_with_lora(w, comfy, style, cond, is_live=is_live)
    model = apply_ip_adapter(w, model, cond.control, comfy, sd_ver)
    latent = w.empty_latent_image(extent.initial, batch)
    prompt_pos, prompt_neg = encode_text_prompt(w, cond, clip, style)
    positive, negative = apply_control(
        w, prompt_pos, prompt_neg, cond.control, extent.initial, comfy, sd_ver
    )
    out_latent = w.ksampler_advanced(model, positive, negative, latent, **sampler_params)
    out_image = scale_refine_and_decode(
        extent, w, style, out_latent, prompt_pos, prompt_neg, cond, seed, model, vae, comfy
    )
    out_image = scale_to_target(extent, w, out_image, comfy)
    w.send_image(out_image)
    return w


def inpaint(comfy: Client, style: Style, image: Image, mask: Mask, cond: Conditioning, seed: int):
    target_bounds = mask.bounds
    sd_ver = resolve_sd_version(style, comfy)
    extent, scaled_image, scaled_mask, _ = prepare_masked(image, mask, sd_ver, style)
    upscale_extent, _ = prepare_extent(target_bounds.extent, sd_ver, style, downscale=False)

    w = ComfyWorkflow(comfy.nodes_inputs)
    model, clip, vae = load_model_with_lora(w, comfy, style, cond)
    in_image = w.load_image(scaled_image)
    in_image = scale_to_initial(extent, w, in_image, comfy)
    in_mask = w.load_mask(scaled_mask)
    in_mask = scale_to_initial(extent, w, in_mask, comfy, is_mask=True)
    cropped_mask = w.load_mask(mask.to_image())

    cond_base = cond.copy()
    cond_base.downscale(image.extent, extent.initial)
    cond_base.area = extent.convert(cond.area, "target", "initial") if cond.area else None
    image_strength = 0.5 if cond.prompt == "" else 0.3
    image_context = create_inpaint_context(scaled_image, cond_base.area or mask.bounds, in_image)
    cond_base.control.append(Control(ControlMode.reference, image_context, image_strength))
    cond_base.control.append(Control(ControlMode.inpaint, in_image, mask=in_mask))
    model = apply_ip_adapter(w, model, cond_base.control, comfy, sd_ver)
    prompt_pos, prompt_neg = encode_text_prompt(w, cond_base, clip, style)
    positive, negative = apply_control(
        w, prompt_pos, prompt_neg, cond_base.control, extent.initial, comfy, sd_ver
    )
    positive = apply_area(w, positive, cond_base, clip)

    batch = compute_batch_size(Extent.largest(extent.initial, upscale_extent.desired))
    latent = w.vae_encode_inpaint(vae, in_image, in_mask)
    latent = w.batch_latent(latent, batch)
    out_latent = w.ksampler_advanced(
        model, positive, negative, latent, **_sampler_params(style, seed=seed, clip_vision=True)
    )
    if extent.refinement_scaling in [ScaleMode.upscale_latent, ScaleMode.upscale_quality]:
        initial_bounds = extent.convert(target_bounds, "target", "initial")

        params = _sampler_params(style, strength=0.5, seed=seed, clip_vision=True)
        upscale_model = w.load_upscale_model(comfy.default_upscaler)
        upscale = w.vae_decode(vae, out_latent)
        upscale = w.crop_image(upscale, initial_bounds)
        upscale = w.upscale_image(upscale_model, upscale)
        upscale = w.scale_image(upscale, upscale_extent.desired)
        latent = w.vae_encode(vae, upscale)

        cond_upscale = cond.copy()
        cond_upscale.crop(target_bounds)
        cond_upscale.control.append(
            Control(ControlMode.inpaint, Image.crop(image, target_bounds), mask=cropped_mask)
        )
        positive_up, negative_up = apply_control(
            w, prompt_pos, prompt_neg, cond_upscale.control, upscale_extent.desired, comfy, sd_ver
        )
        out_latent = w.ksampler_advanced(model, positive_up, negative_up, latent, **params)
        out_image = w.vae_decode(vae, out_latent)
        out_image = scale_to_target(upscale_extent, w, out_image, comfy)
    else:
        desired_bounds = extent.convert(target_bounds, "target", "desired")
        cropped_extent = ScaledExtent(
            desired_bounds.extent,
            desired_bounds.extent,
            desired_bounds.extent,
            target_bounds.extent,
        )

        out_image = w.vae_decode(vae, out_latent)
        out_image = scale(
            extent.initial, extent.desired, extent.refinement_scaling, w, out_image, comfy
        )
        out_image = w.crop_image(out_image, desired_bounds)
        out_image = scale_to_target(cropped_extent, w, out_image, comfy)

    out_masked = w.apply_mask(out_image, cropped_mask)
    w.send_image(out_masked)
    return w


def refine(
    comfy: Client,
    style: Style,
    image: Image,
    cond: Conditioning,
    strength: float,
    seed: int,
    is_live=False,
):
    assert strength > 0 and strength < 1
    sd_ver = resolve_sd_version(style, comfy)
    extent, image, batch = prepare_image(image, sd_ver, style, downscale=False)
    cond.downscale(image.extent, extent.desired)
    sampler_params = _sampler_params(style, strength, seed, is_live=is_live)

    w = ComfyWorkflow(comfy.nodes_inputs)
    model, clip, vae = load_model_with_lora(w, comfy, style, cond, is_live=is_live)
    model = apply_ip_adapter(w, model, cond.control, comfy, sd_ver)
    in_image = w.load_image(image)
    in_image = scale_to_initial(extent, w, in_image, comfy)
    latent = w.vae_encode(vae, in_image)
    if batch > 1 and not is_live:
        latent = w.batch_latent(latent, batch)
    positive, negative = encode_text_prompt(w, cond, clip, style)
    positive, negative = apply_control(
        w, positive, negative, cond.control, extent.desired, comfy, sd_ver
    )
    sampler = w.ksampler_advanced(model, positive, negative, latent, **sampler_params)
    out_image = w.vae_decode(vae, sampler)
    out_image = scale_to_target(extent, w, out_image, comfy)
    w.send_image(out_image)
    return w


def refine_region(
    comfy: Client,
    style: Style,
    image: Image,
    mask: Mask,
    cond: Conditioning,
    strength: float,
    seed: int,
    is_live=False,
):
    assert strength > 0 and strength <= 1

    allow_2pass = strength >= 0.7
    sd_ver = resolve_sd_version(style, comfy)
    extent, image, mask_image, batch = prepare_masked(image, mask, sd_ver, style, allow_2pass)
    cond.downscale(image.extent, extent.desired)
    sampler_params = _sampler_params(style, strength, seed, is_live=is_live)

    w = ComfyWorkflow(comfy.nodes_inputs)
    model, clip, vae = load_model_with_lora(w, comfy, style, cond, is_live=is_live)
    model = apply_ip_adapter(w, model, cond.control, comfy, sd_ver)
    in_image = w.load_image(image)
    in_image = scale_to_initial(extent, w, in_image, comfy)
    in_mask = w.load_mask(mask_image)
    in_mask = scale_to_initial(extent, w, in_mask, comfy, is_mask=True)
    latent = w.vae_encode(vae, in_image)
    latent = w.set_latent_noise_mask(latent, in_mask)
    if batch > 1 and not is_live:
        latent = w.batch_latent(latent, batch)
    if not is_live:
        cond.control.append(Control(ControlMode.inpaint, in_image, mask=in_mask))
    prompt_pos, prompt_neg = encode_text_prompt(w, cond, clip, style)
    positive, negative = apply_control(
        w, prompt_pos, prompt_neg, cond.control, extent.initial, comfy, sd_ver
    )
    out_latent = w.ksampler_advanced(model, positive, negative, latent, **sampler_params)
    out_image = scale_refine_and_decode(
        extent, w, style, out_latent, prompt_pos, prompt_neg, cond, seed, model, vae, comfy
    )
    out_image = scale_to_target(extent, w, out_image, comfy)
    original_mask = w.load_mask(mask.to_image())
    out_masked = w.apply_mask(out_image, original_mask)
    w.send_image(out_masked)
    return w


def create_control_image(
    comfy: Client, image: Image, mode: ControlMode, bounds: Bounds | None = None, seed: int = -1
):
    assert mode not in [ControlMode.reference, ControlMode.face, ControlMode.inpaint]

    target_extent = image.extent
    current_extent = apply_resolution_settings(image.extent)
    if current_extent != target_extent:
        image = Image.scale(image, current_extent)

    w = ComfyWorkflow(comfy.nodes_inputs)
    input = w.load_image(image)
    result = None

    if mode is ControlMode.canny_edge:
        result = w.add("Canny", 1, image=input, low_threshold=0.4, high_threshold=0.8)

    elif mode is ControlMode.hands:
        if bounds is None:
            current_extent = current_extent.multiple_of(64)
            resolution = current_extent.shortest_side
        else:
            input = w.crop_image(input, bounds)
            resolution = bounds.extent.multiple_of(64).shortest_side
        result, _ = w.add(
            "MeshGraphormer-DepthMapPreprocessor",
            2,
            image=input,
            resolution=resolution,
            mask_type="based_on_depth",
            rand_seed=seed if seed != -1 else generate_seed(),
        )
        if bounds is not None:
            result = w.scale_image(result, bounds.extent)
            empty = w.empty_image(current_extent)
            result = w.composite_image_masked(result, empty, None, bounds.x, bounds.y)
    else:
        current_extent = current_extent.multiple_of(64)
        args = {"image": input, "resolution": current_extent.shortest_side}
        if mode is ControlMode.scribble:
            result = w.add("PiDiNetPreprocessor", 1, **args, safe="enable")
            result = w.add("ScribblePreprocessor", 1, image=result, resolution=args["resolution"])
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

    if mode.is_lines:
        result = w.invert_image(result)
    if current_extent != target_extent:
        result = w.scale_image(result, target_extent)

    w.send_image(result)
    return w


def upscale_simple(comfy: Client, image: Image, model: str, factor: float):
    w = ComfyWorkflow(comfy.nodes_inputs)
    upscale_model = w.load_upscale_model(model)
    img = w.load_image(image)
    img = w.upscale_image(upscale_model, img)
    if factor != 4.0:
        img = w.scale_image(img, image.extent * factor)
    w.send_image(img)
    return w


def upscale_tiled(
    comfy: Client, image: Image, model: str, factor: float, style: Style, strength: float, seed: int
):
    sd_ver = resolve_sd_version(style, comfy)
    cond = Conditioning("4k uhd")
    target_extent = image.extent * factor
    if style.preferred_resolution > 0:
        tile_extent = Extent(style.preferred_resolution, style.preferred_resolution)
    elif sd_ver is SDVersion.sd15:
        tile_count = target_extent.longest_side / 768
        tile_extent = (target_extent * (1 / tile_count)).multiple_of(8)
    else:  # SDXL
        tile_extent = Extent(1024, 1024)

    w = ComfyWorkflow(comfy.nodes_inputs)
    img = w.load_image(image)
    checkpoint, clip, vae = load_model_with_lora(w, comfy, style, cond)
    checkpoint = apply_ip_adapter(w, checkpoint, cond.control, comfy, sd_ver)
    upscale_model = w.load_upscale_model(model)
    positive, negative = encode_text_prompt(w, cond, clip, style)
    if sd_ver.has_controlnet_blur:
        blur = [Control(ControlMode.blur, img)]
        positive, negative = apply_control(w, positive, negative, blur, image.extent, comfy, sd_ver)
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
        **_sampler_params(style, seed=seed, advanced=False),
    )
    if not target_extent.is_multiple_of(8):
        img = w.scale_image(img, target_extent)
    w.send_image(img)
    return w
