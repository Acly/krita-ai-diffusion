from __future__ import annotations
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
import math
import random
import re
from pathlib import Path
from typing import Any, NamedTuple, Optional, overload

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


def detect_inpaint_mode(extent: Extent, area: Bounds):
    if area.width >= extent.width or area.height >= extent.height:
        return InpaintMode.expand
    return InpaintMode.fill


def get_inpaint_reference(image: Image, area: Bounds):
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
    return None


def compute_batch_size(extent: Extent, min_size=512, max_batches: Optional[int] = None):
    max_batches = max_batches or settings.batch_size
    desired_pixels = min_size * min_size * max_batches
    requested_pixels = extent.width * extent.height
    return max(1, min(max_batches, desired_pixels // requested_pixels))


class ScaleMode(Enum):
    none = 0
    resize = 1  # downscale, or tiny upscale, use simple scaling like bilinear
    upscale_small = 2  # upscale by small factor (<1.5)
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
            return ScaleMode.upscale_small
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
    return ImageInput(scaled), batch


def prepare_image(image: Image, sd_ver: SDVersion, style: Style, downscale=True):
    scaled, out_image, _, batch = prepare(image.extent, image, None, sd_ver, style, downscale)
    assert out_image is not None
    return ImageInput(scaled, out_image), batch


def prepare_masked(image: Image, mask: Mask, sd_ver: SDVersion, style: Style, downscale=True):
    scaled, out_image, out_mask, batch = prepare(
        image.extent, image, mask, sd_ver, style, downscale
    )
    assert out_image and out_mask
    return ImageInput(scaled, out_image, out_mask), batch


def _scale_images(*imgs: Image | None, target: Extent):
    return [Image.scale(img, target) if img else None for img in imgs]


def generate_seed():
    # Currently only using 32 bit because Qt widgets don't support int64
    return random.randint(0, 2**31 - 1)


def _sampler_params(sampling: SamplingInput, strength=0.0, clip_vision=False, advanced=True):
    sampler_name = {
        "DDIM": "ddim",
        "DPM++ 2M": "dpmpp_2m",
        "DPM++ 2M Karras": "dpmpp_2m",
        "DPM++ 2M SDE": "dpmpp_2m_sde_gpu",
        "DPM++ 2M SDE Karras": "dpmpp_2m_sde_gpu",
        "DPM++ SDE Karras": "dpmpp_sde_gpu",
        "UniPC BH2": "uni_pc_bh2",
        "LCM": "lcm",
        "Lightning": "euler",
        "Euler": "euler",
        "Euler a": "euler_ancestral",
    }[sampling.sampler]
    sampler_scheduler = {
        "DDIM": "ddim_uniform",
        "DPM++ 2M": "normal",
        "DPM++ 2M Karras": "karras",
        "DPM++ 2M SDE": "normal",
        "DPM++ 2M SDE Karras": "karras",
        "DPM++ SDE Karras": "karras",
        "UniPC BH2": "ddim_uniform",
        "LCM": "sgm_uniform",
        "Lightning": "sgm_uniform",
        "Euler": "normal",
        "Euler a": "normal",
    }[sampling.sampler]
    params: dict[str, Any] = dict(
        sampler=sampler_name,
        scheduler=sampler_scheduler,
        steps=sampling.steps,
        cfg=sampling.cfg_scale,
        seed=sampling.seed,
    )
    strength = strength if strength > 0 else sampling.strength
    if advanced:
        if strength < 1.0:
            min_steps = sampling.steps if sampling.sampler == "LCM" else 1
            params["steps"], params["start_at_step"] = _apply_strength(
                strength, params["steps"], min_steps
            )
        else:
            params["start_at_step"] = 0
    if clip_vision:
        params["cfg"] = min(5, sampling.cfg_scale)
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


def load_model_with_lora(w: ComfyWorkflow, models: ModelInput, comfy: Client):
    checkpoint = models.checkpoint
    if checkpoint not in comfy.checkpoints:
        checkpoint = next(iter(comfy.checkpoints.keys()))
        log.warning(f"Style checkpoint {models.checkpoint} not found, using default {checkpoint}")
    model, clip, vae = w.load_checkpoint(checkpoint)

    if models.clip_skip != StyleSettings.clip_skip.default:
        clip = w.clip_set_last_layer(clip, (models.clip_skip * -1))

    if models.vae != StyleSettings.vae.default:
        if models.vae in comfy.vae_models:
            vae = w.load_vae(models.vae)
        else:
            log.warning(f"Style VAE {models.vae} not found, using default VAE from checkpoint")

    for lora in models.loras:
        model, clip = w.load_lora(model, clip, lora.name, lora.strength, lora.strength)

    sdver = comfy.checkpoints[models.checkpoint].sd_version
    lcm_lora = comfy.lora_models["lcm"][sdver]
    is_lcm = any(l.name == lcm_lora for l in models.loras)

    if models.v_prediction_zsnr:
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
    range = (0.0, 1.0)

    _original_extent: Extent | None = None

    def __init__(
        self,
        mode: ControlMode,
        image: Image | Output,
        strength=1.0,
        range: tuple[float, float] = (0.0, 1.0),
        mask: None | Mask | Output = None,
    ):
        self.mode = mode
        self.image = image
        self.strength = strength
        self.mask = mask
        self.range = range

    @staticmethod
    def from_input(i: ControlInput):
        return Control(i.mode, i.image, i.strength, i.range)

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


@dataclass
class Conditioning:
    prompt: str
    negative_prompt: str = ""
    control: list[Control] = field(default_factory=list)
    style_prompt: str = ""
    mask: Output | None = None

    def copy(self):
        return Conditioning(
            self.prompt, self.negative_prompt, [copy(c) for c in self.control], self.style_prompt
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


def encode_text_prompt(w: ComfyWorkflow, cond: Conditioning, clip: Output):
    prompt = cond.prompt
    if prompt != "" and cond.mask:
        prompt = merge_prompt("", cond.style_prompt)
    elif prompt != "":
        prompt = merge_prompt(prompt, cond.style_prompt)
    positive = w.clip_text_encode(clip, prompt)
    negative = w.clip_text_encode(clip, cond.negative_prompt)
    if cond.mask and cond.prompt != "":
        masked = w.clip_text_encode(clip, cond.prompt)
        masked = w.conditioning_set_mask(masked, cond.mask)
        positive = w.conditioning_combine(positive, masked)
    return positive, negative


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
            range=control.range,
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
                    range=control.range,
                    faceid_v2="v2" in ipadapter_model_name,
                )

    # Encode images with their weights into a batch and apply IP-adapter to the model once
    if ipadapter_model_name := comfy.ip_adapter_model[ControlMode.reference][sd_version]:
        ip_images = []
        ip_weights = []
        ip_range = (0.9, 0.1)

        for control in (c for c in control_layers if c.mode is ControlMode.reference):
            if len(ip_images) >= 4:
                raise Exception("Too many control layers of type 'reference image' (maximum is 4)")
            ip_images.append(control.load_image(w))
            ip_weights.append(control.strength)
            ip_range = (min(ip_range[0], control.range[0]), max(ip_range[1], control.range[1]))

        max_weight = max(ip_weights, default=0.0)
        if len(ip_images) > 0 and max_weight > 0:
            ip_weights = [w / max_weight for w in ip_weights]
            clip_vision = w.load_clip_vision(comfy.clip_vision_model)
            ip_adapter = w.load_ip_adapter(ipadapter_model_name)
            embeds = w.encode_ip_adapter(clip_vision, ip_images, ip_weights, noise=0.2)
            model = w.apply_ip_adapter(ip_adapter, embeds, model, max_weight, range=ip_range)

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
    cond: Conditioning,
    sampling: SamplingInput,
    latent: Output,
    prompt_pos: Output,
    prompt_neg: Output,
    model: Output,
    vae: Output,
    sd_ver: SDVersion,
    comfy: Client,
):
    """Handles scaling images from `initial` to `desired` resolution.
    If it is a substantial upscale, runs a high-res SD refinement pass.
    Takes latent as input and returns a decoded image."""

    mode = extent.refinement_scaling
    if mode in [ScaleMode.none, ScaleMode.resize, ScaleMode.upscale_fast]:
        decoded = w.vae_decode(vae, latent)
        return scale(extent.initial, extent.desired, mode, w, decoded, comfy)

    if mode is ScaleMode.upscale_small:
        upscaler = ensure(comfy.upscale_models[UpscalerName.fast_2x])
    else:
        assert mode is ScaleMode.upscale_quality
        upscaler = comfy.default_upscaler

    upscale_model = w.load_upscale_model(upscaler)
    decoded = w.vae_decode(vae, latent)
    upscale = w.upscale_image(upscale_model, decoded)
    upscale = w.scale_image(upscale, extent.desired)
    upscale = w.vae_encode(vae, upscale)
    params = _sampler_params(sampling, strength=0.4)

    positive, negative = apply_control(
        w, prompt_pos, prompt_neg, cond.control, extent.desired, comfy, sd_ver
    )
    result = w.ksampler_advanced(model, positive, negative, upscale, **params)
    image = w.vae_decode(vae, result)
    return image


def generate(
    models: ModelInput,
    extent: ScaledExtent,
    cond: Conditioning,
    sampling: SamplingInput,
    batch_count: int,
    comfy: Client,
):
    sd_version = comfy.checkpoints[models.checkpoint].sd_version

    w = ComfyWorkflow(comfy.nodes_inputs)
    model, clip, vae = load_model_with_lora(w, models, comfy)
    model = apply_ip_adapter(w, model, cond.control, comfy, sd_version)
    latent = w.empty_latent_image(extent.initial, batch_count)
    prompt_pos, prompt_neg = encode_text_prompt(w, cond, clip)
    positive, negative = apply_control(
        w, prompt_pos, prompt_neg, cond.control, extent.initial, comfy, sd_version
    )
    out_latent = w.ksampler_advanced(model, positive, negative, latent, **_sampler_params(sampling))
    out_image = scale_refine_and_decode(
        extent, w, cond, sampling, out_latent, prompt_pos, prompt_neg, model, vae, sd_version, comfy
    )
    out_image = scale_to_target(extent, w, out_image, comfy)
    w.send_image(out_image)
    return w


class InpaintMode(Enum):
    automatic = 0
    fill = 1
    expand = 2
    add_object = 3
    remove_object = 4
    replace_background = 5
    custom = 6


class FillMode(Enum):
    none = 0
    neutral = 1
    blur = 2
    border = 3
    replace = 4
    inpaint = 5


def fill_masked(w: ComfyWorkflow, image: Output, mask: Output, fill: FillMode, comfy: Client):
    if fill is FillMode.blur:
        return w.blur_masked(image, mask, 65, falloff=9)
    elif fill is FillMode.border:
        image = w.fill_masked(image, mask, "navier-stokes")
        return w.blur_masked(image, mask, 65)
    elif fill is FillMode.neutral:
        return w.fill_masked(image, mask, "neutral", falloff=9)
    elif fill is FillMode.inpaint:
        model = w.load_inpaint_model(ensure(comfy.inpaint_models["default"]))
        return w.inpaint_image(model, image, mask)
    elif fill is FillMode.replace:
        return w.fill_masked(image, mask, "neutral")
    return image


@dataclass
class InpaintParams:
    mode: InpaintMode
    target_bounds: Bounds
    fill: FillMode = FillMode.neutral
    use_inpaint_model = False
    use_condition_mask = False
    use_reference = False

    @staticmethod
    def detect(
        mode: InpaintMode, bounds: Bounds, sd_ver: SDVersion, cond: Conditioning, strength: float
    ):
        assert mode is not InpaintMode.automatic
        result = InpaintParams(mode, bounds)

        result.use_inpaint_model = strength > 0.5
        if sd_ver is SDVersion.sd15:
            result.use_condition_mask = (
                mode is InpaintMode.add_object
                and cond.prompt != ""
                and not any(c.mode.is_structural for c in cond.control)
            )

        is_ref_mode = mode in [InpaintMode.fill, InpaintMode.expand]
        result.use_reference = is_ref_mode and cond.prompt == ""

        result.fill = {
            InpaintMode.fill: FillMode.blur,
            InpaintMode.expand: FillMode.border,
            InpaintMode.add_object: FillMode.neutral,
            InpaintMode.remove_object: FillMode.inpaint,
            InpaintMode.replace_background: FillMode.replace,
        }[mode]
        return result

    @staticmethod
    def automatic(bounds: Bounds, sd_ver: SDVersion, cond: Conditioning, image_extent: Extent):
        mode = detect_inpaint_mode(image_extent, bounds)
        return InpaintParams.detect(mode, bounds, sd_ver, cond, strength=1.0)


def inpaint(
    images: ImageInput,
    models: ModelInput,
    cond: Conditioning,
    sampling: SamplingInput,
    params: InpaintParams,
    crop_upscale_extent: Extent,
    batch_count: int,
    comfy: Client,
):
    sd_version = comfy.checkpoints[models.checkpoint].sd_version
    target_bounds = params.target_bounds
    extent = images.extent  # for initial generation with large context
    upscale_extent = ScaledExtent(  # after crop to the masked region
        Extent(0, 0), Extent(0, 0), crop_upscale_extent, target_bounds.extent
    )
    initial_bounds = extent.convert(target_bounds, "target", "initial")

    w = ComfyWorkflow(comfy.nodes_inputs)
    model, clip, vae = load_model_with_lora(w, models, comfy)
    in_image = w.load_image(ensure(images.initial_image))
    in_image = scale_to_initial(extent, w, in_image, comfy)
    in_mask = w.load_mask(ensure(images.initial_mask))
    in_mask = scale_to_initial(extent, w, in_mask, comfy, is_mask=True)
    cropped_mask = w.load_mask(ensure(images.hires_mask))

    cond_base = cond.copy()
    cond_base.downscale(extent.input, extent.initial)
    if params.use_reference:
        reference = get_inpaint_reference(ensure(images.initial_image), initial_bounds) or in_image
        cond_base.control.append(Control(ControlMode.reference, reference, 0.5, (0.2, 0.8)))
    if params.use_inpaint_model and sd_version is SDVersion.sd15:
        cond_base.control.append(Control(ControlMode.inpaint, in_image, mask=in_mask))
    if params.use_condition_mask:
        cond_base.mask = in_mask

    in_image = fill_masked(w, in_image, in_mask, params.fill, comfy)

    model = apply_ip_adapter(w, model, cond_base.control, comfy, sd_version)
    positive, negative = encode_text_prompt(w, cond_base, clip)
    positive, negative = apply_control(
        w, positive, negative, cond_base.control, extent.initial, comfy, sd_version
    )
    if params.use_inpaint_model and sd_version is SDVersion.sdxl:
        positive, negative, latent_inpaint, latent = w.vae_encode_inpaint_conditioning(
            vae, in_image, in_mask, positive, negative
        )
        inpaint_patch = w.load_fooocus_inpaint(**comfy.fooocus_inpaint_models)
        inpaint_model = w.apply_fooocus_inpaint(model, inpaint_patch, latent_inpaint)
    else:
        latent = w.vae_encode(vae, in_image)
        latent = w.set_latent_noise_mask(latent, in_mask)
        inpaint_model = model

    latent = w.batch_latent(latent, batch_count)
    sampler_params = _sampler_params(sampling, clip_vision=params.use_reference)
    out_latent = w.ksampler_advanced(inpaint_model, positive, negative, latent, **sampler_params)

    if extent.refinement_scaling in [ScaleMode.upscale_small, ScaleMode.upscale_quality]:
        if extent.refinement_scaling is ScaleMode.upscale_small:
            upscaler = ensure(comfy.upscale_models[UpscalerName.fast_2x])
        else:
            upscaler = comfy.default_upscaler
        sampler_params = _sampler_params(sampling, strength=0.4)
        upscale_model = w.load_upscale_model(upscaler)
        upscale = w.vae_decode(vae, out_latent)
        upscale = w.crop_image(upscale, initial_bounds)
        upscale = w.upscale_image(upscale_model, upscale)
        upscale = w.scale_image(upscale, upscale_extent.desired)
        latent = w.vae_encode(vae, upscale)
        latent = w.set_latent_noise_mask(latent, cropped_mask)

        cond_upscale = cond.copy()
        cond_upscale.crop(target_bounds)
        if params.use_inpaint_model and sd_version is SDVersion.sd15:
            cond_upscale.control.append(
                Control(ControlMode.inpaint, ensure(images.hires_image), mask=cropped_mask)
            )
        res = upscale_extent.desired
        positive_up, negative_up = encode_text_prompt(w, cond_upscale, clip)
        positive_up, negative_up = apply_control(
            w, positive_up, negative_up, cond_upscale.control, res, comfy, sd_version
        )
        out_latent = w.ksampler_advanced(model, positive_up, negative_up, latent, **sampler_params)
        out_image = w.vae_decode(vae, out_latent)
        out_image = scale_to_target(upscale_extent, w, out_image, comfy)
    else:
        desired_bounds = extent.convert(target_bounds, "target", "desired")
        desired_extent = desired_bounds.extent
        cropped_extent = ScaledExtent(
            desired_extent, desired_extent, desired_extent, target_bounds.extent
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
    image: Image,
    extent: ScaledExtent,
    models: ModelInput,
    cond: Conditioning,
    sampling: SamplingInput,
    batch_count: int,
    comfy: Client,
):
    sd_version = comfy.checkpoints[models.checkpoint].sd_version

    w = ComfyWorkflow(comfy.nodes_inputs)
    model, clip, vae = load_model_with_lora(w, models, comfy)
    model = apply_ip_adapter(w, model, cond.control, comfy, sd_version)
    in_image = w.load_image(image)
    in_image = scale_to_initial(extent, w, in_image, comfy)
    latent = w.vae_encode(vae, in_image)
    if batch_count > 1:
        latent = w.batch_latent(latent, batch_count)
    positive, negative = encode_text_prompt(w, cond, clip)
    positive, negative = apply_control(
        w, positive, negative, cond.control, extent.desired, comfy, sd_version
    )
    sampler = w.ksampler_advanced(model, positive, negative, latent, **_sampler_params(sampling))
    out_image = w.vae_decode(vae, sampler)
    out_image = scale_to_target(extent, w, out_image, comfy)
    w.send_image(out_image)
    return w


def refine_region(
    images: ImageInput,
    models: ModelInput,
    cond: Conditioning,
    sampling: SamplingInput,
    inpaint: InpaintParams,
    batch_count: int,
    comfy: Client,
):
    sd_version = comfy.checkpoints[models.checkpoint].sd_version
    extent = images.extent

    w = ComfyWorkflow(comfy.nodes_inputs)
    model, clip, vae = load_model_with_lora(w, models, comfy)
    model = apply_ip_adapter(w, model, cond.control, comfy, sd_version)
    in_image = w.load_image(ensure(images.initial_image))
    in_image = scale_to_initial(extent, w, in_image, comfy)
    in_mask = w.load_mask(ensure(images.initial_mask))
    in_mask = scale_to_initial(extent, w, in_mask, comfy, is_mask=True)

    prompt_pos, prompt_neg = encode_text_prompt(w, cond, clip)
    if inpaint.use_inpaint_model and sd_version is SDVersion.sd15:
        cond.control.append(Control(ControlMode.inpaint, in_image, mask=in_mask))
    positive, negative = apply_control(
        w, prompt_pos, prompt_neg, cond.control, extent.initial, comfy, sd_version
    )
    if sd_version is SDVersion.sd15 or not inpaint.use_inpaint_model:
        latent = w.vae_encode(vae, in_image)
        latent = w.set_latent_noise_mask(latent, in_mask)
        inpaint_model = model
    else:  # SDXL inpaint model
        positive, negative, latent_inpaint, latent = w.vae_encode_inpaint_conditioning(
            vae, in_image, in_mask, positive, negative
        )
        inpaint_patch = w.load_fooocus_inpaint(**comfy.fooocus_inpaint_models)
        inpaint_model = w.apply_fooocus_inpaint(model, inpaint_patch, latent_inpaint)

    if batch_count > 1:
        latent = w.batch_latent(latent, batch_count)

    out_latent = w.ksampler_advanced(
        inpaint_model, positive, negative, latent, **_sampler_params(sampling)
    )
    out_image = scale_refine_and_decode(
        extent, w, cond, sampling, out_latent, prompt_pos, prompt_neg, model, vae, sd_version, comfy
    )
    out_image = scale_to_target(extent, w, out_image, comfy)
    if extent.target != inpaint.target_bounds.extent:
        out_image = w.crop_image(out_image, inpaint.target_bounds)
    original_mask = w.load_mask(ensure(images.hires_mask))
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
    image: Image,
    extent: ScaledExtent,
    upscale_model_name: str,
    models: ModelInput,
    cond: Conditioning,
    sampling: SamplingInput,
    comfy: Client,
):
    sd_version = comfy.checkpoints[models.checkpoint].sd_version

    w = ComfyWorkflow(comfy.nodes_inputs)
    checkpoint, clip, vae = load_model_with_lora(w, models, comfy)
    checkpoint = apply_ip_adapter(w, checkpoint, cond.control, comfy, sd_version)
    img = w.load_image(image)
    upscale_model = w.load_upscale_model(upscale_model_name)
    positive, negative = encode_text_prompt(w, cond, clip)
    if sd_version.has_controlnet_blur:
        blur = [Control(ControlMode.blur, img)]
        positive, negative = apply_control(
            w, positive, negative, blur, extent.input, comfy, sd_version
        )
    img = w.upscale_tiled(
        image=img,
        model=checkpoint,
        positive=positive,
        negative=negative,
        vae=vae,
        upscale_model=upscale_model,
        factor=extent.target.width / extent.input.width,
        denoise=sampling.strength,
        original_extent=extent.input,
        tile_extent=extent.initial,
        **_sampler_params(sampling, advanced=False),
    )
    if not extent.target.is_multiple_of(8):
        img = w.scale_image(img, extent.target)
    w.send_image(img)
    return w


class WorkflowKind(Enum):
    generate = 0
    inpaint = 1
    refine = 2
    refine_region = 3
    upscale_simple = 4
    upscale_tiled = 5
    control_image = 6


@dataclass
class ImageInput:
    extent: ScaledExtent
    initial_image: Image | None = None
    initial_mask: Image | None = None
    hires_image: Image | None = None
    hires_mask: Image | None = None


@dataclass
class LoraInput:
    name: str
    strength: float

    @staticmethod
    def from_dict(data: dict[str, Any]):
        return LoraInput(data["name"], data["strength"])


@dataclass
class ControlInput:
    mode: ControlMode
    image: Image
    strength: float = 1.0
    range: tuple[float, float] = (0.0, 1.0)

    @staticmethod
    def from_control(control: Control):
        assert isinstance(control.image, Image)
        return ControlInput(control.mode, control.image, control.strength, control.range)


@dataclass
class ModelInput:
    checkpoint: str
    vae: str = ""
    loras: list[LoraInput] = field(default_factory=list)
    clip_skip: int = 0
    v_prediction_zsnr: bool = False

    @staticmethod
    def from_style(style: Style):
        result = ModelInput(
            checkpoint=style.sd_checkpoint,
            vae=style.vae,
            clip_skip=style.clip_skip,
            v_prediction_zsnr=style.v_prediction_zsnr,
        )
        result.add_loras(style.loras)
        return result

    def add_loras(self, loras: list[dict]):
        self.loras += [LoraInput.from_dict(l) for l in loras]

    def add_sampling_lora(self, sampler: str, client: Client):
        sd_version = client.checkpoints[self.checkpoint].sd_version
        if sampler == "LCM":
            if lora := client.lora_models["lcm"][sd_version]:
                self.loras.append(LoraInput(lora, 1.0))
            else:
                raise Exception(f"LCM LoRA model not found for {sd_version.value}")

    def add_control_lora(self, control_layers: list[Control], client: Client):
        sd_version = client.checkpoints[self.checkpoint].sd_version
        face_weight = median_or_zero(
            c.strength for c in control_layers if c.mode is ControlMode.face
        )
        if face_weight > 0:
            if lora := client.lora_models["face"][sd_version]:
                self.loras.append(LoraInput(lora, 0.65 * face_weight))
            else:
                raise Exception(f"IP-Adapter Face LoRA model not found for {sd_version.value}")


@dataclass
class SamplingInput:
    sampler: str
    steps: int
    cfg_scale: float
    strength: float = 1.0
    seed: int = 0

    @staticmethod
    def from_style(style: Style, is_live: bool):
        config = style.get_sampler_config(is_live)
        return SamplingInput(config.sampler, config.steps, config.cfg)


@dataclass
class WorkflowInput:
    kind: WorkflowKind

    images: ImageInput | None = None
    models: ModelInput | None = None
    sampling: SamplingInput | None = None

    prompt: str = ""
    style_prompt: str = ""
    negative_prompt: str = ""
    control: list[ControlInput] = field(default_factory=list)

    inpaint: InpaintParams | None = None
    crop_upscale_extent: Extent | None = None
    upscale_model: str = ""
    batch_count = 1

    @staticmethod
    def create(
        kind: WorkflowKind,
        canvas: Image | Extent,
        mask: Mask | None,
        seed: int,
        style: Style,
        client: Client,
        loras: list[dict],
        conditioning: Conditioning,
        strength: float = 1.0,
        inpaint: InpaintParams | None = None,
        upscale_factor: float = 1.0,
        upscale_model: str = "",
        is_live: bool = False,
    ):
        sd_version = client.checkpoints[style.sd_checkpoint].sd_version

        i = WorkflowInput(kind)
        i.sampling = SamplingInput.from_style(style, is_live)
        i.sampling.seed = seed
        i.sampling.strength = strength
        i.models = ModelInput.from_style(style)
        i.models.add_loras(loras)
        i.models.add_sampling_lora(i.sampling.sampler, client)
        i.models.add_control_lora(conditioning.control, client)
        i.prompt = conditioning.prompt
        i.style_prompt = style.style_prompt
        i.negative_prompt = merge_prompt(conditioning.negative_prompt, style.negative_prompt)

        if kind is WorkflowKind.generate:
            assert isinstance(canvas, Extent)
            i.images, i.batch_count = prepare_extent(
                canvas, sd_version, ensure(style), downscale=not is_live
            )
            conditioning.downscale(canvas, i.images.extent.desired)

        elif kind is WorkflowKind.inpaint:
            assert isinstance(canvas, Image) and mask and inpaint and style
            i.images, _ = prepare_masked(canvas, mask, sd_version, style)
            upscale_extent, _ = prepare_extent(
                mask.bounds.extent, sd_version, style, downscale=False
            )
            i.inpaint = inpaint
            i.crop_upscale_extent = upscale_extent.extent.desired
            i.batch_count = compute_batch_size(
                Extent.largest(i.images.extent.initial, upscale_extent.extent.desired)
            )
            i.images.hires_mask = mask.to_image()
            scaling = i.images.extent.refinement_scaling
            if scaling in [ScaleMode.upscale_small, ScaleMode.upscale_quality]:
                i.images.hires_image = Image.crop(canvas, i.inpaint.target_bounds)
            if inpaint.mode is InpaintMode.remove_object and i.prompt == "":
                i.prompt = "background scenery"

        elif kind is WorkflowKind.refine:
            assert isinstance(canvas, Image) and style
            i.images, i.batch_count = prepare_image(canvas, sd_version, style, downscale=False)
            conditioning.downscale(canvas.extent, i.images.extent.desired)

        elif kind is WorkflowKind.refine_region:
            assert isinstance(canvas, Image) and mask and inpaint and style
            allow_2pass = strength >= 0.7
            i.images, i.batch_count = prepare_masked(
                canvas, mask, sd_version, style, downscale=allow_2pass
            )
            i.images.hires_mask = mask.to_image()
            i.inpaint = inpaint
            conditioning.downscale(canvas.extent, i.images.extent.desired)

        elif kind is WorkflowKind.upscale_tiled:
            assert isinstance(canvas, Image) and style and upscale_model
            i.upscale_model = upscale_model
            target_extent = canvas.extent * upscale_factor
            if style.preferred_resolution > 0:
                tile_extent = Extent(style.preferred_resolution, style.preferred_resolution)
            elif sd_version is SDVersion.sd15:
                tile_count = target_extent.longest_side / 768
                tile_extent = (target_extent * (1 / tile_count)).multiple_of(8)
            else:  # SDXL
                tile_extent = Extent(1024, 1024)
            extent = ScaledExtent(
                canvas.extent, tile_extent, target_extent.multiple_of(8), target_extent
            )
            i.images = ImageInput(extent, canvas)

        else:
            raise Exception(f"Workflow {kind.name} not supported by this constructor")

        i.batch_count = 1 if is_live else i.batch_count
        i.control = [ControlInput.from_control(c) for c in conditioning.control]
        return i

    @staticmethod
    def upscale_simple(image: Image, model: str, factor: float):
        target_extent = image.extent * factor
        extent = ScaledExtent(image.extent, image.extent, target_extent, target_extent)
        i = WorkflowInput(WorkflowKind.upscale_simple, ImageInput(extent, image))
        i.upscale_model = model
        return i

    @staticmethod
    def control_image(
        image: Image, mode: ControlMode, bounds: Bounds | None = None, seed: int = -1
    ):
        i = WorkflowInput(WorkflowKind.control_image)
        i.control = [ControlInput(mode, image)]
        if bounds and seed != -1:
            i.inpaint = InpaintParams(InpaintMode.fill, bounds)
            i.sampling = SamplingInput("", 0, 0, seed=seed)
        return i

    @property
    def extent(self):
        return ensure(self.images).extent

    @property
    def image(self):
        return ensure(ensure(self.images).initial_image)

    @property
    def conditioning(self):
        control = [Control.from_input(c) for c in self.control]
        return Conditioning(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            control=control,
            style_prompt=self.style_prompt,
        )

    @property
    def upscale_factor(self):
        return self.extent.target.width / self.extent.input.width


def generate_workflow(i: WorkflowInput, client: Client):
    if i.kind is WorkflowKind.generate:
        return generate(
            ensure(i.models), i.extent, i.conditioning, ensure(i.sampling), i.batch_count, client
        )
    elif i.kind is WorkflowKind.inpaint:
        return inpaint(
            ensure(i.images),
            ensure(i.models),
            i.conditioning,
            ensure(i.sampling),
            ensure(i.inpaint),
            ensure(i.crop_upscale_extent),
            i.batch_count,
            client,
        )
    elif i.kind is WorkflowKind.refine:
        return refine(
            i.image,
            i.extent,
            ensure(i.models),
            i.conditioning,
            ensure(i.sampling),
            i.batch_count,
            client,
        )
    elif i.kind is WorkflowKind.refine_region:
        return refine_region(
            ensure(i.images),
            ensure(i.models),
            i.conditioning,
            ensure(i.sampling),
            ensure(i.inpaint),
            i.batch_count,
            client,
        )
    elif i.kind is WorkflowKind.upscale_simple:
        return upscale_simple(client, i.image, i.upscale_model, i.upscale_factor)

    elif i.kind is WorkflowKind.upscale_tiled:
        return upscale_tiled(
            i.image,
            i.extent,
            i.upscale_model,
            ensure(i.models),
            i.conditioning,
            ensure(i.sampling),
            client,
        )
    elif i.kind is WorkflowKind.control_image:
        c = i.control[0]
        seed = i.sampling.seed if i.sampling else -1
        bounds = i.inpaint.target_bounds if i.inpaint else None
        return create_control_image(client, c.image, c.mode, bounds, seed)
    else:
        raise ValueError(f"Unsupported workflow kind: {i.kind}")
