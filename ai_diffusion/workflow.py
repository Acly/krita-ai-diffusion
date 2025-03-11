from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Any, NamedTuple
import math
import random

from . import resolution, resources
from .api import ControlInput, ImageInput, CheckpointInput, SamplingInput, WorkflowInput, LoraInput
from .api import ExtentInput, InpaintMode, InpaintParams, FillMode, ConditioningInput, WorkflowKind
from .api import RegionInput, CustomWorkflowInput, UpscaleInput
from .image import Bounds, Extent, Image, ImageCollection, Mask, multiple_of
from .client import ClientModels, ModelDict, resolve_arch
from .files import FileLibrary, FileFormat
from .style import Style, StyleSettings, SamplerPresets
from .resolution import ScaledExtent, ScaleMode, TileLayout, get_inpaint_reference
from .resources import ControlMode, Arch, UpscalerName, ResourceKind, ResourceId
from .settings import PerformanceSettings
from .text import merge_prompt, extract_loras
from .comfy_workflow import ComfyWorkflow, ComfyRunMode, Input, Output, ComfyNode
from .localization import translate as _
from .settings import settings
from .util import ensure, median_or_zero, unique


def detect_inpaint_mode(extent: Extent, area: Bounds):
    if area.width >= extent.width or area.height >= extent.height:
        return InpaintMode.expand
    return InpaintMode.fill


def generate_seed():
    # Currently only using 32 bit because Qt widgets don't support int64
    return random.randint(0, 2**31 - 1)


def _sampling_from_style(style: Style, strength: float, is_live: bool):
    sampler_name = style.live_sampler if is_live else style.sampler
    cfg = style.live_cfg_scale if is_live else style.cfg_scale
    min_steps, max_steps = style.get_steps(is_live=is_live)
    preset = SamplerPresets.instance()[sampler_name]
    result = SamplingInput(
        sampler=preset.sampler,
        scheduler=preset.scheduler,
        cfg_scale=cfg or preset.cfg,
        total_steps=max_steps,
    )
    if strength < 1.0:
        result.total_steps, result.start_step = apply_strength(strength, max_steps, min_steps)
    return result


def apply_strength(strength: float, steps: int, min_steps: int = 0) -> tuple[int, int]:
    start_at_step = round(steps * (1 - strength))

    if min_steps and steps - start_at_step < min_steps:
        steps = math.floor(min_steps / strength)
        start_at_step = steps - min_steps

    return steps, start_at_step


# Given the pair we got from `apply_strength`, reconstruct a weight (in percent).
# representing a natural midpoint of the range.
# For example, if the pair is 4/8, return 50.
# This is used for snapping the strength widget.
# If the resulting step-count is adjusted upward as per above, no such
# midpoint can be reliably determined. In that case, we return None.
def snap_to_percent(steps: int, start_at_step: int, max_steps: int) -> int | None:
    if steps != max_steps:
        return None
    return round((steps - start_at_step) * 100 / steps)


def _sampler_params(sampling: SamplingInput, strength: float | None = None):
    params: dict[str, Any] = dict(
        sampler=sampling.sampler,
        scheduler=sampling.scheduler,
        steps=sampling.total_steps,
        start_at_step=sampling.start_step,
        cfg=sampling.cfg_scale,
        seed=sampling.seed,
    )
    if strength is not None:
        params["steps"], params["start_at_step"] = apply_strength(strength, sampling.total_steps)
    return params


def load_checkpoint_with_lora(w: ComfyWorkflow, checkpoint: CheckpointInput, models: ClientModels):
    arch = checkpoint.version
    model_info = models.checkpoints.get(checkpoint.checkpoint)
    if model_info is None:
        raise RuntimeError(f"Style checkpoint {checkpoint.checkpoint} not found")

    clip, vae = None, None
    match model_info.format:
        case FileFormat.checkpoint:
            model, clip, vae = w.load_checkpoint(model_info.filename)
        case FileFormat.diffusion:
            model = w.load_diffusion_model(model_info.filename)
        case _:
            raise RuntimeError(
                f"Style checkpoint {checkpoint.checkpoint} has an unsupported format {model_info.format.name}"
            )

    if clip is None or arch is Arch.sd3:
        te = models.for_arch(arch).text_encoder
        match arch:
            case Arch.sd15:
                clip = w.load_clip(te["clip_l"], "stable_diffusion")
            case Arch.sdxl | Arch.illu | Arch.illu_v:
                clip = w.load_dual_clip(te["clip_g"], te["clip_l"], type="sdxl")
            case Arch.sd3:
                if te.find("t5"):
                    clip = w.load_triple_clip(te["clip_l"], te["clip_g"], te["t5"])
                else:
                    clip = w.load_dual_clip(te["clip_g"], te["clip_l"], type="sd3")
            case Arch.flux:
                clip = w.load_dual_clip(te["clip_l"], te["t5"], type="flux")
            case _:
                raise RuntimeError(f"No text encoder for model architecture {arch.name}")

    if arch.supports_clip_skip and checkpoint.clip_skip != StyleSettings.clip_skip.default:
        clip = w.clip_set_last_layer(clip, (checkpoint.clip_skip * -1))

    if checkpoint.vae and checkpoint.vae != StyleSettings.vae.default:
        vae = w.load_vae(checkpoint.vae)
    if vae is None:
        vae = w.load_vae(models.for_arch(arch).vae)

    if checkpoint.dynamic_caching and (arch in [Arch.flux, Arch.sd3] or arch.is_sdxl_like):
        model = w.apply_first_block_cache(model, arch)

    for lora in checkpoint.loras:
        model, clip = w.load_lora(model, clip, lora.name, lora.strength, lora.strength)

    if arch is Arch.sd3:
        model = w.model_sampling_sd3(model)

    if checkpoint.v_prediction_zsnr:
        model = w.model_sampling_discrete(model, "v_prediction", zsnr=True)

    is_zsnr = checkpoint.v_prediction_zsnr or arch is Arch.illu_v
    if is_zsnr and checkpoint.rescale_cfg > 0:
        model = w.rescale_cfg(model, checkpoint.rescale_cfg)

    if arch.supports_lcm:
        lcm_lora = models.for_arch(arch).lora.find("lcm")
        if lcm_lora and any(l.name == lcm_lora for l in checkpoint.loras):
            model = w.model_sampling_discrete(model, "lcm")

    if arch.supports_attention_guidance and checkpoint.self_attention_guidance:
        model = w.apply_self_attention_guidance(model)

    return model, Clip(clip, arch), vae


def vae_decode(w: ComfyWorkflow, vae: Output, latent: Output, tiled: bool):
    if tiled:
        return w.vae_decode_tiled(vae, latent)
    return w.vae_decode(vae, latent)


class ImageReshape(NamedTuple):
    """Instructions to optionally crop and/or resize an image.
    The crop is done first. A crop is only performed if the image matches the trigger extent."""

    target_extent: Extent | None
    crop: tuple[Extent, Bounds] | None = None


no_reshape = ImageReshape(None, None)


class ImageOutput:
    """Wraps an image/mask or the output of a loaded image/mask.
    Allows to consume the image with optional resizing and cropping.
    Resize and crop operations are cached to avoid duplicating nodes in the workflow."""

    def __init__(self, image: Image | Output | None, is_mask: bool = False):
        if isinstance(image, Output):
            self.image = None
            self._output = image
        else:
            self.image = image
            self._output = None
        self.is_mask = is_mask
        self._scaled: Output | None = None
        self._cropped: Output | None = None
        self._scale: Extent | None = None
        self._bounds: Bounds | None = None

    def load(
        self,
        w: ComfyWorkflow,
        reshape: Extent | ImageReshape = no_reshape,
        default_image: Output | None = None,
    ):
        if isinstance(reshape, Extent):
            reshape = ImageReshape(reshape)

        if self._output is None:
            if self.image is None:
                self._output = ensure(default_image)
                if reshape.target_extent is not None:
                    self._output = w.scale_image(self._output, reshape.target_extent)
            elif self.is_mask:
                self._output = w.load_mask(self.image)
            else:
                self._output = w.load_image(self.image)

        result = self._output
        if self.image is None:
            return result  # default image
        extent = self.image.extent

        if reshape.crop is not None:
            trigger_extent, crop_bounds = reshape.crop
            if extent == trigger_extent:
                if self._cropped is None or self._bounds != crop_bounds:
                    if self.is_mask:
                        self._cropped = w.crop_mask(result, crop_bounds)
                    else:
                        self._cropped = w.crop_image(result, crop_bounds)
                    self._bounds = crop_bounds
                result = self._cropped
                extent = crop_bounds.extent

        if reshape.target_extent is not None and extent != reshape.target_extent:
            if self._scaled is None or self._scale != reshape.target_extent:
                if self.is_mask:
                    self._scaled = w.scale_mask(result, reshape.target_extent)
                else:
                    self._scaled = w.scale_control_image(result, reshape.target_extent)
                self._scale = reshape.target_extent
            result = self._scaled

        return result

    def __bool__(self):
        return self.image is not None


@dataclass
class Control:
    mode: ControlMode
    image: ImageOutput
    mask: ImageOutput | None = None
    strength: float = 1.0
    range: tuple[float, float] = (0.0, 1.0)

    @staticmethod
    def from_input(i: ControlInput):
        return Control(i.mode, ImageOutput(i.image), None, i.strength, i.range)


class Clip(NamedTuple):
    model: Output
    arch: Arch


class TextPrompt:
    text: str
    language: str
    # Cached values to avoid re-encoding the same text for multiple regions and passes
    _output: Output | None = None
    _clip: Clip | None = None  # can be different due to Lora hooks

    def __init__(self, text: str, language: str):
        self.text = text
        self.language = language

    def encode(self, w: ComfyWorkflow, clip: Clip, style_prompt: str | None = None):
        text = self.text
        if text != "" and style_prompt:
            text = merge_prompt(text, style_prompt, self.language)
        if self._output is None or self._clip != clip:
            if text and self.language:
                text = w.translate(text)
            self._output = w.clip_text_encode(clip.model, text)
            if text == "" and clip.arch is not Arch.sd15:
                self._output = w.conditioning_zero_out(self._output)
            self._clip = clip
        return self._output


@dataclass
class Region:
    mask: ImageOutput
    bounds: Bounds
    positive: TextPrompt
    control: list[Control] = field(default_factory=list)
    loras: list[LoraInput] = field(default_factory=list)
    is_background: bool = False
    clip: Clip | None = None

    @staticmethod
    def from_input(i: RegionInput, index: int, language: str):
        control = [Control.from_input(c) for c in i.control]
        mask = ImageOutput(i.mask, is_mask=True)
        return Region(
            mask,
            i.bounds,
            TextPrompt(i.positive, language),
            control,
            i.loras,
            is_background=index == 0,
        )

    def patch_clip(self, w: ComfyWorkflow, clip: Clip):
        if self.clip is None:
            self.clip = clip
            if len(self.loras) > 0:
                hooks = w.create_hook_lora([(lora.name, lora.strength) for lora in self.loras])
                self.clip = Clip(w.set_clip_hooks(clip.model, hooks), clip.arch)
        return self.clip

    def encode_prompt(self, w: ComfyWorkflow, clip: Clip, style_prompt: str | None = None):
        return self.positive.encode(w, self.patch_clip(w, clip), style_prompt)

    def copy(self):
        control = [copy(c) for c in self.control]
        loras = copy(self.loras)
        return Region(
            self.mask, self.bounds, self.positive, control, loras, self.is_background, self.clip
        )


@dataclass
class Conditioning:
    positive: TextPrompt
    negative: TextPrompt
    control: list[Control] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    style_prompt: str = ""

    @staticmethod
    def from_input(i: ConditioningInput):
        return Conditioning(
            TextPrompt(i.positive, i.language),
            TextPrompt(i.negative, i.language),
            [Control.from_input(c) for c in i.control],
            [Region.from_input(r, idx, i.language) for idx, r in enumerate(i.regions)],
            i.style,
        )

    def copy(self):
        return Conditioning(
            self.positive,
            self.negative,
            [copy(c) for c in self.control],
            [r.copy() for r in self.regions],
            self.style_prompt,
        )

    def downscale(self, original: Extent, target: Extent):
        downscale_control_images(self.all_control, original, target)

    @property
    def all_control(self):
        return self.control + [c for r in self.regions for c in r.control]

    @property
    def language(self):
        return self.positive.language


def downscale_control_images(
    control_layers: list[Control] | list[ControlInput], original: Extent, target: Extent
):
    # Meant to be called during preperation, when desired generation resolution is lower than canvas size:
    # no need to encode and send the full resolution images to the server.
    if original.width > target.width and original.height > target.height:
        for control in control_layers:
            assert isinstance(control.image, Image)
            # Only scale if control image resolution matches canvas resolution
            if control.image.extent == original and control.mode is not ControlMode.canny_edge:
                if isinstance(control, ControlInput):
                    control.image = Image.scale(control.image, target)
                elif control.image.image:
                    control.image.image = Image.scale(control.image.image, target)


def downscale_all_control_images(cond: ConditioningInput, original: Extent, target: Extent):
    downscale_control_images(cond.control, original, target)
    for region in cond.regions:
        downscale_control_images(region.control, original, target)


def encode_text_prompt(
    w: ComfyWorkflow,
    cond: Conditioning,
    clip: Clip,
    regions: Output | None,
):
    if len(cond.regions) <= 1 or all(len(r.loras) == 0 for r in cond.regions):
        positive = cond.positive.encode(w, clip, cond.style_prompt)
        negative = cond.negative.encode(w, clip)
        return positive, negative

    assert regions is not None
    positive = None
    negative = None
    region_masks = w.list_region_masks(regions)

    for i, region in enumerate(cond.regions):
        region_positive = region.encode_prompt(w, clip, cond.style_prompt)
        region_negative = cond.negative.encode(w, region.patch_clip(w, clip))
        mask = w.mask_batch_element(region_masks, i)
        positive, negative = w.combine_masked_conditioning(
            region_positive, region_negative, positive, negative, mask
        )

    assert positive is not None and negative is not None
    return positive, negative


def apply_attention_mask(
    w: ComfyWorkflow,
    model: Output,
    cond: Conditioning,
    clip: Clip,
    shape: Extent | ImageReshape = no_reshape,
):
    if len(cond.regions) == 0:
        return model, None

    if len(cond.regions) == 1:
        region = cond.regions[0]
        cond.positive = region.positive
        cond.control += region.control
        return model, None

    bottom_region = cond.regions[0]
    if bottom_region.is_background:
        regions = w.background_region(bottom_region.encode_prompt(w, clip, cond.style_prompt))
        remaining = cond.regions[1:]
    else:
        regions = w.background_region(cond.positive.encode(w, clip, cond.style_prompt))
        remaining = cond.regions

    for region in remaining:
        mask = region.mask.load(w, shape)
        prompt = region.encode_prompt(w, clip, cond.style_prompt)
        regions = w.define_region(regions, mask, prompt)

    model = w.attention_mask(model, regions)
    return model, regions


def apply_control(
    w: ComfyWorkflow,
    model: Output,
    positive: Output,
    negative: Output,
    control_layers: list[Control],
    shape: Extent | ImageReshape,
    vae: Output,
    models: ModelDict,
):
    models = models.control
    control_lora: ControlMode | None = None

    for control in (c for c in control_layers if c.mode.is_control_net):
        image = control.image.load(w, shape)
        if control.mode is ControlMode.inpaint and models.arch is Arch.sd15:
            assert control.mask is not None, "Inpaint control requires a mask"
            image = w.inpaint_preprocessor(image, control.mask.load(w))
        if control.mode.is_lines:  # ControlNet expects white lines on black background
            image = w.invert_image(image)

        if lora := models.lora.find(control.mode):
            if control_lora is not None:
                raise Exception(
                    _("The following control layers cannot be used together:")
                    + f" {control_lora.text}, {control.mode.text}"
                )
            control_lora = control.mode
            model = w.load_lora_model(model, lora, control.strength)
            positive, negative, __ = w.instruct_pix_to_pix_conditioning(
                positive, negative, vae, image
            )
            continue

        if cn_model := models.find(control.mode):
            controlnet = w.load_controlnet(cn_model)
        elif cn_model := models.find(ControlMode.universal):
            controlnet = w.load_controlnet(cn_model)
            controlnet = w.set_controlnet_type(controlnet, control.mode)
        else:
            raise Exception(f"ControlNet model not found for mode {control.mode}")

        if control.mode is ControlMode.inpaint and models.arch is Arch.flux:
            assert control.mask is not None, "Inpaint control requires a mask"
            mask = control.mask.load(w)
            positive, negative = w.apply_controlnet_inpainting(
                positive, negative, controlnet, vae, image, mask, control.strength, control.range
            )
        else:
            positive, negative = w.apply_controlnet(
                positive, negative, controlnet, image, vae, control.strength, control.range
            )

    positive = apply_style_models(w, positive, control_layers, models)

    return model, positive, negative


def apply_style_models(
    w: ComfyWorkflow, cond: Output, control_layers: list[Control], models: ModelDict
):
    if models.arch is Arch.flux:
        references: Output | None = None

        for control in (c for c in control_layers if c.mode is ControlMode.reference):
            image = control.image.load(w)
            references = w.define_reference_image(
                references, image, control.strength, control.range
            )
        if references is not None:
            clip_vision_model = models.all.resource(ResourceKind.clip_vision, "redux", Arch.flux)
            clip_vision = w.load_clip_vision(clip_vision_model)
            redux = w.load_style_model(models.ip_adapter[ControlMode.reference])
            cond = w.apply_reference_images(cond, clip_vision, redux, references)

    return cond


def apply_ip_adapter(
    w: ComfyWorkflow,
    model: Output,
    control_layers: list[Control],
    models: ModelDict,
    mask: Output | None = None,
):
    if models.arch is Arch.flux:
        return model  # No IP-adapter for Flux, using Style model instead

    models = models.ip_adapter

    # Create a separate embedding for each face ID (though more than 1 is questionable)
    face_layers = [c for c in control_layers if c.mode is ControlMode.face]
    if len(face_layers) > 0:
        clip_vision = w.load_clip_vision(models.clip_vision)
        ip_adapter = w.load_ip_adapter(models[ControlMode.face])
        insight_face = w.load_insight_face()
        for control in face_layers:
            model = w.apply_ip_adapter_face(
                model,
                ip_adapter,
                clip_vision,
                insight_face,
                control.image.load(w),
                control.strength,
                range=control.range,
                mask=mask,
            )

    # Encode images with their weights into a batch and apply IP-adapter to the model once
    def encode_and_apply_ip_adapter(model: Output, control_layers: list[Control], weight_type: str):
        clip_vision = w.load_clip_vision(models.clip_vision)
        ip_adapter = w.load_ip_adapter(models[ControlMode.reference])
        embeds: list[Output] = []
        range = (0.99, 0.01)

        for control in control_layers:
            if len(embeds) >= 5:
                raise Exception(_("Too many control layers of type") + f" '{mode.text}' (max 5)")
            img = control.image.load(w)
            embeds.append(w.encode_ip_adapter(img, control.strength, ip_adapter, clip_vision)[0])
            range = (min(range[0], control.range[0]), max(range[1], control.range[1]))

        combined = w.combine_ip_adapter_embeds(embeds) if len(embeds) > 1 else embeds[0]
        return w.apply_ip_adapter(
            model, ip_adapter, clip_vision, combined, 1.0, weight_type, range, mask
        )

    modes = [
        (ControlMode.reference, "linear"),
        (ControlMode.style, "style transfer"),
        (ControlMode.composition, "composition"),
    ]
    # Chain together different IP-adapter weight types.
    for mode, weight_type in modes:
        ref_layers = [c for c in control_layers if c.mode is mode]
        if len(ref_layers) > 0:
            model = encode_and_apply_ip_adapter(model, ref_layers, weight_type)

    return model


def apply_regional_ip_adapter(
    w: ComfyWorkflow,
    model: Output,
    regions: list[Region],
    shape: Extent | ImageReshape,
    models: ModelDict,
):
    for region in (r for r in regions if r.mask):
        model = apply_ip_adapter(w, model, region.control, models, region.mask.load(w, shape))
    return model


def scale(
    extent: Extent,
    target: Extent,
    mode: ScaleMode,
    w: ComfyWorkflow,
    image: Output,
    models: ModelDict,
):
    """Handles scaling images from `extent` to `target` resolution.
    Uses either lanczos or a fast upscaling model."""

    if mode is ScaleMode.none:
        return image
    elif mode is ScaleMode.resize:
        return w.scale_image(image, target)
    else:
        assert mode is ScaleMode.upscale_fast
        ratio = target.pixel_count / extent.pixel_count
        factor = max(2, min(4, math.ceil(math.sqrt(ratio))))
        upscale_model = w.load_upscale_model(models.upscale[UpscalerName.fast_x(factor)])
        image = w.upscale_image(upscale_model, image)
        image = w.scale_image(image, target)
        return image


def scale_to_initial(
    extent: ScaledExtent, w: ComfyWorkflow, image: Output, models: ModelDict, is_mask=False
):
    if is_mask and extent.target != extent.initial:
        return w.scale_mask(image, extent.initial)
    elif not is_mask:
        return scale(extent.input, extent.initial, extent.initial_scaling, w, image, models)
    else:
        assert is_mask and extent.initial_scaling is ScaleMode.none
        return image


def scale_to_target(extent: ScaledExtent, w: ComfyWorkflow, image: Output, models: ModelDict):
    return scale(extent.desired, extent.target, extent.target_scaling, w, image, models)


def scale_refine_and_decode(
    extent: ScaledExtent,
    w: ComfyWorkflow,
    cond: Conditioning,
    sampling: SamplingInput,
    latent: Output,
    model: Output,
    clip: Clip,
    vae: Output,
    models: ModelDict,
    tiled_vae: bool,
):
    """Handles scaling images from `initial` to `desired` resolution.
    If it is a substantial upscale, runs a high-res SD refinement pass.
    Takes latent as input and returns a decoded image."""

    mode = extent.refinement_scaling
    if mode in [ScaleMode.none, ScaleMode.resize, ScaleMode.upscale_fast]:
        decoded = vae_decode(w, vae, latent, tiled_vae)
        return scale(extent.initial, extent.desired, mode, w, decoded, models)

    model, regions = apply_attention_mask(w, model, cond, clip, extent.desired)
    model = apply_regional_ip_adapter(w, model, cond.regions, extent.desired, models)

    if mode is ScaleMode.upscale_small:
        upscaler = models.upscale[UpscalerName.fast_2x]
    else:
        assert mode is ScaleMode.upscale_quality
        upscaler = models.upscale[UpscalerName.default]

    upscale_model = w.load_upscale_model(upscaler)
    decoded = vae_decode(w, vae, latent, tiled_vae)
    upscale = w.upscale_image(upscale_model, decoded)
    upscale = w.scale_image(upscale, extent.desired)
    latent = w.vae_encode(vae, upscale)
    params = _sampler_params(sampling, strength=0.4)

    positive, negative = encode_text_prompt(w, cond, clip, regions)
    model, positive, negative = apply_control(
        w, model, positive, negative, cond.all_control, extent.desired, vae, models
    )
    result = w.sampler_custom_advanced(model, positive, negative, latent, models.arch, **params)
    image = vae_decode(w, vae, result, tiled_vae)
    return image


def ensure_minimum_extent(w: ComfyWorkflow, image: Output, extent: Extent, min_extent: int):
    # For example, upscale with model requires minimum size of 32x32
    if extent.shortest_side < min_extent:
        image = w.scale_image(image, extent * (min_extent / extent.shortest_side))
    return image


class MiscParams(NamedTuple):
    batch_count: int
    nsfw_filter: float


def generate(
    w: ComfyWorkflow,
    checkpoint: CheckpointInput,
    extent: ScaledExtent,
    cond: Conditioning,
    sampling: SamplingInput,
    misc: MiscParams,
    models: ModelDict,
):
    model, clip, vae = load_checkpoint_with_lora(w, checkpoint, models.all)
    model = apply_ip_adapter(w, model, cond.control, models)
    model_orig = copy(model)
    model, regions = apply_attention_mask(w, model, cond, clip, extent.initial)
    model = apply_regional_ip_adapter(w, model, cond.regions, extent.initial, models)
    latent = w.empty_latent_image(extent.initial, models.arch, misc.batch_count)
    positive, negative = encode_text_prompt(w, cond, clip, regions)
    model, positive, negative = apply_control(
        w, model, positive, negative, cond.all_control, extent.initial, vae, models
    )
    out_latent = w.sampler_custom_advanced(
        model, positive, negative, latent, models.arch, **_sampler_params(sampling)
    )
    out_image = scale_refine_and_decode(
        extent, w, cond, sampling, out_latent, model_orig, clip, vae, models, checkpoint.tiled_vae
    )
    out_image = w.nsfw_filter(out_image, sensitivity=misc.nsfw_filter)
    out_image = scale_to_target(extent, w, out_image, models)
    w.send_image(out_image)
    return w


def fill_masked(w: ComfyWorkflow, image: Output, mask: Output, fill: FillMode, models: ModelDict):
    if fill is FillMode.blur:
        return w.blur_masked(image, mask, 65, falloff=9)
    elif fill is FillMode.border:
        image = w.fill_masked(image, mask, "navier-stokes")
        return w.blur_masked(image, mask, 65)
    elif fill is FillMode.neutral:
        return w.fill_masked(image, mask, "neutral", falloff=9)
    elif fill is FillMode.inpaint:
        model = w.load_inpaint_model(models.inpaint["default"])
        return w.inpaint_image(model, image, mask)
    elif fill is FillMode.replace:
        return w.fill_masked(image, mask, "neutral")
    return image


def apply_grow_feather(w: ComfyWorkflow, mask: Output, inpaint: InpaintParams):
    if inpaint.grow or inpaint.feather:
        mask = w.expand_mask(mask, inpaint.grow, inpaint.feather)
    return mask


def detect_inpaint(
    mode: InpaintMode,
    bounds: Bounds,
    sd_ver: Arch,
    prompt: str,
    control: list[ControlInput],
    strength: float,
):
    assert mode is not InpaintMode.automatic
    result = InpaintParams(mode, bounds)

    if sd_ver is Arch.sd15:
        result.use_inpaint_model = strength > 0.5
        result.use_condition_mask = (
            mode is InpaintMode.add_object
            and prompt != ""
            and not any(c.mode.is_structural for c in control)
        )
    elif sd_ver is Arch.sdxl:
        result.use_inpaint_model = strength > 0.8
    elif sd_ver is Arch.flux:
        result.use_inpaint_model = strength == 1.0

    is_ref_mode = mode in [InpaintMode.fill, InpaintMode.expand]
    result.use_reference = is_ref_mode and prompt == ""

    result.fill = {
        InpaintMode.fill: FillMode.blur,
        InpaintMode.expand: FillMode.border,
        InpaintMode.add_object: FillMode.neutral,
        InpaintMode.remove_object: FillMode.inpaint,
        InpaintMode.replace_background: FillMode.replace,
    }[mode]
    return result


def inpaint_control(image: Output | ImageOutput, mask: Output | ImageOutput, arch: Arch):
    strength, range = 1.0, (0.0, 1.0)
    if arch is Arch.flux:
        strength, range = 0.9, (0.0, 0.5)
    if isinstance(image, Output):
        image = ImageOutput(image)
    if isinstance(mask, Output):
        mask = ImageOutput(mask, is_mask=True)
    return Control(ControlMode.inpaint, image, mask, strength, range)


def inpaint(
    w: ComfyWorkflow,
    images: ImageInput,
    checkpoint: CheckpointInput,
    cond: Conditioning,
    sampling: SamplingInput,
    params: InpaintParams,
    crop_upscale_extent: Extent,
    misc: MiscParams,
    models: ModelDict,
):
    target_bounds = params.target_bounds
    extent = ScaledExtent.from_input(images.extent)  # for initial generation with large context

    is_inpaint_model = params.use_inpaint_model and models.control.find(ControlMode.inpaint) is None
    if is_inpaint_model and models.arch is Arch.flux:
        checkpoint.dynamic_caching = False  # doesn't seem to work with Flux fill model
        sampling.cfg_scale = 30  # set Flux guidance to 30 (typical values don't work well)

    model, clip, vae = load_checkpoint_with_lora(w, checkpoint, models.all)
    model = w.differential_diffusion(model)
    model_orig = copy(model)

    upscale_extent = ScaledExtent(  # after crop to the masked region
        Extent(0, 0), Extent(0, 0), crop_upscale_extent, target_bounds.extent
    )
    initial_bounds = extent.convert(target_bounds, "target", "initial")

    in_image = w.load_image(ensure(images.initial_image))
    in_image = scale_to_initial(extent, w, in_image, models)
    in_mask = w.load_mask(ensure(images.hires_mask))
    in_mask = apply_grow_feather(w, in_mask, params)
    initial_mask = scale_to_initial(extent, w, in_mask, models, is_mask=True)
    cropped_mask = w.crop_mask(in_mask, target_bounds)

    cond_base = cond.copy()
    model, regions = apply_attention_mask(w, model, cond_base, clip, extent.initial)

    if params.use_reference:
        reference = get_inpaint_reference(ensure(images.initial_image), initial_bounds) or in_image
        cond_base.control.append(
            Control(ControlMode.reference, ImageOutput(reference), None, 0.5, (0.2, 0.8))
        )
    inpaint_mask = ImageOutput(initial_mask, is_mask=True)
    if params.use_inpaint_model and models.control.find(ControlMode.inpaint) is not None:
        cond_base.control.append(inpaint_control(in_image, inpaint_mask, models.arch))
    if params.use_condition_mask and len(cond_base.regions) == 0:
        base_prompt = TextPrompt(merge_prompt("", cond_base.style_prompt), cond.language)
        cond_base.regions = [
            Region(ImageOutput(None), Bounds(0, 0, *extent.initial), base_prompt, []),
            Region(inpaint_mask, initial_bounds, cond_base.positive, []),
        ]
    in_image = fill_masked(w, in_image, initial_mask, params.fill, models)

    model = apply_ip_adapter(w, model, cond_base.control, models)
    model = apply_regional_ip_adapter(w, model, cond_base.regions, extent.initial, models)
    positive, negative = encode_text_prompt(w, cond_base, clip, regions)
    model, positive, negative = apply_control(
        w, model, positive, negative, cond_base.all_control, extent.initial, vae, models
    )
    if params.use_inpaint_model and models.arch is Arch.sdxl:
        positive, negative, latent_inpaint, latent = w.vae_encode_inpaint_conditioning(
            vae, in_image, initial_mask, positive, negative
        )
        inpaint_patch = w.load_fooocus_inpaint(**models.fooocus_inpaint)
        inpaint_model = w.apply_fooocus_inpaint(model, inpaint_patch, latent_inpaint)
    elif is_inpaint_model:
        positive, negative, latent_inpaint, latent = w.vae_encode_inpaint_conditioning(
            vae, in_image, initial_mask, positive, negative
        )
        inpaint_model = model
    else:
        latent = w.vae_encode(vae, in_image)
        latent = w.set_latent_noise_mask(latent, initial_mask)
        inpaint_model = model

    latent = w.batch_latent(latent, misc.batch_count)
    out_latent = w.sampler_custom_advanced(
        inpaint_model, positive, negative, latent, models.arch, **_sampler_params(sampling)
    )

    if extent.refinement_scaling in [ScaleMode.upscale_small, ScaleMode.upscale_quality]:
        model = model_orig
        if extent.refinement_scaling is ScaleMode.upscale_small:
            upscaler = models.upscale[UpscalerName.fast_2x]
        else:
            upscaler = models.upscale[UpscalerName.default]
        upscale_mask = cropped_mask
        if crop_upscale_extent != target_bounds.extent:
            upscale_mask = w.scale_mask(cropped_mask, crop_upscale_extent)
        sampler_params = _sampler_params(sampling, strength=0.4)
        upscale_model = w.load_upscale_model(upscaler)
        upscale = vae_decode(w, vae, out_latent, checkpoint.tiled_vae)
        upscale = w.crop_image(upscale, initial_bounds)
        upscale = ensure_minimum_extent(w, upscale, initial_bounds.extent, 32)
        upscale = w.upscale_image(upscale_model, upscale)
        upscale = w.scale_image(upscale, upscale_extent.desired)
        latent = w.vae_encode(vae, upscale)
        latent = w.set_latent_noise_mask(latent, upscale_mask)

        cond_upscale = cond.copy()
        shape = ImageReshape(upscale_extent.desired, crop=(extent.target, target_bounds))

        model, regions = apply_attention_mask(w, model, cond_upscale, clip, shape)
        model = apply_regional_ip_adapter(w, model, cond_upscale.regions, shape, models)
        positive_up, negative_up = encode_text_prompt(w, cond_upscale, clip, regions)

        if params.use_inpaint_model and models.control.find(ControlMode.inpaint) is not None:
            hires_image = ImageOutput(images.hires_image)
            cond_upscale.control.append(inpaint_control(hires_image, upscale_mask, models.arch))
        model, positive_up, negative_up = apply_control(
            w, model, positive_up, negative_up, cond_upscale.all_control, shape, vae, models
        )
        out_latent = w.sampler_custom_advanced(
            model, positive_up, negative_up, latent, models.arch, **sampler_params
        )
        out_image = vae_decode(w, vae, out_latent, checkpoint.tiled_vae)
        out_image = scale_to_target(upscale_extent, w, out_image, models)
    else:
        desired_bounds = extent.convert(target_bounds, "target", "desired")
        desired_extent = desired_bounds.extent
        cropped_extent = ScaledExtent(
            desired_extent, desired_extent, desired_extent, target_bounds.extent
        )
        out_image = vae_decode(w, vae, out_latent, checkpoint.tiled_vae)
        out_image = scale(
            extent.initial, extent.desired, extent.refinement_scaling, w, out_image, models
        )
        out_image = w.crop_image(out_image, desired_bounds)
        out_image = scale_to_target(cropped_extent, w, out_image, models)

    out_image = w.nsfw_filter(out_image, sensitivity=misc.nsfw_filter)
    compositing_mask = w.denoise_to_compositing_mask(cropped_mask)
    out_masked = w.apply_mask(out_image, compositing_mask)
    w.send_image(out_masked)
    return w


def refine(
    w: ComfyWorkflow,
    image: Image,
    extent: ScaledExtent,
    checkpoint: CheckpointInput,
    cond: Conditioning,
    sampling: SamplingInput,
    misc: MiscParams,
    models: ModelDict,
):
    model, clip, vae = load_checkpoint_with_lora(w, checkpoint, models.all)
    model = apply_ip_adapter(w, model, cond.control, models)
    model, regions = apply_attention_mask(w, model, cond, clip, extent.initial)
    model = apply_regional_ip_adapter(w, model, cond.regions, extent.initial, models)
    in_image = w.load_image(image)
    in_image = scale_to_initial(extent, w, in_image, models)
    latent = w.vae_encode(vae, in_image)
    latent = w.batch_latent(latent, misc.batch_count)
    positive, negative = encode_text_prompt(w, cond, clip, regions)
    model, positive, negative = apply_control(
        w, model, positive, negative, cond.all_control, extent.desired, vae, models
    )
    sampler = w.sampler_custom_advanced(
        model, positive, negative, latent, models.arch, **_sampler_params(sampling)
    )
    out_image = vae_decode(w, vae, sampler, checkpoint.tiled_vae)
    out_image = w.nsfw_filter(out_image, sensitivity=misc.nsfw_filter)
    out_image = scale_to_target(extent, w, out_image, models)
    w.send_image(out_image)
    return w


def refine_region(
    w: ComfyWorkflow,
    images: ImageInput,
    checkpoint: CheckpointInput,
    cond: Conditioning,
    sampling: SamplingInput,
    inpaint: InpaintParams,
    misc: MiscParams,
    models: ModelDict,
):
    extent = ScaledExtent.from_input(images.extent)

    model, clip, vae = load_checkpoint_with_lora(w, checkpoint, models.all)
    model = w.differential_diffusion(model)
    model = apply_ip_adapter(w, model, cond.control, models)
    model_orig = copy(model)
    model, regions = apply_attention_mask(w, model, cond, clip, extent.initial)
    model = apply_regional_ip_adapter(w, model, cond.regions, extent.initial, models)
    positive, negative = encode_text_prompt(w, cond, clip, regions)

    in_image = w.load_image(ensure(images.initial_image))
    in_image = scale_to_initial(extent, w, in_image, models)
    in_mask = w.load_mask(ensure(images.hires_mask))
    in_mask = apply_grow_feather(w, in_mask, inpaint)
    initial_mask = scale_to_initial(extent, w, in_mask, models, is_mask=True)

    if inpaint.use_inpaint_model and models.control.find(ControlMode.inpaint) is not None:
        cond.control.append(inpaint_control(in_image, initial_mask, models.arch))
    model, positive, negative = apply_control(
        w, model, positive, negative, cond.all_control, extent.initial, vae, models
    )
    if inpaint.use_inpaint_model and models.arch is Arch.sdxl:
        positive, negative, latent_inpaint, latent = w.vae_encode_inpaint_conditioning(
            vae, in_image, initial_mask, positive, negative
        )
        inpaint_patch = w.load_fooocus_inpaint(**models.fooocus_inpaint)
        inpaint_model = w.apply_fooocus_inpaint(model, inpaint_patch, latent_inpaint)
    else:
        latent = w.vae_encode(vae, in_image)
        latent = w.set_latent_noise_mask(latent, initial_mask)
        inpaint_model = model

    latent = w.batch_latent(latent, misc.batch_count)
    out_latent = w.sampler_custom_advanced(
        inpaint_model, positive, negative, latent, models.arch, **_sampler_params(sampling)
    )
    out_image = scale_refine_and_decode(
        extent, w, cond, sampling, out_latent, model_orig, clip, vae, models, checkpoint.tiled_vae
    )
    out_image = w.nsfw_filter(out_image, sensitivity=misc.nsfw_filter)
    out_image = scale_to_target(extent, w, out_image, models)
    if extent.target != inpaint.target_bounds.extent:
        out_image = w.crop_image(out_image, inpaint.target_bounds)
        in_mask = w.crop_mask(in_mask, inpaint.target_bounds)
    compositing_mask = w.denoise_to_compositing_mask(in_mask)
    out_masked = w.apply_mask(out_image, compositing_mask)
    w.send_image(out_masked)
    return w


def create_control_image(
    w: ComfyWorkflow,
    image: Image,
    mode: ControlMode,
    extent: ScaledExtent,
    bounds: Bounds | None = None,
    seed: int = -1,
):
    assert mode not in [ControlMode.reference, ControlMode.face, ControlMode.inpaint]

    current_extent = extent.input
    input = w.load_image(image)
    result = None

    if mode is ControlMode.hands:
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
        current_extent = current_extent.multiple_of(64).at_least(512)
        args = {"image": input, "resolution": current_extent.shortest_side}
        if mode is ControlMode.scribble:
            result = w.add("PiDiNetPreprocessor", 1, **args, safe="enable")
            result = w.add("ScribblePreprocessor", 1, image=result, resolution=args["resolution"])
        elif mode is ControlMode.line_art:
            result = w.add("LineArtPreprocessor", 1, **args, coarse="disable")
        elif mode is ControlMode.soft_edge:
            args["merge_with_lineart"] = "lineart_standard"
            args["lineart_lower_bound"] = 0.0
            args["lineart_upper_bound"] = 1.0
            args["object_min_size"] = 36
            args["object_connectivity"] = 1
            result = w.add("AnyLineArtPreprocessor_aux", 1, **args)
        elif mode is ControlMode.canny_edge:
            result = w.add("CannyEdgePreprocessor", 1, **args, low_threshold=80, high_threshold=200)
        elif mode is ControlMode.depth:
            model = "depth_anything_v2_vitb.pth"
            result = w.add("DepthAnythingV2Preprocessor", 1, **args, ckpt_name=model)
        elif mode is ControlMode.normal:
            result = w.add("BAE-NormalMapPreprocessor", 1, **args)
        elif mode is ControlMode.pose:
            result = w.estimate_pose(**args)
        elif mode is ControlMode.segmentation:
            result = w.add("OneFormer-COCO-SemSegPreprocessor", 1, **args)

        assert result is not None

    if mode.is_lines:
        result = w.invert_image(result)
    if current_extent != extent.target:
        result = w.scale_image(result, extent.target)

    w.send_image(result)
    return w


def upscale_simple(w: ComfyWorkflow, image: Image, model: str, factor: float):
    upscale_model = w.load_upscale_model(model)
    img = w.load_image(image)
    img = w.upscale_image(upscale_model, img)
    if factor != 4.0:
        img = w.scale_image(img, image.extent * factor)
    w.send_image(img)
    return w


def upscale_tiled(
    w: ComfyWorkflow,
    image: Image,
    extent: ExtentInput,
    checkpoint: CheckpointInput,
    cond: Conditioning,
    sampling: SamplingInput,
    upscale: UpscaleInput,
    misc: MiscParams,
    models: ModelDict,
):
    upscale_factor = extent.initial.width / extent.input.width
    if upscale.tile_overlap >= 0:
        layout = TileLayout(extent.initial, extent.desired.width, upscale.tile_overlap)
    else:
        layout = TileLayout.from_denoise_strength(
            extent.initial, extent.desired.width, sampling.denoise_strength
        )

    model, clip, vae = load_checkpoint_with_lora(w, checkpoint, models.all)
    model = apply_ip_adapter(w, model, cond.control, models)

    in_image = w.load_image(image)
    if upscale.model:
        upscale_model = w.load_upscale_model(upscale.model)
        upscaled = w.upscale_image(upscale_model, in_image)
    else:
        upscaled = in_image
    if extent.input != extent.initial:
        upscaled = w.scale_image(upscaled, extent.initial)
    tile_layout = w.create_tile_layout(upscaled, layout.min_size, layout.padding, layout.blending)

    def tiled_control(control: Control, index: int):
        img = control.image.load(w, extent.initial, default_image=in_image)
        img = w.extract_image_tile(img, tile_layout, index)
        return Control(control.mode, ImageOutput(img), None, control.strength, control.range)

    def tiled_region(region: Region, index: int, tile_bounds: Bounds):
        region_bounds = Bounds.scale(region.bounds, upscale_factor)
        coverage = Bounds.intersection(tile_bounds, region_bounds).area / tile_bounds.area
        if coverage > 0.1:
            if region.mask:
                mask = region.mask.load(w, extent.initial)
                mask = w.extract_mask_tile(mask, tile_layout, index)
                region.mask = ImageOutput(mask, is_mask=True)
            return region
        return None

    out_image = upscaled
    for i in range(layout.total_tiles):
        bounds = layout.bounds(i)
        tile_image = w.extract_image_tile(upscaled, tile_layout, i)
        tile_mask = w.generate_tile_mask(tile_layout, i)

        tile_cond = cond.copy()
        regions = [tiled_region(r, i, bounds) for r in tile_cond.regions]
        tile_cond.regions = [r for r in regions if r is not None]
        tile_model, regions = apply_attention_mask(w, model, tile_cond, clip)
        tile_model = apply_regional_ip_adapter(w, tile_model, tile_cond.regions, no_reshape, models)
        positive, negative = encode_text_prompt(w, tile_cond, clip, regions)

        control = [tiled_control(c, i) for c in tile_cond.all_control]
        tile_model, positive, negative = apply_control(
            w, tile_model, positive, negative, control, no_reshape, vae, models
        )

        latent = w.vae_encode(vae, tile_image)
        latent = w.set_latent_noise_mask(latent, tile_mask)
        sampler = w.sampler_custom_advanced(
            tile_model, positive, negative, latent, models.arch, **_sampler_params(sampling)
        )
        tile_result = vae_decode(w, vae, sampler, checkpoint.tiled_vae)
        out_image = w.merge_image_tile(out_image, tile_layout, i, tile_result)

    out_image = w.nsfw_filter(out_image, sensitivity=misc.nsfw_filter)
    if extent.initial != extent.target:
        out_image = scale(extent.initial, extent.target, ScaleMode.resize, w, out_image, models)
    w.send_image(out_image)
    return w


def expand_custom(
    w: ComfyWorkflow,
    input: CustomWorkflowInput,
    images: ImageInput,
    seed: int,
    models: ClientModels,
):
    custom = ComfyWorkflow.from_dict(input.workflow)
    nodes: dict[int, int] = {}  # map old node IDs to new node IDs
    outputs: dict[Output, Input] = {}

    def map_input(input: Input):
        if isinstance(input, Output):
            mapped = outputs.get(input)
            if mapped is not None:
                return mapped
            else:
                return Output(nodes[input.node], input.output)
        return input

    def get_param(node: ComfyNode, expected_type: type | tuple[type, type] | None = None):
        name = node.input("name", "")
        value = input.params.get(name)
        if value is None:
            raise Exception(f"Missing required parameter '{name}' for custom workflow")
        if expected_type and not isinstance(value, expected_type):
            raise Exception(f"Parameter '{name}' must be of type {expected_type}")
        return value

    for node in custom:
        match node.type:
            case "ETN_KritaCanvas":
                image = ensure(images.initial_image)
                outputs[node.output(0)] = w.load_image(image)
                outputs[node.output(1)] = image.width
                outputs[node.output(2)] = image.height
                outputs[node.output(3)] = seed
            case "ETN_KritaSelection":
                outputs[node.output(0)] = w.load_mask(ensure(images.hires_mask))
            case "ETN_Parameter":
                outputs[node.output(0)] = get_param(node)
            case "ETN_KritaImageLayer":
                img, mask = w.load_image_and_mask(get_param(node, (Image, ImageCollection)))
                outputs[node.output(0)] = img
                outputs[node.output(1)] = mask
            case "ETN_KritaMaskLayer":
                outputs[node.output(0)] = w.load_mask(get_param(node, (Image, ImageCollection)))
            case "ETN_KritaStyle":
                style: Style = get_param(node, Style)
                is_live = node.input("sampler_preset", "auto") == "live"
                checkpoint_input = style.get_models(models.checkpoints.keys())
                sampling = _sampling_from_style(style, 1.0, is_live)
                model, clip, vae = load_checkpoint_with_lora(w, checkpoint_input, models)
                outputs[node.output(0)] = model
                outputs[node.output(1)] = clip.model
                outputs[node.output(2)] = vae
                outputs[node.output(3)] = style.style_prompt
                outputs[node.output(4)] = style.negative_prompt
                outputs[node.output(5)] = sampling.sampler
                outputs[node.output(6)] = sampling.scheduler
                outputs[node.output(7)] = sampling.total_steps
                outputs[node.output(8)] = sampling.cfg_scale
            case _:
                mapped_inputs = {k: map_input(v) for k, v in node.inputs.items()}
                mapped = ComfyNode(node.id, node.type, mapped_inputs)
                nodes[node.id] = w.copy(mapped).node

    w.guess_sample_count()
    return w


###################################################################################################


def prepare(
    kind: WorkflowKind,
    canvas: Image | Extent,
    cond: ConditioningInput,
    style: Style,
    seed: int,
    models: ClientModels,
    files: FileLibrary,
    perf: PerformanceSettings,
    mask: Mask | None = None,
    strength: float = 1.0,
    inpaint: InpaintParams | None = None,
    upscale_factor: float = 1.0,
    upscale: UpscaleInput | None = None,
    is_live: bool = False,
) -> WorkflowInput:
    """
    Takes UI model state, prepares images, normalizes inputs, and returns a WorkflowInput object
    which can be compared and serialized.
    """
    i = WorkflowInput(kind)
    i.conditioning = cond
    i.conditioning.positive, extra_loras = extract_loras(i.conditioning.positive, files.loras)
    i.conditioning.negative = merge_prompt(cond.negative, style.negative_prompt, cond.language)
    i.conditioning.style = style.style_prompt
    for idx, region in enumerate(i.conditioning.regions):
        assert region.mask or idx == 0, "Only the first/bottom region can be without a mask"
        region.positive, region.loras = extract_loras(region.positive, files.loras)
        region.loras = [l for l in region.loras if l not in extra_loras]
    i.sampling = _sampling_from_style(style, strength, is_live)
    i.sampling.seed = seed
    i.models = style.get_models(models.checkpoints.keys())
    i.conditioning.positive += _collect_lora_triggers(i.models.loras, files)
    i.models.loras = unique(i.models.loras + extra_loras, key=lambda l: l.name)
    i.models.dynamic_caching = perf.dynamic_caching
    i.models.tiled_vae = perf.tiled_vae
    arch = i.models.version = resolve_arch(style, models)

    _check_server_has_models(i.models, i.conditioning.regions, models, files, style.name)
    _check_inpaint_model(inpaint, arch, models)

    model_set = models.for_arch(arch)
    has_ip_adapter = model_set.ip_adapter.find(ControlMode.reference) is not None
    i.models.loras += _get_sampling_lora(style, is_live, model_set, models)
    all_control = cond.control + [c for r in cond.regions for c in r.control]
    face_weight = median_or_zero(c.strength for c in all_control if c.mode is ControlMode.face)
    if face_weight > 0:
        i.models.loras.append(LoraInput(model_set.lora["face"], 0.65 * face_weight))

    if kind is WorkflowKind.generate:
        assert isinstance(canvas, Extent)
        i.images, i.batch_count = resolution.prepare_extent(
            canvas, arch, ensure(style), perf, downscale=not is_live
        )
        downscale_all_control_images(i.conditioning, canvas, i.images.extent.desired)

    elif kind is WorkflowKind.inpaint:
        assert isinstance(canvas, Image) and mask and inpaint and style
        i.images, _ = resolution.prepare_image(canvas, arch, style, perf)
        i.images.hires_mask = mask.to_image(canvas.extent)
        upscale_extent, _ = resolution.prepare_extent(
            mask.bounds.extent, arch, style, perf, downscale=False
        )
        i.inpaint = InpaintParams.clamped(inpaint)
        i.inpaint.use_reference = inpaint.use_reference and has_ip_adapter
        i.crop_upscale_extent = upscale_extent.extent.desired
        largest_extent = Extent.largest(i.images.extent.initial, upscale_extent.extent.desired)
        i.batch_count = resolution.compute_batch_size(largest_extent, 512, perf.batch_size)
        scaling = ScaledExtent.from_input(i.images.extent).refinement_scaling
        if scaling in [ScaleMode.upscale_small, ScaleMode.upscale_quality]:
            i.images.hires_image = Image.crop(canvas, i.inpaint.target_bounds)
        if inpaint.mode is InpaintMode.remove_object and i.conditioning.positive == "":
            i.conditioning.positive = "background scenery"

    elif kind is WorkflowKind.refine:
        assert isinstance(canvas, Image) and style
        i.images, i.batch_count = resolution.prepare_image(
            canvas, arch, style, perf, downscale=False
        )
        downscale_all_control_images(i.conditioning, canvas.extent, i.images.extent.desired)

    elif kind is WorkflowKind.refine_region:
        assert isinstance(canvas, Image) and mask and inpaint and style
        allow_2pass = strength >= 0.7
        i.images, i.batch_count = resolution.prepare_image(
            canvas, arch, style, perf, downscale=allow_2pass
        )
        i.images.hires_mask = mask.to_image(canvas.extent)
        i.inpaint = InpaintParams.clamped(inpaint)
        downscale_all_control_images(i.conditioning, canvas.extent, i.images.extent.desired)

    elif kind is WorkflowKind.upscale_tiled:
        assert isinstance(canvas, Image) and style
        target_extent = canvas.extent * upscale_factor
        if style.preferred_resolution > 0:
            tile_size = style.preferred_resolution
        else:
            tile_size = 1024 if arch.is_sdxl_like else 800
        tile_size = max(tile_size, target_extent.longest_side // 12)  # max 12x12 tiles total
        tile_size = multiple_of(tile_size - 128, 8)
        tile_size = Extent(tile_size, tile_size)
        extent = ExtentInput(canvas.extent, target_extent.multiple_of(8), tile_size, target_extent)
        i.images = ImageInput(extent, canvas)
        assert upscale is not None
        i.upscale = upscale
        i.upscale.model = i.upscale.model if upscale_factor > 1 else ""
        i.batch_count = 1

    else:
        raise Exception(f"Workflow {kind.name} not supported by this constructor")

    i.batch_count = 1 if is_live else i.batch_count
    i.nsfw_filter = settings.nsfw_filter
    return i


def prepare_upscale_simple(image: Image, model: str, factor: float):
    target_extent = image.extent * factor
    extent = ExtentInput(image.extent, image.extent, target_extent, target_extent)
    i = WorkflowInput(WorkflowKind.upscale_simple, ImageInput(extent, image))
    i.upscale = UpscaleInput(model)
    return i


def prepare_create_control_image(
    image: Image,
    mode: ControlMode,
    performance_settings: PerformanceSettings,
    bounds: Bounds | None = None,
    seed: int = -1,
) -> WorkflowInput:
    i = WorkflowInput(WorkflowKind.control_image)
    i.control_mode = mode
    i.images = resolution.prepare_control(image, performance_settings)
    if bounds:
        seed = generate_seed() if seed == -1 else seed
        i.inpaint = InpaintParams(InpaintMode.fill, bounds)
        i.sampling = SamplingInput("", "", 1, 1, seed=seed)  # ignored apart from seed
    return i


def create(i: WorkflowInput, models: ClientModels, comfy_mode=ComfyRunMode.server) -> ComfyWorkflow:
    """
    Takes a WorkflowInput object and creates the corresponding ComfyUI workflow prompt.
    This should be a pure function, the workflow is entirely defined by the input.
    """
    workflow = ComfyWorkflow(models.node_inputs, comfy_mode)
    misc = MiscParams(i.batch_count, i.nsfw_filter)

    if i.kind is WorkflowKind.generate:
        return generate(
            workflow,
            ensure(i.models),
            ScaledExtent.from_input(i.extent),
            Conditioning.from_input(ensure(i.conditioning)),
            ensure(i.sampling),
            misc,
            models.for_arch(ensure(i.models).version),
        )
    elif i.kind is WorkflowKind.inpaint:
        return inpaint(
            workflow,
            ensure(i.images),
            ensure(i.models),
            Conditioning.from_input(ensure(i.conditioning)),
            ensure(i.sampling),
            ensure(i.inpaint),
            ensure(i.crop_upscale_extent),
            misc,
            models.for_arch(ensure(i.models).version),
        )
    elif i.kind is WorkflowKind.refine:
        return refine(
            workflow,
            i.image,
            ScaledExtent.from_input(i.extent),
            ensure(i.models),
            Conditioning.from_input(ensure(i.conditioning)),
            ensure(i.sampling),
            misc,
            models.for_arch(ensure(i.models).version),
        )
    elif i.kind is WorkflowKind.refine_region:
        return refine_region(
            workflow,
            ensure(i.images),
            ensure(i.models),
            Conditioning.from_input(ensure(i.conditioning)),
            ensure(i.sampling),
            ensure(i.inpaint),
            misc,
            models.for_arch(ensure(i.models).version),
        )
    elif i.kind is WorkflowKind.upscale_simple:
        return upscale_simple(workflow, i.image, ensure(i.upscale).model, i.upscale_factor)
    elif i.kind is WorkflowKind.upscale_tiled:
        return upscale_tiled(
            workflow,
            i.image,
            i.extent,
            ensure(i.models),
            Conditioning.from_input(ensure(i.conditioning)),
            ensure(i.sampling),
            ensure(i.upscale),
            misc,
            models.for_arch(ensure(i.models).version),
        )
    elif i.kind is WorkflowKind.control_image:
        return create_control_image(
            workflow,
            image=i.image,
            mode=i.control_mode,
            extent=ScaledExtent.from_input(i.extent),
            bounds=i.inpaint.target_bounds if i.inpaint else None,
            seed=i.sampling.seed if i.sampling else -1,
        )
    elif i.kind is WorkflowKind.custom:
        seed = ensure(i.sampling).seed
        return expand_custom(workflow, ensure(i.custom_workflow), ensure(i.images), seed, models)
    else:
        raise ValueError(f"Unsupported workflow kind: {i.kind}")


def _get_sampling_lora(style: Style, is_live: bool, model_set: ModelDict, models: ClientModels):
    sampler_name = style.live_sampler if is_live else style.sampler
    preset = SamplerPresets.instance()[sampler_name]
    if preset.lora:
        file = model_set.lora.find(preset.lora)
        if file is None and preset.lora not in models.loras:
            res = resources.search_path(ResourceKind.lora, model_set.arch, preset.lora)
            if res is None and preset.lora == "lightning":
                raise ValueError(
                    f"The chosen sampler preset '{sampler_name}' requires LoRA "
                    f"'{preset.lora}', which is not supported by {model_set.arch.value}."
                    " Please choose a different sampler."
                )
            elif res is None:
                raise ValueError(
                    _(
                        "Could not find LoRA '{lora}' used by sampler preset '{name}'",
                        lora=preset.lora,
                        name=sampler_name,
                    )
                )
            else:
                raise ValueError(
                    _(
                        "Could not find LoRA '{lora}' ({models}) used by sampler preset '{name}'",
                        lora=preset.lora,
                        name=sampler_name,
                        models=", ".join(res),
                    )
                )
        return [LoraInput(file or preset.lora, 1.0)]
    return []


def _collect_lora_triggers(loras: list[LoraInput], files: FileLibrary):
    def trigger_words(lora: LoraInput) -> str:
        if file := files.loras.find(lora.name):
            return file.meta("lora_triggers", "")
        return ""

    result = " ".join(trigger_words(lora) for lora in loras)
    return " " + result if result else ""


def _check_server_has_loras(
    loras: list[LoraInput], models: ClientModels, files: FileLibrary, style_name: str, arch: Arch
):
    for lora in loras:
        if lora.name not in models.loras:
            if lora_info := files.loras.find_local(lora.name):
                lora.storage_id = lora_info.compute_hash()
                continue  # local file available, can be uploaded to server
            raise ValueError(
                _(
                    "The LoRA '{lora}' used by style '{style}' is not available on the server",
                    lora=lora.name,
                    style=style_name,
                )
            )
        for id, res in models.resources.items():
            lora_arch = ResourceId.parse(id).arch
            if lora.name == res and arch is not lora_arch:
                raise ValueError(
                    _(
                        "Model architecture mismatch for LoRA '{lora}': Cannot use {lora_arch} LoRA with a {checkpoint_arch} checkpoint.",
                        lora=lora.name,
                        lora_arch=lora_arch.value,
                        checkpoint_arch=arch.value,
                    )
                )


def _check_server_has_models(
    input: CheckpointInput,
    regions: list[RegionInput],
    models: ClientModels,
    files: FileLibrary,
    style_name: str,
):
    if input.checkpoint not in models.checkpoints:
        raise ValueError(
            _(
                "The checkpoint '{checkpoint}' used by style '{style}' is not available on the server",
                checkpoint=input.checkpoint,
                style=style_name,
            )
        )

    _check_server_has_loras(input.loras, models, files, style_name, input.version)
    for region in regions:
        _check_server_has_loras(region.loras, models, files, style_name, input.version)

    if input.vae != StyleSettings.vae.default and input.vae not in models.vae:
        raise ValueError(
            _(
                "The VAE '{vae}' used by style '{style}' is not available on the server",
                vae=input.vae,
                style=style_name,
            )
        )


def _check_inpaint_model(inpaint: InpaintParams | None, arch: Arch, models: ClientModels):
    if inpaint and inpaint.use_inpaint_model and arch.has_controlnet_inpaint:
        if models.for_arch(arch).control.find(ControlMode.inpaint) is None:
            if arch is Arch.flux:
                return  # Optional for now, to allow using flux1-fill model instead of inpaint CN
            msg = f"No inpaint model found for {arch.value}."
            res_id = ResourceId(ResourceKind.controlnet, arch, ControlMode.inpaint)
            if res := resources.find_resource(res_id):
                msg += f" Missing '{res.filename}' in folder '{res.folder}'."
            raise ValueError(msg)
