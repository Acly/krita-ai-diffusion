from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Tuple, Literal, overload, Any
from uuid import uuid4
import math
import json

from .image import Bounds, Extent, Image
from .resources import SDVersion


class ComfyRunMode(Enum):
    runtime = 0  # runs as part of same process, transfer images in memory
    server = 1  # runs as a server, transfer images via base64 or websocket


class Output(NamedTuple):
    node: int
    output: int


Output2 = Tuple[Output, Output]
Output3 = Tuple[Output, Output, Output]
Output4 = Tuple[Output, Output, Output, Output]


class ComfyWorkflow:
    """Builder for workflows which can be sent to the ComfyUI prompt API."""

    root: dict[str, dict]
    images: dict[str, Image]
    node_count = 0
    sample_count = 0

    _cache: dict[str, Output | Output2 | Output3 | Output4]
    _nodes_required_inputs: dict[str, dict[str, Any]]
    _run_mode: ComfyRunMode

    def __init__(self, node_inputs: dict | None = None, run_mode=ComfyRunMode.server):
        self.root = {}
        self.images = {}
        self._cache = {}
        self._nodes_required_inputs = node_inputs or {}
        self._run_mode = run_mode

    def add_default_values(self, node_name: str, args: dict):
        if node_inputs := self._nodes_required_inputs.get(node_name, None):
            for k, v in node_inputs.items():
                if k not in args:
                    if len(v) == 1 and isinstance(v[0], list) and len(v[0]) > 0:
                        # enum type, use first value in list of possible values
                        args[k] = v[0][0]
                    elif len(v) > 1 and isinstance(v[1], dict):
                        # other type, try to access default value
                        default = v[1].get("default", None)
                        if default is not None:
                            args[k] = default
        return args

    def dump(self, filepath: str | Path):
        filepath = Path(filepath)
        if filepath.suffix != ".json":
            filepath = filepath / "workflow.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.root, f, indent=4)

    @overload
    def add(self, class_type: str, output_count: Literal[1], **inputs) -> Output: ...

    @overload
    def add(self, class_type: str, output_count: Literal[2], **inputs) -> Output2: ...

    @overload
    def add(self, class_type: str, output_count: Literal[3], **inputs) -> Output3: ...

    @overload
    def add(self, class_type: str, output_count: Literal[4], **inputs) -> Output4: ...

    def add(self, class_type: str, output_count: int, **inputs):
        inputs = self.add_default_values(class_type, inputs)
        normalize = lambda x: [str(x.node), x.output] if isinstance(x, Output) else x
        self.node_count += 1
        self.root[str(self.node_count)] = {
            "class_type": class_type,
            "inputs": {k: normalize(v) for k, v in inputs.items()},
        }
        output = tuple(Output(self.node_count, i) for i in range(output_count))
        return output[0] if output_count == 1 else output

    @overload
    def add_cached(self, class_type: str, output_count: Literal[1], **inputs) -> Output: ...

    @overload
    def add_cached(self, class_type: str, output_count: Literal[3], **inputs) -> Output3: ...

    def add_cached(self, class_type: str, output_count: Literal[1] | Literal[3], **inputs):
        key = class_type + str(inputs)
        result = self._cache.get(key, None)
        if result is None:
            result = self.add(class_type, output_count, **inputs)
            self._cache[key] = result
        return result

    def _add_image(self, image: Image):
        id = str(uuid4())
        self.images[id] = image
        return id

    def ksampler(
        self,
        model: Output,
        positive: Output,
        negative: Output,
        latent_image: Output,
        sampler="dpmpp_2m_sde_gpu",
        scheduler="normal",
        steps=20,
        cfg=7.0,
        denoise=1.0,
        seed=1234,
    ):
        self.sample_count += steps
        return self.add(
            "KSampler",
            1,
            seed=seed,
            sampler_name=sampler,
            scheduler=scheduler,
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
        )

    def ksampler_advanced(
        self,
        model: Output,
        positive: Output,
        negative: Output,
        latent_image: Output,
        sampler="dpmpp_2m_sde_gpu",
        scheduler="normal",
        steps=20,
        start_at_step=0,
        cfg=7.0,
        seed=-1,
    ):
        self.sample_count += steps - start_at_step

        return self.add(
            "KSamplerAdvanced",
            1,
            noise_seed=seed,
            sampler_name=sampler,
            scheduler=scheduler,
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            steps=steps,
            start_at_step=start_at_step,
            end_at_step=steps,
            cfg=cfg,
            add_noise="enable",
            return_with_leftover_noise="disable",
        )

    def differential_diffusion(self, model: Output):
        return self.add("DifferentialDiffusion", 1, model=model)

    def model_sampling_discrete(self, model: Output, sampling: str, zsnr=False):
        return self.add("ModelSamplingDiscrete", 1, model=model, sampling=sampling, zsnr=zsnr)

    def model_sampling_sd3(self, model: Output, shift=3.0):
        return self.add("ModelSamplingSD3", 1, model=model, shift=shift)

    def rescale_cfg(self, model: Output, multiplier=0.7):
        return self.add("RescaleCFG", 1, model=model, multiplier=multiplier)

    def load_checkpoint(self, checkpoint: str):
        return self.add_cached("CheckpointLoaderSimple", 3, ckpt_name=checkpoint)

    def load_dual_clip(self, clip_name1: str, clip_name2: str, type="sd3"):
        return self.add_cached(
            "DualCLIPLoader", 1, clip_name1=clip_name1, clip_name2=clip_name2, type=type
        )

    def load_vae(self, vae_name: str):
        return self.add_cached("VAELoader", 1, vae_name=vae_name)

    def load_controlnet(self, controlnet: str):
        return self.add_cached("ControlNetLoader", 1, control_net_name=controlnet)

    def load_clip_vision(self, clip_name: str):
        return self.add_cached("CLIPVisionLoader", 1, clip_name=clip_name)

    def load_ip_adapter(self, ipadapter_file: str):
        return self.add_cached("IPAdapterModelLoader", 1, ipadapter_file=ipadapter_file)

    def load_upscale_model(self, model_name: str):
        return self.add_cached("UpscaleModelLoader", 1, model_name=model_name)

    def load_lora_model(self, model: Output, lora_name: str, strength: float):
        return self.add(
            "LoraLoaderModelOnly", 1, model=model, lora_name=lora_name, strength_model=strength
        )

    def load_lora(self, model: Output, clip: Output, lora_name, strength_model, strength_clip):
        return self.add(
            "LoraLoader",
            2,
            model=model,
            clip=clip,
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=strength_clip,
        )

    def load_insight_face(self):
        return self.add_cached("IPAdapterInsightFaceLoader", 1, provider="CPU")

    def load_inpaint_model(self, model_name: str):
        return self.add_cached("INPAINT_LoadInpaintModel", 1, model_name=model_name)

    def load_fooocus_inpaint(self, head: str, patch: str):
        return self.add_cached("INPAINT_LoadFooocusInpaint", 1, head=head, patch=patch)

    def empty_latent_image(self, extent: Extent, version: SDVersion, batch_size=1):
        w, h = extent.width, extent.height
        if version is SDVersion.sd3:
            return self.add("EmptySD3LatentImage", 1, width=w, height=h, batch_size=batch_size)
        return self.add("EmptyLatentImage", 1, width=w, height=h, batch_size=batch_size)

    def clip_set_last_layer(self, clip: Output, clip_layer: int):
        return self.add("CLIPSetLastLayer", 1, clip=clip, stop_at_clip_layer=clip_layer)

    def clip_text_encode(self, clip: Output, text: str):
        return self.add("CLIPTextEncode", 1, clip=clip, text=text)

    def conditioning_area(self, conditioning: Output, area: Bounds, strength=1.0):
        return self.add(
            "ConditioningSetArea",
            1,
            conditioning=conditioning,
            x=area.x,
            y=area.y,
            width=area.width,
            height=area.height,
            strength=strength,
        )

    def conditioning_set_mask(self, conditioning: Output, mask: Output, strength=1.0):
        return self.add(
            "ConditioningSetMask",
            1,
            conditioning=conditioning,
            mask=mask,
            strength=strength,
            set_cond_area="default",
        )

    def conditioning_combine(self, a: Output, b: Output):
        return self.add("ConditioningCombine", 1, conditioning_1=a, conditioning_2=b)

    def background_region(self, conditioning: Output):
        return self.add("ETN_BackgroundRegion", 1, conditioning=conditioning)

    def define_region(self, regions: Output, mask: Output, conditioning: Output):
        return self.add(
            "ETN_DefineRegion", 1, regions=regions, mask=mask, conditioning=conditioning
        )

    def attention_mask(self, model: Output, regions: Output):
        return self.add("ETN_AttentionMask", 1, model=model, regions=regions)

    def apply_controlnet(
        self,
        positive: Output,
        negative: Output,
        controlnet: Output,
        image: Output,
        strength=1.0,
        range: tuple[float, float] = (0.0, 1.0),
    ):
        return self.add(
            "ControlNetApplyAdvanced",
            2,
            positive=positive,
            negative=negative,
            control_net=controlnet,
            image=image,
            strength=strength,
            start_percent=range[0],
            end_percent=range[1],
        )

    def encode_ip_adapter(
        self, image: Output, weight: float, ip_adapter: Output, clip_vision: Output
    ):
        return self.add(
            "IPAdapterEncoder",
            2,
            image=image,
            weight=weight,
            ipadapter=ip_adapter,
            clip_vision=clip_vision,
        )

    def combine_ip_adapter_embeds(self, embeds: list[Output]):
        e = {f"embed{i+1}": embed for i, embed in enumerate(embeds)}
        return self.add("IPAdapterCombineEmbeds", 1, method="concat", **e)

    def apply_ip_adapter(
        self,
        model: Output,
        ip_adapter: Output,
        clip_vision: Output,
        embeds: Output,
        weight: float,
        weight_type: str = "linear",
        range: tuple[float, float] = (0.0, 1.0),
        mask: Output | None = None,
    ):
        return self.add(
            "IPAdapterEmbeds",
            1,
            model=model,
            ipadapter=ip_adapter,
            pos_embed=embeds,
            clip_vision=clip_vision,
            weight=weight,
            weight_type=weight_type,
            embeds_scaling="V only",
            start_at=range[0],
            end_at=range[1],
            attn_mask=mask,
        )

    def apply_ip_adapter_face(
        self,
        model: Output,
        ip_adapter: Output,
        clip_vision: Output,
        insightface: Output,
        image: Output,
        weight=1.0,
        range: tuple[float, float] = (0.0, 1.0),
        mask: Output | None = None,
    ):
        return self.add(
            "IPAdapterFaceID",
            1,
            model=model,
            ipadapter=ip_adapter,
            image=image,
            clip_vision=clip_vision,
            insightface=insightface,
            weight=weight,
            weight_faceidv2=weight * 2,
            weight_type="linear",
            start_at=range[0],
            end_at=range[1],
            attn_mask=mask,
        )

    def apply_self_attention_guidance(self, model: Output):
        return self.add("SelfAttentionGuidance", 1, model=model, scale=0.5, blur_sigma=2.0)

    def inpaint_preprocessor(self, image: Output, mask: Output):
        return self.add("InpaintPreprocessor", 1, image=image, mask=mask)

    def apply_fooocus_inpaint(self, model: Output, patch: Output, latent: Output):
        return self.add("INPAINT_ApplyFooocusInpaint", 1, model=model, patch=patch, latent=latent)

    def vae_encode_inpaint_conditioning(
        self, vae: Output, image: Output, mask: Output, positive: Output, negative: Output
    ):
        return self.add(
            "INPAINT_VAEEncodeInpaintConditioning",
            4,
            vae=vae,
            pixels=image,
            mask=mask,
            positive=positive,
            negative=negative,
        )

    def vae_encode(self, vae: Output, image: Output):
        return self.add("VAEEncode", 1, vae=vae, pixels=image)

    def vae_encode_inpaint(self, vae: Output, image: Output, mask: Output):
        return self.add("VAEEncodeForInpaint", 1, vae=vae, pixels=image, mask=mask, grow_mask_by=0)

    def vae_decode(self, vae: Output, latent_image: Output):
        return self.add("VAEDecode", 1, vae=vae, samples=latent_image)

    def set_latent_noise_mask(self, latent: Output, mask: Output):
        return self.add("SetLatentNoiseMask", 1, samples=latent, mask=mask)

    def batch_latent(self, latent: Output, batch_size: int):
        if batch_size == 1:
            return latent
        return self.add("RepeatLatentBatch", 1, samples=latent, amount=batch_size)

    def crop_latent(self, latent: Output, bounds: Bounds):
        return self.add(
            "LatentCrop",
            1,
            samples=latent,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def empty_image(self, extent: Extent, color=0):
        return self.add(
            "EmptyImage", 1, width=extent.width, height=extent.height, color=color, batch_size=1
        )

    def crop_image(self, image: Output, bounds: Bounds):
        return self.add(
            "ImageCrop",
            1,
            image=image,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def scale_image(self, image: Output, extent: Extent, method="lanczos"):
        return self.add(
            "ImageScale",
            1,
            image=image,
            width=extent.width,
            height=extent.height,
            upscale_method=method,
            crop="disabled",
        )

    def scale_control_image(self, image: Output, extent: Extent):
        return self.add(
            "HintImageEnchance",
            1,
            hint_image=image,
            image_gen_width=extent.width,
            image_gen_height=extent.height,
            resize_mode="Just Resize",
        )

    def upscale_image(self, upscale_model: Output, image: Output):
        self.sample_count += 4  # approx, actual number depends on model and image size
        return self.add("ImageUpscaleWithModel", 1, upscale_model=upscale_model, image=image)

    def invert_image(self, image: Output):
        return self.add("ImageInvert", 1, image=image)

    def batch_image(self, batch: Output, image: Output):
        return self.add("ImageBatch", 1, image1=batch, image2=image)

    def inpaint_image(self, model: Output, image: Output, mask: Output):
        return self.add(
            "INPAINT_InpaintWithModel", 1, inpaint_model=model, image=image, mask=mask, seed=834729
        )

    def crop_mask(self, mask: Output, bounds: Bounds):
        return self.add(
            "CropMask",
            1,
            mask=mask,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def scale_mask(self, mask: Output, extent: Extent):
        img = self.mask_to_image(mask)
        scaled = self.scale_image(img, extent, method="bilinear")
        return self.image_to_mask(scaled)

    def image_to_mask(self, image: Output):
        return self.add("ImageToMask", 1, image=image, channel="red")

    def composite_image_masked(
        self, source: Output, destination: Output, mask: Output | None, x=0, y=0
    ):
        return self.add(
            "ImageCompositeMasked",
            1,
            source=source,
            destination=destination,
            mask=mask,
            x=x,
            y=y,
            resize_source=False,
        )

    def mask_to_image(self, mask: Output):
        return self.add("MaskToImage", 1, mask=mask)

    def solid_mask(self, extent: Extent, value=1.0):
        return self.add("SolidMask", 1, width=extent.width, height=extent.height, value=value)

    def fill_masked(self, image: Output, mask: Output, mode="neutral", falloff: int = 0):
        return self.add("INPAINT_MaskedFill", 1, image=image, mask=mask, fill=mode, falloff=falloff)

    def blur_masked(self, image: Output, mask: Output, blur: int, falloff: int = 0):
        return self.add("INPAINT_MaskedBlur", 1, image=image, mask=mask, blur=blur, falloff=falloff)

    def denoise_to_compositing_mask(self, mask: Output, offset=0.05, threshold=0.35):
        return self.add(
            "INPAINT_DenoiseToCompositingMask", 1, mask=mask, offset=offset, threshold=threshold
        )

    def apply_mask(self, image: Output, mask: Output):
        return self.add("ETN_ApplyMaskToImage", 1, image=image, mask=mask)

    def load_image(self, image: Image):
        if self._run_mode is ComfyRunMode.runtime:
            return self.add("ETN_InjectImage", 1, id=self._add_image(image))
        return self.add("ETN_LoadImageBase64", 1, image=image.to_base64())

    def load_mask(self, mask: Image):
        if self._run_mode is ComfyRunMode.runtime:
            return self.add("ETN_InjectMask", 1, id=self._add_image(mask))
        return self.add("ETN_LoadMaskBase64", 1, mask=mask.to_base64())

    def send_image(self, image: Output):
        if self._run_mode is ComfyRunMode.runtime:
            return self.add("ETN_ReturnImage", 1, images=image)
        return self.add("ETN_SendImageWebSocket", 1, images=image)

    def save_image(self, image: Output, prefix: str):
        return self.add("SaveImage", 1, images=image, filename_prefix=prefix)

    def create_tile_layout(self, image: Output, tile_size: int, padding: int, blending: int):
        return self.add(
            "ETN_TileLayout",
            1,
            image=image,
            min_tile_size=tile_size,
            padding=padding,
            blending=blending,
        )

    def extract_image_tile(self, image: Output, layout: Output, index: int):
        return self.add("ETN_ExtractImageTile", 1, image=image, layout=layout, index=index)

    def extract_mask_tile(self, mask: Output, layout: Output, index: int):
        return self.add("ETN_ExtractMaskTile", 1, mask=mask, layout=layout, index=index)

    def merge_image_tile(self, image: Output, layout: Output, index: int, tile: Output):
        return self.add("ETN_MergeImageTile", 1, layout=layout, index=index, tile=tile, image=image)

    def generate_tile_mask(self, layout: Output, index: int):
        return self.add("ETN_GenerateTileMask", 1, layout=layout, index=index)

    def estimate_pose(self, image: Output, resolution: int):
        feat = dict(detect_hand="enable", detect_body="enable", detect_face="enable")
        mdls = dict(bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384.onnx")
        if self._run_mode is ComfyRunMode.runtime:
            # use smaller model, but it requires onnxruntime, see #630
            mdls["bbox_detector"] = "yolo_nas_l_fp16.onnx"
        return self.add("DWPreprocessor", 1, image=image, resolution=resolution, **feat, **mdls)
