from __future__ import annotations
import math
import random
from typing import NamedTuple, Tuple, Literal, overload

from .image import Bounds, Extent, Image


class Output(NamedTuple):
    node: int
    output: int


Output2 = Tuple[Output, Output]
Output3 = Tuple[Output, Output, Output]


class ComfyWorkflow:
    """Builder for workflows which can be sent to the ComfyUI prompt API."""

    node_count = 0
    sample_count = 0

    _cache: dict[str, Output | Output2 | Output3]

    def __init__(self) -> None:
        self.root = {}
        self._cache = {}

    def dump(self, filepath: str):
        with open(filepath, "w") as f:
            for key, value in self.root.items():
                f.write(f"{key} = {value}\n")

    @overload
    def add(self, class_type: str, output_count: Literal[1], **inputs) -> Output:
        ...

    @overload
    def add(self, class_type: str, output_count: Literal[2], **inputs) -> Output2:
        ...

    @overload
    def add(self, class_type: str, output_count: Literal[3], **inputs) -> Output3:
        ...

    def add(self, class_type: str, output_count: int, **inputs):
        normalize = lambda x: [str(x.node), x.output] if isinstance(x, Output) else x
        self.node_count += 1
        self.root[str(self.node_count)] = {
            "class_type": class_type,
            "inputs": {k: normalize(v) for k, v in inputs.items()},
        }
        output = tuple(Output(self.node_count, i) for i in range(output_count))
        return output[0] if output_count == 1 else output

    @overload
    def add_cached(self, class_type: str, output_count: Literal[1], **inputs) -> Output:
        ...

    @overload
    def add_cached(self, class_type: str, output_count: Literal[3], **inputs) -> Output3:
        ...

    def add_cached(self, class_type: str, output_count: Literal[1] | Literal[3], **inputs):
        key = class_type + str(inputs)
        result = self._cache.get(key, None)
        if result is None:
            result = self.add(class_type, output_count, **inputs)
            self._cache[key] = result
        return result

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
        seed=-1,
    ):
        self.sample_count += steps
        return self.add(
            "KSampler",
            1,
            seed=random.getrandbits(64) if seed == -1 else seed,
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

    def load_checkpoint(self, checkpoint: str):
        return self.add_cached("CheckpointLoaderSimple", 3, ckpt_name=checkpoint)

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

    def empty_latent_image(self, width: int, height: int, batch_size=1):
        return self.add("EmptyLatentImage", 1, width=width, height=height, batch_size=batch_size)

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

    def conditioning_combine(self, a: Output, b: Output):
        return self.add("ConditioningCombine", 1, conditioning_1=a, conditioning_2=b)

    def apply_controlnet(
        self, conditioning: Output, controlnet: Output, image: Output, strength=1.0
    ):
        return self.add(
            "ControlNetApply",
            1,
            conditioning=conditioning,
            control_net=controlnet,
            image=image,
            strength=strength,
        )

    def apply_ip_adapter(
        self,
        ipadapter: Output,
        clip_vision: Output,
        image: Output,
        model: Output,
        weight: float,
        noise=0.0,
    ):
        return self.add(
            "IPAdapterApply",
            1,
            ipadapter=ipadapter,
            clip_vision=clip_vision,
            image=image,
            model=model,
            weight=weight,
            noise=noise,
        )

    def inpaint_preprocessor(self, image: Output, mask: Output):
        return self.add("InpaintPreprocessor", 1, image=image, mask=mask)

    def vae_encode(self, vae: Output, image: Output):
        return self.add("VAEEncode", 1, vae=vae, pixels=image)

    def vae_encode_inpaint(self, vae: Output, image: Output, mask: Output):
        return self.add("VAEEncodeForInpaint", 1, vae=vae, pixels=image, mask=mask, grow_mask_by=0)

    def vae_decode(self, vae: Output, latent_image: Output):
        return self.add("VAEDecode", 1, vae=vae, samples=latent_image)

    def set_latent_noise_mask(self, latent: Output, mask: Output):
        return self.add("SetLatentNoiseMask", 1, samples=latent, mask=mask)

    def batch_latent(self, latent: Output, batch_size: int):
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

    def scale_latent(self, latent: Output, extent: Extent):
        return self.add(
            "LatentUpscale",
            1,
            samples=latent,
            width=extent.width,
            height=extent.height,
            upscale_method="nearest-exact",
            crop="disabled",
        )

    def crop_image(self, image: Output, bounds: Bounds):
        return self.add(
            "ETN_CropImage",
            1,
            image=image,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def scale_image(self, image: Output, extent: Extent):
        return self.add(
            "ImageScale",
            1,
            image=image,
            width=extent.width,
            height=extent.height,
            upscale_method="bilinear",
            crop="disabled",
        )

    def upscale_image(self, upscale_model: Output, image: Output):
        return self.add("ImageUpscaleWithModel", 1, upscale_model=upscale_model, image=image)

    def invert_image(self, image: Output):
        return self.add("ImageInvert", 1, image=image)

    def batch_image(self, batch: Output, image: Output):
        return self.add("ImageBatch", 1, image1=batch, image2=image)

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
        scaled = self.scale_image(img, extent)
        return self.image_to_mask(scaled)

    def image_to_mask(self, image: Output):
        return self.add("ImageToMask", 1, image=image, channel="red")

    def mask_to_image(self, mask: Output):
        return self.add("MaskToImage", 1, mask=mask)

    def solid_mask(self, extent: Extent, value=1.0):
        return self.add("SolidMask", 1, width=extent.width, height=extent.height, value=value)

    def apply_mask(self, image: Output, mask: Output):
        return self.add("ETN_ApplyMaskToImage", 1, image=image, mask=mask)

    def load_image(self, image: Image):
        return self.add("ETN_LoadImageBase64", 1, image=image.to_base64())

    def load_mask(self, mask: Image):
        return self.add("ETN_LoadMaskBase64", 1, mask=mask.to_base64())

    def send_image(self, image: Output):
        return self.add("ETN_SendImageWebSocket", 1, images=image)

    def save_image(self, image: Output, prefix: str):
        return self.add("SaveImage", 1, images=image, filename_prefix=prefix)

    def upscale_tiled(
        self,
        image: Output,
        model: Output,
        vae: Output,
        positive: Output,
        negative: Output,
        upscale_model: Output,
        original_extent: Extent,
        factor: float,
        tile_extent: Extent,
        steps: int,
        cfg: float,
        sampler: str,
        scheduler: str,
        denoise: float,
        seed=-1,
    ):
        target_extent = original_extent * factor
        tiles_w = int(math.ceil(target_extent.width / tile_extent.width))
        tiles_h = int(math.ceil(target_extent.height / tile_extent.height))
        self.sample_count += 4 + tiles_w * tiles_h * steps  # approx, ignores padding
        return self.add(
            "UltimateSDUpscale",
            1,
            image=image,
            model=model,
            positive=positive,
            negative=negative,
            vae=vae,
            upscale_model=upscale_model,
            upscale_by=factor,
            seed=random.getrandbits(64) if seed == -1 else seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
            denoise=denoise,
            tile_width=tile_extent.width,
            tile_height=tile_extent.height,
            mode_type="Linear",
            mask_blur=8,
            tile_padding=32,
            seam_fix_mode="None",
            seam_fix_denoise=1.0,
            seam_fix_width=64,
            seam_fix_mask_blur=8,
            seam_fix_padding=16,
            force_uniform_tiles="enable",
        )
