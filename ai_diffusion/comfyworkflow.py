import random
from typing import NamedTuple

from .util import compute_batch_size
from .image import Bounds, Extent, Image
from .settings import settings


class Output(NamedTuple):
    node: int
    output: int


class ComfyWorkflow:
    """Builder for workflows which can be sent to the ComfyUI prompt API."""

    node_count = 0
    sample_count = 0

    def __init__(self) -> None:
        self.root = {}

    def add(self, class_type: str, output_count: int, **inputs):
        normalize = lambda x: [str(x.node), x.output] if isinstance(x, Output) else x
        self.node_count += 1
        self.root[str(self.node_count)] = {
            "class_type": class_type,
            "inputs": {k: normalize(v) for k, v in inputs.items()},
        }
        output = tuple(Output(self.node_count, i) for i in range(output_count))
        return output[0] if output_count == 1 else output

    def ksampler(
        self,
        model,
        positive,
        negative,
        latent_image,
        sampler="dpmpp_2m_sde_gpu",
        scheduler="normal",
        steps=20,
        cfg=7,
        denoise=1,
    ):
        self.sample_count += steps
        return self.add(
            "KSampler",
            1,
            seed=random.getrandbits(64),
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

    def load_checkpoint(self, checkpoint):
        return self.add("CheckpointLoaderSimple", 3, ckpt_name=checkpoint)

    def load_controlnet(self, controlnet):
        return self.add("ControlNetLoader", 1, control_net_name=controlnet)

    def load_clip_vision(self, clip_name):
        return self.add("CLIPVisionLoader", 1, clip_name=clip_name)

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        return self.add(
            "LoraLoader",
            2,
            model=model,
            clip=clip,
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=strength_clip,
        )

    def empty_latent_image(self, width, height, batch_size=1):
        return self.add("EmptyLatentImage", 1, width=width, height=height, batch_size=batch_size)

    def clip_text_encode(self, clip, text):
        return self.add("CLIPTextEncode", 1, clip=clip, text=text)

    def apply_controlnet(self, conditioning, controlnet, image):
        return self.add(
            "ControlNetApply",
            1,
            conditioning=conditioning,
            control_net=controlnet,
            image=image,
            strength=1.0,
        )

    def ip_adapter(self, model_name, model, clip_vision, image, mask, weight, dtype="fp16"):
        return self.add(
            "IPAdapter",
            1,
            model_name=model_name,
            model=model,
            clip_vision=clip_vision,
            image=image,
            mask=mask,
            weight=weight,
            dtype=dtype,
        )

    def inpaint_preprocessor(self, image, mask):
        return self.add("InpaintPreprocessor", 1, image=image, mask=mask)

    def vae_encode(self, vae, image):
        return self.add("VAEEncode", 1, vae=vae, pixels=image)

    def vae_encode_inpaint(self, vae, image, mask):
        return self.add("VAEEncodeForInpaint", 1, vae=vae, pixels=image, mask=mask, grow_mask_by=0)

    def vae_decode(self, vae, latent_image):
        return self.add("VAEDecode", 1, vae=vae, samples=latent_image)

    def set_latent_noise_mask(self, latent, mask):
        return self.add("SetLatentNoiseMask", 1, samples=latent, mask=mask)

    def batch_latent(self, latent, batch_size):
        return self.add("RepeatLatentBatch", 1, samples=latent, amount=batch_size)

    def crop_latent(self, latent, bounds: Bounds):
        return self.add(
            "LatentCrop",
            1,
            samples=latent,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def scale_latent(self, latent, extent):
        return self.add(
            "LatentUpscale",
            1,
            samples=latent,
            width=extent.width,
            height=extent.height,
            upscale_method="nearest-exact",
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

    def crop_mask(self, mask, bounds: Bounds):
        return self.add(
            "CropMask",
            1,
            mask=mask,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def scale_mask(self, mask, extent):
        img = self.mask_to_image(mask)
        scaled = self.scale_image(img, extent)
        return self.image_to_mask(scaled)

    def image_to_mask(self, image):
        return self.add("ImageToMask", 1, image=image, channel="red")

    def mask_to_image(self, mask):
        return self.add("MaskToImage", 1, mask=mask)

    def solid_mask(self, extent: Extent, value=1):
        return self.add("SolidMask", 1, width=extent.width, height=extent.height, value=value)

    def apply_mask(self, image, mask):
        return self.add("ETN_ApplyMaskToImage", 1, image=image, mask=mask)

    def load_image(self, image: Image):
        return self.add("ETN_LoadImageBase64", 1, image=image.to_base64())

    def load_mask(self, mask: Image):
        return self.add("ETN_LoadMaskBase64", 1, mask=mask.to_base64())

    def send_image(self, image):
        return self.add("ETN_SendImageWebSocket", 1, images=image)
