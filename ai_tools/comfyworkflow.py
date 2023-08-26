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

    _i = 0

    def __init__(self) -> None:
        self.root = {}

    def add(self, class_type: str, output_count: int, **inputs):
        normalize = lambda x: [str(x.node), x.output] if isinstance(x, Output) else x
        self._i += 1
        self.root[str(self._i)] = {
            "class_type": class_type,
            "inputs": {k: normalize(v) for k, v in inputs.items()},
        }
        return (
            Output(self._i, 0)
            if output_count == 1
            else tuple(Output(self._i, i) for i in range(output_count))
        )

    def ksampler(self, model, positive, negative, latent_image, steps=20, cfg=7, denoise=1):
        return self.add(
            "KSampler",
            1,
            seed=random.getrandbits(64),
            sampler_name="dpmpp_2m_sde_gpu",
            scheduler="karras",
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

    def empty_latent_image(self, width, height):
        batch = compute_batch_size(
            Extent(width, height), settings.min_image_size, settings.batch_size
        )
        return self.add("EmptyLatentImage", 1, width=width, height=height, batch_size=batch)

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

    def inpaint_preprocessor(self, image, mask):
        return self.add("InpaintPreprocessor", 1, image=image, mask=mask)

    def vae_encode(self, vae, image):
        return self.add("VAEEncode", 1, vae=vae, pixels=image)

    def vae_decode(self, vae, latent_image):
        return self.add("VAEDecode", 1, vae=vae, samples=latent_image)

    def set_latent_noise_mask(self, latent, mask):
        return self.add("SetLatentNoiseMask", 1, samples=latent, mask=mask)

    def latent_upscale(self, latent, extent):
        return self.add(
            "LatentUpscale",
            1,
            samples=latent,
            width=extent.width,
            height=extent.height,
            upscale_method="nearest-exact",
            crop="disabled",
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

    def image_to_mask(self, image):
        return self.add("ImageToMask", 1, image=image, channel="red")

    def mask_to_image(self, mask):
        return self.add("MaskToImage", 1, mask=mask)

    def scale_mask(self, mask, extent):
        img = self.mask_to_image(mask)
        scaled = self.scale_image(img, extent)
        return self.image_to_mask(scaled)

    def apply_mask(self, image, mask):
        return self.add("ETN_ApplyMaskToImage", 1, image=image, mask=mask)

    def load_image(self, image: Image):
        return self.add("ETN_LoadImageBase64", 1, image=image.to_base64())

    def load_mask(self, mask: Image):
        return self.add("ETN_LoadMaskBase64", 1, mask=mask.to_base64())

    def send_image(self, image):
        return self.add("ETN_SendImageWebSocket", 1, images=image)
