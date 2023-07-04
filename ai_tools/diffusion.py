from typing import Union, Sequence
from .image import Extent, Image, ImageCollection
from .network import RequestManager, Interrupted, Progress
from .settings import settings


def _collect_images(result, count: int = ...):
    if "images" in result:
        images = result["images"]
        assert isinstance(images, list)
        if count is not ...:
            images = images[:count]
        return ImageCollection(map(Image.from_base64, images))
    raise Interrupted()


def _make_tiled_vae_payload():
    return {"tiled vae": {"args": [True, 1536]}}  # TODO hardcoded tile size


def _find_controlnet_model(model_list: Sequence[str], model_name: str):
    model = next((model for model in model_list if model.startswith(model_name)), None)
    if model is None:
        raise Exception(
            f"Could not find ControlNet model {model_name}. Make sure to download the model and"
            " place it in the ControlNet models folder."
        )
    return model


def _find_controlnet_processor(processor_list: Sequence[str], processor_name: str):
    if not processor_name in processor_list:
        raise Exception(
            f"Could not find ControlNet processor {processor_name}. Maybe the ControlNet extension"
            " version is too old?"
        )


class Auto1111:
    default_url = "http://127.0.0.1:7860"
    default_upscaler = "Lanczos"
    default_sampler = "DPM++ 2M Karras"

    _requests = RequestManager()
    _controlnet_inpaint_model: str
    _controlnet_tile_model: str

    url: str

    @staticmethod
    async def connect(url=default_url):
        result = Auto1111(url)
        upscalers = await result._get("sdapi/v1/upscalers")
        settings.upscalers = [u["name"] for u in upscalers if not u["name"] == "None"]
        controlnet_models = (await result._get("controlnet/model_list"))["model_list"]
        result._controlnet_inpaint_model = _find_controlnet_model(
            controlnet_models, "control_v11p_sd15_inpaint"
        )
        result._controlnet_tile_model = _find_controlnet_model(
            controlnet_models, "control_v11f1e_sd15_tile"
        )
        controlnet_modules = (await result._get("controlnet/module_list"))["module_list"]
        _find_controlnet_processor(controlnet_modules, "inpaint_only+lama")
        _find_controlnet_processor(controlnet_modules, "tile_resample")
        return result

    def __init__(self, url):
        self.url = url

    async def _get(self, op: str):
        return await self._requests.get(f"{self.url}/{op}")

    async def _post(self, op: str, data: dict, progress: Progress = ...):
        request = self._requests.post(f"{self.url}/{op}", data)
        if progress is not ...:
            while not request.done():
                status = await self._get("sdapi/v1/progress")
                if status["progress"] >= 1:
                    break
                elif status["progress"] > 0:
                    progress(status["progress"])
            progress.finish()
        return await request

    async def txt2img(self, prompt: str, extent: Extent, progress: Progress):
        payload = {
            "prompt": prompt,
            "negative_prompt": settings.negative_prompt,
            "batch_size": settings.batch_size,
            "steps": 30,
            "cfg_scale": 7,
            "width": extent.width,
            "height": extent.height,
            "sampler_index": self.default_sampler,
        }
        result = await self._post("sdapi/v1/txt2img", payload, progress)
        return _collect_images(result)

    async def txt2img_inpaint(
        self, img: Image, mask: Image, prompt: str, extent: Extent, progress: Progress
    ):
        assert img.extent == mask.extent
        cn_payload = {
            "controlnet": {
                "args": [
                    {
                        "input_image": img.to_base64(),
                        "mask": mask.to_base64(),
                        "module": "inpaint_only+lama",
                        "model": self._controlnet_inpaint_model,
                        "control_mode": "ControlNet is more important",
                        "pixel_perfect": True,
                    }
                ]
            }
        }
        payload = {
            "prompt": prompt,
            "negative_prompt": settings.negative_prompt,
            "batch_size": settings.batch_size,
            "steps": 20,
            "cfg_scale": 5,
            "width": extent.width,
            "height": extent.height,
            "alwayson_scripts": cn_payload,
            "sampler_index": "DDIM",
        }
        result = await self._post("sdapi/v1/txt2img", payload, progress)
        return _collect_images(result, count=-1)

    async def _img2img(
        self,
        image: Union[Image, str],
        prompt: str,
        strength: float,
        extent: Extent,
        cfg_scale: int,
        batch_size: int,
        progress: Progress,
    ):
        image = image.to_base64() if isinstance(image, Image) else image
        payload = {
            "init_images": [image],
            "denoising_strength": strength,
            "prompt": prompt,
            "negative_prompt": settings.negative_prompt,
            "batch_size": batch_size,
            "steps": 30,
            "cfg_scale": cfg_scale,
            "width": extent.width,
            "height": extent.height,
            "alwayson_scripts": _make_tiled_vae_payload(),
            "sampler_index": self.default_sampler,
        }
        result = await self._post("sdapi/v1/img2img", payload, progress)
        return _collect_images(result)

    async def img2img(
        self, img: Image, prompt: str, strength: float, extent: Extent, progress: Progress
    ):
        return await self._img2img(img, prompt, strength, extent, 7, settings.batch_size, progress)

    async def img2img_inpaint(
        self,
        img: Image,
        mask: Image,
        prompt: str,
        strength: float,
        extent: Extent,
        progress: Progress,
    ):
        assert img.extent == mask.extent
        cn_payload = {
            "controlnet": {
                "args": [
                    {
                        "module": "inpaint_only",
                        "model": self._controlnet_inpaint_model,
                        "control_mode": "Balanced",
                        "pixel_perfect": True,
                    }
                ]
            }
        }
        payload = {
            "init_images": [img.to_base64()],
            "denoising_strength": strength,
            "mask": mask.to_base64(),
            "mask_blur": 0,
            "inpainting_fill": 1,
            "inpainting_full_res": True,
            "prompt": prompt,
            "negative_prompt": settings.negative_prompt,
            "batch_size": settings.batch_size,
            "steps": 30,
            "cfg_scale": 7,
            "width": extent.width,
            "height": extent.height,
            "alwayson_scripts": cn_payload,
            "sampler_index": self.default_sampler,
        }
        result = await self._post("sdapi/v1/img2img", payload, progress)
        return _collect_images(result, count=-1)

    async def upscale(self, img: Image, target: Extent, prompt: str, progress: Progress):
        upscale_payload = {
            "resize_mode": 1,  # width & height
            "upscaling_resize_w": target.width,
            "upscaling_resize_h": target.height,
            "upscaler_1": settings.upscaler,
            "image": img.to_base64(),
        }
        result = await self._post("sdapi/v1/extra-single-image", upscale_payload)
        result = await self._img2img(
            image=result["image"],
            prompt=f"{settings.upscale_prompt}, {prompt}",
            strength=0.4,
            extent=target,
            cfg_scale=5,
            batch_size=1,
            progress=progress,
        )
        return result

    async def upscale_tiled(self, img: Image, target: Extent, prompt: str, progress: Progress):
        # TODO dead code, consider multi diffusion
        cn_payload = {
            "controlnet": {
                "args": [
                    {
                        "input_image": img.to_base64(),
                        "module": "tile_resample",
                        "model": self._controlnet_tile_model,
                        "control_mode": "Balanced",
                    }
                ]
            }
        }
        upscale_args = [
            None,  # _
            768,  # tile_width
            768,  #  tile_height
            8,  # mask_blur
            32,  # padding
            0,  # seams_fix_width
            0,  # seams_fix_denoise
            0,  # seams_fix_padding
            settings.upscaler_index,
            False,  # save_upscaled_image
            0,  # redraw mode = LINEAR
            False,  # save_seams_fix_image
            0,  # seams_fix_mask_blur
            0,  # seams_fix_type = NONE
            0,  # size type
            0,  # width
            0,  # height
            0,  # scale = FROM_IMG2IMG
        ]
        payload = {
            "init_images": [img.to_base64()],
            "resize_mode": 0,
            "denoising_strength": 0.4,
            "prompt": f"{settings.upscale_prompt}, {prompt}",
            "negative_prompt": settings.negative_prompt,
            "sampler_index": "DPM++ 2M Karras",
            "steps": 30,
            "cfg_scale": 5,
            "width": target.width,
            "height": target.height,
            "script_name": "ultimate sd upscale",
            "script_args": upscale_args,
            "alwayson_scripts": cn_payload,
        }
        result = await self._post("sdapi/v1/img2img", payload, progress)
        return _collect_images(result)

    async def interrupt(self):
        return await self._post("sdapi/v1/interrupt", {})
