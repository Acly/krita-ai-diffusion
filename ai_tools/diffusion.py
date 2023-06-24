import asyncio
import json
from typing import Callable, NamedTuple
from .image import Extent, Image, ImageCollection
from . import settings

from PyQt5.QtCore import QByteArray, QUrl
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply


class NetworkError(Exception):
    def __init__(self, code, msg):
        self.code = code
        super().__init__(self, msg)


class Interrupted(Exception):
    def __init__(self):
        super().__init__(self, "Operation cancelled")


class Request(NamedTuple):
    url: str
    future: asyncio.Future


class RequestManager:
    def __init__(self):
        self._net = QNetworkAccessManager()
        self._net.finished.connect(self._finished)
        self._requests = {}

    def request(self, method, url: str, data: dict = None):
        self._cleanup()

        request = QNetworkRequest(QUrl(url))
        # request.setTransferTimeout({"GET": 30000, "POST": 0}[method]) # requires Qt 5.15 (Krita 5.2)
        if data is not None:
            data_bytes = QByteArray(json.dumps(data).encode("utf-8"))
            request.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
            request.setHeader(QNetworkRequest.ContentLengthHeader, data_bytes.size())

        assert method in ["GET", "POST"]
        if method == "POST":
            reply = self._net.post(request, data_bytes)
        else:
            reply = self._net.get(request)

        future = asyncio.get_running_loop().create_future()
        self._requests[reply] = Request(url, future)
        return future

    def get(self, url: str):
        return self.request("GET", url)

    def post(self, url: str, data: dict):
        return self.request("POST", url, data)

    def _finished(self, reply):
        code = reply.error()
        url, future = self._requests[reply]
        if future.cancelled():
            return  # operation was cancelled, discard result
        if code == QNetworkReply.NoError:
            future.set_result(json.loads(reply.readAll().data()))
        else:
            try:  # extract detailed information from the payload
                data = json.loads(reply.readAll().data())
                err = f'{reply.errorString()} ({data["detail"]})'
            except:
                err = f"{reply.errorString()} ({url})"
            future.set_exception(NetworkError(code, f"Server request failed: {err}"))

    def _cleanup(self):
        self._requests = {
            reply: request for reply, request in self._requests.items() if not reply.isFinished()
        }


class Progress:
    callback: Callable[[float], None]
    scale: float = 1
    offset: float = 0

    def __init__(self, callback: Callable[[float], None], scale: float = 1):
        self.callback = callback
        self.scale = scale

    @staticmethod
    def forward(other, scale: float = 1):
        return Progress(other.callback, scale)

    def __call__(self, progress: float):
        self.callback(self.offset + self.scale * progress)

    def finish(self):
        self.offset = self.offset + self.scale
        self.callback(self.offset)


def _collect_images(result, count: int = ...):
    if "images" in result:
        images = result["images"]
        assert isinstance(images, list)
        if count is not ...:
            images = images[:count]
        return ImageCollection(map(Image.from_base64, images))
    raise Interrupted()


class Auto1111:
    default_url = "http://127.0.0.1:7860"
    default_upscaler = "Lanczos"

    _requests = RequestManager()

    url: str
    negative_prompt = "EasyNegative verybadimagenegative_v1.3"
    upscale_prompt = "highres 8k uhd"
    upscalers = []
    upscaler_index = 0

    @staticmethod
    async def connect(url=default_url):
        result = Auto1111(url)
        upscalers = await result._get("sdapi/v1/upscalers")
        result.upscalers = [u["name"] for u in upscalers]
        result.upscaler_index = result.upscalers.index(Auto1111.default_upscaler)
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
                        "module": "inpaint_only",
                        "model": "control_v11p_sd15_inpaint [ebff9138]",
                        "control_mode": "ControlNet is more important",
                        "pixel_perfect": True,
                    }
                ]
            }
        }
        payload = {
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
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
                        "model": "control_v11p_sd15_inpaint [ebff9138]",
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
            "negative_prompt": self.negative_prompt,
            "batch_size": settings.batch_size,
            "steps": 30,
            "cfg_scale": 7,
            "width": extent.width,
            "height": extent.height,
            "alwayson_scripts": cn_payload,
            "sampler_index": "DPM++ 2M Karras",
        }
        result = await self._post("sdapi/v1/img2img", payload, progress)
        return _collect_images(result, count=-1)

    async def upscale(self, img: Image, target: Extent, prompt: str, progress: Progress):
        cn_payload = {
            "controlnet": {
                "args": [
                    {
                        "input_image": img.to_base64(),
                        "module": "tile_resample",
                        "model": "control_v11f1e_sd15_tile [a371b31b]",
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
            self.upscaler_index,
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
            "prompt": f"{self.upscale_prompt} {prompt}",
            "negative_prompt": self.negative_prompt,
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
