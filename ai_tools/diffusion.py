import asyncio
import json
from typing import Callable, NamedTuple
from .image import Extent, Image

from PyQt5.QtCore import QByteArray, QUrl
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

automatic1111_url = 'http://127.0.0.1:7860/sdapi/v1'
default_upscale_prompt = 'highres 8k uhd'

class NetworkError(Exception):
    def __init__(self, code, msg):
        self.code = code
        super().__init__(self, msg)

class Request(NamedTuple):
    url: str
    future: asyncio.Future

class RequestManager:
    def __init__(self):
        self._net = QNetworkAccessManager()
        self._net.finished.connect(self._finished)
        self._requests = {}

    def request(self, method, url: str, data: dict=None):
        self._cleanup()

        request = QNetworkRequest(QUrl(url))
        if data:
            data_bytes = QByteArray(json.dumps(data).encode("utf-8"))
            request.setHeader(QNetworkRequest.ContentTypeHeader, 'application/json')
            request.setHeader(QNetworkRequest.ContentLengthHeader, data_bytes.size())

        assert method in ['GET', 'POST']
        if method == 'POST':
            reply = self._net.post(request, data_bytes)
        else:
            reply = self._net.get(request)

        future = asyncio.get_running_loop().create_future()
        self._requests[reply] = Request(url, future)
        return future
    
    def get(self, url: str):
        return self.request('GET', url)

    def post(self, url: str, data: dict):
        return self.request('POST', url, data)    
    
    def _finished(self, reply):
        code = reply.error()
        url, future = self._requests[reply]
        if code == QNetworkReply.NoError:
            future.set_result(json.loads(reply.readAll().data()))
        else:
            future.set_exception(NetworkError(code, f'Server request failed: {reply.errorString()} [url={url}]'))

    def _cleanup(self):
        self._requests = {reply: request for reply, request in self._requests.items() if not reply.isFinished()}

requests = RequestManager()


class Progress:
    callback: Callable[[float], None]
    scale: float = 1
    offset: float = 0

    def __init__(self, callback: Callable[[float], None], scale: float=1):
        self.callback = callback
        self.scale = scale

    @staticmethod
    def forward(other, scale: float=1):
        return Progress(other.callback, scale)

    def __call__(self, progress: float):
        self.callback(self.offset + self.scale * progress)

    def finish(self):
        self.offset = self.offset + self.scale
        self.callback(self.offset)

async def auto1111(op: str, data: dict, progress: Progress=...):
    request = requests.post(f'{automatic1111_url}/{op}', data)
    if progress is not ...:
        while not request.done():
            status = await requests.get(f'{automatic1111_url}/progress')
            if status['progress'] >= 1:
                break
            elif status['progress'] > 0:
                progress(status['progress'])
        progress.finish()
    return await request


async def inpaint(img: Image, mask: Image, prompt: str, progress: Progress):
    assert img.extent == mask.extent
    cn_payload = {
        'controlnet': {
            'args': [{
                'input_image': img.to_base64(),
                'mask': mask.to_base64(),
                'module': 'inpaint_only',
                'model': 'control_v11p_sd15_inpaint [ebff9138]',
                'control_mode': 'ControlNet is more important',
                'pixel_perfect': True
    }]}}
    payload = {
        'prompt': prompt,
        'steps': 20,
        'cfg_scale': 5,
        'width': img.width,
        'height': img.height,
        'alwayson_scripts': cn_payload,
        'sampler_index': 'DDIM'
    }
    result = await auto1111('txt2img', payload, progress)
    return Image.from_base64(result['images'][0])
    

async def upscale(img: Image, target: Extent, progress: Progress):
    payload = {
        'init_images': [img.to_base64()],
        'resize_mode': 0,
        'denoising_strength': 0.3,
        'prompt': default_upscale_prompt,
        'sampler_index': 'DPM++ 2M Karras',
        'steps': 30,
        'cfg_scale': 5,
        'width': target.width,
        'height': target.height
    }
    result = await auto1111('img2img', payload, progress)
    return Image.from_base64(result['images'][0])
