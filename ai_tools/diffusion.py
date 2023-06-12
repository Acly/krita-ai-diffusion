import json
import urllib
from urllib.request import urlopen, Request
from typing import Tuple
from . import image
from .image import Extent, Image

from PyQt5.QtCore import QByteArray, QUrl
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest

automatic1111_url = 'http://127.0.0.1:7860/sdapi/v1'
default_upscale_prompt = 'highres 8k uhd'

class RequestManager:
    def __init__(self):
        self._net = QNetworkAccessManager()

    def post(self, url, data, cb):
        data_bytes = QByteArray(json.dumps(data).encode("utf-8"))
        request = QNetworkRequest(QUrl(url))
        request.setHeader(QNetworkRequest.ContentTypeHeader, 'application/json')
        request.setHeader(QNetworkRequest.ContentLengthHeader, data_bytes.size())

        
        reply = self._net.post(request, data_bytes)
        def ret_cb():
            cb(json.loads(reply.readAll().data()))
        reply.finished.connect(ret_cb)

requests = RequestManager()


def inpaint(img: Image, mask: Image, prompt: str, cb):
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
    def ret_cb(result):
        cb(Image.from_base64(result['images'][0]))
    requests.post(f'{automatic1111_url}/txt2img', payload, ret_cb)
    

def upscale(img: Image, target: Extent, cb):
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
    def ret_cb(result):
        cb(Image.from_base64(result['images'][0]))
    requests.post(f'{automatic1111_url}/img2img', payload, ret_cb)
