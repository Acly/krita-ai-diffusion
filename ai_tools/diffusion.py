import json
import urllib
from urllib.request import urlopen, Request
from typing import Tuple
from . import image
from .image import Extent, Image

automatic1111_url = 'http://127.0.0.1:7860/sdapi/v1'
default_upscale_prompt = 'highres 8k uhd'

def post(url, data):
    data_bytes = json.dumps(data).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(data_bytes))}
    req = Request(url, data=data_bytes, headers=headers, method='POST')
    try:
        with urlopen(req) as response:
            data = response.read()
            return json.loads(data)
    except Exception as e:
        print(e)
        raise e

def inpaint(img: Image, mask: Image, prompt: str):
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
    result = post(url=f'{automatic1111_url}/txt2img', data=payload)
    return Image.from_base64(result['images'][0])

def upscale(img: Image, target: Extent):
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
    result = post(url=f'{automatic1111_url}/img2img', data=payload)
    return Image.from_base64(result['images'][0])
