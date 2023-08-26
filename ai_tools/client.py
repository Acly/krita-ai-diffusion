from enum import Enum
import json
import struct
import uuid
from typing import NamedTuple, Union, Sequence
from .image import Extent, Image, ImageCollection
from .network import RequestManager, Interrupted, NetworkError
from .settings import settings
from .util import compute_batch_size
from . import websockets


class ClientEvent(Enum):
    progress = 0
    finished = 1


class ClientMessage(NamedTuple):
    event: ClientEvent
    prompt_id: str
    progress: float
    images: ImageCollection


class Client:
    """HTTP/WebSocket client which sends requests to and listens to messages from a ComfyUI server."""

    default_url = "127.0.0.1:8188"

    _requests = RequestManager()
    _websocket: websockets.WebSocketClientProtocol
    _id: str
    _controlnet_inpaint_model: str
    _controlnet_tile_model: str

    url: str

    @staticmethod
    async def connect(url=default_url):
        client = Client(url)
        try:
            client._websocket = await websockets.connect(
                f"ws://{url}/ws?clientId={client._id}", max_size=2**30, read_limit=2**30
            )
        except OSError as e:
            raise NetworkError(
                e.errno, f"Could not connect to websocket server at {url}: {str(e)}", url
            )
        # upscalers = await result._get("sdapi/v1/upscalers")
        # settings.upscalers = [u["name"] for u in upscalers if not u["name"] == "None"]
        # controlnet_models = (await result._get("controlnet/model_list"))["model_list"]
        # result._controlnet_inpaint_model = _find_controlnet_model(
        #     controlnet_models, "control_v11p_sd15_inpaint"
        # )
        # result._controlnet_tile_model = _find_controlnet_model(
        #     controlnet_models, "control_v11f1e_sd15_tile"
        # )
        # controlnet_modules = (await result._get("controlnet/module_list"))["module_list"]
        # _find_controlnet_processor(controlnet_modules, "inpaint_only+lama")
        # _find_controlnet_processor(controlnet_modules, "tile_resample")
        return client

    def __init__(self, url):
        self.url = url
        self._id = str(uuid.uuid4())

    async def _get(self, op: str):
        return await self._requests.get(f"{self.url}/{op}")

    async def _post(self, op: str, data: dict):
        return await self._requests.post(f"{self.url}/{op}", data)

    async def enqueue(self, prompt: dict):
        data = {"prompt": prompt, "client_id": self._id}
        result = await self._post("prompt", data)
        return result["prompt_id"]

    async def listen(self):
        prompt_id = ""
        images = ImageCollection()

        async for msg in self._websocket:
            if isinstance(msg, bytes):
                image = _extract_message_png_image(msg)
                if image is not None:
                    images.append(image)
            elif isinstance(msg, str):
                msg = json.loads(msg)
                if msg["type"] == "execution_start":
                    prompt_id = msg["data"]["prompt_id"]
                elif msg["type"] == "progress":
                    data = msg["data"]
                    progress = data["value"] / data["max"]
                    yield ClientMessage(ClientEvent.progress, prompt_id, progress, None)
                elif msg["type"] == "executed":
                    prompt_id = _validate_executed_node(msg, len(images))
                    if prompt_id is not None:
                        yield ClientMessage(ClientEvent.finished, prompt_id, 1, images)
                        images = ImageCollection()

    async def interrupt(self):
        return await self._post("interrupt", {})


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


def _extract_message_png_image(data: memoryview):
    s = struct.calcsize(">II")
    if len(data) > s:
        event, format = struct.unpack_from(">II", data)
        # ComfyUI server.py: BinaryEventTypes.PREVIEW_IMAGE=1, PNG=2
        if event == 1 and format == 2:
            return Image.png_from_bytes(data[s:])
    return None


def _validate_executed_node(msg: dict, image_count: int):
    try:
        data = msg["data"]
        output = data["output"]["images"]
        if len(output) != image_count:
            print(
                "[krita-ai-diffusion] received number of images does not match:"
                f" {len(output)} != {image_count}"
            )
        if len(output) > 0 and "source" in output[0] and output[0]["type"] == "output":
            return data["prompt_id"]
    except:
        print("[krita-ai-diffusion] received unknown message format", msg)
        return None
