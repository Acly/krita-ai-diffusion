from enum import Enum
import json
import struct
import uuid
from typing import NamedTuple, Optional, Union, Sequence

from .comfyworkflow import ComfyWorkflow
from .image import Extent, Image, ImageCollection
from .network import RequestManager, Interrupted, NetworkError
from .settings import settings
from .util import compute_batch_size
from .websockets.src import websockets


class ClientEvent(Enum):
    progress = 0
    finished = 1


class ClientMessage(NamedTuple):
    event: ClientEvent
    prompt_id: str
    progress: float
    images: ImageCollection


class PromptInfo(NamedTuple):
    id: str
    node_count: int
    sample_count: int


class Progress:
    _nodes = 0
    _samples = 0
    _info: PromptInfo

    def __init__(self, prompt_info: PromptInfo):
        self._info = prompt_info

    def handle(self, msg: dict):
        prompt_id = msg["data"].get("prompt_id", None)
        if prompt_id is not None and prompt_id != self._info.id:
            return
        if msg["type"] == "executing":
            self._nodes += 1
        elif msg["type"] == "execution_cached":
            self._nodes += len(msg["data"]["nodes"])
        elif msg["type"] == "progress":
            self._samples += 1

    @property
    def value(self):
        # Add +1 to node count so progress doesn't go to 100% until images are received.
        node_part = self._nodes / (self._info.node_count + 1)
        sample_part = self._samples / self._info.sample_count
        return 0.2 * node_part + 0.8 * sample_part


class ResourceKind(Enum):
    checkpoint = "SD Checkpoint"
    controlnet = "ControlNet model"
    clip_vision = "CLIP Vision model"
    ip_adapter = "IP-Adapter model"
    node = "custom node"


class MissingResource(Exception):
    kind: ResourceKind
    names: Optional[Sequence[str]]

    def __init__(self, kind: ResourceKind, names: Optional[Sequence[str]] = None):
        self.kind = kind
        self.names = names

    def __str__(self):
        return f"Missing {self.kind.value}: {', '.join(self.names)}"


class Client:
    """HTTP/WebSocket client which sends requests to and listens to messages from a ComfyUI server."""

    default_url = "127.0.0.1:8188"

    _requests = RequestManager()
    _websocket: websockets.WebSocketClientProtocol
    _id: str
    _prompts: dict

    url: str
    checkpoints: Sequence[str]
    controlnet_model: dict
    clip_vision_model: str
    ip_adapter_model: str

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
        # Retrieve SD checkpoints
        sd = await client._get("object_info/CheckpointLoaderSimple")
        client.checkpoints = sd["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        if len(client.checkpoints) == 0:
            raise MissingResource(ResourceKind.checkpoint)
        if settings.sd_checkpoint == "<No checkpoints found>":
            settings.sd_checkpoint = client.checkpoints[0]

        # Retrieve ControlNet models
        cns = await client._get("object_info/ControlNetLoader")
        cns = cns["ControlNetLoader"]["input"]["required"]["control_net_name"][0]
        client.controlnet_model = {
            "inpaint": _find_controlnet_model(cns, "control_v11p_sd15_inpaint")
        }

        # Retrieve CLIPVision models
        cv = await client._get("object_info/CLIPVisionLoader")
        cv = cv["CLIPVisionLoader"]["input"]["required"]["clip_name"][0]
        client.clip_vision_model = _find_clip_vision_model(cv, "SD1.5")

        # Retrieve IP-Adapter model
        ip = await client._get("object_info/IPAdapter")
        ip = ip["IPAdapter"]["input"]["required"]["model_name"][0]
        client.ip_adapter_model = _find_ip_adapter(ip, "sd15")

        return client

    def __init__(self, url):
        self.url = url
        self._id = str(uuid.uuid4())
        self._prompts = {}

    async def _get(self, op: str):
        return await self._requests.get(f"{self.url}/{op}")

    async def _post(self, op: str, data: dict):
        return await self._requests.post(f"{self.url}/{op}", data)

    async def enqueue(self, workflow: ComfyWorkflow):
        data = {"prompt": workflow.root, "client_id": self._id}
        result = await self._post("prompt", data)
        prompt_id = result["prompt_id"]
        self._prompts[prompt_id] = PromptInfo(prompt_id, workflow.node_count, workflow.sample_count)
        return prompt_id

    async def listen(self):
        prompt_id = None
        progress = None
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
                    progress = Progress(self._prompts[prompt_id])
                elif msg["type"] in ("execution_cached", "executing", "progress") and prompt_id:
                    progress.handle(msg)
                    yield ClientMessage(ClientEvent.progress, prompt_id, progress.value, None)
                elif msg["type"] == "executed":
                    prompt_id = _validate_executed_node(msg, len(images))
                    if prompt_id is not None:
                        yield ClientMessage(ClientEvent.finished, prompt_id, 1, images)
                        del self._prompts[prompt_id]
                        prompt_id = None
                        images = ImageCollection()

    async def interrupt(self):
        return await self._post("interrupt", {})


def _find_controlnet_model(model_list: Sequence[str], model_name: str):
    model = next((model for model in model_list if model.startswith(model_name)), None)
    if model is None:
        raise MissingResource(ResourceKind.controlnet, [model_name])
    return model


def _find_clip_vision_model(model_list: Sequence[str], sdver: str):
    model_name = "clip_vision_g.safetensors" if sdver == "SDXL" else "pytorch_model.bin"
    match = lambda x: (sdver == "SDXL" or sdver in x) and model_name in x
    model = next((m for m in model_list if match(m)), None)
    if model is None:
        full_name = model_name if sdver == "SDXL" else f"{sdver}/{model_name}"
        raise MissingResource(ResourceKind.clip_vision, [full_name])
    return model


def _find_ip_adapter(model_list: Sequence[str], sdver: str):
    model_name = f"ip-adapter_{sdver}"
    model = next((m for m in model_list if model_name in m), None)
    if model is None:
        raise MissingResource(ResourceKind.ip_adapter, [model_name])
    return model


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
