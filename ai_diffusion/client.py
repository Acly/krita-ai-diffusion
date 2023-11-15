from __future__ import annotations
import asyncio
from enum import Enum
from collections import deque
import json
import struct
import uuid
from typing import NamedTuple, Optional, Union, Sequence

from .comfyworkflow import ComfyWorkflow
from .image import Image, ImageCollection
from .network import RequestManager, NetworkError
from .websockets.src.websockets import client as websockets_client
from .websockets.src.websockets import exceptions as websockets_exceptions
from .style import SDVersion, Style, Styles
from .resources import ControlMode, MissingResource, ResourceKind
from . import resources
from .util import ensure, is_windows, client_logger as log


class ClientEvent(Enum):
    progress = 0
    finished = 1
    interrupted = 2
    error = 3
    connected = 4
    disconnected = 5


class ClientMessage(NamedTuple):
    event: ClientEvent
    job_id: str = ""
    progress: float = 0
    images: Optional[ImageCollection] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class JobInfo(NamedTuple):
    id: str
    node_count: int
    sample_count: int


class Progress:
    _nodes = 0
    _samples = 0
    _info: JobInfo

    def __init__(self, job_info: JobInfo):
        self._info = job_info

    def handle(self, msg: dict):
        id = msg["data"].get("prompt_id", None)
        if id is not None and id != self._info.id:
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
        sample_part = self._samples / max(self._info.sample_count, 1)
        return 0.2 * node_part + 0.8 * sample_part


class DeviceInfo(NamedTuple):
    type: str
    name: str
    vram: int  # in GB

    @staticmethod
    def parse(data: dict):
        try:
            name = data["devices"][0]["name"]
            name = name.split(":")[1] if ":" in name else name
            vram = int(round(data["devices"][0]["vram_total"] / (1024**3)))
            return DeviceInfo(data["devices"][0]["type"], name, vram)
        except Exception as e:
            log.error(f"Could not parse device info {data}: {str(e)}")
            return DeviceInfo("cpu", "unknown", 0)


class CheckpointInfo(NamedTuple):
    filename: str
    sd_version: SDVersion
    is_inpaint: bool = False
    is_refiner: bool = False

    @property
    def name(self):
        return self.filename.removesuffix(".safetensors")

    @staticmethod
    def deduce_from_filename(filename: str):
        return CheckpointInfo(
            filename,
            SDVersion.from_checkpoint_name(filename),
            "inpaint" in filename.lower(),
            "refiner" in filename.lower(),
        )


class Client:
    """HTTP/WebSocket client which sends requests to and listens to messages from a ComfyUI server."""

    default_url = "http://127.0.0.1:8188"

    _requests = RequestManager()
    _id: str
    _jobs: deque[JobInfo]
    _active: Optional[JobInfo] = None

    url: str
    checkpoints: dict[str, CheckpointInfo]
    vae_models: list[str]
    lora_models: list[str]
    upscalers: list[str]
    default_upscaler: str
    control_model: dict[ControlMode, dict[SDVersion, str | None]]
    clip_vision_model: str
    ip_adapter_model: dict[SDVersion, str | None]
    ip_adapter_has_weight_type = False
    supported_sd_versions: list[SDVersion]
    device_info: DeviceInfo

    @staticmethod
    async def connect(url=default_url):
        client = Client(parse_url(url))
        log.info(f"Connecting to {client.url}")

        # Retrieve system info
        client.device_info = DeviceInfo.parse(await client._get("system_stats"))

        # Check custom nodes
        nodes = await client._get("object_info")
        missing = [
            package
            for package in resources.required_custom_nodes
            if any(node not in nodes for node in package.nodes)
        ]
        if len(missing) > 0:
            raise MissingResource(ResourceKind.node, missing)

        # Retrieve list of checkpoints
        client._refresh_models(nodes, await client.try_inspect_checkpoints())

        # Retrieve ControlNet models
        cns = nodes["ControlNetLoader"]["input"]["required"]["control_net_name"][0]
        client.control_model = {mode: _find_control_model(cns, mode) for mode in ControlMode}

        # Retrieve CLIPVision models
        cv = nodes["CLIPVisionLoader"]["input"]["required"]["clip_name"][0]
        client.clip_vision_model = _find_clip_vision_model(cv, "SD1.5")

        # Retrieve IP-Adapter model
        ip = nodes["IPAdapterModelLoader"]["input"]["required"]["ipadapter_file"][0]
        client.ip_adapter_model = {
            ver: _find_ip_adapter(ip, ver) for ver in [SDVersion.sd15, SDVersion.sdxl]
        }
        client.ip_adapter_has_weight_type = (
            "weight_type" in nodes["IPAdapterApply"]["input"]["required"]
        )

        # Retrieve upscale models
        client.upscalers = nodes["UpscaleModelLoader"]["input"]["required"]["model_name"][0]
        if len(client.upscalers) == 0:
            raise MissingResource(ResourceKind.upscaler)
        client.default_upscaler = ensure(
            _find_upscaler(client.upscalers, "4x_NMKD-Superscale-SP_178000_G.pth")
        )

        # Check supported SD versions and make sure there is at least one
        missing = {ver: client._check_workload(ver) for ver in [SDVersion.sd15, SDVersion.sdxl]}
        client.supported_sd_versions = [ver for ver, miss in missing.items() if len(miss) == 0]
        if len(client.supported_sd_versions) == 0:
            raise missing[SDVersion.sd15][0]

        _ensure_supported_style(client)
        return client

    def __init__(self, url):
        self.url = url
        self._id = str(uuid.uuid4())
        self._jobs = deque()

    async def _get(self, op: str):
        return await self._requests.get(f"{self.url}/{op}")

    async def _post(self, op: str, data: dict):
        return await self._requests.post(f"{self.url}/{op}", data)

    async def enqueue(self, workflow: ComfyWorkflow):
        data = {"prompt": workflow.root, "client_id": self._id}
        result = await self._post("prompt", data)
        job_id = result["prompt_id"]
        self._jobs.append(JobInfo(job_id, workflow.node_count, workflow.sample_count))
        return job_id

    async def listen(self):
        url = websocket_url(self.url)
        async for websocket in websockets_client.connect(
            f"{url}/ws?clientId={self._id}", max_size=2**30, read_limit=2**30, ping_timeout=60
        ):
            try:
                async for msg in self._listen(websocket):
                    yield msg
            except websockets_exceptions.ConnectionClosedError as e:
                log.warning(f"Websocket connection closed: {str(e)}")
                yield ClientMessage(ClientEvent.disconnected)
            except OSError as e:
                msg = f"Could not connect to websocket server at {url}: {str(e)}"
                yield ClientMessage(ClientEvent.error, error=msg)
            except asyncio.CancelledError:
                await websocket.close()
                self._active = None
                self._jobs.clear()
                break
            except Exception as e:
                log.exception("Unhandled exception in websocket listener")
                yield ClientMessage(ClientEvent.error, error=str(e))

    async def _listen(self, websocket: websockets_client.WebSocketClientProtocol):
        progress = None
        images = ImageCollection()
        result = None

        async for msg in websocket:
            if isinstance(msg, bytes):
                image = _extract_message_png_image(memoryview(msg))
                if image is not None:
                    images.append(image)

            elif isinstance(msg, str):
                msg = json.loads(msg)

                if msg["type"] == "status":
                    yield ClientMessage(ClientEvent.connected)

                if msg["type"] == "execution_start":
                    id = msg["data"]["prompt_id"]
                    self._active = self._start_job(id)
                    if self._active:
                        progress = Progress(self._active)
                        images = ImageCollection()
                        result = None

                if msg["type"] == "execution_interrupted":
                    job = self._get_active_job(msg["data"]["prompt_id"])
                    if job:
                        self._clear_job(job.id)
                        yield ClientMessage(ClientEvent.interrupted, job.id)

                if msg["type"] == "executing" and msg["data"]["node"] is None:
                    job_id = msg["data"]["prompt_id"]
                    if self._clear_job(job_id):
                        # Usually we don't get here because finished, interrupted or error is sent first.
                        # But it may happen if the entire execution is cached and no images are sent.
                        yield ClientMessage(ClientEvent.finished, job_id, 1, images)

                elif msg["type"] in ("execution_cached", "executing", "progress"):
                    if self._active and progress:
                        progress.handle(msg)
                        yield ClientMessage(ClientEvent.progress, self._active.id, progress.value)
                    else:
                        log.error(f"Received message {msg} but there is no active job")

                if msg["type"] == "executed":
                    job = self._get_active_job(msg["data"]["prompt_id"])
                    pose_json = _extract_pose_json(msg)
                    if job and pose_json:
                        result = pose_json
                    elif job and _validate_executed_node(msg, len(images)):
                        self._clear_job(job.id)
                        yield ClientMessage(ClientEvent.finished, job.id, 1, images, result)

                if msg["type"] == "execution_error":
                    job = self._get_active_job(msg["data"]["prompt_id"])
                    if job:
                        error = msg["data"].get("exception_message", "execution_error")
                        traceback = msg["data"].get("traceback", "no traceback")
                        log.error(f"Job {job.id} failed: {error}\n{traceback}")
                        self._clear_job(job.id)
                        yield ClientMessage(ClientEvent.error, job.id, 0, error=error)

    async def interrupt(self):
        await self._post("interrupt", {})

    async def clear_queue(self):
        await self._post("queue", {"clear": True})
        self._jobs.clear()

    async def try_inspect_checkpoints(self):
        try:
            return await self._get("etn/model_info")
        except NetworkError:
            return None  # server has old external tooling version

    @property
    def queued_count(self):
        return len(self._jobs)

    @property
    def is_executing(self):
        return self._active is not None

    async def refresh(self):
        nodes, info = await asyncio.gather(self._get("object_info"), self.try_inspect_checkpoints())
        self._refresh_models(nodes, info)

    def _refresh_models(self, nodes: dict, checkpoint_info: Optional[dict]):
        if checkpoint_info:
            self.checkpoints = {
                filename: CheckpointInfo(
                    filename,
                    SDVersion.from_string(info["base_model"]) or SDVersion.sd15,
                    info.get("is_inpaint", False),
                    info.get("is_refiner", False),
                )
                for filename, info in checkpoint_info.items()
            }
        else:
            self.checkpoints = {
                filename: CheckpointInfo.deduce_from_filename(filename)
                for filename in nodes["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            }
        self.vae_models = nodes["VAELoader"]["input"]["required"]["vae_name"][0]
        self.lora_models = nodes["LoraLoader"]["input"]["required"]["lora_name"][0]

    def _get_active_job(self, id: str) -> Optional[JobInfo]:
        if self._active and self._active.id == id:
            return self._active
        elif self._active:
            log.warning(f"Received message for job {id}, but job {self._active.id} is active")
        if len(self._jobs) == 0:
            log.warning(f"Received unknown job {id}")
            return None
        active = next((j for j in self._jobs if j.id == id), None)
        if active is not None:
            return active
        return None

    def _start_job(self, id: str):
        if self._active is not None:
            log.warning(f"Started job {id}, but {self._active.id} was never finished")
        if len(self._jobs) == 0:
            log.warning(f"Received unknown job {id}")
            return None
        if self._jobs[0].id == id:
            return self._jobs.popleft()
        log.warning(f"Started job {id}, but {self._jobs[0].id} was expected")
        active = next((j for j in self._jobs if j.id == id), None)
        if active is not None:
            self._jobs.remove(active)
            return active
        return None

    def _clear_job(self, job_id: str):
        if self._active is not None and self._active.id == job_id:
            self._active = None
            return True
        return False

    def _check_workload(self, sdver: SDVersion) -> list[MissingResource]:
        missing: list[MissingResource] = []
        if not self.clip_vision_model:
            missing.append(MissingResource(ResourceKind.clip_vision))
        if not self.default_upscaler:
            missing.append(MissingResource(ResourceKind.upscaler))
        if not self.ip_adapter_model[sdver]:
            missing.append(MissingResource(ResourceKind.ip_adapter))
        if not any(cp.sd_version is sdver for cp in self.checkpoints.values()):
            missing.append(MissingResource(ResourceKind.checkpoint))
        if sdver is SDVersion.sd15:
            if not self.control_model[ControlMode.inpaint][SDVersion.sd15]:
                missing.append(MissingResource(ResourceKind.controlnet, ["ControlNet inpaint"]))
            if not self.control_model[ControlMode.blur][SDVersion.sd15]:
                missing.append(MissingResource(ResourceKind.controlnet, ["ControlNet tile"]))
        if len(missing) == 0:
            log.info(f"{sdver.value}: supported")
        else:
            log.info(f"{sdver.value}: missing resources {', '.join(m.kind.value for m in missing)}")
        return missing


def parse_url(url: str):
    url = url.strip("/")
    if not url.startswith("http"):
        url = f"http://{url}"
    return url


def websocket_url(url_http: str):
    return url_http.replace("http", "ws", 1)


def resolve_sd_version(style: Style, client: Optional[Client] = None):
    if style.sd_version is SDVersion.auto:
        if client and style.sd_checkpoint in client.checkpoints:
            return client.checkpoints[style.sd_checkpoint].sd_version
        return style.sd_version.resolve(style.sd_checkpoint)
    return style.sd_version


def filter_supported_styles(styles: Styles, client: Optional[Client] = None):
    if client:
        return [
            style
            for style in styles
            if resolve_sd_version(style, client) in client.supported_sd_versions
            and style.sd_checkpoint in client.checkpoints
        ]
    return list(styles)


def _find_control_model(model_list: Sequence[str], mode: ControlMode):
    def match_filename(path: str, name: str):
        path_sep = "\\" if is_windows else "/"
        return path.startswith(name) or path.split(path_sep)[-1].startswith(name)

    def find(name: Union[str, list, None]):
        if name is None:
            return None
        names = [name] if isinstance(name, str) else name
        matches_name = lambda model: any(match_filename(model, name) for name in names)
        model = next((model for model in model_list if matches_name(model)), None)
        return model

    return {version: find(mode.filenames(version)) for version in [SDVersion.sd15, SDVersion.sdxl]}


def _find_clip_vision_model(model_list: Sequence[str], sdver: str):
    assert sdver == "SD1.5", "Using SD1.5 clip vision model also for SDXL IP-adapter"
    model_name = "pytorch_model.bin"
    match = lambda x: sdver in x and model_name in x
    model = next((m for m in model_list if match(m)), None)
    if model is None:
        full_name = f"{sdver}/{model_name}"
        raise MissingResource(ResourceKind.clip_vision, [full_name])
    return model


def _find_ip_adapter(model_list: Sequence[str], sdver: SDVersion):
    model_name = "ip-adapter_sd15" if sdver is SDVersion.sd15 else "ip-adapter_sdxl_vit-h"
    model = next((m for m in model_list if model_name in m), None)
    return model


def _find_upscaler(model_list: Sequence[str], model_name: str):
    if model_name in model_list:
        return model_name
    log.warning(f"Could not find default upscaler {model_name}, using {model_list[0]} instead")
    return model_list[0]


def _ensure_supported_style(client: Client):
    styles = filter_supported_styles(Styles.list(), client)
    if len(styles) == 0:
        checkpoint = next(
            cp.filename
            for cp in client.checkpoints.values()
            if cp.sd_version in client.supported_sd_versions
        )
        log.info(f"No supported styles found, creating default style with checkpoint {checkpoint}")
        default = next((s for s in Styles.list() if s.filename == "default.json"), None)
        if default:
            default.sd_checkpoint = checkpoint
            default.save()
        else:
            Styles.list().create("default", checkpoint)


def _extract_message_png_image(data: memoryview):
    s = struct.calcsize(">II")
    if len(data) > s:
        event, format = struct.unpack_from(">II", data)
        # ComfyUI server.py: BinaryEventTypes.PREVIEW_IMAGE=1, PNG=2
        if event == 1 and format == 2:
            return Image.png_from_bytes(data[s:])
    return None


def _extract_pose_json(msg: dict):
    try:
        output = msg["data"]["output"]
        if "openpose_json" in output:
            return json.loads(output["openpose_json"][0])
    except Exception as e:
        log.warning(f"Error processing message, error={str(e)}, msg={msg}")
    return None


def _validate_executed_node(msg: dict, image_count: int):
    try:
        output = msg["data"]["output"]
        assert "openpose_json" not in output

        images = output["images"]
        if len(images) != image_count:  # not critical
            log.warning(f"Received number of images does not match: {len(images)} != {image_count}")
        if len(images) > 0 and "source" in images[0] and images[0]["type"] == "output":
            return True
    except Exception as e:
        log.warning(f"Error processing message, error={str(e)}, msg={msg}")
        return False
