from __future__ import annotations
import asyncio
from enum import Enum
from collections import deque
from itertools import product
import json
import struct
import uuid
from typing import NamedTuple, Optional, Sequence

from .comfyworkflow import ComfyWorkflow
from .image import Image, ImageCollection
from .network import RequestManager, NetworkError
from .websockets.src.websockets import client as websockets_client
from .websockets.src.websockets import exceptions as websockets_exceptions
from .style import Style, Styles
from .resources import ControlMode, MissingResource, ResourceKind, SDVersion, UpscalerName
from .resources import ResourceId, resource_id
from .settings import settings
from .util import client_logger as log
from . import resources, util


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


class ClientModels:
    """Collects names of AI models the client has access to."""

    checkpoints: dict[str, CheckpointInfo]
    vae: list[str]
    loras: list[str]
    upscalers: list[str]
    resources: dict[str, str | None]
    node_inputs: dict[str, dict[str, list[str | list | dict]]]

    def resource(
        self, kind: ResourceKind, identifier: ControlMode | UpscalerName | str, version: SDVersion
    ):
        id = ResourceId(kind, version, identifier)
        model = self.resources.get(id.string)
        if model is None:
            raise Exception(f"{id.name} not found")
        return model

    def version_of(self, checkpoint: str):
        return self.checkpoints[checkpoint].sd_version

    def for_version(self, version: SDVersion):
        return ModelDict(self, ResourceKind.upscaler, version)

    def for_checkpoint(self, checkpoint: str):
        return self.for_version(self.version_of(checkpoint))

    @property
    def upscale(self):
        return ModelDict(self, ResourceKind.upscaler, SDVersion.all)

    @property
    def default_upscaler(self):
        return self.resource(ResourceKind.upscaler, UpscalerName.default, SDVersion.all)


class ModelDict:
    """Provides access to filtered list of models matching a certain SD version."""

    _models: ClientModels
    kind: ResourceKind
    version: SDVersion

    def __init__(self, models: ClientModels, kind: ResourceKind, version: SDVersion):
        self._models = models
        self.kind = kind
        self.version = version

    def __getitem__(self, key: ControlMode | UpscalerName | str):
        return self._models.resource(self.kind, key, self.version)

    def find(self, key: ControlMode | UpscalerName | str):
        return self._models.resources.get(resource_id(self.kind, self.version, key))

    def for_version(self, version: SDVersion):
        return ModelDict(self._models, self.kind, version)

    @property
    def clip_vision(self):
        return self._models.resource(ResourceKind.clip_vision, "ip_adapter", SDVersion.all)

    @property
    def upscale(self):
        return ModelDict(self._models, ResourceKind.upscaler, SDVersion.all)

    @property
    def control(self):
        return ModelDict(self._models, ResourceKind.controlnet, self.version)

    @property
    def ip_adapter(self):
        return ModelDict(self._models, ResourceKind.ip_adapter, self.version)

    @property
    def inpaint(self):
        return ModelDict(self._models, ResourceKind.inpaint, SDVersion.all)

    @property
    def lora(self):
        return ModelDict(self._models, ResourceKind.lora, self.version)

    @property
    def fooocus_inpaint(self):
        assert self.version is SDVersion.sdxl
        return dict(
            head=self._models.resource(ResourceKind.inpaint, "fooocus_head", SDVersion.sdxl),
            patch=self._models.resource(ResourceKind.inpaint, "fooocus_patch", SDVersion.sdxl),
        )

    @property
    def all(self):
        return self._models

    @property
    def node_inputs(self):
        return self._models.node_inputs


class Client:
    """HTTP/WebSocket client which sends requests to and listens to messages from a ComfyUI server."""

    default_url = "http://127.0.0.1:8188"

    _requests = RequestManager()
    _id: str
    _jobs: deque[JobInfo]
    _active: Optional[JobInfo] = None

    url: str
    models: ClientModels
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

        # Check for required and optional model resources
        models = client.models
        models.node_inputs = {name: nodes[name]["input"].get("required", None) for name in nodes}
        available_resources = client.models.resources = {}

        control_models = nodes["ControlNetLoader"]["input"]["required"]["control_net_name"][0]
        available_resources.update(_find_control_models(control_models))

        clip_vision_models = nodes["CLIPVisionLoader"]["input"]["required"]["clip_name"][0]
        available_resources.update(_find_clip_vision_model(clip_vision_models))

        ip_adapter_models = nodes["IPAdapterModelLoader"]["input"]["required"]["ipadapter_file"][0]
        available_resources.update(_find_ip_adapters(ip_adapter_models))

        models.upscalers = nodes["UpscaleModelLoader"]["input"]["required"]["model_name"][0]
        available_resources.update(_find_upscalers(models.upscalers))

        inpaint_models = nodes["INPAINT_LoadInpaintModel"]["input"]["required"]["model_name"][0]
        available_resources.update(_find_inpaint_models(inpaint_models))

        loras = nodes["LoraLoader"]["input"]["required"]["lora_name"][0]
        available_resources.update(_find_loras(loras))

        # Retrieve list of checkpoints
        client._refresh_models(nodes, await client.try_inspect_checkpoints())

        # Check supported SD versions and make sure there is at least one
        missing = {ver: client._check_workload(ver) for ver in [SDVersion.sd15, SDVersion.sdxl]}
        client.supported_sd_versions = [ver for ver, miss in missing.items() if len(miss) == 0]
        if len(client.supported_sd_versions) == 0:
            raise missing[SDVersion.sd15][0]

        # Workarounds for DirectML
        if client.device_info.type == "privateuseone":
            # OmniSR causes a crash
            for n in [2, 3, 4]:
                id = resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_x(n))
                available_resources[id] = models.default_upscaler
            # IP-Adapter doesn't work https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/108
            for id in available_resources:
                if id.startswith(ResourceKind.ip_adapter.name):
                    available_resources[id] = None

        _ensure_supported_style(client)
        return client

    def __init__(self, url):
        self.url = url
        self.models = ClientModels()
        self._id = str(uuid.uuid4())
        self._jobs = deque()

    async def _get(self, op: str):
        return await self._requests.get(f"{self.url}/{op}")

    async def _post(self, op: str, data: dict):
        return await self._requests.post(f"{self.url}/{op}", data)

    async def enqueue(self, workflow: ComfyWorkflow, queue_front: bool = False):
        if settings.debug_dump_workflow:
            workflow.dump(util.log_dir)
        data = {"prompt": workflow.root, "client_id": self._id, "front": queue_front}
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
        models = self.models
        if checkpoint_info:
            models.checkpoints = {
                filename: CheckpointInfo(
                    filename,
                    SDVersion.from_string(info["base_model"]) or SDVersion.sd15,
                    info.get("is_inpaint", False),
                    info.get("is_refiner", False),
                )
                for filename, info in checkpoint_info.items()
                if info["base_model"] in ["sd15", "sdxl"]
            }
        else:
            models.checkpoints = {
                filename: CheckpointInfo.deduce_from_filename(filename)
                for filename in nodes["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            }
        models.vae = nodes["VAELoader"]["input"]["required"]["vae_name"][0]
        models.loras = nodes["LoraLoader"]["input"]["required"]["lora_name"][0]
        special_loras = [  # Filter out LCM, FaceID, etc. since they are added automatically
            res for id, res in models.resources.items() if id.startswith(ResourceKind.lora.name)
        ]
        models.loras = [l for l in models.loras if l not in special_loras]

    @property
    def supports_ip_adapter(self):
        return self.device_info.type != "privateuseone"

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
        models = self.models
        missing: list[MissingResource] = []
        for id in resources.required_resource_ids:
            if id.version is not SDVersion.all and id.version is not sdver:
                continue
            if models.resources[id.string] is None:
                missing.append(MissingResource(id.kind, [id.name]))
        if len(missing) == 0:
            log.info(f"{sdver.value}: supported")
        else:
            log.info(f"{sdver.value}: missing {len(missing)} models")
        return missing


def parse_url(url: str):
    url = url.strip("/")
    url = url.replace("0.0.0.0", "127.0.0.1")
    if not url.startswith("http"):
        url = f"http://{url}"
    return url


def websocket_url(url_http: str):
    return url_http.replace("http", "ws", 1)


def resolve_sd_version(style: Style, client: Client | None = None):
    if style.sd_version is SDVersion.auto:
        if client and style.sd_checkpoint in client.models.checkpoints:
            return client.models.version_of(style.sd_checkpoint)
        return style.sd_version.resolve(style.sd_checkpoint)
    return style.sd_version


def filter_supported_styles(styles: Styles, client: Client | None = None):
    if client:
        return [
            style
            for style in styles
            if resolve_sd_version(style, client) in client.supported_sd_versions
            and style.sd_checkpoint in client.models.checkpoints
        ]
    return list(styles)


def _find_model(
    model_list: Sequence[str],
    kind: ResourceKind,
    sdver: SDVersion,
    identifier: ControlMode | UpscalerName | str,
):
    search_paths = resources.search_path(kind, sdver, identifier)
    if search_paths is None:
        return None

    sanitize = lambda m: m.replace("\\", "/").lower()
    matches = (m for m in model_list if any(sanitize(p) in sanitize(m) for p in search_paths))
    # if there are multiple matches, prefer the one with "krita" in the path
    prio = sorted(matches, key=lambda m: 0 if "krita" in m else 1)
    found = next(iter(prio), None)
    model_id = identifier.name if isinstance(identifier, Enum) else identifier
    model_name = f"{kind.value} {model_id}"

    if found is None and resources.is_required(kind, sdver, identifier):
        log.warning(f"Missing {model_name} for {sdver.value}")
        log.info(
            f"-> No model matches search paths: {', '.join(sanitize(p) for p in search_paths)}"
        )
        log.info(f"-> Available models: {', '.join(sanitize(m) for m in model_list)}")
    elif found is None:
        log.info(
            f"Optional {model_name} for {sdver.value} not found (search path:"
            f" {', '.join(search_paths)})"
        )
    else:
        log.info(f"Found {model_name} for {sdver.value}: {found}")
    return found


_sd_versions = [SDVersion.sd15, SDVersion.sdxl]


def _find_control_models(model_list: Sequence[str]):
    kind = ResourceKind.controlnet
    return {
        resource_id(kind, ver, mode): _find_model(model_list, kind, ver, mode)
        for mode, ver in product(ControlMode, _sd_versions)
        if mode.is_control_net
    }


def _find_ip_adapters(model_list: Sequence[str]):
    kind = ResourceKind.ip_adapter
    return {
        resource_id(kind, ver, mode): _find_model(model_list, kind, ver, mode)
        for mode, ver in product(ControlMode, _sd_versions)
        if mode.is_ip_adapter
    }


def _find_clip_vision_model(model_list: Sequence[str]):
    model = _find_model(model_list, ResourceKind.clip_vision, SDVersion.all, "ip_adapter")
    if model is None:
        raise MissingResource(
            ResourceKind.clip_vision,
            resources.search_path(ResourceKind.clip_vision, SDVersion.all, "ip_adapter"),
        )
    return {resource_id(ResourceKind.clip_vision, SDVersion.all, "ip_adapter"): model}


def _find_upscalers(model_list: Sequence[str]):
    kind = ResourceKind.upscaler
    models = {
        resource_id(kind, SDVersion.all, name): _find_model(model_list, kind, SDVersion.all, name)
        for name in UpscalerName
    }
    default_id = resource_id(kind, SDVersion.all, UpscalerName.default)
    if models[default_id] is None and len(model_list) > 0:
        models[default_id] = models[resource_id(kind, SDVersion.all, UpscalerName.fast_4x)]
    return models


def _find_loras(model_list: Sequence[str]):
    kind = ResourceKind.lora
    return {
        resource_id(kind, ver, name): _find_model(model_list, kind, ver, name)
        for name, ver in product(["lcm", "face"], _sd_versions)
    }


def _find_inpaint_models(model_list: Sequence[str]):
    kind = ResourceKind.inpaint
    ids: list[tuple[SDVersion, str]] = [
        (SDVersion.all, "default"),
        (SDVersion.sdxl, "fooocus_head"),
        (SDVersion.sdxl, "fooocus_patch"),
    ]
    return {
        resource_id(kind, ver, name): _find_model(model_list, kind, ver, name) for ver, name in ids
    }


def _ensure_supported_style(client: Client):
    styles = filter_supported_styles(Styles.list(), client)
    if len(styles) == 0:
        checkpoint = next(
            cp.filename
            for cp in client.models.checkpoints.values()
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
            return Image.from_bytes(data[s:])
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
