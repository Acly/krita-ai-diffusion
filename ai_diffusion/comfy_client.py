from __future__ import annotations
import asyncio
import json
import struct
import uuid
from dataclasses import dataclass
from enum import Enum
from collections import deque
from itertools import chain, product
from typing import Any, Optional, Sequence

from .api import WorkflowInput
from .client import Client, CheckpointInfo, ClientMessage, ClientEvent, DeviceInfo, ClientModels
from .client import SharedWorkflow, TranslationPackage, ClientFeatures, TextOutput
from .client import MissingResources, filter_supported_styles, loras_to_upload
from .files import FileFormat
from .image import Image, ImageCollection
from .network import RequestManager, NetworkError
from .websockets.src import websockets
from .style import Styles
from .resources import ControlMode, ResourceId, ResourceKind, Arch
from .resources import CustomNode, UpscalerName, resource_id
from .settings import PerformanceSettings, settings
from .localization import translate as _
from .util import client_logger as log
from .workflow import create as create_workflow
from . import resources, util

if util.is_macos:
    import os

    if "SSL_CERT_FILE" not in os.environ:
        os.environ["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"


@dataclass
class JobInfo:
    local_id: str
    work: WorkflowInput
    front: bool = False
    remote_id: str | asyncio.Future[str] | None = None
    node_count: int = 0
    sample_count: int = 0

    def __str__(self):
        return f"Job[local={self.local_id}, remote={self.remote_id}]"

    @staticmethod
    def create(work: WorkflowInput, front: bool = False):
        return JobInfo(str(uuid.uuid4()), work, front)

    async def get_remote_id(self):
        if isinstance(self.remote_id, asyncio.Future):
            self.remote_id = await self.remote_id
        return self.remote_id


class Progress:
    _nodes = 0
    _samples = 0
    _info: JobInfo

    def __init__(self, job_info: JobInfo):
        self._info = job_info

    def handle(self, msg: dict):
        id = msg["data"].get("prompt_id", None)
        if id is not None and id != self._info.remote_id:
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


class ComfyClient(Client):
    """HTTP/WebSocket client which sends requests to and listens to messages from a ComfyUI server."""

    default_url = "http://127.0.0.1:8188"

    def __init__(self, url):
        self.url = url
        self.models = ClientModels()
        self._requests = RequestManager()
        self._id = str(uuid.uuid4())
        self._active: Optional[JobInfo] = None
        self._features: ClientFeatures = ClientFeatures()
        self._supported_archs: dict[Arch, list[ResourceId]] = {}
        self._messages = asyncio.Queue()
        self._queue = asyncio.Queue()
        self._jobs: deque[JobInfo] = deque()
        self._is_connected = False

    @staticmethod
    async def connect(url=default_url, access_token=""):
        client = ComfyClient(parse_url(url))
        log.info(f"Connecting to {client.url}")

        # Retrieve system info
        client.device_info = DeviceInfo.parse(await client._get("system_stats"))

        # Try to establish websockets connection
        wsurl = websocket_url(client.url)
        try:
            async with websockets.connect(f"{wsurl}/ws?clientId={client._id}"):
                pass
        except Exception as e:
            msg = _("Could not establish websocket connection at") + f" {wsurl}: {str(e)}"
            raise Exception(msg)

        # Check custom nodes
        nodes = await client._get("object_info")
        missing = _check_for_missing_nodes(nodes)
        if len(missing) > 0:
            raise MissingResources(missing)

        client._features = ClientFeatures(
            ip_adapter=True,
            translation=True,
            languages=await _list_languages(client),
            wave_speed="ApplyFBCacheOnModel" in nodes,
        )

        # Check for required and optional model resources
        models = client.models
        models.node_inputs = {name: nodes[name]["input"] for name in nodes}
        available_resources = client.models.resources = {}

        clip_models = nodes["DualCLIPLoader"]["input"]["required"]["clip_name1"][0]
        available_resources.update(_find_text_encoder_models(clip_models))
        if clip_gguf := nodes.get("DualCLIPLoaderGGUF", None):
            clip_gguf_models = clip_gguf["input"]["required"]["clip_name1"][0]
            available_resources.update(_find_text_encoder_models(clip_gguf_models))

        vae_models = nodes["VAELoader"]["input"]["required"]["vae_name"][0]
        available_resources.update(_find_vae_models(vae_models))

        control_models = nodes["ControlNetLoader"]["input"]["required"]["control_net_name"][0]
        available_resources.update(_find_control_models(control_models))

        clip_vision_models = nodes["CLIPVisionLoader"]["input"]["required"]["clip_name"][0]
        available_resources.update(_find_clip_vision_model(clip_vision_models))

        ip_adapter_models = nodes["IPAdapterModelLoader"]["input"]["required"]["ipadapter_file"][0]
        available_resources.update(_find_ip_adapters(ip_adapter_models))

        style_models = nodes["StyleModelLoader"]["input"]["required"]["style_model_name"][0]
        available_resources.update(_find_style_models(style_models))

        models.upscalers = nodes["UpscaleModelLoader"]["input"]["required"]["model_name"][0]
        available_resources.update(_find_upscalers(models.upscalers))

        inpaint_models = nodes["INPAINT_LoadInpaintModel"]["input"]["required"]["model_name"][0]
        available_resources.update(_find_inpaint_models(inpaint_models))

        loras = nodes["LoraLoader"]["input"]["required"]["lora_name"][0]
        available_resources.update(_find_loras(loras))

        # Retrieve list of checkpoints
        checkpoints = await client.try_inspect("checkpoints")
        diffusion_models = await client.try_inspect("diffusion_models")
        diffusion_models.update(await client.try_inspect("unet_gguf"))
        client._refresh_models(nodes, checkpoints, diffusion_models)

        # Check supported base models and make sure there is at least one
        client._supported_archs = {ver: client._check_workload(ver) for ver in Arch.list()}
        supported_workloads = [
            arch for arch, miss in client._supported_archs.items() if len(miss) == 0
        ]
        log.info("Supported workloads: " + ", ".join(arch.value for arch in supported_workloads))
        if len(supported_workloads) == 0:
            raise MissingResources(client._supported_archs)

        # Workarounds for DirectML
        if client.device_info.type == "privateuseone":
            # OmniSR causes a crash
            for n in [2, 3, 4]:
                id = resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_x(n))
                available_resources[id] = models.default_upscaler

        _ensure_supported_style(client)
        return client

    async def _get(self, op: str, timeout: float | None = 30):
        return await self._requests.get(f"{self.url}/{op}", timeout=timeout)

    async def _post(self, op: str, data: dict):
        return await self._requests.post(f"{self.url}/{op}", data)

    async def enqueue(self, work: WorkflowInput, front: bool = False):
        job = JobInfo.create(work, front=front)
        await self._queue.put(job)
        return job.local_id

    async def _report(self, event: ClientEvent, job_id: str, value: float = 0, **kwargs):
        await self._messages.put(ClientMessage(event, job_id, value, **kwargs))

    async def _run(self):
        assert self._is_connected
        try:
            while self._is_connected:
                job = await self._queue.get()
                try:
                    await self._run_job(job)
                except Exception as e:
                    log.exception(f"Unhandled exception while processing {job}")
                    await self._report(ClientEvent.error, job.local_id, error=str(e))
        except asyncio.CancelledError:
            pass

    async def _run_job(self, job: JobInfo):
        await self.upload_loras(job.work, job.local_id)
        workflow = create_workflow(job.work, self.models)
        job.node_count = workflow.node_count
        job.sample_count = workflow.sample_count
        if settings.debug_dump_workflow:
            workflow.dump(util.log_dir)

        data = {"prompt": workflow.root, "client_id": self._id, "front": job.front}
        job.remote_id = asyncio.get_running_loop().create_future()
        self._jobs.append(job)
        try:
            result = await self._post("prompt", data)
            job.remote_id.set_result(result["prompt_id"])
        except Exception as e:
            job.remote_id.set_result("ERROR")
            if self._jobs[0] == job:
                self._jobs.popleft()
            raise e

    async def _listen(self):
        url = websocket_url(self.url)
        async for websocket in websockets.connect(
            f"{url}/ws?clientId={self._id}", max_size=2**30, ping_timeout=60
        ):
            try:
                await self._subscribe_workflows()
                await self._listen_websocket(websocket)
            except websockets.exceptions.ConnectionClosedError as e:
                log.warning(f"Websocket connection closed: {str(e)}")
            except OSError as e:
                msg = _("Could not connect to websocket server at") + f"{url}: {str(e)}"
                await self._report(ClientEvent.error, "", error=msg)
            except asyncio.CancelledError:
                await websocket.close()
                self._active = None
                self._jobs.clear()
                break
            except Exception as e:
                log.exception("Unhandled exception in websocket listener")
                await self._report(ClientEvent.error, "", error=str(e))
            finally:
                await self._report(ClientEvent.disconnected, "")

    async def _listen_websocket(self, websocket: websockets.ClientConnection):
        progress: Progress | None = None
        images = ImageCollection()
        last_images = ImageCollection()
        result = None

        async for msg in websocket:
            if isinstance(msg, bytes):
                image = _extract_message_png_image(memoryview(msg))
                if image is not None:
                    images.append(image)

            elif isinstance(msg, str):
                msg = json.loads(msg)

                if msg["type"] == "status":
                    await self._report(ClientEvent.connected, "")

                if msg["type"] == "execution_start":
                    id = msg["data"]["prompt_id"]
                    self._active = await self._start_job(id)
                    if self._active is not None:
                        progress = Progress(self._active)
                        images = ImageCollection()
                        result = None

                if msg["type"] == "execution_interrupted":
                    job = self._get_active_job(msg["data"]["prompt_id"])
                    if job:
                        self._clear_job(job.remote_id)
                        await self._report(ClientEvent.interrupted, job.local_id)

                if msg["type"] == "executing" and msg["data"]["node"] is None:
                    job_id = msg["data"]["prompt_id"]
                    if local_id := self._clear_job(job_id):
                        if len(images) == 0:
                            # It may happen if the entire execution is cached and no images are sent.
                            images = last_images
                        if len(images) == 0:
                            # Still no images. Potential scenario: execution cached, but previous
                            # generation happened before the client was connected.
                            err = "No new images were generated because the inputs did not change."
                            await self._report(ClientEvent.error, local_id, error=err)
                        else:
                            last_images = images
                            await self._report(
                                ClientEvent.finished, local_id, 1, images=images, result=result
                            )

                elif msg["type"] in ("execution_cached", "executing", "progress"):
                    if self._active is not None and progress is not None:
                        progress.handle(msg)
                        await self._report(
                            ClientEvent.progress, self._active.local_id, progress.value
                        )
                    else:
                        log.error(f"Received message {msg} but there is no active job")

                if msg["type"] == "executed":
                    if job := self._get_active_job(msg["data"]["prompt_id"]):
                        text_output = _extract_text_output(job.local_id, msg)
                        if text_output is not None:
                            await self._messages.put(text_output)
                        pose_json = _extract_pose_json(msg)
                        if pose_json is not None:
                            result = pose_json

                if msg["type"] == "execution_error":
                    job = self._get_active_job(msg["data"]["prompt_id"])
                    if job:
                        error = msg["data"].get("exception_message", "execution_error")
                        traceback = msg["data"].get("traceback", "no traceback")
                        log.error(f"Job {job} failed: {error}\n{traceback}")
                        self._clear_job(job.remote_id)
                        await self._report(ClientEvent.error, job.local_id, error=error)

                if msg["type"] == "etn_workflow_published":
                    name = f"{msg['data']['publisher']['name']} ({msg['data']['publisher']['id']})"
                    workflow = SharedWorkflow(name, msg["data"]["workflow"])
                    await self._report(ClientEvent.published, "", result=workflow)

    async def listen(self):
        self._is_connected = True
        self._job_runner = asyncio.create_task(self._run())
        self._websocket_listener = asyncio.create_task(self._listen())

        try:
            while self._is_connected:
                yield await self._messages.get()
        except asyncio.CancelledError:
            pass

    async def interrupt(self):
        await self._post("interrupt", {})

    async def clear_queue(self):
        while not self._queue.empty():
            try:
                job = self._queue.get_nowait()
                await self._report(ClientEvent.interrupted, job.local_id)
            except asyncio.QueueEmpty:
                break

        await self._post("queue", {"clear": True})
        self._jobs.clear()

    async def disconnect(self):
        if self._is_connected:
            self._is_connected = False
            self._job_runner.cancel()
            self._websocket_listener.cancel()
            await asyncio.gather(
                self._job_runner,
                self._websocket_listener,
                self._unsubscribe_workflows(),
            )

    async def try_inspect(self, folder_name: str) -> dict[str, Any]:
        try:
            return await self._get(f"api/etn/model_info/{folder_name}", timeout=90)
        except NetworkError as e:
            log.error(f"Error while inspecting models in {folder_name}: {str(e)}")
            return {}

    @property
    def queued_count(self):
        return len(self._jobs) + self._queue.qsize()

    @property
    def is_executing(self):
        return self._active is not None

    async def refresh(self):
        nodes, checkpoints, diffusion_models, diffusion_gguf = await asyncio.gather(
            self._get("object_info"),
            self.try_inspect("checkpoints"),
            self.try_inspect("diffusion_models"),
            self.try_inspect("unet_gguf"),
        )
        diffusion_models.update(diffusion_gguf)
        self._refresh_models(nodes, checkpoints, diffusion_models)

    def _refresh_models(self, nodes: dict, checkpoints: dict | None, diffusion_models: dict | None):
        models = self.models

        def parse_model_info(models: dict, model_format: FileFormat):
            parsed = (
                (
                    filename,
                    Arch.from_string(info["base_model"], info.get("type", "eps")),
                    info.get("is_inpaint", False),
                    info.get("is_refiner", False),
                )
                for filename, info in models.items()
            )
            return {
                filename: CheckpointInfo(filename, arch, model_format)
                for filename, arch, is_inpaint, is_refiner in parsed
                if not (arch is None or (is_inpaint and arch is not Arch.flux) or is_refiner)
            }

        if checkpoints:
            models.checkpoints = parse_model_info(checkpoints, FileFormat.checkpoint)
        else:
            models.checkpoints = {
                filename: CheckpointInfo.deduce_from_filename(filename)
                for filename in nodes["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            }
        if diffusion_models:
            models.checkpoints.update(parse_model_info(diffusion_models, FileFormat.diffusion))

        models.vae = nodes["VAELoader"]["input"]["required"]["vae_name"][0]
        models.loras = nodes["LoraLoader"]["input"]["required"]["lora_name"][0]

        if gguf_node := nodes.get("UnetLoaderGGUF", None):
            for name in gguf_node["input"]["required"]["unet_name"][0]:
                if name not in models.checkpoints:
                    models.checkpoints[name] = CheckpointInfo(name, Arch.flux, FileFormat.diffusion)
        else:
            log.info("GGUF support: node is not installed.")

    async def translate(self, text: str, lang: str):
        try:
            return await self._get(f"api/etn/translate/{lang}/{text}")
        except NetworkError as e:
            log.error(f"Could not translate text: {str(e)}")
            return text

    async def _subscribe_workflows(self):
        try:
            await self._post("api/etn/workflow/subscribe", {"client_id": self._id})
        except Exception as e:
            log.error(f"Couldn't subscribe to shared workflows: {str(e)}")

    async def _unsubscribe_workflows(self):
        try:
            await self._post("api/etn/workflow/unsubscribe", {"client_id": self._id})
        except Exception as e:
            log.error(f"Couldn't unsubscribe from shared workflows: {str(e)}")

    @property
    def missing_resources(self):
        return MissingResources(self._supported_archs)

    @property
    def features(self):
        return self._features

    @property
    def performance_settings(self):
        return PerformanceSettings(
            batch_size=settings.batch_size,
            resolution_multiplier=settings.resolution_multiplier,
            max_pixel_count=settings.max_pixel_count,
            tiled_vae=settings.tiled_vae,
            dynamic_caching=settings.dynamic_caching and self.features.wave_speed,
        )

    async def upload_loras(self, work: WorkflowInput, local_job_id: str):
        for file in loras_to_upload(work, self.models):
            try:
                assert file.path is not None
                url = f"{self.url}/api/etn/upload/loras/{file.id}"
                log.info(f"Uploading lora model {file.id} to {url}")
                data = file.path.read_bytes()
                async for sent, total in self._requests.upload(url, data):
                    progress = sent / max(sent, total)
                    await self._report(ClientEvent.upload, local_job_id, progress)

                await self.refresh()
            except Exception as e:
                raise Exception(_("Error during upload of LoRA model") + f" {file.path}: {str(e)}")

    def _get_active_job(self, remote_id: str) -> Optional[JobInfo]:
        if self._active and self._active.remote_id == remote_id:
            return self._active
        elif self._active:
            log.warning(f"Received message for job {remote_id}, but job {self._active} is active")
        if len(self._jobs) == 0:
            log.warning(f"Received unknown job {remote_id}")
            return None
        active = next((j for j in self._jobs if j.remote_id == remote_id), None)
        if active is not None:
            return active
        return None

    async def _start_job(self, remote_id: str):
        if self._active is not None:
            log.warning(f"Started job {remote_id}, but {self._active} was never finished")
        if len(self._jobs) == 0:
            log.warning(f"Received unknown job {remote_id}")
            return None

        if await self._jobs[0].get_remote_id() == remote_id:
            return self._jobs.popleft()

        log.warning(f"Started job {remote_id}, but {self._jobs[0]} was expected")
        for job in self._jobs:
            if await job.get_remote_id() == remote_id:
                self._jobs.remove(job)
                return job
        return None

    def _clear_job(self, job_remote_id: str | asyncio.Future | None):
        if self._active is not None and self._active.remote_id == job_remote_id:
            result = self._active.local_id
            self._active = None
            return result
        return None

    def _check_workload(self, sdver: Arch) -> list[ResourceId]:
        models = self.models
        missing: list[ResourceId] = []
        for id in resources.required_resource_ids:
            if id.arch is not Arch.all and id.arch is not sdver:
                continue
            if models.find(id) is None:
                missing.append(id)
        has_checkpoint = any(cp.arch is sdver for cp in models.checkpoints.values())
        if not has_checkpoint and sdver not in [Arch.illu, Arch.illu_v]:
            missing.append(ResourceId(ResourceKind.checkpoint, sdver, "Diffusion model checkpoint"))
        if len(missing) > 0:
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


def _check_for_missing_nodes(nodes: dict):
    def missing(node: str, package: CustomNode):
        if node not in nodes:
            log.error(f"Missing required node {node} from package {package.name} ({package.url})")
            return True
        return False

    return [
        package
        for package in resources.required_custom_nodes
        if any(missing(node, package) for node in package.nodes)
    ]


def _find_model(
    model_list: Sequence[str],
    kind: ResourceKind,
    sdver: Arch,
    identifier: ControlMode | UpscalerName | str,
):
    search_paths = resources.search_path(kind, sdver, identifier)
    if search_paths is None:
        return None

    def sanitize(p):
        return p.replace("\\", "/").lower()

    def match(filename: str, pattern: str):
        filename = sanitize(filename)
        pattern = pattern.lower()
        return all(p in filename for p in pattern.split("*"))

    matches = (m for p in search_paths for m in model_list if match(m, p))
    # if there are multiple matches, prefer the one with "krita" in the path
    prio = sorted(matches, key=lambda m: 0 if "krita" in m else len(m))
    found = next(iter(prio), None)
    model_id = identifier.name if isinstance(identifier, Enum) else identifier
    model_name = f"{kind.value} {model_id}"

    if found is None and resources.is_required(kind, sdver, identifier):
        log.warning(f"Missing {model_name} for {sdver.value}")
        log.info(f"-> No model matches search paths: {', '.join(p.lower() for p in search_paths)}")
        log.info(f"-> Available models: {', '.join(sanitize(m) for m in model_list)}")
    elif found is None:
        log.info(
            f"Optional {model_name} for {sdver.value} not found (search path:"
            f" {', '.join(search_paths)})"
        )
    else:
        log.info(f"Found {model_name} for {sdver.value}: {found}")
    return found


def find_model(model_list: Sequence[str], id: ResourceId):
    return _find_model(model_list, id.kind, id.arch, id.identifier)


def _find_text_encoder_models(model_list: Sequence[str]):
    kind = ResourceKind.text_encoder
    return {
        resource_id(kind, Arch.all, te): _find_model(model_list, kind, Arch.all, te)
        for te in ["clip_l", "clip_g", "t5"]
    }


def _find_control_models(model_list: Sequence[str]):
    kind = ResourceKind.controlnet
    return {
        resource_id(kind, ver, mode): _find_model(model_list, kind, ver, mode)
        for mode, ver in product(ControlMode, Arch.list())
        if mode.is_control_net
    }


def _find_ip_adapters(model_list: Sequence[str]):
    kind = ResourceKind.ip_adapter
    return {
        resource_id(kind, ver, mode): _find_model(model_list, kind, ver, mode)
        for mode, ver in product(ControlMode, Arch.list())
        if mode.is_ip_adapter
    }


def _find_clip_vision_model(model_list: Sequence[str]):
    clip_vision_sd = ResourceId(ResourceKind.clip_vision, Arch.all, "ip_adapter")
    model = find_model(model_list, clip_vision_sd)
    clip_vision_flux = ResourceId(ResourceKind.clip_vision, Arch.flux, "redux")
    clip_vision_illu = ResourceId(ResourceKind.clip_vision, Arch.illu, "ip_adapter")
    return {
        clip_vision_sd.string: model,
        clip_vision_flux.string: find_model(model_list, clip_vision_flux),
        clip_vision_illu.string: find_model(model_list, clip_vision_illu),
    }


def _find_style_models(model_list: Sequence[str]):
    redux_flux = ResourceId(ResourceKind.ip_adapter, Arch.flux, ControlMode.reference)
    return {redux_flux.string: find_model(model_list, redux_flux)}


def _find_upscalers(model_list: Sequence[str]):
    kind = ResourceKind.upscaler
    models = {
        resource_id(kind, Arch.all, name): _find_model(model_list, kind, Arch.all, name)
        for name in UpscalerName
    }
    default_id = resource_id(kind, Arch.all, UpscalerName.default)
    if models[default_id] is None and len(model_list) > 0:
        models[default_id] = models[resource_id(kind, Arch.all, UpscalerName.fast_4x)]
    return models


def _find_loras(model_list: Sequence[str]):
    kind = ResourceKind.lora
    common_loras = list(product(["hyper", "lcm", "face"], [Arch.sd15, Arch.sdxl]))
    sdxl_loras = [("lightning", Arch.sdxl)]
    flux_loras = [(ControlMode.depth, Arch.flux), (ControlMode.canny_edge, Arch.flux)]
    return {
        resource_id(kind, arch, name): _find_model(model_list, kind, arch, name)
        for name, arch in chain(common_loras, sdxl_loras, flux_loras)
    }


def _find_vae_models(model_list: Sequence[str]):
    kind = ResourceKind.vae
    return {
        resource_id(kind, ver, "default"): _find_model(model_list, kind, ver, "default")
        for ver in Arch.list()
    }


def _find_inpaint_models(model_list: Sequence[str]):
    kind = ResourceKind.inpaint
    ids: list[tuple[Arch, str]] = [
        (Arch.all, "default"),
        (Arch.sdxl, "fooocus_head"),
        (Arch.sdxl, "fooocus_patch"),
    ]
    return {
        resource_id(kind, ver, name): _find_model(model_list, kind, ver, name) for ver, name in ids
    }


def _ensure_supported_style(client: Client):
    styles = filter_supported_styles(Styles.list(), client)
    if len(styles) == 0:
        supported_checkpoints = (
            cp.filename
            for cp in client.models.checkpoints.values()
            if client.supports_arch(cp.arch)
        )
        checkpoint = next(iter(supported_checkpoints), None)
        if checkpoint is None:
            log.warning("No checkpoints found for any of the supported workloads!")
            if len(client.models.checkpoints) == 0:
                raise Exception(_("No diffusion model checkpoints found"))
            return
        log.info(f"No supported styles found, creating default style with checkpoint {checkpoint}")
        default = next((s for s in Styles.list() if s.filename == "default.json"), None)
        if default:
            default.checkpoints = [checkpoint]
            default.save()
        else:
            Styles.list().create("default", checkpoint)


async def _list_languages(client: ComfyClient) -> list[TranslationPackage]:
    try:
        result = await client._get("api/etn/languages")
        return TranslationPackage.from_list(result)
    except NetworkError as e:
        log.error(f"Could not list available languages for translation: {str(e)}")
        return []


def _extract_message_png_image(data: memoryview):
    s = struct.calcsize(">II")
    if len(data) > s:
        event, format = struct.unpack_from(">II", data)
        # ComfyUI server.py: BinaryEventTypes.PREVIEW_IMAGE=1
        if event == 1 and format == 2:  # format: JPEG=1, PNG=2
            return Image.from_bytes(data[s:])
    return None


def _extract_pose_json(msg: dict):
    try:
        output = msg["data"]["output"]
        if output is not None and "openpose_json" in output:
            return json.loads(output["openpose_json"][0])
    except Exception as e:
        log.warning(f"Error processing message, error={str(e)}, msg={msg}")
    return None


def _extract_text_output(job_id: str, msg: dict):
    try:
        output = msg["data"]["output"]
        if output is not None and "text" in output:
            key = msg["data"].get("node")
            payload = output["text"]
            name, text, mime = (None, None, "text/plain")
            if isinstance(payload, list) and len(payload) >= 1:
                payload = payload[0]
            if isinstance(payload, dict):
                text = payload.get("text")
                name = payload.get("name")
                mime = payload.get("content-type", mime)
            elif isinstance(payload, str):
                text = payload
                name = f"Node {key}"
            if text is not None and name is not None:
                result = TextOutput(key, name, text, mime)
                return ClientMessage(ClientEvent.output, job_id, result=result)
    except Exception as e:
        log.warning(f"Error processing message, error={str(e)}, msg={msg}")
    return None
