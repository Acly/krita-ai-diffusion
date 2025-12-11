from __future__ import annotations
import asyncio
import json
import struct
import uuid
from dataclasses import dataclass
from enum import Enum
from collections import deque
from itertools import chain, product
from time import time
from typing import Any, Iterable, Optional, Sequence

from .api import WorkflowInput
from .client import Client, CheckpointInfo, ClientMessage, ClientEvent, DeviceInfo, ClientModels
from .client import SharedWorkflow, TranslationPackage, ClientFeatures, TextOutput, ResizeCommand
from .client import Quantization, MissingResources, filter_supported_styles, loras_to_upload
from .comfy_workflow import ComfyObjectInfo
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
from . import platform_tools, resources, util

if platform_tools.is_macos:
    import os

    if "SSL_CERT_FILE" not in os.environ:
        os.environ["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"


@dataclass
class JobInfo:
    id: str
    work: WorkflowInput
    node_count: int = 0
    sample_count: int = 0

    def __str__(self):
        return f"Job[id={self.id}]"

    @staticmethod
    def create(work: WorkflowInput):
        return JobInfo(str(uuid.uuid4()), work)


class ClientJobQueue:
    """Jobs that have been enqueued on client-side but not yet sent to server.
    Unbounded single-producer/single-consumer queue.
    Always consumed front to back, but jobs can be prioritized when added."""

    def __init__(self):
        self._jobs: deque[JobInfo] = deque()
        self._event = asyncio.Event()

    def put(self, job: JobInfo, front: bool = False):
        if front:
            self._jobs.appendleft(job)
        else:
            self._jobs.append(job)
        self._event.set()

    def _get(self):
        job = self._jobs.popleft()
        if not self._jobs:
            self._event.clear()
        return job

    async def get(self):
        while not self._jobs:
            await self._event.wait()
        return self._get()

    def remove_by_id(self, job_ids: Iterable[str]):
        self._jobs = deque(job for job in self._jobs if job.id not in job_ids)
        if len(self._jobs) == 0:
            self._event.clear()

    def __len__(self):
        return len(self._jobs)


class QueuedJob:
    """A slot for a single job that has been sent to the server but hasn't been started."""

    def __init__(self):
        self._job: JobInfo | None = None
        self._event = asyncio.Event()

    async def wait(self):
        if self._job is not None:
            await self._event.wait()

    def set(self, job: JobInfo):
        assert self._job is None
        self._job = job
        self._event.clear()

    def clear(self):
        self._job = None
        self._event.set()

    def peek(self):
        return self._job

    def get(self):
        job = self._job
        self._job = None
        self._event.set()
        return job

    def __len__(self):
        return 1 if self._job is not None else 0


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


class ComfyClient(Client):
    """HTTP/WebSocket client which sends requests to and listens to messages from a ComfyUI server."""

    default_url = "http://127.0.0.1:8188"

    def __init__(self, url):
        self.url = url
        self.models = ClientModels()
        self._requests = RequestManager()
        self._id = settings.comfyui_client_id
        self._active_job: Optional[JobInfo] = None
        self._waiting_job = QueuedJob()
        self._features: ClientFeatures = ClientFeatures()
        self._supported_archs: dict[Arch, list[ResourceId]] = {}
        self._messages: asyncio.Queue[ClientMessage] = asyncio.Queue()
        self._queue = ClientJobQueue()
        self._is_connected = False

        self._requests.add_header("ngrok-skip-browser-warning", "69420")
        self._requests.add_header("skip_zrok_interstitial", "69420")
        if settings.server_authorization:
            self._requests.set_auth(settings.server_authorization)

    @staticmethod
    async def connect(url=default_url, access_token=""):
        client = ComfyClient(parse_url(url))
        log.info(f"Connecting to {client.url}")

        # Retrieve system info
        client.device_info = DeviceInfo.parse(await client._get("system_stats"))

        # Try to establish websockets connection
        wsurl = websocket_url(client.url)
        wsargs = websocket_args(access_token)
        try:
            async with websockets.connect(f"{wsurl}/ws?clientId={client._id}", **wsargs):
                pass
        except Exception as e:
            msg = _("Could not establish websocket connection at") + f" {wsurl}: {str(e)}"
            raise Exception(msg)

        # Check custom nodes
        log.info("Checking for required custom nodes...")
        nodes = ComfyObjectInfo(await client._get("object_info"))
        missing = _check_for_missing_nodes(nodes)
        if len(missing) > 0 and settings.check_server_resources:
            raise MissingResources(missing)

        client._features = ClientFeatures(
            ip_adapter=True,
            translation=True,
            languages=await _list_languages(client),
            gguf="UnetLoaderGGUF" in nodes,
        )

        # Check for required and optional model resources
        models = client.models
        models.node_inputs = nodes
        available_resources = client.models.resources = {}

        clip_models = nodes.options("DualCLIPLoader", "clip_name1")
        clip_models += nodes.options("DualCLIPLoaderGGUF", "clip_name1")
        available_resources.update(_find_text_encoder_models(clip_models))

        vae_models = nodes.options("VAELoader", "vae_name")
        available_resources.update(_find_vae_models(vae_models))

        control_models = nodes.options("ControlNetLoader", "control_net_name")
        available_resources.update(_find_control_models(control_models))

        clip_vision_models = nodes.options("CLIPVisionLoader", "clip_name")
        available_resources.update(_find_clip_vision_model(clip_vision_models))

        ip_adapter_models = nodes.options("IPAdapterModelLoader", "ipadapter_file")
        available_resources.update(_find_ip_adapters(ip_adapter_models))

        model_patches = nodes.options("ModelPatchLoader", "name")
        available_resources.update(_find_model_patches(model_patches))

        style_models = nodes.options("StyleModelLoader", "style_model_name")
        available_resources.update(_find_style_models(style_models))

        models.upscalers = nodes.options("UpscaleModelLoader", "model_name")
        available_resources.update(_find_upscalers(models.upscalers))

        inpaint_models = nodes.options("INPAINT_LoadInpaintModel", "model_name")
        available_resources.update(_find_inpaint_models(inpaint_models))

        loras = nodes.options("LoraLoader", "lora_name")
        available_resources.update(_find_loras(loras))

        # Workarounds for DirectML
        if client.device_info.type == "privateuseone":
            # OmniSR causes a crash
            for n in [2, 3, 4]:
                id = resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_x(n))
                available_resources[id] = models.default_upscaler

        return client

    async def discover_models(self, refresh: bool):
        if refresh:
            nodes = ComfyObjectInfo(await self._get("object_info"))
        else:
            nodes = self.models.node_inputs

        checkpoints: dict[str, dict] = {}
        diffusion_models: dict[str, dict] = {}
        async for status, result in self.try_inspect("checkpoints"):
            yield status
            checkpoints.update(result)
        async for status, result in self.try_inspect("diffusion_models"):
            yield status
            diffusion_models.update(result)
        async for status, result in self.try_inspect("unet_gguf"):
            yield status
            diffusion_models.update(result)
        self._refresh_models(nodes, checkpoints, diffusion_models)

        # Check supported base models and make sure there is at least one
        self._supported_archs = {ver: self._check_workload(ver) for ver in Arch.list()}
        supported_workloads = [
            arch for arch, miss in self._supported_archs.items() if len(miss) == 0
        ]
        log.info("Supported workloads: " + ", ".join(arch.value for arch in supported_workloads))
        if not refresh and len(supported_workloads) == 0 and settings.check_server_resources:
            raise MissingResources(self._supported_archs)

        _ensure_supported_style(self)

    async def refresh(self):
        async for __ in self.discover_models(refresh=True):
            pass

    async def _get(self, op: str, timeout: float | None = 60):
        return await self._requests.get(f"{self.url}/{op}", timeout=timeout)

    async def _post(self, op: str, data: dict):
        return await self._requests.post(f"{self.url}/{op}", data)

    async def _put(self, op: str, data: bytes):
        return await self._requests.put(f"{self.url}/{op}", data)

    async def enqueue(self, work: WorkflowInput, front: bool = False):
        job = JobInfo.create(work)
        self._queue.put(job, front=front)
        return job.id

    async def _report(self, event: ClientEvent, job_id: str, value: float = 0, **kwargs):
        await self._messages.put(ClientMessage(event, job_id, value, **kwargs))

    async def _run(self):
        assert self._is_connected
        try:
            while self._is_connected:
                await self._waiting_job.wait()  # first wait for slot on server
                job = await self._queue.get()  # then get highest priority job from queue
                try:
                    await self._run_job(job)
                except Exception as e:
                    log.exception(f"Unhandled exception while processing {job}")
                    await self._report(ClientEvent.error, job.id, error=str(e))
        except asyncio.CancelledError:
            pass

    async def _run_job(self, job: JobInfo):
        workflow = create_workflow(job.work, self.models)
        if settings.debug_dump_workflow:
            workflow.embed_images().dump(util.log_dir)

        await self.upload_images(workflow.image_data)
        await self.upload_loras(job.work, job.id)

        job.node_count = workflow.node_count
        job.sample_count = workflow.sample_count
        data = {
            "prompt": workflow.root,
            "client_id": self._id,
            "prompt_id": job.id,
        }
        self._waiting_job.set(job)
        try:
            result = await self._post("prompt", data)
            if result["prompt_id"] != job.id:
                log.error(f"Prompt ID mismatch: {result['prompt_id']} != {job.id}")
                raise ValueError("Prompt ID mismatch - Please update ComfyUI to 0.3.45 or later!")
        except Exception as e:
            self._waiting_job.clear()
            raise e

    async def _listen(self):
        url = websocket_url(self.url)
        args = websocket_args(settings.server_authorization)
        async for websocket in websockets.connect(f"{url}/ws?clientId={self._id}", **args):
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
                self._active_job = None
                self._waiting_job.clear()
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
                    self._active_job = await self._start_job(id)
                    if self._active_job is not None:
                        progress = Progress(self._active_job)
                        images = ImageCollection()
                        result = None

                if msg["type"] == "execution_interrupted":
                    if job := await self._get_active_job(msg["data"]["prompt_id"]):
                        self._clear_job(job.id)
                        await self._report(ClientEvent.interrupted, job.id)

                if msg["type"] == "executing" and msg["data"]["node"] is None:
                    job_id = msg["data"]["prompt_id"]
                    if self._clear_job(job_id):
                        if len(images) == 0:
                            # It may happen if the entire execution is cached and no images are sent.
                            images = last_images
                        if len(images) == 0:
                            # Still no images. Potential scenario: execution cached, but previous
                            # generation happened before the client was connected.
                            err = "No new images were generated because the inputs did not change."
                            await self._report(ClientEvent.error, job_id, error=err)
                        else:
                            last_images = images
                            await self._report(
                                ClientEvent.finished, job_id, 1, images=images, result=result
                            )

                elif msg["type"] in ("execution_cached", "executing", "progress"):
                    if self._active_job is not None and progress is not None:
                        progress.handle(msg)
                        await self._report(
                            ClientEvent.progress, self._active_job.id, progress.value
                        )
                    else:
                        log.warning(f"Received message {msg} but there is no active job")

                if msg["type"] == "executed":
                    if job := await self._get_active_job(msg["data"]["prompt_id"]):
                        images.append(await self._transfer_result_images(msg))
                        text_output = _extract_text_output(job.id, msg)
                        if text_output is not None:
                            await self._messages.put(text_output)
                        resize_cmd = _extract_resize_output(job.id, msg)
                        if resize_cmd is not None:
                            await self._messages.put(resize_cmd)
                        pose_json = _extract_pose_json(msg)
                        if pose_json is not None:
                            result = pose_json

                if msg["type"] == "execution_error":
                    if job := await self._get_active_job(msg["data"]["prompt_id"]):
                        error = msg["data"].get("exception_message", "execution_error")
                        traceback = msg["data"].get("traceback", "no traceback")
                        log.error(f"Job {job} failed: {error}\n{traceback}")
                        self._clear_job(job.id)
                        await self._report(ClientEvent.error, job.id, error=error)

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

    async def cancel(self, job_ids: Iterable[str]):
        # Make sure changes to all queues are processed before suspending this
        # function, otherwise it may interfere with subsequent calls to enqueue.
        tasks = [self._post("queue", {"delete": list(job_ids)})]
        if job := self._waiting_job.peek():
            if job.id in job_ids:
                self._waiting_job.clear()
        self._queue.remove_by_id(job_ids)
        for id in job_ids:
            tasks.append(self._report(ClientEvent.interrupted, id))
        await asyncio.gather(*tasks)

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

    async def try_inspect(self, folder_name: str):
        if "gguf" in folder_name and not self.features.gguf:
            return
        try:
            log.info(f"Inspecting models at {self.url}/api/etn/model_info/{folder_name}")
            start, timeout = time(), 600
            offset, total = 0, 100
            while offset < total and (time() - start) < timeout:
                r = await self._get(f"api/etn/model_info/{folder_name}?offset={offset}&limit=8")
                if "_meta" not in r:  # server doesn't support pagination
                    yield (Client.DiscoverStatus(folder_name, len(r), len(r)), r)
                    return
                total = r["_meta"]["total"]
                del r["_meta"]
                yield (Client.DiscoverStatus(folder_name, offset + len(r), total), r)
                offset += 8
            if offset < total:
                log.warning(f"Timeout while inspecting models, received {offset}/{total} entries")
        except NetworkError as e:
            log.error(f"Error while inspecting models in {folder_name}: {str(e)}")

    @property
    def queued_count(self):
        return len(self._waiting_job) + len(self._queue)

    @property
    def is_executing(self):
        return self._active_job is not None

    def _refresh_models(
        self, nodes: ComfyObjectInfo, checkpoints: dict | None, diffusion_models: dict | None
    ):
        models = self.models

        def parse_model_info(models: dict, model_format: FileFormat):
            parsed = (
                (
                    filename,
                    Arch.from_string(info["base_model"], info.get("type", "eps"), filename),
                    Quantization.from_string(info.get("quant", "none")),
                    info.get("is_inpaint", False),
                    info.get("is_refiner", False),
                )
                for filename, info in models.items()
            )
            return {
                filename: CheckpointInfo(filename, arch, model_format, quant)
                for filename, arch, quant, is_inpaint, is_refiner in parsed
                if not (arch is None or (is_inpaint and arch is not Arch.flux) or is_refiner)
            }

        if checkpoints:
            models.checkpoints = parse_model_info(checkpoints, FileFormat.checkpoint)
        else:
            models.checkpoints = {
                filename: CheckpointInfo.deduce_from_filename(filename)
                for filename in nodes.options("CheckpointLoaderSimple", "ckpt_name")
            }
        if diffusion_models:
            models.checkpoints.update(parse_model_info(diffusion_models, FileFormat.diffusion))

        models.vae = nodes.options("VAELoader", "vae_name")
        models.loras = nodes.options("LoraLoader", "lora_name")

        if "UnetLoaderGGUF" in nodes:
            for name in nodes.options("UnetLoaderGGUF", "unet_name"):
                if name not in models.checkpoints:
                    models.checkpoints[name] = CheckpointInfo(name, Arch.flux, FileFormat.diffusion)
        else:
            log.info("GGUF support: node is not installed.")

    async def _transfer_result_image(self, id: str):
        try:
            data = await self._requests.download(f"{self.url}/api/etn/image/{id}", timeout=300)
            return Image.from_bytes(data)
        except Exception as e:
            log.error(f"Error transferring result image {self.url}/api/etn/image/{id}: {str(e)}")
            raise e

    async def upload_images(self, image_data: dict[str, bytes]):
        for id, data in image_data.items():
            try:
                await self._put(f"api/etn/image/{id}", data)
            except Exception as e:
                log.error(f"Error uploading image {id}: {str(e)}")
                raise RuntimeError(f"Error uploading input image to ComfyUI: {str(e)}") from e

    async def _transfer_result_images(self, msg: dict) -> list[Image]:
        output = msg["data"]["output"]
        if output is not None and "images" in output:
            transfers = []
            for img in output["images"]:
                source = img.get("source")
                id = img.get("id")
                if source == "http" and id is not None:
                    transfers.append(self._transfer_result_image(id))
            return await asyncio.gather(*transfers)
        return []

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
            dynamic_caching=settings.dynamic_caching,
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

    async def _get_active_job(self, job_id: str):
        if self._active_job and self._active_job.id == job_id:
            return self._active_job

        # We might have dropped some messages, see if the message matches the next job
        log.warning(f"Received message for job {job_id}, but active job is {self._active_job}")
        if next_job := await self._start_job(job_id):
            return next_job
        return None

    async def _start_job(self, job_id: str):
        next_job = self._waiting_job.peek()
        if next_job is None:
            log.warning(f"Received unknown job {job_id}, there are no jobs waiting")
            return None
        if next_job.id != job_id:
            log.warning(f"Received unknown job {job_id}, next job is {next_job.id}")
            return None

        next_job = self._waiting_job.get()
        if self._active_job is not None:
            log.warning(f"Started job {job_id}, but {self._active_job} was never finished")
            await self._report(ClientEvent.interrupted, self._active_job.id)
        self._active_job = next_job
        return next_job

    def _clear_job(self, job_id: str):
        if self._active_job is not None and self._active_job.id == job_id:
            self._active_job = None
            return True
        return False

    def _check_workload(self, sdver: Arch) -> list[ResourceId]:
        models = self.models
        missing: list[ResourceId] = []
        for id in resources.required_resource_ids:
            if id.arch is not Arch.all and id.arch is not sdver:
                continue
            if models.find(id) is None:
                missing.append(id)
        has_checkpoint = any(cp.arch is sdver for cp in models.checkpoints.values())
        if not has_checkpoint and sdver is Arch.illu:  # Illu checkpoints are detected as SDXL
            has_checkpoint = any(cp.arch is Arch.sdxl for cp in models.checkpoints.values())
        if not has_checkpoint:
            missing.append(ResourceId(ResourceKind.checkpoint, sdver, "model"))
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


def websocket_args(auth_token: str):
    args: dict[str, Any] = dict(max_size=2**30, ping_timeout=60)
    if auth_token:
        args["extra_headers"] = {"Authorization": f"Bearer {auth_token}"}
    return args


def _check_for_missing_nodes(nodes: ComfyObjectInfo):
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

    matches: list[tuple[str, int]] = []
    for i, pattern in enumerate(search_paths):
        for filename in model_list:
            name = sanitize(filename)
            pattern = pattern.lower()
            if all(p in name for p in pattern.split("*")):
                # prioritize names with "krita" in the path, then earlier matches
                prio = 0 if "krita" in name else i * 100 + len(name)
                matches.append((filename, prio))

    matches = sorted(matches, key=lambda m: m[1])
    found, _ = next(iter(matches), (None, -1))
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
        for te in ["clip_l", "clip_g", "t5", "qwen", "qwen_3"]
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
    clip_vision_sd15 = ResourceId(ResourceKind.clip_vision, Arch.sd15, "ip_adapter")
    clip_vision_sdxl = ResourceId(ResourceKind.clip_vision, Arch.sdxl, "ip_adapter")
    clip_vision_flux = ResourceId(ResourceKind.clip_vision, Arch.flux, "redux")
    clip_vision_illu = ResourceId(ResourceKind.clip_vision, Arch.illu, "ip_adapter")
    return {
        clip_vision_sd15.string: find_model(model_list, clip_vision_sd15),
        clip_vision_sdxl.string: find_model(model_list, clip_vision_sdxl),
        clip_vision_flux.string: find_model(model_list, clip_vision_flux),
        clip_vision_illu.string: find_model(model_list, clip_vision_illu),
    }


def _find_model_patches(model_list: Sequence[str]):
    res = ResourceId(ResourceKind.model_patch, Arch.zimage, ControlMode.universal)
    return {res.string: _find_model(model_list, res.kind, res.arch, res.identifier)}


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
    flux_loras = [
        ("turbo", Arch.flux),
        (ControlMode.depth, Arch.flux),
        (ControlMode.canny_edge, Arch.flux),
    ]
    flux_k_loras = [("turbo", Arch.flux_k)]
    return {
        resource_id(kind, arch, name): _find_model(model_list, kind, arch, name)
        for name, arch in chain(common_loras, sdxl_loras, flux_loras, flux_k_loras)
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
                # Special case: Krita canvas resize command produced by a tooling node
                if mime == "application/x-krita-command" and isinstance(text, str):
                    try:
                        data = json.loads(text)
                        if data.get("action") == "resize_canvas":
                            width = int(data.get("width", 0))
                            height = int(data.get("height", 0))
                            if width > 0 and height > 0:
                                cmd = ResizeCommand(width, height)
                                return ClientMessage(ClientEvent.output, job_id, result=cmd)
                    except Exception as e:
                        log.warning(f"Failed to process Krita command output: {e}")
            elif isinstance(payload, str):
                text = payload
                name = f"Node {key}"
            if text is not None and name is not None:
                result = TextOutput(key, name, text, mime)
                return ClientMessage(ClientEvent.output, job_id, result=result)
    except Exception as e:
        log.warning(f"Error processing message, error={str(e)}, msg={msg}")
    return None


def _extract_resize_output(job_id: str, msg: dict):
    """Extract a Krita canvas resize toggle encoded directly in the UI output."""
    try:
        output = msg["data"]["output"]
        if output is None:
            return None

        resize = output.get("resize_canvas")
        if isinstance(resize, list):
            active = any(bool(item) for item in resize)
        else:
            active = bool(resize)

        if not active:
            return None

        # Use a lightweight dict result; the Krita client will interpret this
        # as "resize canvas to match image extent" on apply.
        return ClientMessage(ClientEvent.output, job_id, result={"resize_canvas": True})
    except Exception as e:
        log.warning(f"Error processing Krita resize output: {e}, msg={msg}")
        return None
