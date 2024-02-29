import asyncio
from datetime import datetime
import json
import os
import platform
import uuid
from dataclasses import dataclass

from .api import WorkflowInput
from .client import Client, ClientEvent, ClientMessage, ClientModels, DeviceInfo, CheckpointInfo
from .image import Image, ImageCollection
from .network import RequestManager
from .resources import SDVersion
from .util import client_logger as log


@dataclass
class JobInfo:
    id: str
    work: WorkflowInput


class CloudClient(Client):
    _requests = RequestManager()
    _queue: asyncio.Queue[JobInfo]
    _token: str = ""
    _current_remote_id: str | None = None

    default_url = os.getenv("CLOUD_ENDPOINT", "http://localhost:3000")

    @staticmethod
    async def connect(url: str, access_token: str):
        if not access_token:
            raise ValueError("Authorization missing for cloud endpoint")
        client = CloudClient(url)
        await client.authenticate(access_token)
        return client

    def __init__(self, url: str):
        self.url = url
        self.models = _models
        self.device_info = DeviceInfo("Cloud", "Remote GPU", 24)
        self._queue = asyncio.Queue()

    async def _get(self, op: str):
        return await self._requests.get(f"{self.url}/api/{op}", bearer=self._token)

    async def _post(self, op: str, data: dict):
        return await self._requests.post(f"{self.url}/api/{op}", data, bearer=self._token)

    async def sign_in(self):
        client_id = str(uuid.uuid4())
        info = f"Generative AI for Krita [Device: {platform.node()}]"
        log.info(f"Sending authorization request for {info} to {self.url}")
        init = await self._post("auth/initiate", dict(client_id=client_id, client_info=info))

        log.info(f"Waiting for completion of authorization at {self.url}{init['url']}")
        yield f"{self.url}{init['url']}"

        auth_confirm = await self._post("auth/confirm", dict(client_id=client_id))
        time = datetime.now()
        while auth_confirm["status"] == "not-found":
            if (datetime.now() - time).seconds > 300:
                raise TimeoutError("Sign-in attempt timed out after 5 minutes")
            await asyncio.sleep(2)
            auth_confirm = await self._post("auth/confirm", dict(client_id=client_id))

        if auth_confirm["status"] == "authorized":
            self._token = auth_confirm["token"]
            log.info(f"Authorization successful")
            yield self._token
        else:
            error = auth_confirm.get("status", "unexpected response")
            raise RuntimeError(f"Authorization could not be confirmed: {error}")

    async def authenticate(self, token: str):
        if not token:
            raise ValueError("Authorization missing for cloud endpoint")
        self._token = token
        user = await self._get("user")
        log.info(f"Connected to {self.url}, user: {user}")

    async def enqueue(self, work: WorkflowInput, front: bool = False):
        work.batch_count = min(work.batch_count, 2)  # TODO: bigger payload
        job = JobInfo(str(uuid.uuid4()), work)
        await self._queue.put(job)
        return job.id

    async def listen(self):
        while True:
            try:
                job = await self._queue.get()
                async for msg in self._process_job(job):
                    yield msg
            except Exception as e:
                log.exception("Unhandled exception in while processing job")
                yield ClientMessage(ClientEvent.error, "", error=str(e))
            except asyncio.CancelledError:
                break

    async def _process_job(self, job: JobInfo):
        inputs = job.work.to_dict()
        data = {"input": {"workflow": inputs}}
        response: dict = await self._post("generate", data)
        remote_id = self._current_remote_id = response["id"]
        yield ClientMessage(ClientEvent.progress, job.id, 0)

        while response["status"] == "IN_QUEUE" or response["status"] == "IN_PROGRESS":
            response = await self._post(f"status/{remote_id}", {})
            log.info(f"Job [id={job.id}, rp={remote_id}] status: {str(response)[:100]}")
            if response["status"] == "IN_PROGRESS":
                if output := response.get("output", None):
                    if progress := output.get("progress", None):
                        yield ClientMessage(ClientEvent.progress, job.id, progress)
            await asyncio.sleep(_poll_interval)
        if response["status"] == "COMPLETED":
            output = response["output"]
            results = ImageCollection(Image.from_base64(img_b64) for img_b64 in output["images"])
            yield ClientMessage(ClientEvent.finished, job.id, 1, results)
        elif response["status"] == "FAILED":
            err_msg, err_trace = _extract_error(response, remote_id)
            log.error(f"Job [id={job.id}, rp={remote_id}] failed\n{err_msg}\n{err_trace}")
            yield ClientMessage(ClientEvent.error, job.id, error=err_msg)
        elif response["status"] == "CANCELLED":
            yield ClientMessage(ClientEvent.interrupted, job.id)
        elif response["status"] == "TIMED_OUT":
            yield ClientMessage(ClientEvent.error, job.id, error="job timed out")
        else:
            log.warning(f"Got unknown job status {response['status']}")

        self._current_remote_id = None

    async def interrupt(self):
        if self._current_remote_id:
            await self._post(f"cancel/{self._current_remote_id}", {})

    async def clear_queue(self):
        self._queue = asyncio.Queue()


def _extract_error(response: dict, job_id: str | None):
    error = response.get("error", f'"Job {job_id} failed (unknown error)"')
    try:
        error_args = json.loads(error)
        err_msg = error_args.get("error_message", error_args)
        err_trace = error_args.get("error_traceback", "No traceback")
    except Exception:
        err_msg = str(error)
        err_trace = "No traceback"
    return err_msg, err_trace


_poll_interval = 0.1  # seconds
_default_policy = {
    "executionTimeout": 2 * 60 * 1000,  # 2 minutes
    "ttl": 10 * 60 * 1000,  # 10 minutes
}

_models = ClientModels()
_models.checkpoints = {
    "dreamshaper_8.safetensors": CheckpointInfo("dreamshaper_8.safetensors", SDVersion.sd15),
    "realisticVisionV51_v51VAE.safetensors": CheckpointInfo(
        "realisticVisionV51_v51VAE.safetensors", SDVersion.sd15
    ),
}
_models.vae = []
_models.loras = []
_models.upscalers = [
    "4x_NMKD-Superscale-SP_178000_G.pth",
    "HAT_SRx4_ImageNet-pretrain.pth",
    "OmniSR_X2_DIV2K.safetensors",
    "OmniSR_X3_DIV2K.safetensors",
    "OmniSR_X4_DIV2K.safetensors",
]
# fmt: off
# TODO: retrieve this from server? or at least share with cloud_worker.py
from ai_diffusion.resources import resource_id, ResourceKind, ControlMode, UpscalerName
_models.resources = {
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.inpaint): "control_v11p_sd15_inpaint_fp16.safetensors",
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.scribble): "control_lora_rank128_v11p_sd15_scribble_fp16.safetensors",
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.scribble): None,
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.line_art): "control_v11p_sd15_lineart_fp16.safetensors",
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.line_art): None,
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.soft_edge): "control_v11p_sd15_softedge_fp16.safetensors",
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.canny_edge): "control_v11p_sd15_canny_fp16.safetensors",
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.canny_edge): None,
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.depth): "control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.depth): None,
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.normal): "control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors",
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.pose): "control_lora_rank128_v11p_sd15_openpose_fp16.safetensors",
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.pose): None,
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.segmentation): None,
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.blur):"control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors",
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.stencil): None,
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.hands): None,
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.hands): None,
    resource_id(ResourceKind.ip_adapter, SDVersion.sd15, ControlMode.reference):"ip-adapter_sd15.safetensors",
    resource_id(ResourceKind.ip_adapter, SDVersion.sdxl, ControlMode.reference): None,
    resource_id(ResourceKind.ip_adapter, SDVersion.sd15, ControlMode.face): None,
    resource_id(ResourceKind.ip_adapter, SDVersion.sdxl, ControlMode.face): None,
    resource_id(ResourceKind.clip_vision, SDVersion.all, "ip_adapter"):"sd1.5/model.safetensors",
    resource_id(ResourceKind.lora, SDVersion.sd15, "lcm"): "lcm-lora-sdv1-5.safetensors",
    resource_id(ResourceKind.lora, SDVersion.sdxl, "lcm"): None,
    resource_id(ResourceKind.lora, SDVersion.sd15, ControlMode.face): None,
    resource_id(ResourceKind.lora, SDVersion.sdxl, ControlMode.face): None,
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.default): UpscalerName.default.value,
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_2x): UpscalerName.fast_2x.value,
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_3x): UpscalerName.fast_3x.value,
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_4x): UpscalerName.fast_4x.value,
    resource_id(ResourceKind.inpaint, SDVersion.sdxl, "fooocus_head"): None,
    resource_id(ResourceKind.inpaint, SDVersion.sdxl, "fooocus_patch"): None,
    resource_id(ResourceKind.inpaint, SDVersion.all, "default"): "MAT_Places512_G_fp16",
}
# fmt: on
