import asyncio
import json
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
    _current_remote_id: str | None = None

    @staticmethod
    async def connect(url: str, dev_mode=False):
        client = CloudClient(url)
        if not dev_mode:
            health = await client._get("health")
            log.info(f"Connected to {url}, health: {health}")
        return client

    def __init__(self, url: str):
        self.url = url
        self.models = ClientModels()
        self.device_info = DeviceInfo("Cloud", "Remote GPU", 48 * 1024)
        self._queue = asyncio.Queue()

    async def _get(self, op: str):
        return await self._requests.get(f"{self.url}/{op}")

    async def _post(self, op: str, data: dict):
        return await self._requests.post(f"{self.url}/{op}", data)

    async def enqueue(self, work: WorkflowInput, front: bool = False):
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
        data = {
            "input": {"workflow": inputs},
            "policy": _default_policy,
        }
        response: dict = await self._post("run", data)
        remote_id = self._current_remote_id = response["id"]

        while response["status"] == "IN_QUEUE" or response["status"] == "IN_PROGRESS":
            response = await self._post(f"status/{remote_id}", {})
            await asyncio.sleep(_poll_interval)
        if response["status"] == "COMPLETED":
            output = response["output"]
            results = ImageCollection(Image.from_base64(img_b64) for img_b64 in output["images"])
            yield ClientMessage(ClientEvent.finished, job.id, 1, results)
        elif response["status"] == "FAILED":
            err_msg, err_trace = _extract_error(response, remote_id)
            log.error(f"Job {job.id} failed: {err_msg}\n{err_trace}")
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
