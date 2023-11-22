import asyncio
import random
from datetime import datetime
from .network import RequestManager
from .util import client_logger as log
from .client import CheckpointInfo, ClientMessage, ClientEvent
from .workflow import Conditioning
from .image import Image, Extent, ImageCollection
from .style import Style
from . import SDVersion, workflow


class Job:
    id: str
    start_time: datetime

    def __init__(self, id: str):
        self.id = id
        self.start_time = datetime.now()


class HordeClient:
    _net: RequestManager
    _jobs: asyncio.Queue[Job]
    _active: Job | None = None

    checkpoints: dict[str, CheckpointInfo]

    @staticmethod
    async def connect(url="https://aihorde.net", apikey="0000000000"):
        client = HordeClient(url, apikey)

        models = await client._get("status/models")
        # {"performance": 568347.3, "queued": 54263808.0, "jobs": 4.0, "eta": 23, "type": "image", "name": "Realistic Vision", "count": 4}
        client.checkpoints = {
            v["name"]: CheckpointInfo(v["name"], SDVersion.sd15)
            for v in models
            if v["type"] == "image"
        }

        return client

    def __init__(self, url: str, apikey: str):
        self._net = RequestManager()
        self._url = url
        self._apikey = apikey
        self._jobs = asyncio.Queue()

    async def _get(self, op: str):
        return await self._net.get(f"{self._url}/api/v2/{op}")

    async def _post(self, op: str, payload: dict):
        headers = {"apikey": self._apikey}
        return await self._net.post(f"{self._url}/api/v2/{op}", payload, headers)

    async def generate(self, style: Style, input_extent: Extent, cond: Conditioning) -> str:
        sampler_params = workflow._sampler_params(style)
        sampler_name = {
            "DDIM": "DDIM",
            "DPM++ 2M": "k_dpmpp_2m",
            "DPM++ 2M Karras": "k_dpmpp_2m",
            "DPM++ 2M SDE": "k_dpmpp_sde",
            "DPM++ 2M SDE Karras": "k_dpmpp_sde",
        }[style.sampler]
        seed = sampler_params.get("seed", random.getrandbits(64))
        payload = {
            "prompt": workflow.merge_prompt(cond.prompt, style.style_prompt),
            "params": {
                "sampler_name": sampler_name,
                "cfg_scale": style.cfg_scale,
                "denoising_strength": 1.0,
                "seed": str(seed),
                "height": input_extent.height,
                "width": input_extent.width,
                "karras": sampler_params["scheduler"] == "karras",
                "steps": sampler_params["steps"],
            },
            "nsfw": False,  # ?
            "models": [style.sd_checkpoint],
        }
        response = await self._post("generate/async", payload)
        log.info(f"Job created: {response}")
        await self._jobs.put(Job(response["id"]))
        return response["id"]

    async def listen(self):
        while True:
            try:
                status = await self._query_status()
                yield status
                if status.event == ClientEvent.progress:
                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                self._active = None
                self._jobs = asyncio.Queue()
                break
            except Exception as e:
                log.exception("Unhandled exception while polling AI Horde")
                yield ClientMessage(ClientEvent.error, error=str(e))

    async def _query_status(self):
        if self._active is None:
            self._active = await self._jobs.get()

        job_id = self._active.id
        status = await self._get(f"generate/check/{job_id}")

        if status["faulted"]:
            log.warning(f"Job {job_id} failed")
            self._active = None
            return ClientMessage(ClientEvent.error, error="Job failed")

        if status["done"]:
            log.info(f"Job {job_id} finished")
            images = await self._retrieve_result(self._active)
            self._active = None
            return ClientMessage(ClientEvent.finished, job_id, images=images)

        if status["waiting"] > 0:
            elapsed = datetime.now() - self._active.start_time
            wait_time = status["wait_time"]
            progress = wait_time / (elapsed.total_seconds() + wait_time)
            return ClientMessage(ClientEvent.progress, job_id, progress)

        return ClientMessage(ClientEvent.progress, job_id, 0.9)  # processing, but progress=?

    async def _retrieve_result(self, job: Job):
        status = await self._get(f"generate/status/{job.id}")
        if not (status["done"] and len(status["generations"]) == 1):
            log.error(f"Job is not actually done or has no results: {status}")
            raise Exception(f"Job returned without result images")
        gen = status["generations"][0]
        img_data = await self._net.get(gen["img"])
        img = Image.from_bytes(img_data, "WEBP")
        return ImageCollection([img])
