import asyncio
import json
import math
import os
import platform
import uuid
from base64 import b64encode
from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from . import __version__ as plugin_version
from .api import WorkflowInput
from .client import (
    Client,
    ClientEvent,
    ClientFeatures,
    ClientJobQueue,
    ClientMessage,
    ClientModels,
    DeviceInfo,
    News,
    ServerError,
    TranslationPackage,
    User,
    loras_to_upload,
)
from .files import File
from .image import ImageCollection, qt_supports_webp
from .localization import translate as _
from .network import NetworkError, RequestManager
from .settings import PerformanceSettings, settings
from .util import clamp, ensure
from .util import client_logger as log


class JobState(Enum):
    send = 1
    generate = 2
    receive = 3
    cancelled = 4
    finalized = 5


@dataclass
class JobInfo:
    local_id: str
    work: WorkflowInput
    remote_id: str | None = None
    worker_id: str | None = None

    state: JobState = JobState.send
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None

    def __str__(self):
        return f"Job[{self.work.kind.name}, local={self.local_id}, remote={self.remote_id}]"

    def next_state(self):
        match self.state:
            case JobState.send:
                return JobState.generate
            case JobState.generate:
                return JobState.receive
            case JobState.receive:
                return JobState.finalized
        return self.state


@dataclass
class JobExecutor:
    queue: ClientJobQueue[JobInfo] = field(default_factory=ClientJobQueue)
    task: asyncio.Task | None = None
    current_job: JobInfo | None = None
    name: str = "Executor"


class CloudClient(Client):
    default_api_url = os.getenv("INTERSTICE_URL", "https://api.interstice.cloud")
    default_web_url = os.getenv("INTERSTICE_WEB_URL", "https://www.interstice.cloud")

    @staticmethod
    async def connect(url: str, access_token: str = ""):
        if not access_token:
            raise ValueError("Authorization missing for cloud endpoint")
        client = CloudClient(url)
        await client.authenticate(access_token)
        return client

    def __init__(self, url: str):
        self.url = url
        self.models = ClientModels()
        self.device_info = DeviceInfo("Cloud", "Remote GPU", 24)
        self._requests = RequestManager()
        self._token: str = ""
        self._user: User | None = None
        self._news: News | None = None
        self._exec_send = JobExecutor(name="send")
        self._exec_generate = JobExecutor(name="generate")
        self._exec_receive = JobExecutor(name="receive")
        self._executors = [self._exec_send, self._exec_generate, self._exec_receive]
        self._messages: asyncio.Queue[ClientMessage] = asyncio.Queue()
        self._features = enumerate_features({})
        self._is_connected = False

    async def _get(self, op: str):
        return await self._requests.get(f"{self.url}/{op}", bearer=self._token)

    async def _post(self, op: str, data: dict):
        return await self._requests.post(f"{self.url}/{op}", data, bearer=self._token)

    async def sign_in(self):
        client_id = str(uuid.uuid4())
        info = f"Generative AI for Krita [Device: {platform.node()}]"
        log.info(f"Sending authorization request for {info} to {self.url}")
        init = await self._post("auth/initiate", {"client_id": client_id, "client_info": info})

        sign_in_url = f"{self.default_web_url}{init['url']}"
        log.info(f"Waiting for completion of authorization at {sign_in_url}")
        yield sign_in_url

        auth_confirm = await self._post("auth/confirm", {"client_id": client_id})
        time = datetime.now(timezone.utc)
        while auth_confirm["status"] == "not-found":
            if (datetime.now(timezone.utc) - time).seconds > 300:
                raise TimeoutError(_("Sign-in attempt timed out after 5 minutes"))
            await asyncio.sleep(2)
            auth_confirm = await self._post("auth/confirm", {"client_id": client_id})

        if auth_confirm["status"] == "authorized":
            self._token = auth_confirm["token"]
            log.info("Authorization successful")
            yield self._token
        else:
            error = auth_confirm.get("status", "unexpected response")
            raise RuntimeError(_("Authorization could not be confirmed: ") + error)

    async def authenticate(self, token: str):
        if not token:
            raise ValueError("Authorization missing for cloud endpoint")
        self._token = token
        try:
            user_data = await self._get(f"user?plugin_version={plugin_version}")
        except NetworkError as e:
            log.error(f"Couldn't authenticate user account: {e.message}")
            self._token = ""
            if e.status == 401:
                e.message = _("The login data is incorrect, please sign in again.")
            raise
        self._user = User(user_data["id"], user_data["name"])
        self._user.images_generated = user_data["images_generated"]
        self._user.credits = user_data["credits"]
        self._features = enumerate_features(user_data)
        if news_text := user_data.get("news"):
            self._news = News.create(news_text)

        model_data = await self._get("plugin/resources")
        self.models = ClientModels.from_dict(model_data)
        log.info(f"Connected to {self.url}, user: {self._user.id}")
        return self._user

    async def enqueue(self, work: WorkflowInput, front: bool = False):
        apply_limits(work, self.features)
        job = JobInfo(str(uuid.uuid4()), work)
        await self._enqueue(job, front)
        return job.local_id

    async def listen(self):
        assert not self._is_connected, "Client is already connected"
        for executor in self._executors:
            executor.task = asyncio.create_task(self._run(executor))
        self._is_connected = True
        yield ClientMessage(ClientEvent.connected)

        try:
            while self._is_connected:
                yield await self._messages.get()
        except asyncio.CancelledError:
            pass
        finally:
            await self.disconnect()

    async def _enqueue(self, job: JobInfo, front: bool = False):
        match job.state:
            case JobState.send:
                self._exec_send.queue.put(job, front)
            case JobState.generate:
                self._exec_generate.queue.put(job, front)
            case JobState.receive:
                self._exec_receive.queue.put(job, front)
            case JobState.cancelled:
                await self._report(ClientEvent.interrupted, job.local_id)
                job.state = JobState.finalized

    async def _process(self, job: JobInfo):
        try:
            match job.state:
                case JobState.send:
                    await self._send(job)
                case JobState.generate:
                    await self._generate(job)
                case JobState.receive:
                    await self._receive(job)
            job.state = job.next_state()
            await self._enqueue(job)

        except NetworkError as e:
            job_id = job.local_id
            if msg := self._process_http_error(e, job_id):
                await self._report(ClientEvent.error, job_id, error=msg)
            else:
                msg = e.message
                log.exception(f"Network error while processing {job}: {msg}")
                if job_id:
                    await self._report(ClientEvent.error, job_id, error=msg)
            job.state = JobState.finalized
        except Exception as e:
            log.exception(f"Unhandled exception while processing {job}")
            await self._report(ClientEvent.error, job.local_id, error=str(e))
            job.state = JobState.finalized

    async def _send(self, job: JobInfo):
        job.input = job.work.to_dict(max_image_size=16 * 1024)

        async for progress in self.send_lora(job.work):
            await self._report(ClientEvent.upload, job.local_id, progress)

        await self.send_images(job.input)

    async def _generate(self, job: JobInfo):
        input = ensure(job.input)
        user = ensure(self.user)
        data = {
            "input": {
                "workflow": input,
                "clientInfo": f"krita-ai-diffusion {plugin_version}",
                "options": {
                    "useWebpCompression": qt_supports_webp(),
                },
            }
        }
        response: dict = await self._post("generate", data)

        job.remote_id = response["id"]
        job.worker_id = response["worker_id"]
        cost = _update_user(user, response.get("user"))
        log.info(f"{job} started, cost was {cost}, {user.credits} tokens remaining")
        await self._report(ClientEvent.progress, job.local_id, 0)

        status = response["status"].lower()
        while status in {"in_queue", "in_progress"}:
            response = await self._post(f"status/{job.remote_id}", {})
            status = response["status"].lower()

            if status == "in_queue":
                await self._report(ClientEvent.queued, job.local_id)

            elif status == "in_progress":
                progress = 0.09
                if output := response.get("output", None):
                    progress = output.get("progress", progress)
                await self._report(ClientEvent.progress, job.local_id, progress)

            if job.state is JobState.cancelled:
                break
            await asyncio.sleep(_poll_interval)

        if status == "completed":
            job.output = response["output"]

        elif status == "failed":
            err_msg, err_trace = _extract_error(response, job.remote_id)
            log.error(f"{job} failed\n{err_msg}\n{err_trace}")
            await self._report(ClientEvent.error, job.local_id, error=err_msg)
            job.state = JobState.finalized

        elif status == "violation":
            err_msg = response.get("error", "content violation")
            log.warning(f"{job} was aborted: {err_msg}")
            await self._report(ClientEvent.error, job.local_id, error=err_msg)
            job.state = JobState.finalized

        elif status == "cancelled" or job.state is JobState.cancelled:
            log.info(f"{job} was cancelled")
            await self._report(ClientEvent.interrupted, job.local_id)
            job.state = JobState.finalized

        elif status == "timed_out":
            log.warning(f"{job} timed out")
            msg = _("Generation took too long and was cancelled (timeout)")
            await self._report(ClientEvent.error, job.local_id, error=msg)
            job.state = JobState.finalized
        else:
            log.warning(f"Got unknown job status {status}")

    async def _receive(self, job: JobInfo):
        output = ensure(job.output)
        images = await self.receive_images(output["images"])
        pose = output.get("pose", None)
        log.info(f"{job} completed, got {len(images)} images{', got pose' if pose else ''}")
        lora_warning = output.get("lora_warning", False)
        if lora_warning:
            log.warning(f"{job} encountered LoRA that could not be applied to the checkpoint")
        error = "incompatible_lora" if lora_warning else None
        await self._report(
            ClientEvent.finished, job.local_id, 1, images=images, result=pose, error=error
        )

    async def _run(self, executor: JobExecutor):
        try:
            while self._is_connected:
                executor.current_job = await executor.queue.get()
                await self._process(executor.current_job)
                executor.current_job = None
        except asyncio.CancelledError:
            pass

    async def disconnect(self):
        if self._is_connected:
            try:
                self._is_connected = False
                tasks = [e.task for e in self._executors if e.task]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks)
            except Exception:
                log.exception("Error while disconnecting from cloud client")

    async def _report(self, event: ClientEvent, job_id: str, value: float = 0, **kwargs):
        await self._messages.put(ClientMessage(event, job_id, value, **kwargs))

    async def interrupt(self):
        if job := self._exec_send.current_job:
            job.state = JobState.cancelled
        if (job := self._exec_generate.current_job) and job.remote_id and job.worker_id:
            response = await self._post(f"cancel/{job.worker_id}/{job.remote_id}", {})
            log.info(f"Requested cancellation of {job}: {response}")
            job.state = JobState.cancelled
        # If a job is in the receive stage, we let it finish normally.

    async def cancel(self, job_ids: Iterable[str]):
        for executor in (self._exec_send, self._exec_generate, self._exec_receive):
            for job in executor.queue:
                if job.local_id in job_ids:
                    job.state = JobState.cancelled

    @property
    def user(self):
        return self._user

    @property
    def news(self):
        return self._news

    @property
    def performance_settings(self):
        return PerformanceSettings(
            batch_size=clamp(settings.batch_size, 4, 8),
            resolution_multiplier=settings.resolution_multiplier,
            max_pixel_count=clamp(settings.max_pixel_count, 1, 8),
            dynamic_caching=False,
        )

    @property
    def features(self):
        return self._features

    async def send_images(self, inputs: dict, max_inline_size=4096):
        if image_data := inputs.get("image_data"):
            blob, offsets = image_data["bytes"], image_data["offsets"]
            if _base64_size(len(blob)) < max_inline_size:
                encoded = b64encode(blob).decode("utf-8")
                inputs["image_data"] = {"base64": encoded, "offsets": offsets}
            else:
                s3_object = await self._upload_image(blob)
                inputs["image_data"] = {"s3_object": s3_object, "offsets": offsets}

    async def _upload_image(self, data: bytes):
        upload_info = await self._post("upload/image", {})
        log.info(f"Uploading image input to temporary transfer {upload_info['url']}")
        await self._requests.put(upload_info["url"], data)
        return upload_info["object"]

    async def send_lora(self, workflow: WorkflowInput):
        for file in loras_to_upload(workflow, self.models):
            async for progress in self._upload_lora(file):
                yield progress

    async def _upload_lora(self, lora: File):
        assert lora.path and lora.hash and lora.size
        upload = await self._post("upload/lora", {"hash": lora.hash, "size": lora.size})
        if upload["status"] == "too-large":
            max_size = int(upload.get("max", 0)) / (1024 * 1024)
            raise ValueError(
                _("LoRA model is too large to upload") + f" (max {max_size} MB) {lora.name}"
            )
        if upload["status"] == "limit-exceeded":
            raise ValueError(_("Can't upload LoRA model, limit exceeded") + f" {lora.name}")
        if upload["status"] == "cached":
            return  # already uploaded
        log.info(
            f"Uploading LoRA model {lora.name} to cloud (hash={lora.hash}, url={upload['url']})"
        )
        try:
            data = lora.path.read_bytes()
            async for sent, total in self._requests.upload(upload["url"], data, sha256=lora.hash):
                yield sent / max(total, 1)
        except NetworkError as e:
            log.error(f"LoRA model upload failed [{e.status}]: {e.message}")
            raise ServerError(
                _("Connection error during upload of LoRA model") + f" {lora.name}"
            ) from e
        except Exception as e:
            raise ServerError(_("Error during upload of LoRA model") + f" {lora.name}") from e

    async def receive_images(self, images: dict):
        offsets = images.get("offsets")
        if not (isinstance(offsets, list) and len(offsets) > 0):
            raise ValueError(f"Could not read result images, invalid offsets: {offsets}")
        if url := images.get("url"):
            log.info(f"Downloading result images from temporary transfer {url}")
            data = await self._requests.download(url)
            return ImageCollection.from_bytes(data, offsets)
        elif b64 := images.get("base64"):
            return ImageCollection.from_base64(b64, offsets)
        else:
            raise ValueError(f"No result images found in server response: {str(images)[:80]}")

    async def compute_cost(self, input: WorkflowInput):
        response = await self._post("admin/cost", input.to_dict())
        return int(response.decode())

    def _process_http_error(self, e: NetworkError, job_id: str):
        if e.status == 402 and e.data and self.user:  # 402 Payment Required
            try:
                data = copy(e.data)
                data["url"] = f"{self.default_web_url}/user"
                self.user.credits = e.data["credits"]
                return ClientMessage(
                    ClientEvent.payment_required, job_id, result=e.data, error=e.message
                )
            except Exception:
                log.warning(f"Could not parse 402 error: {e.data}")
        return None


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


def _update_user(user: User, response: dict | None):
    if response:
        cost = max(0, user.credits - response["credits"])
        user.images_generated = response["images_generated"]
        user.credits = response["credits"]
        return cost
    else:
        log.warning("Did not receive updated user data from server")
        return 0


def enumerate_features(user_data: dict):
    return ClientFeatures(
        ip_adapter=True,
        translation=True,
        languages=[
            TranslationPackage("zh", "Chinese"),
            TranslationPackage("fr", "French"),
            TranslationPackage("de", "German"),
            TranslationPackage("ru", "Russian"),
            TranslationPackage("es", "Spanish"),
        ],
        max_upload_size=user_data.get("max_upload_size", 300 * 1024 * 1024),
        max_control_layers=user_data.get("max_control_layers", 4),
    )


def apply_limits(work: WorkflowInput, features: ClientFeatures):
    if work.models:
        work.models.self_attention_guidance = False
    if work.conditioning:
        work.conditioning.control = work.conditioning.control[: features.max_control_layers]
        for region in work.conditioning.regions:
            region.control = region.control[: features.max_control_layers]
    if work.sampling:
        work.sampling.total_steps = min(work.sampling.total_steps, 1000)


def _base64_size(size: int):
    return math.ceil(size / 3) * 4


_poll_interval = 0.5  # seconds
