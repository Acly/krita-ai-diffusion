import asyncio
import json
import math
import os
import platform
import uuid
from base64 import b64encode
from copy import copy
from datetime import datetime
from dataclasses import dataclass
from itertools import chain

from .api import WorkflowInput
from .client import Client, ClientEvent, ClientMessage, ClientModels, DeviceInfo, CheckpointInfo
from .client import ClientFeatures, TranslationPackage, User, loras_to_upload
from .image import ImageCollection, qt_supports_webp
from .network import RequestManager, NetworkError
from .files import File
from .resources import Arch, ResourceKind, ControlMode, UpscalerName, resource_id
from .settings import PerformanceSettings, settings
from .localization import translate as _
from .util import clamp, ensure, client_logger as log
from . import resources, __version__ as plugin_version


@dataclass
class JobInfo:
    local_id: str
    work: WorkflowInput
    remote_id: str | None = None
    worker_id: str | None = None

    def __str__(self):
        return f"Job[{self.work.kind.name}, local={self.local_id}, remote={self.remote_id}]"


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
        self.models = models
        self.device_info = DeviceInfo("Cloud", "Remote GPU", 24)
        self._requests = RequestManager()
        self._token: str = ""
        self._user: User | None = None
        self._current_job: JobInfo | None = None
        self._cancel_requested: bool = False
        self._queue: asyncio.Queue[JobInfo] = asyncio.Queue()
        self._features = enumerate_features({})

    async def _get(self, op: str):
        return await self._requests.get(f"{self.url}/{op}", bearer=self._token)

    async def _post(self, op: str, data: dict):
        return await self._requests.post(f"{self.url}/{op}", data, bearer=self._token)

    async def sign_in(self):
        client_id = str(uuid.uuid4())
        info = f"Generative AI for Krita [Device: {platform.node()}]"
        log.info(f"Sending authorization request for {info} to {self.url}")
        init = await self._post("auth/initiate", dict(client_id=client_id, client_info=info))

        sign_in_url = f"{self.default_web_url}{init['url']}"
        log.info(f"Waiting for completion of authorization at {sign_in_url}")
        yield sign_in_url

        auth_confirm = await self._post("auth/confirm", dict(client_id=client_id))
        time = datetime.now()
        while auth_confirm["status"] == "not-found":
            if (datetime.now() - time).seconds > 300:
                raise TimeoutError(_("Sign-in attempt timed out after 5 minutes"))
            await asyncio.sleep(2)
            auth_confirm = await self._post("auth/confirm", dict(client_id=client_id))

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
            raise e
        self._user = User(user_data["id"], user_data["name"])
        self._user.images_generated = user_data["images_generated"]
        self._user.credits = user_data["credits"]
        self._features = enumerate_features(user_data)
        log.info(f"Connected to {self.url}, user: {self._user.id}")
        return self._user

    async def enqueue(self, work: WorkflowInput, front: bool = False):
        apply_limits(work, self.features)
        job = JobInfo(str(uuid.uuid4()), work)
        await self._queue.put(job)
        return job.local_id

    async def listen(self):
        yield ClientMessage(ClientEvent.connected)
        while True:
            try:
                self._current_job = await self._queue.get()
                self._cancel_requested = False
                async for msg in self._process_job(self._current_job):
                    yield msg
                    if self._cancel_requested:
                        yield ClientMessage(ClientEvent.interrupted, self._current_job.local_id)
                        break

            except NetworkError as e:
                job_id = self._current_job.local_id if self._current_job else ""
                if msg := self._process_http_error(e, job_id):
                    yield msg
                else:
                    msg = e.message
                    log.exception(f"Network error while processing {self._current_job}: {msg}")
                    if job_id:
                        yield ClientMessage(ClientEvent.error, job_id, error=msg)
            except Exception as e:
                log.exception(f"Unhandled exception while processing {self._current_job}")
                if self._current_job is not None:
                    yield ClientMessage(ClientEvent.error, self._current_job.local_id, error=str(e))
            except asyncio.CancelledError:
                break
            finally:
                self._current_job = None

    async def _process_job(self, job: JobInfo):
        user = ensure(self.user)
        inputs = job.work.to_dict(max_image_size=16 * 1024)

        async for progress in self.send_lora(job.work):
            yield ClientMessage(ClientEvent.upload, job.local_id, progress)

        await self.send_images(inputs)

        data = {
            "input": {
                "workflow": inputs,
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
        yield ClientMessage(ClientEvent.progress, job.local_id, 0)

        status = response["status"].lower()
        while status == "in_queue" or status == "in_progress":
            response = await self._post(f"status/{job.worker_id}/{job.remote_id}", {})
            status = response["status"].lower()

            if status == "in_queue":
                yield ClientMessage(ClientEvent.queued, job.local_id)

            elif status == "in_progress":
                progress = 0.09
                if output := response.get("output", None):
                    progress = output.get("progress", progress)
                yield ClientMessage(ClientEvent.progress, job.local_id, progress)
            await asyncio.sleep(_poll_interval)

        if status == "completed":
            output = response["output"]
            images = await self.receive_images(output["images"])
            pose = output.get("pose", None)
            log.info(f"{job} completed, got {len(images)} images{', got pose' if pose else ''}")
            lora_warning = output.get("lora_warning", False)
            if lora_warning:
                log.warning(f"{job} encountered LoRA that could not be applied to the checkpoint")
            error = "incompatible_lora" if lora_warning else None
            yield ClientMessage(ClientEvent.finished, job.local_id, 1, images, pose, error=error)

        elif status == "failed":
            err_msg, err_trace = _extract_error(response, job.remote_id)
            log.error(f"{job} failed\n{err_msg}\n{err_trace}")
            yield ClientMessage(ClientEvent.error, job.local_id, error=err_msg)

        elif status == "cancelled":
            log.info(f"{job} was cancelled")
            yield ClientMessage(ClientEvent.interrupted, job.local_id)

        elif status == "timed_out":
            log.warning(f"{job} timed out")
            yield ClientMessage(ClientEvent.error, job.local_id, error="job timed out")
        else:
            log.warning(f"Got unknown job status {status}")

    async def interrupt(self):
        if job := self._current_job:
            self._cancel_requested = True
            if job.remote_id and job.worker_id:
                response = await self._post(f"cancel/{job.worker_id}/{job.remote_id}", {})
                log.info(f"Requested cancellation of {job}: {response}")

    async def clear_queue(self):
        self._queue = asyncio.Queue()

    @property
    def user(self):
        return self._user

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
        upload = await self._post("upload/lora", dict(hash=lora.hash, size=lora.size))
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
            raise Exception(_("Connection error during upload of LoRA model") + f" {lora.name}")
        except Exception as e:
            raise Exception(_("Error during upload of LoRA model") + f" {lora.name}") from e

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


def _checkpoint_info(id: str, arch: Arch):
    models = chain(resources.default_checkpoints, resources.deprecated_models)
    res = next(m for m in models if m.id.identifier == id and m.arch == arch)
    return (res.filename, CheckpointInfo(res.filename, res.arch))


_poll_interval = 0.5  # seconds

models = ClientModels()
models.checkpoints = {
    filename: info
    for filename, info in (
        _checkpoint_info(name, arch)
        for name, arch in [
            ("dreamshaper", Arch.sd15),
            ("realistic_vision", Arch.sd15),
            ("serenity", Arch.sd15),
            ("flat2d_animerge", Arch.sd15),
            ("realvis", Arch.sdxl),
            ("zavychroma", Arch.sdxl),
            ("flux_schnell", Arch.flux),
            ("noobai", Arch.illu_v),
        ]
    )
}
models.vae = []
models.loras = [
    "Hyper-SD15-8steps-CFG-lora.safetensors",
    "Hyper-SDXL-8steps-CFG-lora.safetensors",
    "ip-adapter-faceid-plusv2_sd15_lora.safetensors",
    "ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
]
models.upscalers = [
    "4x_NMKD-Superscale-SP_178000_G.pth",
    "HAT_SRx4_ImageNet-pretrain.pth",
    "OmniSR_X2_DIV2K.safetensors",
    "OmniSR_X3_DIV2K.safetensors",
    "OmniSR_X4_DIV2K.safetensors",
]
# fmt: off
models.resources = {
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.inpaint): "control_v11p_sd15_inpaint_fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.universal): "xinsir-controlnet-union-sdxl-1.0-promax.safetensors",
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.scribble): "control_lora_rank128_v11p_sd15_scribble_fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.line_art): "control_v11p_sd15_lineart_fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.soft_edge): "control_v11p_sd15_softedge_fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.canny_edge): "control_v11p_sd15_canny_fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.depth): "control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.pose): "control_lora_rank128_v11p_sd15_openpose_fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.blur):"control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.stencil): "control_v1p_sd15_qrcode_monster.safetensors",
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.scribble): "noob-sdxl-controlnet-scribble_pidinet.fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.line_art): "noob-sdxl-controlnet-lineart_anime.fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.canny_edge): "noob_sdxl_controlnet_canny.fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.depth): "noob-sdxl-controlnet-depth_midas-v1-1.fp16.safetensors",
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.pose): "noobaiXLControlnet_openposeModel.safetensors",
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.blur): "noob-sdxl-controlnet-tile.fp16.safetensors",
    resource_id(ResourceKind.ip_adapter, Arch.sd15, ControlMode.reference): "ip-adapter_sd15.safetensors",
    resource_id(ResourceKind.ip_adapter, Arch.sdxl, ControlMode.reference): "ip-adapter_sdxl_vit-h.safetensors",
    resource_id(ResourceKind.ip_adapter, Arch.illu, ControlMode.reference): "noobIPAMARK1_mark1.safetensors",
    resource_id(ResourceKind.ip_adapter, Arch.sd15, ControlMode.face): "ip-adapter-faceid-plusv2_sd15.bin",
    resource_id(ResourceKind.ip_adapter, Arch.sdxl, ControlMode.face): "ip-adapter-faceid-plusv2_sdxl.bin",
    resource_id(ResourceKind.ip_adapter, Arch.flux, ControlMode.reference): "flux1-redux-dev.safetensors",
    resource_id(ResourceKind.clip_vision, Arch.all, "ip_adapter"): "clip-vision_vit-h.safetensors",
    resource_id(ResourceKind.clip_vision, Arch.illu, "ip_adapter"): "clip-vision_vit-g.safetensors",
    resource_id(ResourceKind.clip_vision, Arch.flux, "redux"): "sigclip_vision_patch14_384.safetensors",
    resource_id(ResourceKind.lora, Arch.sd15, "hyper"): "Hyper-SD15-8steps-CFG-lora.safetensors",
    resource_id(ResourceKind.lora, Arch.sdxl, "hyper"): "Hyper-SDXL-8steps-CFG-lora.safetensors",
    resource_id(ResourceKind.lora, Arch.sd15, ControlMode.face): "ip-adapter-faceid-plusv2_sd15_lora.safetensors",
    resource_id(ResourceKind.lora, Arch.sdxl, ControlMode.face): "ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.default): UpscalerName.default.value,
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_2x): UpscalerName.fast_2x.value,
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_3x): UpscalerName.fast_3x.value,
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_4x): UpscalerName.fast_4x.value,
    resource_id(ResourceKind.inpaint, Arch.sdxl, "fooocus_head"): "fooocus_inpaint_head.pth",
    resource_id(ResourceKind.inpaint, Arch.sdxl, "fooocus_patch"): "inpaint_v26.fooocus.patch",
    resource_id(ResourceKind.inpaint, Arch.all, "default"): "MAT_Places512_G_fp16.safetensors",
}
# fmt: on
