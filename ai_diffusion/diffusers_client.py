"""
Diffusers Client for Qwen Image Layered Pipeline

HTTP client that communicates with the local diffusers server
for layered image generation and segmentation.
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from dataclasses import dataclass
from typing import Iterable

from .api import WorkflowInput, WorkflowKind, LayeredInput, DiffusersInput, DiffusersMode
from .client import (
    Client,
    ClientEvent,
    ClientMessage,
    ClientModels,
    DeviceInfo,
    ClientFeatures,
    ClientJobQueue,
    CheckpointInfo,
)
from .files import FileFormat
from .image import Image, ImageCollection
from .network import RequestManager, NetworkError
from .resources import Arch
from .settings import PerformanceSettings, settings
from .util import client_logger as log


@dataclass
class JobInfo:
    """Local job tracking info."""

    local_id: str
    work: WorkflowInput
    remote_id: str | None = None
    cancelled: bool = False

    def __str__(self):
        return f"DiffusersJob[local={self.local_id}, remote={self.remote_id}]"


@dataclass
class ModelLoadingStatus:
    """Status of model loading/downloading."""

    status: str  # not_loaded, loading, loaded, error
    message: str
    progress: float  # 0.0 - 1.0
    error: str | None = None

    @property
    def is_loading(self) -> bool:
        return self.status == "loading"

    @property
    def is_loaded(self) -> bool:
        return self.status == "loaded"

    @property
    def is_error(self) -> bool:
        return self.status == "error"


class DiffusersClient(Client):
    """HTTP client for local diffusers server with polling-based status checks."""

    default_url = "http://127.0.0.1:8189"

    def __init__(self, url: str, token: str = ""):
        self.url = url
        self.token = token
        self.models = ClientModels()
        self.device_info = DeviceInfo("cuda", "Local GPU", 0)
        self._requests = RequestManager()
        self._current_job: JobInfo | None = None
        self._cancel_requested: bool = False
        self._disconnecting: bool = False
        self._queue: ClientJobQueue[JobInfo] = ClientJobQueue()
        self._features = ClientFeatures(
            ip_adapter=False,  # Qwen doesn't use IP-Adapter
            translation=False,
            languages=[],
            max_upload_size=0,
            max_control_layers=0,
        )

    def clear_queue(self):
        """Clear all pending jobs and signal disconnect."""
        self._disconnecting = True
        self._cancel_requested = True
        # Clear the queue - mark all jobs as cancelled
        for job in self._queue._jobs:
            job.cancelled = True
        self._queue._jobs.clear()
        self._queue._event.set()  # Wake up any waiting get()

    @staticmethod
    async def connect(url: str = default_url, access_token: str = "") -> Client:
        """Connect to diffusers server and verify it's running.

        Args:
            url: Server URL (http:// or https://)
            access_token: Bearer token for authentication (optional)
        """
        client = DiffusersClient(url, token=access_token)
        log.info(f"Connecting to diffusers server at {client.url}")

        try:
            # Get system stats to verify server is running
            # Use longer timeout for initial connection (GPU init can be slow)
            stats = await client._get("system_stats", timeout=120)
            client.device_info = DeviceInfo.parse(stats)
            log.info(f"Connected to diffusers server: {client.device_info.name}")

            # Set up minimal models info - Qwen uses HuggingFace model
            client.models.checkpoints = {
                "Qwen-Image-Layered": CheckpointInfo(
                    "Qwen-Image-Layered",
                    Arch.sdxl,  # Use SDXL as closest approximation
                    FileFormat.diffusion,
                )
            }

        except NetworkError as e:
            msg = f"Could not connect to diffusers server at {url}: {e.message}"
            log.error(msg)
            raise Exception(msg) from e
        except Exception as e:
            msg = f"Could not connect to diffusers server at {url}: {str(e)}"
            log.error(msg)
            raise Exception(msg) from e

        return client

    async def _get(self, op: str, timeout: float = 30):
        """GET request to server."""
        return await self._requests.get(
            f"{self.url}/{op}", timeout=timeout, bearer=self.token or None
        )

    async def _post(self, op: str, data: dict):
        """POST request to server."""
        return await self._requests.post(
            f"{self.url}/{op}", data, bearer=self.token or None
        )

    async def get_model_status(self) -> ModelLoadingStatus:
        """Get current model loading status."""
        try:
            data = await self._get("model_status")
            return ModelLoadingStatus(
                status=data.get("status", "not_loaded"),
                message=data.get("message", ""),
                progress=data.get("progress", 0.0),
                error=data.get("error"),
            )
        except Exception as e:
            log.warning(f"Failed to get model status: {e}")
            return ModelLoadingStatus("error", str(e), 0.0, str(e))

    async def request_model_load(self) -> bool:
        """Request the server to start loading the model."""
        try:
            response = await self._post("load_model", {})
            status = response.get("status", "")
            return status in ("loading_started", "loading", "already_loaded")
        except Exception as e:
            log.warning(f"Failed to request model load: {e}")
            return False

    async def get_vram_usage(self) -> dict:
        """Get current VRAM usage from server.

        Returns dict with:
            - backend: str (cuda, rocm, mps, cpu)
            - devices: list of dicts with vram_total, vram_used, vram_percent
        """
        try:
            return await self._get("system_stats")
        except Exception as e:
            log.warning(f"Failed to get VRAM stats: {e}")
            return {"devices": [], "backend": "unknown"}

    async def enqueue(self, work: WorkflowInput, front: bool = False) -> str:
        """Add a job to the queue."""
        job = JobInfo(str(uuid.uuid4()), work)
        self._queue.put(job, front)
        log.info(f"Enqueued {job}")
        return job.local_id

    async def listen(self):
        """Process jobs from queue and poll for results."""
        yield ClientMessage(ClientEvent.connected)

        while not self._disconnecting:
            try:
                # Use a timeout so we can check _disconnecting periodically
                try:
                    self._current_job = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                self._cancel_requested = False

                if self._disconnecting:
                    break

                if self._current_job.cancelled:
                    yield ClientMessage(ClientEvent.interrupted, self._current_job.local_id)
                    continue

                async for msg in self._process_job(self._current_job):
                    if self._disconnecting:
                        break
                    yield msg
                    if self._cancel_requested:
                        yield ClientMessage(
                            ClientEvent.interrupted, self._current_job.local_id
                        )
                        break

            except NetworkError as e:
                job_id = self._current_job.local_id if self._current_job else ""
                log.error(f"Network error while processing {self._current_job}: {e.message}")
                if job_id:
                    yield ClientMessage(ClientEvent.error, job_id, error=e.message)

            except Exception as e:
                log.exception(f"Unhandled exception while processing {self._current_job}")
                if self._current_job is not None:
                    yield ClientMessage(
                        ClientEvent.error, self._current_job.local_id, error=str(e)
                    )

            except asyncio.CancelledError:
                break

            finally:
                self._current_job = None

        log.info("Client listen loop exited")

    async def _process_job(self, job: JobInfo):
        """Convert WorkflowInput to diffusers API and poll for results."""
        work = job.work

        # Build request from WorkflowInput
        request = self._build_request(work)

        log.info(f"Submitting {job} to diffusers server")

        # Submit job
        try:
            response = await self._post("generate", request)
        except NetworkError as e:
            log.error(f"Failed to submit job: {e.message}")
            raise

        job.remote_id = response.get("job_id")
        log.info(f"{job} submitted, remote_id={job.remote_id}")

        yield ClientMessage(ClientEvent.progress, job.local_id, 0)

        # Poll for results
        poll_interval = 0.5  # seconds
        while True:
            if self._cancel_requested:
                # Request interrupt
                try:
                    await self._post("interrupt", {})
                except Exception as e:
                    log.warning(f"Failed to request interrupt: {e}")
                break

            try:
                status = await self._get(f"status/{job.remote_id}")
            except NetworkError as e:
                log.warning(f"Failed to get status: {e.message}")
                await asyncio.sleep(poll_interval)
                continue

            status_value = status.get("status", "unknown")

            if status_value in ("pending", "processing"):
                progress = status.get("progress", 0.5)
                message = status.get("message")  # Model loading status, etc.
                yield ClientMessage(ClientEvent.progress, job.local_id, progress, message=message)
                await asyncio.sleep(poll_interval)

            elif status_value == "completed":
                # Convert base64 images to ImageCollection
                images = self._decode_images(status.get("images", []))
                log.info(f"{job} completed with {len(images)} layers")
                yield ClientMessage(
                    ClientEvent.finished,
                    job.local_id,
                    1.0,
                    images=images,
                )
                break

            elif status_value == "failed":
                error = status.get("error", "Unknown error")
                log.error(f"{job} failed: {error}")
                yield ClientMessage(ClientEvent.error, job.local_id, error=error)
                break

            elif status_value == "interrupted":
                log.info(f"{job} was interrupted")
                yield ClientMessage(ClientEvent.interrupted, job.local_id)
                break

            else:
                log.warning(f"Unknown status: {status_value}")
                await asyncio.sleep(poll_interval)

    def _build_request(self, work: WorkflowInput) -> dict:
        """Convert WorkflowInput to diffusers server request format."""
        # Use separate paths for layered vs general diffusers to avoid regressions
        if work.kind in (WorkflowKind.layered_generate, WorkflowKind.layered_segment):
            return self._build_layered_request(work)
        else:
            return self._build_diffusers_request(work)

    def _build_layered_request(self, work: WorkflowInput) -> dict:
        """Build request for Qwen layered generation/segmentation (original format)."""
        request: dict = {}
        layered = work.layered or LayeredInput()

        # Mode
        if work.kind == WorkflowKind.layered_generate:
            request["mode"] = "layered_generate"
        else:
            request["mode"] = "layered_segment"

        # Prompt
        if work.conditioning:
            request["prompt"] = work.conditioning.positive
            request["negative_prompt"] = work.conditioning.negative or " "
        else:
            request["prompt"] = ""
            request["negative_prompt"] = " "

        # Sampling params
        if work.sampling:
            request["seed"] = work.sampling.seed
            request["cfg_scale"] = work.sampling.cfg_scale
            request["num_inference_steps"] = work.sampling.total_steps
        else:
            request["seed"] = 42
            request["cfg_scale"] = 4.0
            request["num_inference_steps"] = 50

        # Layered params
        request["layers"] = layered.num_layers
        request["resolution"] = layered.resolution
        request["cfg_normalize"] = layered.cfg_normalize
        request["use_en_prompt"] = layered.use_en_prompt

        # Input image for segmentation mode
        if work.kind == WorkflowKind.layered_segment and work.images:
            if work.images.initial_image:
                request["image"] = self._encode_image(work.images.initial_image)

        return request

    def _build_diffusers_request(self, work: WorkflowInput) -> dict:
        """Build request for general diffusers generation (txt2img, img2img, inpaint)."""
        request: dict = {}
        diffusers = work.diffusers or DiffusersInput()

        # Mode
        mode_map = {
            WorkflowKind.diffusers_generate: "text_to_image",
            WorkflowKind.diffusers_img2img: "img2img",
            WorkflowKind.diffusers_inpaint: "inpaint",
        }
        request["mode"] = mode_map.get(work.kind, "text_to_image")

        # Model ID (empty = use server default)
        request["model_id"] = diffusers.model_id

        # Prompt
        if work.conditioning:
            request["prompt"] = work.conditioning.positive
            request["negative_prompt"] = work.conditioning.negative or ""
        else:
            request["prompt"] = ""
            request["negative_prompt"] = ""

        # Sampling params
        if work.sampling:
            request["seed"] = work.sampling.seed
            request["num_inference_steps"] = work.sampling.total_steps
            request["guidance_scale"] = work.sampling.cfg_scale
        else:
            request["seed"] = -1
            request["num_inference_steps"] = diffusers.num_inference_steps
            request["guidance_scale"] = diffusers.guidance_scale

        # Resolution
        request["width"] = diffusers.width
        request["height"] = diffusers.height

        # Strength (for img2img/inpaint)
        request["strength"] = diffusers.strength

        # Input image (for img2img, inpaint)
        if work.kind in (WorkflowKind.diffusers_img2img, WorkflowKind.diffusers_inpaint):
            if work.images and work.images.initial_image:
                request["image"] = self._encode_image(work.images.initial_image)

        # Mask image (for inpaint mode)
        if work.kind == WorkflowKind.diffusers_inpaint:
            if work.images and work.images.hires_mask:
                request["mask"] = self._encode_image(work.images.hires_mask)

        # Optimization settings from preset
        request["offload"] = diffusers.offload
        request["quantization"] = diffusers.quantization
        request["quantize_transformer"] = diffusers.quantize_transformer
        request["quantize_text_encoder"] = diffusers.quantize_text_encoder
        request["vae_tiling"] = diffusers.vae_tiling
        request["ramtorch"] = diffusers.ramtorch

        return request

    def _encode_image(self, image: Image) -> str:
        """Encode image to base64 PNG."""
        return image.to_base64()

    def _decode_images(self, images_b64: list[str]) -> ImageCollection:
        """Decode base64 images to ImageCollection."""
        images = ImageCollection()
        for b64 in images_b64:
            try:
                data = base64.b64decode(b64)
                img = Image.from_bytes(data)
                images.append(img)
            except Exception as e:
                log.warning(f"Failed to decode image: {e}")
        return images

    async def interrupt(self):
        """Interrupt the current job."""
        self._cancel_requested = True
        if self._current_job and self._current_job.remote_id:
            try:
                await self._post("interrupt", {})
                log.info(f"Requested interrupt for {self._current_job}")
            except Exception as e:
                log.warning(f"Failed to request interrupt: {e}")

    async def cancel(self, job_ids: Iterable[str]):
        """Cancel queued jobs."""
        for job in self._queue:
            if job.local_id in job_ids:
                job.cancelled = True
                log.info(f"Cancelled queued job {job.local_id}")

    async def disconnect(self):
        """Disconnect from server."""
        log.info("Disconnecting from diffusers server")

    @property
    def features(self) -> ClientFeatures:
        return self._features

    @property
    def performance_settings(self) -> PerformanceSettings:
        return PerformanceSettings(
            batch_size=1,  # Qwen generates one set of layers at a time
            resolution_multiplier=1.0,
            max_pixel_count=1,
        )

    @property
    def user(self):
        return None  # No user authentication for local server

    def supports_arch(self, arch: Arch) -> bool:
        # Qwen is its own architecture, accept all for now
        return True
