"""Mock implementation of the Client interface for use in tests."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

from ai_diffusion.backend.api import WorkflowInput
from ai_diffusion.backend.client import (
    CheckpointInfo,
    Client,
    ClientEvent,
    ClientFeatures,
    ClientJobQueue,
    ClientMessage,
    ClientModels,
    DeviceInfo,
    MissingResources,
)
from ai_diffusion.backend.resources import (
    Arch,
    ControlMode,
    ResourceKind,
    UpscalerName,
    resource_id,
)
from ai_diffusion.image import Bounds, Extent, Image, ImageCollection
from ai_diffusion.settings import PerformanceSettings


class MockClient(Client):
    """Scriptable Client stub.

    Usage::

        client = MockClient()
        # Make the next connect() call raise an error:
        client.connect_error = NetworkError(0, "refused", "ws://localhost")
        # Pre-load messages that listen() will yield:
        client.messages = [
            ClientMessage(ClientEvent.connected, ""),
            ClientMessage(ClientEvent.progress, "job-1", 0.5),
        ]
    """

    url: str = "ws://mock"
    models: ClientModels
    device_info: DeviceInfo

    def __init__(self):
        self.models = ClientModels()
        self.device_info = DeviceInfo("cpu", "Mock GPU", 8)
        self._features = ClientFeatures()
        self._performance = PerformanceSettings()
        self._missing: MissingResources | None = None

        # Populate default resources so Model and workflow tests work out of the box
        # without requiring per-test boilerplate.
        _checkpoint = "test_sd15.safetensors"
        self.models.checkpoints[_checkpoint] = CheckpointInfo(_checkpoint, Arch.sd15)
        _upscaler = "4x_NMKD-Superscale-SP_178000_G.pth"
        self.models.upscalers = [_upscaler]
        self.models.resources = {
            resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.inpaint): (
                "control_v11p_sd15_inpaint.pth"
            ),
            resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.default): _upscaler,
        }

        # Scriptable behaviour
        self.connect_error: Exception | None = None
        self.messages: list[ClientMessage] = []

        # Introspection helpers
        self.connect_count = 0
        self.disconnect_count = 0
        self.connected = False
        self.enqueued: list[WorkflowInput] = []

        # Internal queue fed by push() for listen() consumers
        self._queue: asyncio.Queue[ClientMessage | None] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Client interface
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self.connect_count += 1
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    async def discover_models(self, refresh: bool) -> AsyncGenerator[Client.DiscoverStatus, Any]:  # type: ignore[override]
        yield Client.DiscoverStatus("checkpoints", 1, 1)

    async def enqueue(self, work: WorkflowInput, front: bool = False) -> str:
        job_id = f"mock-job-{len(self.enqueued)}"
        self.enqueued.append(work)
        return job_id

    async def listen(self) -> AsyncGenerator[ClientMessage, Any]:  # type: ignore[override]
        # Drain any pre-loaded messages first
        for msg in self.messages:
            yield msg
        self.messages = []

        # Then yield messages pushed via push() until None sentinel
        while True:
            msg = await self._queue.get()
            if msg is None:
                break
            yield msg

    async def interrupt(self):
        pass

    async def cancel(self, job_ids: Any):
        pass

    async def disconnect(self):
        self.disconnect_count += 1
        self.connected = False

    @property
    def missing_resources(self) -> MissingResources | None:
        return self._missing

    @missing_resources.setter
    def missing_resources(self, value: MissingResources | None):
        self._missing = value

    @property
    def features(self) -> ClientFeatures:
        return self._features

    @property
    def performance_settings(self) -> PerformanceSettings:
        return self._performance

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def push(self, msg: ClientMessage):
        """Send a message to active listen() consumers."""
        self._queue.put_nowait(msg)

    def close(self):
        """Signal listen() to stop iterating."""
        self._queue.put_nowait(None)


class _Job(NamedTuple):
    id: str
    work: WorkflowInput


class FakeClient(Client):
    """Testing client that simulates a connected server without running any inference.

    Exposes a small selection of default models. Generation jobs return a dummy image
    picked from a fixed list (resized/cropped to the requested output resolution) after
    a short delay to simulate generation time. All other workflow inputs are ignored.

    Queuing mimics real clients: a new job starts processing immediately if the client
    is idle, otherwise it waits in queue. Only one job is processed at a time, queued
    jobs are processed in order (`front=True` prioritizes).

    Used by scripts/design.py via the --fake-client flag.
    """

    url = "ws://fake"
    generation_delay = 1.0

    _image_dir = Path(__file__).parent.parent / "images"
    _dummy_images: ClassVar[list[str]] = [
        "beach_1536x1024.webp",
        "lake_1536x1024.webp",
        "flowers.webp",
        "watercolor.webp",
    ]

    def __init__(self):
        self.models = ClientModels()
        self.device_info = DeviceInfo("cuda", "Fake GPU", 24)
        self._features = ClientFeatures()
        self._performance = PerformanceSettings()
        self._images: list[Image] = []
        self._job_counter = itertools.count(1)
        self._image_counter = itertools.count()
        self._jobs: ClientJobQueue[_Job] = ClientJobQueue()
        self._messages: asyncio.Queue[ClientMessage] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._active_job: _Job | None = None
        self._active_task: asyncio.Task | None = None
        self._disconnecting = False

        # A few checkpoints covering different architectures, matching built-in styles
        for filename, arch in [
            ("dreamshaper_8.safetensors", Arch.sd15),
            ("RealVisXL_V5.0_fp16.safetensors", Arch.sdxl),
            ("flux1-schnell-fp8.safetensors", Arch.flux),
        ]:
            self.models.checkpoints[filename] = CheckpointInfo(filename, arch)

        upscaler = "4x_NMKD-Superscale-SP_178000_G.pth"
        self.models.upscalers = [upscaler]
        hyper_sd15 = "Hyper-SD15-8steps-CFG-lora.safetensors"
        hyper_sdxl = "Hyper-SDXL-8steps-CFG-lora.safetensors"
        self.models.loras = [hyper_sd15, hyper_sdxl]
        self.models.resources = {
            resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.default): upscaler,
            resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.inpaint): (
                "control_v11p_sd15_inpaint.pth"
            ),
            resource_id(ResourceKind.lora, Arch.sd15, "hyper"): hyper_sd15,
            resource_id(ResourceKind.lora, Arch.sdxl, "hyper"): hyper_sdxl,
        }

    # ------------------------------------------------------------------
    # Client interface
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._disconnecting = False
        self._images = [Image.load(self._image_dir / name) for name in self._dummy_images]

    async def discover_models(self, refresh: bool) -> AsyncGenerator[Client.DiscoverStatus, Any]:  # type: ignore[override]
        yield Client.DiscoverStatus("checkpoints", 1, 1)

    async def enqueue(self, work: WorkflowInput, front: bool = False) -> str:
        job = _Job(f"fake-job-{next(self._job_counter)}", work)
        self._jobs.put(job, front=front)
        return job.id

    async def listen(self) -> AsyncGenerator[ClientMessage, Any]:  # type: ignore[override]
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())
        while True:
            yield await self._messages.get()

    async def interrupt(self):
        # Like the real client's /interrupt: stops the active job, queued jobs continue
        if self._active_job is not None:
            await self._report(ClientEvent.interrupted, self._active_job.id)
            if self._active_task is not None:
                self._active_task.cancel()

    async def cancel(self, job_ids: Iterable[str]):
        for job_id in job_ids:
            if self._active_job is not None and self._active_job.id == job_id:
                await self.interrupt()
            elif self._remove_waiting(job_id):
                await self._report(ClientEvent.interrupted, job_id)

    async def disconnect(self):
        self._disconnecting = True
        if self._worker is not None:
            self._worker.cancel()  # also cancels the active job it is awaiting
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        self._jobs.remove_if(lambda _: True)

    @property
    def features(self) -> ClientFeatures:
        return self._features

    @property
    def performance_settings(self) -> PerformanceSettings:
        return self._performance

    # ------------------------------------------------------------------
    # Fake generation
    # ------------------------------------------------------------------

    async def _run(self):
        """Worker loop: takes jobs from the queue and processes them one at a time."""
        try:
            while True:
                job = await self._jobs.get()
                self._active_job = job
                self._active_task = asyncio.create_task(self._process(job))
                try:
                    await self._active_task
                except asyncio.CancelledError:
                    if self._disconnecting:
                        raise
                    # else: job was interrupted via interrupt() / cancel(), move on
                finally:
                    self._active_job = None
                    self._active_task = None
        except asyncio.CancelledError:
            pass  # disconnect

    async def _process(self, job: _Job):
        try:
            await self._report(ClientEvent.queued, job.id)
            await asyncio.sleep(self.generation_delay / 2)
            await self._report(ClientEvent.progress, job.id, 0.5)
            await asyncio.sleep(self.generation_delay / 2)
            work = job.work
            images = ImageCollection(
                self._make_image(work) for _ in range(max(1, work.batch_count))
            )
            await self._report(ClientEvent.finished, job.id, 1, images=images)
        except Exception as e:  # CancelledError propagates to the worker loop
            await self._report(ClientEvent.error, job.id, error=str(e))

    def _remove_waiting(self, job_id: str) -> bool:
        count = len(self._jobs)
        self._jobs.remove_if(lambda job: job.id == job_id)
        return len(self._jobs) < count

    def _make_image(self, work: WorkflowInput):
        extent = work.images.extent.target if work.images else Extent(512, 512)
        source = self._images[next(self._image_counter) % len(self._images)]
        # Scale to cover the target extent (keeping aspect ratio), then crop the center
        scale = max(extent.width / source.width, extent.height / source.height)
        image = Image.scale(source, source.extent * scale)
        if image.extent != extent:
            x = (image.width - extent.width) // 2
            y = (image.height - extent.height) // 2
            image = Image.crop(image, Bounds(x, y, extent.width, extent.height))
        return image

    async def _report(self, event: ClientEvent, job_id: str, progress: float = 0, **kwargs):
        await self._messages.put(ClientMessage(event, job_id, progress, **kwargs))
