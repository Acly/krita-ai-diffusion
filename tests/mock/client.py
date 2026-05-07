"""Mock implementation of the Client interface for use in tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from ai_diffusion.api import WorkflowInput
from ai_diffusion.client import (
    CheckpointInfo,
    Client,
    ClientFeatures,
    ClientMessage,
    ClientModels,
    DeviceInfo,
    MissingResources,
)
from ai_diffusion.resources import Arch, ControlMode, ResourceKind, UpscalerName, resource_id
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
