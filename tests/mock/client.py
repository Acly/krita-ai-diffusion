"""Mock implementation of the Client interface for use in tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from ai_diffusion.client import (
    Client,
    ClientFeatures,
    ClientMessage,
    ClientModels,
    DeviceInfo,
    MissingResources,
)
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

        # Scriptable behaviour
        self.connect_error: Exception | None = None
        self.messages: list[ClientMessage] = []

        # Introspection helpers
        self.connect_count = 0
        self.disconnect_count = 0
        self.connected = False

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

    async def enqueue(self, work: Any, front: bool = False) -> str:
        return "mock-job"

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
