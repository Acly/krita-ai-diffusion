import asyncio
import socket
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ai_diffusion.backend.server import Server, ServerState, _available_port, _server_ready_url

from .conftest import qtapp


@qtapp
async def test_concurrent_start_launches_only_one_process(tmp_path: Path, monkeypatch):
    server = Server(str(tmp_path))
    server.state = ServerState.stopped
    entered = asyncio.Event()
    release = asyncio.Event()

    async def start(port):
        entered.set()
        await release.wait()
        server.state = ServerState.running
        server.url = "127.0.0.1:8199"
        return server.url

    implementation = AsyncMock(side_effect=start)
    monkeypatch.setattr(server, "_start", implementation)
    first = asyncio.create_task(server.start())
    await asyncio.wait_for(entered.wait(), timeout=2)
    assert server.state is ServerState.starting
    second = asyncio.create_task(server.start())
    await asyncio.sleep(0)
    release.set()
    assert await first == await second == "127.0.0.1:8199"
    assert implementation.await_count == 1


def test_busy_default_port_selects_another_port():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        port = busy.getsockname()[1]
        assert _available_port(port, explicit=False) != port
        with pytest.raises(RuntimeError, match="already in use"):
            _available_port(port, explicit=True)


def test_ready_url_strips_ansi_and_ignores_unrelated_messages():
    assert (
        _server_ready_url(
            "\x1b[32m[INFO]\x1b[0m To see the GUI go to: http://127.0.0.1:8199\x1b[0m"
        )
        == "127.0.0.1:8199"
    )
    assert _server_ready_url("Port 8188 is already in use") is None
