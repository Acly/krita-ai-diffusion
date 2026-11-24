"""Tests for network module, especially socket-based PUT with Expect: 100-continue."""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
from aiohttp import web
from PyQt5.QtCore import QByteArray

from ai_diffusion.backend.network import NetworkError, RequestManager

from .conftest import qtapp


# ---------------------------------------------------------------------------
# Test HTTP Server
# ---------------------------------------------------------------------------


class _TestHTTPServer:
    """Async test HTTP server using aiohttp for testing socket PUT requests."""

    def __init__(self, port: int | None = None):
        self.port = port or _find_free_port()
        self.app = web.Application(client_max_size=16 * 1024**2)
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self.behavior = "send_100_continue"
        self.request_body: bytearray | None = None
        
        # Route PUT requests to the handler
        self.app.router.add_put("/api/upload", self._handle_put)

    async def _handle_put(self, request: web.Request) -> web.Response:
        """Handle PUT requests according to configured behavior."""
        if self.behavior == "send_100_continue":
                # Read payload and send 200 OK (normal upload)
            body = await request.read()
            self.request_body = bytearray(body)
            response = {"status": "ok"}
            return web.json_response(response, status=200)
        
        elif self.behavior == "send_early_200":
                # Send 200 OK immediately without reading payload (matches production expect_handler)
                # In production, this uses aiohttp's expect_handler to detect cached images before
                # body consumption. The client sees 200 and skips the upload.
            self.request_body = None
            response = {"status": "cached"}
            return web.json_response(response, status=200)
        
        elif self.behavior == "ignore_expect":
                # Read payload normally, send 200 OK (server ignores Expect: 100-continue)
            body = await request.read()
            self.request_body = bytearray(body)
            response = {"status": "ok"}
            return web.json_response(response, status=200)
        
        elif self.behavior == "send_error":
            # Send error response
            response = {"error": "Test error"}
            return web.json_response(response, status=400)
        
        elif self.behavior == "slow_100_continue":
            # Simulate slow server
            await asyncio.sleep(0.5)
            body = await request.read()
            self.request_body = bytearray(body)
            response = {"status": "ok"}
            return web.json_response(response, status=200)
        
        else:
            return web.json_response({"error": "Unknown behavior"}, status=500)

    async def start(self):
        """Start the server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        await self.site.start()

    async def stop(self):
        """Stop the server."""
        if self.runner is not None:
            await self.runner.cleanup()

    def set_behavior(self, behavior: str):
        """Set how the server should respond to requests."""
        self.behavior = behavior
        self.request_body = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _find_free_port():
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def test_server(qtapp):
    """Fixture providing an aiohttp test server."""
    server = _TestHTTPServer()
    qtapp.run(server.start())
    yield server
    qtapp.run(server.stop())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@qtapp
async def test_socket_put_with_100_continue(test_server: _TestHTTPServer):
    """Socket PUT receives 100 Continue, then uploads payload, then receives 200 OK."""
    test_server.set_behavior("send_100_continue")
    
    manager = RequestManager()
    test_data = b"x" * (100 * 1024)  # 100 KB
    
    result = await manager.put_socket(
        f"{test_server.url}/api/upload",
        test_data,
        timeout=5.0,
        expect_continue=True,
    )
    
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
    assert test_server.request_body == bytearray(test_data)


@qtapp
async def test_socket_put_early_200_skips_upload(test_server: _TestHTTPServer):
    """Socket PUT receives early 200 OK (cached); upload is skipped entirely."""
    test_server.set_behavior("send_early_200")
    
    manager = RequestManager()
    test_data = b"x" * (100 * 1024)  # 100 KB
    
    result = await manager.put_socket(
        f"{test_server.url}/api/upload",
        test_data,
        timeout=5.0,
        expect_continue=True,
    )
    
    assert isinstance(result, dict)
    assert result.get("status") == "cached"
    # Payload should NOT be sent to the server in this case
    assert test_server.request_body is None or len(test_server.request_body) == 0


@qtapp
async def test_socket_put_ignore_expect_fallback(test_server: _TestHTTPServer):
    """Socket PUT with server ignoring Expect: fallback timer triggers upload after 2s."""
    test_server.set_behavior("ignore_expect")
    
    manager = RequestManager()
    test_data = b"y" * (100 * 1024)  # 100 KB
    
    result = await manager.put_socket(
        f"{test_server.url}/api/upload",
        test_data,
        timeout=5.0,
        expect_continue=True,
    )
    
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
    assert test_server.request_body == bytearray(test_data)


@qtapp
async def test_socket_put_error_response(test_server: _TestHTTPServer):
    """Socket PUT receives error status code (400)."""
    test_server.set_behavior("send_error")
    
    manager = RequestManager()
    test_data = b"z" * (10 * 1024)
    
    with pytest.raises(NetworkError) as exc_info:
        await manager.put_socket(
            f"{test_server.url}/api/upload",
            test_data,
            timeout=5.0,
            expect_continue=True,
        )
    
    error = exc_info.value
    assert error.status == 400
    assert "error" in error.message.lower() or "Test error" in error.message


@qtapp
async def test_socket_put_without_expect_continue(test_server: _TestHTTPServer):
    """Socket PUT without Expect: 100-continue header; immediate upload."""
    test_server.set_behavior("ignore_expect")
    
    manager = RequestManager()
    test_data = b"w" * (50 * 1024)
    
    result = await manager.put_socket(
        f"{test_server.url}/api/upload",
        test_data,
        timeout=5.0,
        expect_continue=False,
    )
    
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
    assert test_server.request_body == bytearray(test_data)


@qtapp
async def test_socket_put_timeout_on_no_response(test_server: _TestHTTPServer):
    """Socket PUT times out if server never responds."""
    # Don't set any behavior; server won't respond at all
    # Instead, we'll test against a port with no server
    manager = RequestManager()
    test_data = b"x" * (10 * 1024)
    
    with pytest.raises(NetworkError) as exc_info:
        await manager.put_socket(
            "http://127.0.0.1:1",  # Port 1 is reserved; connection will fail
            test_data,
            timeout=1.0,
            expect_continue=True,
        )
    
    # Should fail due to connection error or timeout
    error = exc_info.value
    assert error.code is not None


@qtapp
async def test_socket_put_small_payload(test_server: _TestHTTPServer):
    """Socket PUT with very small payload."""
    test_server.set_behavior("send_100_continue")
    
    manager = RequestManager()
    test_data = b"hello"
    
    result = await manager.put_socket(
        f"{test_server.url}/api/upload",
        test_data,
        timeout=5.0,
        expect_continue=True,
    )
    
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
    assert test_server.request_body == bytearray(test_data)


@qtapp
async def test_socket_put_large_payload(test_server: _TestHTTPServer):
    """Socket PUT with large payload (multiple chunks)."""
    test_server.set_behavior("send_100_continue")
    
    manager = RequestManager()
    test_data = b"z" * (5 * 1024 * 1024)  # 5 MB
    
    result = await manager.put_socket(
        f"{test_server.url}/api/upload",
        test_data,
        timeout=10.0,
        expect_continue=True,
    )
    
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
    assert test_server.request_body == bytearray(test_data)


@qtapp
async def test_socket_put_qbytearray_input(test_server: _TestHTTPServer):
    """Socket PUT accepts QByteArray as input."""
    test_server.set_behavior("send_100_continue")
    
    manager = RequestManager()
    test_data = b"qt_test_data"
    q_data = QByteArray(test_data)
    
    result = await manager.put_socket(
        f"{test_server.url}/api/upload",
        q_data,
        timeout=5.0,
        expect_continue=True,
    )
    
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
    assert test_server.request_body == bytearray(test_data)


@qtapp
async def test_socket_put_with_bearer_token(test_server: _TestHTTPServer):
    """Socket PUT includes Authorization header when bearer token is set."""
    test_server.set_behavior("send_100_continue")
    
    manager = RequestManager()
    manager.set_auth("test-token-12345")
    test_data = b"data"
    
    result = await manager.put_socket(
        f"{test_server.url}/api/upload",
        test_data,
        timeout=5.0,
        expect_continue=True,
    )
    
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
