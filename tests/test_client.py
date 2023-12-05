import asyncio
from pathlib import Path
import pytest

from ai_diffusion import eventloop
from ai_diffusion.resources import ControlMode
from ai_diffusion.comfyworkflow import ComfyWorkflow
from ai_diffusion.network import NetworkError
from ai_diffusion.image import Image, Extent
from ai_diffusion.client import Client, ClientEvent, parse_url, resolve_sd_version, websocket_url
from ai_diffusion.style import SDVersion, Style
from ai_diffusion.server import Server, ServerState, ServerBackend
from .config import server_dir, default_checkpoint


@pytest.fixture(scope="session")
def comfy_server(qtapp):
    server = Server(str(server_dir))
    server.backend = ServerBackend.cpu
    assert server.state is ServerState.stopped, (
        f"Expected server installation at {server_dir}. To create the default installation run"
        " `pytest tests/test_server.py --test-install`"
    )
    yield qtapp.run(server.start(port=8189))
    qtapp.run(server.stop())


def make_default_workflow(steps=20):
    w = ComfyWorkflow()
    model, clip, vae = w.load_checkpoint(default_checkpoint[SDVersion.sd15])
    positive = w.clip_text_encode(clip, "a photo of a cat")
    negative = w.clip_text_encode(clip, "a photo of a dog")
    latent_image = w.empty_latent_image(512, 512)
    latent_result = w.ksampler_advanced(model, positive, negative, latent_image, steps=steps)
    result_image = w.vae_decode(vae, latent_result)
    w.send_image(result_image)
    return w


def make_trivial_workflow():
    img = Image.create(Extent(16, 16))
    w = ComfyWorkflow()
    w.send_image(w.load_image(img))
    return w


def test_connect_bad_url(qtapp, comfy_server):
    async def main():
        with pytest.raises(NetworkError):
            await Client.connect("bad_url")

    qtapp.run(main())


@pytest.mark.parametrize("cancel_point", ["after_enqueue", "after_start", "after_sampling"])
def test_cancel(qtapp, comfy_server, cancel_point):
    async def main():
        client = await Client.connect(comfy_server)
        job_id = None
        interrupted = False
        stage = 0

        async for msg in client.listen():
            if stage == 0:
                assert msg.event is not ClientEvent.finished
                assert msg.job_id == job_id or msg.job_id == ""
                if not job_id:
                    job_id = await client.enqueue(make_default_workflow(steps=200))
                    assert client.queued_count == 1
                if not interrupted:
                    if cancel_point == "after_enqueue":
                        await client.interrupt()
                        interrupted = True
                    if cancel_point == "after_start" and msg.event is ClientEvent.progress:
                        await client.interrupt()
                        interrupted = True
                    if cancel_point == "after_sampling" and msg.progress > 0.1:
                        await client.interrupt()
                        interrupted = True
                if msg.event is ClientEvent.interrupted:
                    assert msg.job_id == job_id
                    assert client.is_executing == False and client.queued_count == 0

                    job_id = await client.enqueue(make_trivial_workflow())
                    stage = 1
                    assert client.queued_count == 1
                elif msg.event is ClientEvent.progress:
                    assert client.is_executing

            elif stage == 1:
                assert msg.event is not ClientEvent.interrupted
                assert msg.job_id == job_id or msg.job_id == ""
                if msg.event is ClientEvent.finished:
                    assert msg.images is not None and len(msg.images) > 0
                    assert msg.images[0].extent == Extent(16, 16)
                    break

        assert client.is_executing == False and client.queued_count == 0

    qtapp.run(main())


def test_disconnect(qtapp, comfy_server):
    async def listen(client: Client):
        async for msg in client.listen():
            assert msg.event is ClientEvent.connected

    async def main():
        client = await Client.connect(comfy_server)
        task = eventloop._loop.create_task(listen(client))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.is_executing == False and client.queued_count == 0

    qtapp.run(main())


@pytest.mark.parametrize(
    "url,expected_http,expected_ws",
    [
        ("http://localhost:8000", "http://localhost:8000", "ws://localhost:8000"),
        ("http://localhost:8000/", "http://localhost:8000", "ws://localhost:8000"),
        ("http://localhost:8000/foo", "http://localhost:8000/foo", "ws://localhost:8000/foo"),
        ("http://127.0.0.1:1234", "http://127.0.0.1:1234", "ws://127.0.0.1:1234"),
        ("localhost:8000", "http://localhost:8000", "ws://localhost:8000"),
        ("https://localhost:8000", "https://localhost:8000", "wss://localhost:8000"),
    ],
)
def test_parse_url(url, expected_http, expected_ws):
    parsed = parse_url(url)
    assert parsed == expected_http and websocket_url(parsed) == expected_ws


def check_client_info(client: Client):
    assert client.device_info.type in ["cpu", "cuda"]
    assert client.device_info.name != ""
    assert client.device_info.vram > 0

    assert len(client.checkpoints) > 0
    for filename, cp in client.checkpoints.items():
        assert cp.filename == filename
        assert cp.filename.startswith(cp.name)
        assert cp.is_inpaint == ("inpaint" in cp.name.lower())
        assert cp.is_refiner == ("refiner" in cp.name.lower())

    assert len(client.control_model) > 0
    inpaint = client.control_model[ControlMode.inpaint][SDVersion.sd15]
    assert inpaint and "inpaint" in inpaint


def check_resolve_sd_version(client: Client, sd_version: SDVersion):
    checkpoint = next(cp for cp in client.checkpoints.values() if cp.sd_version == sd_version)
    style = Style(Path("dummy"))
    style.sd_version = SDVersion.auto
    style.sd_checkpoint = checkpoint.filename
    assert resolve_sd_version(style, client) == sd_version
    assert resolve_sd_version(style, None) == sd_version


def test_info(pytestconfig, qtapp, comfy_server):
    async def main():
        client = await Client.connect(comfy_server)
        check_client_info(client)
        await client.refresh()
        check_client_info(client)
        check_resolve_sd_version(client, SDVersion.sd15)
        # check_resolve_sd_version(client, SDVersion.sdxl) # no SDXL checkpoint in default installation

    qtapp.run(main())
