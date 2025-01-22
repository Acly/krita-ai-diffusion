import asyncio
from pathlib import Path
import pytest

from ai_diffusion import eventloop, resources
from ai_diffusion.api import WorkflowInput, WorkflowKind, LoraInput
from ai_diffusion.api import CheckpointInput, ImageInput, SamplingInput, ConditioningInput
from ai_diffusion.resources import ControlMode
from ai_diffusion.network import NetworkError
from ai_diffusion.image import Extent
from ai_diffusion.client import ClientEvent, resolve_arch
from ai_diffusion.comfy_client import ComfyClient, parse_url, websocket_url
from ai_diffusion.style import Arch, Style
from ai_diffusion.server import Server, ServerState, ServerBackend
from ai_diffusion.files import FileLibrary, File, FileFormat
from ai_diffusion.util import ensure
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


def make_default_work(size=512, steps=20):
    return WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput(default_checkpoint[Arch.sd15]),
        images=ImageInput.from_extent(Extent(size, size)),
        conditioning=ConditioningInput("a photo of a cat", "a photo of a dog"),
        sampling=SamplingInput("euler", "normal", cfg_scale=7.0, total_steps=steps),
    )


def test_connect_bad_url(qtapp, comfy_server):
    async def main():
        with pytest.raises(NetworkError):
            await ComfyClient.connect("bad_url")

    qtapp.run(main())


@pytest.mark.parametrize("cancel_point", ["after_enqueue", "after_start", "after_sampling"])
def test_cancel(qtapp, comfy_server, cancel_point):
    async def main():
        client = await ComfyClient.connect(comfy_server)
        job_id = None
        interrupted = False
        stage = 0

        async for msg in client.listen():
            if msg.event is ClientEvent.error:
                assert False, msg.error

            elif stage == 0:
                assert msg.event is not ClientEvent.finished
                assert msg.job_id == job_id or msg.job_id == ""
                if not job_id:
                    job_id = await client.enqueue(make_default_work(steps=200))
                    assert client.queued_count == 1
                if not interrupted:
                    if cancel_point == "after_enqueue":
                        await client.clear_queue()
                        interrupted = True
                    if cancel_point == "after_start" and msg.event is ClientEvent.progress:
                        await client.interrupt()
                        interrupted = True
                    if cancel_point == "after_sampling" and msg.progress > 0.1:
                        await client.interrupt()
                        interrupted = True
                if msg.event is ClientEvent.interrupted:
                    assert msg.job_id == job_id
                    assert not client.is_executing and client.queued_count == 0

                    job_id = await client.enqueue(make_default_work(size=320, steps=1))
                    stage = 1
                    assert client.queued_count == 1
                elif msg.event is ClientEvent.progress:
                    assert client.is_executing

            elif stage == 1:
                assert msg.event is not ClientEvent.interrupted
                assert msg.job_id == job_id or msg.job_id == ""
                if msg.event is ClientEvent.finished:
                    assert msg.images is not None and len(msg.images) > 0
                    assert msg.images[0].extent == Extent(320, 320)
                    break

        assert not client.is_executing and client.queued_count == 0

    qtapp.run(main())


def test_disconnect(qtapp, comfy_server):
    async def listen(client: ComfyClient):
        async for msg in client.listen():
            assert msg.event is ClientEvent.connected

    async def main():
        client = await ComfyClient.connect(comfy_server)
        task = eventloop._loop.create_task(listen(client))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client.is_executing and client.queued_count == 0

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


def check_client_info(client: ComfyClient):
    assert client.device_info.type in ["cpu", "cuda"]
    assert client.device_info.name != ""
    assert client.device_info.vram > 0

    assert len(client.models.checkpoints) > 0
    for filename, cp in client.models.checkpoints.items():
        assert cp.filename == filename
        assert cp.filename.startswith(cp.name)
        assert cp.format is FileFormat.checkpoint

    assert len(client.models.resources) >= len(resources.required_resource_ids)
    inpaint = client.models.for_arch(Arch.sd15).control[ControlMode.inpaint]
    assert inpaint and "inpaint" in inpaint


def check_resolve_sd_version(client: ComfyClient, arch: Arch):
    checkpoint = next(cp for cp in client.models.checkpoints.values() if cp.arch == arch)
    style = Style(Path("dummy"))
    style.architecture = Arch.auto
    style.checkpoints = [checkpoint.filename]
    assert resolve_arch(style, client) == arch
    assert resolve_arch(style, None) == arch


def test_info(pytestconfig, qtapp, comfy_server):
    async def main():
        client = await ComfyClient.connect(comfy_server)
        check_client_info(client)
        await client.refresh()
        check_client_info(client)
        check_resolve_sd_version(client, Arch.sd15)
        # check_resolve_sd_version(client, Arch.sdxl) # no SDXL checkpoint in default installation

    qtapp.run(main())


def test_upload_lora(qtapp, comfy_server, tmp_path: Path):
    lora_path = tmp_path / "test-lora.safetensors"
    lora_path.write_bytes(b"testdata" * 1024 * 1024)

    files = FileLibrary.instance()
    file = files.loras.add(File.local(lora_path, compute_hash=True))

    async def main():
        client = await ComfyClient.connect(comfy_server)
        if file.id in client.models.loras:
            client.models.loras.remove(file.id)

        input = make_default_work()
        assert input.models is not None
        input.models.loras = [LoraInput(file.id, 1.0, storage_id=ensure(file.hash))]

        task = asyncio.get_running_loop().create_task(client.upload_loras(input, "JOB-ID"))
        upload_progress = 0
        async for msg in client.listen():
            if msg.event is ClientEvent.upload:
                assert msg.job_id == "JOB-ID"
                assert msg.progress >= upload_progress
                upload_progress = msg.progress
                if upload_progress == 1.0:
                    break

        await task
        assert file.id in client.models.loras

    qtapp.run(main())
