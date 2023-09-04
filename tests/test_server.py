from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtNetwork import QNetworkAccessManager
import pytest
import shutil

from ai_diffusion import network, Server, ServerState, ServerBackend

test_dir = Path(__file__).parent / ".server"
comfy_dir = Path("C:/Dev/ComfyUI")


@pytest.fixture(scope="session", autouse=True)
def clear_downloads():
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(exist_ok=True)


def test_download(qtapp):
    async def main():
        net = QNetworkAccessManager()
        with TemporaryDirectory() as tmp:
            url = "https://github.com/Acly/krita-ai-diffusion/archive/refs/tags/v0.1.0.zip"
            path = Path(tmp) / "test.zip"
            async for progress in network.download(net, url, path):
                assert progress == -1 or (progress >= 0 and progress <= 1)
            assert path.exists() and path.stat().st_size > 0

    qtapp.run(main())


def test_install_and_run(qtapp, pytestconfig):
    """Test installing and running ComfyUI server from scratch.
    Downloads ~5GB of data, so it does not run by default.
    """
    if not pytestconfig.getoption("--test-install"):
        pytest.skip("Only runs with --test-install")

    server = Server(str(test_dir))
    server.backend = ServerBackend.cpu
    assert server.state in [
        ServerState.missing_comfy,
        ServerState.missing_python,
        ServerState.missing_resources,
    ]

    def log(msg):
        print(msg)

    async def main():
        await server.install(log)
        assert server.state is ServerState.stopped

        url = await server.start(log)
        assert server.state is ServerState.running
        assert url == "127.0.0.1:8188"

        await server.stop()
        assert server.state is ServerState.stopped

    qtapp.run(main())


def test_run_external(qtapp):
    if not comfy_dir.exists():
        pytest.skip("External ComfyUI installation not found")

    server = Server(comfy_dir)
    server.backend = ServerBackend.cpu
    assert server.state in [ServerState.stopped, ServerState.missing_resources]

    def log(msg):
        print(msg)

    async def main():
        url = await server.start(log)
        assert server.state is ServerState.running
        assert url == "127.0.0.1:8188"

        await server.stop()
        assert server.state is ServerState.stopped

    qtapp.run(main())
