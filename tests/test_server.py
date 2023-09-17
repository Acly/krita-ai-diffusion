from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtNetwork import QNetworkAccessManager
import pytest
import shutil

from ai_diffusion import network, Server, ServerState, ServerBackend, InstallationProgress

test_dir = Path(__file__).parent / ".server"
comfy_dir = Path("C:/Dev/ComfyUI")


@pytest.fixture(scope="session", autouse=True)
def clear_downloads():
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(exist_ok=True)


@pytest.mark.parametrize("mode", ["from_scratch", "resume"])
def test_download(qtapp, mode):
    async def main():
        net = QNetworkAccessManager()
        with TemporaryDirectory() as tmp:
            url = "https://github.com/Acly/krita-ai-diffusion/archive/refs/tags/v0.1.0.zip"
            path = Path(tmp) / "test.zip"
            if mode == "resume":
                part = Path(tmp) / "test.zip.part"
                part.touch()
                part.write_bytes(b"1234567890")
            got_finished = False
            async for progress in network.download(net, url, path):
                if progress and progress.total > 0:
                    assert progress.value >= 0 and progress.value <= 1
                    assert progress.received <= progress.total
                    assert progress.speed >= 0
                    got_finished = got_finished or progress.value == 1
                elif progress and progress.total == 0:
                    assert progress.value == -1
            assert got_finished and path.exists() and path.stat().st_size > 0

    qtapp.run(main())


def test_install_and_run(qtapp, pytestconfig):
    """Test installing and running ComfyUI server from scratch.
    Downloads ~5GB of data, so it does not run by default.
    """
    if not pytestconfig.getoption("--test-install"):
        pytest.skip("Only runs with --test-install")

    server = Server(str(test_dir))
    server.backend = ServerBackend.cpu
    assert server.state in [ServerState.not_installed, ServerState.missing_resources]

    def handle_progress(report: InstallationProgress):
        assert (
            report.progress is None
            or report.progress.value == -1
            or report.progress.value >= 0
            and report.progress.value <= 1
        )
        assert report.stage != ""
        if report.progress is None:
            print(report.stage, report.message)

    async def main():
        await server.install(handle_progress)
        assert server.state is ServerState.stopped

        url = await server.start()
        assert server.state is ServerState.running
        assert url == "127.0.0.1:8188"

        await server.stop()
        assert server.state is ServerState.stopped

    qtapp.run(main())


def test_run_external(qtapp, pytestconfig):
    if not pytestconfig.getoption("--test-install"):
        pytest.skip("Only runs with --test-install")
    if not comfy_dir.exists():
        pytest.skip("External ComfyUI installation not found")

    server = Server(comfy_dir)
    server.backend = ServerBackend.cpu
    assert server.state in [ServerState.stopped, ServerState.missing_resources]

    async def main():
        url = await server.start()
        assert server.state is ServerState.running
        assert url == "127.0.0.1:8188"

        await server.stop()
        assert server.state is ServerState.stopped

    qtapp.run(main())
