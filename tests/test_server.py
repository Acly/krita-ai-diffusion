from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtNetwork import QNetworkAccessManager
import asyncio
import pytest
import shutil

from ai_diffusion import network, server, resources
from ai_diffusion.style import Arch
from ai_diffusion.server import Server, ServerState, ServerBackend, InstallationProgress
from ai_diffusion.resources import VerificationState
from .config import test_dir, server_dir

workload_sd15 = [p.name for p in resources.required_models if p.arch is Arch.sd15]
workload_sd15 += [resources.default_checkpoints[0].name]


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


def clear_test_server():
    if server_dir.exists():
        shutil.rmtree(server_dir, ignore_errors=True)
    server_dir.mkdir(exist_ok=True)


def test_install_and_run(qtapp, pytestconfig, local_download_server):
    """Test installing and running ComfyUI server from scratch.
    * Takes a while, only runs with --test-install
    * Starts and downloads from local file server instead of huggingface/civitai
      * Required to run `scripts/download_models.py -m scripts/downloads` to download models once
      * Remove `local_download_server` fixture to download from original urls
    * Also tests upgrading server from "previous" version
      * In this case it's the same version, but it removes & re-installs anyway
    """
    if not pytestconfig.getoption("--test-install"):
        pytest.skip("Only runs with --test-install")

    clear_test_server()

    server = Server(str(server_dir))
    server.backend = ServerBackend.cpu
    assert server.state in [ServerState.not_installed, ServerState.missing_resources]

    last_stage = ""

    def handle_progress(report: InstallationProgress):
        nonlocal last_stage
        assert (
            not isinstance(report.progress, network.DownloadProgress)
            or report.progress.value == -1
            or (report.progress.value >= 0 and report.progress.value <= 1)
        )
        assert report.stage != ""
        if report.progress is None and report.stage != last_stage:
            last_stage = report.stage
            print(report.stage)

    async def main():
        await server.install(handle_progress)
        assert server.state is ServerState.missing_resources
        await server.download_required(handle_progress)
        assert server.state is ServerState.missing_resources
        await server.download(workload_sd15, handle_progress)
        assert server.state is ServerState.stopped and server.version == resources.version

        url = await server.start(port=8191)
        assert server.state is ServerState.running
        assert url == "127.0.0.1:8191"

        await server.stop()
        assert server.state is ServerState.stopped

        version_file = server_dir / ".version"
        assert version_file.exists()
        with version_file.open("w", encoding="utf-8") as f:
            f.write("1.0.42")
        server.check_install()
        assert server.upgrade_required
        await server.upgrade(handle_progress)
        assert server.state is ServerState.stopped and server.version == resources.version

    qtapp.run(main())


def test_run_external(qtapp, pytestconfig):
    if not pytestconfig.getoption("--test-install"):
        pytest.skip("Only runs with --test-install")
    if not (server_dir / "ComfyUI").exists():
        pytest.skip("ComfyUI installation not found")

    server = Server(str(server_dir))
    server.backend = ServerBackend.cpu
    assert server.has_python
    assert server.state in [ServerState.stopped, ServerState.missing_resources]

    async def main():
        url = await server.start(port=8192)
        assert server.state is ServerState.running
        assert url == "127.0.0.1:8192"

        await server.stop()
        assert server.state is ServerState.stopped

    qtapp.run(main())


def test_verify_and_fix(qtapp, pytestconfig, local_download_server):
    if not pytestconfig.getoption("--test-install"):
        pytest.skip("Only runs with --test-install")

    server = Server(str(server_dir))
    server.backend = ServerBackend.cpu
    # Requires test_install_and_run to setup the server
    assert server.state in [ServerState.stopped]

    model_file = resources.required_models[0]
    model_path = server_dir / model_file.files[0].path
    assert model_path.exists()

    # Break the model file
    with model_path.open("w", encoding="utf-8") as f:
        f.write("test")

    def handle_progress(report: InstallationProgress):
        print(report.stage, report.message)

    async def main():
        verify_result = await server.verify(handle_progress)
        assert len(verify_result) == 1
        mismatch = verify_result[0]
        assert mismatch.state is VerificationState.mismatch
        assert mismatch.file.path == model_file.files[0].path
        assert server.state is ServerState.stopped

        await server.fix_models(verify_result, handle_progress)
        fixed_result = await server.verify(handle_progress)
        assert len(fixed_result) == 0
        assert server.state is ServerState.stopped

    qtapp.run(main())


def test_uninstall(qtapp, pytestconfig, local_download_server):
    if not pytestconfig.getoption("--test-install"):
        pytest.skip("Only runs with --test-install")

    temp_server_dir = test_dir / "temp_server"
    if temp_server_dir.exists():
        shutil.rmtree(temp_server_dir, ignore_errors=True)
    server = Server(str(temp_server_dir))
    server.backend = ServerBackend.cpu
    assert server.state is ServerState.not_installed

    def handle_progress(report: InstallationProgress):
        print(report.stage, report.message)

    async def main():
        await server.install(handle_progress)
        await server.download_required(handle_progress)
        await server.uninstall(handle_progress)
        assert server.state is ServerState.not_installed
        assert (temp_server_dir / "models").exists()
        assert server.comfy_dir is None

        server.state = ServerState.stopped
        await server.uninstall(handle_progress, delete_models=True)
        assert server.state is ServerState.not_installed
        assert not temp_server_dir.exists()

    qtapp.run(main())


def test_install_if_missing(qtapp):
    installed = False

    async def install_func(target: Path):
        nonlocal installed
        installed = True
        target.mkdir()
        (target / "file").touch()

    async def main():
        nonlocal installed

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "test"
            await server.install_if_missing(target, install_func, target)
            assert installed and (target / "file").exists()

            installed = False
            await server.install_if_missing(target, install_func, target)
            assert not installed

    qtapp.run(main())


def test_try_install(qtapp):
    async def install_func(target: Path):
        target.mkdir()
        raise Exception("test")

    async def main():
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "test"
            with pytest.raises(Exception):
                await server.install_if_missing(target, install_func, target)
            assert not target.exists()

            target.mkdir()
            with pytest.raises(Exception):
                await server.try_install(target, install_func, target)
            assert target.exists()

    qtapp.run(main())


@pytest.mark.parametrize("scenario", ["default", "target-empty", "target-exists", "source-missing"])
def test_rename_extracted_folder(scenario):
    with TemporaryDirectory() as tmp:
        source = Path(tmp) / "test-sffx"
        if scenario != "source-missing":
            source.mkdir()
            (source / "file").touch()

        target = Path(tmp) / "test"
        if scenario in ["target-exists", "target-empty"]:
            target.mkdir()
        if scenario == "target-exists":
            (target / "file").touch()

        try:
            asyncio.run(server.rename_extracted_folder("Test", target, "sffx"))
            assert not source.exists()
            assert (target / "file").exists()
        except Exception:
            assert scenario in ["target-exists", "source-missing"]


@pytest.mark.parametrize("scenario", ["regular-file", "large-file", "model-file"])
def test_safe_remove_dir(scenario):
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test"
        path.mkdir()
        if scenario == "regular-file":
            (path / "file").touch()
        elif scenario == "large-file":
            large_file = path / "large_file"
            with large_file.open("wb") as f:
                f.write(b"0" * 1032)
        elif scenario == "model-file":
            (path / "model.safetensors").touch()
        try:
            server.safe_remove_dir(path, max_size=1024)
            assert scenario == "regular-file" and not path.exists()
        except Exception:
            assert scenario != "regular-file"


def test_python_version(qtapp):
    async def main():
        py, major, minor = await server.get_python_version(Path("python"))
        assert py.startswith("Python 3.")
        assert major is not None and minor is not None and major >= 3 and minor >= 9
        pip, major, minor = await server.get_python_version(Path("pip"))
        assert major is None and minor is None
        assert pip.startswith("pip ")

    qtapp.run(main())


common_errors = {
    "timeout": {
        "original": "pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='download.pytorch.org', port=443): Read timed out.",
        "expected": "Connection to download.pytorch.org timed out during download. Please make sure you have a stable internet connection and try again.",
    }
}


@pytest.mark.parametrize("errors", common_errors.values(), ids=common_errors.keys())
def test_common_errors(errors):
    assert server.parse_common_errors(errors["original"]) == errors["expected"]
