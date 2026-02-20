import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_diffusion.platform_tools import ZipFile, create_process, get_cuda_devices
from ai_diffusion.util import ensure


def test_create_process():
    async def main():
        process = await create_process(sys.executable, "--version")
        output = await ensure(process.stdout).read()
        assert output.decode().startswith("Python")
        await process.wait()
        assert process.returncode == 0

    asyncio.run(main())


def test_long_path_zip_file():
    with TemporaryDirectory() as dir:
        file = Path(dir) / "test.zip"
        with ZipFile(file, "w") as zip:
            zip.writestr("test.txt", "test")
            zip.writestr("test2.txt", "test2")
        long_path = Path(dir) / ("l" + "o" * 150 + "ng") / ("l" + "o" * 150 + "ng")
        with ZipFile(file, "r") as zip:
            zip.extractall(long_path)
        assert (long_path / "test.txt").read_text() == "test"
        assert (long_path / "test2.txt").read_text() == "test2"


def test_cuda_devices():
    devices = get_cuda_devices()
    assert len(devices) == 0 or devices[0][0] >= 3
