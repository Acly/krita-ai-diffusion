import asyncio
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from ai_diffusion import util
from ai_diffusion.util import ZipFile
from ai_diffusion.util import batched, ensure, sanitize_prompt, find_unused_path


def test_batched():
    iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n = 3
    expected_output = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    assert list(batched(iterable, n)) == expected_output


def test_sanitize_prompt():
    assert sanitize_prompt("") == "no prompt"
    assert sanitize_prompt("a" * 50) == "a" * 40
    assert sanitize_prompt("bla\nblub\n<x:2:4> (neg) [pos]") == "blablubx24 neg pos"


def test_unused_path():
    with TemporaryDirectory() as dir:
        file = Path(dir) / "test.txt"
        assert find_unused_path(file) == file
        file.touch()
        assert find_unused_path(file) == Path(dir) / "test-1.txt"
        (Path(dir) / "test-1.txt").touch()
        assert find_unused_path(file) == Path(dir) / "test-2.txt"


def test_create_process():
    async def main():
        process = await util.create_process(sys.executable, "--version")
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
