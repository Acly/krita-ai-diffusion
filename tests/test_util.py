import asyncio
import struct
import sys
import zlib
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

        def test_format_like_automatic1111_basic():
            class Bounds:
                width = 512
                height = 768

            class Params:
                seed = 12345
                bounds = Bounds()
                metadata = {
                    "prompt": "A cat",
                    "negative_prompt": "dog",
                    "sampler": "Euler - euler_a (20 / 7.5)",
                    "checkpoint": "model.ckpt",
                    "strength": 0.8,
                    "loras": [],
                }

            result = util._format_like_automatic1111(Params())
            assert "Prompt: A cat" in result
            assert "Negative prompt: dog" in result
            assert (
                "Steps: 20, Sampler: euler_a, CFG scale: 7.5, Seed: 12345, Size: 512x768, Model hash: unknown, Model: model.ckpt, Denoising strength: 0.8"
                in result
            )


def test_format_like_automatic1111_sampler_unmatched():
    class Bounds:
        width = 256
        height = 256

    class Params:
        seed = 42
        bounds = Bounds()
        metadata = {
            "prompt": "Test",
            "negative_prompt": "",
            "sampler": "UnknownSampler",
            "checkpoint": "unknown.ckpt",
            "loras": [],
        }

    result = util._format_like_automatic1111(Params())
    assert "Sampler: UnknownSampler" in result
    assert "Steps: Unknown" in result
    assert "CFG scale: Unknown" in result


def test_format_like_automatic1111_loras_dict_and_tuple():
    class Bounds:
        width = 128
        height = 128

    class Params:
        seed = 1
        bounds = Bounds()
        metadata = {
            "prompt": "Prompt",
            "negative_prompt": "",
            "sampler": "Euler - euler_a (10 / 5.0)",
            "checkpoint": "loramodel.ckpt",
            "loras": [{"name": "lora1", "weight": 0.7}, ("lora2", 0.5), ["lora3", 0.9]],
        }

    result = util._format_like_automatic1111(Params())
    assert "<lora:lora1:0.7>" in result
    assert "<lora:lora2:0.5>" in result
    assert "<lora:lora3:0.9>" in result


def test_format_like_automatic1111_strength_none_and_one():
    class Bounds:
        width = 64
        height = 64

    class ParamsNone:
        seed = 0
        bounds = Bounds()
        metadata = {
            "prompt": "Prompt",
            "negative_prompt": "",
            "sampler": "Euler - euler_a (5 / 2.0)",
            "checkpoint": "model.ckpt",
            "strength": None,
            "loras": [],
        }

    class ParamsOne:
        seed = 0
        bounds = Bounds()
        metadata = {
            "prompt": "Prompt",
            "negative_prompt": "",
            "sampler": "Euler - euler_a (5 / 2.0)",
            "checkpoint": "model.ckpt",
            "strength": 1.0,
            "loras": [],
        }

    result_none = util._format_like_automatic1111(ParamsNone())
    result_one = util._format_like_automatic1111(ParamsOne())
    assert "Denoising strength" not in result_none
    assert "Denoising strength" not in result_one


def test_format_like_automatic1111_missing_metadata_fields():
    class Bounds:
        width = 100
        height = 200

    class Params:
        seed = 999
        bounds = Bounds()
        metadata = {}

    result = util._format_like_automatic1111(Params())
    assert "Prompt: " in result
    assert "Negative prompt: " in result
    assert "Steps: Unknown" in result
    assert "Sampler: " in result
    assert "CFG scale: Unknown" in result
    assert "Seed: 999" in result
    assert "Size: 100x200" in result
    assert "Model: Unknown" in result


def test_add_png_itxt_valid(tmp_path):
    # Create a minimal valid PNG file
    png_header = b"\x89PNG\r\n\x1a\n"
    ihdr_chunk = (
        b"\x00\x00\x00\rIHDR" + b"\x00\x00\x00\x01" + b"\x00\x00\x00\x01" + b"\x08\x02\x00\x00\x00"
    )
    ihdr_crc = struct.pack(
        ">I",
        zlib.crc32(b"IHDR" + b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00") & 0xFFFFFFFF,
    )
    iend_chunk = b"\x00\x00\x00\x00IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    png_data = png_header + ihdr_chunk + ihdr_crc + iend_chunk

    file_path = tmp_path / "test.png"
    file_path.write_bytes(png_data)

    util._add_png_itxt(file_path, "testkey", "testvalue")

    # Check that the file still starts with PNG header
    data = file_path.read_bytes()
    assert data.startswith(png_header)
    # Check that iTXt chunk is present
    assert b"iTXt" in data
    assert b"testkey" in data
    assert b"testvalue" in data


def test_add_png_itxt_invalid(tmp_path):
    # Not a PNG file
    file_path = tmp_path / "not_png.txt"
    file_path.write_text("not a png")
    try:
        util._add_png_itxt(file_path, "key", "value")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Not a valid PNG file" in str(e)
