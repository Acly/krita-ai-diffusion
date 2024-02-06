import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
from ai_diffusion import util
from ai_diffusion.util import batched, sanitize_prompt, find_unused_path, get_path_dict, ZipFile


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


def test_path_dict():
    # test with mixed forward/backward slashes only works on windows
    bs = "\\" if util.is_windows else "/"
    paths = [
        "f1.txt",
        "f2.txxt",
        "d1/f1.txt",
        f"d2{bs}f1.txt",
        f"d3/d4{bs}d5/f1.txt",
        "d1/f2.txt",
        str(Path("w1/f1.txt")),
        str(Path("w1/noext")),
        str(Path("w1/w2/.noext")),
    ]
    expected = {
        "f1.txt": "f1.txt",
        "f2.txxt": "f2.txxt",
        "d1": {"f1.txt": "d1/f1.txt", "f2.txt": "d1/f2.txt"},
        "d2": {"f1.txt": f"d2{bs}f1.txt"},
        "d3": {"d4": {"d5": {"f1.txt": f"d3/d4{bs}d5/f1.txt"}}},
        "w1": {"f1.txt": paths[-3], "noext": paths[-2], "w2": {".noext": paths[-1]}},
    }
    assert get_path_dict(paths) == expected


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
