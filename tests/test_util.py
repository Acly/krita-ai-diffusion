import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
from ai_diffusion.util import batched, get_path_dict, ZipFile


def test_batched():
    iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n = 3
    expected_output = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    assert list(batched(iterable, n)) == expected_output


def test_path_dict():
    paths = [
        "f1.txt",
        "f2.txxt",
        "d1/f1.txt",
        "d2/f1.txt",
        "d3/d4/d5/f1.txt",
        "d1/f2.txt",
        str(Path("w1/f1.txt")),
        str(Path("w1/noext")),
        str(Path("w1/w2/.noext")),
    ]
    expected = {
        "f1.txt": None,
        "f2.txxt": None,
        "d1": {"f1.txt": None, "f2.txt": None},
        "d2": {"f1.txt": None},
        "d3": {"d4": {"d5": {"f1.txt": None}}},
        "w1": {"f1.txt": None, "noext": None, "w2": {".noext": None}},
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
