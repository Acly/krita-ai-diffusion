import asyncio
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from ai_diffusion import util
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
