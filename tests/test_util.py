import pytest
from ai_diffusion.util import batched


def test_batched():
    iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n = 3
    expected_output = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    assert list(batched(iterable, n)) == expected_output
