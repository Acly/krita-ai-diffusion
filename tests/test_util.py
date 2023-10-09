import pytest
from ai_diffusion import Extent
from ai_diffusion.util import compute_batch_size, batched


@pytest.mark.parametrize(
    "extent, min_size, max_batches, expected",
    [
        (Extent(512, 512), 512, 4, 4),
        (Extent(512, 512), 512, 6, 6),
        (Extent(1024, 512), 512, 8, 4),
        (Extent(1024, 1024), 512, 8, 2),
        (Extent(2048, 1024), 512, 6, 1),
        (Extent(256, 256), 512, 4, 4),
    ],
)
def test_compute_batch_size(extent, min_size, max_batches, expected):
    assert compute_batch_size(extent, min_size, max_batches) == expected


def test_batched():
    iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n = 3
    expected_output = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    assert list(batched(iterable, n)) == expected_output
