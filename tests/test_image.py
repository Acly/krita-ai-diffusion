import pytest
import numpy as np
from PyQt5.QtGui import QImage, qRgba
from PyQt5.QtCore import Qt, QByteArray
from ai_diffusion.image import Mask, Bounds, Extent, Image, ImageCollection


def test_extent_compare():
    assert (Extent(4, 3) < Extent(4, 4)) == True
    assert (Extent(3, 4) < Extent(4, 4)) == True
    assert (Extent(4, 4) < Extent(4, 4)) == False


def create_test_image(w, h):
    img = QImage(w, h, QImage.Format_ARGB32)
    for y in range(h):
        for x in range(w):
            img.setPixel(x, y, qRgba(x, y, 0, 255))
    return Image(img)


def test_image_rgba():
    img = create_test_image(2, 5)
    assert img.extent == Extent(2, 5)
    assert img.is_rgba and not img.is_mask


def test_image_mask():
    qimg = QImage(2, 5, QImage.Format_Grayscale8)
    qimg.fill(123)
    img = Image(qimg)
    assert img.extent == Extent(2, 5)
    assert img.is_mask and not img.is_rgba
    assert img.pixel(1, 1) == 123


def test_base64():
    img = create_test_image(15, 21)
    encoded = img.to_base64()
    decoded = Image.from_base64(encoded)
    assert img == decoded


def test_image_make_opaque():
    img = Image(QImage(2, 2, QImage.Format_ARGB32))
    img.set_pixel(0, 0, (0, 0, 0, 0))
    img.set_pixel(1, 0, (0, 0, 0, 155))
    img.set_pixel(0, 1, (42, 42, 42, 255))
    img.make_opaque(Qt.GlobalColor.white)
    assert (
        img.pixel(0, 0) == (255, 255, 255, 255)
        and img.pixel(1, 0) == (100, 100, 100, 255)
        and img.pixel(0, 1) == (42, 42, 42, 255)
    )


def test_image_to_array():
    img = create_test_image(2, 2)
    expected = np.array(
        [[[0, 0, 0, 1], [0, 0, 1 / 255, 1]], [[0, 1 / 255, 0, 1], [0, 1 / 255, 1 / 255, 1]]],
        np.float32,
    )
    assert np.all(np.isclose(img.to_array(), expected))


def test_image_compare():
    img1 = create_test_image(2, 2)
    img2 = create_test_image(2, 2)
    assert Image.compare(img1, img2) < 0.0001


def test_image_collection_each():
    col = ImageCollection([create_test_image(2, 2), create_test_image(2, 2)])
    col.each(lambda img: img.set_pixel(0, 0, (42, 42, 42, 42)))
    assert col[0].pixel(0, 0) == (42, 42, 42, 42) and col[1].pixel(0, 0) == (42, 42, 42, 42)


def test_image_collection_map():
    col = ImageCollection([create_test_image(2, 2), create_test_image(2, 2)])
    sub = col.map(lambda img: Image.crop(img, Bounds(0, 0, 2, 1)))
    assert sub[0].extent == (2, 1) and sub[1].extent == (2, 1)


def test_pad_bounds():
    bounds = Bounds(3, 1, 5, 9)
    result = Bounds.pad(bounds, 2, multiple=1)
    assert result == Bounds(1, -1, 9, 13)


def test_pad_bounds_multiple():
    bounds = Bounds(3, 2, 5, 9)
    result = Bounds.pad(bounds, 0, multiple=4)
    assert result == Bounds(2, 1, 8, 12)


def test_pad_bounds_min_size():
    bounds = Bounds(3, 2, 5, 9)
    result = Bounds.pad(bounds, 2, min_size=10, multiple=1)
    assert result == Bounds(1, 0, 10, 13)


def test_pad_square():
    bounds = Bounds(0, 0, 8, 2)
    result = Bounds.pad(bounds, 2, square=True, multiple=1)
    assert result == Bounds(0, -2, 8, 6)


@pytest.mark.parametrize(
    "input,expected",
    [
        (Bounds(-1, 3, 5, 9), Bounds(0, 1, 4, 9)),
        (Bounds(-1, 3, 5, 9), Bounds(0, 1, 4, 9)),
        (Bounds(2, 3, 2, 5), Bounds(2, 3, 2, 5)),
    ],
)
def test_clamp_bounds(input, expected):
    result = Bounds.clamp(input, Extent(4, 10))
    assert result == expected


@pytest.mark.parametrize(
    "input,target,expected",
    [
        (Bounds(0, 0, 1, 2), Bounds(0, 0, 2, 2), Bounds(0, 0, 1, 2)),
        (Bounds(0, 0, 1, 2), Bounds(0, 0, 1, 1), Bounds(0, 0, 1, 1)),
        (Bounds(2, 4, 1, 2), Bounds(1, 4, 2, 2), Bounds(1, 0, 1, 2)),
        (Bounds(0, 0, 1, 2), Bounds(1, 4, 2, 2), Bounds(0, 0, 1, 2)),
        (Bounds(3, 4, 5, 6), Bounds(1, 1, 2, 3), Bounds(0, 0, 2, 3)),
    ],
)
def test_bounds_apply_crop(input, target, expected):
    result = Bounds.apply_crop(input, target)
    assert result == expected


@pytest.mark.parametrize(
    "input,min_size,max_extent,expected",
    [
        (Bounds(0, 0, 1, 2), 2, Extent(4, 4), Bounds(0, 0, 2, 2)),
        (Bounds(3, 4, 5, 6), 8, Extent(12, 12), Bounds(3, 4, 8, 8)),
        (Bounds(3, 4, 5, 6), 8, Extent(10, 10), Bounds(2, 2, 8, 8)),
        (Bounds(3, 4, 5, 6), 8, Extent(8, 7), None),
        (Bounds(3, 4, 5, 6), 5, Extent(8, 7), Bounds(3, 1, 5, 6)),
    ],
)
def test_bounds_minimum_size(input, min_size, max_extent, expected):
    result = Bounds.minimum_size(input, min_size, max_extent)
    assert result == expected


def test_mask_to_image():
    data = QByteArray(b"\x00\x01\x02\xff")
    mask = Mask(Bounds(0, 0, 2, 2), data)
    img = mask.to_image(Extent(2, 2))
    assert (
        img.pixel(0, 0) == 0
        and img.pixel(1, 0) == 1
        and img.pixel(0, 1) == 2
        and img.pixel(1, 1) == 255
    )


def test_mask_to_image_offset():
    data = QByteArray(b"\x00\x01\x02\xff")
    mask = Mask(Bounds(1, 2, 2, 2), data)
    img = mask.to_image(Extent(4, 4))
    assert (
        img.pixel(1, 2) == 0
        and img.pixel(2, 2) == 1
        and img.pixel(1, 3) == 2
        and img.pixel(2, 3) == 255
    )


def test_mask_to_image_no_extent():
    data = QByteArray(b"\x00\x01\x02\xff")
    mask = Mask(Bounds(1, 2, 2, 2), data)
    img = mask.to_image()
    assert img.width == 2 and img.height == 2
    assert (
        img.pixel(0, 0) == 0
        and img.pixel(1, 0) == 1
        and img.pixel(0, 1) == 2
        and img.pixel(1, 1) == 255
    )


def test_mask_rectangle():
    mask = Mask.rectangle(Bounds(1, 2, 6, 5), feather=2)
    # fmt: off
    assert mask.to_array() == [
        127, 159, 191, 191, 159, 127,
        159, 191, 223, 223, 191, 159,
        191, 223, 255, 255, 223, 191,
        159, 191, 223, 223, 191, 159,
        127, 159, 191, 191, 159, 127]
    # fmt: on


def test_downscale():
    img = create_test_image(12, 8)
    result = Image.scale(img, Extent(6, 4))
    assert result.width == 6 and result.height == 4
