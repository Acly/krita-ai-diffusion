import pytest
from PyQt5.QtGui import QImage, qRgba
from PyQt5.QtCore import QByteArray
from ai_tools import image, Mask, Bounds, Extent, Image, ImageCollection


def create_test_image(w, h):
    img = QImage(w, h, QImage.Format_ARGB32)
    for y in range(h):
        for x in range(w):
            img.setPixel(x, y, qRgba(x, y, 0, 255))
    return Image(img)


def test_base64():
    img = create_test_image(15, 21)
    encoded = img.to_base64()
    decoded = Image.from_base64(encoded)
    assert img == decoded


def test_image_collection_each():
    col = ImageCollection([create_test_image(2, 2), create_test_image(2, 2)])
    col.each(lambda img: img.set_pixel(0, 0, (42, 42, 42, 42)))
    assert col[0].pixel(0, 0) == (42, 42, 42, 42) and col[1].pixel(0, 0) == (42, 42, 42, 42)


def test_image_collection_map():
    col = ImageCollection([create_test_image(2, 2), create_test_image(2, 2)])
    sub = col.map(lambda img: Image.sub_region(img, Bounds(0, 0, 2, 1)))
    assert sub[0].extent == (2, 1) and sub[1].extent == (2, 1)


def test_pad_bounds():
    bounds = Bounds(3, 1, 5, 9)
    result = Bounds.pad(bounds, 2, 1)
    assert result == Bounds(1, -1, 9, 13)


def test_pad_bounds_multiple():
    bounds = Bounds(3, 2, 5, 9)
    result = Bounds.pad(bounds, 0, 4)
    assert result == Bounds(3, 2, 8, 12)


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


def test_mask_to_image():
    data = QByteArray(b"\x00\x01\x02\xff")
    mask = Mask(Bounds(0, 0, 2, 2), data)
    img = mask.to_image(Extent(2, 2))
    assert (
        img.pixel(0, 0) == (0, 0, 0, 255)
        and img.pixel(1, 0) == (1, 1, 1, 255)
        and img.pixel(0, 1) == (2, 2, 2, 255)
        and img.pixel(1, 1) == (255, 255, 255, 255)
    )


def test_mask_to_image_offset():
    data = QByteArray(b"\x00\x01\x02\xff")
    mask = Mask(Bounds(1, 2, 2, 2), data)
    img = mask.to_image(Extent(4, 4))
    assert (
        img.pixel(1, 2) == (0, 0, 0, 255)
        and img.pixel(2, 2) == (1, 1, 1, 255)
        and img.pixel(1, 3) == (2, 2, 2, 255)
        and img.pixel(2, 3) == (255, 255, 255, 255)
    )


def test_mask_to_image_no_extent():
    data = QByteArray(b"\x00\x01\x02\xff")
    mask = Mask(Bounds(1, 2, 2, 2), data)
    img = mask.to_image()
    assert img.width == 2 and img.height == 2
    assert (
        img.pixel(0, 0) == (0, 0, 0, 255)
        and img.pixel(1, 0) == (1, 1, 1, 255)
        and img.pixel(0, 1) == (2, 2, 2, 255)
        and img.pixel(1, 1) == (255, 255, 255, 255)
    )


def test_apply_mask():
    data = QByteArray(b"\x00\x01\x02\xff")
    mask = Mask(Bounds(1, 2, 2, 2), data)
    img = create_test_image(4, 4)
    Mask.apply(img, mask)
    assert (
        img.pixel(0, 0) == (0, 0, 0, 0)
        and img.pixel(1, 2) == (1, 2, 0, 0)
        and img.pixel(2, 2) == (2, 2, 0, 1)
        and img.pixel(1, 3) == (1, 3, 0, 2)
        and img.pixel(2, 3) == (2, 3, 0, 255)
        and img.pixel(3, 3) == (3, 3, 0, 0)
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
