import pytest
import numpy as np
from PyQt5.QtGui import QImage, qRgba
from PyQt5.QtCore import Qt, QByteArray
from PIL import Image as PILImage
from ai_diffusion.image import Mask, Bounds, Extent, Image, ImageCollection
from .config import image_dir, result_dir, reference_dir


def test_extent_compare():
    assert Extent(4, 3) < Extent(4, 4)
    assert Extent(3, 4) < Extent(4, 4)
    assert not (Extent(4, 4) < Extent(4, 4))


def test_extent_scale_pixel_count():
    assert Extent(4, 3).scale_to_pixel_count(12) == Extent(4, 3)
    assert Extent(4, 3).scale_to_pixel_count(24) == Extent(6, 4)
    assert Extent(4, 3).scale_to_pixel_count(50) == Extent(8, 6)
    assert Extent(2, 8).scale_to_pixel_count(55) == Extent(4, 15)
    assert Extent(10, 8).scale_to_pixel_count(60) == Extent(9, 7)


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
        [
            [
                [0, 0, 0, 1],
                [1 / 255, 0, 0, 1],
            ],
            [
                [0, 1 / 255, 0, 1],
                [1 / 255, 1 / 255, 0, 1],
            ],
        ],
        np.float32,
    )
    assert np.all(np.isclose(img.to_array(), expected))


def test_image_compare():
    img1 = create_test_image(2, 2)
    img2 = create_test_image(2, 2)
    assert Image.compare(img1, img2) < 0.0001


def test_image_from_pil():
    pil_img = PILImage.new("RGBA", (2, 2), (255, 0, 0, 255))
    img = Image.from_pil(pil_img)
    assert img.extent == Extent(2, 2)
    assert img.pixel(0, 0) == (255, 0, 0, 255)


@pytest.mark.skip("Benchmark")
def test_image_compress_speed():
    from PyQt5.QtGui import QImageWriter
    from PyQt5.QtCore import QBuffer, QByteArray, QFile, QIODevice
    from timeit import default_timer

    img = Image.load("tests/images/beach_1536x1024.webp")

    print("\nQImage (lossy)")

    for q in range(10, 101, 10):
        start = default_timer()

        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QBuffer.OpenModeFlag.WriteOnly)
        writer = QImageWriter(buffer, QByteArray(b"webp"))
        writer.setQuality(q)
        writer.write(img._qimage)
        buffer.close()

        end = default_timer()
        print(f"Quality {q} | Time {end - start:.3f}s | Size {len(byte_array) // 1024} kB")

        file = QFile(f"beach_1536x1024_q{q}.webp")
        file.open(QIODevice.OpenModeFlag.WriteOnly)
        file.write(byte_array)
        file.close()

    from PIL import Image as PILImage
    from io import BytesIO

    print("\nPillow (lossless)")

    pil_img = PILImage.open("tests/images/beach_1536x1024.webp")
    for q in range(10, 101, 10):
        start = default_timer()
        buffer = BytesIO()
        pil_img.save(buffer, "WEBP", lossless=True, quality=q)
        end = default_timer()
        print(
            f"Compression {q} | Time {end - start:.3f}s | Size {len(buffer.getvalue()) // 1024} kB"
        )


def test_image_equal():
    red1 = Image.create(Extent(2, 2), Qt.GlobalColor.red)
    red2 = Image.create(Extent(2, 2), Qt.GlobalColor.red)
    green = Image.create(Extent(2, 2), Qt.GlobalColor.green)
    assert red1 == red2
    assert red1 != green


def test_draw_image():
    base = Image.create(Extent(32, 32), Qt.GlobalColor.white)
    icon = Image.create(Extent(4, 4), Qt.GlobalColor.red)
    base.draw_image(icon, offset=(7, 23))
    for y in range(32):
        for x in range(32):
            if 7 <= x < 11 and 23 <= y < 27:
                assert base.pixel(x, y) == (255, 0, 0, 255)
            else:
                assert base.pixel(x, y) == (255, 255, 255, 255)


def test_mask_subtract():
    lhs = Mask.load(image_dir / "mask_op_left.webp").to_image()
    rhs = Mask.load(image_dir / "mask_op_right.webp").to_image()
    result = Image.mask_subtract(lhs, rhs)
    result.save(result_dir / "mask_op_subtract.png")
    reference = Mask.load(reference_dir / "mask" / "mask_op_subtract.png").to_image()
    assert Image.compare(result, reference) < 0.0001


def test_mask_add():
    lhs = Mask.load(image_dir / "mask_op_left.webp").to_image()
    rhs = Mask.load(image_dir / "mask_op_right.webp").to_image()
    result = Image.mask_add(lhs, rhs)
    result.save(result_dir / "mask_op_add.png")
    reference = Mask.load(reference_dir / "mask" / "mask_op_add.png").to_image()
    assert Image.compare(result, reference) < 0.0001


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
    assert result == Bounds(-1, -2, 10, 6)


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
    "input,bounds,expected",
    [
        (Bounds(0, 0, 1, 2), Bounds(0, 0, 2, 2), Bounds(0, 0, 1, 2)),
        (Bounds(0, 0, 1, 2), Bounds(0, 0, 1, 1), Bounds(0, 0, 1, 1)),
        (Bounds(2, 4, 1, 2), Bounds(1, 4, 2, 2), Bounds(2, 4, 1, 2)),
        (Bounds(2, 4, 7, 9), Bounds(1, 4, 2, 2), Bounds(2, 4, 1, 2)),
        (Bounds(-1, 5, 3, 3), Bounds(0, 6, 5, 2), Bounds(0, 6, 2, 2)),
    ],
)
def test_restrict_bounds(input: Bounds, bounds: Bounds, expected: Bounds):
    result = Bounds.restrict(input, bounds)
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


def test_bounds_expand():
    bounds = Bounds(1, 2, 4, 5)
    other = Bounds(3, 0, 4, 4)
    result = Bounds.expand(bounds, other)
    assert result == Bounds(1, 0, 6, 7)


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
