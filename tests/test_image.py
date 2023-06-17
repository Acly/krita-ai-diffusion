from PyQt5.QtGui import QImage, qRgba
from PyQt5.QtCore import QByteArray
from ai_tools import image, Mask, Bounds, Extent, Image

def create_test_image(w, h):
    img = QImage(w, h, QImage.Format_ARGB32)
    for y in range(h):
        for x in range(w):
            img.setPixel(x, y , qRgba(x, y, 0, 255))
    return Image(img)

def test_base64():
    img = create_test_image(15, 21)
    encoded = img.to_base64()
    decoded = Image.from_base64(encoded)
    assert img == decoded

def test_pad_bounds():
    bounds = Bounds(3, 2, 5, 9)
    extent = Extent(9, 14)
    result = Bounds.pad(bounds, 0, 4, extent)
    assert result.x == 1 and result.width == 8\
       and result.y == 2 and result.height == 12

def test_mask_to_image():
    data = QByteArray(b'\x00\x01\x02\xff')
    mask = Mask(Bounds(0, 0, 2, 2), data)
    img = mask.to_image(Extent(2, 2))
    assert img.pixel(0, 0) == (0, 0, 0, 255)\
       and img.pixel(1, 0) == (1, 1, 1, 255)\
       and img.pixel(0, 1) == (2, 2, 2, 255)\
       and img.pixel(1, 1) == (255, 255, 255, 255)

def test_mask_to_image_offset():
    data = QByteArray(b'\x00\x01\x02\xff')
    mask = Mask(Bounds(1, 2, 2, 2), data)
    img = mask.to_image(Extent(4, 4))
    assert img.pixel(1, 2) == (0, 0, 0, 255)\
       and img.pixel(2, 2) == (1, 1, 1, 255)\
       and img.pixel(1, 3) == (2, 2, 2, 255)\
       and img.pixel(2, 3) == (255, 255, 255, 255)

def test_mask_to_image_no_extent():
    data = QByteArray(b'\x00\x01\x02\xff')
    mask = Mask(Bounds(1, 2, 2, 2), data)
    img = mask.to_image()
    assert img.width == 2 and img.height == 2
    assert img.pixel(0, 0) == (0, 0, 0, 255)\
       and img.pixel(1, 0) == (1, 1, 1, 255)\
       and img.pixel(0, 1) == (2, 2, 2, 255)\
       and img.pixel(1, 1) == (255, 255, 255, 255)
    
def test_apply_mask():
    data = QByteArray(b'\x00\x01\x02\xff')
    mask = Mask(Bounds(1, 2, 2, 2), data)
    img = create_test_image(4, 4)
    Mask.apply(img, mask)
    assert img.pixel(0, 0) == (0, 0, 0, 0)\
       and img.pixel(1, 2) == (1, 2, 0, 0)\
       and img.pixel(2, 2) == (2, 2, 0, 1)\
       and img.pixel(1, 3) == (1, 3, 0, 2)\
       and img.pixel(2, 3) == (2, 3, 0, 255)\
       and img.pixel(3, 3) == (3, 3, 0, 0)

def test_mask_rectangle():
    mask = Mask.rectangle(Bounds(1, 2, 6, 5), feather=2)
    assert mask.to_array() == [
        127, 159, 191, 191, 159, 127,
        159, 191, 223, 223, 191, 159,
        191, 223, 255, 255, 223, 191,
        159, 191, 223, 223, 191, 159,
        127, 159, 191, 191, 159, 127]

def test_downscale():
    img = create_test_image(12, 8)
    result = Image.downscale(img, 6)
    assert result.width == 6 and result.height == 4

