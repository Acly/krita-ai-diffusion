from __future__ import annotations
from enum import Enum
from math import ceil, sqrt
from PyQt5.QtGui import QImage, QImageWriter, QPixmap, QIcon, QPainter, QColorSpace
from PyQt5.QtGui import qRgba, qRed, qGreen, qBlue, qAlpha, qGray
from PyQt5.QtCore import Qt, QByteArray, QBuffer, QRect, QSize
from typing import Callable, Iterable, SupportsIndex, Tuple, NamedTuple, Union, Optional
from itertools import product
from pathlib import Path
from .settings import settings


def multiple_of(number, multiple):
    """Round up to the nearest multiple of a number."""
    return ((number + multiple - 1) // multiple) * multiple


class Extent(NamedTuple):
    width: int
    height: int

    def __mul__(self, scale: float | SupportsIndex):
        if isinstance(scale, (float, int)):
            return Extent(round(self.width * scale), round(self.height * scale))
        raise NotImplementedError()

    def at_least(self, min_size: int):
        return Extent(max(self.width, min_size), max(self.height, min_size))

    def multiple_of(self, multiple: int):
        return Extent(multiple_of(self.width, multiple), multiple_of(self.height, multiple))

    def is_multiple_of(self, multiple: int):
        return self.width % multiple == 0 and self.height % multiple == 0

    def scale_keep_aspect(self, target: Extent):
        scale = min(target.width / self.width, target.height / self.height)
        return self * scale

    def scale_to_pixel_count(self, pixel_count: int):
        scale = sqrt(pixel_count / self.pixel_count)
        return self * scale

    @property
    def longest_side(self):
        return max(self.width, self.height)

    @property
    def shortest_side(self):
        return min(self.width, self.height)

    @property
    def average_side(self):
        return (self.width + self.height) // 2

    @property
    def diagonal(self):
        return sqrt(self.width**2 + self.height**2)

    @property
    def pixel_count(self):
        return self.width * self.height

    @staticmethod
    def from_qsize(qsize: QSize):
        return Extent(qsize.width(), qsize.height())

    @staticmethod
    def largest(a, b):
        return a if a.width * a.height > b.width * b.height else b

    @staticmethod
    def ratio(a: "Extent", b: "Extent"):
        return sqrt(a.pixel_count / b.pixel_count)


class Bounds(NamedTuple):
    x: int
    y: int
    width: int
    height: int

    @property
    def offset(self):
        return (self.x, self.y)

    @property
    def extent(self):
        return Extent(self.width, self.height)

    @property
    def is_zero(self):
        return self.width == 0 and self.height == 0

    def is_within(self, x: int, y: int):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    @staticmethod
    def scale(b: "Bounds", scale: float):
        if scale == 1:
            return b

        def apply(x):
            return int(round(x * scale))

        return Bounds(apply(b.x), apply(b.y), apply(b.width), apply(b.height))

    @staticmethod
    def pad(bounds: "Bounds", padding: int, min_size=0, multiple=8, square=False):
        """Grow bounds by adding `padding` evenly on all side. Add additional padding if the area
        is still smaller than `min_size` and ensure the result is a multiple of `multiple`.
        If `square` is set, works towards making width and height balanced.
        """

        def pad_scalar(x, size, pad):
            padded_size = size + 2 * pad
            new_size = multiple_of(max(padded_size, min_size), multiple)
            new_x = x - (new_size - size) // 2
            return new_x, new_size

        pad_x, pad_y = padding, padding
        if square and bounds.width > bounds.height:
            pad_x = max(0, pad_x - (bounds.width - bounds.height) // 2)
        elif square and bounds.height > bounds.width:
            pad_y = max(0, pad_y - (bounds.height - bounds.width) // 2)

        new_x, new_width = pad_scalar(bounds.x, bounds.width, pad_x)
        new_y, new_height = pad_scalar(bounds.y, bounds.height, pad_y)
        return Bounds(new_x, new_y, new_width, new_height)

    @staticmethod
    def clamp(bounds: "Bounds", extent: Extent):
        """Clamp mask bounds to be inside an image region. Bounds extent should remain unchanged,
        unless it is larger than the image extent.
        """

        def impl(off, size, max_size):
            if size >= max_size:
                return 0, max_size
            off = max(off, 0)
            excess = max((off + size) - max_size, 0)
            return off - excess, size

        x, width = impl(bounds.x, bounds.width, extent.width)
        y, height = impl(bounds.y, bounds.height, extent.height)
        return Bounds(x, y, width, height)

    @staticmethod
    def restrict(bounds: "Bounds", within: "Bounds"):
        """Restrict bounds to be inside another bounds."""
        x = max(within.x, bounds.x)
        y = max(within.y, bounds.y)
        width = min(within.x + within.width, bounds.x + bounds.width) - x
        height = min(within.y + within.height, bounds.y + bounds.height) - y
        return Bounds(x, y, width, height)

    @staticmethod
    def expand(bounds: "Bounds", include: "Bounds"):
        """Expand bounds to include another bounds."""
        x = min(bounds.x, include.x)
        y = min(bounds.y, include.y)
        width = max(bounds.x + bounds.width, include.x + include.width) - x
        height = max(bounds.y + bounds.height, include.y + include.height) - y
        return Bounds(x, y, width, height)

    @staticmethod
    def apply_crop(bounds: "Bounds", image_bounds: "Bounds"):
        """Adjust bounds area after the image has been cropped."""
        x = bounds.x - image_bounds.x
        y = bounds.y - image_bounds.y
        result = Bounds(x, y, bounds.width, bounds.height)
        return Bounds.clamp(result, image_bounds.extent)

    @staticmethod
    def minimum_size(bounds: "Bounds", min_size: int, max_extent: Extent):
        """Return bounds extended to a minimum size if they still fit."""
        if any(x < min_size for x in max_extent):
            return None  # doesn't fit, image too small
        result = Bounds(
            bounds.x, bounds.y, max(bounds.width, min_size), max(bounds.height, min_size)
        )
        return Bounds.clamp(result, max_extent)

    @staticmethod
    def from_qrect(qrect: QRect):
        return Bounds(qrect.x(), qrect.y(), qrect.width(), qrect.height())


def extent_equal(a: QImage, b: QImage):
    return a.width() == b.width() and a.height() == b.height()


class ImageFileFormat(Enum):
    # Low compression rate, fast but large files. Good for local use, but maybe not optimal
    # for remote server where images are transferred via internet.
    png = ("PNG", 85)

    webp = ("WEBP", 90)


class Image:
    def __init__(self, qimage: QImage):
        assert qimage.format() in [QImage.Format_ARGB32, QImage.Format_Grayscale8]
        self._qimage = qimage

    @staticmethod
    def load(filepath: Union[str, Path]):
        image = QImage()
        success = image.load(str(filepath))
        assert success, f"Failed to load image {filepath}"
        return Image(image.convertToFormat(QImage.Format_ARGB32))

    @staticmethod
    def create(extent: Extent, fill=None):
        img = Image(QImage(extent.width, extent.height, QImage.Format_ARGB32))
        if fill is not None:
            img._qimage.fill(fill)
        return img

    @property
    def width(self):
        return self._qimage.width()

    @property
    def height(self):
        return self._qimage.height()

    @property
    def extent(self):
        return Extent(self.width, self.height)

    @property
    def is_rgba(self):
        return self._qimage.format() == QImage.Format_ARGB32

    @property
    def is_mask(self):
        return self._qimage.format() == QImage.Format_Grayscale8

    @staticmethod
    def from_base64(data: str):
        bytes = QByteArray.fromBase64(data.encode("utf-8"))
        return Image.from_bytes(bytes)

    @staticmethod
    def from_bytes(data: QByteArray | memoryview, format: str | None = None):
        img = QImage.fromData(data, format)
        assert img and not img.isNull(), "Failed to load image from memory"
        return Image(img.convertToFormat(QImage.Format_ARGB32))

    @staticmethod
    def scale(img: "Image", target: Extent):
        mode = Qt.AspectRatioMode.IgnoreAspectRatio
        quality = Qt.TransformationMode.SmoothTransformation
        scaled = img._qimage.scaled(target.width, target.height, mode, quality)
        return Image(scaled.convertToFormat(QImage.Format_ARGB32))

    @staticmethod
    def scale_to_fit(img: "Image", target: Extent):
        return Image.scale(img, img.extent.scale_keep_aspect(target))

    @staticmethod
    def crop(img: "Image", bounds: Bounds):
        return Image(img._qimage.copy(*bounds))

    @staticmethod
    def compare(a, b):
        assert extent_equal(a._qimage, b._qimage)
        import numpy as np

        # Compute RMSE
        a = a.to_array()
        b = b.to_array()
        return np.sqrt(np.mean((a - b) ** 2))

    def pixel(self, x: int, y: int):
        c = self._qimage.pixel(x, y)
        if self.is_rgba:
            return (qRed(c), qGreen(c), qBlue(c), qAlpha(c))
        else:
            return qGray(c)

    def set_pixel(self, x: int, y: int, color: Tuple[int, int, int, int]):
        # Note: this is slow, only used for testing
        r, g, b, a = color
        self._qimage.setPixel(x, y, qRgba(r, g, b, a))

    def make_opaque(self, background=Qt.GlobalColor.white):
        painter = QPainter(self._qimage)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
        painter.fillRect(self._qimage.rect(), background)
        painter.end()

    @property
    def data(self):
        ptr = self._qimage.bits()
        assert ptr is not None, "Accessing data of invalid image"
        ptr.setsize(self._qimage.byteCount())
        return QByteArray(ptr.asstring())

    @property
    def size(self):  # in bytes
        return self._qimage.byteCount()

    def to_array(self):
        import numpy as np

        w, h = self.extent
        bits = self._qimage.constBits()
        assert bits is not None, "Accessing data of invalid image"
        ptr = bits.asarray(w * h * 4)
        array = np.frombuffer(ptr, np.uint8).reshape(w, h, 4)  # type: ignore
        return array.astype(np.float32) / 255

    def write(self, buffer: QBuffer, format=ImageFileFormat.png):
        # Compression takes time for large images and blocks the UI, might be worth to thread.
        format_str, quality = format.value
        if format is ImageFileFormat.webp:
            writer = QImageWriter(buffer, QByteArray(b"webp"))
            writer.setQuality(quality)
            writer.write(self._qimage)
        else:
            self._qimage.save(buffer, format_str, quality)

    def to_bytes(self, format=ImageFileFormat.png):
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QBuffer.OpenModeFlag.WriteOnly)
        self.write(buffer, format)
        buffer.close()
        return byte_array

    def to_base64(self, format=ImageFileFormat.png):
        byte_array = self.to_bytes(format)
        return byte_array.toBase64().data().decode("utf-8")

    def to_pixmap(self):
        return QPixmap.fromImage(self._qimage)

    def to_icon(self):
        return QIcon(self.to_pixmap())

    def draw_image(self, image: "Image", offset: tuple[int, int] = (0, 0)):
        w, h = self.extent
        x, y = offset[0] if offset[0] >= 0 else w + offset[0], (
            offset[1] if offset[1] >= 0 else h + offset[1]
        )
        painter = QPainter(self._qimage)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawImage(x, y, image._qimage)
        painter.end()

    def save(self, filepath: Union[str, Path]):
        success = self._qimage.save(str(filepath))
        assert success, f"Failed to save image to {filepath}"

    def debug_save(self, name):
        if settings.debug_image_folder:
            self.save(Path(settings.debug_image_folder, f"{name}.png"))

    def __eq__(self, other):
        return self._qimage == other._qimage


class ImageCollection:
    _items: list[Image]

    def __init__(self, items: Optional[Iterable[Image]] = None):
        self._items = []
        if items is not None:
            self.append(items)

    def append(self, items: Union[Image, Iterable[Image]]):
        if isinstance(items, ImageCollection):
            self._items.extend(items)
        elif isinstance(items, Image):
            self._items.append(items)
        else:  # some iterable
            for item in items:
                self.append(item)

    def map(self, func: Callable[[Image], Image]):
        result = []
        for img in self._items:
            r = func(img)
            assert isinstance(r, Image)
            result.append(r)
        return ImageCollection(result)

    def each(self, func: Callable[[Image], None]):
        for img in self._items:
            func(img)

    def remove(self, index: int):
        return self._items.pop(index)

    def save(self, filepath: Union[Path, str]):
        filepath = Path(filepath)
        suffix = filepath.suffix
        filepath = filepath.with_suffix("")
        for i, img in enumerate(self._items):
            img.save(filepath.with_name(f"{filepath.name}_{i}").with_suffix(suffix))

    def debug_save(self, name):
        for i, img in enumerate(self._items):
            img.debug_save(f"{name}_{i}")

    @property
    def size(self):
        return sum(i.size for i in self)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i: int):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)


class Mask:
    _data: Optional[QByteArray]

    bounds: Bounds
    image: QImage

    def __init__(self, bounds: Bounds, data: Union[QImage, QByteArray]):
        self.bounds = bounds
        if isinstance(data, QImage):
            self.image = data
        else:
            assert len(data) == bounds.width * bounds.height
            self._data = data
            self.image = QImage(
                self._data.data(),
                bounds.width,
                bounds.height,
                bounds.width,
                QImage.Format_Grayscale8,
            )
            assert not self.image.isNull()

    @staticmethod
    def rectangle(bounds: Bounds, feather=0):
        # Note: for testing only, where Krita selection is not available
        m = [255 for i in range(bounds.width * bounds.height)]
        if feather > 0:
            for x, y in product(range(bounds.width), range(bounds.height)):
                l = min(0, x - feather) * -1
                t = min(0, y - feather) * -1
                r = max(0, x + feather + 1 - bounds.width)
                b = max(0, y + feather + 1 - bounds.height)
                alpha = (
                    64 * l // feather + 64 * t // feather + 64 * r // feather + 64 * b // feather
                )
                m[y * bounds.width + x] = 255 - alpha
        return Mask(bounds, QByteArray(bytes(m)))

    @staticmethod
    def load(filepath: Union[str, Path]):
        mask = QImage()
        success = mask.load(str(filepath))
        assert success, f"Failed to load mask {filepath}"
        mask.setColorSpace(QColorSpace())
        mask = mask.convertToFormat(QImage.Format.Format_Grayscale8)
        return Mask(Bounds(0, 0, mask.width(), mask.height()), mask)

    @staticmethod
    def crop(mask: "Mask", bounds: Bounds):
        return Mask(bounds, mask.image.copy(*bounds))

    def value(self, x: int, y: int):
        if self.bounds.is_within(x, y):
            return qGray(self.image.pixel(x, y))
        return 0

    def to_array(self):
        e = self.bounds.extent
        return [self.value(x, y) for y in range(e.height) for x in range(e.width)]

    def to_image(self, extent: Optional[Extent] = None):
        if extent is None:
            return Image(self.image)
        img = QImage(extent.width, extent.height, QImage.Format_Grayscale8)
        img.fill(0)
        painter = QPainter(img)
        painter.drawImage(self.bounds.x, self.bounds.y, self.image)
        return Image(img)
