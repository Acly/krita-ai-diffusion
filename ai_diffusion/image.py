from __future__ import annotations

import struct
import zlib
from collections.abc import Callable, Iterable
from math import sqrt
from pathlib import Path
from typing import NamedTuple, SupportsIndex

from PyQt5.QtCore import QBuffer, QByteArray, QFile, QIODevice, QRect, QSize, Qt
from PyQt5.QtGui import (
    QColorSpace,
    QIcon,
    QImage,
    QImageReader,
    QImageWriter,
    QPainter,
    QPixmap,
    qAlpha,
    qBlue,
    qGray,
    qGreen,
    qRed,
    qRgba,
)

from .platform_tools import is_linux
from .settings import ImageFileFormat, settings
from .util import clamp, ensure
from .util import client_logger as log


def multiple_of(number, multiple):
    """Round up to the nearest multiple of a number."""
    return ((number + multiple - 1) // multiple) * multiple


class Extent(NamedTuple):
    width: int
    height: int

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
    def from_points(start: Point, end: Point):
        return Extent(end.x - start.x, end.y - start.y)

    @staticmethod
    def from_qsize(qsize: QSize):
        return Extent(qsize.width(), qsize.height())

    @staticmethod
    def largest(a: Extent, b: Extent):
        return a if a.width * a.height > b.width * b.height else b

    @staticmethod
    def min(a: Extent, b: Extent):
        return Extent(min(a.width, b.width), min(a.height, b.height))

    @staticmethod
    def ratio(a: Extent, b: Extent):
        return sqrt(a.pixel_count / b.pixel_count)

    def __add__(self, other):
        return Extent(self.width + other.width, self.height + other.height)

    def __sub__(self, other: Extent):
        return Extent(self.width - other.width, self.height - other.height)

    def __mul__(self, scale: float | SupportsIndex):
        if isinstance(scale, (float, int)):
            return Extent(round(self.width * scale), round(self.height * scale))
        raise NotImplementedError()

    def __floordiv__(self, div: int):
        return Extent(self.width // div, self.height // div)


class Point(NamedTuple):
    x: int
    y: int

    def __add__(self, other):
        x, y = other[0], other[1]
        return Point(self.x + x, self.y + y)

    def __sub__(self, other: Point):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(self.x * other.x, self.y * other.y)
        return Point(self.x * other, self.y * other)

    def __floordiv__(self, div: int):
        return Point(self.x // div, self.y // div)

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def clamp(self, bounds: Bounds):
        return Point(
            clamp(self.x, bounds.x, bounds.x + bounds.width),
            clamp(self.y, bounds.y, bounds.y + bounds.height),
        )


class Bounds(NamedTuple):
    x: int
    y: int
    width: int
    height: int

    @staticmethod
    def from_extent(extent: Extent):
        return Bounds(0, 0, extent.width, extent.height)

    @staticmethod
    def from_points(start: Point, end: Point):
        return Bounds(start.x, start.y, end.x - start.x, end.y - start.y)

    @property
    def offset(self):
        return (self.x, self.y)

    @property
    def extent(self):
        return Extent(self.width, self.height)

    @property
    def is_zero(self):
        return self.width * self.height == 0

    def is_within(self, x: int, y: int):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    @staticmethod
    def scale(b: Bounds, scale: float):
        if scale == 1:
            return b

        def apply(x):
            return round(x * scale)

        return Bounds(apply(b.x), apply(b.y), apply(b.width), apply(b.height))

    @staticmethod
    def pad(bounds: Bounds, padding: int, min_size=0, multiple=8, square=False):
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
            pad_x = max(pad_x // 2, pad_x - (bounds.width - bounds.height) // 2)
        elif square and bounds.height > bounds.width:
            pad_y = max(pad_y // 2, pad_y - (bounds.height - bounds.width) // 2)

        new_x, new_width = pad_scalar(bounds.x, bounds.width, pad_x)
        new_y, new_height = pad_scalar(bounds.y, bounds.height, pad_y)
        return Bounds(new_x, new_y, new_width, new_height)

    @staticmethod
    def clamp(bounds: Bounds, extent: Extent):
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
    def restrict(bounds: Bounds, within: Bounds):
        """Restrict bounds to be inside another bounds."""
        x = max(within.x, bounds.x)
        y = max(within.y, bounds.y)
        width = max(0, min(within.x + within.width, bounds.x + bounds.width) - x)
        height = max(0, min(within.y + within.height, bounds.y + bounds.height) - y)
        return Bounds(x, y, width, height)

    @staticmethod
    def expand(bounds: Bounds, include: Bounds):
        """Expand bounds to include another bounds."""
        x = min(bounds.x, include.x)
        y = min(bounds.y, include.y)
        width = max(bounds.x + bounds.width, include.x + include.width) - x
        height = max(bounds.y + bounds.height, include.y + include.height) - y
        return Bounds(x, y, width, height)

    @staticmethod
    def apply_crop(bounds: Bounds, image_bounds: Bounds):
        """Adjust bounds area after the image has been cropped."""
        x = bounds.x - image_bounds.x
        y = bounds.y - image_bounds.y
        result = Bounds(x, y, bounds.width, bounds.height)
        return Bounds.clamp(result, image_bounds.extent)

    @staticmethod
    def at_least(bounds: Bounds, min_size: int):
        """Return bounds with width and height being at least `min_size`."""
        return Bounds(bounds.x, bounds.y, max(bounds.width, min_size), max(bounds.height, min_size))

    @staticmethod
    def minimum_size(bounds: Bounds, min_size: int, max_extent: Extent):
        """Return bounds extended to a minimum size if they still fit."""
        if any(x < min_size for x in max_extent):
            return None  # doesn't fit, image too small
        return Bounds.clamp(Bounds.at_least(bounds, min_size), max_extent)

    @staticmethod
    def intersection(a: Bounds, b: Bounds):
        x = max(a.x, b.x)
        y = max(a.y, b.y)
        width = min(a.x + a.width, b.x + b.width) - x
        height = min(a.y + a.height, b.y + b.height) - y
        return Bounds(x, y, max(0, width), max(0, height))

    @staticmethod
    def union(a: Bounds, b: Bounds):
        x = min(a.x, b.x)
        y = min(a.y, b.y)
        width = max(a.x + a.width, b.x + b.width) - x
        height = max(a.y + a.height, b.y + b.height) - y
        return Bounds(x, y, width, height)

    @property
    def area(self):
        return self.width * self.height

    def relative_to(self, reference: Bounds):
        """Return bounds relative to another bounds."""
        return Bounds(self.x - reference.x, self.y - reference.y, self.width, self.height)

    @staticmethod
    def from_qrect(qrect: QRect):
        return Bounds(qrect.x(), qrect.y(), qrect.width(), qrect.height())


def extent_equal(a: QImage, b: QImage):
    return a.width() == b.width() and a.height() == b.height()


_qt_supports_webp = None


def qt_supports_webp():
    global _qt_supports_webp
    if _qt_supports_webp is None:
        _qt_supports_webp = QByteArray(b"webp") in QImageWriter.supportedImageFormats()
    return _qt_supports_webp


class Image:
    def __init__(self, qimage: QImage):
        self._qimage = qimage

    @staticmethod
    def load(filepath: str | Path):
        image = QImage()
        success = image.load(str(filepath))
        assert success, f"Failed to load image {filepath}"
        return Image(image)

    @staticmethod
    def create(extent: Extent, fill=None):
        img = Image(QImage(extent.width, extent.height, QImage.Format.Format_ARGB32))
        if fill is not None:
            img._qimage.fill(fill)
        return img

    @staticmethod
    def from_packed_bytes(data: QByteArray, extent: Extent, channels=4):
        assert channels in {4, 1}
        stride = extent.width * channels
        format = QImage.Format.Format_ARGB32 if channels == 4 else QImage.Format.Format_Grayscale8
        qimg = QImage(data, extent.width, extent.height, stride, format)
        return Image(qimg)

    @staticmethod
    def copy(image: Image):
        return Image(QImage(image._qimage))

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
        return self._qimage.format() in [
            QImage.Format.Format_Indexed8,
            QImage.Format.Format_ARGB32,
            QImage.Format.Format_RGB32,
            QImage.Format.Format_RGBA8888,
        ]

    @property
    def is_mask(self):
        return self._qimage.format() == QImage.Format.Format_Grayscale8

    @staticmethod
    def from_base64(data: str):
        bytes = QByteArray.fromBase64(data.encode("utf-8"))
        return Image.from_bytes(bytes)

    @staticmethod
    def from_bytes(data: QBuffer | QByteArray | memoryview | bytes, format: str | None = None):
        if isinstance(data, QBuffer):
            buffer = data
        else:
            if not isinstance(data, QByteArray):
                data = QByteArray(bytearray(data))
            buffer = QBuffer(data)
            buffer.open(QBuffer.OpenModeFlag.ReadOnly)
        if format:
            loader = QImageReader(buffer, format.encode("utf-8"))
        else:
            loader = QImageReader(buffer)

        img = QImage()
        if loader.read(img):
            return Image(img)
        else:
            raise RuntimeError(f"Failed to load image from buffer: {loader.errorString()}")

    @staticmethod
    def from_pil(pil_image):
        assert pil_image.mode == "RGBA"
        qimage = QImage(
            pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format.Format_RGBA8888
        )
        return Image(qimage)

    @staticmethod
    def scale(img: Image, target: Extent):
        if isinstance(img, DummyImage):
            return DummyImage(target)
        if img.extent == target:
            return img
        mode = Qt.AspectRatioMode.IgnoreAspectRatio
        quality = Qt.TransformationMode.SmoothTransformation
        scaled = img._qimage.scaled(target.width, target.height, mode, quality)
        return Image(scaled)

    @staticmethod
    def scale_to_fit(img: Image, target: Extent):
        return Image.scale(img, img.extent.scale_keep_aspect(target))

    @staticmethod
    def crop(img: Image, bounds: Bounds):
        return Image(img._qimage.copy(*bounds))

    @staticmethod
    def _mask_op(lhs: Image, rhs: Image, mode: QPainter.CompositionMode):
        assert extent_equal(lhs._qimage, rhs._qimage)
        assert lhs.is_mask and rhs.is_mask
        result = lhs._qimage.copy()
        result.reinterpretAsFormat(QImage.Format.Format_Alpha8)
        rhs._qimage.reinterpretAsFormat(QImage.Format.Format_Alpha8)
        painter = QPainter(result)
        painter.setCompositionMode(mode)
        painter.drawImage(0, 0, rhs._qimage)
        painter.end()
        rhs._qimage.reinterpretAsFormat(QImage.Format.Format_Grayscale8)
        result.reinterpretAsFormat(QImage.Format.Format_Grayscale8)
        return Image(result)

    @staticmethod
    def save_png_w_itxt(img_path: str | Path, png_data: bytes, keyword: str, text: str):
        if png_data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Not a valid PNG file")

        offset = 8
        ihdr_inserted = False

        with open(img_path, "wb") as f:
            # Write PNG header
            f.write(png_data[:8])

            while offset < len(png_data):
                length = struct.unpack(">I", png_data[offset : offset + 4])[0]
                chunk_type = png_data[offset + 4 : offset + 8]
                chunk_data = png_data[offset + 8 : offset + 8 + length]
                crc = png_data[offset + 8 + length : offset + 12 + length]
                offset += 12 + length

                # Write original chunk
                f.write(struct.pack(">I", length))
                f.write(chunk_type)
                f.write(chunk_data)
                f.write(crc)

                if not ihdr_inserted and chunk_type == b"IHDR":
                    # Insert iTXt chunk after IHDR
                    keyword_bytes = keyword.encode("latin1")
                    text_bytes = text.encode("utf-8")
                    itxt_data = (
                        keyword_bytes
                        + b"\x00"
                        + b"\x00"  # compression flag: 0 (not compressed)
                        + b"\x00"  # compression method: 0
                        + b"\x00"  # language tag: empty
                        + b"\x00"  # translated keyword: empty
                        + text_bytes
                    )
                    f.write(struct.pack(">I", len(itxt_data)))
                    f.write(b"iTXt")
                    f.write(itxt_data)
                    f.write(struct.pack(">I", zlib.crc32(b"iTXt" + itxt_data) & 0xFFFFFFFF))
                    ihdr_inserted = True

    @classmethod
    def mask_subtract(cls, lhs: Image, rhs: Image):
        return cls._mask_op(rhs, lhs, QPainter.CompositionMode.CompositionMode_SourceOut)

    @classmethod
    def mask_add(cls, lhs: Image, rhs: Image):
        return cls._mask_op(lhs, rhs, QPainter.CompositionMode.CompositionMode_SourceOver)

    @staticmethod
    def compare(img_a: Image, img_b: Image):
        assert extent_equal(img_a._qimage, img_b._qimage)
        import numpy as np

        # Compute RMSE
        a = img_a.to_array()
        b = img_b.to_array()
        return np.sqrt(np.mean((a - b) ** 2))

    def pixel(self, x: int, y: int):
        c = self._qimage.pixel(x, y)
        if self.is_rgba:
            return (qRed(c), qGreen(c), qBlue(c), qAlpha(c))
        else:
            return qGray(c)

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int, int]):
        # Note: this is slow, only used for testing
        r, g, b, a = color
        self._qimage.setPixel(x, y, qRgba(r, g, b, a))

    def make_opaque(self, background=Qt.GlobalColor.white):
        painter = QPainter(self._qimage)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
        painter.fillRect(self._qimage.rect(), background)
        painter.end()

    def invert(self):
        self._qimage.invertPixels()

    def average(self):
        assert self.is_mask
        avg = Image.scale(self, Extent(1, 1)).pixel(0, 0)
        avg = avg[0] if isinstance(avg, tuple) else avg
        return avg / 255

    @property
    def data(self):
        self.to_krita_format()
        if self._qimage.bytesPerLine() != self._qimage.width() * (self._qimage.depth() // 8):
            # QImage scanlines are padded to 32-bit, which can be a problem with mask formats
            buffer = QByteArray()
            for i in range(self._qimage.height()):
                ptr = ensure(self._qimage.scanLine(i), "Accessing data of invalid image")
                buffer.append(ptr.asstring(self._qimage.width() * (self._qimage.depth() // 8)))
            return buffer
        else:
            ptr = ensure(self._qimage.constBits(), "Accessing data of invalid image")
            return QByteArray(ptr.asstring(self._qimage.byteCount()))

    @property
    def size(self):  # in bytes
        return self._qimage.byteCount()

    def to_array(self):
        import numpy as np

        self.to_numpy_format()
        w, h = self.extent
        c = 4 if self.is_rgba else 1
        bits = self._qimage.constBits()
        assert bits is not None, "Accessing data of invalid image"
        ptr = bits.asarray(w * h * c)
        array = np.frombuffer(ptr, np.uint8).reshape(h, w, c)  # type: ignore
        return array.astype(np.float32) / 255

    def write(
        self, buffer: QIODevice, format=ImageFileFormat.png, override_quality: int | None = None
    ):
        # Compression takes time for large images and blocks the UI, might be worth to thread.
        if not qt_supports_webp():
            format = format.no_webp_fallback
        format_str = format.extension
        quality = override_quality if override_quality is not None else format.quality
        writer = QImageWriter(buffer, QByteArray(format_str.encode("utf-8")))
        writer.setQuality(quality)
        result = writer.write(self._qimage)
        if not result:
            info = f"[{self.width}x{self.height} format={self._qimage.format()}] -> {format_str}@{quality}"
            if is_linux and format_str == "webp":
                log.warning(
                    "To enable support for writing webp images, you may need to install the 'qt5-imageformats' package."
                )
                global _qt_supports_webp
                _qt_supports_webp = False
                self.write(buffer, format.no_webp_fallback)
            raise RuntimeError(f"Failed to write image to buffer: {writer.errorString()} {info}")

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
        self.to_krita_format()
        return QPixmap.fromImage(self._qimage)

    def to_icon(self):
        return QIcon(self.to_pixmap())

    def to_mask(self, bounds: Bounds | None = None):
        assert self.is_mask
        return Mask(bounds or Bounds(0, 0, *self.extent), self._qimage)

    def draw_image(self, image: Image, offset: tuple[int, int] = (0, 0), keep_alpha=False):
        mode = QPainter.CompositionMode.CompositionMode_SourceOver
        if keep_alpha:
            mode = QPainter.CompositionMode.CompositionMode_SourceAtop
        painter = QPainter(self._qimage)
        painter.setCompositionMode(mode)
        painter.drawImage(*offset, image._qimage)
        painter.end()

    def save(
        self,
        filepath: str | Path,
        format: ImageFileFormat | None = None,
        quality: int | None = None,
    ):
        fmt = format or ImageFileFormat.from_extension(filepath)
        file = QFile(str(filepath))
        if not file.open(QFile.OpenModeFlag.WriteOnly):
            raise RuntimeError(f"Failed to open {filepath} for writing: {file.errorString()}")
        try:
            self.write(file, fmt, quality)
        finally:
            file.close()

    def save_png_with_metadata(
        self, filepath: str | Path, metadata_text: str, format: ImageFileFormat | None = None
    ):
        png_bytes = bytes(self.to_bytes(format or ImageFileFormat.png))
        self.save_png_w_itxt(filepath, png_bytes, "parameters", metadata_text)

    def debug_save(self, name):
        if settings.debug_image_folder:
            self.save(Path(settings.debug_image_folder, f"{name}.png"))

    def to_krita_format(self):
        if self.is_rgba and self._qimage.format() != QImage.Format.Format_ARGB32:
            self._qimage = self._qimage.convertToFormat(QImage.Format.Format_ARGB32)
        return self

    def to_numpy_format(self):
        if self.is_rgba and self._qimage.format() != QImage.Format.Format_RGBA8888:
            self._qimage = self._qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        return self

    def __eq__(self, other):
        return isinstance(other, Image) and self._qimage == other._qimage


class DummyImage(Image):
    _extent: Extent

    def __init__(self, extent: Extent):
        super().__init__(QImage())
        self._extent = extent

    @property
    def width(self):
        return self._extent.width

    @property
    def height(self):
        return self._extent.height

    def __eq__(self, other):
        return isinstance(other, DummyImage) and self.extent == other.extent

    def __hash__(self):
        return hash(self.extent)


class ImageCollection:
    _items: list[Image]

    def __init__(self, items: Iterable[Image] | None = None):
        self._items = []
        if items is not None:
            self.append(items)

    def append(self, items: Image | Iterable[Image]):
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

    def save(self, filepath: Path | str):
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

    def to_bytes(self, format=ImageFileFormat.webp):
        offsets = []
        data = QByteArray()
        result = QBuffer(data)
        result.open(QBuffer.OpenModeFlag.WriteOnly)
        for img in self:
            offsets.append(result.pos())
            img.write(result, format)
        result.close()
        return data, offsets

    @staticmethod
    def from_bytes(data: QByteArray | bytes, offsets: list[int]):
        if isinstance(data, bytes):
            data = QByteArray(data)

        images = ImageCollection()
        buffer = QBuffer(data)
        buffer.open(QBuffer.OpenModeFlag.ReadOnly)
        for i, offset in enumerate(offsets):
            buffer.seek(offset)
            images.append(Image.from_bytes(buffer))
        buffer.close()
        return images

    def to_base64(self, format=ImageFileFormat.png):
        bytes, offsets = self.to_bytes(format)
        return bytes.toBase64().data().decode("utf-8"), offsets

    @staticmethod
    def from_base64(data: str, offsets: list[int]):
        bytes = QByteArray.fromBase64(data.encode("utf-8"))
        return ImageCollection.from_bytes(bytes, offsets)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i: int):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)


class Mask:
    def __init__(self, bounds: Bounds, data: QImage | QByteArray):
        self.bounds = bounds
        if isinstance(data, QImage):
            self.image: QImage = data
        else:
            assert len(data) == bounds.width * bounds.height
            self.image = Image.from_packed_bytes(data, bounds.extent, channels=1)._qimage
            assert not self.image.isNull()

    @staticmethod
    def transparent(bounds: Bounds):
        return Mask(bounds, QByteArray(bytes(bounds.width * bounds.height)))

    @staticmethod
    def rectangle(bounds: Bounds, context: Bounds):
        # Note: for testing only, where Krita selection is not available
        m = []
        for y in range(context.height):
            for x in range(context.width):
                if (
                    x >= bounds.x
                    and x < bounds.x + bounds.width
                    and y >= bounds.y
                    and y < bounds.y + bounds.height
                ):
                    m.append(255)
                else:
                    m.append(0)
        return Mask(context, QByteArray(bytes(m)))

    @staticmethod
    def load(filepath: str | Path):
        mask = QImage()
        success = mask.load(str(filepath))
        assert success, f"Failed to load mask {filepath}"
        mask.setColorSpace(QColorSpace())
        mask = mask.convertToFormat(QImage.Format.Format_Grayscale8)
        return Mask(Bounds(0, 0, mask.width(), mask.height()), mask)

    @staticmethod
    def crop(mask: Mask, bounds: Bounds):
        return Mask(bounds, mask.image.copy(*bounds))

    def value(self, x: int, y: int):
        if self.bounds.is_within(x, y):
            return qGray(self.image.pixel(x, y))
        return 0

    def to_array(self):
        e = self.bounds.extent
        return [self.value(x, y) for y in range(e.height) for x in range(e.width)]

    def to_image(self, extent: Extent | None = None):
        if extent is None:
            return Image(self.image)
        img = QImage(extent.width, extent.height, QImage.Format_Grayscale8)
        img.fill(0)
        painter = QPainter(img)
        painter.drawImage(self.bounds.x, self.bounds.y, self.image)
        return Image(img)
