from PyQt5.QtGui import QImage, QPixmap, QIcon, qRgba, qRed, qGreen, qBlue, qAlpha
from PyQt5.QtCore import Qt, QByteArray, QBuffer
from typing import Callable, Iterable, Tuple, NamedTuple, Union
from itertools import product
from pathlib import Path
from .settings import settings


def round_to_multiple(number, multiple):
    if multiple == 1:
        return number
    return multiple * (number // multiple + 1)


class Extent(NamedTuple):
    width: int
    height: int


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

    def is_within(self, x: int, y: int):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    @staticmethod
    def scale(b, scale: float):
        def apply(x):
            return int(round(x * scale))

        return Bounds(apply(b.x), apply(b.y), apply(b.width), apply(b.height))

    @staticmethod
    def pad(bounds, padding: int, multiple: int):
        new_width = round_to_multiple(bounds.width + 2 * padding, multiple)
        new_height = round_to_multiple(bounds.height + 2 * padding, multiple)
        new_x = bounds.x - padding
        new_y = bounds.y - padding
        return Bounds(new_x, new_y, new_width, new_height)

    @staticmethod
    def clamp(bounds, extent: Extent):
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


def extent_equal(a: QImage, b: QImage):
    return a.width() == b.width() and a.height() == b.height()


class Image:
    def __init__(self, qimage: QImage):
        assert qimage.format() == QImage.Format_ARGB32
        self._qimage = qimage

    @staticmethod
    def load(filepath: Union[str, Path]):
        image = QImage()
        success = image.load(str(filepath))
        assert success, f"Failed to load image {filepath}"
        return Image(image.convertToFormat(QImage.Format_ARGB32))

    @staticmethod
    def create(extent: Extent):
        return Image(QImage(extent.width, extent.height, QImage.Format_ARGB32))

    @property
    def width(self):
        return self._qimage.width()

    @property
    def height(self):
        return self._qimage.height()

    @property
    def extent(self):
        return Extent(self.width, self.height)

    @staticmethod
    def from_base64(data: str):
        bytes = QByteArray.fromBase64(data.encode("utf-8"))
        img = QImage.fromData(bytes, "PNG").convertToFormat(QImage.Format_ARGB32)
        return Image(img)

    @staticmethod
    def scale(img, target: Union[Extent, int]):
        mode = Qt.AspectRatioMode.IgnoreAspectRatio
        if isinstance(target, int):
            mode = Qt.AspectRatioMode.KeepAspectRatio
            target = Extent(target, target)
        scaled = img._qimage.scaled(
            target.width, target.height, mode, Qt.TransformationMode.SmoothTransformation
        )
        return Image(scaled.convertToFormat(QImage.Format_ARGB32))

    @staticmethod
    def sub_region(img, bounds: Bounds):
        return Image(img._qimage.copy(*bounds))

    def pixel(self, x: int, y: int):
        c = self._qimage.pixel(x, y)
        return (qRed(c), qGreen(c), qBlue(c), qAlpha(c))

    def set_pixel(self, x: int, y: int, color: Tuple[int, int, int, int]):
        r, g, b, a = color
        self._qimage.setPixel(x, y, qRgba(r, g, b, a))

    @property
    def data(self):
        ptr = self._qimage.bits()
        ptr.setsize(self._qimage.byteCount())
        return QByteArray(ptr.asstring())

    def to_base64(self):
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QBuffer.WriteOnly)
        self._qimage.save(buffer, "PNG")
        buffer.close()
        return byte_array.toBase64().data().decode("utf-8")

    def to_icon(self):
        return QIcon(QPixmap.fromImage(self._qimage))

    def save(self, filepath: Union[str, Path]):
        success = self._qimage.save(str(filepath))
        assert success, "Failed to save image to f{filepath}"

    def debug_save(self, name):
        if settings.debug_image_folder:
            self.save(Path(settings.debug_image_folder, f"{name}.png"))

    def __eq__(self, other):
        return self._qimage == other._qimage


class ImageCollection:
    def __init__(self, items: Iterable[Image] = ...):
        self._items = []
        if items is not ...:
            self.append(items)

    def append(self, items: Iterable[Image]):
        for item in items:
            if isinstance(item, ImageCollection):
                self._items.extend(item)
            else:
                assert isinstance(item, Image)
                self._items.append(item)

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

    def save(self, filepath: Union[Path, str]):
        filepath = Path(filepath)
        suffix = filepath.suffix
        filepath = filepath.with_suffix("")
        for i, img in enumerate(self._items):
            img.save(filepath.with_name(f"{filepath.name}_{i}").with_suffix(suffix))

    def debug_save(self, name):
        for i, img in enumerate(self._items):
            img.debug_save(f"{name}_{i}")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


class Mask:
    bounds: Bounds
    data: QByteArray

    def __init__(self, bounds: Bounds, data: QByteArray):
        self.bounds = bounds
        self.data = data

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
    def apply(img: Image, mask):
        if img.width == mask.bounds.width and img.height == mask.bounds.height:
            x_offset, y_offset = 0, 0
        else:
            x_offset, y_offset = mask.bounds.offset

        for y in range(img.height):
            for x in range(img.width):
                r, g, b, _ = img.pixel(x, y)
                a = mask.value(x - x_offset, y - y_offset)
                img.set_pixel(x, y, (r, g, b, a))

    def value(self, x: int, y: int):
        if self.bounds.is_within(x, y):
            return int.from_bytes(self.data[y * self.bounds.width + x], "big")
        return 0

    def to_array(self):
        return [x[0] for x in self.data]

    def to_image(self, extent: Extent = ...):
        offset = (self.bounds.x, self.bounds.y)
        if extent is ...:
            extent = self.bounds.extent
            offset = (0, 0)
        img = QImage(extent.width, extent.height, QImage.Format_ARGB32)
        img.fill(0)
        for y in range(self.bounds.height):
            for x in range(self.bounds.width):
                a = self.data[y * self.bounds.width + x][0]
                col = qRgba(a, a, a, 255)
                img.setPixel(offset[0] + x, offset[1] + y, col)
        return Image(img)
