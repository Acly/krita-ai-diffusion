"""Mock of the krita module that is normally only available inside the Krita application.

Only the subset of the API needed to run the tests outside of Krita is implemented here.
Methods return simple default values unless a test needs to configure specific behaviour.
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, QObject, QRect, Qt, QUuid, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QDockWidget, QDoubleSpinBox, QHBoxLayout, QSlider, QWidget

IS_MOCK = True


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class Action(QObject):
    """Minimal stub for a Krita action (QAction equivalent)."""

    triggered = pyqtSignal()

    def __init__(self, name: str = ""):
        super().__init__()
        self._name = name

    def trigger(self) -> None:
        self.triggered.emit()

    def setEnabled(self, enabled: bool) -> None:
        pass


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


class Notifier(QObject):
    """Stub for Krita's Notifier object."""

    applicationClosing = pyqtSignal()
    imageCreated = pyqtSignal()
    imageClosed = pyqtSignal()
    imageSaved = pyqtSignal()
    viewCreated = pyqtSignal()
    viewClosed = pyqtSignal()
    windowCreated = pyqtSignal()

    def __init__(self):
        super().__init__()

    def setActive(self, active: bool) -> None:
        pass


# ---------------------------------------------------------------------------
# View / Canvas
# ---------------------------------------------------------------------------


class View(QObject):
    """Stub for a Krita View."""

    def __init__(self):
        super().__init__()


class Canvas(QObject):
    """Stub for a Krita Canvas."""

    def __init__(self):
        super().__init__()
        self._view: View | None = None

    def view(self) -> View | None:
        return self._view


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


class Window(QObject):
    """Stub for a Krita Window."""

    def __init__(self):
        super().__init__()

    def createAction(self, name: str, text: str = "", menu: str = "") -> Action:
        return Action(name)


# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------


class Extension(QObject):
    """Stub base class for Krita extensions."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def setup(self) -> None:
        pass

    def createActions(self, window: Window) -> None:
        pass


# ---------------------------------------------------------------------------
# DockWidget
# ---------------------------------------------------------------------------


class DockWidget(QDockWidget):
    """Stub base class matching Krita's DockWidget (a QDockWidget with canvasChanged)."""

    def __init__(self):
        super().__init__()

    def canvasChanged(self, canvas: Canvas) -> None:
        """Called by Krita when the active canvas changes; override in subclasses."""


# ---------------------------------------------------------------------------
# DockWidgetFactory / DockWidgetFactoryBase
# ---------------------------------------------------------------------------


class DockWidgetFactoryBase:
    DockLeft = 0
    DockRight = 1
    DockTop = 2
    DockBottom = 3


class DockWidgetFactory(DockWidgetFactoryBase):
    def __init__(self, name: str, position: int, widget_class: type):
        self._name = name
        self._position = position
        self._widget_class = widget_class


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkerboard(width: int, height: int, square: int = 32) -> bytes:
    """Return BGRA bytes for a neutral gray checkerboard pattern (no per-pixel loop)."""
    light_px = bytes([210, 210, 210, 255])
    dark_px = bytes([170, 170, 170, 255])

    def _row(start_light: bool) -> bytes:
        row: bytearray = bytearray()
        x, light = 0, start_light
        while x < width:
            n = min(square, width - x)
            row += (light_px if light else dark_px) * n
            light = not light
            x += n
        return bytes(row)

    row_a, row_b = _row(True), _row(False)
    result: bytearray = bytearray()
    y, sq_row = 0, 0
    while y < height:
        n = min(square, height - y)
        result += (row_a if sq_row % 2 == 0 else row_b) * n
        y += n
        sq_row += 1
    return bytes(result)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class Node(QObject):
    """Mock of krita.Node (a layer / mask node in the document tree)."""

    pixelDataChanged = pyqtSignal()

    def __init__(
        self, name: str = "layer", node_type: str = "paintlayer", parent: Node | None = None
    ):
        super().__init__()
        self._name = name
        self._type = node_type
        self._parent = parent
        self._children: list[Node] = []
        self._document: Document | None = None  # set by Document when added to the tree
        self._visible = True
        self._locked = False
        self._blending_mode = "normal"
        self._unique_id = QUuid.createUuid()
        self._animated = False
        # Pixel storage: flat BGRA bytes, stride = _pixel_width * 4.
        self._pixel_data: bytearray = bytearray()
        self._bounds = (
            0,
            0,
            0,
            0,
        )  # (x, y, width, height) of the non-transparent pixels in _pixel_data

    # --- tree navigation ---

    def uniqueId(self) -> QUuid:
        return self._unique_id

    def name(self) -> str:
        return self._name

    def setName(self, name: str) -> None:
        self._name = name

    def type(self) -> str:
        return self._type

    def parentNode(self) -> Node | None:
        return self._parent

    def childNodes(self) -> list[Node]:
        return list(self._children)

    def addChildNode(self, node: Node, above: Node | None) -> bool:
        node._parent = self
        node._document = self._document
        if above is None:
            self._children.append(node)
        else:
            try:
                idx = self._children.index(above)
            except ValueError:
                self._children.append(node)
            else:
                self._children.insert(idx, node)
        if self._document is not None:
            if node._type == "paintlayer":
                node.pixelDataChanged.connect(self._document.pixelDataChanged)
            self._document.nodesChanged.emit()
        return True

    def removeChildNode(self, node: Node) -> None:
        self._children = [c for c in self._children if c is not node]
        if self._document is not None:
            self._document.nodesChanged.emit()

    def remove(self) -> None:
        if self._parent is not None:
            self._parent.removeChildNode(self)

    # --- properties ---

    def visible(self) -> bool:
        return self._visible

    def setVisible(self, value: bool) -> None:
        self._visible = value

    def locked(self) -> bool:
        return self._locked

    def setLocked(self, value: bool) -> None:
        self._locked = value

    def blendingMode(self) -> str:
        return self._blending_mode

    def setBlendingMode(self, mode: str) -> None:
        self._blending_mode = mode

    def animated(self) -> bool:
        return self._animated

    def hasKeyframeAtTime(self, time: int) -> bool:
        return False

    def colorDepth(self) -> str:
        return "U8"

    # --- pixel data ---

    def bounds(self):
        return QRect(*self._bounds)

    def pixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        from ai_diffusion.image import Bounds, Image

        b = self._bounds
        if b[2] == 0 or b[3] == 0:
            return QByteArray(bytes(w * h * 4))

        img = self._to_image()
        img = Image.crop(img, Bounds(x, y, w, h))
        return img.to_packed_bytes()

    def projectionPixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        if self._type == "grouplayer":
            from ai_diffusion.image import BlendMode, Extent, Image

            result = Image.create(Extent(w, h), fill=0)
            # _children is ordered bottom-to-top (last element = topmost layer);
            # composite bottom-to-top so the last child ends up on top.
            for child in self._children:
                if child._visible:
                    child_data = child.projectionPixelData(x, y, w, h)
                    child_img = Image.from_packed_bytes(child_data, Extent(w, h))
                    result.draw_image(child_img, (0, 0), blend=BlendMode.alpha)
            return result.to_packed_bytes()
        return self.pixelData(x, y, w, h)

    def setPixelData(self, value: QByteArray, x: int, y: int, w: int, h: int) -> bool:
        from ai_diffusion.image import BlendMode, Bounds, Extent, Image

        b = self._bounds
        if b[2] == 0 or b[3] == 0:
            self._bounds = (x, y, w, h)
            self._pixel_data = bytearray(value.data())
            self.pixelDataChanged.emit()
            return True

        if x < b[0] or y < b[1] or x + w > b[0] + b[2] or y + h > b[1] + b[3]:
            # expand bounds to include the new data
            x0 = min(x, b[0])
            y0 = min(y, b[1])
            x1 = max(x + w, b[0] + b[2])
            y1 = max(y + h, b[1] + b[3])
            self._bounds = (x0, y0, x1 - x0, y1 - y0)
            # expand pixel data to match the new bounds
            img = self._to_image()
            img = Image.crop(img, Bounds(*self._bounds))
            self._pixel_data = bytearray(img.to_packed_bytes().data())

        src = Image.from_packed_bytes(value, Extent(w, h))
        dst = self._to_image()
        dst.draw_image(src, (x, y), blend=BlendMode.replace)
        self._pixel_data = bytearray(dst.to_packed_bytes().data())
        self.pixelDataChanged.emit()
        return True

    def pixelDataAtTime(self, x: int, y: int, w: int, h: int, time: int) -> QByteArray:
        return QByteArray(bytes(w * h * 4))

    def thumbnail(self, w: int, h: int):
        from PyQt6.QtGui import QImage

        return QImage(w, h, QImage.Format.Format_ARGB32)

    def _to_image(self):
        from ai_diffusion.image import Extent, Image

        e = Extent(*self._bounds[2:])
        return Image.from_packed_bytes(QByteArray(self._pixel_data), e)


def _traverse_nodes(node: Node):
    yield node
    for child in node.childNodes():
        yield from _traverse_nodes(child)


# ---------------------------------------------------------------------------
# VectorLayer
# ---------------------------------------------------------------------------


class VectorLayer(Node):
    def __init__(self, name: str = "vector layer"):
        super().__init__(name, "vectorlayer")
        self._shapes: list[Shape] = []

    def shapes(self) -> list[Shape]:
        return list(self._shapes)

    def addShapesFromSvg(self, svg: str) -> list[Shape]:
        shape = Shape()
        self._shapes.append(shape)
        return [shape]


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class Shape(QObject):
    def __init__(self):
        super().__init__()
        self._z_index = 0

    def setZIndex(self, z: int) -> None:
        self._z_index = z

    def zIndex(self) -> int:
        return self._z_index


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class Selection(QObject):
    def __init__(self):
        super().__init__()
        self._x = 0
        self._y = 0
        self._width = 0
        self._height = 0
        self._data = QByteArray()

    def x(self) -> int:
        return self._x

    def y(self) -> int:
        return self._y

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def pixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        result = bytearray(w * h)  # default: fully unselected (0)
        if not self._data or self._width == 0 or self._height == 0:
            return QByteArray(bytes(result))
        # Intersection of requested rect with stored rect
        x0 = max(x, self._x)
        x1 = min(x + w, self._x + self._width)
        y0 = max(y, self._y)
        y1 = min(y + h, self._y + self._height)
        if x0 >= x1 or y0 >= y1:
            return QByteArray(bytes(result))
        src = self._data.data()
        n = x1 - x0
        for ry in range(y0, y1):
            src_off = (ry - self._y) * self._width + (x0 - self._x)
            dst_off = (ry - y) * w + (x0 - x)
            result[dst_off : dst_off + n] = src[src_off : src_off + n]
        return QByteArray(bytes(result))

    def setPixelData(self, data: QByteArray, x: int, y: int, w: int, h: int) -> None:
        self._data = data
        self._x = x
        self._y = y
        self._width = w
        self._height = h

    def duplicate(self) -> Selection:
        copy = Selection()
        copy._x = self._x
        copy._y = self._y
        copy._width = self._width
        copy._height = self._height
        copy._data = QByteArray(self._data)
        return copy

    def invert(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class Document(QObject):
    """Mock of krita.Document (an open image/canvas in Krita)."""

    pixelDataChanged = pyqtSignal()
    nodesChanged = pyqtSignal()
    activeNodeChanged = pyqtSignal(object)  # emits Node | None

    def __init__(self, width: int = 512, height: int = 512):
        super().__init__()
        self._width = width
        self._height = height
        self._color_model = "RGBA"
        self._color_depth = "U8"
        self._resolution = 72.0
        self._filename = ""
        self._current_time = 0
        self._playback_start = 0
        self._playback_end = 0
        self._selection: Selection | None = None
        self._annotations: dict[str, QByteArray] = {}

        # Build a minimal layer tree: a root group node with one paint layer.
        self._root = Node("root", "grouplayer")
        self._root._document = self
        background = Node("Background", "paintlayer", parent=self._root)
        background._document = self
        background.pixelDataChanged.connect(self.pixelDataChanged)
        self._root._children = [background]
        self._active_node: Node = background

        # Initialise the background layer with a visible checkerboard.
        background.setPixelData(QByteArray(_make_checkerboard(width, height)), 0, 0, width, height)

    # --- canvas dimensions ---

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def resolution(self) -> float:
        return self._resolution

    def setResolution(self, xres: float, yres: float) -> None:
        self._resolution = xres

    # --- color ---

    def colorModel(self) -> str:
        return self._color_model

    def colorDepth(self) -> str:
        return self._color_depth

    # --- file ---

    def fileName(self) -> str:
        return self._filename

    # --- layer tree ---

    def rootNode(self) -> Node:
        return self._root

    def activeNode(self) -> Node | None:
        return self._active_node

    def setActiveNode(self, node: Node) -> None:
        self._active_node = node
        self.activeNodeChanged.emit(node)

    def createNode(self, name: str, node_type: str) -> Node:
        return Node(name, node_type)

    def createVectorLayer(self, name: str) -> VectorLayer:
        return VectorLayer(name)

    def createGroupLayer(self, name: str) -> Node:
        return Node(name, "grouplayer")

    def createTransparencyMask(self, name: str) -> Node:
        return Node(name, "transparencymask")

    # --- selection ---

    def selection(self) -> Selection | None:
        return self._selection

    def setSelection(self, selection: Selection | None) -> None:
        self._selection = selection

    # --- pixel data ---

    def pixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        from ai_diffusion.image import Extent, Image

        projection = Image.flatten(
            Image.from_packed_bytes(n.pixelData(x, y, w, h), Extent(w, h))
            for n in _traverse_nodes(self._root)
            if n._type == "paintlayer"
        )
        return projection.to_packed_bytes()

    def refreshProjection(self) -> None:
        pass

    # --- transform ---

    def scaleImage(self, w: int, h: int, res_x: float, res_y: float, filter: str) -> None:
        self._width = w
        self._height = h
        cb = QByteArray(_make_checkerboard(w, h))
        for node in self._root.childNodes():
            if node._type == "paintlayer":
                node.setPixelData(cb, 0, 0, w, h)

    def resizeImage(self, x: int, y: int, w: int, h: int) -> None:
        self._width = w
        self._height = h
        cb = QByteArray(_make_checkerboard(w, h))
        for node in self._root.childNodes():
            if node._type == "paintlayer":
                node.setPixelData(cb, 0, 0, w, h)

    # --- annotations ---

    def annotation(self, key: str) -> QByteArray:
        return self._annotations.get(key, QByteArray())

    def setAnnotation(self, key: str, description: str, data: QByteArray) -> None:
        self._annotations[key] = data

    def removeAnnotation(self, key: str) -> None:
        self._annotations.pop(key, None)

    # --- animation ---

    def currentTime(self) -> int:
        return self._current_time

    def playBackStartTime(self) -> int:
        return self._playback_start

    def playBackEndTime(self) -> int:
        return self._playback_end

    def importAnimation(self, files: list[str], offset: int, step: int) -> bool:
        return True

    def close(self) -> bool:
        krita = Krita.instance()
        if self not in krita._documents:
            return False
        krita._documents = [d for d in krita._documents if d is not self]
        if krita._active_document is self:
            krita._active_document = None
        return True


# ---------------------------------------------------------------------------
# Krita application singleton
# ---------------------------------------------------------------------------


class Krita(QObject):
    """Mock of the krita.Krita application object (the Krita singleton)."""

    _instance: Krita | None = None

    def __init__(self):
        super().__init__()
        self._documents: list[Document] = []
        self._active_document: Document | None = None
        self._notifier = Notifier()
        self._actions: dict[str, Action] = {}

    @staticmethod
    def instance() -> Krita:
        if Krita._instance is None:
            Krita._instance = Krita()
        return Krita._instance

    def version(self) -> str:
        return "5.2.0"

    def icon(self, name: str) -> QIcon:
        return QIcon()

    def action(self, name: str) -> Action:
        if name not in self._actions:
            self._actions[name] = Action(name)
        return self._actions[name]

    def notifier(self) -> Notifier:
        return self._notifier

    def addExtension(self, extension: Extension) -> None:
        pass

    def addDockWidgetFactory(self, factory: DockWidgetFactory) -> None:
        pass

    def activeDocument(self) -> Document | None:
        return self._active_document

    def documents(self) -> list[Document]:
        return self._documents

    def setActiveDocument(self, doc: Document | None) -> None:
        self._active_document = doc

    def openDocument(self, filename: str) -> Document:
        # If there is already an active document that hasn't been registered yet
        # (simulates Krita completing the load of a pending document), register
        # it and return it rather than creating a new one.
        if self._active_document is not None and self._active_document not in self._documents:
            self._documents.append(self._active_document)
            return self._active_document
        doc = Document()
        doc._filename = filename
        self._documents.append(doc)
        self._active_document = doc
        return doc


class DoubleParseSpinBox(QWidget):
    def __init__(self):
        super().__init__()

        self._layout = QHBoxLayout(self)
        self.setLayout(self._layout)

        self._widget = QDoubleSpinBox()
        self._layout.addWidget(self._widget)

    def widget(self):
        return self._widget

    def stepBy(self, steps: int):
        self._widget.stepBy(steps)


class DoubleSliderSpinBox(DoubleParseSpinBox):
    draggingFinished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._layout.insertWidget(0, self._slider)

        self._slider.valueChanged.connect(self._update_spinbox_from_slider)
        self.widget().valueChanged.connect(self._update_slider_from_spinbox)

    def setRange(self, min: float, max: float, decimals: int = 0, compute_fast_step: bool = True):
        self.widget().setRange(min, max)
        self.widget().setDecimals(decimals)
        self._slider.setRange(int(min), int(max))

    def setSoftMinimum(self, min: float):
        pass

    def setSoftMaximum(self, max: float):
        pass

    def _update_slider_from_spinbox(self):
        value = self.widget().value()
        if abs(self._slider.value() / 100.0 - value) > 0.01:
            self._slider.setValue(int(value * 100))

    def _update_spinbox_from_slider(self):
        value = self._slider.value() / 100.0
        if abs(self.widget().value() - value) > 0.01:
            self.widget().setValue(value)

    def setValue(self, value: float):
        self.widget().setValue(value)
        self._slider.setValue(int(value * 100))

    def value(self):
        return self.widget().value()

    def isDragging(self) -> bool:
        return self._slider.isSliderDown()
