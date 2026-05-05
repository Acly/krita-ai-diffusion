"""Mock of the krita module that is normally only available inside the Krita application.

Only the subset of the API needed to run the tests outside of Krita is implemented here.
Methods return simple default values unless a test needs to configure specific behaviour.
"""

from __future__ import annotations

from PyQt5.QtCore import QByteArray, QObject, QUuid, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDockWidget

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
        self._visible = True
        self._locked = False
        self._blending_mode = "normal"
        self._unique_id = QUuid.createUuid()
        self._animated = False
        # Pixel storage: flat BGRA bytes, stride = _pixel_width * 4.
        self._pixel_data: bytearray = bytearray()
        self._pixel_width: int = 0

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
        if above is None:
            self._children.append(node)
        else:
            try:
                idx = self._children.index(above)
            except ValueError:
                self._children.append(node)
            else:
                self._children.insert(idx, node)
        return True

    def removeChildNode(self, node: Node) -> None:
        self._children = [c for c in self._children if c is not node]

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
        from PyQt5.QtCore import QRect

        if self._pixel_width > 0 and self._pixel_data:
            return QRect(0, 0, self._pixel_width, len(self._pixel_data) // (self._pixel_width * 4))
        return QRect(0, 0, 0, 0)

    def pixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        if not self._pixel_data or self._pixel_width == 0:
            return QByteArray(bytes(w * h * 4))
        result = bytearray()
        for row in range(y, y + h):
            start = (row * self._pixel_width + x) * 4
            result += self._pixel_data[start : start + w * 4]
        return QByteArray(bytes(result))

    def projectionPixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        return self.pixelData(x, y, w, h)

    def setPixelData(self, value: QByteArray, x: int, y: int, w: int, h: int) -> bool:
        src = bytes(value)
        if x == 0 and y == 0 and (not self._pixel_data or self._pixel_width != w):
            # Full or first write – store as-is and record stride.
            self._pixel_width = w
            self._pixel_data = bytearray(src)
        else:
            if not self._pixel_data:
                return False
            for row in range(h):
                dst = ((y + row) * self._pixel_width + x) * 4
                self._pixel_data[dst : dst + w * 4] = src[row * w * 4 : (row + 1) * w * 4]
        self.pixelDataChanged.emit()
        return True

    def pixelDataAtTime(self, x: int, y: int, w: int, h: int, time: int) -> QByteArray:
        return QByteArray(bytes(w * h * 4))

    def thumbnail(self, w: int, h: int):
        from PyQt5.QtGui import QImage

        return QImage(w, h, QImage.Format.Format_ARGB32)


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
        return self._data

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
        self._paint_layer = Node("Background", "paintlayer", parent=self._root)
        self._root._children = [self._paint_layer]
        self._active_node: Node = self._paint_layer

        # Initialise the background layer with a visible checkerboard and
        # forward its pixelDataChanged signal to the document level.
        self._paint_layer.setPixelData(
            QByteArray(_make_checkerboard(width, height)), 0, 0, width, height
        )
        self._paint_layer.pixelDataChanged.connect(self.pixelDataChanged)

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

    # --- pixel data ---

    def pixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        """Return composite pixel data by delegating to the first paint layer."""
        return self._paint_layer.pixelData(x, y, w, h)

    def refreshProjection(self) -> None:
        pass

    # --- transform ---

    def scaleImage(self, w: int, h: int, res_x: float, res_y: float, filter: str) -> None:
        self._width = w
        self._height = h
        self._paint_layer.setPixelData(QByteArray(_make_checkerboard(w, h)), 0, 0, w, h)

    def resizeImage(self, x: int, y: int, w: int, h: int) -> None:
        self._width = w
        self._height = h
        self._paint_layer.setPixelData(QByteArray(_make_checkerboard(w, h)), 0, 0, w, h)

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
