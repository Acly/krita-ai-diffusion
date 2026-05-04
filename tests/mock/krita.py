"""Mock of the krita module that is normally only available inside the Krita application.

Only the subset of the API needed to run the tests outside of Krita is implemented here.
Methods return simple default values unless a test needs to configure specific behaviour.
"""

from __future__ import annotations

from PyQt5.QtCore import QByteArray, QObject, QUuid

IS_MOCK = True


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class Node(QObject):
    """Mock of krita.Node (a layer / mask node in the document tree)."""

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
        self._pixel_data = QByteArray()
        self._animated = False

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

        return QRect(0, 0, 0, 0)

    def pixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        return QByteArray(bytes(w * h * 4))

    def projectionPixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        return QByteArray(bytes(w * h * 4))

    def setPixelData(self, data: QByteArray, x: int, y: int, w: int, h: int) -> bool:
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

    # --- canvas dimensions ---

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def resolution(self) -> float:
        return self._resolution

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
        return QByteArray(bytes(w * h * 4))

    def refreshProjection(self) -> None:
        pass

    # --- transform ---

    def scaleImage(self, w: int, h: int, res_x: float, res_y: float, filter: str) -> None:
        self._width = w
        self._height = h

    def resizeImage(self, x: int, y: int, w: int, h: int) -> None:
        self._width = w
        self._height = h

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


# ---------------------------------------------------------------------------
# Krita application singleton
# ---------------------------------------------------------------------------


class Krita(QObject):
    """Mock of the krita.Krita application object (the Krita singleton)."""

    _instance: Krita | None = None

    def __init__(self):
        super().__init__()
        self._document = Document()

    @staticmethod
    def instance() -> Krita:
        if Krita._instance is None:
            Krita._instance = Krita()
        return Krita._instance

    def activeDocument(self) -> Document | None:
        return self._document

    def documents(self) -> list[Document]:
        return [self._document]

    def setActiveDocument(self, doc: Document | None) -> None:
        self._document = doc
