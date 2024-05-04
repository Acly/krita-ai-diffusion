from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, NamedTuple, cast
from weakref import WeakValueDictionary
import krita
from krita import Krita
from PyQt5.QtCore import QObject, QUuid, QByteArray, QTimer, pyqtSignal
from PyQt5.QtGui import QImage

from .image import Extent, Bounds, Mask, Image
from .pose import Pose
from . import eventloop


class Document(QObject):
    """Document interface. Used as placeholder when there is no open Document in Krita."""

    selection_bounds_changed = pyqtSignal()
    current_time_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

    @property
    def extent(self):
        return Extent(0, 0)

    @property
    def filename(self) -> str:
        return ""

    def check_color_mode(self) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
        return True, None

    def create_mask_from_selection(
        self,
        grow: float = 0.0,
        feather: float = 0.0,
        padding: float = 0.0,
        multiple=8,
        min_size=0,
        square=False,
        invert=False,
    ) -> Mask | None:
        raise NotImplementedError

    def get_mask_bounds(self, layer: krita.Node) -> Bounds:
        raise NotImplementedError

    def get_image(
        self, bounds: Bounds | None = None, exclude_layers: list[krita.Node] | None = None
    ) -> Image:
        raise NotImplementedError

    def get_layer_image(self, layer: krita.Node, bounds: Bounds | None) -> Image:
        raise NotImplementedError

    def get_layer_mask(self, layer: krita.Node, bounds: Bounds | None) -> Image:
        raise NotImplementedError

    def insert_layer(
        self,
        name: str,
        img: Image | None = None,
        bounds: Bounds | None = None,
        make_active=True,
        below: krita.Node | None = None,
    ) -> krita.Node:
        raise NotImplementedError

    def insert_vector_layer(self, name: str, svg: str) -> krita.Node:
        raise NotImplementedError

    def set_layer_content(self, layer: krita.Node, img: Image, bounds: Bounds, make_visible=True):
        raise NotImplementedError

    def hide_layer(self, layer: krita.Node):
        raise NotImplementedError

    def move_to_top(self, layer: krita.Node):
        raise NotImplementedError

    def resize(self, extent: Extent):
        raise NotImplementedError

    def annotate(self, key: str, value: QByteArray):
        pass

    def find_annotation(self, key: str) -> QByteArray | None:
        return None

    def remove_annotation(self, key: str):
        pass

    def add_pose_character(self, layer: krita.Node):
        raise NotImplementedError

    def create_layer_observer(self) -> LayerObserver:
        return LayerObserver(None)

    def import_animation(self, files: list[Path], offset: int = 0):
        raise NotImplementedError

    @property
    def active_layer(self) -> krita.Node:
        raise NotImplementedError

    @active_layer.setter
    def active_layer(self, layer: krita.Node):
        pass

    @property
    def selection_bounds(self) -> Bounds | None:
        return None

    @property
    def resolution(self) -> float:
        return 0.0

    @property
    def playback_time_range(self) -> tuple[int, int]:
        return 0, 0

    @property
    def current_time(self) -> int:
        return 0

    @property
    def is_valid(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return Krita.instance().activeDocument() is None


class KritaDocument(Document):
    """Wrapper around a Krita Document (opened image). Manages multiple image layers and
    allows to retrieve and modify pixel data."""

    _doc: krita.Document
    _id: QUuid
    _poller: QTimer
    _selection_bounds: Bounds | None = None
    _current_time: int = 0
    _instances: WeakValueDictionary[str, KritaDocument] = WeakValueDictionary()

    def __init__(self, krita_document: krita.Document):
        super().__init__()
        self._doc = krita_document
        self._id = krita_document.rootNode().uniqueId()
        self._poller = QTimer()
        self._poller.setInterval(20)
        self._poller.timeout.connect(self._poll)
        self._poller.start()
        self._instances[self._id.toString()] = self

    @classmethod
    def active(cls):
        if doc := Krita.instance().activeDocument():
            id = doc.rootNode().uniqueId().toString()
            return cls._instances.get(id) or KritaDocument(doc)
        return None

    @property
    def extent(self):
        return Extent(self._doc.width(), self._doc.height())

    @property
    def filename(self):
        return self._doc.fileName()

    def check_color_mode(self):
        model = self._doc.colorModel()
        msg_fmt = "Incompatible document: Color {0} must be {1} (current {0}: {2})"
        if model != "RGBA":
            return False, msg_fmt.format("model", "RGB/Alpha", model)
        depth = self._doc.colorDepth()
        if depth != "U8":
            return False, msg_fmt.format("depth", "8-bit integer", depth)
        return True, None

    def create_mask_from_selection(
        self,
        grow: float = 0.0,
        feather: float = 0.0,
        padding: float = 0.0,
        multiple=8,
        min_size=0,
        square=False,
        invert=False,
    ):
        user_selection = self._doc.selection()
        if not user_selection:
            return None

        if _selection_is_entire_document(user_selection, self.extent):
            return None

        selection = user_selection.duplicate()
        original_bounds = Bounds(
            selection.x(), selection.y(), selection.width(), selection.height()
        )
        original_bounds = Bounds.clamp(original_bounds, self.extent)
        size_factor = original_bounds.extent.diagonal
        grow_pixels = int(grow * size_factor)
        feather_radius = int(feather * size_factor)
        padding_pixels = int(padding * size_factor)

        if grow_pixels > 0:
            selection.grow(grow_pixels, grow_pixels)
        if feather_radius > 0:
            selection.feather(feather_radius)
        if invert:
            selection.invert()

        bounds = _selection_bounds(selection)
        bounds = Bounds.pad(
            bounds, padding_pixels, multiple=multiple, min_size=min_size, square=square
        )
        bounds = Bounds.clamp(bounds, self.extent)
        data = selection.pixelData(*bounds)
        return Mask(bounds, data)

    def get_mask_bounds(self, layer: krita.Node):
        assert layer.type() in ["transparencymask", "selectionmask"]
        b = layer.bounds()  # Unfortunately layer.bounds() returns the whole image
        # Use a selection to get just the bounds that contain pixels > 0
        s = krita.Selection()
        data = layer.pixelData(b.x(), b.y(), b.width(), b.height())
        s.setPixelData(data, b.x(), b.y(), b.width(), b.height())
        return Bounds(s.x(), s.y(), s.width(), s.height())

    def get_image(
        self, bounds: Bounds | None = None, exclude_layers: list[krita.Node] | None = None
    ):
        excluded: list[krita.Node] = []
        if exclude_layers:
            for layer in filter(lambda l: l.visible(), exclude_layers):
                layer.setVisible(False)
                excluded.append(layer)
        if len(excluded) > 0:
            self._doc.refreshProjection()

        bounds = bounds or Bounds(0, 0, self._doc.width(), self._doc.height())
        img = QImage(self._doc.pixelData(*bounds), *bounds.extent, QImage.Format.Format_ARGB32)

        for layer in excluded:
            layer.setVisible(True)
        if len(excluded) > 0:
            self._doc.refreshProjection()
        return Image(img)

    def get_layer_image(self, layer: krita.Node, bounds: Bounds | None):
        bounds = bounds or Bounds.from_qrect(layer.bounds())
        data: QByteArray = layer.projectionPixelData(*bounds)
        assert data is not None and data.size() >= bounds.extent.pixel_count * 4
        return Image(QImage(data, *bounds.extent, QImage.Format.Format_ARGB32))

    def get_layer_mask(self, layer: krita.Node, bounds: Bounds | None):
        bounds = bounds or Bounds.from_qrect(layer.bounds())
        if layer.type() in ["transparencymask", "selectionmask"]:
            data: QByteArray = layer.pixelData(*bounds)
            assert data is not None and data.size() >= bounds.extent.pixel_count
            return Image(QImage(data, *bounds.extent, QImage.Format.Format_Grayscale8))
        else:
            img = self.get_layer_image(layer, bounds)
            alpha = img._qimage.convertToFormat(QImage.Format.Format_Alpha8)
            alpha.reinterpretAsFormat(QImage.Format.Format_Grayscale8)
            return Image(alpha)

    def insert_layer(
        self,
        name: str,
        img: Image | None = None,
        bounds: Bounds | None = None,
        make_active=True,
        below: krita.Node | None = None,
    ):
        with RestoreActiveLayer(self) if not make_active else nullcontext():
            layer = self._doc.createNode(name, "paintlayer")
            self._doc.rootNode().addChildNode(layer, _find_layer_above(self._doc, below))
            if img and bounds:
                layer.setPixelData(img.data, *bounds)
                self.refresh(layer)
            return layer

    def insert_vector_layer(self, name: str, svg: str):
        layer = self._doc.createVectorLayer(name)
        self._doc.rootNode().addChildNode(layer, None)
        layer.addShapesFromSvg(svg)
        self.refresh(layer)
        return layer

    def set_layer_content(self, layer: krita.Node, img: Image, bounds: Bounds, make_visible=True):
        layer_bounds = Bounds.from_qrect(layer.bounds())
        if layer_bounds != bounds and not layer_bounds.is_zero:
            # layer.cropNode(*bounds)  <- more efficient, but clutters the undo stack
            blank = Image.create(layer_bounds.extent, fill=0)
            layer.setPixelData(blank.data, *layer_bounds)
        layer.setPixelData(img.data, *bounds)
        if make_visible:
            layer.setVisible(True)
        if layer.visible():
            self.refresh(layer)
        return layer

    def hide_layer(self, layer: krita.Node):
        layer.setVisible(False)
        self.refresh(layer)
        return layer

    def move_to_top(self, layer: krita.Node):
        parent = layer.parentNode()
        if parent.childNodes()[-1] == layer:
            return  # already top-most layer
        with RestoreActiveLayer(self):
            parent.removeChildNode(layer)
            parent.addChildNode(layer, None)

    def resize(self, extent: Extent):
        res = self._doc.resolution()
        self._doc.scaleImage(extent.width, extent.height, res, res, "Bilinear")

    def annotate(self, key: str, value: QByteArray):
        self._doc.setAnnotation(f"ai_diffusion/{key}", f"AI Diffusion Plugin: {key}", value)

    def find_annotation(self, key: str) -> QByteArray | None:
        result = self._doc.annotation(f"ai_diffusion/{key}")
        return result if result.size() > 0 else None

    def remove_annotation(self, key: str):
        self._doc.removeAnnotation(f"ai_diffusion/{key}")

    def add_pose_character(self, layer: krita.Node):
        assert layer.type() == "vectorlayer"
        _pose_layers.add_character(cast(krita.VectorLayer, layer))

    def create_layer_observer(self):
        return LayerObserver(self._doc)

    def import_animation(self, files: list[Path], offset: int = 0):
        success = self._doc.importAnimation([str(f) for f in files], offset, 1)
        if not success and len(files) > 0:
            folder = files[0].parent
            raise RuntimeError(f"Failed to import animation from {folder}")

    def refresh(self, layer: krita.Node):
        # Hacky way of refreshing the projection of a layer, avoids a full document refresh
        layer.setBlendingMode(layer.blendingMode())

    @property
    def active_layer(self):
        return self._doc.activeNode()

    @active_layer.setter
    def active_layer(self, layer: krita.Node):
        self._doc.setActiveNode(layer)

    @property
    def selection_bounds(self):
        return self._selection_bounds

    @property
    def resolution(self):
        return self._doc.resolution() / 72.0  # KisImage::xRes which is applied to vectors

    @property
    def playback_time_range(self):
        return self._doc.playBackStartTime(), self._doc.playBackEndTime()

    @property
    def current_time(self):
        return self._doc.currentTime()

    @property
    def is_valid(self):
        return self._doc in Krita.instance().documents()

    @property
    def is_active(self):
        return self._doc == Krita.instance().activeDocument()

    def _poll(self):
        if self.is_valid:
            selection = self._doc.selection()
            selection_bounds = _selection_bounds(selection) if selection else None
            if selection_bounds != self._selection_bounds:
                self._selection_bounds = selection_bounds
                self.selection_bounds_changed.emit()

            current_time = self.current_time
            if current_time != self._current_time:
                self._current_time = current_time
                self.current_time_changed.emit()
        else:
            self._poller.stop()

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, KritaDocument):
            return self._id == other._id
        return False


def _traverse_layers(node: krita.Node, type_filter=None):
    for child in node.childNodes():
        yield from _traverse_layers(child, type_filter)
        if not type_filter or child.type() in type_filter:
            yield child


def _find_layer_above(doc: krita.Document, layer_below: krita.Node | None):
    if layer_below:
        nodes = doc.rootNode().childNodes()
        index = nodes.index(layer_below)
        if index >= 1:
            return nodes[index - 1]
    return None


def _selection_bounds(selection: krita.Selection):
    return Bounds(selection.x(), selection.y(), selection.width(), selection.height())


def _selection_is_entire_document(selection: krita.Selection, extent: Extent):
    bounds = _selection_bounds(selection)
    if bounds.x > 0 or bounds.y > 0:
        return False
    if bounds.width + bounds.x < extent.width or bounds.height + bounds.y < extent.height:
        return False
    mask = selection.pixelData(*bounds)
    is_opaque = all(x == b"\xff" for x in mask)
    return is_opaque


class RestoreActiveLayer:
    layer: krita.Node | None = None

    def __init__(self, document: Document):
        self.document = document

    def __enter__(self):
        self.layer = self.document.active_layer

    def __exit__(self, exc_type, exc_value, traceback):
        # Some operations like inserting a new layer change the active layer as a side effect.
        # It doesn't happen directly, so changing it back in the same call doesn't work.
        eventloop.run(self._restore())

    async def _restore(self):
        if self.layer:
            if self.layer == self.document.active_layer:
                # Maybe whatever event we expected to change the active layer hasn't happened yet.
                await eventloop.wait_until(
                    lambda: self.document.active_layer != self.layer, no_error=True
                )
            self.document.active_layer = self.layer


class LayerObserver(QObject):
    """Periodically checks the document for changes in the layer structure. Krita doesn't expose
    Python events for these kinds of changes, so we have to poll and compare.
    """

    class Desc(NamedTuple):
        id: QUuid
        name: str
        node: krita.Node

    image_layer_types = [
        "paintlayer",
        "vectorlayer",
        "grouplayer",
        "filelayer",
        "clonelayer",
        "filterlayer",
    ]

    mask_layer_types = ["transparencymask", "selectionmask"]

    changed = pyqtSignal()
    active_changed = pyqtSignal()

    _doc: krita.Document | None
    _layers: list[Desc]
    _active: QUuid | None
    _timer: QTimer

    def __init__(self, doc: krita.Document | None):
        super().__init__()
        self._doc = doc
        self._layers = []
        if doc is not None:
            self._active = doc.activeNode().uniqueId()
            self.update()
            self._timer = QTimer()
            self._timer.setInterval(500)
            self._timer.timeout.connect(self.update)
            self._timer.start()

    def update(self):
        if self._doc is None:
            return
        root_node = self._doc.rootNode()
        if root_node is None:
            return  # Document has been closed

        active = self._doc.activeNode()
        if active is None:
            return

        if active.uniqueId() != self._active:
            self._active = active.uniqueId()
            self.active_changed.emit()

        layers = [self.Desc(l.uniqueId(), l.name(), l) for l in _traverse_layers(root_node)]
        if len(layers) != len(self._layers) or any(
            a.id != b.id or a.name != b.name for a, b in zip(layers, self._layers)
        ):
            self._layers = layers
            self.changed.emit()

    def find(self, id: QUuid):
        return next((l.node for l in self._layers if l.id == id), None)

    def updated(self):
        self.update()
        return self

    def siblings(self, node: krita.Node | None, filter_type: str | None = None):
        below: list[krita.Node] = []
        above: list[krita.Node] = []
        if self._doc is None:
            return below, above

        if node is not None:
            parent = node.parentNode()
            current = below
        else:
            parent = self._doc.rootNode()
            current = above
        for l in self._layers:
            if l.node.parentNode() == parent and (not filter_type or l.node.type() == filter_type):
                if l.node == node:
                    current = above
                else:
                    current.append(l.node)
        return below, above

    @property
    def root(self):
        assert self._doc is not None
        return self._doc.rootNode()

    @property
    def images(self):
        return [l.node for l in self._layers if l.node.type() in self.image_layer_types]

    @property
    def masks(self):
        return [l.node for l in self._layers if l.node.type() in self.mask_layer_types]

    def __iter__(self):
        return (l.node for l in self._layers)

    def __getitem__(self, index: int):
        return self._layers[index].node


class PoseLayers:
    _layers: dict[str, Pose] = {}
    _timer = QTimer()

    def __init__(self):
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    def update(self):
        doc = KritaDocument.active()
        if not doc:
            return
        layer = doc.active_layer
        if not layer or layer.type() != "vectorlayer":
            return

        layer = cast(krita.VectorLayer, layer)
        pose = self._layers.setdefault(layer.uniqueId(), Pose(doc.extent))
        self._update(layer, layer.shapes(), pose, doc.resolution)

    def add_character(self, layer: krita.VectorLayer):
        doc = KritaDocument.active()
        assert doc is not None
        pose = self._layers.setdefault(layer.uniqueId(), Pose(doc.extent))
        svg = Pose.create_default(doc.extent, pose.people_count).to_svg()
        shapes = layer.addShapesFromSvg(svg)
        self._update(layer, shapes, pose, doc.resolution)

    def _update(
        self, layer: krita.VectorLayer, shapes: list[krita.Shape], pose: Pose, resolution: float
    ):
        changes = pose.update(shapes, resolution)  # type: ignore
        if changes:
            shapes = layer.addShapesFromSvg(changes)
            for shape in shapes:
                shape.setZIndex(-1)


_pose_layers = PoseLayers()
