from __future__ import annotations
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from typing import Literal, NamedTuple, cast
from weakref import WeakValueDictionary
import krita
from krita import Krita
from PyQt5.QtCore import QObject, QUuid, QByteArray, QTimer, pyqtSignal
from PyQt5.QtGui import QImage

from .image import Extent, Bounds, Mask, Image
from .pose import Pose
from .util import ensure
from . import eventloop


class Document(QObject):
    """Document interface. Used as placeholder when there is no open Document in Krita."""

    selection_bounds_changed = pyqtSignal()
    current_time_changed = pyqtSignal()

    _layers: LayerObserver

    def __init__(self):
        super().__init__()
        self._layers = LayerObserver(None)

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

    def get_image(
        self, bounds: Bounds | None = None, exclude_layers: list[Layer] | None = None
    ) -> Image:
        raise NotImplementedError

    def resize(self, extent: Extent):
        raise NotImplementedError

    def annotate(self, key: str, value: QByteArray):
        pass

    def find_annotation(self, key: str) -> QByteArray | None:
        return None

    def remove_annotation(self, key: str):
        pass

    def add_pose_character(self, layer: Layer):
        raise NotImplementedError

    def import_animation(self, files: list[Path], offset: int = 0):
        raise NotImplementedError

    @property
    def layers(self) -> LayerObserver:
        return self._layers

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
    _layers: LayerObserver
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
        self._layers = LayerObserver(krita_document)

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

    @property
    def layers(self):
        return self._layers

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

    def get_image(self, bounds: Bounds | None = None, exclude_layers: list[Layer] | None = None):
        excluded: list[Layer] = []
        if exclude_layers:
            for layer in filter(lambda l: l.is_visible, exclude_layers):
                layer.hide()
                excluded.append(layer)
        if len(excluded) > 0:
            self._doc.refreshProjection()

        bounds = bounds or Bounds(0, 0, self._doc.width(), self._doc.height())
        img = QImage(self._doc.pixelData(*bounds), *bounds.extent, QImage.Format.Format_ARGB32)

        for layer in excluded:
            layer.show()
        if len(excluded) > 0:
            self._doc.refreshProjection()
        return Image(img)

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

    def add_pose_character(self, layer: Layer):
        assert layer.type is LayerType.vector
        _pose_layers.add_character(cast(krita.VectorLayer, layer.node))

    def import_animation(self, files: list[Path], offset: int = 0):
        success = self._doc.importAnimation([str(f) for f in files], offset, 1)
        if not success and len(files) > 0:
            folder = files[0].parent
            raise RuntimeError(f"Failed to import animation from {folder}")

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


def _traverse_layers(node: krita.Node, type_filter: list[str] | None = None):
    for child in node.childNodes():
        yield from _traverse_layers(child, type_filter)
        if not type_filter or child.type() in type_filter:
            yield child


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
    layer: Layer | None = None

    def __init__(self, layers: LayerObserver):
        self._observer = layers

    def __enter__(self):
        self.layer = self._observer.active

    def __exit__(self, exc_type, exc_value, traceback):
        # Some operations like inserting a new layer change the active layer as a side effect.
        # It doesn't happen directly, so changing it back in the same call doesn't work.
        eventloop.run(self._restore())

    async def _restore(self):
        if self.layer:
            if self.layer.is_active:
                # Maybe whatever event we expected to change the active layer hasn't happened yet.
                await eventloop.wait_until(
                    lambda: self.layer is not None and not self.layer.is_active, no_error=True
                )
            self._observer.active = self.layer


class LayerType(Enum):
    paint = "paintlayer"
    vector = "vectorlayer"
    group = "grouplayer"
    file = "filelayer"
    clone = "clonelayer"
    filter = "filterlayer"
    transparency = "transparencymask"
    selection = "selectionmask"

    @property
    def is_image(self):
        return not self.is_mask

    @property
    def is_mask(self):
        return self in [LayerType.transparency, LayerType.selection]


class Layer(QObject):
    """Wrapper around a Krita Node. Provides convenience methods and polling-based events."""

    _observer: LayerObserver
    _node: krita.Node
    _name: str

    renamed = pyqtSignal(str)
    removed = pyqtSignal()

    def __init__(self, observer: LayerObserver, node: krita.Node):
        super().__init__()
        self._observer = observer
        self._node = node
        self._name = node.name()

    @property
    def id(self):
        return self._node.uniqueId()

    @property
    def id_string(self):
        return self._node.uniqueId().toString()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if self._name == value:
            return
        self._name = value
        self._node.setName(value)
        self.renamed.emit(value)

    @property
    def type(self):
        return LayerType(self._node.type())

    @property
    def was_removed(self):
        return self._observer.updated().find(self.id) is None

    @property
    def is_visible(self):
        return self._node.visible()

    @is_visible.setter
    def is_visible(self, value):
        self._node.setVisible(value)

    def hide(self):
        self._node.setVisible(False)

    def show(self):
        self._node.setVisible(True)

    @property
    def is_active(self):
        return self is self._observer.active

    @property
    def is_locked(self):
        return self._node.locked()

    @is_locked.setter
    def is_locked(self, value):
        self._node.setLocked(value)

    @property
    def bounds(self):
        return Bounds.from_qrect(self._node.bounds())

    @property
    def parent_layer(self):
        if parent := self._node.parentNode():
            return self._observer.wrap(parent)
        return None

    @property
    def child_layers(self):
        return [self._observer.wrap(child) for child in self._node.childNodes()]

    @property
    def is_root(self):
        return self._node.parentNode() is None

    def get_pixels(self, bounds: Bounds | None = None, time: int | None = None):
        bounds = bounds or self.bounds
        if time is None:
            data: QByteArray = self._node.projectionPixelData(*bounds)
        else:
            data: QByteArray = self._node.pixelDataAtTime(time, *bounds)
        assert data is not None and data.size() >= bounds.extent.pixel_count * 4
        return Image(QImage(data, *bounds.extent, QImage.Format.Format_ARGB32))

    def write_pixels(self, img: Image, bounds: Bounds | None = None, make_visible=True):
        layer_bounds = self.bounds
        bounds = bounds or layer_bounds
        if layer_bounds != bounds and not layer_bounds.is_zero:
            # layer.cropNode(*bounds)  <- more efficient, but clutters the undo stack
            blank = Image.create(layer_bounds.extent, fill=0)
            self._node.setPixelData(blank.data, *layer_bounds)
        self._node.setPixelData(img.data, *bounds)
        if make_visible:
            self.show()
        if self.is_visible:
            self.refresh()

    def get_mask(self, bounds: Bounds | None):
        bounds = bounds or self.bounds
        if self.type.is_mask:
            data: QByteArray = self._node.pixelData(*bounds)
            assert data is not None and data.size() >= bounds.extent.pixel_count
            return Image(QImage(data, *bounds.extent, QImage.Format.Format_Grayscale8))
        else:
            img = self.get_pixels(bounds)
            alpha = img._qimage.convertToFormat(QImage.Format.Format_Alpha8)
            alpha.reinterpretAsFormat(QImage.Format.Format_Grayscale8)
            return Image(alpha)

    def move_to_top(self):
        parent = self._node.parentNode()
        if parent.childNodes()[-1] == self._node:
            return  # already top-most layer
        with RestoreActiveLayer(self._observer):
            parent.removeChildNode(self.node)
            parent.addChildNode(self.node, None)

    def refresh(self):
        # Hacky way of refreshing the projection of a layer, avoids a full document refresh
        self._node.setBlendingMode(self._node.blendingMode())

    def thumbnail(self, size: Extent):
        return self.node.thumbnail(*size)

    def remove(self):
        self._node.remove()
        self._observer.update()

    def compute_bounds(self):
        bounds = self.bounds
        if self.type.is_mask:
            # Unfortunately node.bounds() returns the whole image
            # Use a selection to get just the bounds that contain pixels > 0
            s = krita.Selection()
            data = self.node.pixelData(*bounds)
            s.setPixelData(data, *bounds)
            return Bounds(s.x(), s.y(), s.width(), s.height())
        elif self.type is LayerType.group:
            for child in self.child_layers:
                if child.type is LayerType.transparency:
                    bounds = child.compute_bounds()
        return bounds

    @property
    def siblings(self):
        below: list[Layer] = []
        above: list[Layer] = []
        parent = self.parent_layer

        if parent is None:
            return below, above

        current = below
        for l in parent.child_layers:
            if l == self:
                current = above
            else:
                current.append(l)
        return below, above

    @property
    def sibling_above(self):
        nodes = ensure(self.parent_layer).child_layers
        index = nodes.index(self)
        if index >= 1:
            return nodes[index - 1]
        return self

    @property
    def is_animated(self):
        return self._node.animated()

    @property
    def node(self):
        return self._node

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Layer):
            return self.id == other.id
        return False


class LayerObserver(QObject):
    """Periodically checks the document for changes in the layer structure. Krita doesn't expose
    Python events for these kinds of changes, so we have to poll and compare.
    """

    changed = pyqtSignal()
    active_changed = pyqtSignal()

    _doc: krita.Document | None
    _root: Layer | None
    _layers: dict[QUuid, Layer]
    _active: QUuid
    _timer: QTimer

    def __init__(self, doc: krita.Document | None):
        super().__init__()
        self._doc = doc
        self._layers = {}
        if doc is not None:
            root = doc.rootNode()
            self._root = Layer(self, root)
            self._layers = {self._root.id: self._root}
            self._active = doc.activeNode().uniqueId()
            self.update()
            self._timer = QTimer()
            self._timer.setInterval(500)
            self._timer.timeout.connect(self.update)
            self._timer.start()
        else:
            self._root = None
            self._active = QUuid()

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

        removals = set(self._layers.keys())
        changes = False
        for n in _traverse_layers(root_node):
            id = n.uniqueId()
            if id in self._layers:
                removals.remove(id)
                layer = self._layers[id]
                if layer.name != n.name():
                    layer.name = n.name()
                    changes = True
            else:
                self._layers[id] = Layer(self, n)
                changes = True

        removals.remove(self.root.id)
        for id in removals:
            self._layers[id].removed.emit()
            del self._layers[id]

        if removals or changes:
            self.changed.emit()

    def wrap(self, node: krita.Node) -> Layer:
        layer = self.find(node.uniqueId())
        if layer is None:
            layer = self.updated()._layers[node.uniqueId()]
        return layer

    def find(self, id: QUuid) -> Layer | None:
        if self._doc is None:
            return None
        return self._layers.get(id)

    def updated(self):
        self.update()
        return self

    @property
    def root(self):
        assert self._root is not None
        return self._root

    @property
    def active(self):
        assert self._doc is not None
        layer = self.find(self._doc.activeNode().uniqueId())
        if layer is None:
            layer = self.updated()._layers[self._active]
        return layer

    @active.setter
    def active(self, layer: Layer):
        if self._doc is not None:
            self._doc.setActiveNode(layer.node)
            self.update()

    def create(
        self,
        name: str,
        img: Image | None = None,
        bounds: Bounds | None = None,
        make_active=True,
        parent: Layer | None = None,
        above: Layer | None = None,
    ):
        doc = ensure(self._doc)
        node = doc.createNode(name, "paintlayer")
        layer = self._insert(node, parent, above, make_active)
        if img and bounds:
            layer.node.setPixelData(img.data, *bounds)
            layer.refresh()
        return layer

    def _insert(
        self,
        node: krita.Node,
        parent: Layer | None = None,
        above: Layer | None = None,
        make_active=True,
    ):
        if above is not None:
            parent = parent or above.parent_layer
        parent = parent or self.root
        with RestoreActiveLayer(self) if not make_active else nullcontext():
            parent.node.addChildNode(node, above.node if above else None)
            return self.updated().wrap(node)

    def create_vector(self, name: str, svg: str):
        doc = ensure(self._doc)
        node = doc.createVectorLayer(name)
        doc.rootNode().addChildNode(node, None)
        node.addShapesFromSvg(svg)
        layer = self.updated().wrap(node)
        layer.refresh()
        return layer

    def create_mask(self, name: str, img: Image, bounds: Bounds, parent: Layer | None = None):
        assert img.is_mask
        doc = ensure(self._doc)
        node = doc.createTransparencyMask(name)
        node.setPixelData(img.data, *bounds)
        return self._insert(node, parent=parent)

    def create_group(self, name: str, above: Layer | None = None):
        doc = ensure(self._doc)
        node = doc.createGroupLayer(name)
        return self._insert(node, above)

    def create_group_for(self, layer: Layer):
        doc = ensure(self._doc)
        group = self.wrap(doc.createGroupLayer(f"{layer.name} Group"))
        parent = ensure(layer.parent_layer, "Cannot group root layer")
        parent.node.addChildNode(group.node, layer.node)
        parent.node.removeChildNode(layer.node)
        group.node.addChildNode(layer.node, None)
        return group

    _image_types = [t.value for t in LayerType if t.is_image]
    _mask_types = [t.value for t in LayerType if t.is_mask]

    @property
    def images(self):
        if self._doc is None:
            return []
        return [self.wrap(n) for n in _traverse_layers(self._doc.rootNode(), self._image_types)]

    @property
    def masks(self):
        if self._doc is None:
            return []
        return [self.wrap(n) for n in _traverse_layers(self._doc.rootNode(), self._mask_types)]

    def __bool__(self):
        return self._doc is not None


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
        layer = doc.layers.active
        if not layer or layer.type is not LayerType.vector:
            return

        layer = cast(krita.VectorLayer, layer.node)
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
