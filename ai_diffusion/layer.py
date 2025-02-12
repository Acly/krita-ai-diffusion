from __future__ import annotations
from contextlib import contextmanager, nullcontext
from enum import Enum
import krita
from PyQt5.QtCore import QObject, QUuid, QByteArray, QTimer, pyqtSignal
from PyQt5.QtGui import QImage

from .image import Extent, Bounds, Image, ImageCollection
from .util import acquire_elements, ensure, maybe, client_logger as log
from . import eventloop


class LayerType(Enum):
    paint = "paintlayer"
    vector = "vectorlayer"
    group = "grouplayer"
    file = "filelayer"
    clone = "clonelayer"
    fill = "filllayer"
    filter = "filterlayer"
    transparency = "transparencymask"
    selection = "selectionmask"
    filtermask = "filtermask"
    transform = "transformmask"
    colorize = "colorizemask"

    @property
    def is_image(self):
        return self in [  # Layers that contain color pixel data
            LayerType.paint,
            LayerType.vector,
            LayerType.group,
            LayerType.file,
            LayerType.clone,
            LayerType.filter,
            LayerType.fill,
        ]

    @property
    def is_mask(self):  # Layers that contain alpha pixel data
        return self in [LayerType.transparency, LayerType.selection]

    @property
    def is_filter(self):
        return self in [  # Layers which modify their parent layer
            LayerType.transparency,
            LayerType.selection,
            LayerType.filtermask,
            LayerType.transform,
            LayerType.colorize,
        ]


class Layer(QObject):
    """Wrapper around a Krita Node. Provides pythonic interface, read and write pixels
    from/to QImage. Exposes some events based on polling done in LayerManager.
    Layer objects are cached, there is a guarantee only one instance exists per layer node.
    """

    def __init__(self, manager: LayerManager, node: krita.Node, is_confirmed=True):
        super().__init__()
        self._manager = manager
        self._node = node
        self._name = node.name()
        self._parent = maybe(krita.Node.uniqueId, node.parentNode())
        self._is_confirmed = is_confirmed

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

    @property
    def type(self):
        return LayerType(self._node.type())

    @property
    def is_confirmed(self):
        return self._is_confirmed

    @property
    def was_removed(self):
        return self._manager.updated().find(self.id) is None

    @property
    def is_visible(self):
        return self._node.visible()

    @is_visible.setter
    def is_visible(self, value):
        self._node.setVisible(value)

    def hide(self):
        self._node.setVisible(False)
        self.refresh()

    def show(self):
        self._node.setVisible(True)
        self.refresh()

    @property
    def is_active(self):
        return self is self._manager.active

    @property
    def is_locked(self):
        return self._node.locked()

    @is_locked.setter
    def is_locked(self, value):
        self._node.setLocked(value)

    @property
    def bounds(self):
        # In Krita layer bounds can be larger than the image - this property clamps them
        bounds = Bounds.from_qrect(self._node.bounds())
        bounds = Bounds.restrict(bounds, Bounds(0, 0, *self._manager.image_extent))
        return bounds

    @property
    def parent_layer(self):
        return maybe(self._manager.find, self._parent)

    @property
    def child_layers(self):
        return [
            self._manager.wrap(child)
            for child in acquire_elements(self._node.childNodes())
            if _is_real(child)
        ]

    @property
    def is_root(self):
        return self._node.parentNode() is None

    def get_pixels(self, bounds: Bounds | None = None, time: int | None = None):
        bounds = bounds or self.bounds
        assert self._node.colorDepth() == "U8", "Operation only supports 8-bit images"
        if time is None:
            data: QByteArray = self._node.projectionPixelData(*bounds)
        else:
            data: QByteArray = self._node.pixelDataAtTime(*bounds, time)
        assert data is not None and data.size() >= bounds.extent.pixel_count * 4
        return Image.from_packed_bytes(data, bounds.extent)

    def write_pixels(
        self,
        img: Image,
        bounds: Bounds | None = None,
        make_visible=True,
        keep_alpha=False,
        silent=False,
    ):
        layer_bounds = self.bounds
        bounds = bounds or layer_bounds
        if keep_alpha:
            composite = self.get_pixels(bounds)
            composite.draw_image(img, keep_alpha=True)
            img = composite
        elif layer_bounds != bounds and not layer_bounds.is_zero:
            # layer.cropNode(*bounds)  <- more efficient, but clutters the undo stack
            blank = Image.create(layer_bounds.extent, fill=0)
            self._node.setPixelData(blank.data, *layer_bounds)
        self._node.setPixelData(img.data, *bounds)
        if make_visible:
            self.is_visible = True
        if not silent and self.is_visible:
            self.refresh()

    def get_mask(self, bounds: Bounds | None = None, time: int | None = None):
        bounds = bounds or self.bounds
        if self.type.is_mask:
            if time is None:
                data: QByteArray = self._node.pixelData(*bounds)
            else:
                data: QByteArray = self._node.pixelDataAtTime(*bounds, time)
            assert data is not None and data.size() >= bounds.extent.pixel_count
            return Image.from_packed_bytes(data, bounds.extent, channels=1)
        else:
            img = self.get_pixels(bounds, time)
            alpha = img._qimage.convertToFormat(QImage.Format.Format_Alpha8)
            alpha.reinterpretAsFormat(QImage.Format.Format_Grayscale8)
            return Image(alpha)

    def _get_frames(self, fn, bounds: Bounds | None = None):
        doc = ensure(self._manager._doc)
        bounds = bounds or self.bounds
        time_range = range(doc.playBackStartTime(), doc.playBackEndTime() + 1)
        return ImageCollection(
            (fn(bounds, time) for time in time_range if self._node.hasKeyframeAtTime(time))
        )

    def get_pixel_frames(self, bounds: Bounds | None = None):
        return self._get_frames(self.get_pixels, bounds)

    def get_mask_frames(self, bounds: Bounds | None = None):
        return self._get_frames(self.get_mask, bounds)

    def move_to_top(self):
        parent = self._node.parentNode()
        if acquire_elements(parent.childNodes())[-1] == self._node:
            return  # already top-most layer
        with RestoreActiveLayer(self._manager):
            parent.removeChildNode(self.node)
            parent.addChildNode(self.node, None)

    def refresh(self):
        # Hacky way of refreshing the projection of a layer, avoids a full document refresh
        self._node.setBlendingMode(self._node.blendingMode())

    def thumbnail(self, size: Extent):
        return self.node.thumbnail(*size)

    def remove(self):
        self._node.remove()
        self._manager.update()

    def remove_later(self):
        eventloop.run(self._remove_later())

    async def _remove_later(self):
        self.remove()

    def compute_bounds(self):
        bounds = self.bounds
        if bounds.is_zero:
            return bounds
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

    def poll(self):
        self._is_confirmed = True
        changed = False
        if self._name != self._node.name():
            self._name = self._node.name()
            changed = True

        new_parent = maybe(krita.Node.uniqueId, self._node.parentNode())
        if self._parent != new_parent:
            self._parent = new_parent
            self._manager.parent_changed.emit(self)
            changed = True

        return changed

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Layer):
            return self.id == other.id
        return False


class RestoreActiveLayer:
    previous: Layer | None = None
    target: Layer | None = None

    def __init__(self, layers: LayerManager):
        self._observer = layers

    def __enter__(self):
        self.previous = self._observer.active
        self.target = self.previous
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Some operations like inserting a new layer change the active layer as a side effect.
        # It doesn't happen directly, so changing it back in the same call doesn't work.
        eventloop.run(self._restore())

    async def _restore(self):
        if self.previous and self.target:
            if self.previous.is_active:
                # Maybe whatever event we expected to change the active layer hasn't happened yet.
                await eventloop.wait_until(
                    lambda: self.previous is not None and not self.previous.is_active, no_error=True
                )
            self._observer.active = self.target


class LayerManager(QObject):
    """Periodically checks the document for changes in the layer structure. Krita doesn't expose
    Python events for these kinds of changes, so we have to poll and compare.
    Provides helpers to quickly create new layers and groups with initial content.
    """

    changed = pyqtSignal()
    active_changed = pyqtSignal()
    parent_changed = pyqtSignal(Layer)
    removed = pyqtSignal(Layer)

    _doc: krita.Document | None
    _layers: dict[QUuid, Layer]
    _active_id: QUuid
    _last_active: Layer | None = None
    _timer: QTimer
    _is_updating: bool = False

    def __init__(self, doc: krita.Document | None):
        super().__init__()
        self._doc = doc
        self._layers = {}
        if doc is not None:
            root = doc.rootNode()
            self._layers = {root.uniqueId(): Layer(self, root)}
            self._active_id = doc.activeNode().uniqueId()
            self.update()
            self._timer = QTimer()
            self._timer.setInterval(500)
            self._timer.timeout.connect(self.update)
            self._timer.start()
        else:
            self._active_id = QUuid()

    def __del__(self):
        if self._doc is not None:
            self._timer.stop()

    @contextmanager
    def _update_guard(self):
        self._is_updating = True
        try:
            yield
        finally:
            self._is_updating = False

    def update(self):
        if self._doc is None:
            return
        if self._is_updating:
            return
        root_node = self._doc.rootNode()
        if root_node is None:
            return  # Document has been closed

        active = self._doc.activeNode()
        if active is None:
            return

        with self._update_guard():
            removals = set(self._layers.keys())
            changes = False
            for n in traverse_layers(root_node):
                id = n.uniqueId()
                if id in self._layers:
                    removals.remove(id)
                    layer = self._layers[id]
                    changes = layer.poll() or changes
                else:
                    self._layers[id] = Layer(self, n)
                    changes = True

            removals.discard(root_node.uniqueId())
            for id in removals:
                if self._layers[id].is_confirmed:
                    self.removed.emit(self._layers[id])
                    del self._layers[id]

            active_id = active.uniqueId()
            if active_id != self._active_id and active_id in self._layers:
                self._active_id = active_id
                self.active_changed.emit()

            if removals or changes:
                self.changed.emit()

    def wrap(self, node: krita.Node) -> Layer:
        layer = self.find(node.uniqueId())
        if layer is None:
            layer = Layer(self, node, is_confirmed=False)
            self._layers[node.uniqueId()] = layer
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
        assert self._doc is not None
        root = ensure(self._doc.rootNode(), "Document root node was None")
        return self.wrap(root)

    @property
    def active(self):
        try:
            assert self._doc is not None
            layer = self.find(self._doc.activeNode().uniqueId())
            if layer is None:
                layer = self.updated()._layers.get(self._active_id)
            if layer is None:
                # Active layer is not in the layer tree yet, can happen immediately after creating
                # a new layer or merging existing layers.
                layer = self._last_active
            else:
                self._last_active = layer
            return ensure(layer, "Active layer not found in layer tree (no fallback)")
        except Exception as e:
            log.error(f"Error getting active layer: {e}")
            return self.root

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
        if img and bounds:
            node.setPixelData(img.data, *bounds)
        layer = self._insert(node, parent, above, make_active)
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
        group_node = doc.createGroupLayer(f"{layer.name} Group")
        parent = ensure(layer.parent_layer, "Cannot group root layer")
        parent.node.addChildNode(group_node, layer.node)
        parent.node.removeChildNode(layer.node)
        group_node.addChildNode(layer.node, None)
        return self.wrap(group_node)

    def update_layer_image(self, layer: Layer, image: Image, bounds: Bounds, keep_alpha=False):
        """Update layer pixel data by creating a new layer to allow undo."""
        layer_bounds = layer.bounds
        if not keep_alpha:
            layer_bounds = Bounds.union(layer_bounds, bounds)
        content = layer.get_pixels(layer_bounds)
        content.draw_image(image, bounds.relative_to(layer_bounds).offset, keep_alpha=keep_alpha)
        replacement = self.create(layer.name, content, layer_bounds, above=layer)
        layer.remove_later()
        return replacement

    _image_types = [t.value for t in LayerType if t.is_image]
    _mask_types = [t.value for t in LayerType if t.is_mask]

    @property
    def all(self) -> list[Layer]:
        if self._doc is None:
            return []
        return [self.wrap(n) for n in traverse_layers(self._doc.rootNode())]

    @property
    def images(self) -> list[Layer]:
        if self._doc is None:
            return []
        return [self.wrap(n) for n in traverse_layers(self._doc.rootNode(), self._image_types)]

    @property
    def masks(self) -> list[Layer]:
        if self._doc is None:
            return []
        return [self.wrap(n) for n in traverse_layers(self._doc.rootNode(), self._mask_types)]

    @property
    def image_extent(self):
        if doc := self._doc:
            return Extent(doc.width(), doc.height())
        return Extent(1, 1)

    def __bool__(self):
        return self._doc is not None


def traverse_layers(node: krita.Node, type_filter: list[str] | None = None):
    for child in acquire_elements(node.childNodes()):
        type = child.type()
        if _is_real(type) and (not type_filter or type in type_filter):
            yield child
        yield from traverse_layers(child, type_filter)


def _is_real(node_type: krita.Node | str):
    # Krita sometimes inserts "fake" nodes for processing, like decorations-wrapper-layer
    # They don't have a layer type and we want to ignore them
    if isinstance(node_type, krita.Node):
        node_type = node_type.type()
    return node_type != ""
