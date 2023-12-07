from __future__ import annotations
from typing import cast
import krita
from krita import Krita
from PyQt5.QtCore import QObject, QUuid, QByteArray, QTimer, pyqtSignal
from PyQt5.QtGui import QImage

from .image import Extent, Bounds, Mask, Image
from .pose import Pose
from .util import client_logger as log


class Document:
    """Document interface. Used as placeholder when there is no open Document in Krita."""

    @property
    def extent(self):
        return Extent(0, 0)

    @property
    def is_active(self):
        return Krita.instance().activeDocument() is None

    @property
    def is_valid(self):
        return True

    def check_color_mode(self):
        return True, None

    def create_mask_from_selection(
        self, grow: float, feather: float, padding: float, min_size=0, square=False
    ) -> tuple[Mask, Bounds] | tuple[None, None]:
        raise NotImplementedError

    def create_mask_from_layer(self, padding: float, is_inpaint: bool) -> tuple[Mask, Bounds, None]:
        raise NotImplementedError

    def get_image(
        self, bounds: Bounds | None = None, exclude_layers: list[krita.Node] | None = None
    ):
        raise NotImplementedError

    def get_layer_image(self, layer: krita.Node, bounds: Bounds | None) -> Image:
        raise NotImplementedError

    def insert_layer(
        self, name: str, img: Image, bounds: Bounds, below: krita.Node | None = None
    ) -> krita.Node:
        raise NotImplementedError

    def insert_vector_layer(
        self, name: str, svg: str, below: krita.Node | None = None
    ) -> krita.Node:
        raise NotImplementedError

    def set_layer_content(self, layer: krita.Node, img: Image, bounds: Bounds):
        raise NotImplementedError

    def hide_layer(self, layer: krita.Node):
        raise NotImplementedError

    def resize(self, extent: Extent):
        raise NotImplementedError

    def add_pose_character(self, layer: krita.Node):
        raise NotImplementedError

    def create_layer_observer(self) -> LayerObserver:
        return LayerObserver(None)

    @property
    def active_layer(self) -> krita.Node:
        raise NotImplementedError

    @property
    def resolution(self):
        return 0


class KritaDocument(Document):
    """Wrapper around a Krita Document (opened image). Manages multiple image layers and
    allows to retrieve and modify pixel data."""

    _doc: krita.Document

    def __init__(self, krita_document: krita.Document):
        self._doc = krita_document

    @staticmethod
    def active():
        doc = Krita.instance().activeDocument()
        return KritaDocument(doc) if doc else None

    @property
    def extent(self):
        return Extent(self._doc.width(), self._doc.height())

    @property
    def is_active(self):
        return self._doc == Krita.instance().activeDocument()

    @property
    def is_valid(self):
        return self._doc in Krita.instance().documents()

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
        self, grow: float, feather: float, padding: float, min_size=0, square=False
    ):
        user_selection = self._doc.selection()
        if not user_selection:
            return None, None

        if _selection_is_entire_document(user_selection, self.extent):
            return None, None

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

        bounds = _selection_bounds(selection)
        bounds = Bounds.pad(bounds, padding_pixels, multiple=8, min_size=min_size, square=square)
        bounds = Bounds.clamp(bounds, self.extent)
        data = selection.pixelData(*bounds)
        return Mask(bounds, data), original_bounds

    def create_mask_from_layer(self, padding: float, is_inpaint: bool):
        image_bounds = Bounds(0, 0, *self.extent)
        if context_selection := self._doc.selection():
            image_bounds = Bounds.clamp(_selection_bounds(context_selection), self.extent)

        assert self.active_layer.type() == "selectionmask"
        layer = cast(krita.SelectionMask, self.active_layer)
        mask_selection = layer.selection()
        mask_bounds = image_bounds
        if is_inpaint:
            mask_bounds = _selection_bounds(mask_selection)
            pad = int(mask_bounds.extent.diagonal * padding)
            mask_bounds = Bounds.pad(mask_bounds, pad, 512, 8)
            mask_bounds = Bounds.restrict(mask_bounds, image_bounds)

        data: QByteArray = layer.projectionPixelData(*mask_bounds)
        assert data is not None and data.size() >= mask_bounds.extent.pixel_count
        mask = Mask(mask_bounds, data)
        return mask, image_bounds, None

    def get_image(
        self, bounds: Bounds | None = None, exclude_layers: list[krita.Node] | None = None
    ):
        excluded: list[krita.Node] = []
        if exclude_layers:
            for layer in filter(lambda l: l.visible(), exclude_layers):
                layer.setVisible(False)
                excluded.append(layer)
        if len(excluded) > 0:
            # This is quite slow and blocks the UI. Maybe async spinning on tryBarrierLock works?
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

    def insert_layer(self, name: str, img: Image, bounds: Bounds, below: krita.Node | None = None):
        layer = self._doc.createNode(name, "paintlayer")
        above = _find_layer_above(self._doc, below)
        self._doc.rootNode().addChildNode(layer, above)
        layer.setPixelData(img.data, *bounds)
        self._doc.refreshProjection()
        return layer

    def insert_vector_layer(self, name: str, svg: str, below: krita.Node | None = None):
        layer = self._doc.createVectorLayer(name)
        above = _find_layer_above(self._doc, below)
        self._doc.rootNode().addChildNode(layer, above)
        layer.addShapesFromSvg(svg)
        self._doc.refreshProjection()
        return layer

    def set_layer_content(self, layer: krita.Node, img: Image, bounds: Bounds):
        layer_bounds = Bounds.from_qrect(layer.bounds())
        if layer_bounds != bounds:
            # layer.cropNode(*bounds)  <- more efficient, but clutters the undo stack
            blank = Image.create(layer_bounds.extent, fill=0)
            layer.setPixelData(blank.data, *layer_bounds)
        layer.setPixelData(img.data, *bounds)
        layer.setVisible(True)
        self._doc.refreshProjection()
        return layer

    def hide_layer(self, layer: krita.Node):
        layer.setVisible(False)
        self._doc.refreshProjection()
        return layer

    def resize(self, extent: Extent):
        res = self._doc.resolution()
        self._doc.scaleImage(extent.width, extent.height, res, res, "Bilinear")

    def add_pose_character(self, layer: krita.Node):
        assert layer.type() == "vectorlayer"
        _pose_layers.add_character(cast(krita.VectorLayer, layer))

    def create_layer_observer(self):
        return LayerObserver(self._doc)

    @property
    def active_layer(self):
        return self._doc.activeNode()

    @property
    def resolution(self):
        return self._doc.resolution() / 72.0  # KisImage::xRes which is applied to vectors


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


class LayerObserver(QObject):
    managed_layer_types = [
        "paintlayer",
        "vectorlayer",
        "grouplayer",
        "filelayer",
        "clonelayer",
        "filterlayer",
    ]

    changed = pyqtSignal()

    _doc: krita.Document | None
    _layers: list[krita.Node]
    _timer: QTimer

    def __init__(self, doc: krita.Document | None):
        super().__init__()
        self._doc = doc
        self._layers = []
        if doc is not None:
            self._timer = QTimer()
            self._timer.setInterval(500)
            self._timer.timeout.connect(self.update)
            self._timer.start()

    def update(self):
        assert self._doc is not None
        root_node = self._doc.rootNode()
        if root_node is None:
            return  # Document has been closed
        layers = list(_traverse_layers(root_node, self.managed_layer_types))
        if len(layers) != len(self._layers) or any(
            a.uniqueId() != b.uniqueId() for a, b in zip(layers, self._layers)
        ):
            self._layers = layers
            self.changed.emit()

    def find(self, id: QUuid):
        return next((l for l in self._layers if l.uniqueId() == id), None)

    def __iter__(self):
        return iter(self._layers)

    def __getitem__(self, index):
        return self._layers[index]

    def __len__(self):
        return len(self._layers)


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
