from __future__ import annotations
from typing import cast
import krita
from krita import Krita
from PyQt5.QtCore import QUuid, QByteArray, QTimer
from PyQt5.QtGui import QImage

from .image import Extent, Bounds, Mask, Image
from .pose import Pose


class Document:
    _doc: krita.Document

    def __init__(self, krita_document: krita.Document):
        self._doc = krita_document

    @staticmethod
    def active():
        doc = Krita.instance().activeDocument()
        return Document(doc) if doc else None

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

    def create_mask_from_selection(self, grow: float, feather: float, padding: float):
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

        bounds = Bounds(selection.x(), selection.y(), selection.width(), selection.height())
        bounds = Bounds.pad(bounds, padding_pixels, multiple=8)
        bounds = Bounds.clamp(bounds, self.extent)
        data = selection.pixelData(*bounds)
        return Mask(bounds, data), original_bounds

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

    @property
    def image_layers(self):
        allowed_layer_types = [
            "paintlayer",
            "vectorlayer",
            "grouplayer",
            "filelayer",
            "clonelayer",
            "filterlayer",
        ]
        return list(_traverse_layers(self._doc.rootNode(), allowed_layer_types))

    def find_layer(self, id: QUuid):
        return next((layer for layer in self.image_layers if layer.uniqueId() == id), None)

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


def _selection_is_entire_document(selection: krita.Selection, extent: Extent):
    bounds = Bounds(selection.x(), selection.y(), selection.width(), selection.height())
    if bounds.x > 0 or bounds.y > 0:
        return False
    if bounds.width + bounds.x < extent.width or bounds.height + bounds.y < extent.height:
        return False
    mask = selection.pixelData(*bounds)
    is_opaque = all(x == b"\xff" for x in mask)
    return is_opaque


class PoseLayers:
    _layers: dict[str, Pose] = {}
    _timer = QTimer()

    def __init__(self):
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    def update(self):
        doc = Document.active()
        if not doc:
            return
        layer = doc.active_layer
        if not layer or layer.type() != "vectorlayer":
            return

        layer = cast(krita.VectorLayer, layer)
        pose = self._layers.setdefault(layer.uniqueId(), Pose(doc.extent))
        self._update(layer, layer.shapes(), pose, doc.resolution)

    def add_character(self, layer: krita.VectorLayer):
        doc = Document.active()
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
