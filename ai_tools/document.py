import krita
from krita import Krita
from .image import Extent, Bounds, Mask, Image
from PyQt5.QtGui import QImage


class Document:
    def __init__(self, krita_document):
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

    def create_mask_from_selection(self):
        user_selection = self._doc.selection()
        if not user_selection:
            return None

        extent = self.extent
        size_factor = min(extent.width, extent.height)
        feather_radius = min(5, size_factor // 32)
        selection = user_selection.duplicate()
        selection.grow(feather_radius, feather_radius)
        selection.feather(feather_radius)

        bounds = Bounds(selection.x(), selection.y(), selection.width(), selection.height())
        bounds = Bounds.pad(bounds, size_factor // 32, 8)
        bounds = Bounds.clamp(bounds, extent)
        data = selection.pixelData(*bounds)
        return Mask(bounds, data)

    def get_image(self):
        img = QImage(
            self._doc.pixelData(0, 0, self._doc.width(), self._doc.height()),
            self._doc.width(),
            self._doc.height(),
            QImage.Format_ARGB32,
        )
        return Image(img)

    def insert_layer(self):
        layer = self._doc.createNode("New Layer", "paintLayer")
        self._doc.rootNode().addChildNode(layer, None)
        layer.setLocked(True)
        return layer

    def set_layer_pixels(self, layer: krita.Node, img: Image, bounds: Bounds):
        # TODO make sure image extent and format match
        layer.setPixelData(img.data, *bounds)
        self._doc.refreshProjection()
