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

    def create_mask_from_selection(self):
        user_selection = self._doc.selection()
        if not user_selection:
            return None

        extent = self.extent
        size_factor = min(extent.width, extent.height)
        selection = user_selection.duplicate()
        selection.feather(min(5, size_factor // 32))

        bounds = Bounds(selection.x(), selection.y(), selection.width(), selection.height())
        bounds = Bounds.pad(bounds, size_factor // 32, 8, extent)
        data = selection.pixelData(*bounds)
        return Mask(bounds, data)

    def get_image(self):
        img = QImage(
            self._doc.pixelData(0, 0, self._doc.width(), self._doc.height()),
            self._doc.width(), self._doc.height(), QImage.Format_ARGB32)
        return Image(img)
    
    def insert_layer(self, name: str, img: Image, bounds: Bounds):
        assert img.extent == bounds.extent
        layer = self._doc.createNode(name, "paintLayer")
        self._doc.rootNode().addChildNode(layer, None)
        self.set_layer_pixels(layer, img, bounds)
        layer.setLocked(True)
        return layer
    
    def set_layer_pixels(self, layer: krita.Node, img: Image, bounds: Bounds):
        # TODO make sure image extent and format match
        layer.setPixelData(img.data, *bounds)
        self._doc.refreshProjection()
