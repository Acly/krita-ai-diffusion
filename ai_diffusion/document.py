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

    def create_mask_from_selection(self, min_size):
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
        bounds = Bounds.pad(bounds, size_factor // 32, min_size=min_size, multiple=8)
        bounds = Bounds.clamp(bounds, extent)
        data = selection.pixelData(*bounds)
        return Mask(bounds, data)

    def get_image(self, exclude_layer=None):
        restore_layer = False
        if exclude_layer and exclude_layer.visible():
            exclude_layer.setVisible(False)
            # This is quite slow and blocks the UI. Maybe async spinning on tryBarrierLock works?
            self._doc.refreshProjection()
            restore_layer = True
        img = QImage(
            self._doc.pixelData(0, 0, self._doc.width(), self._doc.height()),
            self._doc.width(),
            self._doc.height(),
            QImage.Format_ARGB32,
        )
        if restore_layer:
            exclude_layer.setVisible(True)
            self._doc.refreshProjection()
        return Image(img)

    def insert_layer(self, name: str, img: Image, bounds: Bounds):
        layer = self._doc.createNode(name, "paintLayer")
        self._doc.rootNode().addChildNode(layer, None)
        layer.setPixelData(img.data, *bounds)
        layer.setLocked(True)
        self._doc.refreshProjection()
        return layer

    def set_layer_content(self, layer, img: Image, bounds: Bounds):
        layer_bounds = Bounds.from_qrect(layer.bounds())
        if layer_bounds != bounds:
            # layer.cropNode(*bounds)  <- more efficient, but clutters the undo stack
            blank = Image.create(layer_bounds.extent, fill=0)
            layer.setPixelData(blank.data, *layer_bounds)
        layer.setPixelData(img.data, *bounds)
        layer.setVisible(True)
        self._doc.refreshProjection()
        return layer
