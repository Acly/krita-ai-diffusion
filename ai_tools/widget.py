from typing import Tuple, Optional
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QWidget, QLineEdit
from PyQt5.QtCore import QByteArray
from PyQt5.QtGui import QImage
from krita import Krita, DockWidget, DockWidgetFactory, DockWidgetFactoryBase

from . import diffusion
from . import image
from . import settings
from . import workflow
from .image import Extent, Bounds, Mask, Image


class ImageDiffusionWidget(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Diffusion")

        frame = QWidget(self)
        frame.setLayout(QHBoxLayout())
        self.setWidget(frame)

        self.prompt_textbox = QLineEdit(frame)
        frame.layout().addWidget(self.prompt_textbox)

        generate_button = QPushButton("Generate", frame)
        generate_button.clicked.connect(self.inpaint)
        frame.layout().addWidget(generate_button)

    def create_mask_from_selection(self):
        doc = Krita.instance().activeDocument()
        user_selection = doc.selection()
        if not user_selection:
            return None

        extent = Extent(doc.width(), doc.height())
        size_factor = min(doc.width(), doc.height())
        selection = user_selection.duplicate()
        selection.feather(min(5, size_factor // 50))

        bounds = Bounds(selection.x(), selection.y(), selection.width(), selection.height())
        bounds = Bounds.pad(bounds, size_factor // 50, 8, extent)
        data = selection.pixelData(*bounds)
        return Mask(bounds, data)

    def get_image(self):
        doc = Krita.instance().activeDocument()
        img = QImage(
            doc.pixelData(0, 0, doc.width(), doc.height()),
            doc.width(),
            doc.height(),
            QImage.Format_ARGB32)
        return Image(img)

    def insert_layer(self, name: str, img: Image, bounds: Bounds):
        assert img.extent == bounds.extent
        doc = Krita.instance().activeDocument()
        layer = doc.createNode(name, "paintLayer")
        doc.rootNode().addChildNode(layer, None)
        # TODO make sure image extent and format match
        layer.setPixelData(img.data, *bounds)
        doc.refreshProjection()

    def inpaint(self):
        prompt = self.prompt_textbox.text()
        self.prompt_textbox.clear()
        image = self.get_image()
        image.debug_save('krita_document')
        mask = self.create_mask_from_selection()
        if not mask:
            raise Exception('A selection is required for inpaint')
        
        result = workflow.inpaint(image, mask, prompt)
        self.insert_layer(f'diffusion {prompt}', result, mask.bounds)

    def canvasChanged(self, canvas):
        pass


Krita.instance().addDockWidgetFactory(
    DockWidgetFactory(
        "imageDiffusion", DockWidgetFactoryBase.DockTop, ImageDiffusionWidget
    )
)
