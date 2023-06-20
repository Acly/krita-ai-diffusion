import asyncio

from PyQt5.QtWidgets import QSlider, QPushButton, QWidget, QPlainTextEdit, QLabel, QProgressBar, QGridLayout, QSizePolicy, QListWidget, QListView, QListWidgetItem
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtCore import Qt, QSize
from krita import Krita, DockWidget, DockWidgetFactory, DockWidgetFactoryBase
import krita

from . import eventloop
from . import diffusion
from . import workflow
from .image import Extent, Bounds, Mask, Image, ImageCollection
from .document import Document
from .diffusion import Progress


class ImageDiffusionWidget(DockWidget):
    _task: asyncio.Task = None
    _result_images: ImageCollection = None
    _result_layer: krita.Node = None
    _result_bounds: Bounds = None
    _doc: Document = None

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Diffusion")

        frame = QWidget(self)
        layout = QGridLayout(frame)
        frame.setLayout(layout)
        self.setWidget(frame)

        self.prompt_textbox = QPlainTextEdit(frame)
        self.prompt_textbox.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.prompt_textbox.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.prompt_textbox.setTabChangesFocus(True)
        self.prompt_textbox.setPlaceholderText('Optional prompt: describe the content you want to see, or leave empty.')
        fm = QFontMetrics(self.prompt_textbox.document().defaultFont())
        self.prompt_textbox.setFixedHeight(fm.lineSpacing() * 2 + 4)
        layout.addWidget(self.prompt_textbox, 0, 0, 1, 2)

        strength_text = QLabel(frame)
        strength_text.setText('Strength: 100%')
        layout.addWidget(strength_text, 2, 0)

        self.strength_slider = QSlider(Qt.Horizontal, frame)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setSingleStep(5)
        self.strength_slider.setValue(100)
        self.strength_slider.valueChanged.connect(lambda v: strength_text.setText(f'Strength: {v}%'))
        layout.addWidget(self.strength_slider, 1, 0)

        self.generate_button = QPushButton("Generate", frame)
        self.generate_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.generate_button.setEnabled(Document.active() is not None)
        self.generate_button.clicked.connect(self.generate)
        layout.addWidget(self.generate_button, 1, 1, 2, 1)

        self.progress_bar = QProgressBar(frame)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar, 3, 0, 1, 2)

        self.error_text = QLabel(frame)
        self.error_text.setStyleSheet('font-weight: bold; color: red;')
        self.error_text.setWordWrap(True)
        self.error_text.setVisible(False)
        layout.addWidget(self.error_text, 4, 0, 1, 2)

        self.preview_list = QListWidget(frame)
        self.preview_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_list.setResizeMode(QListView.Adjust)
        self.preview_list.setFlow(QListView.LeftToRight)
        self.preview_list.setViewMode(QListWidget.IconMode)
        self.preview_list.setIconSize(QSize(128, 128))
        self.preview_list.currentItemChanged.connect(self.apply_preview)
        layout.setRowStretch(5, 1)
        layout.addWidget(self.preview_list, 5, 0, 1, 2)

        self.discard_button = QPushButton('Discard', frame)
        self.discard_button.setEnabled(False)
        self.discard_button.clicked.connect(self.discard_result)
        layout.addWidget(self.discard_button, 6, 0)

        self.apply_button = QPushButton('Apply', frame)
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_result)
        layout.addWidget(self.apply_button, 6, 1)

    def report_progress(self, value: float):
        if self._task and self._task.done():
            self.progress_bar.setValue(100)
        elif self._task:
            self.progress_bar.setValue(int(value * 100))

    def report_error(self, message: str):
        print('[krita-ai-tools]', message)
        self.error_text.setText(message)
        self.error_text.setVisible(True)

    def add_preview(self, images: ImageCollection):
        images.each(lambda img: self.preview_list.addItem(QListWidgetItem(img.to_icon(), None)))

    def apply_preview(self, current, previous):
        index = self.preview_list.row(current)
        if index >= 0: 
            self._doc.set_layer_pixels(self._result_layer, self._result_images[index], self._result_bounds)

    def reset(self):
        if self._result_layer:
            self._result_layer.remove()
        self.progress_bar.reset()
        self.error_text.setVisible(False)
        self.preview_list.clear()
        self.apply_button.setEnabled(False)
        self.discard_button.setEnabled(False)
        self._result_images = None
        self._result_layer = None
        self._result_bounds = None
        self._doc = None

    def generate(self):
        if self._task and not self._task.done():
            return self.cancel()

        assert self.generate_button.text() == 'Generate'
        self.reset()
        self.generate_button.setText('Cancel')
        prompt = self.prompt_textbox.toPlainText()
        strength = self.strength_slider.value() / 100

        doc = Document.active()
        self._doc = doc
        image = doc.get_image()
        image.debug_save('krita_document')
        mask = doc.create_mask_from_selection()
        if not mask:
            raise Exception('A selection is required for inpaint')
        
        async def run_task():
            try:
                if strength >= 0.99:
                    result = await workflow.generate(
                        image, mask, prompt, Progress(self.report_progress))
                else:
                    result = await workflow.refine(
                        image, mask, prompt, strength, Progress(self.report_progress))
                if result:
                    self._result_images = result
                    self._result_layer = doc.insert_layer(f'[Preview] {prompt}', result[0], mask.bounds)
                    self._result_bounds = mask.bounds
                    self.add_preview(result)
                    self.apply_button.setEnabled(True)
                    self.discard_button.setEnabled(True)
            except Exception as e:
                self.report_error(str(e))
                raise e
            finally:
                self.generate_button.setText('Generate')
        self._task = eventloop.run(run_task())

    def cancel(self):
        assert self.generate_button.text() == 'Cancel'
        eventloop.run(diffusion.interrupt())

    def apply_result(self):
        assert self._result_layer is not None
        new_layer = self._result_layer
        self._result_layer = self._result_layer.duplicate()
        parent = new_layer.parentNode()
        parent.addChildNode(self._result_layer, new_layer)
        new_layer.setLocked(False)
        new_layer.setName(new_layer.name().replace('[Preview]', '[Diffusion]'))

    def discard_result(self):
        assert self._result_layer is not None
        self.reset()

    def canvasChanged(self, canvas):
        self.generate_button.setEnabled(Document.active() is not None)


Krita.instance().addDockWidgetFactory(
    DockWidgetFactory(
        "imageDiffusion", DockWidgetFactoryBase.DockTop, ImageDiffusionWidget
    )
)
