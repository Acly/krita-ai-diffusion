import asyncio

from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSlider, QPushButton, QWidget, QLineEdit, QPlainTextEdit, QLabel, QProgressBar
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt
from krita import Krita, DockWidget, DockWidgetFactory, DockWidgetFactoryBase

from . import eventloop
from . import diffusion
from . import workflow
from .image import Extent, Bounds, Mask, Image
from .document import Document
from .diffusion import Progress


class ImageDiffusionWidget(DockWidget):
    _task: asyncio.Task = None

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Diffusion")

        frame = QWidget(self)
        frame.setLayout(QVBoxLayout())
        self.setWidget(frame)

        # self.prompt_textbox = QPlainTextEdit(frame)
        # self.prompt_textbox.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.prompt_textbox.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.prompt_textbox.setTabChangesFocus(True)
        self.prompt_textbox = QLineEdit(frame)
        frame.layout().addWidget(self.prompt_textbox)

        actions = QWidget(frame)
        actions.setLayout(QHBoxLayout())
        frame.layout().addWidget(actions)

        self.strength_slider = QSlider(Qt.Horizontal, actions)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setSingleStep(5)
        self.strength_slider.setValue(100)
        actions.layout().addWidget(self.strength_slider)

        strength_text = QLabel(actions)
        strength_text.setText('Strength: 100%')
        actions.layout().addWidget(strength_text)
        self.strength_slider.valueChanged.connect(lambda v: strength_text.setText(f'Strength: {v}%'))

        self.generate_button = QPushButton("Generate", actions)
        self.generate_button.clicked.connect(self.execute)
        actions.layout().addWidget(self.generate_button)

        self.progress_bar = QProgressBar(frame)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        frame.layout().addWidget(self.progress_bar)

    def report_progress(self, value):
        if self._task and self._task.done():
            self.progress_bar.setValue(100)
        elif self._task:
            self.progress_bar.setValue(int(value * 100))

    def execute(self):
        if self._task and not self._task.done():
            return self.cancel()

        assert self.generate_button.text() == 'Generate'
        self.progress_bar.reset()
        self.generate_button.setText('Cancel')
        prompt = self.prompt_textbox.text()
        strength = self.strength_slider.value() / 100

        doc = Document.active()
        image = doc.get_image()
        image.debug_save('krita_document')
        mask = doc.create_mask_from_selection()
        if not mask:
            raise Exception('A selection is required for inpaint')
        
        async def run_task():
            if strength >= 0.99:
                result = await workflow.generate(image, mask, prompt, Progress(self.report_progress))
            else:
                result = await workflow.refine(image, mask, prompt, strength, Progress(self.report_progress))
            if result:
                doc.insert_layer(f'diffusion {prompt}', result, mask.bounds)            
            self.generate_button.setText('Generate')
        self._task = eventloop.run(run_task())

    def cancel(self):
        assert self.generate_button.text() == 'Cancel'
        eventloop.run(diffusion.interrupt())

    def canvasChanged(self, canvas):
        pass


Krita.instance().addDockWidgetFactory(
    DockWidgetFactory(
        "imageDiffusion", DockWidgetFactoryBase.DockTop, ImageDiffusionWidget
    )
)
