from __future__ import annotations
from enum import Enum
from typing import Callable
import asyncio

from PyQt5.QtWidgets import (
    QSlider,
    QPushButton,
    QWidget,
    QPlainTextEdit,
    QLabel,
    QProgressBar,
    QGridLayout,
    QSizePolicy,
    QListWidget,
    QListView,
    QListWidgetItem,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
)
from PyQt5.QtGui import QFontMetrics, QGuiApplication
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from krita import Krita, DockWidget, DockWidgetFactory, DockWidgetFactoryBase
import krita

from . import eventloop
from . import diffusion
from . import workflow
from .image import Bounds, ImageCollection
from .document import Document
from .diffusion import Progress


class State(Enum):
    setup = 0
    generating = 1
    preview = 2


class Model:
    _doc: Document
    _layer: krita.Node = None
    _bounds: Bounds = None
    _report_progress: Callable[[], None]

    state = State.setup
    prompt = ""
    strength = 1.0
    progress = 0.0
    results: ImageCollection = None
    error = ""
    task: asyncio.Task = None

    def __init__(self, document):
        self._doc = document

    async def run_diffusion(self, report_progress: Callable[[], None]):
        assert self.state is State.setup
        self._report_progress = report_progress
        self.progress = 0.0

        image = self._doc.get_image()
        mask = self._doc.create_mask_from_selection()
        assert mask, "A selection is required for inpaint"

        try:
            self.state = State.generating
            if self.strength >= 0.99:
                self.results = await workflow.generate(
                    image, mask, self.prompt, Progress(self.report_progress)
                )
            else:
                self.results = await workflow.refine(
                    image, mask, self.prompt, self.strength, Progress(self.report_progress)
                )
            self._layer = self._doc.insert_layer(
                f"[Preview] {self.prompt}", self.results[0], mask.bounds
            )
            self._bounds = mask.bounds
            self.state = State.preview
        except diffusion.Interrupted:
            self.reset()
        except Exception as e:
            self.report_error(str(e))

    def cancel(self):
        assert self.state is State.generating and self.task is not None
        self.task.cancel()
        self.reset()

    def report_progress(self, value):
        self.progress = value
        self._report_progress()

    def report_error(self, message: str):
        print("[krita-ai-tools]", message)
        self.state = State.setup
        self.error = message

    def show_preview(self, index: int):
        self._doc.set_layer_pixels(self._layer, self.results[index], self._bounds)

    def apply_current_result(self):
        """Apply selected result by duplicating the preview layer and inserting it below.
        This allows to apply multiple results (eg. to combine them afterwards by erasing parts).
        """
        new_layer = self._layer
        self._layer = self._layer.duplicate()
        parent = new_layer.parentNode()
        parent.addChildNode(self._layer, new_layer)
        new_layer.setLocked(False)
        new_layer.setName(new_layer.name().replace("[Preview]", "[Diffusion]"))

    def reset(self):
        if self._layer:
            self._layer.remove()
            self._layer = None
        self.results = None
        self.state = State.setup
        self.progress = 0
        self.error = ""

    @property
    def is_active(self):
        return self._doc.is_active

    @property
    def is_valid(self):
        return self._doc.is_valid


class SetupWidget(QWidget):
    _model: Model = ...

    started = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        self.setLayout(layout)

        self.prompt_textbox = QPlainTextEdit(self)
        self.prompt_textbox.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.prompt_textbox.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.prompt_textbox.setTabChangesFocus(True)
        self.prompt_textbox.setPlaceholderText(
            "Optional prompt: describe the content you want to see, or leave empty."
        )
        self.prompt_textbox.textChanged.connect(self.change_prompt)
        fm = QFontMetrics(self.prompt_textbox.document().defaultFont())
        self.prompt_textbox.setFixedHeight(fm.lineSpacing() * 2 + 4)
        layout.addWidget(self.prompt_textbox, 0, 0, 1, 3)

        strength_text = QLabel(self)
        strength_text.setText("Strength")
        layout.addWidget(strength_text, 1, 0)

        self.strength_input = QSpinBox(self)
        self.strength_input.setMinimum(0)
        self.strength_input.setMaximum(100)
        self.strength_input.setSingleStep(5)
        self.strength_input.setSuffix("%")
        self.strength_input.valueChanged.connect(self.change_strength)
        layout.addWidget(self.strength_input, 1, 2)

        self.strength_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setSingleStep(5)
        self.strength_slider.valueChanged.connect(self.change_strength)
        layout.addWidget(self.strength_slider, 1, 1)

        self.generate_button = QPushButton("Generate", self)
        self.generate_button.clicked.connect(self.generate)
        layout.addWidget(self.generate_button, 2, 0, 1, 3)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar, 3, 0, 1, 3)

        self.error_text = QLabel(self)
        self.error_text.setStyleSheet("font-weight: bold; color: red;")
        self.error_text.setWordWrap(True)
        self.error_text.setVisible(False)
        layout.addWidget(self.error_text, 4, 0, 1, 3)

        layout.addWidget(QWidget(self), 5, 0, 1, 3)
        layout.setRowStretch(5, 1)  # Fill remaining space with dummy

    @property
    def model(self):
        assert self._model is not ...
        return self._model

    @model.setter
    def model(self, model: Model):
        self._model = model

    def update(self):
        model = self.model
        self.prompt_textbox.setPlainText(model.prompt)
        self.strength_input.setValue(int(model.strength * 100))
        self.progress_bar.setValue(int(model.progress * 100))
        self.generate_button.setText("Cancel" if model.state is State.generating else "Generate")
        self.error_text.setText(model.error)
        self.error_text.setVisible(model.error != "")

    def reset(self):
        self.progress_bar.reset()
        self.error_text.setVisible(False)

    def generate(self):
        if self.model.state is State.generating:
            return self.cancel()

        assert self.generate_button.text() == "Generate"
        self.reset()
        self.generate_button.setText("Cancel")
        self.started.emit()

    def report_progress(self):
        if self.model.state is State.generating:
            self.progress_bar.setValue(self.model.progress)

    def cancel(self):
        assert self.generate_button.text() == "Cancel"
        eventloop.run(diffusion.interrupt())
        self.model.cancel()
        self.update()

    def change_prompt(self):
        self.model.prompt = self.prompt_textbox.toPlainText()

    def change_strength(self, value: int):
        self.model.strength = value / 100
        if self.strength_input.value() != value:
            self.strength_input.setValue(value)
        if self.strength_slider.value() != value:
            self.strength_slider.setValue(value)


class PreviewWidget(QWidget):
    _model: Model = ...

    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        self.setLayout(layout)

        self.preview_list = QListWidget(self)
        self.preview_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_list.setResizeMode(QListView.Adjust)
        self.preview_list.setFlow(QListView.LeftToRight)
        self.preview_list.setViewMode(QListWidget.IconMode)
        self.preview_list.setIconSize(QSize(128, 128))
        self.preview_list.currentItemChanged.connect(self.show_preview)
        layout.setRowStretch(0, 1)
        layout.addWidget(self.preview_list, 0, 0, 1, 2)

        self.discard_button = QPushButton("Discard", self)
        self.discard_button.clicked.connect(self.discard_result)
        layout.addWidget(self.discard_button, 1, 0)

        self.apply_button = QPushButton("Apply", self)
        self.apply_button.clicked.connect(self.apply_result)
        layout.addWidget(self.apply_button, 1, 1)

    @property
    def model(self):
        assert self._model is not ...
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model != model:
            self.preview_list.clear()
            self._model = model

    def update(self):
        if self.model.state is not State.preview:
            return
        if len(self.model.results) != self.preview_list.count():
            self.preview_list.clear()
            self.model.results.each(
                lambda img: self.preview_list.addItem(QListWidgetItem(img.to_icon(), None))
            )

    def show_preview(self, current, previous):
        index = self.preview_list.row(current)
        if index >= 0:
            self.model.show_preview(index)

    def apply_result(self):
        self.model.apply_current_result()
        if int(QGuiApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier) == 0:
            self.finish()

    def discard_result(self):
        self.finish()

    def finish(self):
        self.preview_list.clear()
        self.model.reset()
        self.finished.emit()


class WelcomeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("AI Image Diffusion", self))
        self.layout().addWidget(
            QLabel("Create a new document or open an existing image to start.", self)
        )


class ImageDiffusionWidget(DockWidget):
    _models: list[Model] = []

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Diffusion")
        self._welcome = WelcomeWidget()
        self._setup = SetupWidget()
        self._preview = PreviewWidget()
        self._frame = QStackedWidget(self)
        self._frame.addWidget(self._welcome)
        self._frame.addWidget(self._setup)
        self._frame.addWidget(self._preview)
        self.setWidget(self._frame)

        self._setup.started.connect(self.generate)
        self._preview.finished.connect(self.update)

    def canvasChanged(self, canvas):
        # Remove models for documents that have been closed
        self._models = [m for m in self._models if m.is_valid]

        # Switch to or create model for active document
        if not (canvas is None or Document.active() is None):
            model = self.model
            if model is None:
                model = Model(Document.active())
                self._models.append(model)
            self._setup.model = model
            self._preview.model = model
            self.update()

    @property
    def model(self):
        return next((m for m in self._models if m.is_active), None)

    def update(self):
        if self.model is None:
            self._frame.setCurrentWidget(self._welcome)
        elif self.model.state in [State.setup, State.generating]:
            self._setup.update()
            self._frame.setCurrentWidget(self._setup)
        elif self.model.state is State.preview:
            self._preview.update()
            self._frame.setCurrentWidget(self._preview)
        else:
            assert False, "Unhandled model state"

    def report_progress(self):
        self._setup.update()

    def reset(self):
        self.model.reset()
        self._setup.reset()
        self._preview.reset()
        self._frame.setCurrentWidget(self._setup)

    def generate(self):
        model = self.model

        async def run_task():
            await model.run_diffusion(self.report_progress)
            if model == self.model:  # still viewing the same canvas
                self.update()

        model.task = eventloop.run(run_task())


Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("imageDiffusion", DockWidgetFactoryBase.DockTop, ImageDiffusionWidget)
)
