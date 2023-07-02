from __future__ import annotations
from enum import Flag
from typing import Callable, Optional
from pathlib import Path
import asyncio
import sys
import traceback

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
from PyQt5.QtGui import QFontMetrics, QGuiApplication, QIcon
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from krita import Krita, DockWidget, DockWidgetFactory, DockWidgetFactoryBase
import krita

from .. import (
    eventloop,
    workflow,
    Bounds,
    ImageCollection,
    Image,
    Mask,
    Extent,
    Document,
    Progress,
    Auto1111,
    Interrupted,
    NetworkError,
    settings,
)
from .server import diffusion_server, ServerState

_icon_path = Path(__file__).parent.parent / "icons"


class State(Flag):
    setup = 0
    generating = 1
    preview = 2


class Model:
    _doc: Document
    _layer: Optional[krita.Node] = None
    _image: Optional[Image] = None
    _mask: Optional[Mask] = None
    _extent: Optional[Extent] = None
    _bounds: Optional[Bounds] = None
    _report_progress: Callable[[], None]

    state = State.setup
    prompt = ""
    strength = 1.0
    progress = 0.0
    results: ImageCollection
    error = ""
    task: Optional[asyncio.Task] = None

    def __init__(self, document: Document):
        self._doc = document
        self.results = ImageCollection()

    def setup(self):
        """Retrieve the current image and selection mask as inputs for the next generation(s)."""
        self._mask = self._doc.create_mask_from_selection()
        if self._mask is not None or self.strength < 1.0:
            self._image = self._doc.get_image()
            self._bounds = self._mask.bounds if self._mask else Bounds(0, 0, *self._image.extent)
        else:
            self._extent = self._doc.extent
            self._bounds = Bounds(0, 0, *self._extent)

    async def generate(self, report_progress: Callable[[], None]):
        try:
            assert State.generating not in self.state
            assert diffusion_server.state is ServerState.connected

            diffusion = diffusion_server.diffusion
            image, mask = self._image, self._mask
            self.state = self.state | State.generating
            self._report_progress = report_progress
            self.progress = 0.0
            progress = Progress(self.report_progress)

            if image is None and mask is None:
                assert self._extent is not None and self.strength == 1
                results = await workflow.generate(diffusion, self._extent, self.prompt, progress)
            elif mask is None and self.strength < 1:
                assert image is not None
                results = await workflow.refine(
                    diffusion, image, self.prompt, self.strength, progress
                )
            elif self.strength == 1:
                assert image is not None and mask is not None
                results = await workflow.inpaint(diffusion, image, mask, self.prompt, progress)
            else:
                assert image is not None and mask is not None and self.strength < 1
                results = await workflow.refine_region(
                    diffusion, image, mask, self.prompt, self.strength, progress
                )
            self.results.append(results)
            self.state = State.preview
            if self._layer is None:
                self._layer = self._doc.insert_layer(
                    f"[Preview] {self.prompt}", self.results[0], self._bounds
                )
        except Interrupted:
            self.reset()
        except NetworkError as e:
            self.report_error(e.message, f"[url={e.url}, code={e.code}]")
        except AssertionError as e:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            self.report_error("Error: Internal assertion failed.")
        except Exception as e:
            self.report_error(str(e))

    def cancel(self):
        assert State.generating in self.state and self.task is not None
        diffusion_server.interrupt()
        self.task.cancel()
        self.state = self.state & (~State.generating)
        self.reset()

    def report_progress(self, value):
        self.progress = value
        self._report_progress()

    def report_error(self, message: str, details: Optional[str] = None):
        print("[krita-ai-tools]", message, details)
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
        """Discard all results, cancel any running generation, and go back to the setup stage.
        Setup configuration like prompt and strength is not reset.
        """
        if self.state is (State.preview | State.generating) and not self.task.cancelled():
            self.cancel()
        if self._layer:
            self._layer.setLocked(False)
            self._layer.remove()
            self._layer = None
        self._image = None
        self._mask = None
        self._extent = None
        self.results = ImageCollection()
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
    cancelled = pyqtSignal()

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
        layout.addWidget(self.generate_button, 2, 0, 1, 2)

        self.settings_button = QPushButton(QIcon(str(_icon_path / "settings.svg")), "", self)
        self.settings_button.clicked.connect(self.show_settings)
        layout.addWidget(self.settings_button, 2, 2)

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

    def generate(self):
        if self.model.state is State.generating:
            return self.cancel()

        assert self.generate_button.text() == "Generate"
        self.model.error = ""
        self.model.progress = 0
        self.generate_button.setText("Cancel")
        self.started.emit()
        self.update()

    def cancel(self):
        assert self.generate_button.text() == "Cancel"
        self.model.cancel()
        self.update()
        self.cancelled.emit()

    def change_prompt(self):
        self.model.prompt = self.prompt_textbox.toPlainText()

    def change_strength(self, value: int):
        self.model.strength = value / 100
        if self.strength_input.value() != value:
            self.strength_input.setValue(value)
        if self.strength_slider.value() != value:
            self.strength_slider.setValue(value)

    def show_settings(self):
        Krita.instance().action("ai_tools_settings").trigger()


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
        self.preview_list.setIconSize(QSize(96, 96))
        self.preview_list.currentItemChanged.connect(self.show_preview)
        layout.setRowStretch(0, 1)
        layout.addWidget(self.preview_list, 0, 0, 1, 3)

        self.generate_progress = QProgressBar(self)
        self.generate_progress.setMinimum(0)
        self.generate_progress.setMaximum(100)
        self.generate_progress.setTextVisible(False)
        self.generate_progress.setFixedHeight(6)
        layout.addWidget(self.generate_progress, 1, 0, 1, 3)

        self.apply_button = QPushButton(QIcon(str(_icon_path / "apply.svg")), "Apply", self)
        self.apply_button.clicked.connect(self.apply_result)
        layout.addWidget(self.apply_button, 2, 0)

        self.discard_button = QPushButton(QIcon(str(_icon_path / "discard.svg")), "Discard", self)
        self.discard_button.clicked.connect(self.discard_result)
        layout.addWidget(self.discard_button, 2, 1)

        self.generate_button = QPushButton("Generate more", self)
        self.generate_button.clicked.connect(self.generate_more)
        layout.addWidget(self.generate_button, 2, 2)

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
        if State.preview not in self.model.state:
            return
        if len(self.model.results) != self.preview_list.count():
            self.preview_list.clear()
            self.model.results.each(
                lambda img: self.preview_list.addItem(QListWidgetItem(img.to_icon(), None))
            )
        self.generate_button.setEnabled(self.model.state is State.preview)
        self.generate_progress.setValue(int(self.model.progress * 100))

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

    def generate_more(self):
        self.generate_button.setEnabled(False)
        self.generate_progress.setValue(0)
        model = self.model

        async def run_task():
            await model.generate(report_progress=self.update)
            if model == self.model:  # still viewing the same canvas
                self.update()

        model.task = eventloop.run(run_task())

    def finish(self):
        self.preview_list.clear()
        self.model.reset()
        self.finished.emit()


class WelcomeWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("AI Image Diffusion", self))

        self._connect_status = QLabel("Not connected to Automatic1111 server.", self)
        layout.addWidget(self._connect_status)

        self._connect_error = QLabel(self)
        self._connect_error.setVisible(False)
        self._connect_error.setWordWrap(True)
        self._connect_error.setStyleSheet("font-weight: bold; color: red;")
        layout.addWidget(self._connect_error)

        self._connect_button = QPushButton("Connect", self)
        self._connect_button.clicked.connect(diffusion_server.connect)
        layout.addWidget(self._connect_button)

        self._settings_button = QPushButton("Settings", self)
        self._settings_button.clicked.connect(self.show_settings)
        layout.addWidget(self._settings_button)

        layout.addStretch()

        diffusion_server.changed.connect(self.update)

    def update(self):
        if diffusion_server.state in [ServerState.disconnected, ServerState.error]:
            self._connect_status.setText("Not connected to Automatic1111 server.")
            self._connect_button.setVisible(True)
        if diffusion_server.state is ServerState.error:
            self._connect_error.setText(diffusion_server.error)
            self._connect_error.setVisible(True)
        if diffusion_server.state is ServerState.connecting:
            self._connect_status.setText(
                f"Connecting to Automatic1111 server at {Auto1111.default_url}..."
            )
            self._connect_button.setVisible(False)
        if diffusion_server.state is ServerState.connected:
            self._connect_status.setText(
                f"Connected to Automatic1111 server at {diffusion_server.diffusion.url}.\n\nCreate"
                " a new document or open an existing image to start."
            )
            self._connect_button.setVisible(False)
            self._connect_error.setVisible(False)

    def show_settings(self):
        Krita.instance().action("ai_tools_settings").trigger()


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
        self._setup.cancelled.connect(self.cancel)
        self._preview.finished.connect(self.update)
        diffusion_server.changed.connect(self.update)

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
        if self.model is None or diffusion_server is None:
            self._frame.setCurrentWidget(self._welcome)
        elif self.model.state in [State.setup, State.generating]:
            self._setup.update()
            self._frame.setCurrentWidget(self._setup)
        elif State.preview in self.model.state:
            self._preview.update()
            self._frame.setCurrentWidget(self._preview)
        else:
            assert False, "Unhandled model state"

    def reset(self):
        self.model.reset()
        self._setup.reset()
        self._preview.reset()
        self._frame.setCurrentWidget(self._setup)

    def generate(self):
        model = self.model
        model.setup()

        async def run_task():
            await model.generate(report_progress=self._setup.update)
            if model == self.model:  # still viewing the same canvas
                self.update()

        model.task = eventloop.run(run_task())

    def cancel(self):
        eventloop.run(diffusion_server.interrupt())


Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("imageDiffusion", DockWidgetFactoryBase.DockTop, ImageDiffusionWidget)
)
