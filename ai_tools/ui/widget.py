from __future__ import annotations
from typing import Callable, Optional
from pathlib import Path

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

from .. import Auto1111
from .model import Model, ModelRegistry, State
from .server import DiffusionServer, ServerState

_icon_path = Path(__file__).parent.parent / "icons"


class SetupWidget(QWidget):
    _model: Optional[Model] = None

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
        assert self._model is not None
        return self._model

    @model.setter
    def model(self, model: Model):
        self._model = model

    def update(self):
        model = self.model
        self.prompt_textbox.setPlainText(model.prompt)
        self.strength_input.setValue(int(model.strength * 100))
        self.generate_button.setText("Cancel" if model.state is State.generating else "Generate")
        self.error_text.setText(model.error)
        self.error_text.setVisible(model.error != "")
        self.update_progress()

    def update_progress(self):
        self.progress_bar.setValue(int(self.model.progress * 100))

    def generate(self):
        if self.model.state is State.generating:
            return self.cancel()

        assert self.generate_button.text() == "Generate"
        self.generate_button.setText("Cancel")
        self.model.setup()
        self.model.generate()
        self.update()

    def cancel(self):
        assert self.generate_button.text() == "Cancel"
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

    def show_settings(self):
        Krita.instance().action("ai_tools_settings").trigger()


class PreviewWidget(QWidget):
    _model: Optional[Model] = None

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
        assert self._model is not None
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
        self.update_progress()

    def update_progress(self):
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
        self.model.generate()

    def finish(self):
        self.preview_list.clear()
        self.model.reset()


class WelcomeWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("AI Image Generation", self))

        self._connect_status = QLabel("Not connected to Automatic1111 server.", self)
        layout.addWidget(self._connect_status)

        self._connect_error = QLabel(self)
        self._connect_error.setVisible(False)
        self._connect_error.setWordWrap(True)
        self._connect_error.setStyleSheet("font-weight: bold; color: red;")
        layout.addWidget(self._connect_error)

        self._connect_button = QPushButton("Connect", self)
        self._connect_button.clicked.connect(DiffusionServer.instance().connect)
        layout.addWidget(self._connect_button)

        self._settings_button = QPushButton("Settings", self)
        self._settings_button.clicked.connect(self.show_settings)
        layout.addWidget(self._settings_button)

        layout.addStretch()

        DiffusionServer.instance().changed.connect(self.update)

    def update(self):
        server = DiffusionServer.instance()
        if server.state in [ServerState.disconnected, ServerState.error]:
            self._connect_status.setText("Not connected to Automatic1111 server.")
            self._connect_button.setVisible(True)
        if server.state is ServerState.error:
            self._connect_error.setText(server.error)
            self._connect_error.setVisible(True)
        if server.state is ServerState.connecting:
            self._connect_status.setText(
                f"Connecting to Automatic1111 server at {Auto1111.default_url}..."
            )
            self._connect_button.setVisible(False)
        if server.state is ServerState.connected:
            self._connect_status.setText(
                f"Connected to Automatic1111 server at {server.diffusion.url}.\n\nCreate"
                " a new document or open an existing image to start."
            )
            self._connect_button.setVisible(False)
            self._connect_error.setVisible(False)

    def show_settings(self):
        Krita.instance().action("ai_tools_settings").trigger()


class ImageDiffusionWidget(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Generation")
        self._welcome = WelcomeWidget()
        self._setup = SetupWidget()
        self._preview = PreviewWidget()
        self._frame = QStackedWidget(self)
        self._frame.addWidget(self._welcome)
        self._frame.addWidget(self._setup)
        self._frame.addWidget(self._preview)
        self.setWidget(self._frame)

        DiffusionServer.instance().changed.connect(self.update)
        ModelRegistry.instance().created.connect(self.register_model)

    def canvasChanged(self, canvas):
        self.update()

    def register_model(self, model):
        model.changed.connect(self.update)
        model.progress_changed.connect(self.update_progress)

    def update(self):
        model = Model.active()
        server = DiffusionServer.instance()
        if model is None or server.state in [ServerState.disconnected, ServerState.error]:
            self._frame.setCurrentWidget(self._welcome)
        elif model.state in [State.setup, State.generating]:
            self._setup.model = model
            self._setup.update()
            self._frame.setCurrentWidget(self._setup)
        elif State.preview in model.state:
            self._preview.model = model
            self._preview.update()
            self._frame.setCurrentWidget(self._preview)
        else:
            assert False, "Unhandled model state"

    def update_progress(self):
        model = Model.active()
        if model.state is State.generating:
            self._setup.update_progress()
        elif State.preview in model.state:
            self._preview.update_progress()


Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("imageDiffusion", DockWidgetFactoryBase.DockTop, ImageDiffusionWidget)
)
