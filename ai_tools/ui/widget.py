from __future__ import annotations
from typing import Callable, Optional
from pathlib import Path

from PyQt5.QtWidgets import (
    QAction,
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
    QMenu,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QHBoxLayout,
    QVBoxLayout,
)
from PyQt5.QtGui import QFontMetrics, QGuiApplication, QIcon
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from krita import Krita, DockWidget

from .. import Client
from . import actions
from .model import Model, ModelRegistry, Job, JobQueue, State
from .connection import Connection, ConnectionState

_icon_path = Path(__file__).parent.parent / "icons"


class QueueWidget(QToolButton):
    _style = """
        QToolButton {{ border: none; border-radius: 6px; background-color: {color}; color: white; }}
        QToolButton::menu-indicator {{ width: 0px; }}"""
    _inactive_color = "#606060"
    _active_color = "#53728E"

    def __init__(self, parent):
        super().__init__(parent)

        queue_menu = QMenu(self)
        queue_menu.addAction(self._create_action("Cancel active", actions.cancel))
        self.setMenu(queue_menu)

        self.setStyleSheet(self._style.format(color=self._inactive_color))
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setArrowType(Qt.NoArrow)

    def update(self, jobs: JobQueue):
        count = jobs.count(State.queued)
        if jobs.any_executing():
            self.setStyleSheet(self._style.format(color=self._active_color))
            if count > 0:
                self.setToolTip(f"Generating image. {count} jobs queued - click to cancel.")
            else:
                self.setToolTip(f"Generating image. Click to cancel.")
        else:
            self.setStyleSheet(self._style.format(color=self._inactive_color))
            self.setToolTip("Idle.")
        self.setText(f"+{count} ")

    def _create_action(self, name: str, func: Callable[[], None]):
        action = QAction(name, self)
        action.triggered.connect(func)
        return action


class GenerationWidget(QWidget):
    _model: Optional[Model] = None

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
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
        layout.addWidget(self.prompt_textbox)

        strength_text = QLabel(self)
        strength_text.setText("Strength")

        self.strength_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setSingleStep(5)
        self.strength_slider.valueChanged.connect(self.change_strength)

        self.strength_input = QSpinBox(self)
        self.strength_input.setMinimum(0)
        self.strength_input.setMaximum(100)
        self.strength_input.setSingleStep(5)
        self.strength_input.setSuffix("%")
        self.strength_input.valueChanged.connect(self.change_strength)

        strength_layout = QHBoxLayout()
        strength_layout.addWidget(strength_text)
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_input)
        layout.addLayout(strength_layout)

        self.generate_button = QPushButton("Generate", self)
        self.generate_button.clicked.connect(self.generate)

        self.queue_button = QueueWidget(self)

        self.settings_button = QToolButton(self)
        self.settings_button.setIcon(QIcon(str(_icon_path / "settings.svg")))
        self.settings_button.setStyleSheet("border: none")
        self.settings_button.clicked.connect(self.show_settings)

        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.generate_button)
        actions_layout.addWidget(self.queue_button)
        actions_layout.addWidget(self.settings_button)
        layout.addLayout(actions_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        self.error_text = QLabel(self)
        self.error_text.setStyleSheet("font-weight: bold; color: red;")
        self.error_text.setWordWrap(True)
        self.error_text.setVisible(False)
        layout.addWidget(self.error_text)

        self.preview_list = QListWidget(self)
        self.preview_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_list.setResizeMode(QListView.Adjust)
        self.preview_list.setFlow(QListView.LeftToRight)
        self.preview_list.setViewMode(QListWidget.IconMode)
        self.preview_list.setIconSize(QSize(96, 96))
        self.preview_list.currentItemChanged.connect(self.show_preview)
        self.preview_list.itemDoubleClicked.connect(self.apply_result)
        layout.addWidget(self.preview_list)

        self.apply_button = QPushButton(QIcon(str(_icon_path / "apply.svg")), "Apply", self)
        self.apply_button.clicked.connect(self.apply_selected_result)
        layout.addWidget(self.apply_button)

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
        model = self.model
        self.prompt_textbox.setPlainText(model.prompt)
        self.strength_input.setValue(int(model.strength * 100))
        self.error_text.setText(model.error)
        self.error_text.setVisible(model.error != "")
        self.update_progress()

        # Clear items which are too old and have been discarded
        previews = self.preview_list
        first_id = next((job.id for job in self.model.history), None)
        while previews.count() > 0 and previews.item(0).data(Qt.UserRole) != first_id:
            previews.takeItem(0)
        # Add new items not yet in the list
        row = 0
        for job in self.model.history:
            if row < previews.count() and previews.item(row).data(Qt.UserRole) == job.id:
                row += len(job.results)
            else:
                self.add_preview(job)

    def update_progress(self):
        self.progress_bar.setValue(int(self.model.progress * 100))
        self.queue_button.update(self.model.jobs)

    def generate(self):
        self.model.generate()
        self.update()

    def cancel(self):
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

    def add_preview(self, job: Job):
        for i, img in enumerate(job.results):
            item = QListWidgetItem(img.to_icon(), None)
            item.setData(Qt.UserRole, job.id)
            item.setData(Qt.UserRole + 1, i)
            item.setData(Qt.ToolTipRole, job.prompt)
            self.preview_list.addItem(item)
        self.preview_list.scrollToBottom()

    def show_preview(self, current: Optional[QListWidgetItem], previous):
        if current:
            job_id = current.data(Qt.UserRole)
            index = current.data(Qt.UserRole + 1)
            self.model.show_preview(job_id, index)
        else:
            self.model.hide_preview()

    def apply_selected_result(self):
        self.model.apply_current_result()

    def apply_result(self, item: QListWidgetItem):
        self.show_preview(item, None)
        self.apply_selected_result()


class WelcomeWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("AI Image Generation", self))

        self._connect_status = QLabel("Not connected to ComfyUI server.", self)
        layout.addWidget(self._connect_status)

        self._connect_error = QLabel(self)
        self._connect_error.setVisible(False)
        self._connect_error.setWordWrap(True)
        self._connect_error.setStyleSheet("font-weight: bold; color: red;")
        layout.addWidget(self._connect_error)

        self._connect_button = QPushButton("Connect", self)
        self._connect_button.clicked.connect(Connection.instance().connect)
        layout.addWidget(self._connect_button)

        self._settings_button = QPushButton("Settings", self)
        self._settings_button.clicked.connect(self.show_settings)
        layout.addWidget(self._settings_button)

        layout.addStretch()

        Connection.instance().changed.connect(self.update)

    def update(self):
        connection = Connection.instance()
        if connection.state in [ConnectionState.disconnected, ConnectionState.error]:
            self._connect_status.setText("Not connected to ComfyUI server.")
            self._connect_button.setVisible(True)
        if connection.state is ConnectionState.error:
            self._connect_error.setText(connection.error)
            self._connect_error.setVisible(True)
        if connection.state is ConnectionState.connecting:
            self._connect_status.setText(f"Connecting to ComfyUI server at {Client.default_url}...")
            self._connect_button.setVisible(False)
        if connection.state is ConnectionState.connected:
            self._connect_status.setText(
                f"Connected to ComfyUI server at {connection.client.url}.\n\nCreate"
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
        self._generation = GenerationWidget()
        self._frame = QStackedWidget(self)
        self._frame.addWidget(self._welcome)
        self._frame.addWidget(self._generation)
        self.setWidget(self._frame)

        Connection.instance().changed.connect(self.update)
        ModelRegistry.instance().created.connect(self.register_model)

    def canvasChanged(self, canvas):
        self.update()

    def register_model(self, model):
        model.changed.connect(self.update)
        model.progress_changed.connect(self._generation.update_progress)

    def update(self):
        model = Model.active()
        connection = Connection.instance()
        if model is None or connection.state in [
            ConnectionState.disconnected,
            ConnectionState.error,
        ]:
            self._frame.setCurrentWidget(self._welcome)
        else:
            self._generation.model = model
            self._generation.update()
            self._frame.setCurrentWidget(self._generation)
