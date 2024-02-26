from __future__ import annotations
from PyQt5.QtCore import Qt, QMetaObject
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QProgressBar,
    QLabel,
    QRadioButton,
    QSizePolicy,
)

from ..properties import Binding, bind, bind_combo, bind_toggle, Bind
from ..model import Model
from ..image import Extent, Image
from ..root import root
from ..settings import settings
from . import theme
from .widget import (
    WorkspaceSelectWidget,
    StyleSelectWidget,
    TextPromptWidget,
    StrengthWidget,
    ControlLayerButton,
    ControlListWidget,
    QueueButton,
)


class AnimationWidget(QWidget):

    _model: Model
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        settings.changed.connect(self.update_settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 0)
        self.setLayout(layout)

        self.workspace_select = WorkspaceSelectWidget(self)
        self.style_select = StyleSelectWidget(self, show_quality=True)

        style_layout = QHBoxLayout()
        style_layout.addWidget(self.workspace_select)
        style_layout.addWidget(self.style_select)
        layout.addLayout(style_layout)

        self.prompt_textbox = TextPromptWidget(parent=self)
        self.prompt_textbox.line_count = settings.prompt_line_count

        self.negative_textbox = TextPromptWidget(line_count=1, is_negative=True, parent=self)
        self.negative_textbox.setVisible(settings.show_negative_prompt)

        prompt_layout = QVBoxLayout()
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(2)
        prompt_layout.addWidget(self.prompt_textbox)
        prompt_layout.addWidget(self.negative_textbox)
        layout.addLayout(prompt_layout)

        self.control_list = ControlListWidget(self)
        layout.addWidget(self.control_list)

        self.strength_slider = StrengthWidget(parent=self)
        self.add_control_button = ControlLayerButton(self)
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.add_control_button)
        layout.addLayout(strength_layout)

        mode_layout = QHBoxLayout()
        self.batch_mode_button = QRadioButton("Full Animation", self)
        self.frame_mode_button = QRadioButton("Single Frame", self)
        mode_layout.addWidget(self.batch_mode_button)
        mode_layout.addWidget(self.frame_mode_button)
        layout.addLayout(mode_layout)

        self.generate_button = QPushButton(self)
        self.generate_button.setMinimumHeight(int(self.generate_button.sizeHint().height() * 1.2))

        self.queue_button = QueueButton(parent=self, supports_batch=False)
        self.queue_button.setMinimumHeight(self.generate_button.minimumHeight())

        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.generate_button)
        actions_layout.addWidget(self.queue_button)
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

        self.target_layer = QComboBox(self)
        layout.addWidget(self.target_layer)

        self.preview_area = QLabel(self)
        self.preview_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_area.setAlignment(
            Qt.AlignmentFlag(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        )
        layout.addWidget(self.preview_area)

        self.update_mode()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            self._model_bindings = [
                bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
                bind(model, "style", self.style_select, "value"),
                bind(model.animation, "sampling_quality", self.style_select, "quality"),
                bind(model, "prompt", self.prompt_textbox, "text"),
                bind(model, "negative_prompt", self.negative_textbox, "text"),
                bind(model, "strength", self.strength_slider, "value"),
                bind_toggle(model.animation, "batch_mode", self.batch_mode_button),
                bind_combo(model.animation, "target_layer", self.target_layer),
                model.animation.batch_mode_changed.connect(self.update_mode),
                model.animation.target_image_changed.connect(self.show_result),
                model.progress_changed.connect(self.update_progress),
                model.error_changed.connect(self.error_text.setText),
                model.has_error_changed.connect(self.error_text.setVisible),
                model.layers.changed.connect(self.update_target_layers),
                self.add_control_button.clicked.connect(model.control.add),
                self.prompt_textbox.activated.connect(model.animation.generate),
                self.negative_textbox.activated.connect(model.animation.generate),
                self.generate_button.clicked.connect(model.animation.generate),
            ]
            self.control_list.model = model
            self.queue_button.model = model
            self.update_mode()
            self.update_target_layers()
            self.preview_area.clear()

    def update_progress(self):
        self.progress_bar.setValue(int(self.model.progress * 100))

    def update_settings(self, key: str, value):
        if key == "prompt_line_count":
            self.prompt_textbox.line_count = value
        elif key == "show_negative_prompt":
            self.negative_textbox.text = ""
            self.negative_textbox.setVisible(value)

    def update_mode(self):
        self.target_layer.setVisible(not self.model.animation.batch_mode)
        if self.model.animation.batch_mode:
            self.preview_area.clear()
            self.generate_button.setText("Generate Animation")
            self.generate_button.setToolTip(
                "Generate images from the active layer for all keyframes within start and end time."
                " The active layer must contain an animation!"
            )
        else:
            self.generate_button.setText("Generate Frame")
            self.generate_button.setToolTip(
                "Generate a single frame from the current canvas and insert it into the target layer."
            )

    def update_target_layers(self):
        with theme.SignalBlocker(self.target_layer):
            self.target_layer.clear()
            for layer in self._model.layers.images:
                self.target_layer.addItem(f"Target layer: {layer.name()}", layer.uniqueId())
        if self.model.animation.target_layer.isNull():
            self.model.animation.target_layer = self.target_layer.currentData()
        else:
            current_index = self.target_layer.findData(self.model.animation.target_layer)
            if current_index >= 0:
                self.target_layer.setCurrentIndex(current_index)

    def show_result(self, image: Image):
        target = Extent.from_qsize(self.preview_area.size())
        img = Image.scale_to_fit(image, target)
        self.preview_area.setPixmap(img.to_pixmap())
        self.preview_area.setMinimumSize(256, 256)
