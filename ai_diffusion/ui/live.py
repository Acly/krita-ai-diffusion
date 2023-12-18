from __future__ import annotations
from PyQt5.QtCore import QMetaObject, Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolButton,
    QLabel,
    QSpinBox,
    QSizePolicy,
    QProgressBar,
)

from ..properties import Binding, bind, Bind
from ..image import Extent, Image
from ..model import Model
from ..root import root
from ..settings import settings
from .widget import (
    WorkspaceSelectWidget,
    StyleSelectWidget,
    TextPromptWidget,
    StrengthWidget,
    ControlLayerButton,
    ControlListWidget,
)
from . import theme


class LiveWidget(QWidget):
    _play_icon = theme.icon("play")
    _pause_icon = theme.icon("pause")

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

        self.active_button = QToolButton(self)
        self.active_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.active_button.setIcon(self._play_icon)
        self.active_button.setAutoRaise(True)
        self.active_button.setToolTip("Start/stop live preview")
        self.active_button.clicked.connect(self.toggle_active)

        self.apply_button = QToolButton(self)
        self.apply_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.apply_button.setIcon(theme.icon("copy-image"))
        self.apply_button.setAutoRaise(True)
        self.apply_button.setEnabled(False)
        self.apply_button.setToolTip("Copy the current result to the image as a new layer")

        self.style_select = StyleSelectWidget(self)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.workspace_select)
        controls_layout.addWidget(self.active_button)
        controls_layout.addWidget(self.apply_button)
        controls_layout.addWidget(self.style_select)
        layout.addLayout(controls_layout)

        self.strength_slider = StrengthWidget(parent=self)

        self.seed_input = QSpinBox(self)
        self.seed_input.setMinimum(0)
        self.seed_input.setMaximum(2**31 - 1)
        self.seed_input.setPrefix("Seed: ")
        self.seed_input.setToolTip(
            "The seed controls the random part of the output. The same seed value will always"
            " produce the same result."
        )

        self.random_seed_button = QToolButton(self)
        self.random_seed_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.random_seed_button.setIcon(theme.icon("random"))
        self.random_seed_button.setAutoRaise(True)
        self.random_seed_button.setToolTip(
            "Generate a random seed value to get a variation of the image."
        )

        params_layout = QHBoxLayout()
        params_layout.addWidget(self.strength_slider)
        params_layout.addWidget(self.seed_input)
        params_layout.addWidget(self.random_seed_button)
        layout.addLayout(params_layout)

        self.control_list = ControlListWidget(self)
        self.add_control_button = ControlLayerButton(self)
        self.prompt_textbox = TextPromptWidget(line_count=1, parent=self)
        self.negative_textbox = TextPromptWidget(line_count=1, is_negative=True, parent=self)
        self.negative_textbox.setVisible(settings.show_negative_prompt)

        prompt_layout = QVBoxLayout()
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(2)
        prompt_layout.addWidget(self.prompt_textbox)
        prompt_layout.addWidget(self.negative_textbox)
        cond_layout = QHBoxLayout()
        cond_layout.addLayout(prompt_layout)
        cond_layout.addWidget(self.add_control_button)
        layout.addLayout(cond_layout)
        layout.addWidget(self.control_list)

        self.error_text = QLabel(self)
        self.error_text.setStyleSheet("font-weight: bold; color: red;")
        self.error_text.setWordWrap(True)
        self.error_text.setVisible(False)
        layout.addWidget(self.error_text)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Loading %p%")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: transparent;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {theme.grey};
                width: 20px;
            }}""")
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.preview_area = QLabel(self)
        self.preview_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_area.setAlignment(
            Qt.AlignmentFlag(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        )
        layout.addWidget(self.preview_area)

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
                bind(model.live, "strength", self.strength_slider, "value"),
                bind(model.live, "seed", self.seed_input, "value"),
                bind(model, "prompt", self.prompt_textbox, "text"),
                bind(model, "negative_prompt", self.negative_textbox, "text"),
                model.live.is_active_changed.connect(self.update_is_active),
                model.live.has_result_changed.connect(self.apply_button.setEnabled),
                self.apply_button.clicked.connect(model.live.copy_result_to_layer),
                self.add_control_button.clicked.connect(model.control.add),
                self.random_seed_button.clicked.connect(model.live.generate_seed),
                model.error_changed.connect(self.error_text.setText),
                model.has_error_changed.connect(self.error_text.setVisible),
                model.progress_changed.connect(self.update_progress),
                model.live.result_available.connect(self.show_result),
            ]
            self.control_list.model = model

    def update_settings(self, key: str, value):
        if key == "show_negative_prompt":
            self.negative_textbox.text = ""
            self.negative_textbox.setVisible(value)

    def toggle_active(self):
        self.model.live.is_active = not self.model.live.is_active

    def update_is_active(self):
        self.active_button.setIcon(
            self._pause_icon if self.model.live.is_active else self._play_icon
        )

    def update_progress(self):
        if self.model.live.result is None:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(int(self.model.progress * 100))

    def show_result(self, image: Image):
        self.progress_bar.setVisible(False)
        target = Extent.from_qsize(self.preview_area.size())
        img = Image.scale_to_fit(image, target)
        self.preview_area.setPixmap(img.to_pixmap())
        self.preview_area.setMinimumSize(256, 256)
