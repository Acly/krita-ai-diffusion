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
from ..localization import translate as _
from ..root import root
from .control import ControlListWidget
from .region import ActiveRegionWidget, PromptHeader
from .widget import WorkspaceSelectWidget, StyleSelectWidget, StrengthWidget
from .widget import ErrorBox, create_wide_tool_button
from . import theme


class LivePreviewArea(QLabel):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft))

    def show_image(self, image: Image):
        target = Extent.from_qsize(self.size())
        img = Image.scale_to_fit(image, target)
        self.setPixmap(img.to_pixmap())
        self.setMinimumSize(256, 256)


class LiveWidget(QWidget):
    _play_icon = theme.icon("play")
    _pause_icon = theme.icon("pause")

    _record_icon = theme.icon("record")
    _record_active_icon = theme.icon("record-active")

    _model: Model
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 0)
        self.setLayout(layout)

        self.workspace_select = WorkspaceSelectWidget(self)

        self.active_button = QToolButton(self)
        self.active_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.active_button.setIcon(self._play_icon)
        self.active_button.setAutoRaise(True)
        self.active_button.setToolTip(_("Start/stop live preview"))
        self.active_button.clicked.connect(self.toggle_active)

        self.record_button = QToolButton(self)
        self.record_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.record_button.setIcon(self._record_icon)
        self.record_button.setAutoRaise(True)
        self.record_button.setToolTip(
            _("Start live generation and insert images as keyframes into an animation")
        )
        self.record_button.clicked.connect(self.toggle_record)

        self.apply_button = QToolButton(self)
        self.apply_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.apply_button.setIcon(theme.icon("apply"))
        self.apply_button.setAutoRaise(True)
        self.apply_button.setEnabled(False)
        self.apply_button.setToolTip(_("Copy the current result to the active layer"))
        self.apply_button.clicked.connect(self.apply_result)

        self.apply_layer_button = QToolButton(self)
        self.apply_layer_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.apply_layer_button.setIcon(theme.icon("apply-layer"))
        self.apply_layer_button.setAutoRaise(True)
        self.apply_layer_button.setEnabled(False)
        self.apply_layer_button.setToolTip(_("Create a new layer with the current result"))
        self.apply_layer_button.clicked.connect(self.apply_result_layer)

        self.style_select = StyleSelectWidget(self)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.workspace_select)
        controls_layout.addWidget(self.active_button)
        controls_layout.addWidget(self.record_button)
        controls_layout.addWidget(self.apply_button)
        controls_layout.addWidget(self.apply_layer_button)
        controls_layout.addWidget(self.style_select)
        layout.addLayout(controls_layout)

        self.strength_slider = StrengthWidget(parent=self)

        self.seed_input = QSpinBox(self)
        self.seed_input.setMinimum(0)
        self.seed_input.setMaximum(2**31 - 1)
        self.seed_input.setPrefix(_("Seed") + ": ")
        self.seed_input.setToolTip(
            _(
                "The seed controls the random part of the output. The same seed value will always produce the same result."
            )
        )

        self.random_seed_button = QToolButton(self)
        self.random_seed_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.random_seed_button.setIcon(theme.icon("random"))
        self.random_seed_button.setAutoRaise(True)
        self.random_seed_button.setToolTip(
            _("Generate a random seed value to get a variation of the image.")
        )

        params_layout = QHBoxLayout()
        params_layout.addWidget(self.strength_slider)
        params_layout.addWidget(self.seed_input)
        params_layout.addWidget(self.random_seed_button)
        layout.addLayout(params_layout)

        self.control_list = ControlListWidget(self)
        self.add_control_button = create_wide_tool_button(
            "control-add", _("Add Control Layer"), self
        )
        self.add_region_button = create_wide_tool_button("region-add", _("Add Region"), self)
        prompt_buttons_layout = QVBoxLayout()
        prompt_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        prompt_buttons_layout.setSpacing(2)
        prompt_buttons_layout.addWidget(self.add_region_button)
        prompt_buttons_layout.addWidget(self.add_control_button)

        self.region_widget = ActiveRegionWidget(self._model.regions, self, header=PromptHeader.icon)
        self.region_widget.is_slim = True
        self.region_widget.focused.connect(self.focus_active_region)

        self.prompt_widget = ActiveRegionWidget(self._model.regions, self, header=PromptHeader.icon)
        self.prompt_widget.is_slim = True
        self.prompt_widget.focused.connect(self.focus_root_region)

        prompt_text_layout = QVBoxLayout()
        prompt_text_layout.setSpacing(2)
        prompt_text_layout.addWidget(self.region_widget)
        prompt_text_layout.addWidget(self.prompt_widget)

        cond_layout = QHBoxLayout()
        cond_layout.addLayout(prompt_text_layout)
        cond_layout.addLayout(prompt_buttons_layout)
        layout.addLayout(cond_layout)
        layout.addWidget(self.control_list)

        self.error_box = ErrorBox(self)
        layout.addWidget(self.error_box)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Loading %p%")
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                background: transparent;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {theme.grey};
                width: 20px;
            }}"""
        )
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.preview_area = LivePreviewArea(self)
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
                bind(model, "seed", self.seed_input, "value"),
                bind(model, "error", self.error_box, "error", Bind.one_way),
                model.live.is_active_changed.connect(self.update_is_active),
                model.live.is_recording_changed.connect(self.update_is_recording),
                model.live.has_result_changed.connect(self.apply_button.setEnabled),
                model.live.has_result_changed.connect(self.apply_layer_button.setEnabled),
                self.add_region_button.clicked.connect(model.regions.create_region_layer),
                self.add_control_button.clicked.connect(model.regions.add_control),
                self.random_seed_button.clicked.connect(model.generate_seed),
                model.progress_changed.connect(self.update_progress),
                model.live.result_available.connect(self.show_result),
                model.regions.active_changed.connect(self.update_region),
                model.layers.active_changed.connect(self.update_region),
            ]
            self.apply_button.setEnabled(model.live.has_result)
            self.apply_layer_button.setEnabled(model.live.has_result)
            self.prompt_widget.region = model.regions
            self.region_widget.root = model.regions
            self.strength_slider.model = model
            self.update_region()
            self.update_is_active()
            self.update_is_recording()
            self.preview_area.clear()

    def toggle_active(self):
        self.model.live.is_active = not self.model.live.is_active

    def toggle_record(self):
        self.model.live.is_recording = not self.model.live.is_recording

    def update_is_active(self):
        self.active_button.setIcon(
            self._pause_icon if self.model.live.is_active else self._play_icon
        )

    def update_region(self):
        has_regions = len(self.model.regions) > 0
        self.region_widget.setVisible(has_regions)
        self.region_widget.region = self.model.regions.region_for_active_layer
        self.prompt_widget.header_style = PromptHeader.icon if has_regions else PromptHeader.none
        self.control_list.model = self.model.regions.active_or_root.control

    def focus_root_region(self):
        if len(self.model.regions) > 0:
            self.model.regions.active = self.model.regions

    def focus_active_region(self):
        self.model.regions.active = self.model.regions.region_for_active_layer

    def update_is_recording(self):
        self.record_button.setIcon(
            self._record_active_icon if self.model.live.is_recording else self._record_icon
        )

    def update_progress(self):
        if self.model.live.result is None:
            if self.model.progress > 0:
                self.progress_bar.setFormat("Loading %p%")
            else:
                self.progress_bar.setFormat("Initializing...")
            self.progress_bar.setValue(int(self.model.progress * 100))
            self.progress_bar.setVisible(True)

    def show_result(self, image: Image):
        self.progress_bar.setVisible(False)
        self.preview_area.show_image(image)

    def apply_result(self):
        self.model.live.apply_result()

    def apply_result_layer(self):
        self.model.live.apply_result(layer_only=True)
