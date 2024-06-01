from PyQt5.QtCore import Qt, QMetaObject
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QProgressBar,
    QLabel,
    QComboBox,
    QSlider,
    QDoubleSpinBox,
    QGroupBox,
    QCheckBox,
)

from ..properties import Binding, Bind, bind, bind_combo, bind_toggle
from ..resources import UpscalerName
from ..model import Model
from ..root import root
from .theme import SignalBlocker, set_text_clipped
from .widget import WorkspaceSelectWidget, StyleSelectWidget, StrengthWidget, QueueButton


class UpscaleWidget(QWidget):
    _model: Model
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        root.connection.state_changed.connect(self.update_models)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 0)
        self.setLayout(layout)

        self.workspace_select = WorkspaceSelectWidget(self)
        self.model_select = QComboBox(self)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.workspace_select)
        model_layout.addWidget(self.model_select)
        layout.addLayout(model_layout)

        self.factor_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.factor_slider.setMinimum(100)
        self.factor_slider.setMaximum(400)
        self.factor_slider.setTickInterval(50)
        self.factor_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.factor_slider.setSingleStep(50)
        self.factor_slider.setPageStep(50)
        self.factor_slider.valueChanged.connect(self.change_factor_slider)

        self.factor_input = QDoubleSpinBox(self)
        self.factor_input.setMinimum(1.0)
        self.factor_input.setMaximum(4.0)
        self.factor_input.setSingleStep(0.5)
        self.factor_input.setPrefix("Scale: ")
        self.factor_input.setSuffix("x")
        self.factor_input.setDecimals(2)
        self.factor_input.valueChanged.connect(self.change_factor)

        factor_layout = QHBoxLayout()
        factor_layout.addWidget(self.factor_slider)
        factor_layout.addWidget(self.factor_input)
        layout.addLayout(factor_layout)

        self.target_label = QLabel("Target size:", self)
        layout.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addSpacing(6)

        self.refinement_checkbox = QGroupBox("Refine upscaled image", self)
        self.refinement_checkbox.setCheckable(True)

        self.style_select = StyleSelectWidget(self)
        self.strength_slider = StrengthWidget(slider_range=(20, 50), parent=self)

        self.unblur_checkbox = QCheckBox("Use input as guidance ", self)

        self.use_prompt = QCheckBox("Use prompt", self)
        self.prompt_label = QLabel(self)
        self.use_prompt.toggled.connect(self.prompt_label.setEnabled)

        group_layout = QVBoxLayout(self.refinement_checkbox)
        group_layout.addWidget(self.style_select)
        group_layout.addWidget(self.strength_slider)
        group_layout.addWidget(self.unblur_checkbox)
        group_layout.addWidget(self.use_prompt)
        group_layout.addWidget(self.prompt_label)
        self.refinement_checkbox.setLayout(group_layout)
        layout.addWidget(self.refinement_checkbox)
        self.factor_input.setMinimumWidth(self.strength_slider._input.width() + 10)

        self.upscale_button = QPushButton("Upscale", self)
        self.upscale_button.setMinimumHeight(int(self.upscale_button.sizeHint().height() * 1.2))
        self.upscale_button.clicked.connect(self.upscale)

        self.queue_button = QueueButton(supports_batch=False, parent=self)
        self.queue_button.setMinimumHeight(self.upscale_button.minimumHeight())

        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.upscale_button)
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

        layout.addStretch()

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
                bind_combo(model.upscale, "upscaler", self.model_select),
                model.upscale.factor_changed.connect(self.update_factor),
                model.upscale.target_extent_changed.connect(self.update_target_extent),
                bind_toggle(model.upscale, "use_diffusion", self.refinement_checkbox),
                bind(model, "style", self.style_select, "value"),
                bind(model.upscale, "strength", self.strength_slider, "value"),
                bind_toggle(model.upscale, "use_unblur", self.unblur_checkbox),
                bind_toggle(model.upscale, "use_prompt", self.use_prompt),
                model.regions.modified.connect(self._update_prompt),
                model.progress_changed.connect(self.update_progress),
                model.error_changed.connect(self.error_text.setText),
                model.has_error_changed.connect(self.error_text.setVisible),
            ]
            self.queue_button.model = model
            self.update_factor(model.upscale.factor)
            self.update_target_extent()
            self.update_progress()

    def update_models(self):
        if client := root.connection.client_if_connected:
            with SignalBlocker(self.model_select):
                self.model_select.clear()
                for file in sorted(client.models.upscalers, key=_upscaler_order):
                    if file == UpscalerName.default.value:
                        name = f"Default ({file.removesuffix('.pth')})"
                        self.model_select.addItem(name, file)
                    elif file == UpscalerName.fast_4x.value:
                        name = f"Fast ({file.removesuffix('.safetensors')})"
                        self.model_select.addItem(name, file)
                    elif file == UpscalerName.quality.value:
                        name = f"Quality ({file.removesuffix('.pth')})"
                        self.model_select.addItem(name, file)
                    elif file == UpscalerName.sharp.value:
                        name = f"Sharp ({file.removesuffix('.pth')})"
                        self.model_select.addItem(name, file)
                    elif file in [UpscalerName.fast_2x.value, UpscalerName.fast_3x.value]:
                        pass
                    else:
                        self.model_select.addItem(file, file)
                selected = self.model_select.findData(self.model.upscale.upscaler)
                self.model_select.setCurrentIndex(max(selected, 0))

    def update_factor(self, value: float):
        with SignalBlocker(self.factor_input), SignalBlocker(self.factor_slider):
            self.factor_slider.setValue(int(value * 100))
            self.factor_input.setValue(value)

    def update_progress(self):
        self.progress_bar.setValue(int(self.model.progress * 100))

    def upscale(self):
        self.model.upscale_image()

    def change_factor_slider(self, value: int | float):
        rounded = round(value / 50) * 50
        if rounded != value:
            self.factor_slider.setValue(rounded)
        else:
            self.model.upscale.factor = value / 100

    def change_factor(self, value: float):
        self.model.upscale.factor = value

    def update_target_extent(self):
        e = self.model.upscale.target_extent
        self.target_label.setText(f"Target size: {e.width} x {e.height}")

    def _update_prompt(self):
        text = self.model.regions.positive
        if len(self.model.regions) > 0:
            text = f"<b>{len(self.model.regions)} Regions</b> | {text}"
        set_text_clipped(self.prompt_label, text)


def _upscaler_order(filename: str):
    return {
        UpscalerName.default.value: 0,
        UpscalerName.fast_4x.value: 1,
        UpscalerName.quality.value: 2,
        UpscalerName.sharp.value: 3,
    }.get(filename, 99)
