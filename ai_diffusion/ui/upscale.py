from PyQt5.QtCore import Qt, QMetaObject, QEvent, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QProgressBar,
    QLabel,
    QComboBox,
    QSlider,
    QDoubleSpinBox,
    QGroupBox,
)

from ..properties import Binding, Bind, bind, bind_combo, bind_toggle
from ..resources import ControlMode, UpscalerName
from ..model import Model
from ..jobs import JobKind
from ..localization import translate as _
from ..root import root
from .theme import SignalBlocker, set_text_clipped
from .widget import WorkspaceSelectWidget, StyleSelectWidget, StrengthWidget, QueueButton
from .widget import GenerateButton, ErrorBox
from .settings_widgets import WarningIcon
from .switch import SwitchWidget
from . import theme


class FactorWidget(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None):
        super().__init__(parent)
        self._value = 1.0

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(100)
        self.slider.setMaximum(400)
        self.slider.setTickInterval(50)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setSingleStep(50)
        self.slider.setPageStep(50)
        self.slider.valueChanged.connect(self.change_factor_slider)

        self.input = QDoubleSpinBox(self)
        self.input.setMinimum(1.0)
        self.input.setMaximum(4.0)
        self.input.setSingleStep(0.5)
        self.input.setPrefix(_("Scale") + ": ")
        self.input.setSuffix("x")
        self.input.setDecimals(2)
        self.input.valueChanged.connect(self.change_factor)

        self.target_label = QLabel(self)
        self.target_label.setStyleSheet(f"color: {theme.grey};")

        value_layout = QHBoxLayout()
        value_layout.addWidget(self.slider)
        value_layout.addWidget(self.input)
        layout = QVBoxLayout(self)
        layout.addLayout(value_layout)
        layout.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignRight)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: float):
        if value != self._value:
            self._value = value
            with SignalBlocker(self.input), SignalBlocker(self.slider):
                self.slider.setValue(int(value * 100))
                self.input.setValue(value)
            self.update_target_extent()
            self.value_changed.emit(value)

    def change_factor_slider(self, value: int | float):
        rounded = round(value / 50) * 50
        if rounded != value:
            self.slider.setValue(rounded)
        else:
            self.value = value / 100

    def change_factor(self, value: float):
        self.value = value

    def update_target_extent(self):
        e = root.active_model.upscale.target_extent
        if self.slider.isSliderDown() or self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            self.target_label.setText(_("Target size") + f": {e.width} x {e.height}")
        else:
            self.target_label.setText("")

    def enterEvent(self, a0: QEvent | None):
        self.update_target_extent()
        super().enterEvent(a0)

    def leaveEvent(self, a0: QEvent | None):
        self.update_target_extent()
        super().leaveEvent(a0)


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

        self.factor_widget = FactorWidget(self)
        self.factor_widget.value_changed.connect(self._update_factor)
        layout.addWidget(self.factor_widget)

        self.refinement_checkbox = QGroupBox(_("Refine upscaled image"), self)
        self.refinement_checkbox.setCheckable(True)

        self.style_select = StyleSelectWidget(self)
        self.strength_slider = StrengthWidget(slider_range=(20, 50), parent=self)

        self.unblur_combo = QComboBox(self)
        self.unblur_combo.addItem(_("Off"), 0)
        self.unblur_combo.addItem(_("Unblur - Medium"), 1)
        self.unblur_combo.addItem(_("Unblur - Strong"), 2)
        unblur_layout = QHBoxLayout()
        unblur_layout.addWidget(QLabel(_("Image guidance"), self), 2)
        unblur_layout.addWidget(self.unblur_combo, 1)
        root.connection.models_changed.connect(self._update_unblur_enabled)

        self.use_prompt_switch = SwitchWidget(self)
        self.use_prompt_switch.toggled.connect(self._update_prompt)
        self.use_prompt_value = QLabel(_("Off"), self)
        self.prompt_warning = WarningIcon(self)
        self.prompt_label = QLabel(self)
        self.prompt_label.setMinimumWidth(40)
        self.prompt_label.setEnabled(False)
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel(_("Use Prompt"), self))
        prompt_layout.addWidget(self.prompt_label, 1)
        prompt_layout.addWidget(self.prompt_warning)
        prompt_layout.addWidget(self.use_prompt_value)
        prompt_layout.addWidget(self.use_prompt_switch)

        group_layout = QVBoxLayout(self.refinement_checkbox)
        group_layout.addWidget(self.style_select)
        group_layout.addWidget(self.strength_slider)
        group_layout.addLayout(unblur_layout)
        group_layout.addLayout(prompt_layout)
        self.refinement_checkbox.setLayout(group_layout)
        layout.addWidget(self.refinement_checkbox)
        self.factor_widget.input.setMinimumWidth(self.strength_slider._input.width() + 10)

        self.upscale_button = GenerateButton(JobKind.upscaling, self)
        self.upscale_button.operation = _("Upscale")
        self.upscale_button.clicked.connect(self.upscale)

        self.queue_button = QueueButton(supports_batch=False, parent=self)
        self.queue_button.setFixedHeight(self.upscale_button.height() - 2)

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

        self.error_box = ErrorBox(self)
        layout.addWidget(self.error_box)

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
                bind(model.upscale, "factor", self.factor_widget, "value"),
                bind_toggle(model.upscale, "use_diffusion", self.refinement_checkbox),
                bind(model, "style", self.style_select, "value"),
                bind(model.upscale, "strength", self.strength_slider, "value"),
                bind_combo(model.upscale, "unblur_strength", self.unblur_combo),
                bind_toggle(model.upscale, "use_prompt", self.use_prompt_switch),
                bind(model.upscale, "can_generate", self.upscale_button, "enabled", Bind.one_way),
                bind(model, "error", self.error_box, "error", Bind.one_way),
                model.upscale.use_prompt_changed.connect(self._update_prompt),
                model.regions.modified.connect(self._update_prompt),
                model.regions.added.connect(self._update_prompt),
                model.regions.removed.connect(self._update_prompt),
                model.progress_changed.connect(self.update_progress),
                model.style_changed.connect(self._update_unblur_enabled),
            ]
            self.upscale_button.model = model
            self.queue_button.model = model
            self._update_prompt()
            self._update_unblur_enabled()
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

    def update_progress(self):
        self.progress_bar.setValue(int(self.model.progress * 100))

    def upscale(self):
        self.model.upscale_image()

    def _update_unblur_enabled(self):
        has_unblur = False
        if client := root.connection.client_if_connected:
            models = client.models.for_arch(self.model.arch)
            has_unblur = models.control.find(ControlMode.blur, allow_universal=True) is not None
        self.unblur_combo.setEnabled(has_unblur)
        if not has_unblur:
            self.unblur_combo.setToolTip(_("The tile/unblur control model is not installed."))
        else:
            self.unblur_combo.setToolTip(
                _(
                    "When enabled, the low resolution image is used as guidance for refining the upscaled image.\nThis produces results which are closer to the original while enhancing local details."
                )
            )

    def _update_prompt(self):
        self.use_prompt_value.setText(_("On") if self.model.upscale.use_prompt else _("Off"))
        text = self.model.regions.positive
        if len(self.model.regions) > 0:
            text = f"<b>{len(self.model.regions)} " + _("Regions") + f"</b> | {text}"
        padding = 8
        if self.model.upscale.use_prompt and len(self.model.regions) == 0:
            padding += self.prompt_warning.icon_size
            self.prompt_warning.show_message(
                _(
                    "Text prompt regions have not been set up.\nIt is not recommended to use a single text description for tiled upscale,\nunless it can be generally applied to all parts of the image."
                )
            )
        else:
            self.prompt_warning.hide()
        set_text_clipped(self.prompt_label, text, padding=padding)

    def _update_factor(self):
        if self.factor_widget.value == 1.0 and self.model.upscale.use_diffusion:
            self.upscale_button.operation = _("Refine")
        else:
            self.upscale_button.operation = _("Upscale")


def _upscaler_order(filename: str):
    return {
        UpscalerName.default.value: 0,
        UpscalerName.fast_4x.value: 1,
        UpscalerName.quality.value: 2,
        UpscalerName.sharp.value: 3,
    }.get(filename, 99)
