from __future__ import annotations

from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import QWidget, QLabel, QSlider, QToolButton, QCheckBox
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QGridLayout, QFrame
from PyQt5.QtCore import Qt, QMetaObject, QSize, pyqtSignal

from ..resources import ControlMode
from ..properties import Binding, bind, bind_combo, bind_toggle
from ..control import ControlLayer
from ..model import Model
from .interval_slider import IntervalSlider
from .theme import SignalBlocker
from . import theme


class ControlWidget(QWidget):
    _model: Model
    _control: ControlLayer
    _connections: list[QMetaObject.Connection | Binding]

    def __init__(self, model: Model, control: ControlLayer, parent: ControlListWidget):
        super().__init__(parent)
        self._model = model
        self._control = control

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.mode_select = QComboBox(self)
        self.mode_select.setStyleSheet(theme.flat_combo_stylesheet)
        for mode in (m for m in ControlMode if m is not ControlMode.inpaint):
            icon = theme.icon(f"control-{mode.name}")
            self.mode_select.addItem(icon, mode.text, mode)

        self.layer_select = QComboBox(self)
        self.layer_select.setMinimumContentsLength(20)
        self.layer_select.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLength
        )
        self._update_layers()
        self._model.layers.changed.connect(self._update_layers)

        self.preset_slider = QSlider(self)
        self.preset_slider.setOrientation(Qt.Orientation.Horizontal)
        self.preset_slider.setMinimumWidth(40)
        self.preset_slider.setRange(0, control.max_preset_value)
        self.preset_slider.setValue(control.preset_value)
        self.preset_slider.setSingleStep(1)
        self.preset_slider.setPageStep(2)
        self.preset_slider.setTickInterval(2)
        self.preset_slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.preset_slider.setToolTip("Control strength: how much the layer affects the image")

        self.error_text = QLabel(self)
        self.error_text.setStyleSheet(f"color: {theme.red};")
        self.error_text.setVisible(not control.is_supported)
        self._set_error(control.error_text)

        self.generate_tool_button = _create_generate_button(
            self, Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.generate_tool_button.clicked.connect(control.generate)

        self.add_pose_tool_button = _create_add_pose_button(
            self, Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.add_pose_tool_button.clicked.connect(self._add_pose_character)

        self.expand_button = QToolButton(self)
        self.expand_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.expand_button.setIcon(theme.icon("more"))
        self.expand_button.setToolTip("Show/hide advanced settings")
        self.expand_button.setAutoRaise(True)
        self.expand_button.setCheckable(True)
        self.expand_button.setChecked(False)
        self.expand_button.clicked.connect(self._toggle_extended)

        self.remove_button = QToolButton(self)
        self.remove_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.remove_button.setIcon(theme.icon("remove"))
        self.remove_button.setToolTip("Remove control layer")
        self.remove_button.setAutoRaise(True)
        self.remove_button.clicked.connect(self.remove)

        bar_layout = QHBoxLayout()
        bar_layout.addWidget(self.mode_select)
        bar_layout.addWidget(self.layer_select, 3)
        bar_layout.addWidget(self.generate_tool_button)
        bar_layout.addWidget(self.add_pose_tool_button)
        bar_layout.addWidget(self.preset_slider, 1)
        bar_layout.addWidget(self.error_text, 3)
        bar_layout.addWidget(self.expand_button)
        bar_layout.addWidget(self.remove_button)
        layout.addLayout(bar_layout)

        line = QFrame(self)
        line.setObjectName("LeftIndent")
        line.setStyleSheet(f"#LeftIndent {{ color: {theme.line};  }}")
        line.setFrameShape(QFrame.Shape.VLine)
        line.setLineWidth(1)

        extended_layout = QVBoxLayout()
        extended_layout.setContentsMargins(8, 2, 4, 6)
        pad_layout = QHBoxLayout()
        pad_layout.setContentsMargins(10, 0, 0, 0)
        pad_layout.addWidget(line)
        pad_layout.addLayout(extended_layout)

        self.extended_widget = QWidget(self)
        self.extended_widget.setVisible(False)
        self.extended_widget.setLayout(pad_layout)
        layout.addWidget(self.extended_widget)

        self.generate_button = _create_generate_button(
            self.extended_widget, Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.generate_button.clicked.connect(control.generate)

        self.add_pose_button = _create_add_pose_button(
            self.extended_widget, Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.add_pose_button.clicked.connect(self._add_pose_character)

        self.custom_checkbox = QCheckBox(self.extended_widget)
        self.custom_checkbox.setText("Use custom values")
        self.custom_checkbox.setChecked(control.use_custom_strength)

        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.custom_checkbox, stretch=1)
        actions_layout.addWidget(self.generate_button)
        actions_layout.addWidget(self.add_pose_button)
        extended_layout.addLayout(actions_layout)

        self.strength_slider = QSlider(self.extended_widget)
        self.strength_slider.setOrientation(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 75)
        self.strength_slider.setValue(control.strength)
        self.strength_slider.setSingleStep(1)
        self.strength_slider.setPageStep(10)
        self.strength_label = QLabel("1.0", self.extended_widget)

        self.range_slider = IntervalSlider(
            low=0, high=20, minimum=0, maximum=20, parent=self.extended_widget
        )
        self.range_slider.intervalChanged.connect(self._set_range)
        self.range_start_label = QLabel("0.00", self.extended_widget)
        self.range_end_label = QLabel("1.00", self.extended_widget)

        slider_layout = QGridLayout()
        slider_layout.setSpacing(8)
        slider_layout.addWidget(QLabel("Strength:"), 0, 0)
        slider_layout.addWidget(self.strength_slider, 0, 2)
        slider_layout.addWidget(self.strength_label, 0, 3)
        slider_layout.addWidget(QLabel("Range:"), 1, 0)
        slider_layout.addWidget(self.range_start_label, 1, 1)
        slider_layout.addWidget(self.range_slider, 1, 2)
        slider_layout.addWidget(self.range_end_label, 1, 3)
        extended_layout.addLayout(slider_layout)

        self._update_visibility()
        self._update_pose_utils()
        self._update_strength()
        self._update_range()
        self._update_custom_values()

        self._connections = [
            bind_combo(control, "mode", self.mode_select),
            bind_combo(control, "layer_id", self.layer_select),
            bind_toggle(control, "use_custom_strength", self.custom_checkbox),
            bind(control, "preset_value", self.preset_slider, "value"),
            bind(control, "strength", self.strength_slider, "value"),
            control.use_custom_strength_changed.connect(self._update_custom_values),
            control.strength_changed.connect(self._update_strength),
            control.start_changed.connect(self._update_range),
            control.end_changed.connect(self._update_range),
            control.has_active_job_changed.connect(self._update_job_active),
            control.error_text_changed.connect(self._set_error),
            control.is_supported_changed.connect(self._update_visibility),
            control.can_generate_changed.connect(self._update_visibility),
            control.mode_changed.connect(self._update_visibility),
            control.is_pose_vector_changed.connect(self._update_pose_utils),
        ]

    def disconnect_all(self):
        Binding.disconnect_all(self._connections)

    def _update_layers(self):
        layers = reversed(self._model.layers.images)
        with SignalBlocker(self.layer_select):
            self.layer_select.clear()
            index = -1
            for layer in layers:
                self.layer_select.addItem(layer.name(), layer.uniqueId())
                if layer.uniqueId() == self._control.layer_id:
                    index = self.layer_select.count() - 1
            if index == -1 and self._control in self._model.control:
                self.remove()
            else:
                self.layer_select.setCurrentIndex(index)

    def remove(self):
        self._model.control.remove(self._control)

    def resizeEvent(self, a0: QResizeEvent | None):
        super().resizeEvent(a0)
        self._update_visibility()

    def _add_pose_character(self):
        self._model.document.add_pose_character(self._control.layer)

    def _update_visibility(self):
        is_small = self.width() < 450
        is_pose = self._control.mode is ControlMode.pose

        def controls():
            self.layer_select.setVisible(self._control.is_supported)
            self.preset_slider.setVisible(self._control.is_supported)
            self.expand_button.setVisible(self._control.is_supported)
            self.generate_button.setVisible(self._control.can_generate and is_small)
            self.generate_tool_button.setVisible(self._control.can_generate and not is_small)
            self.add_pose_button.setVisible(is_pose and is_small)
            self.add_pose_tool_button.setVisible(is_pose and not is_small)
            if not self._control.is_supported:
                self.expand_button.setChecked(False)

        def error():
            self.error_text.setVisible(not self._control.is_supported)

        if self._control.is_supported:
            error()
            controls()
        else:  # always hide things to hide first to make space in the layout
            controls()
            error()

    def _update_range(self):
        self.range_slider.setInterval(int(self._control.start * 20), int(self._control.end * 20))
        self.range_start_label.setText(f"{self._control.start:.2f}")
        self.range_end_label.setText(f"{self._control.end:.2f}")

    def _set_range(self, low: int, high: int):
        self._control.start = low / 20
        self._control.end = high / 20

    def _update_strength(self):
        self.strength_label.setText(
            f"{self._control.strength / ControlLayer.strength_multiplier:.2f}"
        )

    def _update_job_active(self):
        self.generate_button.setEnabled(not self._control.has_active_job)
        self.generate_tool_button.setEnabled(not self._control.has_active_job)
        self.layer_select.setEnabled(not self._control.has_active_job)

    def _update_custom_values(self):
        self.preset_slider.setEnabled(not self._control.use_custom_strength)
        self.strength_slider.setEnabled(self._control.use_custom_strength)
        self.range_slider.setEnabled(self._control.use_custom_strength)

    def _update_pose_utils(self):
        self.add_pose_button.setEnabled(self._control.is_pose_vector)
        self.add_pose_button.setToolTip(
            "Add new character pose to selected layer"
            if self._control.is_pose_vector
            else "Disabled: selected layer must be a vector layer to add a pose"
        )

    def _toggle_extended(self):
        self.extended_widget.setVisible(self.expand_button.isChecked())

    def _set_error(self, error: str):
        parts = error.split("[", 2)
        self.error_text.setText(parts[0])
        if len(parts) > 1:
            self.error_text.setToolTip(f"Missing one of the following models: {parts[1][:-1]}")


def _create_generate_button(parent, style: Qt.ToolButtonStyle):
    button = QToolButton(parent)
    button.setToolButtonStyle(style)
    button.setText("From Image")
    button.setIcon(theme.icon("control-generate"))
    button.setToolTip("Generate control layer from current image")
    return button


def _create_add_pose_button(parent, style: Qt.ToolButtonStyle):
    button = QToolButton(parent)
    button.setToolButtonStyle(style)
    button.setText("Add Skeleton")
    button.setIcon(theme.icon("add-pose"))
    return button


class ControlListWidget(QWidget):
    _controls: list[ControlWidget]
    _model: Model
    _model_connections: list[QMetaObject.Connection]

    changed = pyqtSignal()

    def __init__(self, model: Model, parent=None):
        super().__init__(parent)
        self._model = model
        self._controls = []
        self._model_connections = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model != model:
            Binding.disconnect_all(self._model_connections)
            self._model = model
            while len(self._controls) > 0:
                self._remove_widget(self._controls[0])
            for control in self._model.control:
                self._add_widget(control)
            self._model_connections = [
                model.control.added.connect(self._add_widget),
                model.control.removed.connect(self._remove_widget),
            ]

    def _add_widget(self, control: ControlLayer):
        widget = ControlWidget(self._model, control, self)
        self._controls.append(widget)
        self._layout.addWidget(widget)

    def _remove_widget(self, widget: ControlWidget | ControlLayer):
        if isinstance(widget, ControlLayer):
            widget = next(w for w in self._controls if w._control == widget)
        self._controls.remove(widget)
        widget.disconnect_all()
        widget.deleteLater()


class ControlLayerButton(QToolButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setIcon(theme.icon("control-add"))
        self.setToolTip("Add control layer")
        self.setAutoRaise(True)
        icon_height = self.iconSize().height()
        self.setIconSize(QSize(int(icon_height * 1.25), icon_height))
