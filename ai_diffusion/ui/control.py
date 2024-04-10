from __future__ import annotations

from PyQt5.QtWidgets import QWidget, QLabel, QSpinBox, QDoubleSpinBox, QToolButton
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QMetaObject, QSize, pyqtSignal

from ..resources import ControlMode
from ..properties import Binding, bind, bind_combo
from ..control import ControlLayer
from ..model import Model
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

        layout = QHBoxLayout(self)
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

        self.generate_button = QToolButton(self)
        self.generate_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.generate_button.setIcon(theme.icon("control-generate"))
        self.generate_button.setToolTip("Generate control layer from current image")
        self.generate_button.clicked.connect(control.generate)

        self.add_pose_button = QToolButton(self)
        self.add_pose_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.add_pose_button.setIcon(theme.icon("add-pose"))
        self.add_pose_button.clicked.connect(self._add_pose_character)

        self.strength_spin = QSpinBox(self)
        self.strength_spin.setRange(0, 100)
        self.strength_spin.setValue(int(control.strength * 100))
        self.strength_spin.setSuffix("%")
        self.strength_spin.setSingleStep(10)
        self.strength_spin.setToolTip("Control strength")

        self.end_spin = QDoubleSpinBox(self)
        self.end_spin.setRange(0.0, 1.0)
        self.end_spin.setValue(control.end)
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setToolTip("Control ending step ratio")

        self.error_text = QLabel(self)
        self.error_text.setStyleSheet(f"color: {theme.red};")
        self.error_text.setVisible(not control.is_supported)
        self._set_error(control.error_text)

        self.remove_button = QToolButton(self)
        self.remove_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.remove_button.setIcon(theme.icon("remove"))
        self.remove_button.setToolTip("Remove control layer")
        button_height = self.remove_button.iconSize().height()
        self.remove_button.setIconSize(QSize(int(button_height * 1.25), button_height))
        self.remove_button.setAutoRaise(True)
        self.remove_button.clicked.connect(self.remove)

        layout.addWidget(self.mode_select)
        layout.addWidget(self.layer_select, 1)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.add_pose_button)
        layout.addWidget(self.strength_spin)
        layout.addWidget(self.end_spin)
        layout.addWidget(self.error_text, 1)
        layout.addWidget(self.remove_button)

        self._update_visibility()
        self._update_pose_utils()

        self._connections = [
            bind_combo(control, "mode", self.mode_select),
            bind_combo(control, "layer_id", self.layer_select),
            bind(control, "strength", self.strength_spin, "value"),
            bind(control, "end", self.end_spin, "value"),
            control.has_active_job_changed.connect(
                lambda x: self.generate_button.setEnabled(not x)
            ),
            control.has_active_job_changed.connect(lambda x: self.layer_select.setEnabled(not x)),
            control.error_text_changed.connect(self._set_error),
            control.is_supported_changed.connect(self._update_visibility),
            control.can_generate_changed.connect(self._update_visibility),
            control.show_end_changed.connect(self._update_visibility),
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

    def _add_pose_character(self):
        self._model.document.add_pose_character(self._control.layer)

    def _update_visibility(self):
        def controls():
            self.layer_select.setVisible(self._control.is_supported)
            self.generate_button.setVisible(self._control.can_generate)
            self.add_pose_button.setVisible(
                self._control.is_supported and self._control.mode is ControlMode.pose
            )
            self.strength_spin.setVisible(self._control.is_supported)
            self.end_spin.setVisible(self._control.show_end)

        def error():
            self.error_text.setVisible(not self._control.is_supported)

        if self._control.is_supported:
            error()
            controls()
        else:  # always hide things to hide first to make space in the layout
            controls()
            error()

    def _update_pose_utils(self):
        self.add_pose_button.setEnabled(self._control.is_pose_vector)
        self.add_pose_button.setToolTip(
            "Add new character pose to selected layer"
            if self._control.is_pose_vector
            else "Disabled: selected layer must be a vector layer to add a pose"
        )

    def _set_error(self, error: str):
        parts = error.split("[", 2)
        self.error_text.setText(parts[0])
        if len(parts) > 1:
            self.error_text.setToolTip(f"Missing one of the following models: {parts[1][:-1]}")


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
