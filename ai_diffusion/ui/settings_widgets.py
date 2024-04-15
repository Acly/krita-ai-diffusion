from __future__ import annotations

from enum import Enum
from typing import Any
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QComboBox,
    QSlider,
    QWidget,
    QScrollArea,
    QFrame,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon

from ..settings import Setting, settings
from .switch import SwitchWidget
from .theme import add_header


class ExpanderButton(QToolButton):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setIconSize(QSize(8, 8))
        self.setStyleSheet("QToolButton { border: none; font-weight: bold }")
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setText(" " + text)
        self._toggle(False)
        self.toggled.connect(self._toggle)

    def _toggle(self, value: bool):
        self.setArrowType(Qt.ArrowType.DownArrow if value else Qt.ArrowType.RightArrow)


class SettingWidget(QWidget):
    value_changed = pyqtSignal()

    _checkbox: QCheckBox | None = None
    _layout: QHBoxLayout
    _widget: QWidget

    def __init__(self, setting: Setting, parent=None):
        super().__init__(parent)

        self._key_label = QLabel(f"<b>{setting.name}</b><br>{setting.desc}")
        self._key_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 2, 0, 2)
        self._layout.addWidget(self._key_label)
        self._layout.addStretch(1)
        self.setLayout(self._layout)

    def set_widget(self, widget: QWidget):
        self._widget = widget
        self._layout.addWidget(widget)

    def add_button(self, icon: QIcon, tooltip: str, handler):
        button = QToolButton(self)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setIcon(icon)
        button.setToolTip(tooltip)
        button.clicked.connect(handler)
        self._layout.addWidget(button)

    def add_checkbox(self, text: str):
        self._checkbox = QCheckBox(text, self)
        self._checkbox.toggled.connect(lambda v: self._widget.setEnabled(v))
        self._layout.removeWidget(self._widget)
        self._layout.addWidget(self._checkbox)
        self._layout.addWidget(self._widget)
        return self._checkbox

    @property
    def visible(self):
        return self.isVisible()

    @visible.setter
    def visible(self, v: bool):
        self.setVisible(v)

    @property
    def enabled(self):
        return self._widget.isEnabled()

    @enabled.setter
    def enabled(self, v: bool):
        self._widget.setEnabled(v)
        if self._checkbox is not None:
            self._checkbox.setChecked(v)

    @property
    def indent(self):
        return self._layout.contentsMargins().left() / 16

    @indent.setter
    def indent(self, v: int):
        self._layout.setContentsMargins(v * 16, 2, 0, 2)

    def _notify_value_changed(self):
        self.value_changed.emit()


class SpinBoxSetting(SettingWidget):
    def __init__(self, setting: Setting, parent=None, minimum=0, maximum=100, step=1, suffix=""):
        super().__init__(setting, parent)

        self._spinbox = QSpinBox(self)
        self._spinbox.setMinimumWidth(100)
        self._spinbox.setMinimum(minimum)
        self._spinbox.setMaximum(maximum)
        self._spinbox.setSingleStep(step)
        self._spinbox.setSuffix(suffix)
        self._spinbox.valueChanged.connect(self._notify_value_changed)
        self.set_widget(self._spinbox)

    @property
    def value(self):
        return self._spinbox.value()

    @value.setter
    def value(self, v):
        self._spinbox.setValue(v)

    def add_checkbox(self, text: str):
        self._spinbox.setSpecialValueText("Default")
        return super().add_checkbox(text)


class SliderSetting(SettingWidget):
    _is_float = False

    def __init__(
        self,
        setting: Setting,
        parent=None,
        minimum: int | float = 0,
        maximum: int | float = 100,
        format="{}",
    ):
        super().__init__(setting, parent)
        self._format_string = format
        self._is_float = isinstance(setting.default, float)

        slider_widget = QWidget(self)
        slider_layout = QHBoxLayout()
        slider_widget.setLayout(slider_layout)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setMinimumWidth(200)
        self._slider.setMaximumWidth(300)
        self._slider.setMinimum(round(minimum * self.multiplier))
        self._slider.setMaximum(round(maximum * self.multiplier))
        self._slider.setSingleStep(1)
        self._slider.valueChanged.connect(self._change_value)
        self._label = QLabel(str(self._slider.value()), self)
        self._label.setMinimumWidth(16)
        slider_layout.addWidget(self._slider)
        slider_layout.addWidget(self._label)
        self.set_widget(slider_widget)

    def _change_value(self, value: int):
        self._label.setText(self._format_string.format(self.value))
        self.value_changed.emit()

    @property
    def multiplier(self):
        return 1 if not self._is_float else 10

    @property
    def value(self):
        x = self._slider.value()
        return x if not self._is_float else x / self.multiplier

    @value.setter
    def value(self, v: int | float):
        x = int(v) if not self._is_float else round(v * self.multiplier)
        self._slider.setValue(x)


class ComboBoxSetting(SettingWidget):
    _suppress_change = False
    _enum_type = None
    _original_text = ""

    def __init__(self, setting: Setting, parent=None):
        super().__init__(setting, parent)
        self._combo = QComboBox(self)
        if isinstance(setting.default, Enum):
            self._enum_type = type(setting.default)
            self.set_items(self._enum_type)
        elif setting.items:
            self.set_items(setting.items)

        self._combo.setMinimumWidth(230)
        self._combo.currentIndexChanged.connect(self._change_value)
        self.set_widget(self._combo)
        self._original_text = self._key_label.text()

    def set_items(self, items: list[str] | type[Enum] | list[tuple[str, Any, QIcon]]):
        self._suppress_change = True
        self._combo.clear()
        if isinstance(items, type):
            for e in items:
                self._combo.addItem(e.value, e.name)
        else:
            for name in items:
                if isinstance(name, str):
                    self._combo.addItem(name, name)
                elif len(name) == 2:
                    self._combo.addItem(name[0], name[1])
                elif len(name) == 3:
                    self._combo.addItem(name[2], name[0], name[1])
        self._suppress_change = False

    def _change_value(self):
        if not self._suppress_change:
            self.value_changed.emit()

    def set_text(self, text):
        self._key_label.setText(self._original_text + text)

    @property
    def value(self):
        if self._enum_type is not None:
            return self._enum_type[self._combo.currentData()]
        else:
            return self._combo.currentData()

    @value.setter
    def value(self, v):
        if self._enum_type is not None:
            v = v.name
        index = self._combo.findData(v, Qt.ItemDataRole.UserRole)
        self._combo.setCurrentIndex(index)


class TextSetting(SettingWidget):
    def __init__(self, setting: Setting, parent=None):
        super().__init__(setting, parent)
        self._edit = QLineEdit(self)
        self._edit.setMinimumWidth(230)
        self._edit.setMaximumWidth(300)
        self._edit.textChanged.connect(self._notify_value_changed)
        self.set_widget(self._edit)

    @property
    def value(self):
        return self._edit.text()

    @value.setter
    def value(self, v):
        self._edit.setText(v)


class LineEditSetting(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, setting: Setting, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        add_header(layout, setting)

        self._edit = QLineEdit(self)
        self._edit.textChanged.connect(self._change_value)
        layout.addWidget(self._edit)

    def _change_value(self):
        self.value_changed.emit()

    @property
    def value(self):
        return self._edit.text()

    @value.setter
    def value(self, v):
        self._edit.setText(v)


class SwitchSetting(SettingWidget):
    _text: tuple[str, str]

    def __init__(self, setting: Setting, text=("On", "Off"), parent=None):
        super().__init__(setting, parent)
        self._text = text

        self._label = QLabel(text[0], self)
        self._switch = SwitchWidget(self)
        self._switch.toggled.connect(self._notify_value_changed)
        self._layout.addWidget(self._label)
        self.set_widget(self._switch)

    def _update_text(self):
        self._label.setText(self._text[0 if self._switch.is_checked else 1])

    def _notify_value_changed(self):
        self._update_text()
        super()._notify_value_changed()

    @property
    def value(self):
        return self._switch.is_checked

    @value.setter
    def value(self, v):
        self._switch.is_checked = v
        self._update_text()


class SettingsTab(QWidget):
    _write_guard: SettingsWriteGuard
    _widgets: dict
    _layout: QVBoxLayout

    def __init__(self, title: str):
        super().__init__()
        self._write_guard = SettingsWriteGuard()
        self._widgets = {}

        frame_layout = QVBoxLayout()
        self.setLayout(frame_layout)
        _add_title(frame_layout, title)

        inner = QWidget(self)
        self._layout = QVBoxLayout()
        inner.setLayout(self._layout)

        scroll = QScrollArea(self)
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        frame_layout.addWidget(scroll)

    def add(self, name: str, widget):
        self._layout.addWidget(widget)
        self._widgets[name] = widget
        widget.value_changed.connect(self.write)

    def _read(self):
        pass

    def read(self):
        with self._write_guard:
            for name, widget in self._widgets.items():
                widget.value = getattr(settings, name)
            self._read()

    def _write(self):
        pass

    def write(self, *ignored):
        if not self._write_guard:
            for name, widget in self._widgets.items():
                setattr(settings, name, widget.value)
            self._write()
            settings.save()


class SettingsWriteGuard:
    """Avoid feedback loop when reading settings and updating the UI."""

    _locked = False

    def __enter__(self):
        self._locked = True

    def __exit__(self, *ignored):
        self._locked = False

    def __bool__(self):
        return self._locked


def _add_title(layout: QVBoxLayout, title: str):
    title_label = QLabel(title)
    title_label.setStyleSheet("font-size: 12pt")
    layout.addWidget(title_label)
    layout.addSpacing(6)
