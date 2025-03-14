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
from PyQt5.QtCore import Qt, QAbstractItemModel, QSize, pyqtSignal
from PyQt5.QtGui import QIcon

from ..localization import translate as _
from ..settings import Setting, settings
from .switch import SwitchWidget
from .theme import add_header, icon


class ExpanderButton(QToolButton):
    def __init__(self, text: str | None = None, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setIconSize(QSize(8, 8))
        self.setStyleSheet("QToolButton { border: none; font-weight: bold }")
        if text is not None:
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.setText(" " + text)
        else:
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            self.setMinimumHeight(24)
            self.setMinimumWidth(18)
        self._toggle(False)
        self.toggled.connect(self._toggle)

    def _toggle(self, value: bool):
        self.setArrowType(Qt.ArrowType.DownArrow if value else Qt.ArrowType.RightArrow)


class WarningIcon(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.icon_size = int(1.2 * self.fontMetrics().height())
        warning_icon = icon("warning").pixmap(self.icon_size, self.icon_size)
        self.setPixmap(warning_icon)
        self.setVisible(False)

    def show_message(self, text: str):
        self.setToolTip(text)
        self.setVisible(True)

    def hide(self):
        self.setVisible(False)


class SettingWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, setting: Setting, parent=None):
        super().__init__(parent)

        self._key_label = QLabel(f"<b>{setting.name}</b><br>{setting.desc}")
        self._key_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._checkbox: QCheckBox | None = None
        self._widget: QWidget | None = None

        self._indent = 0
        self._show_label = True
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._key_label)
        self._layout.addStretch(1)
        self.setLayout(self._layout)
        self._set_margins()

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
        widget = self._widget
        assert widget is not None
        checkbox = self._checkbox = QCheckBox(text, self)
        checkbox.toggled.connect(lambda v: widget.setEnabled(v))
        self._layout.removeWidget(self._widget)
        self._layout.addWidget(checkbox)
        self._layout.addWidget(self._widget)
        return checkbox

    @property
    def visible(self):
        return self.isVisible()

    @visible.setter
    def visible(self, v: bool):
        self.setVisible(v)

    @property
    def enabled(self):
        return self._widget and self._widget.isEnabled()

    @enabled.setter
    def enabled(self, v: bool):
        if self._widget is not None:
            self._widget.setEnabled(v)
        if self._checkbox is not None:
            self._checkbox.setChecked(v)

    @property
    def indent(self):
        return self._indent

    @indent.setter
    def indent(self, v: int):
        self._indent = v
        self._set_margins()

    @property
    def show_label(self):
        return self._show_label

    @show_label.setter
    def show_label(self, v: bool):
        self._show_label = v
        self._key_label.setVisible(v)
        self._set_margins()

    def _notify_value_changed(self):
        self.value_changed.emit()

    def _set_margins(self):
        self.setContentsMargins(self._indent * 16, 4 if self._show_label else 0, 0, 0)


class FileListSetting(SettingWidget):
    _files: list[str]

    def __init__(self, setting: Setting, files: list[str], parent=None):
        super().__init__(setting, parent)
        self._list_items: list[tuple[str, QCheckBox]] = []
        self.list_widget = QWidget(self)
        self._list_layout = QHBoxLayout()
        self._list_layout.setContentsMargins(4, 0, 0, 2)
        self.list_widget.setLayout(self._list_layout)
        self._label = QLabel(_("Disabled"), self)
        self._layout.addWidget(self._label)
        self._widget = self.list_widget

        self._set_files(files)

    def reset_files(self, files: list[str]):
        old_value = self.value
        self._set_files(files)
        self.value = old_value

    def _set_files(self, files: list[str]):
        files = sorted(files, key=lambda x: x.lower())
        self._files = files
        for f, w in self._list_items:
            self._list_layout.removeWidget(w)
        if item := self._list_layout.itemAt(0):
            self._list_layout.removeItem(item)
        self._list_items.clear()
        for file_ in self._files:
            checkbox = QCheckBox(file_)
            checkbox.toggled.connect(self._notify_value_changed)
            self._list_layout.addWidget(checkbox)
            self._list_items.append((file_, checkbox))
        self._list_layout.addStretch()

    def _notify_value_changed(self):
        self._update_label()
        return super()._notify_value_changed()

    def _update_label(self):
        self._label.setText(_("Enabled") if len(self.value) > 0 else _("Disabled"))

    @property
    def value(self):
        return [file for file, checkbox in self._list_items if checkbox.isChecked()]

    @value.setter
    def value(self, v):
        for file, checkbox in self._list_items:
            checkbox.setChecked(file in v)
        self._update_label()


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


ComboItemList = list[str] | list[tuple[str, Any]] | list[tuple[str, Any, QIcon]] | type[Enum]


class ComboBoxSetting(SettingWidget):
    _suppress_change = False
    _enum_type = None
    _original_text = ""

    def __init__(self, setting: Setting, model: QAbstractItemModel | None = None, parent=None):
        super().__init__(setting, parent)
        self._combo = QComboBox(self)
        if model is not None:
            self._combo.setModel(model)
        elif setting.items:
            self.set_items(setting.items)
        elif isinstance(setting.default, Enum):
            self._enum_type = type(setting.default)
            self.set_items(self._enum_type)

        self._combo.setMinimumWidth(230)
        self._combo.activated.connect(self._change_value)
        self.set_widget(self._combo)
        self._original_text = self._key_label.text()

    def set_items(self, items: ComboItemList):
        self._suppress_change = True
        self._combo.clear()
        if isinstance(items, type):
            for e in items:
                self._combo.addItem(e.value, e.name)
        else:
            for name in items:
                if isinstance(name, str):
                    self._combo.addItem(name, name)
                elif isinstance(name, Enum):
                    self._combo.addItem(name.value, name.name)
                    self._enum_type = type(name)
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

    def __init__(self, setting: Setting, text=(_("On"), _("Off")), parent=None):
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
    _widgets: dict[str, SettingWidget]
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
                if widget.enabled:
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
    font = title_label.font()
    font.setPointSize(font.pointSize() + 2)
    title_label.setFont(font)
    layout.addWidget(title_label)
    layout.addSpacing(6)
