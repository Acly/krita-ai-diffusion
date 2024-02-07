from __future__ import annotations

import functools
import os.path
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Optional, cast
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QPushButton,
    QCheckBox,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QRadioButton,
    QToolButton,
    QComboBox,
    QSlider,
    QWidget,
    QMenu,
    QAction,
)
from PyQt5.QtCore import Qt, QSize, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QGuiApplication, QIcon, QCursor
from krita import Krita

from ..resources import SDVersion, CustomNode, MissingResource, ResourceKind, required_models
from ..settings import Setting, Settings, ServerMode, PerformancePreset, settings
from ..server import Server
from ..client import resolve_sd_version
from ..style import Style, Styles, StyleSettings
from ..root import root
from .. import util, __version__
from ..connection import ConnectionState, apply_performance_preset
from .server import ServerWidget
from .switch import SwitchWidget
from .theme import add_header, icon, sd_version_icon, red, yellow, green, grey


def _add_title(layout: QVBoxLayout, title: str):
    title_label = QLabel(title)
    title_label.setStyleSheet("font-size: 12pt")
    layout.addWidget(title_label)
    layout.addSpacing(6)


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
    def __init__(self, setting: Setting, parent=None, minimum=0, maximum=100, step=0, suffix=""):
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


def _menu_width(menu: QMenu) -> int:
    if not menu.isEmpty():
        last_action = menu.actions()[-1]
        action_rect = menu.actionGeometry(last_action)
        return action_rect.right()
    else:
        return 0


class LoraList(QWidget):
    class Item(QWidget):
        changed = pyqtSignal()
        removed = pyqtSignal(QWidget)

        def __init__(self, lora_names, parent=None):
            super().__init__(parent)
            self.setContentsMargins(0, 0, 0, 0)

            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            self.setLayout(layout)

            self._select_value = ""
            self._select = QPushButton(self)
            self._select.setStyleSheet("QPushButton {text-align: left; padding: 0.2em 0.4em;}")
            self._select.setMenu(self._build_menu(util.get_path_dict(lora_names)))

            self._strength = QSpinBox(self)
            self._strength.setMinimum(-400)
            self._strength.setMaximum(400)
            self._strength.setSingleStep(5)
            self._strength.setValue(100)
            self._strength.setPrefix("Strength: ")
            self._strength.setSuffix("%")
            self._strength.valueChanged.connect(self._update)

            self._remove = QToolButton(self)
            self._remove.setIcon(icon("discard"))
            self._remove.clicked.connect(self.remove)

            layout.addWidget(self._select, 3)
            layout.addWidget(self._strength, 1)
            layout.addWidget(self._remove)

        def _update(self):
            self.changed.emit()

        def remove(self):
            self.removed.emit(self)

        def _build_menu(self, values: dict, title="") -> QMenu:
            menu = QMenu(title, self)
            for k, v in values.items():
                if isinstance(v, str):
                    action = QAction(k, self)
                    action.triggered.connect(functools.partial(self._select_update, v))
                    menu.addAction(action)
                else:
                    menu.addMenu(self._build_menu(v, k))

            if screen := QGuiApplication.screenAt(QCursor.pos()):
                if _menu_width(menu) > screen.availableSize().width():
                    menu.setStyleSheet("QMenu{menu-scrollable: 1;}")

            return menu

        def _select_update(self, text):
            self._select_value = text
            self._select.setText(text)
            self._update()

        @property
        def value(self):
            return dict(name=self._select_value, strength=self._strength.value() / 100)

        @value.setter
        def value(self, v):
            self._select_value = v["name"]
            self._select.setText(v["name"])
            self._strength.setValue(int(v["strength"] * 100))

    value_changed = pyqtSignal()

    open_folder_button: Optional[QToolButton] = None

    _loras: list[str]
    _items: list[Item]

    def __init__(self, setting: Setting, parent=None):
        super().__init__(parent)
        self._loras = []
        self._items = []

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_text_layout = QVBoxLayout()
        add_header(header_text_layout, setting)
        header_layout.addLayout(header_text_layout, 3)

        self._add_button = QPushButton("Add", self)
        self._add_button.setMinimumWidth(100)
        self._add_button.clicked.connect(self._add_item)
        align_right_center = Qt.AlignmentFlag(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        header_layout.addWidget(self._add_button, 1, align_right_center)

        self._refresh_button = QToolButton(self)
        self._refresh_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._refresh_button.setIcon(Krita.instance().icon("reload-preset"))
        self._refresh_button.setToolTip("Look for new LoRA files")
        self._refresh_button.clicked.connect(root.connection.refresh)
        header_layout.addWidget(self._refresh_button, 0, align_right_center)

        if settings.server_mode is ServerMode.managed:
            self.open_folder_button = QToolButton(self)
            self.open_folder_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            self.open_folder_button.setIcon(Krita.instance().icon("document-open"))
            self.open_folder_button.setToolTip("Open folder containing LoRA files")
            header_layout.addWidget(self.open_folder_button, 0, align_right_center)

        self._layout.addLayout(header_layout)

        self._item_list = QVBoxLayout()
        self._item_list.setContentsMargins(0, 0, 0, 0)
        self._layout.addLayout(self._item_list)

    def _add_item(self, lora=None):
        assert self._item_list is not None
        item = self.Item(self._loras, self)
        if isinstance(lora, dict):
            item.value = lora
        item.changed.connect(self._update_item)
        item.removed.connect(self._remove_item)
        self._items.append(item)
        self._item_list.addWidget(item)
        self.value_changed.emit()

    def _remove_item(self, item: QWidget):
        self._items.remove(item)
        self._item_list.removeWidget(item)
        item.deleteLater()
        self.value_changed.emit()

    def _update_item(self):
        self.value_changed.emit()

    @property
    def names(self):
        return self._loras

    @names.setter
    def names(self, v):
        self._loras = v
        for item in self._items:
            item._select.setMenu(item._build_menu(util.get_path_dict(self._loras)))

    @property
    def value(self):
        return [item.value for item in self._items]

    @value.setter
    def value(self, v):
        while not len(self._items) == 0:
            self._remove_item(self._items[-1])
        for lora in v:
            self._add_item(lora)


class SettingsWriteGuard:
    """Avoid feedback loop when reading settings and updating the UI."""

    _locked = False

    def __enter__(self):
        self._locked = True

    def __exit__(self, *ignored):
        self._locked = False

    def __bool__(self):
        return self._locked


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


class ConnectionSettings(SettingsTab):
    def __init__(self, server: Server):
        super().__init__("Server Configuration")

        add_header(self._layout, Settings._server_mode)
        self._server_managed = QRadioButton("Local server managed by Krita plugin", self)
        self._server_external = QRadioButton("Connect to external Server (local or remote)", self)
        self._server_managed.toggled.connect(self._change_server_mode)
        info_managed = QLabel(
            "Let the Krita plugin install and manage a local server on your machine", self
        )
        info_external = QLabel("You are responsible to set up and start the server yourself", self)
        for button in (self._server_managed, self._server_external):
            button.setStyleSheet("font-weight:bold")
        for label in (info_managed, info_external):
            label.setContentsMargins(20, 0, 0, 0)

        self._server_widget = ServerWidget(server, self)
        self._connection_widget = QWidget(self)
        self._server_stack = QStackedWidget(self)
        self._server_stack.addWidget(self._server_widget)
        self._server_stack.addWidget(self._connection_widget)

        connection_layout = QVBoxLayout()
        self._connection_widget.setLayout(connection_layout)

        add_header(connection_layout, Settings._server_url)
        server_layout = QHBoxLayout()
        self._server_url = QLineEdit(self._connection_widget)
        self._server_url.textChanged.connect(self.write)
        server_layout.addWidget(self._server_url)
        self._connect_button = QPushButton("Connect", self._connection_widget)
        self._connect_button.clicked.connect(self._connect)
        server_layout.addWidget(self._connect_button)
        connection_layout.addLayout(server_layout)

        self._connection_status = QLabel(self._connection_widget)
        self._connection_status.setWordWrap(True)
        self._connection_status.setTextFormat(Qt.TextFormat.RichText)
        self._connection_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self._connection_status.setOpenExternalLinks(True)

        open_log_button = QLabel(f"<a href='file://{util.log_path}'>View log files</a>", self)
        open_log_button.setToolTip(str(util.log_path))
        open_log_button.linkActivated.connect(self._open_logs)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self._connection_status)
        status_layout.addWidget(open_log_button, alignment=Qt.AlignmentFlag.AlignRight)

        connection_layout.addLayout(status_layout)
        connection_layout.addStretch()

        self._layout.addWidget(self._server_managed)
        self._layout.addWidget(info_managed)
        self._layout.addWidget(self._server_external)
        self._layout.addWidget(info_external)
        self._layout.addWidget(self._server_stack)

        root.connection.state_changed.connect(self.update_server_status)
        self.update_server_status()

    @property
    def server_mode(self):
        if self._server_managed.isChecked():
            return ServerMode.managed
        elif self._server_external.isChecked():
            return ServerMode.external
        else:
            return ServerMode.undefined

    @server_mode.setter
    def server_mode(self, mode: ServerMode):
        if self.server_mode != mode:
            self._server_managed.setChecked(mode is ServerMode.managed)
            self._server_external.setChecked(mode is ServerMode.external)
        self._server_stack.setCurrentWidget(
            self._server_widget if mode is ServerMode.managed else self._connection_widget
        )

    def update_ui(self):
        self._server_widget.update()

    def _read(self):
        self.server_mode = settings.server_mode
        self._server_url.setText(settings.server_url)

    def _write(self):
        settings.server_mode = self.server_mode
        settings.server_url = self._server_url.text()

    def _change_server_mode(self, checked: bool):
        self.server_mode = ServerMode.managed if checked else ServerMode.external
        self.write()

    def _connect(self):
        root.connection.connect(settings.server_url)

    def update_server_status(self):
        server = root.connection
        self._connect_button.setEnabled(server.state != ConnectionState.connecting)
        if server.state == ConnectionState.connected:
            self._connection_status.setText("Connected")
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
        elif server.state == ConnectionState.connecting:
            self._connection_status.setText("Connecting")
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
        elif server.state == ConnectionState.disconnected:
            self._connection_status.setText("Disconnected")
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
        elif server.state == ConnectionState.error:
            self._connection_status.setText(f"<b>Error</b>: {server.error}")
            self._connection_status.setStyleSheet(f"color: {red};")
            if server.missing_resource is not None:
                self._handle_missing_resource(server.missing_resource)

    def _handle_missing_resource(self, resource: MissingResource):
        if resource.kind is ResourceKind.checkpoint:
            self._connection_status.setText(
                "<b>Error</b>: No checkpoints found!\nCheckpoints must be placed into"
                " ComfyUI/models/checkpoints."
            )
        elif resource.kind is ResourceKind.upscaler:
            names = cast(list[str], resource.names)
            self._connection_status.setText(
                "<b>Error</b>: Could not find Upscaler model"
                f" {', '.join(names)}. Make sure to download the model and place it in the"
                " ComfyUI/models/upscale_models folder."
            )
        elif resource.kind is ResourceKind.controlnet:
            names = cast(list[str], resource.names)
            self._connection_status.setText(
                f"<b>Error</b>: Could not find ControlNet model {', '.join(names)}. Make"
                " sure to download the model and place it in the ComfyUI/models/controlnet"
                " folder."
            )
        elif resource.kind is ResourceKind.clip_vision:
            res = [r for r in required_models if r.kind is ResourceKind.clip_vision]
            model = res[0].folder / res[0].filename
            self._connection_status.setText(
                f"<b>Error</b>: Could not find CLIPVision model {model.name} for SD1.5. Make sure"
                f" to download the model and place it in ComfyUI/{model.parent.as_posix()}"
            )
        elif resource.kind is ResourceKind.ip_adapter:
            names = cast(list[str], resource.names)
            self._connection_status.setText(
                "<b>Error</b>: Could not find IPAdapter model"
                f" {', '.join(names)}. Make sure to download the model and place"
                " it in the ComfyUI/models/ipadapter folder."
            )
        elif resource.kind is ResourceKind.inpaint:
            names = cast(list[str], resource.names)
            self._connection_status.setText(
                "<b>Error</b>: Could not find Inpaint model"
                f" {', '.join(names)}. Make sure to download the model and place"
                " it in the ComfyUI/models/inpaint folder."
            )
        elif resource.kind is ResourceKind.node:
            nodes = cast(list[CustomNode], resource.names)
            self._connection_status.setText(
                "<b>Error</b>: The following ComfyUI custom nodes are missing:<ul>"
                + "\n".join((f"<li>{p.name} <a href='{p.url}'>{p.url}</a></li>" for p in nodes))
                + "</ul>Please install them, restart the server and try again."
            )

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_path)))


class StylePresets(SettingsTab):
    _checkpoint_advanced_widgets: list[SettingWidget]
    _default_sampler_widgets: list[SettingWidget]
    _live_sampler_widgets: list[SettingWidget]

    def __init__(self, server: Server):
        super().__init__("Style Presets")
        self.server = server

        frame = QFrame(self)
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setLineWidth(1)
        frame_layout = QHBoxLayout()
        frame.setLayout(frame_layout)

        self._style_list = QComboBox(self)
        self._populate_style_list()
        self._style_list.currentIndexChanged.connect(self._change_style)
        frame_layout.addWidget(self._style_list)

        self._create_style_button = QToolButton(self)
        self._create_style_button.setIcon(Krita.instance().icon("list-add"))
        self._create_style_button.setToolTip("Create a new style")
        self._create_style_button.clicked.connect(self._create_style)
        frame_layout.addWidget(self._create_style_button)

        self._delete_style_button = QToolButton(self)
        self._delete_style_button.setIcon(Krita.instance().icon("deletelayer"))
        self._delete_style_button.setToolTip("Delete the current style")
        self._delete_style_button.clicked.connect(self._delete_style)
        frame_layout.addWidget(self._delete_style_button)

        self._refresh_button = QToolButton(self)
        self._refresh_button.setIcon(Krita.instance().icon("reload-preset"))
        self._refresh_button.setToolTip("Look for new style files")
        self._refresh_button.clicked.connect(self._update_style_list)
        frame_layout.addWidget(self._refresh_button)

        self._open_folder_button = QToolButton(self)
        self._open_folder_button.setIcon(Krita.instance().icon("document-open"))
        self._open_folder_button.setToolTip("Open folder containing style files")
        self._open_folder_button.clicked.connect(self._open_style_folder)
        frame_layout.addWidget(self._open_folder_button)

        self._layout.addWidget(frame)

        self._style_widgets = {}

        def add(name: str, widget: SettingWidget):
            self._style_widgets[name] = widget
            self._layout.addWidget(widget)
            widget.value_changed.connect(self.write)
            return widget

        add("name", TextSetting(StyleSettings.name, self))
        self._style_widgets["name"].value_changed.connect(self._update_name)

        add("sd_checkpoint", ComboBoxSetting(StyleSettings.sd_checkpoint, self))
        self._style_widgets["sd_checkpoint"].add_button(
            Krita.instance().icon("reload-preset"),
            "Look for new checkpoint files",
            root.connection.refresh,
        )
        self._checkpoint_warning = QLabel(self)
        self._checkpoint_warning.setStyleSheet(f"font-style: italic; color: {yellow};")
        self._checkpoint_warning.setVisible(False)
        self._layout.addWidget(self._checkpoint_warning, alignment=Qt.AlignmentFlag.AlignRight)

        checkpoint_advanced = ExpanderButton("Checkpoint configuration (advanced)", self)
        checkpoint_advanced.toggled.connect(self._toggle_checkpoint_advanced)
        self._layout.addWidget(checkpoint_advanced)

        self._checkpoint_advanced_widgets = [add("vae", ComboBoxSetting(StyleSettings.vae, self))]

        self._clip_skip = add(
            "clip_skip", SpinBoxSetting(StyleSettings.clip_skip, self, 0, 12, step=1)
        )
        clip_skip_check = self._clip_skip.add_checkbox("Override")
        clip_skip_check.toggled.connect(self._toggle_clip_skip)
        self._checkpoint_advanced_widgets.append(self._clip_skip)

        self._resolution_spin = add(
            "preferred_resolution",
            SpinBoxSetting(StyleSettings.preferred_resolution, self, 0, 2048, step=8),
        )
        resolution_check = self._resolution_spin.add_checkbox("Override")
        resolution_check.toggled.connect(self._toggle_preferred_resolution)
        self._checkpoint_advanced_widgets.append(self._resolution_spin)

        self._checkpoint_advanced_widgets.append(
            add("v_prediction_zsnr", SwitchSetting(StyleSettings.v_prediction_zsnr, parent=self))
        )
        self._toggle_checkpoint_advanced(False)

        add("loras", LoraList(StyleSettings.loras, self))
        add("style_prompt", LineEditSetting(StyleSettings.style_prompt, self))
        add("negative_prompt", LineEditSetting(StyleSettings.negative_prompt, self))

        sdesc = "Configure sampler type, steps and CFG to tweak the quality of generated images."
        add_header(self._layout, Setting("Sampler Settings", "", sdesc))

        default_sampler_button = ExpanderButton("Quality Preset (generate and upscale)", self)
        default_sampler_button.toggled.connect(self._toggle_default_sampler)
        self._layout.addSpacing(4)
        self._layout.addWidget(default_sampler_button)
        self._default_sampler_widgets = [
            add("sampler", ComboBoxSetting(StyleSettings.sampler, self)),
            add("sampler_steps", SliderSetting(StyleSettings.sampler_steps, self, 1, 100)),
            add("cfg_scale", SliderSetting(StyleSettings.cfg_scale, self, 1.0, 20.0)),
        ]
        self._toggle_default_sampler(False)

        live_sampler_button = ExpanderButton("Performance Preset (live mode)", self)
        live_sampler_button.toggled.connect(self._toggle_live_sampler)
        self._layout.addSpacing(4)
        self._layout.addWidget(live_sampler_button)
        self._live_sampler_widgets = [
            add("live_sampler", ComboBoxSetting(StyleSettings.live_sampler, self)),
            add("live_sampler_steps", SliderSetting(StyleSettings.live_sampler_steps, self, 1, 50)),
            add("live_cfg_scale", SliderSetting(StyleSettings.live_cfg_scale, self, 0.1, 14.0)),
        ]
        self._toggle_live_sampler(False)

        for widget in chain(
            self._checkpoint_advanced_widgets,
            self._default_sampler_widgets,
            self._live_sampler_widgets,
        ):
            widget.indent = 1

        self._layout.addStretch()

        if settings.server_mode is ServerMode.managed:
            self._style_widgets["sd_checkpoint"].add_button(
                Krita.instance().icon("document-open"),
                "Open the folder where checkpoints are stored",
                self._open_checkpoints_folder,
            )
        if self._style_widgets["loras"].open_folder_button:
            self._style_widgets["loras"].open_folder_button.clicked.connect(self._open_lora_folder)

    @property
    def current_style(self) -> Style:
        return Styles.list()[self._style_list.currentIndex()]

    @current_style.setter
    def current_style(self, style: Style):
        index = Styles.list().find(style.filename)[1]
        if index >= 0:
            self._style_list.setCurrentIndex(index)
            self._read_style(style)

    def update_model_lists(self):
        with self._write_guard:
            self._read()

    def _create_style(self):
        checkpoint = self._style_widgets["sd_checkpoint"].value
        # make sure the new style is in the combobox before setting it as the current style
        new_style = Styles.list().create(checkpoint=checkpoint)
        self._update_style_list()
        self.current_style = new_style

    def _delete_style(self):
        Styles.list().delete(self.current_style)
        self._update_style_list()

    def _open_style_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Styles.list().folder)))

    def _populate_style_list(self):
        self._style_list.addItems([f"{style.name} ({style.filename})" for style in Styles.list()])

    def _update_style_list(self):
        previous = None
        if self._style_list.count() > 0:
            previous = self._style_list.currentText()
            self._style_list.clear()
        Styles.list().reload()
        self._populate_style_list()
        if previous is not None:
            self._style_list.setCurrentText(previous)

    def _update_name(self):
        index = self._style_list.currentIndex()
        style = self.current_style
        self._style_list.setItemText(index, f"{style.name} ({style.filename})")
        Styles.list().name_changed.emit()

    def _change_style(self):
        self._read_style(self.current_style)

    def _open_checkpoints_folder(self):
        if self.server.comfy_dir is not None:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.server.comfy_dir / "models" / "checkpoints"))
            )

    def _open_lora_folder(self):
        if self.server.comfy_dir is not None:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.server.comfy_dir / "models" / "loras"))
            )

    def _set_checkpoint_warning(self):
        self._checkpoint_warning.setVisible(False)
        if client := root.connection.client_if_connected:
            version = resolve_sd_version(self.current_style, client)
            if self.current_style.sd_checkpoint not in client.checkpoints:
                self._checkpoint_warning.setText(
                    "The checkpoint used by this style is not installed."
                )
                self._checkpoint_warning.setVisible(True)
            elif version not in client.supported_sd_versions:
                self._checkpoint_warning.setText(
                    f"This is a {version.value} checkpoint, but the {version.value} workload has"
                    " not been installed."
                )
                self._checkpoint_warning.setVisible(True)

    def _toggle_preferred_resolution(self, checked: bool):
        if checked and self._resolution_spin.value == 0:
            sd_ver = resolve_sd_version(self.current_style, root.connection.client_if_connected)
            self._resolution_spin.value = 640 if sd_ver is SDVersion.sd15 else 1024
        elif not checked and self._resolution_spin.value > 0:
            self._resolution_spin.value = 0

    def _toggle_clip_skip(self, checked: bool):
        if checked and self._clip_skip.value == 0:
            sd_ver = resolve_sd_version(self.current_style, root.connection.client_if_connected)
            self._clip_skip.value = 1 if sd_ver is SDVersion.sd15 else 2
        elif not checked and self._clip_skip.value > 0:
            self._clip_skip.value = 0

    def _toggle_checkpoint_advanced(self, checked: bool):
        for widget in self._checkpoint_advanced_widgets:
            widget.visible = checked

    def _toggle_default_sampler(self, checked: bool):
        for widget in self._default_sampler_widgets:
            widget.visible = checked

    def _toggle_live_sampler(self, checked: bool):
        for widget in self._live_sampler_widgets:
            widget.visible = checked

    def _read_style(self, style: Style):
        with self._write_guard:
            for name, widget in self._style_widgets.items():
                widget.value = getattr(style, name)
        self._set_checkpoint_warning()
        self._clip_skip.enabled = style.clip_skip > 0
        self._resolution_spin.enabled = style.preferred_resolution > 0

    def _read(self):
        if client := root.connection.client_if_connected:
            default_vae = cast(str, StyleSettings.vae.default)
            checkpoints = [
                (cp.name, cp.filename, sd_version_icon(cp.sd_version, client))
                for cp in client.checkpoints.values()
                if not (cp.is_refiner or cp.is_inpaint)
            ]
            self._style_widgets["sd_checkpoint"].set_items(checkpoints)
            self._style_widgets["loras"].names = client.loras
            self._style_widgets["vae"].set_items([default_vae] + client.vae_models)
        self._read_style(self.current_style)

    def _write(self):
        style = self.current_style
        for name, widget in self._style_widgets.items():
            setattr(style, name, widget.value)
        self._set_checkpoint_warning()
        style.save()


class DiffusionSettings(SettingsTab):
    def __init__(self):
        super().__init__("Diffusion Settings")

        S = Settings
        self.add("selection_grow", SliderSetting(S._selection_grow, self, 0, 25, "{} %"))
        self.add("selection_feather", SliderSetting(S._selection_feather, self, 0, 25, "{} %"))
        self.add("selection_padding", SliderSetting(S._selection_padding, self, 0, 25, "{} %"))

        self._layout.addStretch()


class InterfaceSettings(SettingsTab):
    def __init__(self):
        super().__init__("Interface Settings")

        S = Settings
        self.add("prompt_line_count", SpinBoxSetting(S._prompt_line_count, self, 1, 10))
        self.add(
            "show_negative_prompt", SwitchSetting(S._show_negative_prompt, ("Show", "Hide"), self)
        )
        self.add("show_control_end", SwitchSetting(S._show_control_end, ("Show", "Hide"), self))
        self.add("auto_preview", SwitchSetting(S._auto_preview, parent=self))
        self.add("new_seed_after_apply", SwitchSetting(S._new_seed_after_apply, parent=self))
        self.add("debug_dump_workflow", SwitchSetting(S._debug_dump_workflow, parent=self))

        self._layout.addStretch()


class HistorySizeWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, maximum: int, step: int, parent=None):
        super().__init__(parent)

        self._history_size = QSpinBox(self)
        self._history_size.setMinimum(5)
        self._history_size.setMaximum(maximum)
        self._history_size.setSingleStep(step)
        self._history_size.setSuffix(" MB")
        self._history_size.valueChanged.connect(self._change_value)

        self._history_usage = QLabel(self)
        self._history_usage.setStyleSheet(f"font-style:italic; color: {green};")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._history_size)
        layout.addWidget(self._history_usage)
        self.setLayout(layout)

    def _change_value(self):
        self.value_changed.emit()

    @property
    def value(self):
        return self._history_size.value()

    @value.setter
    def value(self, v):
        self._history_size.setValue(v)

    def update_usage(self, usage: float):
        self._history_usage.setText(f"Currently using {usage:.1f} MB")


class PerformanceSettings(SettingsTab):
    def __init__(self):
        super().__init__("Performance Settings")

        add_header(self._layout, Settings._history_size)
        self._history_size = HistorySizeWidget(maximum=10000, step=100, parent=self)
        self._history_size.value_changed.connect(self.write)
        self._layout.addWidget(self._history_size)

        add_header(self._layout, Settings._history_storage)
        self._history_storage = HistorySizeWidget(maximum=100, step=5, parent=self)
        self._history_storage.value_changed.connect(self.write)
        self._layout.addWidget(self._history_storage)

        add_header(self._layout, Settings._performance_preset)
        self._device_info = QLabel(self)
        self._device_info.setStyleSheet(f"font-style:italic")
        self._layout.addWidget(self._device_info)

        self._performance_preset = QComboBox(self)
        for preset in PerformancePreset:
            self._performance_preset.addItem(preset.value)
        self._performance_preset.currentIndexChanged.connect(self._change_performance_preset)
        self._layout.addWidget(self._performance_preset, alignment=Qt.AlignmentFlag.AlignLeft)

        self._advanced = QWidget(self)
        self._advanced.setEnabled(settings.performance_preset is PerformancePreset.custom)
        self._advanced.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._advanced)
        advanced_layout = QVBoxLayout()
        self._advanced.setLayout(advanced_layout)

        self._batch_size = SliderSetting(Settings._batch_size, self._advanced, 1, 16)
        self._batch_size.value_changed.connect(self.write)
        advanced_layout.addWidget(self._batch_size)

        self._resolution_multiplier = SliderSetting(
            Settings._resolution_multiplier, self._advanced, 0.3, 1.5, "{:.1f}x"
        )
        self._resolution_multiplier.value_changed.connect(self.write)
        advanced_layout.addWidget(self._resolution_multiplier)

        self._max_pixel_count = SpinBoxSetting(
            Settings._max_pixel_count, self._advanced, 1, 99, 1, " MP"
        )
        self._max_pixel_count.value_changed.connect(self.write)
        advanced_layout.addWidget(self._max_pixel_count)

        self._layout.addStretch()

    def _change_performance_preset(self, index):
        self.write()
        is_custom = settings.performance_preset is PerformancePreset.custom
        self._advanced.setEnabled(is_custom)
        if (
            settings.performance_preset is PerformancePreset.auto
            and root.connection.state is ConnectionState.connected
        ):
            apply_performance_preset(settings, root.connection.client.device_info)
        if not is_custom:
            self.read()

    def update_device_info(self):
        if root.connection.state is ConnectionState.connected:
            client = root.connection.client
            self._device_info.setText(
                f"Device: [{client.device_info.type.upper()}] {client.device_info.name} ("
                f"{client.device_info.vram} GB)"
            )

    def _read(self):
        self._history_size.value = settings.history_size
        self._history_size.update_usage(root.active_model.jobs.memory_usage)
        self._history_storage.value = settings.history_storage
        self._history_storage.update_usage(root.get_active_model_used_storage() / (1024**2))
        self._batch_size.value = settings.batch_size
        self._performance_preset.setCurrentIndex(
            list(PerformancePreset).index(settings.performance_preset)
        )
        self._resolution_multiplier.value = settings.resolution_multiplier
        self._max_pixel_count.value = settings.max_pixel_count
        self.update_device_info()

    def _write(self):
        settings.history_size = self._history_size.value
        settings.history_storage = self._history_storage.value
        settings.batch_size = int(self._batch_size.value)
        settings.resolution_multiplier = self._resolution_multiplier.value
        settings.max_pixel_count = self._max_pixel_count.value
        settings.performance_preset = list(PerformancePreset)[
            self._performance_preset.currentIndex()
        ]


class SettingsDialog(QDialog):
    connection: ConnectionSettings
    styles: StylePresets
    performance: PerformanceSettings

    _instance = None

    @classmethod
    def instance(cls) -> "SettingsDialog":
        assert cls._instance is not None
        return cls._instance

    def __init__(self, server: Server):
        super().__init__()
        type(self)._instance = self

        self.setWindowTitle("Configure Image Diffusion")
        self.setMinimumSize(QSize(840, 480))
        if screen := QGuiApplication.screenAt(QCursor.pos()):
            size = screen.availableSize()
            self.resize(QSize(max(900, int(size.width() * 0.6)), int(size.height() * 0.8)))

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.connection = ConnectionSettings(server)
        self.styles = StylePresets(server)
        self.diffusion = DiffusionSettings()
        self.interface = InterfaceSettings()
        self.performance = PerformanceSettings()

        self._stack = QStackedWidget(self)
        self._list = QListWidget(self)
        self._list.setFixedWidth(120)

        def create_list_item(text: str, widget: QWidget):
            item = QListWidgetItem(text, self._list)
            item.setSizeHint(QSize(112, 24))
            self._stack.addWidget(widget)

        create_list_item("Connection", self.connection)
        create_list_item("Styles", self.styles)
        create_list_item("Diffusion", self.diffusion)
        create_list_item("Interface", self.interface)
        create_list_item("Performance", self.performance)

        self._list.setCurrentRow(0)
        self._list.currentRowChanged.connect(self._change_page)
        layout.addWidget(self._list)

        inner = QVBoxLayout()
        layout.addLayout(inner)
        inner.addWidget(self._stack)
        inner.addSpacing(6)

        self._restore_button = QPushButton("Restore Defaults", self)
        self._restore_button.clicked.connect(self.restore_defaults)

        version_label = QLabel(f"Plugin version: {__version__}", self)
        version_label.setStyleSheet(f"font-style:italic; color: {grey};")

        self._close_button = QPushButton("Ok", self)
        self._close_button.clicked.connect(self._close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._restore_button)
        button_layout.addStretch()
        button_layout.addWidget(version_label)
        button_layout.addStretch()
        button_layout.addWidget(self._close_button)
        inner.addLayout(button_layout)

        root.connection.state_changed.connect(self._update_connection)
        root.connection.models_changed.connect(self.styles.update_model_lists)

    def read(self):
        self.connection.read()
        self.styles.read()
        self.diffusion.read()
        self.interface.read()
        self.performance.read()

    def restore_defaults(self):
        settings.restore()
        settings.save()
        self.read()

    def show(self, style: Optional[Style] = None):
        self.read()
        super().show()
        if style:
            self._list.setCurrentRow(1)
            self.styles.current_style = style
        self._close_button.setFocus()

    def _change_page(self, index):
        self._stack.setCurrentIndex(index)

    def _update_connection(self):
        self.connection.update_server_status()
        if root.connection.state == ConnectionState.connected:
            self.performance.update_device_info()

    def _close(self):
        _ = self.close()
