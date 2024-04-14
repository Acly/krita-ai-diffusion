from __future__ import annotations

import functools
from enum import Enum
from itertools import chain
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
from PyQt5.QtCore import Qt, QMetaObject, QSize, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QGuiApplication, QIcon, QCursor
from krita import Krita

from ..client import User, resolve_sd_version
from ..cloud_client import CloudClient
from ..resources import SDVersion, CustomNode, MissingResource, ResourceKind, required_models
from ..settings import Setting, Settings, ServerMode, PerformancePreset, settings
from ..server import Server
from ..style import Style, Styles, StyleSettings, SamplerPresets
from ..root import root
from ..connection import ConnectionState, apply_performance_preset
from ..properties import Binding
from .. import eventloop, util, __version__
from .server import ServerWidget
from .switch import SwitchWidget
from .theme import SignalBlocker, add_header, icon, sd_version_icon, red, yellow, green, grey


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

        self.setEnabled(settings.server_mode is not ServerMode.cloud)
        settings.changed.connect(self._handle_settings_change)

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

    def _handle_settings_change(self, key: str, value):
        if key == "server_mode":
            self.setEnabled(value is not ServerMode.cloud)

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


class UserWidget(QFrame):
    _user: User | None = None
    _connections: list[QMetaObject.Connection | Binding]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connections = []

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setVisible(False)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self._user_name = QLabel("", self)
        self._user_name.setStyleSheet("font-weight:bold")
        user_name_layout = QHBoxLayout()
        user_name_layout.addWidget(QLabel("Account:", self), 0)
        user_name_layout.addWidget(self._user_name, 1)
        layout.addLayout(user_name_layout)

        self._images_generated = QLabel("", self)
        image_count_layout = QHBoxLayout()
        image_count_layout.addWidget(QLabel("Total generated:", self), 0)
        image_count_layout.addWidget(self._images_generated, 1)
        layout.addLayout(image_count_layout)

        self._tokens_remaining = QLabel("", self)
        self._tokens_remaining.setStyleSheet("font-weight:bold")
        image_remaining_layout = QHBoxLayout()
        image_remaining_layout.addWidget(QLabel("Image tokens remaining:", self), 0)
        image_remaining_layout.addWidget(self._tokens_remaining, 1)
        layout.addLayout(image_remaining_layout)

        self._logout_button = QPushButton("Sign out", self)
        self._logout_button.setMinimumWidth(200)
        self._logout_button.clicked.connect(self._logout)
        layout.addWidget(self._logout_button)

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, user: User | None):
        if self._user is not user:
            Binding.disconnect_all(self._connections)
            self.setVisible(user is not None)

            self._user = user
            if user is not None:
                self._user_name.setText(user.name)
                self._connections = [
                    user.images_generated_changed.connect(self._update_counts),
                    user.credits_changed.connect(self._update_counts),
                ]
                self._update_counts()

    def _update_counts(self):
        user = util.ensure(self.user)
        self._images_generated.setText(str(user.images_generated))
        self._tokens_remaining.setText(str(user.credits))

    def _logout(self):
        eventloop.run(self._disconnect_and_logout())

    async def _disconnect_and_logout(self):
        await root.connection.disconnect()
        settings.access_token = ""
        settings.save()


class CloudWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 12, 4, 4)
        self.setLayout(layout)

        service_url = CloudClient.default_url
        service_url_text = service_url.removeprefix("https://").removesuffix("/")
        service_label = QLabel(f"<a href='{service_url}'>{service_url_text}</a>", self)
        service_label.setStyleSheet("font-size: 12pt")
        service_label.setTextFormat(Qt.TextFormat.RichText)
        service_label.setOpenExternalLinks(True)
        layout.addWidget(service_label)

        self._connection_status = QLabel(self)
        self._connection_status.setWordWrap(True)
        self._connection_status.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._connection_status)

        self.connect_button = QPushButton("Login", self)
        self.connect_button.setMinimumWidth(200)
        self.connect_button.setMinimumHeight(int(1.3 * self.connect_button.sizeHint().height()))
        self.connect_button.clicked.connect(self._connect)

        self._sign_out_button = QPushButton("Sign out", self)
        self._sign_out_button.setVisible(False)
        self._sign_out_button.setMinimumWidth(200)
        self._sign_out_button.clicked.connect(self._sign_out)

        self._user_widget = UserWidget(self)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.connect_button)
        buttons_layout.addWidget(self._sign_out_button)

        connect_layout = QHBoxLayout()
        connect_layout.addLayout(buttons_layout)
        connect_layout.addWidget(self._user_widget)
        connect_layout.addStretch()
        layout.addLayout(connect_layout)

        layout.addStretch()

    def update_connection_state(self, state: ConnectionState):
        is_connected = state == ConnectionState.connected
        self.connect_button.setVisible(not is_connected)
        self._sign_out_button.setVisible(False)
        self._user_widget.user = root.connection.user

        if state in [ConnectionState.auth_missing, ConnectionState.auth_error]:
            self.connect_button.setText("Sign in")
            self.connect_button.setEnabled(True)
            self._connection_status.setText("Disconnected")
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
        elif state is ConnectionState.auth_pending:
            self.connect_button.setText("Sign in")
            self.connect_button.setEnabled(False)
            self._connection_status.setText("Waiting for sign-in to complete...")
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
            self._connection_status.setVisible(True)
        elif state is ConnectionState.connected:
            self._connection_status.setText("Connected")
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
            self._user_widget.user = root.connection.user
        else:
            can_connect = state in [ConnectionState.disconnected, ConnectionState.error]
            self.connect_button.setEnabled(can_connect)
            self.connect_button.setText("Connect" if can_connect else "Connected")

        if state in [ConnectionState.error, ConnectionState.auth_error]:
            error = root.connection.error or "Unknown error"
            self._connection_status.setText(f"<b>Error</b>: {error.removeprefix('Error: ')}")
            self._connection_status.setStyleSheet(f"color: {red}; font-weight:bold")
            self._connection_status.setVisible(True)
            if settings.access_token:
                self._sign_out_button.setVisible(True)

    def _connect(self):
        connection = root.connection
        if connection.state in [ConnectionState.auth_missing, ConnectionState.auth_error]:
            connection.sign_in()
        else:
            connection.connect()

    def _sign_out(self):
        settings.access_token = ""
        settings.save()


class ConnectionSettings(SettingsTab):
    def __init__(self, server: Server):
        super().__init__("Server Configuration")

        self._server_cloud = QRadioButton("Online Service [BETA]", self)
        self._server_managed = QRadioButton("Local Managed Server", self)
        self._server_external = QRadioButton("Custom Server (local or remote)", self)
        info_cloud = QLabel("Generate images via GPU Cloud Service", self)
        info_managed = QLabel(
            "Let the Krita plugin install and run a local server on your machine", self
        )
        info_external = QLabel(
            "Connect to a running ComfyUI instance which you set up and maintain yourself", self
        )
        for button in (self._server_cloud, self._server_managed, self._server_external):
            button.setStyleSheet("font-weight:bold")
            button.toggled.connect(self._change_server_mode)
        for label in (info_cloud, info_managed, info_external):
            label.setContentsMargins(20, 0, 0, 0)

        self._cloud_widget = CloudWidget(self)
        self._server_widget = ServerWidget(server, self)
        self._connection_widget = QWidget(self)
        self._server_stack = QStackedWidget(self)
        self._server_stack.addWidget(self._cloud_widget)
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

        open_log_button = QLabel(f"<a href='file://{util.log_dir}'>View log files</a>", self)
        open_log_button.setToolTip(str(util.log_dir))
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
        self._layout.addWidget(self._server_cloud)
        self._layout.addWidget(info_cloud)
        self._layout.addWidget(self._server_stack)

        root.connection.state_changed.connect(self.update_server_status)
        self.update_server_status()

    @property
    def server_mode(self):
        if self._server_cloud.isChecked():
            return ServerMode.cloud
        elif self._server_managed.isChecked():
            return ServerMode.managed
        elif self._server_external.isChecked():
            return ServerMode.external
        else:
            return ServerMode.undefined

    @server_mode.setter
    def server_mode(self, mode: ServerMode):
        if self.server_mode != mode:
            self._server_cloud.setChecked(mode is ServerMode.cloud)
            self._server_managed.setChecked(mode is ServerMode.managed)
            self._server_external.setChecked(mode is ServerMode.external)
        widget = {
            ServerMode.cloud: self._cloud_widget,
            ServerMode.managed: self._server_widget,
            ServerMode.external: self._connection_widget,
        }[mode]
        self._server_stack.setCurrentWidget(widget)

    def update_ui(self):
        self._server_widget.update()

    def _read(self):
        self.server_mode = settings.server_mode
        self._server_url.setText(settings.server_url)

    def _write(self):
        settings.server_mode = self.server_mode
        settings.server_url = self._server_url.text()

    def _change_server_mode(self, checked: bool):
        if self._server_cloud.isChecked():
            self.server_mode = ServerMode.cloud
        elif self._server_managed.isChecked():
            self.server_mode = ServerMode.managed
        elif self._server_external.isChecked():
            self.server_mode = ServerMode.external
        self.write()

    def _connect(self):
        root.connection.connect()

    def update_server_status(self):
        connection = root.connection
        self._cloud_widget.update_connection_state(connection.state)
        self._connect_button.setEnabled(connection.state != ConnectionState.connecting)
        if connection.state == ConnectionState.connected:
            self._connection_status.setText("Connected")
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
        elif connection.state == ConnectionState.connecting:
            self._connection_status.setText("Connecting")
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
        elif connection.state == ConnectionState.disconnected:
            self._connection_status.setText("Disconnected")
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
        elif connection.state == ConnectionState.error:
            self._connection_status.setText(f"<b>Error</b>: {connection.error}")
            self._connection_status.setStyleSheet(f"color: {red};")
            if connection.missing_resource is not None:
                self._handle_missing_resource(connection.missing_resource)

    def _handle_missing_resource(self, resource: MissingResource):
        if resource.kind is ResourceKind.checkpoint:
            self._connection_status.setText(
                "<b>Error</b>: No checkpoints found!\nCheckpoints must be placed into"
                " ComfyUI/models/checkpoints."
            )
        elif resource.kind is ResourceKind.node:
            nodes = cast(list[CustomNode], resource.names)
            self._connection_status.setText(
                "<b>Error</b>: The following ComfyUI custom nodes are missing:<ul>"
                + "\n".join((f"<li>{p.name} <a href='{p.url}'>{p.url}</a></li>" for p in nodes))
                + "</ul>Please install them, restart the server and try again."
            )
        else:
            search_paths = resource.search_path_string.replace("\n", "<br>")
            self._connection_status.setText(
                f"<b>Error</b>: {str(resource)}<br>{search_paths}<br><br>"
                "See <a href='https://github.com/Acly/krita-ai-diffusion/wiki/ComfyUI-Setup'>Custom ComfyUI Setup</a> for required models.<br>"
                "Check the client.log file for more details."
            )

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_dir)))


class SamplerWidget(QWidget):

    prefix: str

    value_changed = pyqtSignal()

    def __init__(self, prefix: str, title: str, parent):
        super().__init__(parent)
        self.prefix = prefix

        expander = ExpanderButton(title, self)
        expander.toggled.connect(self._toggle_expand)

        self._preset = QComboBox(self)
        self._preset.addItems(SamplerPresets.instance().names())
        self._preset.setMinimumWidth(230)
        self._preset.currentIndexChanged.connect(self._select_preset)

        header_layout = QHBoxLayout()
        header_layout.addWidget(expander)
        header_layout.addStretch()
        header_layout.addWidget(self._preset)

        self._steps = SliderSetting(StyleSettings.sampler_steps, self, 1, 100)
        self._steps.indent = 1
        self._steps.value_changed.connect(self.notify_changed)

        self._cfg = SliderSetting(StyleSettings.cfg_scale, self, 1.0, 20.0)
        self._cfg.indent = 1
        self._cfg.value_changed.connect(self.notify_changed)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addLayout(header_layout)
        layout.addWidget(self._steps)
        layout.addWidget(self._cfg)
        self.setLayout(layout)

        self._toggle_expand(False)

    def _toggle_expand(self, expanded: bool):
        self._steps.setVisible(expanded)
        self._cfg.setVisible(expanded)

    def _select_preset(self, index: int):
        name = self._preset.currentText()
        preset = SamplerPresets.instance()[name]
        self._steps.value = preset.steps
        self._cfg.value = preset.cfg

    def notify_changed(self):
        self.value_changed.emit()

    def read(self, style: Style):
        self._preset.setCurrentText(getattr(style, f"{self.prefix}sampler"))
        self._steps.value = getattr(style, f"{self.prefix}sampler_steps")
        self._cfg.value = getattr(style, f"{self.prefix}cfg_scale")

    def write(self, style: Style):
        setattr(style, f"{self.prefix}sampler", self._preset.currentText())
        setattr(style, f"{self.prefix}sampler_steps", self._steps.value)
        setattr(style, f"{self.prefix}cfg_scale", self._cfg.value)


class StylePresets(SettingsTab):
    _checkpoint_advanced_widgets: list[SettingWidget]
    _default_sampler_widgets: list[SettingWidget]
    _live_sampler_widgets: list[SettingWidget]

    def __init__(self, server: Server):
        super().__init__("Style Presets")
        self.server = server

        self._style_list = QComboBox(self)
        self._style_list.currentIndexChanged.connect(self._change_style)

        self._create_style_button = QToolButton(self)
        self._create_style_button.setIcon(Krita.instance().icon("list-add"))
        self._create_style_button.setToolTip("Create a new style")
        self._create_style_button.clicked.connect(self._create_style)

        self._delete_style_button = QToolButton(self)
        self._delete_style_button.setIcon(Krita.instance().icon("deletelayer"))
        self._delete_style_button.setToolTip("Delete the current style")
        self._delete_style_button.clicked.connect(self._delete_style)

        self._refresh_button = QToolButton(self)
        self._refresh_button.setIcon(Krita.instance().icon("reload-preset"))
        self._refresh_button.setToolTip("Look for new style files")
        self._refresh_button.clicked.connect(Styles.list().reload)

        self._open_folder_button = QToolButton(self)
        self._open_folder_button.setIcon(Krita.instance().icon("document-open"))
        self._open_folder_button.setToolTip("Open folder containing style files")
        self._open_folder_button.clicked.connect(self._open_style_folder)

        self._show_builtin_checkbox = QCheckBox("Show pre-installed styles", self)
        self._show_builtin_checkbox.toggled.connect(self.write)

        style_control_layout = QHBoxLayout()
        style_control_layout.setContentsMargins(0, 0, 0, 0)
        style_control_layout.addWidget(self._style_list)
        style_control_layout.addWidget(self._create_style_button)
        style_control_layout.addWidget(self._delete_style_button)
        style_control_layout.addWidget(self._refresh_button)
        style_control_layout.addWidget(self._open_folder_button)
        frame_layout = QVBoxLayout()
        frame_layout.addLayout(style_control_layout)
        frame_layout.addWidget(self._show_builtin_checkbox, alignment=Qt.AlignmentFlag.AlignRight)

        frame = QFrame(self)
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setLineWidth(1)
        frame.setLayout(frame_layout)
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

        self._clip_skip = add("clip_skip", SpinBoxSetting(StyleSettings.clip_skip, self, 0, 12))
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
        self._checkpoint_advanced_widgets.append(
            add(
                "self_attention_guidance",
                SwitchSetting(StyleSettings.self_attention_guidance, parent=self),
            )
        )
        self._toggle_checkpoint_advanced(False)

        add("loras", LoraList(StyleSettings.loras, self))
        add("style_prompt", LineEditSetting(StyleSettings.style_prompt, self))
        add("negative_prompt", LineEditSetting(StyleSettings.negative_prompt, self))

        sdesc = "Configure sampler type, steps and CFG to tweak the quality of generated images."
        add_header(self._layout, Setting("Sampler Settings", "", sdesc))

        self._default_sampler = SamplerWidget("", "Quality Preset (generate and upscale)", self)
        self._default_sampler.value_changed.connect(self.write)
        self._layout.addWidget(self._default_sampler)

        self._live_sampler = SamplerWidget("live_", "Performance Preset (live mode)", self)
        self._live_sampler.value_changed.connect(self.write)
        self._layout.addWidget(self._live_sampler)

        self._layout.addStretch()

        if settings.server_mode is ServerMode.managed:
            self._style_widgets["sd_checkpoint"].add_button(
                Krita.instance().icon("document-open"),
                "Open the folder where checkpoints are stored",
                self._open_checkpoints_folder,
            )
        if self._style_widgets["loras"].open_folder_button:
            self._style_widgets["loras"].open_folder_button.clicked.connect(self._open_lora_folder)

        self._populate_style_list()
        Styles.list().changed.connect(self._update_style_list)

    @property
    def current_style(self) -> Style:
        styles = Styles.list()
        return styles.find(self._style_list.currentData()) or styles.default

    @current_style.setter
    def current_style(self, style: Style):
        index = self._style_list.findData(style.filename)
        if index >= 0:
            self._style_list.setCurrentIndex(index)
            self._read_style(style)

    def update_model_lists(self):
        with self._write_guard:
            self._read()

    def _create_style(self):
        cp = self._style_widgets["sd_checkpoint"].value or StyleSettings.sd_checkpoint.default
        new_style = Styles.list().create(checkpoint=cp)
        self.current_style = new_style

    def _delete_style(self):
        Styles.list().delete(self.current_style)

    def _open_style_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Styles.list().user_folder)))

    def _populate_style_list(self):
        for style in Styles.list().filtered():
            self._style_list.addItem(f"{style.name} ({style.filename})", style.filename)

    def _update_style_list(self):
        previous = None
        with SignalBlocker(self._style_list):
            if self._style_list.count() > 0:
                previous = self._style_list.currentData()
                self._style_list.clear()
            self._populate_style_list()
            if previous is not None:
                i = self._style_list.findData(previous)
                self._style_list.setCurrentIndex(max(0, i))
        self._change_style()

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
            if self.current_style.sd_checkpoint not in client.models.checkpoints:
                self._checkpoint_warning.setText(
                    "The checkpoint used by this style is not installed."
                )
                self._checkpoint_warning.setVisible(True)
            else:
                version = resolve_sd_version(self.current_style, client)
                if not client.supports_version(version):
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

    def _read_style(self, style: Style):
        with self._write_guard:
            for name, widget in self._style_widgets.items():
                widget.value = getattr(style, name)
            self._default_sampler.read(style)
            self._live_sampler.read(style)
        self._set_checkpoint_warning()
        self._clip_skip.enabled = style.clip_skip > 0
        self._resolution_spin.enabled = style.preferred_resolution > 0

    def _read(self):
        self._show_builtin_checkbox.setChecked(settings.show_builtin_styles)
        if client := root.connection.client_if_connected:
            default_vae = cast(str, StyleSettings.vae.default)
            checkpoints = [
                (cp.name, cp.filename, sd_version_icon(cp.sd_version, client))
                for cp in client.models.checkpoints.values()
                if not (cp.is_refiner or cp.is_inpaint)
            ]
            self._style_widgets["sd_checkpoint"].set_items(checkpoints)
            self._style_widgets["loras"].names = client.models.loras
            self._style_widgets["vae"].set_items([default_vae] + client.models.vae)
        self._read_style(self.current_style)

    def _write(self):
        if settings.show_builtin_styles != self._show_builtin_checkbox.isChecked():
            settings.show_builtin_styles = self._show_builtin_checkbox.isChecked()
        style = self.current_style
        for name, widget in self._style_widgets.items():
            if widget.value is not None:
                setattr(style, name, widget.value)
        self._default_sampler.write(style)
        self._live_sampler.write(style)
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

        self._open_folder_link = QLabel(
            f"<a href='file://{util.user_data_dir}'>Open Settings folder</a>", self
        )
        self._open_folder_link.linkActivated.connect(self._open_settings_folder)
        self._open_folder_link.setToolTip(str(util.user_data_dir))

        self._close_button = QPushButton("Ok", self)
        self._close_button.clicked.connect(self._close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._restore_button)
        button_layout.addStretch()
        button_layout.addWidget(version_label)
        button_layout.addStretch()
        button_layout.addWidget(self._open_folder_link)
        button_layout.addSpacing(8)
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

    def _open_settings_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.user_data_dir)))

    def _close(self):
        _ = self.close()
