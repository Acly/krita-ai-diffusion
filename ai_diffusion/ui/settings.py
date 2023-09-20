from enum import Enum
from pathlib import Path
from typing import Optional
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
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QRadioButton,
    QToolButton,
    QComboBox,
    QSlider,
    QWidget,
)
from PyQt5.QtCore import Qt, QSize, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QGuiApplication
from krita import Krita

from .. import (
    MissingResource,
    ResourceKind,
    Setting,
    Settings,
    settings,
    Server,
    ServerMode,
    SDVersion,
    Style,
    Styles,
    StyleSettings,
    PerformancePreset,
)
from .connection import Connection, ConnectionState, apply_performance_preset
from .model import Model
from .server import ServerWidget
from .theme import add_header, icon, red, yellow, green, grey


def _add_title(layout: QVBoxLayout, title: str):
    title_label = QLabel(title)
    title_label.setStyleSheet("font-size: 12pt")
    layout.addWidget(title_label)
    layout.addSpacing(6)


class SettingWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, setting: Setting, parent=None):
        super().__init__(parent)

        self._key_label = QLabel(f"<b>{setting.name}</b><br>{setting.desc}")
        self._key_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 2, 0, 2)
        self._layout.addWidget(self._key_label, alignment=Qt.AlignLeft)
        self.setLayout(self._layout)


class SpinBoxSetting(SettingWidget):
    def __init__(self, setting: Setting, parent=None, minimum=0, maximum=100, suffix=""):
        super().__init__(setting, parent)

        self._spinbox = QSpinBox(self)
        self._spinbox.setMinimumWidth(100)
        self._spinbox.setMinimum(minimum)
        self._spinbox.setMaximum(maximum)
        self._spinbox.setSingleStep(1)
        self._spinbox.setSuffix(suffix)
        self._spinbox.valueChanged.connect(self._change_value)
        self._layout.addWidget(self._spinbox, alignment=Qt.AlignRight)

    def _change_value(self, value: int):
        self.value_changed.emit()

    @property
    def value(self):
        return self._spinbox.value()

    @value.setter
    def value(self, v):
        self._spinbox.setValue(v)


class SliderSetting(SettingWidget):
    def __init__(self, setting: Setting, parent=None, minimum=0, maximum=100):
        super().__init__(setting, parent)

        slider_widget = QWidget(self)
        slider_layout = QHBoxLayout()
        slider_widget.setLayout(slider_layout)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setMinimumWidth(200)
        self._slider.setMaximumWidth(300)
        self._slider.setMinimum(minimum)
        self._slider.setMaximum(maximum)
        self._slider.setSingleStep(1)
        self._slider.valueChanged.connect(self._change_value)
        self._label = QLabel(str(self._slider.value()), self)
        self._label.setMinimumWidth(12)
        slider_layout.addWidget(self._slider)
        slider_layout.addWidget(self._label)
        self._layout.addWidget(slider_widget, alignment=Qt.AlignRight)

    def _change_value(self, value: int):
        self._label.setText(str(value))
        self.value_changed.emit()

    @property
    def value(self):
        return self._slider.value()

    @value.setter
    def value(self, v):
        self._slider.setValue(v)


class ComboBoxSetting(SettingWidget):
    _suppress_change = False
    _enum_type = None
    _original_text = ""

    def __init__(self, setting: Setting, parent=None):
        super().__init__(setting, parent)
        self._combo = QComboBox(self)
        if isinstance(setting.default, Enum):
            self._enum_type = type(setting.default)
            for i, e in enumerate(self._enum_type):
                self._combo.addItem(e.value)
                self._combo.setItemData(i, e.name, Qt.UserRole)
        elif setting.items:
            self._combo.addItems(setting.items)

        self._combo.setMinimumWidth(230)
        self._combo.currentIndexChanged.connect(self._change_value)
        self._layout.addWidget(self._combo, alignment=Qt.AlignRight)
        self._original_text = self._key_label.text()

    def set_items(self, items):
        self._suppress_change = True
        self._combo.clear()
        self._combo.addItems(items)
        self._suppress_change = False

    def _change_value(self):
        if not self._suppress_change:
            self.value_changed.emit()

    def set_text(self, text):
        self._key_label.setText(self._original_text + text)

    @property
    def value(self):
        if self._enum_type is not None:
            name = self._combo.itemData(self._combo.currentIndex(), Qt.UserRole)
            return self._enum_type[name]
        else:
            return self._combo.currentText()

    @value.setter
    def value(self, v):
        if self._enum_type is not None:
            index = self._combo.findData(v.name, Qt.UserRole)
            self._combo.setCurrentIndex(index)
        else:
            self._combo.setCurrentText(v)


class TextSetting(SettingWidget):
    def __init__(self, setting: Setting, parent=None):
        super().__init__(setting, parent)
        self._edit = QLineEdit(self)
        self._edit.setMinimumWidth(230)
        self._edit.setMaximumWidth(300)
        self._edit.textChanged.connect(self._change_value)
        self._layout.addWidget(self._edit, alignment=Qt.AlignRight)

    def _change_value(self):
        self.value_changed.emit()

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

            self._select = QComboBox(self)
            self._select.addItems(lora_names)
            self._select.currentIndexChanged.connect(self._update)

            self._strength = QSpinBox(self)
            self._strength.setMinimum(-100)
            self._strength.setMaximum(100)
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

        @property
        def value(self):
            return dict(name=self._select.currentText(), strength=self._strength.value() / 100)

        @value.setter
        def value(self, v):
            self._select.setCurrentText(v["name"])
            self._strength.setValue(int(v["strength"] * 100))

    value_changed = pyqtSignal()

    def __init__(self, setting: Setting, parent=None):
        super().__init__(parent)
        self._loras = []
        self._items = []
        self._item_list = None

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_text_layout = QVBoxLayout()
        add_header(header_text_layout, setting)

        self._add_button = QPushButton("Add", self)
        self._add_button.setMinimumWidth(100)
        self._add_button.clicked.connect(self._add_item)

        header_layout.addLayout(header_text_layout, 3)
        header_layout.addWidget(self._add_button, 1, Qt.AlignRight | Qt.AlignVCenter)
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
            item._select.clear()
            item._select.addItems(v)

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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
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
        self._connection_status.setTextFormat(Qt.RichText)
        self._connection_status.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self._connection_status.setOpenExternalLinks(True)
        connection_layout.addWidget(self._connection_status)
        connection_layout.addStretch()

        self._layout.addWidget(self._server_managed)
        self._layout.addWidget(info_managed)
        self._layout.addWidget(self._server_external)
        self._layout.addWidget(info_external)
        self._layout.addWidget(self._server_stack)

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

    def update(self):
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
        Connection.instance().connect(settings.server_url)

    def update_server_status(self):
        server = Connection.instance()
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
                " [ComfyUI]/model/checkpoints."
            )

        elif resource.kind is ResourceKind.controlnet:
            self._connection_status.setText(
                f"<b>Error</b>: Could not find ControlNet model {', '.join(resource.names)}. Make"
                " sure to download the model and place it in the [ComfyUI]/models/controlnet"
                " folder."
            )
        elif resource.kind is ResourceKind.clip_vision:
            model = Path("models", "clip_vision", resource.names[0])
            self._connection_status.setText(
                f"<b>Error</b>: Could not find CLIPVision model {model.name} for SD1.5. Make sure"
                f" to download the model and place it in [ComfyUI]/{model.parent.as_posix()}"
            )
        elif resource.kind is ResourceKind.ip_adapter:
            self._connection_status.setText(
                f"<b>Error</b>: Could not find IPAdapter model {', '.join(resource.names)}. Make"
                " sure to download the model and place it in the"
                " [ComfyUI]/custom_nodes/IPAdapter-ComfyUI/models folder."
            )
        elif resource.kind is ResourceKind.node:
            self._connection_status.setText(
                "<b>Error</b>: The following ComfyUI custom nodes are missing:<ul>"
                + "\n".join(
                    (f"<li>{p.name} <a href='{p.url}'>{p.url}</a></li>" for p in resource.names)
                )
                + "</ul>Please install them, restart the server and try again."
            )


class StylePresets(SettingsTab):
    def __init__(self):
        super().__init__("Style Presets")

        frame = QFrame(self)
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setLineWidth(1)
        frame_layout = QHBoxLayout()
        frame.setLayout(frame_layout)

        self._style_list = QComboBox(self)
        self._populate_style_list()
        self._style_list.currentIndexChanged.connect(self._change_style)
        frame_layout.addWidget(self._style_list)

        self._refresh_button = QToolButton(self)
        self._refresh_button.setIcon(Krita.instance().icon("reload-preset"))
        self._refresh_button.clicked.connect(self._update_style_list)
        frame_layout.addWidget(self._refresh_button)

        self._open_folder_button = QToolButton(self)
        self._open_folder_button.setIcon(Krita.instance().icon("document-open"))
        self._open_folder_button.clicked.connect(self._open_folder)
        frame_layout.addWidget(self._open_folder_button)
        
        self._create_style_button = QToolButton(self)
        self._create_style_button.setIcon(Krita.instance().icon("list-add"))
        self._create_style_button.clicked.connect(self._create_style)
        frame_layout.addWidget(self._create_style_button)
        
        self._delete_style_button = QToolButton(self)
        self._delete_style_button.setIcon(Krita.instance().icon("deletelayer"))
        self._delete_style_button.clicked.connect(self._delete_style)
        frame_layout.addWidget(self._delete_style_button)

        self._layout.addWidget(frame)

        self._style_widgets = {}

        def add(name: str, widget: QWidget):
            self._style_widgets[name] = widget
            self._layout.addWidget(widget)
            widget.value_changed.connect(self.write)

        add("name", TextSetting(StyleSettings.name, self))
        add("sd_checkpoint", ComboBoxSetting(StyleSettings.sd_checkpoint, self))
        add("loras", LoraList(StyleSettings.loras, self))
        add("style_prompt", LineEditSetting(StyleSettings.style_prompt, self))
        add("negative_prompt", LineEditSetting(StyleSettings.negative_prompt, self))
        add("sd_version", ComboBoxSetting(StyleSettings.sd_version, self))
        add("vae", ComboBoxSetting(StyleSettings.vae, self))
        add("sampler", ComboBoxSetting(StyleSettings.sampler, self))
        add("sampler_steps", SliderSetting(StyleSettings.sampler_steps, self, 1, 100))
        add(
            "sampler_steps_upscaling",
            SliderSetting(StyleSettings.sampler_steps_upscaling, self, 1, 100),
        )
        add("cfg_scale", SliderSetting(StyleSettings.cfg_scale, self, 1, 20))
        self._layout.addStretch()
        self._style_widgets["name"].value_changed.connect(self._update_name)

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
        self._read()

    def _create_style(self):
        # make sure the new style is in the combobox before setting it as the current style
        new_style = Styles.list().create()
        self._update_style_list()
        self.current_style = new_style

    def _delete_style(self):
        Styles.list().delete(self.current_style)
        self._update_style_list()

    def _open_folder(self):
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

    def _set_sd_version_text(self):
        if self.current_style.sd_version is SDVersion.auto:
            actual = self.current_style.sd_version_resolved
            self._style_widgets["sd_version"].set_text(f". <i>Detected {actual.value}</i>")
        else:
            self._style_widgets["sd_version"].set_text("")

    def _read_style(self, style: Style):
        with self._write_guard:
            for name, widget in self._style_widgets.items():
                widget.value = getattr(style, name)

    def _read(self):
        if Connection.instance().state == ConnectionState.connected:
            client = Connection.instance().client
            self._style_widgets["sd_checkpoint"].set_items(client.checkpoints)
            self._style_widgets["loras"].names = client.lora_models
            self._style_widgets["vae"].set_items([StyleSettings.vae.default] + client.vae_models)
        self._read_style(self.current_style)
        self._set_sd_version_text()

    def _write(self):
        style = self.current_style
        for name, widget in self._style_widgets.items():
            setattr(style, name, widget.value)
        self._set_sd_version_text()
        style.save()


class DiffusionSettings(SettingsTab):
    def __init__(self):
        super().__init__("Diffusion Settings")

        self.add("random_seed", TextSetting(Settings._random_seed, self))
        self._fixed_seed_checkbox = QCheckBox("Use fixed seed", self)
        self._fixed_seed_checkbox.stateChanged.connect(self.write)
        self._layout.addWidget(self._fixed_seed_checkbox)

        self._layout.addStretch()

    def _read(self):
        self._fixed_seed_checkbox.setChecked(settings.fixed_seed)
        self._widgets["random_seed"].setEnabled(settings.fixed_seed)

    def _write(self):
        settings.fixed_seed = self._fixed_seed_checkbox.isChecked()
        self._widgets["random_seed"].setEnabled(settings.fixed_seed)


class PerformanceSettings(SettingsTab):
    def __init__(self):
        super().__init__("Performance Settings")

        add_header(self._layout, Settings._history_size)
        self._history_size = QSpinBox(self)
        self._history_size.setMinimum(8)
        self._history_size.setMaximum(1024 * 16)
        self._history_size.setSingleStep(100)
        self._history_size.setSuffix(" MB")
        self._history_size.valueChanged.connect(self.write)
        self._history_usage = QLabel(self)
        self._history_usage.setStyleSheet(f"font-style:italic; color: {green};")
        history_layout = QHBoxLayout()
        history_layout.addWidget(self._history_size)
        history_layout.addWidget(self._history_usage)
        self._layout.addLayout(history_layout)

        add_header(self._layout, Settings._performance_preset)
        self._device_info = QLabel(self)
        self._device_info.setStyleSheet(f"font-style:italic")
        self._layout.addWidget(self._device_info)

        self._performance_preset = QComboBox(self)
        for preset in PerformancePreset:
            self._performance_preset.addItem(preset.value)
        self._performance_preset.currentIndexChanged.connect(self._change_performance_preset)
        self._layout.addWidget(self._performance_preset, alignment=Qt.AlignLeft)

        self._advanced = QWidget(self)
        self._advanced.setEnabled(settings.performance_preset is PerformancePreset.custom)
        self._advanced.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._advanced)
        advanced_layout = QVBoxLayout()
        self._advanced.setLayout(advanced_layout)

        self._batch_size = SliderSetting(Settings._batch_size, self._advanced, 1, 16)
        self._batch_size.value_changed.connect(self.write)
        advanced_layout.addWidget(self._batch_size)

        self._diffusion_tile_size = SpinBoxSetting(
            Settings._diffusion_tile_size, self._advanced, 768, 4096 * 2
        )
        self._diffusion_tile_size.value_changed.connect(self.write)
        advanced_layout.addWidget(self._diffusion_tile_size)

        self._layout.addStretch()

    def _change_performance_preset(self, index):
        self.write()
        is_custom = settings.performance_preset is PerformancePreset.custom
        self._advanced.setEnabled(is_custom)
        if (
            settings.performance_preset is PerformancePreset.auto
            and Connection.instance().state is ConnectionState.connected
        ):
            apply_performance_preset(settings, Connection.instance().client.device_info)
        if not is_custom:
            self.read()

    def update_device_info(self):
        if Connection.instance().state is ConnectionState.connected:
            client = Connection.instance().client
            self._device_info.setText(
                f"Device: [{client.device_info.type.upper()}] {client.device_info.name} ("
                f"{client.device_info.vram} GB)"
            )

    def _read(self):
        memory_usage = 0
        if Model.active() is not None:
            memory_usage = Model.active().jobs.memory_usage
        self._history_size.setValue(settings.history_size)
        self._history_usage.setText(f"Currently using {memory_usage:.1f} MB")
        self._batch_size.value = settings.batch_size
        self._performance_preset.setCurrentIndex(
            list(PerformancePreset).index(settings.performance_preset)
        )
        self._diffusion_tile_size.value = settings.diffusion_tile_size
        self.update_device_info()

    def _write(self):
        settings.history_size = self._history_size.value()
        settings.batch_size = self._batch_size.value
        settings.diffusion_tile_size = self._diffusion_tile_size.value
        settings.performance_preset = list(PerformancePreset)[
            self._performance_preset.currentIndex()
        ]


class SettingsDialog(QDialog):
    connection: ConnectionSettings
    styles: StylePresets
    performance: PerformanceSettings

    _instance = None

    @classmethod
    def instance(Class):
        assert Class._instance is not None
        return Class._instance

    def __init__(self, main_window: QMainWindow, server: Server):
        super().__init__()
        type(self)._instance = self

        self.setWindowTitle("Configure Image Diffusion")
        self.setMinimumSize(QSize(840, 480))
        screen_size = QGuiApplication.primaryScreen().availableSize()
        self.resize(
            QSize(max(900, int(screen_size.width() * 0.6)), int(screen_size.height() * 0.8))
        )

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.connection = ConnectionSettings(server)
        self.styles = StylePresets()
        self.diffusion = DiffusionSettings()
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

        self._close_button = QPushButton("Ok", self)
        self._close_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._restore_button)
        button_layout.addStretch()
        button_layout.addWidget(self._close_button)
        inner.addLayout(button_layout)

        Connection.instance().changed.connect(self._update_connection)

    def read(self):
        self.connection.read()
        self.styles.read()
        self.diffusion.read()
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
        if Connection.instance().state == ConnectionState.connected:
            self.styles.update_model_lists()
            self.performance.update_device_info()
