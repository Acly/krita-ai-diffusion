from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QComboBox,
    QSlider,
    QWidget,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal

from .. import Client, MissingResource, ResourceKind, Setting, Settings, settings, GPUMemoryPreset
from .connection import Connection, ConnectionState
from .model import Model


class SliderWithValue(QWidget):
    value_changed = pyqtSignal(int)

    def __init__(self, minimum=0, maximum=100, parent=None, format_string="{}"):
        super().__init__(parent)
        self._format_string = format_string

        layout = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setMinimum(minimum)
        self._slider.setMaximum(maximum)
        self._slider.setSingleStep(1)
        self._slider.valueChanged.connect(self._change_value)
        self._label = QLabel(str(self._slider.value()), self)
        layout.addWidget(self._slider)
        layout.addWidget(self._label)
        self.setLayout(layout)

    def _change_value(self, value: int):
        self._label.setText(self._format_string.format(value))
        self.value_changed.emit(value)

    @property
    def value(self):
        return self._slider.value()

    @value.setter
    def value(self, v):
        self._slider.setValue(v)


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
            self._strength.setSuffix("%")
            self._strength.valueChanged.connect(self._update)

            self._remove = QToolButton(self)
            self._remove.setText("Remove")
            self._remove.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self._remove.clicked.connect(self.remove)

            layout.addWidget(self._select, 2)
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

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loras = []
        self._items = []
        self._item_list = None

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        self._item_list = QVBoxLayout()
        self._item_list.setContentsMargins(0, 0, 0, 0)
        self._layout.insertLayout(0, self._item_list)

        self._add_button = QPushButton("Add", self)
        self._add_button.clicked.connect(self._add_item)
        self._layout.addWidget(self._add_button)

    def _add_item(self, lora=None):
        assert self._item_list is not None
        item = self.Item(self._loras, self)
        if isinstance(lora, dict):
            item.value = lora
        item.changed.connect(self._update_item)
        item.removed.connect(self._remove_item)
        self._items.append(item)
        self._item_list.addWidget(item)
        self.changed.emit()

    def _remove_item(self, item: QWidget):
        self._items.remove(item)
        self._item_list.removeWidget(item)
        item.deleteLater()
        self.changed.emit()

    def _update_item(self):
        self.changed.emit()

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


def _add_title(layout: QVBoxLayout, title: str):
    title_label = QLabel(title)
    title_label.setStyleSheet("font-size: 16px")
    layout.addWidget(title_label)
    layout.addSpacing(6)


def _add_header(layout: QVBoxLayout, setting: Setting):
    title_label = QLabel(setting.name)
    title_label.setStyleSheet("font-weight:bold")
    desc_label = QLabel(setting.desc)
    desc_label.setWordWrap(True)
    layout.addSpacing(6)
    layout.addWidget(title_label)
    layout.addWidget(desc_label)


class SettingsWriteGuard:
    """Avoid feedback loop when reading settings and updating the UI."""

    _locked = False

    def __enter__(self):
        self._locked = True

    def __exit__(self, *ignored):
        self._locked = False

    def __bool__(self):
        return self._locked


class ConnectionSettings(QWidget):
    def __init__(self):
        super().__init__()
        self._write_guard = SettingsWriteGuard()

        layout = QVBoxLayout()
        self.setLayout(layout)
        _add_title(layout, "Server Configuration")

        _add_header(layout, Settings._server_url)
        server_layout = QHBoxLayout()
        self._server_url = QLineEdit(self)
        self._server_url.textChanged.connect(self.write)
        server_layout.addWidget(self._server_url)
        self._connect_button = QPushButton("Connect", self)
        self._connect_button.clicked.connect(Connection.instance().connect)
        server_layout.addWidget(self._connect_button)
        layout.addLayout(server_layout)

        self._connection_status = QLabel(self)
        self._connection_status.setWordWrap(True)
        self._connection_status.setTextFormat(Qt.RichText)
        self._connection_status.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self._connection_status.setOpenExternalLinks(True)
        layout.addWidget(self._connection_status)

        layout.addStretch()

        Connection.instance().changed.connect(self._update_server_status)
        self._update_server_status()

    def read(self):
        with self._write_guard:
            self._server_url.setText(settings.server_url)

    def write(self, *ignored):
        if not self._write_guard:
            settings.server_url = self._server_url.text()
            settings.save()

    def _update_server_status(self):
        server = Connection.instance()
        self._connect_button.setEnabled(server.state != ConnectionState.connecting)
        if server.state == ConnectionState.connected:
            self._connection_status.setText("Connected")
            self._connection_status.setStyleSheet("color: #3b3; font-weight:bold")
        elif server.state == ConnectionState.connecting:
            self._connection_status.setText("Connecting")
            self._connection_status.setStyleSheet("color: #cc3; font-weight:bold")
        elif server.state == ConnectionState.disconnected:
            self._connection_status.setText("Disconnected")
            self._connection_status.setStyleSheet("color: #888; font-style:italic")
        elif server.state == ConnectionState.error:
            self._connection_status.setText(f"<b>Error</b>: {server.error}")
            self._connection_status.setStyleSheet("color: red;")
            if server.missing_resource is not None:
                self._handle_missing_resource(server.missing_resource)

    def _handle_missing_resource(self, resource: MissingResource):
        if resource.kind is ResourceKind.checkpoint:
            self._connection_status.setText(
                "<b>Error</b>: No checkpoints found!\nCheckpoints must be placed into"
                " <ComfyUI>/model/checkpoints.\n<a"
                " href='https://civitai.com/models'>Civitai.com</a> has a large collection"
                " of checkpoints available for download."
            )

        elif resource.kind is ResourceKind.controlnet:
            self._connection_status.setText(
                f"<b>Error</b>: Could not find ControlNet model {', '.join(resource.names)}. Make"
                " sure to download the model and place it in the <ComfyUI>/models/controlnet"
                " folder."
            )
        elif resource.kind is ResourceKind.clip_vision:
            self._connection_status.setText(
                f"<b>Error</b>: Could not find CLIPVision model {', '.join(resource.names)}. Make"
                " sure to download the model and place it in the <ComfyUI>/models/clip_vision"
                " folder."
            )
        elif resource.kind is ResourceKind.ip_adapter:
            self._connection_status.setText(
                f"<b>Error</b>: Could not find IPAdapter model {', '.join(resource.names)}. Make"
                " sure to download the model and place it in the"
                " <ComfyUI>/custom_nodes/IPAdapter-ComfyUI/models folder."
            )
        elif resource.kind is ResourceKind.node:
            self._connection_status.setText(
                "<b>Error</b>: The following ComfyUI custom nodes are missing:<ul>"
                + "\n".join(
                    (
                        f"<li>{p[:p.index('|')]} <a"
                        f" href='{p[p.index('|')+1 :]}'>{p[p.index('|')+1 :]}</a></li>"
                        for p in resource.names
                    )
                )
                + "</ul>Please install them, restart the server and try again."
            )


class DiffusionSettings(QWidget):
    def __init__(self):
        super().__init__()
        self._write_guard = SettingsWriteGuard()

        frame_layout = QVBoxLayout()
        self.setLayout(frame_layout)
        _add_title(frame_layout, "Image Diffusion Settings")

        inner = QWidget(self)
        layout = QVBoxLayout()
        inner.setLayout(layout)

        _add_header(layout, Settings._sd_checkpoint)
        self._sd_checkpoint = QComboBox(inner)
        self._sd_checkpoint.currentIndexChanged.connect(self.write)
        layout.addWidget(self._sd_checkpoint, alignment=Qt.AlignLeft)

        _add_header(layout, Settings._loras)
        self._loras = LoraList(inner)
        self._loras.changed.connect(self.write)
        layout.addWidget(self._loras)

        _add_header(layout, Settings._style_prompt)
        self._style_prompt = QLineEdit(inner)
        self._style_prompt.textChanged.connect(self.write)
        layout.addWidget(self._style_prompt)

        _add_header(layout, Settings._negative_prompt)
        self._negative_prompt = QLineEdit(inner)
        self._negative_prompt.textChanged.connect(self.write)
        layout.addWidget(self._negative_prompt)

        _add_header(layout, Settings._upscale_prompt)
        self._upscale_prompt = QLineEdit(inner)
        self._upscale_prompt.textChanged.connect(self.write)
        layout.addWidget(self._upscale_prompt)

        _add_header(layout, Settings._min_image_size)
        self._min_image_size = QSpinBox(inner)
        self._min_image_size.setMinimumWidth(100)
        self._min_image_size.setMaximum(1024)
        self._min_image_size.valueChanged.connect(self.write)
        layout.addWidget(self._min_image_size, alignment=Qt.AlignLeft)

        _add_header(layout, Settings._max_image_size)
        self._max_image_size = QSpinBox(inner)
        self._max_image_size.setMinimumWidth(100)
        self._max_image_size.setMaximum(2048)
        self._max_image_size.valueChanged.connect(self.write)
        layout.addWidget(self._max_image_size, alignment=Qt.AlignLeft)

        _add_header(layout, Settings._sampler)
        self._sampler = QComboBox(inner)
        self._sampler.addItem("DDIM")
        self._sampler.addItem("DPM++ 2M SDE")
        self._sampler.addItem("DPM++ 2M SDE Karras")
        self._sampler.currentIndexChanged.connect(self.write)
        layout.addWidget(self._sampler, alignment=Qt.AlignLeft)

        _add_header(layout, Settings._sampler_steps)
        self._sampler_steps = SliderWithValue(1, 100, inner)
        self._sampler_steps.value_changed.connect(self.write)
        layout.addWidget(self._sampler_steps)

        _add_header(layout, Settings._sampler_steps_upscaling)
        self._sampler_steps_upscaling = SliderWithValue(1, 100, inner)
        self._sampler_steps_upscaling.value_changed.connect(self.write)
        layout.addWidget(self._sampler_steps_upscaling)

        _add_header(layout, Settings._cfg_scale)
        self._cfg_scale = SliderWithValue(1, 20, inner)
        self._cfg_scale.value_changed.connect(self.write)
        layout.addWidget(self._cfg_scale)

        layout.addStretch()

        scroll = QScrollArea(self)
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        frame_layout.addWidget(scroll)

    def read(self):
        with self._write_guard:
            self._sd_checkpoint.clear()
            if Connection.instance().state == ConnectionState.connected:
                client = Connection.instance().client
                for checkpoint in client.checkpoints:
                    self._sd_checkpoint.addItem(checkpoint)
                self._loras.names = client.lora_models

            self._sd_checkpoint.setCurrentText(settings.sd_checkpoint)
            self._loras.value = settings.loras
            self._style_prompt.setText(settings.style_prompt)
            self._negative_prompt.setText(settings.negative_prompt)
            self._upscale_prompt.setText(settings.upscale_prompt)
            self._min_image_size.setValue(settings.min_image_size)
            self._max_image_size.setValue(settings.max_image_size)
            self._sampler.setCurrentText(settings.sampler)
            self._sampler_steps.value = settings.sampler_steps
            self._sampler_steps_upscaling.value = settings.sampler_steps_upscaling
            self._cfg_scale.value = settings.cfg_scale

    def write(self, *ignored):
        if not self._write_guard:
            settings.sd_checkpoint = self._sd_checkpoint.currentText()
            settings.loras = self._loras.value
            settings.style_prompt = self._style_prompt.text()
            settings.negative_prompt = self._negative_prompt.text()
            settings.upscale_prompt = self._upscale_prompt.text()
            settings.min_image_size = self._min_image_size.value()
            settings.max_image_size = self._max_image_size.value()
            settings.sampler = self._sampler.currentText()
            settings.sampler_steps = self._sampler_steps.value
            settings.sampler_steps_upscaling = self._sampler_steps_upscaling.value
            settings.cfg_scale = self._cfg_scale.value
            settings.save()


class PerformanceSettings(QWidget):
    def __init__(self):
        super().__init__()
        self._write_guard = SettingsWriteGuard()

        layout = QVBoxLayout()
        self.setLayout(layout)
        _add_title(layout, "Performance Settings")

        _add_header(layout, Settings._history_size)
        self._history_size = QSpinBox(self)
        self._history_size.setMinimum(8)
        self._history_size.setMaximum(1024 * 16)
        self._history_size.setSingleStep(100)
        self._history_size.setSuffix(" MB")
        self._history_size.valueChanged.connect(self.write)
        self._history_usage = QLabel(self)
        self._history_usage.setStyleSheet("font-style:italic; color: #3b3;")
        history_layout = QHBoxLayout()
        history_layout.addWidget(self._history_size)
        history_layout.addWidget(self._history_usage)
        layout.addLayout(history_layout)

        _add_header(layout, Settings._gpu_memory_preset)
        self._gpu_memory_preset = QComboBox(self)
        for preset in GPUMemoryPreset:
            self._gpu_memory_preset.addItem(preset.text)
        self._gpu_memory_preset.currentIndexChanged.connect(self._change_gpu_memory_preset)
        layout.addWidget(self._gpu_memory_preset, alignment=Qt.AlignLeft)

        self._advanced = QWidget(self)
        self._advanced.setEnabled(settings.gpu_memory_preset is GPUMemoryPreset.custom)
        self._advanced.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._advanced)
        advanced_layout = QVBoxLayout()
        self._advanced.setLayout(advanced_layout)

        _add_header(advanced_layout, Settings._batch_size)
        self._batch_size_slider = SliderWithValue(1, 16, self._advanced)
        self._batch_size_slider.value_changed.connect(self.write)
        advanced_layout.addWidget(self._batch_size_slider)

        _add_header(advanced_layout, Settings._diffusion_tile_size)
        self._diffusion_tile_size = QSpinBox(self._advanced)
        self._diffusion_tile_size.setMinimum(768)
        self._diffusion_tile_size.setMaximum(4096 * 2)
        self._diffusion_tile_size.setMinimumWidth(100)
        self._diffusion_tile_size.valueChanged.connect(self.write)
        advanced_layout.addWidget(self._diffusion_tile_size, alignment=Qt.AlignLeft)

        layout.addStretch()

    def _change_gpu_memory_preset(self, index):
        self.write()
        is_custom = settings.gpu_memory_preset is GPUMemoryPreset.custom
        self._advanced.setEnabled(is_custom)
        if not is_custom:
            self.read()

    def read(self):
        with self._write_guard:
            memory_usage = Model.active().jobs.memory_usage
            self._history_size.setValue(settings.history_size)
            self._history_usage.setText(f"Currently using {memory_usage:.1f} MB")
            self._batch_size_slider.value = settings.batch_size
            self._gpu_memory_preset.setCurrentIndex(settings.gpu_memory_preset.value)
            self._diffusion_tile_size.setValue(settings.diffusion_tile_size)

    def write(self, *ignored):
        if not self._write_guard:
            settings.history_size = self._history_size.value()
            settings.batch_size = self._batch_size_slider.value
            settings.diffusion_tile_size = self._diffusion_tile_size.value()
            settings.gpu_memory_preset = GPUMemoryPreset(self._gpu_memory_preset.currentIndex())
            settings.save()


class SettingsDialog(QDialog):
    _server: ConnectionSettings
    _diffusion: DiffusionSettings
    _performance: PerformanceSettings

    def __init__(self, main_window: QMainWindow):
        super().__init__()
        self.setMinimumSize(QSize(640, 480))
        self.setMaximumSize(QSize(800, 2048))
        self.resize(QSize(800, int(main_window.height() * 0.8)))
        self.setWindowTitle("Configure Image Diffusion")

        layout = QHBoxLayout()
        self.setLayout(layout)

        self._list = QListWidget(self)
        self._list.setFixedWidth(120)

        def create_list_item(text: str):
            item = QListWidgetItem(text, self._list)
            item.setSizeHint(QSize(112, 20))

        create_list_item("Connection")
        create_list_item("Diffusion")
        create_list_item("Performance")
        self._list.setCurrentRow(0)
        self._list.currentRowChanged.connect(self._change_page)
        layout.addWidget(self._list)

        inner = QVBoxLayout()
        layout.addLayout(inner)

        self._stack = QStackedWidget(self)
        self._connection = ConnectionSettings()
        self._diffusion = DiffusionSettings()
        self._performance = PerformanceSettings()
        self._stack.addWidget(self._connection)
        self._stack.addWidget(self._diffusion)
        self._stack.addWidget(self._performance)
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

    def read(self):
        self._connection.read()
        self._diffusion.read()
        self._performance.read()

    def restore_defaults(self):
        settings.restore()
        self.read()

    def show(self):
        self.read()
        super().show()
        self._close_button.setFocus()

    def _change_page(self, index):
        self._stack.setCurrentIndex(index)
