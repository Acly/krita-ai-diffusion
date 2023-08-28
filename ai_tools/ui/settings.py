from pathlib import Path
from typing import Callable
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QSpinBox,
    QStackedWidget,
    QComboBox,
    QSlider,
    QWidget,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap

from .. import Client, Setting, Settings, settings, GPUMemoryPreset
from .connection import Connection, ConnectionState


class SliderWithValue(QWidget):
    value_changed = pyqtSignal(int)

    def __init__(self, minimum=0, maximum=100, parent=None):
        super().__init__(parent)
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
        self._label.setText(str(value))
        self.value_changed.emit(value)

    @property
    def value(self):
        return self._slider.value()

    @value.setter
    def value(self, v):
        self._slider.setValue(v)


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
        settings.save()

    def __bool__(self):
        return self._locked


class ConnectionSettings(QWidget):
    def __init__(self):
        super().__init__()
        self._write_guard = SettingsWriteGuard()

        layout = QVBoxLayout()
        self.setLayout(layout)
        _add_title(layout, "Server configuration")

        _add_header(layout, Settings._server_url)
        self._connection_status = QLabel(self)
        self._connection_status.setWordWrap(True)
        layout.addWidget(self._connection_status)

        server_layout = QHBoxLayout()
        self._server_url = QLineEdit(self)
        self._server_url.textChanged.connect(self.write)
        server_layout.addWidget(self._server_url)
        self._connect_button = QPushButton("Connect", self)
        self._connect_button.clicked.connect(Connection.instance().connect)
        server_layout.addWidget(self._connect_button)
        layout.addLayout(server_layout)

        layout.addStretch()

        Connection.instance().changed.connect(self._update_server_status)
        self._update_server_status()

    def read(self):
        with self._write_guard:
            self._server_url.setText(settings.server_url)

    def write(self, *ignored):
        if not self._write_guard:
            settings.server_url = self._server_url.text()

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


class DiffusionSettings(QWidget):
    def __init__(self):
        super().__init__()
        self._write_guard = SettingsWriteGuard()

        layout = QVBoxLayout()
        self.setLayout(layout)
        _add_title(layout, "Diffusion model settings")

        _add_header(layout, Settings._negative_prompt)
        self._negative_prompt = QLineEdit(self)
        self._negative_prompt.textChanged.connect(self.write)
        layout.addWidget(self._negative_prompt)

        _add_header(layout, Settings._upscale_prompt)
        self._upscale_prompt = QLineEdit(self)
        self._upscale_prompt.textChanged.connect(self.write)
        layout.addWidget(self._upscale_prompt)

        _add_header(layout, Settings._upscaler)
        self._upscaler = QComboBox(self)
        self._upscaler.currentIndexChanged.connect(self.write)
        layout.addWidget(self._upscaler)

        _add_header(layout, Settings._min_image_size)
        self._min_image_size = QSpinBox(self)
        self._min_image_size.setMaximum(1024)
        self._min_image_size.valueChanged.connect(self.write)
        layout.addWidget(self._min_image_size)

        _add_header(layout, Settings._max_image_size)
        self._max_image_size = QSpinBox(self)
        self._max_image_size.setMaximum(2048)
        self._max_image_size.valueChanged.connect(self.write)
        layout.addWidget(self._max_image_size)

        layout.addStretch()

    def read(self):
        with self._write_guard:
            self._negative_prompt.setText(settings.negative_prompt)
            self._upscale_prompt.setText(settings.upscale_prompt)
            self._min_image_size.setValue(settings.min_image_size)
            self._max_image_size.setValue(settings.max_image_size)
            self._upscaler.clear()
            for index, upscaler in enumerate(settings.upscalers):
                self._upscaler.insertItem(index, upscaler)
            try:
                self._upscaler.setCurrentIndex(settings.upscaler_index)
            except Exception as e:
                self._upscaler.setCurrentIndex(0)

    def write(self, *ignored):
        if not self._write_guard:
            settings.negative_Prompt = self._negative_prompt.text()
            settings.upscale_prompt = self._upscale_prompt.text()
            settings.min_image_size = self._min_image_size.value()
            settings.max_image_size = self._max_image_size.value()
            settings.upscaler = self._upscaler.currentText()


class PerformanceSettings(QWidget):
    def __init__(self):
        super().__init__()
        self._write_guard = SettingsWriteGuard()

        layout = QVBoxLayout()
        self.setLayout(layout)
        _add_title(layout, "Performance settings")

        _add_header(layout, Settings._gpu_memory_preset)
        self._gpu_memory_preset = QComboBox(self)
        for preset in GPUMemoryPreset:
            self._gpu_memory_preset.addItem(preset.text)
        self._gpu_memory_preset.currentIndexChanged.connect(self._change_gpu_memory_preset)
        layout.addWidget(self._gpu_memory_preset)

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

        _add_header(advanced_layout, Settings._vae_endoding_tile_size)
        self._vae_endoding_tile_size = QSpinBox(self._advanced)
        self._vae_endoding_tile_size.setMinimum(768)
        self._vae_endoding_tile_size.setMaximum(4096)
        self._vae_endoding_tile_size.valueChanged.connect(self.write)
        advanced_layout.addWidget(self._vae_endoding_tile_size)

        _add_header(advanced_layout, Settings._diffusion_tile_size)
        self._diffusion_tile_size = QSpinBox(self._advanced)
        self._diffusion_tile_size.setMinimum(768)
        self._diffusion_tile_size.setMaximum(4096 * 2)
        self._diffusion_tile_size.valueChanged.connect(self.write)
        advanced_layout.addWidget(self._diffusion_tile_size)

        layout.addStretch()

    def _change_gpu_memory_preset(self, index):
        self.write()
        is_custom = settings.gpu_memory_preset is GPUMemoryPreset.custom
        self._advanced.setEnabled(is_custom)
        if not is_custom:
            self.read()

    def read(self):
        with self._write_guard:
            self._batch_size_slider.value = settings.batch_size
            self._gpu_memory_preset.setCurrentIndex(settings.gpu_memory_preset.value)
            self._vae_endoding_tile_size.setValue(settings.vae_endoding_tile_size)
            self._diffusion_tile_size.setValue(settings.diffusion_tile_size)

    def write(self, *ignored):
        if not self._write_guard:
            settings.batch_size = self._batch_size_slider.value
            settings.vae_endoding_tile_size = self._vae_endoding_tile_size.value()
            settings.diffusion_tile_size = self._diffusion_tile_size.value()
            settings.gpu_memory_preset = GPUMemoryPreset(self._gpu_memory_preset.currentIndex())


class SettingsDialog(QDialog):
    _server: ConnectionSettings
    _diffusion: DiffusionSettings
    _performance: PerformanceSettings

    def __init__(self):
        super().__init__()
        self.setMinimumSize(QSize(640, 480))
        self.setMaximumSize(QSize(800, 2048))
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
