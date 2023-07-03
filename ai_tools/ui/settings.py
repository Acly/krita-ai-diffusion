from pathlib import Path
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QPushButton,
    QLabel,
    QLineEdit,
    QSpinBox,
    QComboBox,
    QSlider,
    QWidget,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap

from .. import Auto1111, Setting, Settings, settings
from .server import DiffusionServer, ServerState

_icon_path = Path(__file__).parent.parent / "icons"


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


class SettingsDialog(QDialog):
    _write_on_update = True

    def __init__(self):
        super().__init__()
        self.setMinimumSize(QSize(640, 480))
        self.setMaximumSize(QSize(800, 2048))
        self.setWindowTitle("Configure AI Tools")

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self._add_header(Settings._server_url)
        self._connection_status = QLabel(self)
        self._connection_status.setWordWrap(True)
        self._layout.addWidget(self._connection_status)

        server_layout = QHBoxLayout()
        self._server_url = QLineEdit(self)
        self._server_url.textChanged.connect(self.write)
        server_layout.addWidget(self._server_url)
        self._connect_button = QPushButton("Connect", self)
        self._connect_button.clicked.connect(DiffusionServer.instance().connect)
        server_layout.addWidget(self._connect_button)
        self._layout.addLayout(server_layout)

        DiffusionServer.instance().changed.connect(self.update_server_status)
        self.update_server_status()

        self._add_header(Settings._batch_size)
        self._batch_size_slider = SliderWithValue(1, 16, self)
        self._batch_size_slider.value_changed.connect(self.write)
        self._layout.addWidget(self._batch_size_slider)

        self._add_header(Settings._upscaler)
        self._upscaler = QComboBox(self)
        self._upscaler.currentIndexChanged.connect(self.write)
        self._layout.addWidget(self._upscaler)

        self._add_header(Settings._min_image_size)
        self._min_image_size = QSpinBox(self)
        self._min_image_size.setMaximum(1024)
        self._min_image_size.valueChanged.connect(self.write)
        self._layout.addWidget(self._min_image_size)

        self._add_header(Settings._max_image_size)
        self._max_image_size = QSpinBox(self)
        self._max_image_size.setMaximum(2048)
        self._max_image_size.valueChanged.connect(self.write)
        self._layout.addWidget(self._max_image_size)

        self._layout.addStretch()
        self._layout.addSpacing(6)

        self._restore_button = QPushButton("Restore Defaults", self)
        self._restore_button.clicked.connect(self.restore_defaults)

        self._close_button = QPushButton("Ok", self)
        self._close_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._restore_button)
        button_layout.addStretch()
        button_layout.addWidget(self._close_button)
        self._layout.addLayout(button_layout)

    def _add_header(self, setting: Setting):
        title_label = QLabel(setting.name, self)
        title_label.setStyleSheet("font-weight:bold")
        desc_label = QLabel(setting.desc, self)
        desc_label.setWordWrap(True)
        self._layout.addWidget(title_label)
        self._layout.addWidget(desc_label)

    def read(self):
        self._write_on_update = False
        try:
            self._server_url.setText(settings.server_url)
            self._batch_size_slider.value = settings.batch_size
            self._min_image_size.setValue(settings.min_image_size)
            self._max_image_size.setValue(settings.max_image_size)
            self._upscaler.clear()
            for index, upscaler in enumerate(settings.upscalers):
                self._upscaler.insertItem(index, upscaler)
            try:
                self._upscaler.setCurrentIndex(settings.upscaler_index)
            except Exception as e:
                print("[krita-ai-tools] Can't find upscaler", settings.upscaler)
                self._upscaler.setCurrentIndex(0)
        finally:
            self._write_on_update = True

    def write(self, *ignored):
        if self._write_on_update:
            settings.server_url = self._server_url.text()
            settings.batch_size = self._batch_size_slider.value
            settings.min_image_size = self._min_image_size.value()
            settings.max_image_size = self._max_image_size.value()
            settings.upscaler = self._upscaler.currentText()
            settings.save()

    def restore_defaults(self):
        settings.restore()
        self.read()

    def update_server_status(self):
        server = DiffusionServer.instance()
        self._connect_button.setEnabled(server.state != ServerState.connecting)
        if server.state == ServerState.connected:
            self._connection_status.setText("Connected")
            self._connection_status.setStyleSheet("color: #3b3; font-weight:bold")
        elif server.state == ServerState.connecting:
            self._connection_status.setText("Connecting")
            self._connection_status.setStyleSheet("color: #cc3; font-weight:bold")
        elif server.state == ServerState.disconnected:
            self._connection_status.setText("Disconnected")
            self._connection_status.setStyleSheet("color: #888; font-style:italic")
        elif server.state == ServerState.error:
            self._connection_status.setText(f"<b>Error</b>: {server.error}")
            self._connection_status.setStyleSheet("color: red;")

    def show(self):
        self.read()
        super().show()
        self._close_button.setFocus()
