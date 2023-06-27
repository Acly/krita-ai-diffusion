import os
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QPushButton,
    QLabel,
    QSpinBox,
    QComboBox,
    QSlider,
    QWidget,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal


class Setting:
    def __init__(self, name: str, default, desc: str):
        self.name = name
        self.desc = desc
        self.default = default


class Settings:
    default_path = Path(__file__).parent / "settings.json"

    _min_image_size = Setting(
        "Minimum image size",
        512,
        (
            "Generation will run at a resolution of at least the configured value, "
            "even if the selected input image content is smaller. "
            "Results are automatically downscaled to fit the target area if needed."
        ),
    )

    _max_image_size = Setting(
        "Maximum image size",
        768,
        (
            "Initial image generation will run with a resolution no higher than the value "
            "configured here. If the resolution of the target area is higher, the results "
            "will be upscaled afterwards."
        ),
    )

    _batch_size = Setting(
        "Batch size",
        2,
        (
            "Number of low resolution images which are generated at once. Improves generation "
            "speed but requires more GPU memory (VRAM)."
        ),
    )

    _upscaler = Setting(
        "Upscaler",
        "Lanczos",
        ("The algorithm to use whenever images need to be resized to a higher resolution."),
    )
    upscalers = ["Lanczos"]

    @property
    def upscaler_index(self):
        return self.upscalers.index(self.upscaler)

    # Folder where intermediate images are stored for debug purposes (default: None)
    debug_image_folder = os.environ.get("KRITA_AI_TOOLS_DEBUG_IMAGE")

    def __init__(self):
        self.restore()

    def __getattr__(self, name: str):
        if name in self._values:
            return self._values[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        if name in self._values:
            self._values[name] = value
        else:
            object.__setattr__(self, name, value)

    def restore(self):
        self.__dict__["_values"] = {
            k[1:]: v.default for k, v in Settings.__dict__.items() if isinstance(v, Setting)
        }

    def save(self, path: Path = ...):
        path = self.default_path if path is ... else path
        with open(path, "w") as file:
            file.write(json.dumps(self._values))

    def load(self, path: Path = ...):
        path = self.default_path if path is ... else path
        if not path.exists():
            self.save()  # create new file with defaults
            return
        with open(path, "r") as file:
            contents = json.loads(file.read())
            for k, v in contents.items():
                setting = getattr(Settings, f"_{k}", None)
                if setting is not None:
                    if isinstance(setting.default, type(v)):
                        self._values[k] = v
                    else:
                        raise Exception(f"{v} is not a valid value for '{k}'")


settings = Settings()


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
            settings.batch_size = self._batch_size_slider.value
            settings.min_image_size = self._min_image_size.value()
            settings.max_image_size = self._max_image_size.value()
            settings.upscaler = self._upscaler.currentText()
            settings.save()

    def restore_defaults(self):
        settings.restore()
        self.read()

    def show(self):
        self.read()
        super().show()
        self._close_button.setFocus()
