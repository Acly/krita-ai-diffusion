from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication, QPalette, QIcon, QPixmap, QFontMetrics
from PyQt5.QtWidgets import QVBoxLayout, QLabel
from pathlib import Path

from ..settings import Setting

is_dark = QGuiApplication.palette().color(QPalette.Window).lightness() < 128

green = "#3b3" if is_dark else "#292"
yellow = "#cc3" if is_dark else "#762"
red = "#c33"
grey = "#888" if is_dark else "#555"
highlight = "#8df" if is_dark else "#346"

background_inactive = "#606060"
background_active = QGuiApplication.palette().highlight().color().name()

icon_path = Path(__file__).parent.parent / "icons"


def icon(name: str):
    theme = "dark" if is_dark else "light"
    return QIcon(str(icon_path / f"{name}-{theme}.svg"))


def logo():
    return QPixmap(str(icon_path / "logo-128.png"))


def add_header(layout: QVBoxLayout, setting: Setting):
    title_label = QLabel(setting.name)
    title_label.setStyleSheet("font-weight:bold")
    desc_label = QLabel(setting.desc)
    desc_label.setWordWrap(True)
    layout.addSpacing(6)
    layout.addWidget(title_label)
    layout.addWidget(desc_label)


def set_text_clipped(label: QLabel, text: str):
    metrics = QFontMetrics(label.font())
    elided = metrics.elidedText(text, Qt.ElideRight, label.width() - 2)
    label.setText(elided)


class EventSuppression:
    def __init__(self):
        self.active = False

    def __enter__(self):
        self.active = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False

    def __bool__(self):
        return self.active
