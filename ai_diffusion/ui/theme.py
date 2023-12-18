from __future__ import annotations
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtGui import QGuiApplication, QPalette, QIcon, QPixmap, QFontMetrics
from PyQt5.QtWidgets import QVBoxLayout, QLabel
from pathlib import Path

from ..settings import Setting, util
from ..style import SDVersion
from ..client import Client

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
    path = icon_path / f"{name}-{theme}.svg"
    if not path.exists():
        path = path.with_suffix(".png")
    if not path.exists():
        util.client_logger.error(f"Icon {name} not found for them {theme}")
        return QIcon()
    return QIcon(str(path))


def sd_version_icon(version: SDVersion, client: Client | None = None):
    if client and version not in client.supported_sd_versions:
        return icon("warning")
    elif version is SDVersion.sd15:
        return icon("sd-version-15")
    elif version is SDVersion.sdxl:
        return icon("sd-version-xl")
    else:
        util.client_logger.warning(f"Unresolved SD version {version}, cannot fetch icon")
        return icon("warning")


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
    elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, label.width() - 2)
    label.setText(elided)


class SignalBlocker:
    def __init__(self, obj: QObject):
        self._obj = obj

    def __enter__(self):
        self._obj.blockSignals(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._obj.blockSignals(False)
