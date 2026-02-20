from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QObject, QSize, Qt
from PyQt5.QtGui import QFontMetrics, QGuiApplication, QIcon, QPalette, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..client import Client
from ..files import FileFormat
from ..localization import translate as _
from ..platform_tools import is_windows
from ..settings import Setting
from ..style import Arch
from ..util import client_logger as log

_palette = QGuiApplication.palette()
is_dark = _palette.color(QPalette.ColorRole.Window).lightness() < 128

base = _palette.color(QPalette.ColorRole.Base).name()
green = "#30b030" if is_dark else "#209020"
yellow = "#c0c030" if is_dark else "#706020"
red = "#d07a40" if is_dark else "#c07630"
grey = "#888" if is_dark else "#606060"
highlight = "#8df" if is_dark else "#357"
progress_alt = "#a16207" if is_dark else "#ca8a04"
active = _palette.color(QPalette.ColorRole.Highlight).name()
line = _palette.color(QPalette.ColorRole.Background).darker(120).name()
line_base = _palette.color(QPalette.ColorRole.Base).darker(120).name()

flat_combo_stylesheet = f"""
    QComboBox {{ border: none; background-color: transparent; padding: 1px 12px 1px 2px; }}
    QComboBox QAbstractItemView {{ selection-color: {highlight}; }}
"""

copy_to_clipboard_string = _("Copy to clipboard")  # keeping translations for future use

icon_path = Path(__file__).parent.parent / "icons"


def icon(name: str):
    theme = "dark" if is_dark else "light"
    path = icon_path / f"{name}-{theme}.svg"
    if not path.exists():
        path = path.with_suffix(".png")
    if not path.exists():
        log.error(f"Icon {name} not found for them {theme}")
        return QIcon()
    return QIcon(str(path))


def checkpoint_icon(arch: Arch, format: FileFormat | None = None, client: Client | None = None):
    if client:
        if not client.supports_arch(arch):
            return icon("warning")
        if format is FileFormat.diffusion and not client.models.for_arch(arch).has_te_vae:
            return icon("warning")
    if arch is Arch.sd15:
        return icon("sd-version-15")
    elif arch is Arch.sdxl:
        return icon("sd-version-xl")
    elif arch is Arch.sd3:
        return icon("sd-version-3")
    elif arch is Arch.flux:
        return icon("sd-version-flux")
    elif arch is Arch.flux_k:
        return icon("sd-version-flux-k")
    elif arch.is_flux2:
        return icon("sd-version-flux-2")
    elif arch is Arch.illu:
        return icon("sd-version-illu")
    elif arch is Arch.illu_v:
        return icon("sd-version-illu-v")
    elif arch is Arch.chroma:
        return icon("sd-version-chroma")
    elif arch.is_qwen_like:
        return icon("sd-version-qwen")
    elif arch is Arch.zimage:
        return icon("sd-version-z-image")
    else:
        log.warning(f"Unresolved SD version {arch}, cannot fetch icon")
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


def set_text_clipped(label: QLabel, text: str, padding=4):
    metrics = QFontMetrics(label.font())
    elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, label.width() - padding)
    label.setText(elided)


def screen_scale(widget: QWidget, size: QSize):
    if is_windows:  # Not sure about other OS
        scale = widget.logicalDpiX() / 96.0
        return QSize(int(size.width() * scale), int(size.height() * scale))
    return size


class SignalBlocker:
    def __init__(self, obj: QObject):
        self._obj = obj

    def __enter__(self):
        self._obj.blockSignals(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._obj.blockSignals(False)
