from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path

from PyQt5.QtCore import QRect, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QGuiApplication, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..backend.lora_manager import (
    LoraInfo,
    arch_for_base_model,
    fetch_commercial_use,
    fetch_loras_pages,
    fetch_new_loras,
    fetch_preview_bytes,
    load_cached_loras,
    lora_preload_task,
    save_lora_cache,
    set_favorite,
)
from ..localization import translate as _
from ..model.root import root
from ..settings import settings
from ..text import format_option_group
from ..util import ensure
from ..wildcards import WildcardLibrary
from . import theme

_PREVIEW_SIZE_DEFAULT = 96
_PREVIEW_SIZE_MIN = 48
_PREVIEW_SIZE_MAX = 384
_TAG_ALL = "__all__"
_MAX_TAG_ENTRIES = 30
_TRIGGER_ALL = "__all_triggers__"
_FORMAT_RANDOM = "random"
_FORMAT_SEQUENTIAL = "sequential"
_FORMAT_SEPARATE = "separate"
_MULTI_TRIGGERS_NONE = "none"
_MULTI_TRIGGERS_FIRST = "first"
_MULTI_TRIGGERS_ALL = "all"
_ARCH_ANY = "__any__"
_SORT_NAME = "name"
_SORT_DATE = "date"
_POS_END = "end"
_POS_START = "start"
_POS_CURSOR = "cursor"
_NSFW_ALL = "all"
_NSFW_SAFE = "safe"
_NSFW_HIDE_EXPLICIT = "hide_explicit"
_KNOWN_ARCHES = [
    "sd15",
    "sdxl",
    "illu",
    "sd3",
    "flux",
    "flux_k",
    "chroma",
    "qwen",
    "anima",
    "zimage",
    "ernie",
    "krea2",
    "moss",
]
# full names for the base-model filter, matching the labels in the style editor
_ARCH_LABELS = {
    "sd15": "SD 1.5",
    "sdxl": "SD XL",
    "illu": "Illustrious",
    "sd3": "SD 3",
    "flux": "Flux",
    "flux_k": "Flux Kontext",
    "chroma": "Chroma",
    "qwen": "Qwen",
    "anima": "Anima",
    "zimage": "Z-Image",
    "ernie": "ERNIE Image",
    "krea2": "Krea 2",
    "moss": "MOSS-Image",
}
_VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov")


def _is_video_url(url: str) -> bool:
    # the real filename is in the `path` query param (e.g. /api/lm/previews?path=...mp4),
    # so check the whole url rather than stripping the query string
    lower = url.lower()
    return any(lower.endswith(ext) for ext in _VIDEO_EXTENSIONS)


_COMMERCIAL_COLORS = {"yes": QColor(60, 170, 75), "no": QColor(205, 60, 55)}


def _with_tag(pixmap: QPixmap, text: str) -> QPixmap:
    """Small label pill, bottom-left: the base model name."""
    if not text:
        return pixmap
    result = QPixmap(pixmap)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    font = painter.font()
    font.setBold(True)
    font.setPixelSize(max(9, result.width() // 12))
    painter.setFont(font)
    metrics = painter.fontMetrics()
    pad = 4
    text_w = metrics.horizontalAdvance(text)
    h = metrics.height() + 2
    w = min(result.width() - 4, text_w + pad * 2)
    x, y = 2, result.height() - h - 2
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(0, 0, 0, 165))
    painter.drawRoundedRect(x, y, w, h, 3, 3)
    painter.setPen(QColor(255, 255, 255))
    elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, w - pad * 2)
    painter.drawText(QRect(x, y, w, h), Qt.AlignmentFlag.AlignCenter, elided)
    painter.end()
    return result


def _base_model_label(base_model: str) -> str:
    if not base_model:
        return ""
    arch = arch_for_base_model(base_model)
    return _ARCH_LABELS.get(arch, base_model)


def _with_badges(pixmap: QPixmap, favorite: bool, commercial: str, base_model: str = "") -> QPixmap:
    """Overlay a favorite star (top-right), a commercial-use $ square (bottom-right):
    green = commercial image use allowed, red = not allowed, grey = unknown, and a
    base-model name tag (bottom-left)."""
    result = QPixmap(pixmap)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    w, h = result.width(), result.height()
    b = max(12, w // 6)

    # commercial-use square, bottom-right
    x, y = w - b - 2, h - b - 2
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_COMMERCIAL_COLORS.get(commercial, QColor(110, 110, 110)))
    painter.drawRoundedRect(x, y, b, b, 3, 3)
    painter.setPen(QColor(255, 255, 255))
    font = painter.font()
    font.setBold(True)
    font.setPixelSize(max(8, int(b * 0.75)))
    painter.setFont(font)
    painter.drawText(QRect(x, y, b, b), Qt.AlignmentFlag.AlignCenter, "$")

    # favorite star, top-right
    if favorite:
        sb = max(14, w // 5)
        star_font = painter.font()
        star_font.setPixelSize(sb)
        painter.setFont(star_font)
        rect = QRect(w - sb - 2, 0, sb, sb + 2)
        painter.setPen(QColor(0, 0, 0, 160))  # subtle outline for contrast
        painter.drawText(rect.translated(1, 1), Qt.AlignmentFlag.AlignCenter, "★")
        painter.setPen(QColor(240, 200, 60))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "★")

    painter.end()
    return _with_tag(result, base_model)


def _visible_range(grid, overscan: int = 24) -> range:
    """Indices of items in/near the viewport. Lazy preview loaders use this so they
    don't call visualItemRect on every item (thousands) on each scroll tick."""
    count = grid.count()
    if count == 0:
        return range(0)
    vp = ensure(grid.viewport()).rect()
    first = grid.indexAt(vp.topLeft())
    last = grid.indexAt(vp.bottomRight())
    start = first.row() if first.isValid() else 0
    end = last.row() if last.isValid() else start + 200  # empty space below last row
    return range(max(0, start - overscan), min(count, end + overscan + 1))


_ffmpeg_path = shutil.which("ffmpeg")


def _extract_video_frame(data: bytes) -> bytes | None:
    """Extract the first frame of a video as JPEG using ffmpeg. Returns None if
    ffmpeg is not installed or extraction fails. Runs blocking - call in executor."""
    if _ffmpeg_path is None:
        return None
    src = out = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(data)
            src = Path(f.name)
        out = src.with_suffix(".jpg")
        subprocess.run(
            [_ffmpeg_path, "-y", "-i", str(src), "-frames:v", "1", "-f", "image2", str(out)],
            capture_output=True,
            check=False,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # Windows only
        )
        if out.exists() and out.stat().st_size > 0:
            return out.read_bytes()
        return None
    except Exception:
        return None
    finally:
        for p in (src, out):
            if p is not None:
                p.unlink(missing_ok=True)


class LoraBrowser(QWidget):
    """The local LoRA library - one tab of LoraPickerDialog."""

    lora_selected = pyqtSignal(str, float)  # name, strength

    def __init__(self, current_arch: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._current_arch = current_arch
        self._all_loras: list[LoraInfo] = []
        self._filtered: list[LoraInfo] = []
        self._preview_cache: dict[str, QPixmap] = {}  # original, unscaled
        self._pending_previews: set[str] = set()
        self._pending_commercial: set[str] = set()
        self._loading = False
        self._preview_size = settings.lora_browser_size

        # ── row 1: search ──
        self._search = QLineEdit(self)
        self._search.setPlaceholderText(_("Search LoRAs…"))
        self._search.textChanged.connect(self._apply_filter)

        self._refresh_btn = QToolButton(self)
        self._refresh_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._refresh_btn.setIcon(theme.icon("reset"))
        self._refresh_btn.setText(_("Reload list"))
        self._refresh_btn.setToolTip(_("Reload the list from Lora Manager (fast)"))
        self._refresh_btn.clicked.connect(self._force_reload)

        self._rescan_btn = QToolButton(self)
        self._rescan_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._rescan_btn.setIcon(theme.icon("comfyui"))
        self._rescan_btn.setText(_("Scan server"))
        self._rescan_btn.setToolTip(_("Look for new LoRA files on the server (slow, full rescan)"))
        self._rescan_btn.clicked.connect(self._rescan_server)

        row1 = QHBoxLayout()
        row1.addWidget(self._search, 1)
        row1.addWidget(self._refresh_btn)
        row1.addWidget(self._rescan_btn)

        # ── row 2: base model + tag dropdowns ──
        arch_label = QLabel(_("Base Model:"), self)
        self._arch_combo = QComboBox(self)
        self._arch_combo.addItem(_("Any"), _ARCH_ANY)
        for arch in sorted(_KNOWN_ARCHES, key=lambda a: _ARCH_LABELS.get(a, a)):
            self._arch_combo.addItem(_ARCH_LABELS.get(arch, arch), arch)
        idx = self._arch_combo.findData(current_arch)
        self._arch_combo.setCurrentIndex(max(idx, 0))
        self._arch_combo.currentIndexChanged.connect(self._apply_filter)

        tag_label = QLabel(_("Tag:"), self)
        self._tag_combo = QComboBox(self)
        self._tag_combo.addItem(_("All"), _TAG_ALL)
        self._tag_combo.currentIndexChanged.connect(self._apply_filter)

        self._favorites_only = QCheckBox(_("Favorites"), self)
        self._favorites_only.toggled.connect(self._apply_filter)

        self._nsfw_combo = QComboBox(self)
        self._nsfw_combo.addItem(_("All"), _NSFW_ALL)
        self._nsfw_combo.addItem(_("Safe Only"), _NSFW_SAFE)
        self._nsfw_combo.addItem(_("Hide Explicit"), _NSFW_HIDE_EXPLICIT)
        self._nsfw_combo.setToolTip(
            _(
                "Filter by the CivitAI content rating of the preview image.\n"
                "Safe Only: hides R and above. Hide Explicit: hides X/XXX only."
            )
        )
        self._nsfw_combo.currentIndexChanged.connect(self._apply_filter)

        sort_label = QLabel(_("Sort:"), self)
        self._sort_combo = QComboBox(self)
        self._sort_combo.addItem(_("Name"), _SORT_NAME)
        self._sort_combo.addItem(_("Date Added"), _SORT_DATE)
        idx = self._sort_combo.findData(settings.lora_browser_sort)
        self._sort_combo.setCurrentIndex(max(idx, 0))
        self._sort_combo.currentIndexChanged.connect(self._apply_filter)
        self._sort_combo.currentIndexChanged.connect(self._save_sort_setting)

        size_label = QLabel(_("Size:"), self)
        self._size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._size_slider.setMinimum(_PREVIEW_SIZE_MIN)
        self._size_slider.setMaximum(_PREVIEW_SIZE_MAX)
        self._size_slider.setValue(self._preview_size)
        self._size_slider.setFixedWidth(90)
        self._size_slider.valueChanged.connect(self._on_preview_size_changed)
        self._size_slider.sliderReleased.connect(self._save_size_setting)

        row2 = QHBoxLayout()
        row2.addWidget(arch_label)
        row2.addWidget(self._arch_combo, 1)
        row2.addWidget(tag_label)
        row2.addWidget(self._tag_combo, 1)
        row2.addWidget(self._favorites_only)
        row2.addWidget(self._nsfw_combo)
        row2.addWidget(sort_label)
        row2.addWidget(self._sort_combo)
        row2.addWidget(size_label)
        row2.addWidget(self._size_slider)

        # ── grid ──
        self._grid = QListWidget(self)
        self._grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(QSize(self._preview_size, self._preview_size))
        self._grid.setGridSize(QSize(self._preview_size + 16, self._preview_size + 40))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setMovement(QListWidget.Movement.Static)
        self._grid.setWordWrap(True)
        self._grid.setSpacing(4)
        self._grid.itemSelectionChanged.connect(self._on_selection_changed)
        self._grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._grid.customContextMenuRequested.connect(self._show_context_menu)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._load_visible_previews)
        ensure(self._grid.verticalScrollBar()).valueChanged.connect(self._schedule_visible_previews)

        # ── bottom bar ──
        self._selected_label = QLabel(_("No LoRA selected"), self)
        self._selected_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        strength_label = QLabel(_("Strength:"), self)
        self._strength = QDoubleSpinBox(self)
        self._strength.setMinimum(0.0)
        self._strength.setMaximum(2.0)
        self._strength.setSingleStep(0.05)
        self._strength.setValue(1.0)
        self._strength.setDecimals(2)
        self._strength.setFixedWidth(72)

        self._include_triggers = QCheckBox(_("+ triggers:"), self)
        self._include_triggers.setChecked(True)
        self._include_triggers.setToolTip(
            _("Also insert the selected trigger word group into the prompt")
        )
        self._include_triggers.toggled.connect(self._update_trigger_combo_enabled)

        self._trigger_combo = QComboBox(self)
        self._trigger_combo.setMinimumWidth(160)
        self._trigger_combo.setToolTip(
            _("CivitAI lists alternative trigger phrases - pick which one to insert")
        )

        # multi-select controls (shown instead of the above when 2+ LoRAs are selected)
        self._format_combo = QComboBox(self)
        self._format_combo.addItem(_("Random {a|b}"), _FORMAT_RANDOM)
        self._format_combo.addItem(_("Sequential [[a|b]]"), _FORMAT_SEQUENTIAL)
        self._format_combo.addItem(_("Separate (all)"), _FORMAT_SEPARATE)
        self._format_combo.setToolTip(
            _(
                "Random: one is picked per generation. Sequential: cycles through in batch order."
                " Separate: adds all LoRAs together, no wildcard."
            )
        )

        self._multi_trigger_mode = QComboBox(self)
        self._multi_trigger_mode.addItem(_("No trigger words"), _MULTI_TRIGGERS_NONE)
        self._multi_trigger_mode.addItem(_("First trigger group"), _MULTI_TRIGGERS_FIRST)
        self._multi_trigger_mode.addItem(_("All trigger groups"), _MULTI_TRIGGERS_ALL)
        self._multi_trigger_mode.setCurrentIndex(1)
        self._format_combo.setVisible(False)
        self._multi_trigger_mode.setVisible(False)

        self._position_combo = QComboBox(self)
        self._position_combo.addItem(_("at End"), _POS_END)
        self._position_combo.addItem(_("at Start"), _POS_START)
        self._position_combo.addItem(_("at Cursor"), _POS_CURSOR)
        self._position_combo.setToolTip(_("Where to insert the LoRA in the prompt"))

        self._add_btn = QPushButton(_("Add to Prompt"), self)
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._add_to_prompt)

        self._copy_btn = QPushButton(_("Copy"), self)
        self._copy_btn.setEnabled(False)
        self._copy_btn.setToolTip(
            _("Copy the tags to the clipboard instead of adding to the prompt")
        )
        self._copy_btn.clicked.connect(self._copy_to_clipboard)

        self._save_wildcard_btn = QPushButton(_("Save as Wildcard…"), self)
        self._save_wildcard_btn.setToolTip(
            _("Save the selected LoRAs (with the trigger words chosen above) as a wildcard file")
        )
        self._save_wildcard_btn.setVisible(False)
        self._save_wildcard_btn.clicked.connect(self._save_selection_as_wildcard)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self._selected_label, 1)
        bottom_layout.addWidget(strength_label)
        bottom_layout.addWidget(self._strength)
        bottom_layout.addWidget(self._include_triggers)
        bottom_layout.addWidget(self._trigger_combo)
        bottom_layout.addWidget(self._format_combo)
        bottom_layout.addWidget(self._multi_trigger_mode)
        bottom_layout.addWidget(self._position_combo)
        bottom_layout.addWidget(self._add_btn)
        bottom_layout.addWidget(self._copy_btn)
        bottom_layout.addWidget(self._save_wildcard_btn)

        # ── status ──
        self._status = QLabel(_("Loading…"), self)
        self._status.setStyleSheet(f"color: {theme.grey}; font-style: italic;")

        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self._grid, 1)
        layout.addWidget(self._status)
        layout.addLayout(bottom_layout)
        self.setLayout(layout)

        self._load_loras()

    # ── data loading ──

    def _load_loras(self, force_refresh: bool = False):
        if self._loading:
            return
        client = root.connection.client_if_connected
        if client is None:
            self._status.setText(_("Not connected to ComfyUI"))
            return

        if not force_refresh:
            if (preload := lora_preload_task(client.url)) is not None:
                self._status.setText(_("Loading…"))
                self._loading = True
                eventloop.run(self._await_preload(preload))
                return
            cached = load_cached_loras(client.url)
            if cached:
                self._all_loras = cached
                self._rebuild_filters()
                self._apply_filter()
                self._status.setText(f"{len(cached)} {_('LoRAs (cached)')}")
                return

        self._status.setText(_("Loading…"))
        self._grid.clear()
        self._all_loras = []
        self._loading = True
        eventloop.run(self._fetch_progressive(client))

    async def _await_preload(self, preload):
        # the startup preload is already fetching - wait for it instead of fetching twice
        try:
            await preload
        finally:
            self._loading = False
        self._load_loras()

    def _force_reload(self):
        self._load_loras(force_refresh=True)

    def _rescan_server(self):
        # Look for LoRA files added since the list was loaded. ComfyUI only re-reads
        # its file lists (no model inspection - LoRAs don't need it), and Lora Manager
        # is asked for its newest entries until nothing new turns up, instead of
        # reloading every LoRA. "Reload list" stays the full reload.
        if root.connection.client_if_connected is None:
            self._status.setText(_("Not connected to ComfyUI"))
            return
        self._rescan_btn.setEnabled(False)
        self._status.setText(_("Scanning server for new LoRA files…"))
        # connection emits models_changed when the async refresh finishes
        root.connection.models_changed.connect(self._on_server_scanned)
        root.connection.refresh(quick=True)

    def _on_server_scanned(self):
        try:
            root.connection.models_changed.disconnect(self._on_server_scanned)
        except (TypeError, RuntimeError):
            pass
        client = root.connection.client_if_connected
        if client is None or self._loading:  # a full load already brings in new files
            self._rescan_btn.setEnabled(True)
            return
        if not self._all_loras:
            self._rescan_btn.setEnabled(True)
            self._load_loras(force_refresh=True)
            return
        eventloop.run(self._merge_new_loras(client))

    async def _merge_new_loras(self, client):
        try:
            new = await fetch_new_loras(client._requests, client.url, self._all_loras)
        finally:
            self._rescan_btn.setEnabled(True)
        if new is None:  # files were removed or the answer was unexpected
            self._load_loras(force_refresh=True)
            return
        if new:
            self._all_loras = new + self._all_loras
            save_lora_cache(client.url, self._all_loras)
            self._rebuild_filters()
            self._apply_filter()
        self._status.setText(
            _("{n} new LoRAs").format(n=len(new))
            + f" - {len(self._filtered)} / {len(self._all_loras)} LoRAs"
        )

    def shutdown(self):
        # drop the pending scan callback so it doesn't fire on a destroyed browser.
        # Not tied to hiding: switching to another tab while a scan runs must not
        # lose the callback that re-enables the button when it finishes.
        try:
            root.connection.models_changed.disconnect(self._on_server_scanned)
        except (TypeError, RuntimeError):
            pass

    async def _fetch_progressive(self, client):
        try:
            async for batch in fetch_loras_pages(client._requests, client.url):
                self._all_loras.extend(batch)
                self._rebuild_filters()
                self._apply_filter()
                self._status.setText(f"{_('Loading…')} ({len(self._all_loras)})")
        finally:
            self._loading = False
            if self._all_loras:
                save_lora_cache(client.url, self._all_loras)
                self._status.setText(f"{len(self._filtered)} / {len(self._all_loras)} LoRAs")
            else:
                self._status.setText(_("LoRA Manager not installed or no LoRAs found"))

    # ── filtering ──

    def _rebuild_filters(self):
        counts: dict[str, int] = {}
        for lora in self._all_loras:
            for tag in lora.tags:
                counts[tag] = counts.get(tag, 0) + 1
        # keep the most common tags (relevance cap), then list them alphabetically
        top_tags = sorted(counts, key=lambda t: -counts[t])[:_MAX_TAG_ENTRIES]
        top_tags.sort(key=str.lower)

        current = self._tag_combo.currentData()
        self._tag_combo.blockSignals(True)
        self._tag_combo.clear()
        self._tag_combo.addItem(_("All"), _TAG_ALL)
        for tag in top_tags:
            self._tag_combo.addItem(f"{tag} ({counts[tag]})", tag)
        idx = self._tag_combo.findData(current)
        self._tag_combo.setCurrentIndex(max(idx, 0))
        self._tag_combo.blockSignals(False)

    def _apply_filter(self):
        search = self._search.text().lower()
        arch = self._arch_combo.currentData()
        arch = "" if arch == _ARCH_ANY else (arch or "")
        active_tag = self._tag_combo.currentData()
        favorites_only = self._favorites_only.isChecked()
        nsfw_mode = self._nsfw_combo.currentData()

        def matches(lora: LoraInfo) -> bool:
            if favorites_only and not lora.favorite:
                return False
            if nsfw_mode == _NSFW_SAFE and lora.nsfw_level >= 8:  # R and above
                return False
            if nsfw_mode == _NSFW_HIDE_EXPLICIT and lora.nsfw_level >= 16:  # X/XXX
                return False
            if arch:
                if lora.base_model:
                    # has base_model info: must match, even if we don't recognize
                    # the string (otherwise unmapped models like "NoobAI" would
                    # bypass every arch filter)
                    if arch_for_base_model(lora.base_model) != arch:
                        return False
                # else: no base_model info at all -> can't tell, show anyway
            if active_tag and active_tag != _TAG_ALL:
                if active_tag not in lora.tags:
                    return False
            if search:
                haystack = (lora.name + " " + lora.display_name + " " + " ".join(lora.tags)).lower()
                if search not in haystack:
                    return False
            return True

        self._filtered = [l for l in self._all_loras if matches(l)]
        if self._sort_combo.currentData() == _SORT_DATE:
            self._filtered.sort(key=lambda l: l.modified, reverse=True)
        else:
            self._filtered.sort(key=lambda l: (l.display_name or l.name).lower())
        self._populate_grid()

    def _lora_key(self, lora: LoraInfo) -> str:
        return lora.sha256 or lora.name

    def _populate_grid(self):
        # the list is rebuilt as more LoRAs stream in (or filters/sort change) -
        # remember what was selected so it doesn't vanish out from under the user
        selected_keys = {
            self._lora_key(i.data(Qt.ItemDataRole.UserRole)) for i in self._grid.selectedItems()
        }
        self._grid.clear()
        cell_size = self._grid.gridSize()
        to_reselect: list[QListWidgetItem] = []
        for lora in self._filtered:
            item = QListWidgetItem(lora.display_name or lora.name)
            item.setData(Qt.ItemDataRole.UserRole, lora)
            # Force the item's clickable rect to fill the whole grid cell.
            # Without this, Qt sizes it to the wrapped text content, so
            # short single-line names get a tiny hit area while longer
            # wrapped names happen to fill the cell - ctrl-click then only
            # seems to "work" on multi-line items.
            item.setSizeHint(cell_size)
            fav = "★ " if lora.favorite else ""
            item.setToolTip(
                f"{fav}{lora.display_name}\nFile: {lora.name}\nBase: {lora.base_model or '?'}\n"
                + (f"Triggers: {' | '.join(lora.trigger_words)}" if lora.trigger_words else "")
            )
            self._set_tile_icon(item, lora)
            self._grid.addItem(item)
            if self._lora_key(lora) in selected_keys:
                to_reselect.append(item)
        if to_reselect:
            with theme.SignalBlocker(self._grid):
                for item in to_reselect:
                    item.setSelected(True)
            self._on_selection_changed()  # signals were blocked - refresh the bottom bar once
        if not self._loading:
            self._status.setText(f"{len(self._filtered)} / {len(self._all_loras)} LoRAs")
        self._schedule_visible_previews()

    def _tile_base(self, lora: LoraInfo) -> QPixmap:
        size = self._preview_size
        if lora.sha256 in self._preview_cache:
            return self._preview_cache[lora.sha256].scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        if lora.preview_url and _is_video_url(lora.preview_url):
            return theme.icon("play").pixmap(size, size)
        blank = QPixmap(size, size)
        blank.fill(Qt.GlobalColor.transparent)
        return blank

    def _set_tile_icon(self, item: QListWidgetItem, lora: LoraInfo):
        # preview (or placeholder) with favorite star + commercial-use badge on top
        item.setIcon(
            QIcon(
                _with_badges(
                    self._tile_base(lora),
                    lora.favorite,
                    lora.commercial,
                    _base_model_label(lora.base_model),
                )
            )
        )

    def _on_preview_size_changed(self, value: int):
        self._preview_size = value
        self._grid.setIconSize(QSize(value, value))
        self._grid.setGridSize(QSize(value + 16, value + 40))
        for i in range(self._grid.count()):
            item = ensure(self._grid.item(i))
            lora: LoraInfo = item.data(Qt.ItemDataRole.UserRole)
            item.setSizeHint(self._grid.gridSize())
            self._set_tile_icon(item, lora)
        self._schedule_visible_previews()

    def _save_size_setting(self):
        settings.lora_browser_size = self._preview_size
        settings.save()

    def _save_sort_setting(self):
        settings.lora_browser_sort = self._sort_combo.currentData()
        settings.save()

    # ── lazy preview loading (only visible items) ──

    def _schedule_visible_previews(self):
        self._preview_timer.start()

    def _load_visible_previews(self):
        viewport_rect = ensure(self._grid.viewport()).rect()
        client = root.connection.client_if_connected
        if client is None:
            return
        # only scan around the viewport - iterating all items (up to thousands of
        # LoRAs) with visualItemRect on every scroll tick is what made it lag
        for i in _visible_range(self._grid):
            item = ensure(self._grid.item(i))
            rect = self._grid.visualItemRect(item)
            if not rect.intersects(viewport_rect):
                continue
            lora: LoraInfo = item.data(Qt.ItemDataRole.UserRole)
            # commercial-use badge (lazy, per-model metadata call)
            if (
                lora.commercial == ""
                and lora.file_path
                and lora.sha256 not in self._pending_commercial
            ):
                self._pending_commercial.add(lora.sha256)
                eventloop.run(self._load_commercial(lora, item))
            if not lora.preview_url or lora.sha256 in self._preview_cache:
                continue
            if _is_video_url(lora.preview_url) and _ffmpeg_path is None:
                continue  # no decoder available - placeholder set in _populate_grid
            if lora.sha256 in self._pending_previews:
                continue
            self._pending_previews.add(lora.sha256)
            eventloop.run(self._load_preview(lora, item))

    async def _load_commercial(self, lora: LoraInfo, item: QListWidgetItem):
        client = root.connection.client_if_connected
        if client is None:
            return
        result = await fetch_commercial_use(client._requests, client.url, lora.file_path)
        lora.commercial = result or "unknown"  # "unknown" = fetched but no info (avoid refetch)
        try:
            self._set_tile_icon(item, lora)
        except RuntimeError:
            pass  # item removed before the fetch finished

    async def _load_preview(self, lora: LoraInfo, item: QListWidgetItem):
        client = root.connection.client_if_connected
        if client is None:
            return
        data = await fetch_preview_bytes(client._requests, lora.preview_url)
        is_video = _is_video_url(lora.preview_url)
        if data and is_video:
            # extract first frame via ffmpeg (off the UI thread)
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _extract_video_frame, data)
        if data:
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            if not pixmap.isNull():
                if is_video:  # small play badge so animated previews are recognizable
                    badge = theme.icon("play").pixmap(24, 24)
                    painter = QPainter(pixmap)
                    painter.drawPixmap(pixmap.width() - 28, pixmap.height() - 28, badge)
                    painter.end()
                self._preview_cache[lora.sha256] = pixmap
                self._set_tile_icon(item, lora)
            else:
                # format not decodable by Qt (e.g. some animated webp builds) - placeholder
                item.setIcon(theme.icon("filter"))

    # ── context menu / favorites ──

    def _show_context_menu(self, pos):
        item = self._grid.itemAt(pos)
        if item is None:
            return
        lora: LoraInfo = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        label = _("Remove from Favorites") if lora.favorite else _("Add to Favorites")
        menu.addAction(label, lambda: self._toggle_favorite(lora, item))
        if lora.civitai_model_id:
            menu.addAction(_("Open on CivitAI"), lambda: self._open_civitai(lora))
        menu.exec(self._grid.mapToGlobal(pos))

    def _save_selection_as_wildcard(self):
        items = self._grid.selectedItems()
        entries = self._multi_lora_entries(items)
        if not entries:
            return
        name, ok = QInputDialog.getText(
            self, _("Save as Wildcard"), _("Wildcard file name (may include a folder/ prefix):")
        )
        if not ok or not name.strip():
            return
        if WildcardLibrary.instance().create(name, entries):
            key = name.strip().strip("/\\").lower()
            QMessageBox.information(self, _("Saved"), f"__{key}__ " + _("created."))
        else:
            QMessageBox.warning(
                self,
                _("Save Failed"),
                _("Could not create the wildcard file - a file with that name may already exist."),
            )

    def _open_civitai(self, lora: LoraInfo):
        QDesktopServices.openUrl(QUrl(f"https://civitai.com/models/{lora.civitai_model_id}"))

    def _toggle_favorite(self, lora: LoraInfo, item: QListWidgetItem):
        client = root.connection.client_if_connected
        if client is None:
            return

        async def _apply():
            new_value = not lora.favorite
            if await set_favorite(client._requests, client.url, lora.file_path, new_value):
                lora.favorite = new_value
                save_lora_cache(client.url, self._all_loras)
                self._apply_filter()  # refresh tooltip/star and the favorites filter

        eventloop.run(_apply())

    # ── selection / insertion ──

    def _on_selection_changed(self):
        items = self._grid.selectedItems()
        is_multi = len(items) > 1

        self._include_triggers.setVisible(not is_multi)
        self._trigger_combo.setVisible(not is_multi)
        self._format_combo.setVisible(is_multi)
        self._multi_trigger_mode.setVisible(is_multi)
        self._save_wildcard_btn.setVisible(is_multi)

        if is_multi:
            names = ", ".join(i.data(Qt.ItemDataRole.UserRole).display_name for i in items[:3])
            more = f" +{len(items) - 3}" if len(items) > 3 else ""
            self._selected_label.setText(f"{len(items)} {_('selected')}: {names}{more}")
            self._add_btn.setEnabled(True)
        elif items:
            lora: LoraInfo = items[0].data(Qt.ItemDataRole.UserRole)
            fav = "★ " if lora.favorite else ""
            self._selected_label.setText(f"{fav}{lora.display_name}  [{lora.base_model or '?'}]")
            self._add_btn.setEnabled(True)
            self._trigger_combo.clear()
            if lora.trigger_words:
                self._trigger_combo.addItem(_("All"), _TRIGGER_ALL)
                for group in lora.trigger_words:
                    self._trigger_combo.addItem(group, group)
                self._trigger_combo.setCurrentIndex(0)
            self._update_trigger_combo_enabled()
        else:
            self._selected_label.setText(_("No LoRA selected"))
            self._add_btn.setEnabled(False)
            self._trigger_combo.clear()
        self._copy_btn.setEnabled(len(items) > 0)

    def _update_trigger_combo_enabled(self):
        enabled = self._include_triggers.isChecked() and self._trigger_combo.count() > 0
        self._trigger_combo.setEnabled(enabled)

    def _build_addition(self) -> str:
        """Build the text (lora tags + triggers) for the current selection, in the
        chosen format. Shared by 'Add to Prompt' and 'Copy'."""
        items = self._grid.selectedItems()
        if not items:
            return ""
        if len(items) > 1:
            return self._build_multi_lora_block(items)
        lora: LoraInfo = items[0].data(Qt.ItemDataRole.UserRole)
        strength = self._strength.value()
        parts = [f"<lora:{lora.name}:{strength:.2f}>"]
        if self._include_triggers.isChecked() and self._trigger_combo.currentData():
            selected = self._trigger_combo.currentData()
            if selected == _TRIGGER_ALL:
                parts.append("\n----\n".join(lora.trigger_words))
            else:
                parts.append(selected)
        return " ".join(parts)

    def _add_to_prompt(self):
        items = self._grid.selectedItems()
        model = root.active_model
        if not items or model is None:
            return
        addition = self._build_addition()
        if not addition:
            return
        if len(items) > 1:
            self.lora_selected.emit("", self._strength.value())
        else:
            lora: LoraInfo = items[0].data(Qt.ItemDataRole.UserRole)
            self.lora_selected.emit(lora.name, self._strength.value())

        position = self._position_combo.currentData()
        # insert at the prompt widget's cursor if possible, else fall back to end
        if position == _POS_CURSOR and self._insert_at_cursor(addition):
            return

        region = model.regions.active_or_root
        current = region.positive
        if position == _POS_START:
            region.positive = addition + "\n" + current.lstrip("\n")
        else:  # end (also the cursor fallback)
            region.positive = current.rstrip("\n") + "\n" + addition

    def _insert_at_cursor(self, addition: str) -> bool:
        # the prompt widget that owns the positive field is an ancestor - walk up
        # rather than checking the direct parent, which is the dialog holding the tabs
        widget = None
        node = self.parent()
        while node is not None and widget is None:
            widget = getattr(node, "positive", None)
            node = node.parent()
        if widget is None or not hasattr(widget, "textCursor"):
            return False
        cursor = widget.textCursor()
        cursor.insertText(addition)
        widget.setTextCursor(cursor)
        return True

    def _copy_to_clipboard(self):
        addition = self._build_addition()
        if not addition:
            return
        if clipboard := QGuiApplication.clipboard():
            clipboard.setText(addition)
            self._status.setText(_("Copied to clipboard"))

    def _multi_lora_entries(self, items: list[QListWidgetItem]) -> list[str]:
        """One line per selected LoRA: its tag plus trigger words per the current mode."""
        strength = self._strength.value()
        trigger_mode = self._multi_trigger_mode.currentData()
        entries = []
        for item in items:
            lora: LoraInfo = item.data(Qt.ItemDataRole.UserRole)
            tag = f"<lora:{lora.name}:{strength:.2f}>"
            trigger_text = ""
            if trigger_mode == _MULTI_TRIGGERS_FIRST and lora.trigger_words:
                trigger_text = lora.trigger_words[0]
            elif trigger_mode == _MULTI_TRIGGERS_ALL and lora.trigger_words:
                trigger_text = ", ".join(lora.trigger_words)
            entries.append(f"{tag} {trigger_text}".strip())
        return entries

    def _build_multi_lora_block(self, items: list[QListWidgetItem]) -> str:
        entries = self._multi_lora_entries(items)
        fmt = self._format_combo.currentData()
        if fmt == _FORMAT_SEPARATE:  # all LoRAs together, no wildcard
            return "\n".join(entries)
        return format_option_group(entries, fmt == _FORMAT_SEQUENTIAL)


class LoraPickerDialog(QDialog):
    """The LoRA library and CivitAI search as two tabs of one window.

    Looking for a LoRA and looking for one to download are the same errand, so they
    share a window instead of the second one being another button in the prompt bar.
    """

    def __init__(self, current_arch: str, parent: QWidget | None = None):
        super().__init__(parent)
        from .civitai_picker import CivitaiBrowser  # imported here: it imports from us

        self.setWindowTitle(_("LoRA Browser"))
        self.setMinimumSize(640, 480)
        self.resize(880, 620)
        # Non-modal: Krita stays interactive while this dialog is open
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        self.library = LoraBrowser(current_arch, self)
        self.civitai = CivitaiBrowser("lora", current_arch, self)

        self._tabs = QTabWidget(self)
        self._tabs.addTab(self.library, _("Library"))
        self._tabs.addTab(self.civitai, _("CivitAI"))

        close_btn = QPushButton(_("Close"), self)
        close_btn.clicked.connect(self.close)  # type: ignore (slot returns bool)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(close_btn)

        layout = QVBoxLayout()
        layout.addWidget(self._tabs, 1)
        layout.addLayout(bottom)
        self.setLayout(layout)

    def closeEvent(self, a0):
        self.library.shutdown()
        super().closeEvent(a0)
