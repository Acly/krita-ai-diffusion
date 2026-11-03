from __future__ import annotations

import asyncio
from pathlib import Path

from PyQt5.QtCore import QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..backend import workflow
from ..backend.client import CheckpointInfo
from ..backend.lora_manager import (
    LoraInfo,
    arch_for_base_model,
    fetch_checkpoints_pages,
    fetch_preview_bytes,
    style_family_for_base_model,
)
from ..backend.resources import Arch
from ..files import FileFormat
from ..localization import translate as _
from ..model.root import root
from ..settings import settings
from ..style import Styles
from ..util import ensure
from . import theme
from .generate_across import AcrossOptions, generate_across_checkpoints
from .generate_sets import GenerateSetBar
from .lora_picker import (
    _extract_video_frame,
    _ffmpeg_path,
    _is_video_url,
    _visible_range,
    _with_tag,
)


def _with_style_badge(pixmap: QPixmap) -> QPixmap:
    """Small green checkmark, top-left: a style already exists for this checkpoint."""
    result = QPixmap(pixmap)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    b = max(14, result.width() // 6)
    x, y = 2, 2
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(60, 170, 75))
    painter.drawRoundedRect(x, y, b, b, 3, 3)
    painter.setPen(QColor(255, 255, 255))
    font = painter.font()
    font.setBold(True)
    font.setPixelSize(max(9, int(b * 0.75)))
    painter.setFont(font)
    painter.drawText(QRect(x, y, b, b), Qt.AlignmentFlag.AlignCenter, "✓")
    painter.end()
    return result


_PREVIEW_SIZE_DEFAULT = 128
_PREVIEW_SIZE_MIN = 64
_PREVIEW_SIZE_MAX = 256
_BASE_ANY = "__any__"
_SORT_NAME = "name"
_SORT_DATE = "date"
_NSFW_ALL = "all"
_NSFW_SAFE = "safe"
_NSFW_HIDE_EXPLICIT = "hide_explicit"
# ComfyUI loader nodes whose file lists hold the models a style can point at
_LOADERS = [
    ("CheckpointLoaderSimple", "ckpt_name", FileFormat.checkpoint),
    ("UNETLoader", "unet_name", FileFormat.diffusion),
    ("UnetLoaderGGUF", "unet_name", FileFormat.diffusion),
]


class CheckpointBrowser(QWidget):
    """Browse Lora Manager checkpoints and bulk-create Krita styles from them,
    cloning the settings of a chosen template style.

    Embedded as a tab of the style dialog: creating a style from a checkpoint is a
    step of picking a style, not a separate errand, and a dialog opened out of a
    dialog was the wrong shape for it."""

    styles_created = pyqtSignal(int)  # number of styles created

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._all: list[LoraInfo] = []
        self._filtered: list[LoraInfo] = []
        self._preview_cache: dict[str, QPixmap] = {}
        self._pending_previews: set[str] = set()
        self._loading = False
        self._preview_size = settings.checkpoint_browser_size

        # ── row 1: search + refresh ──
        self._search = QLineEdit(self)
        self._search.setPlaceholderText(_("Search checkpoints…"))
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)

        self._refresh_btn = QToolButton(self)
        self._refresh_btn.setIcon(theme.icon("reset"))
        self._refresh_btn.setToolTip(_("Reload the checkpoint list from Lora Manager"))
        self._refresh_btn.clicked.connect(self._load)

        self._rescan_btn = QToolButton(self)
        self._rescan_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._rescan_btn.setIcon(theme.icon("comfyui"))
        self._rescan_btn.setText(_("Scan server"))
        self._rescan_btn.setToolTip(
            _("Look for new checkpoint files on the ComfyUI server (checkpoints only)")
        )
        self._rescan_btn.clicked.connect(lambda: self._rescan_server())
        self._after_scan = None

        row1 = QHBoxLayout()
        row1.addWidget(self._search, 1)
        row1.addWidget(self._refresh_btn)
        row1.addWidget(self._rescan_btn)

        # ── row 2: filters ──
        base_label = QLabel(_("Base Model:"), self)
        self._base_combo = QComboBox(self)
        self._base_combo.addItem(_("Any"), _BASE_ANY)
        self._base_combo.currentIndexChanged.connect(self._apply_filter)

        self._favorites_only = QCheckBox(_("Favorites"), self)
        self._favorites_only.toggled.connect(self._apply_filter)

        self._hide_styled = QCheckBox(_("Hide checkpoints with a style"), self)
        self._hide_styled.setToolTip(
            _("Hide checkpoints that already have a style created for them")
        )
        self._hide_styled.toggled.connect(self._apply_filter)

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
        idx = self._sort_combo.findData(settings.checkpoint_browser_sort)
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
        row2.addWidget(base_label)
        row2.addWidget(self._base_combo, 1)
        row2.addWidget(self._favorites_only)
        row2.addWidget(self._hide_styled)
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

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._load_visible_previews)
        ensure(self._grid.verticalScrollBar()).valueChanged.connect(self._schedule_visible_previews)

        # ── bottom bar: template + create ──
        template_label = QLabel(_("Template style:"), self)
        self._template_combo = QComboBox(self)
        self._template_combo.setMinimumWidth(180)
        self._template_combo.setToolTip(
            _(
                "New styles copy this style's settings (sampler, steps, etc.), only the checkpoint differs"
            )
        )

        self._selected_label = QLabel(_("No checkpoint selected"), self)
        self._selected_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        seed_label = QLabel(_("Seed:"), self)
        self._seed_input = QSpinBox(self)
        self._seed_input.setRange(-1, 2**31 - 1)
        self._seed_input.setValue(-1)
        self._seed_input.setSpecialValueText(_("random"))  # -1 shows "random"
        self._seed_input.setToolTip(_("Seed used for all checkpoints (-1 = random, same for all)"))
        self._seed_input.setFixedWidth(110)

        self._generate_btn = QPushButton(_("Generate across"), self)
        self._generate_btn.setEnabled(False)
        self._generate_btn.setToolTip(
            _(
                "Run the current prompt/settings once per selected checkpoint, same seed,"
                " so you can compare them side by side in the history"
            )
        )
        self._generate_btn.clicked.connect(self._generate_across)
        self._across_options = AcrossOptions(self)

        self._create_btn = QPushButton(_("Create Styles"), self)
        self._create_btn.setEnabled(False)
        self._create_btn.clicked.connect(self._create_styles)

        self._set_bar = GenerateSetBar(
            "checkpoint_sets", self._selected_keys, self._apply_set, self
        )

        bottom = QHBoxLayout()
        bottom.addWidget(self._selected_label, 1)
        bottom.addWidget(template_label)
        bottom.addWidget(self._template_combo)
        bottom.addWidget(seed_label)
        bottom.addWidget(self._seed_input)
        bottom.addWidget(self._across_options)
        bottom.addWidget(self._generate_btn)
        bottom.addWidget(self._create_btn)

        # ── status ──
        self._status = QLabel(_("Loading…"), self)
        self._status.setStyleSheet(f"color: {theme.grey}; font-style: italic;")

        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self._grid, 1)
        layout.addWidget(self._status)
        layout.addWidget(self._set_bar)
        layout.addLayout(bottom)
        self.setLayout(layout)

        self._populate_templates()
        self._load()

    # ── templates ──

    def _populate_templates(self):
        current = self._template_combo.currentData()
        self._template_combo.blockSignals(True)
        self._template_combo.clear()
        self._template_combo.addItem(_("(none - blank style)"), None)
        active = root.active_model.style if root.active_model else None
        for style in sorted(Styles.list(), key=lambda s: s.name.lower()):
            self._template_combo.addItem(style.name, style)
        idx = self._template_combo.findData(active) if active else -1
        if idx < 0:
            idx = self._template_combo.findData(current)
        self._template_combo.setCurrentIndex(max(idx, 0))
        self._template_combo.blockSignals(False)

    # ── data loading ──

    def _load(self):
        if self._loading:
            return
        client = root.connection.client_if_connected
        if client is None:
            self._status.setText(_("Not connected to ComfyUI"))
            return
        self._status.setText(_("Loading…"))
        self._grid.clear()
        self._all = []
        self._loading = True
        eventloop.run(self._fetch(client))

    async def _fetch(self, client):
        try:
            async for batch in fetch_checkpoints_pages(client._requests, client.url):
                self._all.extend(batch)
                self._rebuild_filters()
                self._apply_filter()
                self._status.setText(f"{_('Loading…')} ({len(self._all)})")
        finally:
            self._loading = False
            if self._all:
                self._status.setText(f"{len(self._filtered)} / {len(self._all)} checkpoints")
            else:
                self._status.setText(_("No checkpoints found (requires ComfyUI-Lora-Manager)"))

    # ── filtering ──

    def _rebuild_filters(self):
        bases = sorted({c.base_model for c in self._all if c.base_model})
        current = self._base_combo.currentData()
        self._base_combo.blockSignals(True)
        self._base_combo.clear()
        self._base_combo.addItem(_("Any"), _BASE_ANY)
        for bm in bases:
            self._base_combo.addItem(bm, bm)
        idx = self._base_combo.findData(current)
        self._base_combo.setCurrentIndex(max(idx, 0))
        self._base_combo.blockSignals(False)

    def _apply_filter(self):
        search = self._search.text().strip().lower()
        base = self._base_combo.currentData()
        base = "" if base in (None, _BASE_ANY) else base
        favorites_only = self._favorites_only.isChecked()
        hide_styled = self._hide_styled.isChecked()
        has_style = self._styled_stems() if hide_styled else set()
        nsfw_mode = self._nsfw_combo.currentData()

        def matches(c: LoraInfo) -> bool:
            if favorites_only and not c.favorite:
                return False
            if nsfw_mode == _NSFW_SAFE and c.nsfw_level >= 8:  # R and above
                return False
            if nsfw_mode == _NSFW_HIDE_EXPLICIT and c.nsfw_level >= 16:  # X/XXX
                return False
            if hide_styled and c.name in has_style:
                return False
            if base and c.base_model != base:
                return False
            if search:
                haystack = (c.name + " " + c.display_name + " " + " ".join(c.tags)).lower()
                if search not in haystack:
                    return False
            return True

        self._filtered = [c for c in self._all if matches(c)]
        if self._sort_combo.currentData() == _SORT_DATE:
            self._filtered.sort(key=lambda c: c.modified, reverse=True)
        else:
            self._filtered.sort(key=lambda c: (c.display_name or c.name).lower())
        self._populate_grid()

    def _styled_stems(self) -> set[str]:
        """Checkpoint stems (filename without extension) already used by an
        existing style, so the browser can flag "you already have a style
        for this" instead of the user finding out by trial and error."""
        stems = set()
        for style in Styles.list():
            if style.checkpoints:
                stems.add(Path(style.checkpoints[0]).stem)
        return stems

    def _populate_grid(self):
        self._grid.clear()
        cell = self._grid.gridSize()
        has_style = self._styled_stems()
        for c in self._filtered:
            label = c.display_name or c.name
            if c.version and c.version.lower() not in label.lower():
                label = f"{label}  ({c.version})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, c)
            item.setSizeHint(cell)
            fav = "★ " if c.favorite else ""
            version_line = f"\nVersion: {c.version}" if c.version else ""
            styled_line = "\n✓ Style already exists" if c.name in has_style else ""
            item.setToolTip(
                f"{fav}{c.display_name}{version_line}\nFile: {c.name}\nBase: {c.base_model or '?'}{styled_line}"
            )
            self._set_tile_icon(item, c, has_style)
            self._grid.addItem(item)
        if not self._loading:
            self._status.setText(f"{len(self._filtered)} / {len(self._all)} checkpoints")
        self._schedule_visible_previews()

    def _tile_base(self, c: LoraInfo) -> QPixmap:
        size = self._preview_size
        if c.sha256 in self._preview_cache:
            return self._preview_cache[c.sha256].scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        if c.preview_url and _is_video_url(c.preview_url):
            return theme.icon("play").pixmap(size, size)
        blank = QPixmap(size, size)
        blank.fill(Qt.GlobalColor.transparent)
        return blank

    def _base_model_label(self, base_model: str) -> str:
        label = style_family_for_base_model(base_model)
        return base_model if label == "Auto" else label

    def _set_tile_icon(self, item: QListWidgetItem, c: LoraInfo, has_style: set[str] | None = None):
        base = self._tile_base(c)
        styled = c.name in has_style if has_style is not None else c.name in self._styled_stems()
        if styled:
            base = _with_style_badge(base)
        base = _with_tag(base, self._base_model_label(c.base_model))
        item.setIcon(QIcon(base))

    def _on_preview_size_changed(self, value: int):
        self._preview_size = value
        self._grid.setIconSize(QSize(value, value))
        self._grid.setGridSize(QSize(value + 16, value + 40))
        for i in range(self._grid.count()):
            item = ensure(self._grid.item(i))
            c: LoraInfo = item.data(Qt.ItemDataRole.UserRole)
            item.setSizeHint(self._grid.gridSize())
            self._set_tile_icon(item, c)
        self._schedule_visible_previews()

    def _save_size_setting(self):
        settings.checkpoint_browser_size = self._preview_size
        settings.save()

    def _save_sort_setting(self):
        settings.checkpoint_browser_sort = self._sort_combo.currentData()
        settings.save()

    # ── lazy preview loading ──

    def _schedule_visible_previews(self):
        self._preview_timer.start()

    def _load_visible_previews(self):
        viewport_rect = ensure(self._grid.viewport()).rect()
        client = root.connection.client_if_connected
        if client is None:
            return
        for i in _visible_range(self._grid):  # only scan around the viewport
            item = ensure(self._grid.item(i))
            if not self._grid.visualItemRect(item).intersects(viewport_rect):
                continue
            c: LoraInfo = item.data(Qt.ItemDataRole.UserRole)
            if (
                not c.preview_url
                or c.sha256 in self._preview_cache
                or c.sha256 in self._pending_previews
            ):
                continue
            if _is_video_url(c.preview_url) and _ffmpeg_path is None:
                continue  # no decoder available - play placeholder set in _populate_grid
            self._pending_previews.add(c.sha256)
            eventloop.run(self._load_preview(c, item))

    async def _load_preview(self, c: LoraInfo, item: QListWidgetItem):
        client = root.connection.client_if_connected
        if client is None:
            return
        data = await fetch_preview_bytes(client._requests, c.preview_url)
        is_video = _is_video_url(c.preview_url)
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
                self._preview_cache[c.sha256] = pixmap
                self._set_tile_icon(item, c)
            else:
                item.setIcon(theme.icon("filter"))

    # ── selection / create ──

    def _on_selection_changed(self):
        items = self._grid.selectedItems()
        self._create_btn.setEnabled(len(items) > 0)
        self._generate_btn.setEnabled(len(items) > 0 and root.active_model is not None)
        if not items:
            self._selected_label.setText(_("No checkpoint selected"))
        else:
            names = ", ".join(i.data(Qt.ItemDataRole.UserRole).display_name for i in items[:3])
            more = f" +{len(items) - 3}" if len(items) > 3 else ""
            self._selected_label.setText(f"{len(items)} {_('selected')}: {names}{more}")

    def _resolve_checkpoints(self, infos: list[LoraInfo]):
        """Map checkpoints to server file names.

        The client's model list leaves out checkpoints whose base model the server
        could not classify, and ones added after its model cache was written. Those
        are looked up in ComfyUI's loader options instead and registered with the
        architecture Lora Manager reports, which is also returned so a style can
        store it - that is what keeps it resolvable after a reconnect."""
        client = root.connection.client_if_connected
        if client is None:
            return {}, list(infos)
        models = client.models
        known = {Path(k).stem: k for k in models.checkpoints}
        loaders: dict[str, tuple[str, FileFormat]] = {}
        for node, input, format in _LOADERS:
            if node in models.node_inputs:
                for name in models.node_inputs.options(node, input):
                    loaders.setdefault(Path(name).stem, (name, format))

        resolved: dict[str, tuple[str, Arch | None]] = {}
        skipped: list[LoraInfo] = []
        for c in infos:
            if (identifier := known.get(c.name)) is not None:
                resolved[c.name] = (identifier, None)
            elif (entry := loaders.get(c.name)) is not None:
                identifier, format = entry
                arch_name = arch_for_base_model(c.base_model)
                arch = Arch[arch_name] if arch_name else Arch.from_checkpoint_name(identifier)
                models.checkpoints[identifier] = CheckpointInfo(identifier, arch, format)
                resolved[c.name] = (identifier, arch)
            else:
                skipped.append(c)
        return resolved, skipped

    @staticmethod
    def _infos(items: list[QListWidgetItem]) -> list[LoraInfo]:
        return [i.data(Qt.ItemDataRole.UserRole) for i in items]

    def _offer_scan(self, title: str, skipped: list[LoraInfo], retry):
        """Tell which checkpoints the server doesn't list, and offer a checkpoint scan
        that retries just those - the usual cause is a file added after connecting."""
        names = "\n".join(c.display_name or c.name for c in skipped)
        box = QMessageBox(
            QMessageBox.Icon.Warning,
            title,
            _("These checkpoints were not found on the ComfyUI server:") + f"\n\n{names}",
            QMessageBox.StandardButton.Close,
            self,
        )
        scan = box.addButton(_("Scan server and retry"), QMessageBox.ButtonRole.AcceptRole)
        box.exec()
        if box.clickedButton() is scan:
            self._rescan_server(then=lambda: retry(skipped))

    def _rescan_server(self, then=None):
        if root.connection.client_if_connected is None:
            self._status.setText(_("Not connected to ComfyUI"))
            return
        if not self._rescan_btn.isEnabled():  # a scan is already running
            self._status.setText(_("A server scan is already running, try again when it is done"))
            return
        self._after_scan = then
        self._rescan_btn.setEnabled(False)
        self._status.setText(_("Scanning server for new checkpoint files…"))
        # connection emits models_changed when the async refresh finishes
        root.connection.models_changed.connect(self._on_server_scanned)
        root.connection.refresh(quick=True)

    def _on_server_scanned(self):
        try:
            root.connection.models_changed.disconnect(self._on_server_scanned)
        except (TypeError, RuntimeError):
            pass
        self._rescan_btn.setEnabled(True)
        after, self._after_scan = self._after_scan, None
        if after is not None:
            after()
        else:
            self._load()  # also refreshes the "has a style" badges

    def shutdown(self):
        # drop the pending scan callback so it doesn't fire on a closed browser. The
        # dialog is kept for a reopen, so leave the button usable - the scan itself
        # still finishes and updates the model list.
        try:
            root.connection.models_changed.disconnect(self._on_server_scanned)
        except (TypeError, RuntimeError):
            pass
        self._after_scan = None
        self._rescan_btn.setEnabled(True)

    def _create_styles(self):
        items = self._grid.selectedItems()
        if not items:
            return
        template = self._template_combo.currentData()

        template_name = template.name if template else _("blank style")
        confirm = QMessageBox.question(
            self,
            _("Create Styles from Checkpoints"),
            _("Create {n} styles from template '{t}'?").format(n=len(items), t=template_name),
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._create_styles_for(self._infos(items), template)

    def _create_styles_for(self, infos: list[LoraInfo], template):
        resolved, skipped = self._resolve_checkpoints(infos)
        created = 0
        for c in infos:
            if c.name not in resolved:
                continue
            identifier, arch = resolved[c.name]
            style = Styles.list().create(
                filename=f"{c.name}.json", checkpoint=identifier, copy_from=template
            )
            # Styles.create copies ALL settings from the template last, including
            # its checkpoint and architecture - so re-apply ours afterwards
            style.checkpoints = [identifier]
            style.architecture = arch or Arch.auto
            name = c.display_name or c.name
            if c.version and c.version.lower() not in name.lower():
                name = f"{name} ({c.version})"
            style.name = name
            style.base_model_family = style_family_for_base_model(c.base_model)
            style.save()
            created += 1

        if created:
            self.styles_created.emit(created)
            self._apply_filter()  # "has a style" badges
        self._status.setText(_("Created {n} styles").format(n=created))
        if skipped:
            self._offer_scan(
                _("Create Styles from Checkpoints"),
                skipped,
                lambda retry: self._create_styles_for(retry, template),
            )

    def _selected_keys(self) -> list[str]:
        return [i.data(Qt.ItemDataRole.UserRole).name for i in self._grid.selectedItems()]

    def _apply_set(self, names: list[str]):
        # a set can hold checkpoints the current filters hide - reset them,
        # otherwise restoring a set would silently drop half of it
        self._search.clear()
        self._favorites_only.setChecked(False)
        self._hide_styled.setChecked(False)
        self._base_combo.setCurrentIndex(0)
        self._nsfw_combo.setCurrentIndex(0)
        self._apply_filter()

        wanted = set(names)
        self._grid.clearSelection()
        found = set()
        first = None
        for i in range(self._grid.count()):
            item = ensure(self._grid.item(i))
            name = item.data(Qt.ItemDataRole.UserRole).name
            if name in wanted:
                item.setSelected(True)
                found.add(name)
                first = first or item
        if first is not None:
            self._grid.scrollToItem(first)
        text = _("Selected {n} checkpoints").format(n=len(found))
        if missing := len(wanted - found):
            text += "  " + _("({n} no longer available)").format(n=missing)
        self._status.setText(text)

    def _generate_across(self):
        items = self._grid.selectedItems()
        if items:
            self._generate_across_for(self._infos(items))

    def _generate_across_for(self, infos: list[LoraInfo], seed: int | None = None):
        model = root.active_model
        if model is None:
            return
        if seed is None:
            # -1 in the seed box means "pick a random one and reuse it for all"
            seed = self._seed_input.value()
            if seed < 0:
                seed = workflow.generate_seed()
        resolved, skipped = self._resolve_checkpoints(infos)
        ids = [resolved[c.name][0] for c in infos if c.name in resolved]
        if skipped:
            # a retry after the scan keeps the seed, so all runs stay comparable
            self._offer_scan(
                _("Generate across"), skipped, lambda retry: self._generate_across_for(retry, seed)
            )
        if not ids:
            self._status.setText(_("None of the selected checkpoints are on the server"))
            return

        client = root.connection.client_if_connected
        if client is None or not generate_across_checkpoints(
            model, client.models, ids, seed, self._across_options, self
        ):
            return
        self._status.setText(_("Queued {n} checkpoints (seed {s})").format(n=len(ids), s=seed))
