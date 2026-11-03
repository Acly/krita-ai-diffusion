from __future__ import annotations

import asyncio
from pathlib import Path

from PyQt5.QtCore import QEvent, QObject, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QKeyEvent, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..backend import workflow
from ..backend.client import filter_supported_styles, resolve_arch
from ..backend.lora_manager import fetch_checkpoints_pages, fetch_preview_bytes
from ..localization import translate as _
from ..model.connection import ConnectionState
from ..model.root import root
from ..settings import settings
from ..style import Style, Styles, sort_recent_styles
from ..util import ensure
from . import theme
from .generate_across import AcrossOptions, generate_across_styles
from .generate_sets import GenerateSetBar
from .lora_picker import (
    _extract_video_frame,
    _ffmpeg_path,
    _is_video_url,
    _visible_range,
    _with_tag,
)

_ARCH_ANY = "__any__"
_VIEW_LIST = "list"
_VIEW_GRID = "grid"
_SORT_NAME = "name"
_SORT_DATE = "date"


def _tint_pixmap(pixmap: QPixmap, color) -> QPixmap:
    tinted = QPixmap(pixmap.size())
    tinted.fill(Qt.GlobalColor.transparent)
    painter = QPainter(tinted)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), color)
    painter.end()
    return tinted


_star_pixmap = _tint_pixmap(QPixmap(str(theme.icon_path / "star.png")), Qt.GlobalColor.white)


def _favorite_icon(base: QIcon) -> QIcon:
    """Overlay a white star badge on the bottom-right corner of a checkpoint icon."""
    size = 16
    pixmap = base.pixmap(size, size)
    if pixmap.isNull():
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
    badge_size = max(7, size // 2)
    star = _star_pixmap.scaled(
        badge_size,
        badge_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    painter = QPainter(pixmap)
    painter.drawPixmap(pixmap.width() - badge_size, pixmap.height() - badge_size, star)
    painter.end()
    return QIcon(pixmap)


class StyleBrowser(QWidget):
    """The style preset list - one tab of StylePickerDialog."""

    style_selected = pyqtSignal(Style)

    def __init__(self, current: Style, parent: QWidget | None = None):
        super().__init__(parent)
        self._current = current
        self._styles: list[Style] = []
        self._recent: set[str] = set()
        self._favorites_only = False
        self._ckpt_previews: dict[str, str] = {}  # checkpoint stem -> preview url
        self._preview_cache: dict[str, QPixmap] = {}  # stem -> pixmap
        self._pending_previews: set[str] = set()
        self._view = settings.style_browser_view
        self._thumb_size = settings.style_browser_size

        self._search = QLineEdit(self)
        self._search.setPlaceholderText(_("Search styles…"))
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)

        self._refresh_btn = QToolButton(self)
        self._refresh_btn.setIcon(theme.icon("reset"))
        self._refresh_btn.setToolTip(_("Reload styles and checkpoint thumbnails"))
        self._refresh_btn.clicked.connect(self._refresh)

        arch_label = QLabel(_("Base Model:"), self)
        self._arch_combo = QComboBox(self)
        self._arch_combo.addItem(_("Any"), _ARCH_ANY)
        self._arch_combo.currentIndexChanged.connect(self._apply_filter)

        self._favorites_check = QCheckBox(_("Favorites"), self)
        self._favorites_check.setIcon(QIcon(_star_pixmap))
        self._favorites_check.toggled.connect(self._set_favorites_only)

        self._view_combo = QComboBox(self)
        self._view_combo.addItem(_("List"), _VIEW_LIST)
        self._view_combo.addItem(_("Grid"), _VIEW_GRID)
        idx = self._view_combo.findData(self._view)
        self._view_combo.setCurrentIndex(max(idx, 0))
        self._view_combo.currentIndexChanged.connect(self._set_view_mode)
        self._view_combo.currentIndexChanged.connect(self._save_view_setting)

        self._sort_combo = QComboBox(self)
        self._sort_combo.addItem(_("Name"), _SORT_NAME)
        self._sort_combo.addItem(_("Date Added"), _SORT_DATE)
        idx = self._sort_combo.findData(settings.style_browser_sort)
        self._sort_combo.setCurrentIndex(max(idx, 0))
        self._sort_combo.currentIndexChanged.connect(self._apply_filter)
        self._sort_combo.currentIndexChanged.connect(self._save_sort_setting)

        size_label = QLabel(_("Size:"), self)
        self._size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._size_slider.setMinimum(32)
        self._size_slider.setMaximum(512)
        self._size_slider.setValue(self._thumb_size)
        self._size_slider.setFixedWidth(90)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        self._size_slider.sliderReleased.connect(self._save_size_setting)

        row1 = QHBoxLayout()
        row1.addWidget(self._search, 1)
        row1.addWidget(self._refresh_btn)

        row2 = QHBoxLayout()
        row2.addWidget(arch_label)
        row2.addWidget(self._arch_combo, 1)
        row2.addWidget(self._favorites_check)
        row2.addWidget(self._view_combo)
        row2.addWidget(self._sort_combo)
        row2.addWidget(size_label)
        row2.addWidget(self._size_slider)

        self._list = QListWidget(self)
        self._list.setIconSize(QSize(self._thumb_size, self._thumb_size))
        self._list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._list.itemActivated.connect(self._activate)
        self._list.itemSelectionChanged.connect(self._update_actions)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.installEventFilter(self)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._load_visible_previews)
        if scrollbar := ensure(self._list.verticalScrollBar()):
            scrollbar.valueChanged.connect(self._schedule_visible_previews)

        hint = QLabel(
            _("Double-click to select. Ctrl/Shift-click several, then Generate across."), self
        )
        hint.setStyleSheet(f"color: {theme.grey}; font-style: italic;")

        seed_label = QLabel(_("Seed:"), self)
        self._seed_input = QSpinBox(self)
        self._seed_input.setRange(-1, 2**31 - 1)
        self._seed_input.setValue(-1)
        self._seed_input.setSpecialValueText(_("random"))
        self._seed_input.setToolTip(_("Seed used for all styles (-1 = random, same for all)"))
        self._seed_input.setFixedWidth(110)

        self._generate_btn = QPushButton(_("Generate across"), self)
        self._generate_btn.setEnabled(False)
        self._generate_btn.setToolTip(
            _("Run the current prompt once per selected style, same seed, to compare them")
        )
        self._generate_btn.clicked.connect(self._generate_across)
        self._across_options = AcrossOptions(self)

        self._favorite_btn = QPushButton(_("Favorite"), self)
        self._favorite_btn.setEnabled(False)
        self._favorite_btn.clicked.connect(self._toggle_selected_favorite)

        self._delete_btn = QPushButton(_("Delete"), self)
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._delete_selected)

        self._set_bar = GenerateSetBar("style_sets", self._selected_keys, self._apply_set, self)

        bottom = QHBoxLayout()
        bottom.addWidget(self._set_bar)
        bottom.addStretch(1)
        bottom.addWidget(seed_label)
        bottom.addWidget(self._seed_input)
        bottom.addWidget(self._across_options)
        bottom.addWidget(self._generate_btn)
        bottom.addWidget(self._favorite_btn)
        bottom.addWidget(self._delete_btn)

        self._status = QLabel(self)
        self._status.setStyleSheet(f"color: {theme.grey}; font-style: italic;")

        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._status)
        layout.addWidget(hint)
        layout.addLayout(bottom)
        self.setLayout(layout)

        Styles.list().changed.connect(self._reload)
        settings.changed.connect(self._on_settings_changed)
        # setCurrentIndex on _view_combo ran before currentIndexChanged was connected,
        # so a saved Grid setting never reached the list widget - it stayed in Qt's
        # default List mode until the user touched the combo. Apply it explicitly.
        self._set_view_mode()
        self._reload()
        self._load_checkpoint_previews()

    def _set_view_mode(self):
        self._view = self._view_combo.currentData()
        size = self._thumb_size
        # Batched layout avoids laying out the whole (possibly large) list at once;
        # uniform sizes lets Qt skip per-item geometry lookups during scroll - only
        # safe in grid mode, list mode mixes in shorter section-header rows
        self._list.setLayoutMode(QListWidget.LayoutMode.Batched)
        self._list.setBatchSize(100)
        if self._view == _VIEW_GRID:
            self._list.setViewMode(QListWidget.ViewMode.IconMode)
            self._list.setFlow(QListWidget.Flow.LeftToRight)
            self._list.setWrapping(True)
            self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
            self._list.setMovement(QListWidget.Movement.Static)
            self._list.setWordWrap(True)
            self._list.setSpacing(4)
            self._list.setIconSize(QSize(size, size))
            self._list.setUniformItemSizes(True)
        else:
            self._list.setViewMode(QListWidget.ViewMode.ListMode)
            self._list.setFlow(QListWidget.Flow.TopToBottom)
            self._list.setWrapping(False)
            self._list.setResizeMode(QListWidget.ResizeMode.Fixed)
            self._list.setMovement(QListWidget.Movement.Static)
            self._list.setWordWrap(False)
            self._list.setSpacing(0)
            self._list.setIconSize(QSize(size, size))
            # now safe: headers and content rows are both given the same sizeHint
            self._list.setUniformItemSizes(True)
        self._apply_filter()

    def _on_size_changed(self, value: int):
        self._thumb_size = value
        self._set_view_mode()

    def _save_size_setting(self):
        settings.style_browser_size = self._thumb_size
        settings.save()

    def _save_sort_setting(self):
        settings.style_browser_sort = self._sort_combo.currentData()
        settings.save()

    def _save_view_setting(self):
        settings.style_browser_view = self._view_combo.currentData()
        settings.save()

    # ── checkpoint thumbnails from Lora Manager ──

    def _refresh(self):
        self._ckpt_previews.clear()
        self._preview_cache.clear()
        self._pending_previews.clear()
        self._reload()
        self._load_checkpoint_previews()

    def _style_stem(self, style: Style) -> str:
        return Path(style.checkpoints[0]).stem if style.checkpoints else ""

    def _scaled_icon(self, pixmap: QPixmap, label: str = "") -> QIcon:
        # pre-scale to the icon size - handing QListWidget a full-resolution pixmap
        # makes it rescale on every repaint, which is what killed scroll performance
        s = self._thumb_size
        scaled = pixmap.scaled(
            s, s, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        if label:
            scaled = _with_tag(scaled, label)
        return QIcon(scaled)

    def _load_checkpoint_previews(self):
        client = self._client()
        if client is not None:
            eventloop.run(self._fetch_checkpoint_previews(client))

    async def _fetch_checkpoint_previews(self, client):
        try:
            async for batch in fetch_checkpoints_pages(client._requests, client.url):
                for c in batch:
                    if c.preview_url:
                        self._ckpt_previews[c.name] = c.preview_url
        finally:
            self._schedule_visible_previews()

    def _schedule_visible_previews(self):
        self._preview_timer.start()

    def _load_visible_previews(self):
        # only touch items currently in the viewport - applying a cached preview is
        # cheap, only a cache miss triggers a network fetch (one per checkpoint)
        viewport_rect = ensure(self._list.viewport()).rect()
        if self._client() is None:
            return
        for i in _visible_range(self._list):  # only scan around the viewport
            item = ensure(self._list.item(i))
            if item is None or item.data(Qt.ItemDataRole.UserRole) is None:
                continue
            if item.data(Qt.ItemDataRole.UserRole + 2):  # preview already applied
                continue
            if not self._list.visualItemRect(item).intersects(viewport_rect):
                continue
            stem = item.data(Qt.ItemDataRole.UserRole + 1)
            if not stem:
                continue
            if stem in self._preview_cache:
                family_label = item.data(Qt.ItemDataRole.UserRole + 3) or ""
                item.setIcon(self._scaled_icon(self._preview_cache[stem], family_label))
                item.setData(Qt.ItemDataRole.UserRole + 2, True)
                continue
            if stem in self._pending_previews:
                continue
            url = self._ckpt_previews.get(stem)
            if not url or (_is_video_url(url) and _ffmpeg_path is None):
                continue
            self._pending_previews.add(stem)
            eventloop.run(self._load_preview(stem, url, item))

    async def _load_preview(self, stem: str, url: str, item: QListWidgetItem):
        client = self._client()
        if client is None:
            return
        data = await fetch_preview_bytes(client._requests, url)
        if data and _is_video_url(url):
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _extract_video_frame, data)
        if not data:
            return
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        if pixmap.isNull():
            return
        self._preview_cache[stem] = pixmap
        # apply to the requesting item; others sharing the stem pick it up from the
        # cache when they next scroll into view (see _load_visible_previews)
        try:
            family_label = item.data(Qt.ItemDataRole.UserRole + 3) or ""
            item.setIcon(self._scaled_icon(pixmap, family_label))
            item.setData(Qt.ItemDataRole.UserRole + 2, True)
        except RuntimeError:
            pass  # item was removed (list rebuilt) before the fetch finished

    def _client(self):
        if root.connection.state is ConnectionState.connected:
            return root.connection.client_if_connected
        return None

    def _on_settings_changed(self, name: str, value: object):
        if name == "favorite_styles":
            self._apply_filter()

    def _set_favorites_only(self, value: bool):
        self._favorites_only = value
        self._apply_filter()

    def _is_favorite(self, style: Style) -> bool:
        return style.filename in settings.favorite_styles

    def _toggle_favorite(self, style: Style):
        favorites = list(settings.favorite_styles)
        if style.filename in favorites:
            favorites.remove(style.filename)
        else:
            favorites.append(style.filename)
        settings.favorite_styles = favorites
        settings.save()

    def _reload(self):
        client = self._client()
        filtered = filter_supported_styles(Styles.list().filtered(), client)
        recent, remaining = sort_recent_styles(
            filtered, settings.recent_styles, settings.recent_styles_count
        )
        if self._current not in recent and self._current not in remaining:
            remaining.insert(0, self._current)
        self._recent = {s.filename for s in recent}
        self._styles = recent + remaining

        families = sorted({s.effective_family(resolve_arch(s, client)) for s in self._styles})
        current_arch = self._arch_combo.currentData()
        self._arch_combo.blockSignals(True)
        self._arch_combo.clear()
        self._arch_combo.addItem(_("Any"), _ARCH_ANY)
        for family in families:
            self._arch_combo.addItem(family, family)
        idx = self._arch_combo.findData(current_arch)
        self._arch_combo.setCurrentIndex(max(idx, 0))
        self._arch_combo.blockSignals(False)

        self._apply_filter()

    def _apply_filter(self):
        client = self._client()
        search = self._search.text().strip().lower()
        family_filter = self._arch_combo.currentData()
        family_filter = "" if family_filter in (None, _ARCH_ANY) else family_filter

        self._list.clear()
        section = None  # None | "favorites" | "recent" | "all"
        shown = 0

        # group into contiguous sections (favorites > recent > all) so each
        # header appears exactly once - favorites can be scattered through the
        # recent+remaining ordering otherwise
        def bucket(style: Style) -> int:
            if self._is_favorite(style):
                return 0
            if style.filename in self._recent:
                return 1
            return 2

        def sort_value(style: Style):
            if self._sort_combo.currentData() == _SORT_DATE:
                try:
                    return -style.filepath.stat().st_ctime
                except OSError:
                    return 0.0
            return style.name.lower()

        def full_key(style: Style):
            b = bucket(style)
            # "Recently Used" keeps its own recency order regardless of the sort
            # dropdown - that's the point of the section
            return (b, 0) if b == 1 else (b, sort_value(style))

        for style in sorted(self._styles, key=full_key):
            arch = resolve_arch(style, client)
            if family_filter and style.effective_family(arch) != family_filter:
                continue
            is_fav = self._is_favorite(style)
            if self._favorites_only and not is_fav:
                continue
            haystack = (style.name + " " + " ".join(style.checkpoints)).lower()
            if search and search not in haystack:
                continue

            is_recent = style.filename in self._recent
            wanted_section = "favorites" if is_fav else ("recent" if is_recent else "all")
            # section headers only make sense in the list view (a full-width row);
            # the grid view shows a flat grid in the same favorites>recent>all order
            if self._view == _VIEW_LIST and wanted_section != section:
                section = wanted_section
                label = {
                    "favorites": _("Favorites"),
                    "recent": _("Recently Used"),
                    "all": _("All Styles"),
                }[section]
                header = QListWidgetItem(label)
                header.setFlags(Qt.ItemFlag.NoItemFlags)
                # same height as content rows - required for setUniformItemSizes,
                # which is what makes scrolling fast (see _set_view_mode)
                header.setSizeHint(QSize(9999, self._thumb_size + 8))
                self._list.addItem(header)

            stem = self._style_stem(style)
            family_label = style.effective_family(arch)
            applied = False
            if stem in self._preview_cache:  # real checkpoint thumbnail if we have it
                icon = self._scaled_icon(self._preview_cache[stem], family_label)
                applied = True
            else:
                icon = theme.checkpoint_icon(arch, client=client)
                if is_fav:
                    icon = _favorite_icon(icon)
                pixmap = _with_tag(icon.pixmap(self._thumb_size, self._thumb_size), family_label)
                icon = QIcon(pixmap)
            item = QListWidgetItem(icon, style.name)
            item.setData(Qt.ItemDataRole.UserRole, style.filename)
            item.setData(Qt.ItemDataRole.UserRole + 1, stem)
            item.setData(Qt.ItemDataRole.UserRole + 2, applied)
            item.setData(Qt.ItemDataRole.UserRole + 3, family_label)
            if self._view == _VIEW_GRID:
                item.setSizeHint(QSize(self._thumb_size + 16, self._thumb_size + 40))
                item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            else:
                item.setSizeHint(QSize(9999, self._thumb_size + 8))
            if style == self._current:
                item.setSelected(True)
            self._list.addItem(item)
            shown += 1

        self._status.setText(f"{shown} / {len(self._styles)} styles")
        self._schedule_visible_previews()

    def _activate(self, item: QListWidgetItem):
        filename = item.data(Qt.ItemDataRole.UserRole)
        if filename is None:
            return
        if style := Styles.list().find(filename):
            self._current = style
            self.style_selected.emit(style)
            ensure(self.window()).close()  # picking a style is the point - close the window

    def _selected_style(self) -> Style | None:
        items = self._list.selectedItems()
        if not items:
            return None
        filename = items[0].data(Qt.ItemDataRole.UserRole)
        return Styles.list().find(filename) if filename else None

    def _selected_styles(self) -> list[Style]:
        result = []
        for item in self._list.selectedItems():
            filename = item.data(Qt.ItemDataRole.UserRole)
            if filename and (style := Styles.list().find(filename)):
                result.append(style)
        return result

    def _selected_keys(self) -> list[str]:
        return [s.filename for s in self._selected_styles()]

    def _apply_set(self, filenames: list[str]):
        # a set can hold styles the current filters hide - reset them, otherwise
        # restoring a set would silently drop half of it
        self._search.clear()
        self._favorites_check.setChecked(False)
        self._arch_combo.setCurrentIndex(0)
        self._apply_filter()

        wanted = set(filenames)
        self._list.clearSelection()
        found = set()
        first = None
        for i in range(self._list.count()):
            item = ensure(self._list.item(i))
            filename = item.data(Qt.ItemDataRole.UserRole)
            if filename in wanted:
                item.setSelected(True)
                found.add(filename)
                first = first or item
        if first is not None:
            self._list.scrollToItem(first)
        missing = len(wanted - found)
        text = _("Selected {n} styles").format(n=len(found))
        if missing:
            text += "  " + _("({n} no longer available)").format(n=missing)
        self._status.setText(text)
        self._update_actions()

    def _generate_across(self):
        styles = self._selected_styles()
        model = root.active_model
        client = root.connection.client_if_connected
        if not styles or model is None or client is None:
            return
        seed = self._seed_input.value()
        if seed < 0:
            seed = workflow.generate_seed()
        generate_across_styles(model, client.models, styles, seed, self._across_options, self)

    def _is_builtin(self, style: Style) -> bool:
        return style.filename.startswith("built-in/")

    def _update_actions(self):
        selected = self._selected_styles()
        style = selected[0] if selected else None
        self._favorite_btn.setEnabled(style is not None)
        self._delete_btn.setEnabled(style is not None and not self._is_builtin(style))
        self._generate_btn.setEnabled(bool(selected) and root.active_model is not None)
        if style is not None:
            self._favorite_btn.setText(
                _("Remove Favorite") if self._is_favorite(style) else _("Add Favorite")
            )
        else:
            self._favorite_btn.setText(_("Favorite"))

    def _toggle_selected_favorite(self):
        if style := self._selected_style():
            self._toggle_favorite(style)
            self._update_actions()

    def _delete_selected(self):
        self._delete_style(self._selected_style())

    def _delete_style(self, style: Style | None):
        if style is None or self._is_builtin(style):
            return
        confirm = QMessageBox.question(
            self,
            _("Delete Style"),
            _("Delete the style '{name}'? This removes its file and cannot be undone.").format(
                name=style.name
            ),
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            Styles.list().delete(style)  # _reload runs via the changed signal

    def _show_context_menu(self, pos):
        item = self._list.itemAt(pos)
        if item is None or item.data(Qt.ItemDataRole.UserRole) is None:
            return
        filename = item.data(Qt.ItemDataRole.UserRole)
        style = Styles.list().find(filename)
        if style is None:
            return
        menu = theme.style_menu(QMenu(self))
        label = _("Remove from Favorites") if self._is_favorite(style) else _("Add to Favorites")
        menu.addAction(label + "\tF", lambda: self._toggle_favorite(style))
        if not self._is_builtin(style):
            menu.addSeparator()
            menu.addAction(_("Delete Style"), lambda: self._delete_style(style))
        menu.exec(self._list.mapToGlobal(pos))

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a0 is self._list and a1 and a1.type() == QEvent.Type.ShortcutOverride:
            assert isinstance(a1, QKeyEvent)
            if a1.key() == Qt.Key.Key_F:
                self._toggle_selected_favorite()
                a1.accept()
                return True
        return super().eventFilter(a0, a1)


class StylePickerDialog(QDialog):
    """Styles, the checkpoints they can be built from, and CivitAI in one window.

    Creating a style from a checkpoint used to open a second dialog on top of this
    one, and downloading a checkpoint a third. As tabs they share a window and their
    state: a checkpoint downloaded in the CivitAI tab shows up in the Checkpoints
    tab, and styles created there land in the list behind the first tab.
    """

    style_selected = pyqtSignal(Style)

    def __init__(self, current: Style, parent: QWidget | None = None):
        super().__init__(parent)
        from .checkpoint_picker import CheckpointBrowser
        from .civitai_picker import CivitaiBrowser

        self.setWindowTitle(_("Select Style"))
        self.setMinimumSize(560, 480)
        self.resize(880, 620)
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        self.styles = StyleBrowser(current, self)
        self.checkpoints = CheckpointBrowser(self)
        self.civitai = CivitaiBrowser("checkpoint", parent=self)

        self._tabs = QTabWidget(self)
        self._tabs.addTab(self.styles, _("Styles"))
        self._tabs.addTab(self.checkpoints, _("Checkpoints"))
        self._tabs.addTab(self.civitai, _("CivitAI"))

        self.styles.style_selected.connect(self.style_selected)
        # styles created from checkpoints belong in the list right away
        self.checkpoints.styles_created.connect(lambda _count: self.styles._reload())

        close_btn = QPushButton(_("Close"), self)
        close_btn.clicked.connect(self.close)  # type: ignore (slot returns bool)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(close_btn)

        layout = QVBoxLayout()
        layout.addWidget(self._tabs, 1)
        layout.addLayout(bottom)
        self.setLayout(layout)

    def set_current(self, style: Style):
        """Point the style list at the style that is active now, for a reopen."""
        self.styles._current = style
        self.styles._reload()

    def closeEvent(self, a0):
        self.checkpoints.shutdown()
        super().closeEvent(a0)
