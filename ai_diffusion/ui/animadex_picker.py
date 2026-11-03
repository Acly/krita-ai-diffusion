from __future__ import annotations

import asyncio
import html
import random
from datetime import datetime, timezone

from PyQt5.QtCore import QSize, Qt, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices, QGuiApplication, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..backend import animadex
from ..backend.animadex import Entry, Facet, SearchResult
from ..backend.animadex_catalog import Catalog, ImportFailed, offline_facets, wildcard_lines
from ..localization import translate as _
from ..model.root import root
from ..settings import settings
from ..text import format_option_group
from ..util import client_logger as log
from ..util import ensure
from ..wildcards import WildcardLibrary
from . import theme

_POS_END = "end"
_POS_START = "start"
_POS_CURSOR = "cursor"
_FORMAT_RANDOM = "random"
_FORMAT_SEQUENTIAL = "sequential"
_FORMAT_SEPARATE = "separate"
_SOURCE_AUTO = "auto"
_SOURCE_OFFLINE = "offline"
_SOURCE_ONLINE = "online"
_SIZE_MIN = 64
_SIZE_MAX = 256
_OFFLINE_PAGE_SIZE = 96
_ANY = ""
# Architectures trained on danbooru tags, which is what AnimaDex triggers are.
_TAG_ARCHS = ("anima", "illu", "illu_v")

_facet_order = {
    "characters": ["copyright", "gender", "hair_color", "hair_length", "eye_color"],
    "artists": ["category", "score"],
}


def _visible_range(grid: QListWidget, overscan: int = 24) -> range:
    count = grid.count()
    if count == 0:
        return range(0)
    vp = ensure(grid.viewport()).rect()
    first = grid.indexAt(vp.topLeft())
    last = grid.indexAt(vp.bottomRight())
    start = first.row() if first.isValid() else 0
    end = last.row() if last.isValid() else start + 200
    return range(max(0, start - overscan), min(count, end + overscan + 1))


class AnimadexBrowser(QWidget):
    """Browse AnimaDex characters and artists and put their trigger phrase into the
    prompt. Reads the imported offline catalogue when there is one (instant, no
    rate limit), and the live site otherwise."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._catalog = Catalog.instance()
        self._result = SearchResult()
        self._entries: list[Entry] = []
        self._search_seq = 0
        self._loading = False
        self._importing = False
        self._seed = random.randint(1, 1_000_000)
        self._size = settings.animadex_browser_size
        self._thumbs: dict[str, QPixmap] = {}  # thumb url -> unscaled pixmap
        self._pending: set[str] = set()
        self._facets: dict[tuple[str, str], list[Facet]] = {}  # (source, mode) -> facets

        # ── top row ──
        self._mode_combo = QComboBox(self)
        self._mode_combo.addItem(_("Artists"), "artists")
        self._mode_combo.addItem(_("Characters"), "characters")
        idx = self._mode_combo.findData(settings.animadex_browser_mode)
        self._mode_combo.setCurrentIndex(idx if idx >= 0 else 1)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self._search = QLineEdit(self)
        self._search.setPlaceholderText(_("Search name, series, tags… (comma = AND)"))
        self._search.setClearButtonEnabled(True)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self._new_search)
        self._search.textChanged.connect(lambda _t: self._search_timer.start())
        self._search.returnPressed.connect(self._new_search)

        self._sort_combo = QComboBox(self)
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)

        self._source_combo = QComboBox(self)
        self._source_combo.addItem(_("Auto"), _SOURCE_AUTO)
        self._source_combo.addItem(_("Live site"), _SOURCE_ONLINE)
        self._source_combo.addItem(_("Offline catalogue"), _SOURCE_OFFLINE)
        self._source_combo.setToolTip(
            _(
                "Auto uses the imported offline catalogue if there is one, else the live"
                " site. Only the live site knows LoRA links, votes and artist categories."
            )
        )
        idx = self._source_combo.findData(settings.animadex_browser_source)
        self._source_combo.setCurrentIndex(max(idx, 0))
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)

        self._size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._size_slider.setRange(_SIZE_MIN, _SIZE_MAX)
        self._size_slider.setValue(self._size)
        self._size_slider.setFixedWidth(90)
        self._size_slider.setToolTip(_("Thumbnail size"))
        self._size_slider.valueChanged.connect(self._on_size_changed)
        self._size_slider.sliderReleased.connect(self._save_size)

        row1 = QHBoxLayout()
        row1.addWidget(self._mode_combo)
        row1.addWidget(self._search, 1)
        row1.addWidget(QLabel(_("Sort:"), self))
        row1.addWidget(self._sort_combo)
        row1.addWidget(QLabel(_("Source:"), self))
        row1.addWidget(self._source_combo)
        row1.addWidget(self._size_slider)

        # ── facet row (rebuilt per mode) ──
        self._facet_row = QHBoxLayout()
        self._facet_combos: dict[str, QComboBox] = {}
        self._loras_only = QCheckBox(_("Has LoRA"), self)
        self._loras_only.setToolTip(_("Only characters with an Anima LoRA on CivitAI (live site)"))
        self._loras_only.toggled.connect(self._new_search)

        # ── offline catalogue row ──
        self._catalog_label = QLabel(self)
        self._catalog_label.setStyleSheet(f"color: {theme.grey};")
        self._catalog_label.setWordWrap(True)
        self._import_btn = QToolButton(self)
        self._import_btn.setText(_("Import…"))
        self._import_btn.setToolTip(
            _(
                "Import the offline catalogue with an export token from"
                " animadex.net → Account → Offline dataset export"
            )
        )
        self._import_btn.clicked.connect(self._import_with_token)
        self._update_btn = QToolButton(self)
        self._update_btn.setText(_("Update"))
        self._update_btn.setToolTip(_("Fetch catalogue changes since the last import"))
        self._update_btn.clicked.connect(lambda: self._run_import(""))
        self._wildcard_btn = QToolButton(self)
        self._wildcard_btn.setText(_("Save as Wildcard…"))
        self._wildcard_btn.setToolTip(
            _("Write the trigger of every result of the current search into a wildcard file")
        )
        self._wildcard_btn.clicked.connect(self._save_as_wildcard)
        catalog_row = QHBoxLayout()
        catalog_row.addWidget(self._catalog_label, 1)
        catalog_row.addWidget(self._wildcard_btn)
        catalog_row.addWidget(self._import_btn)
        catalog_row.addWidget(self._update_btn)

        # ── grid ──
        self._grid = QListWidget(self)
        self._grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setMovement(QListWidget.Movement.Static)
        self._grid.setWordWrap(True)
        self._grid.setSpacing(4)
        self._grid.setUniformItemSizes(True)
        self._apply_grid_size()
        self._grid.itemSelectionChanged.connect(self._on_selection_changed)
        self._grid.itemDoubleClicked.connect(lambda _item: self._add_to_prompt())
        self._grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._grid.customContextMenuRequested.connect(self._show_context_menu)
        ensure(self._grid.verticalScrollBar()).valueChanged.connect(self._on_scroll)

        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(120)
        self._thumb_timer.timeout.connect(self._load_visible_thumbs)

        self._status = QLabel(self)
        self._status.setStyleSheet(f"color: {theme.grey}; font-style: italic;")
        self._arch_hint = QLabel(self)
        self._arch_hint.setWordWrap(True)
        self._arch_hint.setStyleSheet(f"color: {theme.yellow};")
        self._net_hint = QLabel(self)
        self._net_hint.setWordWrap(True)
        self._net_hint.setStyleSheet(f"color: {theme.yellow};")
        self._net_hint.setVisible(False)

        # ── bottom row ──
        self._selected_label = QLabel(_("Nothing selected"), self)
        self._selected_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._selected_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._selected_label.setTextFormat(Qt.TextFormat.RichText)

        self._with_tags = QCheckBox(_("+ tags"), self)
        self._with_tags.setToolTip(
            _("Also insert the character's appearance tags (hair, eyes, outfit…)")
        )
        self._with_tags.setChecked(settings.animadex_with_tags)
        self._with_tags.toggled.connect(self._on_with_tags_changed)

        # multi-select only, same choices as the LoRA browser
        self._format_combo = QComboBox(self)
        self._format_combo.addItem(_("Random {a|b}"), _FORMAT_RANDOM)
        self._format_combo.addItem(_("Sequential [[a|b]]"), _FORMAT_SEQUENTIAL)
        self._format_combo.addItem(_("Separate (all)"), _FORMAT_SEPARATE)
        self._format_combo.setToolTip(
            _(
                "Random: one is picked per generation. Sequential: cycles through in batch order."
                " Separate: adds all of them together, no wildcard."
            )
        )
        idx = self._format_combo.findData(settings.animadex_multi_format)
        self._format_combo.setCurrentIndex(max(idx, 0))
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        self._format_combo.setVisible(False)

        self._position_combo = QComboBox(self)
        self._position_combo.addItem(_("at End"), _POS_END)
        self._position_combo.addItem(_("at Start"), _POS_START)
        self._position_combo.addItem(_("at Cursor"), _POS_CURSOR)

        self._add_btn = QPushButton(_("Add to Prompt"), self)
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._add_to_prompt)

        bottom = QHBoxLayout()
        bottom.addWidget(self._selected_label, 1)
        bottom.addWidget(self._with_tags)
        bottom.addWidget(self._format_combo)
        bottom.addWidget(QLabel(_("Insert:"), self))
        bottom.addWidget(self._position_combo)
        bottom.addWidget(self._add_btn)

        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(self._facet_row)
        layout.addLayout(catalog_row)
        layout.addWidget(self._grid, 1)
        layout.addWidget(self._status)
        layout.addWidget(self._arch_hint)
        layout.addWidget(self._net_hint)
        layout.addLayout(bottom)
        self.setLayout(layout)

        self._rebuild_sort_combo()
        self._update_catalog_label()
        self.update_arch_hint()
        self._rebuild_facets()

    # ── state ──

    @property
    def mode(self) -> str:
        return self._mode_combo.currentData()

    @property
    def source(self) -> str:
        chosen = self._source_combo.currentData()
        if chosen == _SOURCE_AUTO:
            return _SOURCE_OFFLINE if self._catalog.available else _SOURCE_ONLINE
        return chosen

    @property
    def host(self) -> str:
        return settings.animadex_host

    def _filters(self) -> dict[str, list[str]]:
        return {k: [c.currentData()] for k, c in self._facet_combos.items() if c.currentData()}

    def update_arch_hint(self):
        model = root.active_model
        arch = model.arch.name if model else ""
        if arch and arch not in _TAG_ARCHS:
            self._arch_hint.setText(
                _(
                    "AnimaDex triggers are danbooru tags made for Anima - the current model"
                    " may not recognize them."
                )
            )
            self._arch_hint.setVisible(True)
        else:
            self._arch_hint.setVisible(False)

    def _update_catalog_label(self):
        state = self._catalog.state
        if self._catalog.available and state.imported_at:
            when = (
                datetime
                .fromtimestamp(state.imported_at, tz=timezone.utc)
                .astimezone()
                .strftime("%Y-%m-%d %H:%M")
            )
            text = _("Offline catalogue imported") + f" {when}"
            if state.version:
                text += f" ({state.version})"
        else:
            text = _("No offline catalogue - browsing the live site. Import for instant search.")
        self._catalog_label.setText(text)
        self._update_btn.setEnabled(bool(state.manifest) and not self._importing)
        self._import_btn.setEnabled(not self._importing)
        self._loras_only.setVisible(self.mode == "characters" and self.source == _SOURCE_ONLINE)
        self._wildcard_btn.setToolTip(
            _(
                "Write the triggers of the selected entries into a wildcard file - or of every"
                " result of the current search when nothing is selected"
            )
            if self.source == _SOURCE_OFFLINE
            else _(
                "Write the triggers of the selected entries into a wildcard file - or of the"
                " results loaded so far when nothing is selected"
            )
        )

    # ── sort / facets ──

    def _rebuild_sort_combo(self):
        sorts = [("count", _("Popular")), ("az", _("A-Z")), ("random", _("Random"))]
        if self.mode == "artists":
            sorts.append(("score", _("Artwork Score")))
        if self.source == _SOURCE_ONLINE:
            sorts += [("liked", _("Most Liked")), ("favourited", _("Most Favourited"))]
            sorts.append(("recent", _("Recently Added")))
        self._sort_combo.blockSignals(True)
        self._sort_combo.clear()
        for key, label in sorts:
            self._sort_combo.addItem(label, key)
        idx = self._sort_combo.findData(settings.animadex_browser_sort)
        self._sort_combo.setCurrentIndex(max(idx, 0))
        self._sort_combo.blockSignals(False)

    def _rebuild_facets(self):
        while self._facet_row.count():
            item = self._facet_row.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None and widget is not self._loras_only:
                widget.deleteLater()
        self._facet_combos.clear()
        mode, source = self.mode, self.source
        allowed = offline_facets[mode] if source == _SOURCE_OFFLINE else _facet_order[mode]
        for key in _facet_order[mode]:
            if key not in allowed:
                continue
            combo = QComboBox(self)
            combo.setMinimumWidth(90)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            combo.addItem(_("Any") + " " + key.replace("_", " "), _ANY)
            combo.currentIndexChanged.connect(self._new_search)
            self._facet_combos[key] = combo
            self._facet_row.addWidget(combo)
        self._facet_row.addWidget(self._loras_only)
        self._facet_row.addStretch(1)
        eventloop.run(self._load_facets(source, mode))

    async def _load_facets(self, source: str, mode: str):
        key = (source, mode)
        facets = self._facets.get(key)
        if facets is None:
            if source == _SOURCE_OFFLINE:
                loop = asyncio.get_running_loop()
                # the first access parses the CSV - keep that off the UI thread
                facets = await loop.run_in_executor(
                    None, lambda: self._catalog.facets(mode, limit=0)
                )
            else:
                facets = animadex.load_cached_facets(mode, self.host)
                if facets is None:
                    facets = await animadex.fetch_facets(mode, self.host)
                    if facets:
                        animadex.save_facet_cache(facets, mode, self.host)
            self._facets[key] = facets
        if (self.source, self.mode) != key:
            return  # switched away while loading
        for facet in facets:
            combo = self._facet_combos.get(facet.key)
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.setItemText(0, _("Any") + " " + facet.label.lower())
            values = facet.values
            if facet.key == "copyright":  # thousands of series: alphabetical is findable
                values = sorted(values, key=lambda v: v.label.lower())
            for v in values:
                combo.addItem(f"{v.label} ({v.count:,})", v.value)
            combo.blockSignals(False)
        self._new_search()

    # ── searching ──

    def _new_search(self):
        self._search_seq += 1
        self._entries = []
        self._result = SearchResult()
        self._grid.clear()
        self._pending.clear()
        self._on_selection_changed()
        self._load_page(1)

    def _load_page(self, page: int):
        self._loading = True
        self._status.setText(_("Loading…"))
        eventloop.run(self._search_page(self._search_seq, page))

    async def _search_page(self, seq: int, page: int):
        mode, source = self.mode, self.source
        query, filters, sort = (
            self._search.text().strip(),
            self._filters(),
            self._sort_combo.currentData(),
        )
        try:
            if source == _SOURCE_OFFLINE:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._catalog.search(
                        mode, query, filters, sort, page, _OFFLINE_PAGE_SIZE, self._seed
                    ),
                )
            else:
                loras_only = self._loras_only.isChecked() and mode == "characters"
                result = await animadex.search(
                    mode, query, filters, sort, page, loras_only, self._seed, self.host
                )
        except Exception as e:
            log.warning(f"AnimaDex search failed: {e}")
            result = SearchResult()
        if seq != self._search_seq:
            return  # a newer search superseded this one
        self._loading = False
        self._result = result
        entries = [e for e in result.entries if not e.is_hidden]
        self._entries.extend(entries)
        for entry in entries:
            self._add_tile(entry)
        if result.total == 0 and source == _SOURCE_ONLINE and not (query or filters):
            self._status.setText(_("Could not reach AnimaDex - check the site in the settings."))
        else:
            self._status.setText(
                f"{len(self._entries):,} / {result.total:,} "
                + (_("artists") if mode == "artists" else _("characters"))
            )
        self._thumb_timer.start()

    def _on_scroll(self, value: int):
        self._thumb_timer.start()
        bar = ensure(self._grid.verticalScrollBar())
        if not self._loading and self._result.has_more and value >= bar.maximum() - bar.pageStep():
            self._load_page(self._result.page + 1)

    # ── tiles ──

    def _apply_grid_size(self):
        self._grid.setIconSize(QSize(self._size, self._size))
        self._grid.setGridSize(QSize(self._size + 16, self._size + 44))

    def _tile_icon(self, entry: Entry) -> QIcon:
        pixmap = self._thumbs.get(entry.thumb_url)
        if pixmap is None:
            blank = QPixmap(self._size, self._size)
            blank.fill(Qt.GlobalColor.transparent)
            return QIcon(blank)
        return QIcon(
            pixmap.scaled(
                self._size,
                self._size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _add_tile(self, entry: Entry):
        text = entry.name
        if entry.copyright_name and self.mode == "characters":
            text += f"\n{entry.copyright_name}"
        if entry.loras:
            text += "  ◆"
        item = QListWidgetItem(self._tile_icon(entry), text)
        item.setData(Qt.ItemDataRole.UserRole, entry)
        item.setSizeHint(self._grid.gridSize())
        item.setToolTip(self._tooltip(entry))
        self._grid.addItem(item)

    def _tooltip(self, entry: Entry) -> str:
        lines = [f"<b>{html.escape(entry.name)}</b>", f"<code>{html.escape(entry.prompt)}</code>"]
        if entry.tags:
            lines.append(f"<i>{', '.join(entry.tags)}</i>")
        stats = [f"{entry.count:,} " + _("danbooru posts")]
        if entry.score:
            stats.append(_("score") + f" {entry.score:.0%}")
        if entry.up_votes or entry.fav_count:
            stats.append(f"▲{entry.up_votes} ▼{entry.down_votes} ♥{entry.fav_count}")
        lines.append(" · ".join(stats))
        if entry.loras:
            lines.append(_("LoRAs on CivitAI:") + " " + ", ".join(l.name for l in entry.loras))
        return "<br>".join(lines)

    def _on_size_changed(self, value: int):
        self._size = value
        self._apply_grid_size()
        for i in range(self._grid.count()):
            item = ensure(self._grid.item(i))
            item.setSizeHint(self._grid.gridSize())
            item.setIcon(self._tile_icon(item.data(Qt.ItemDataRole.UserRole)))
        self._thumb_timer.start()

    def _save_size(self):
        settings.animadex_browser_size = self._size
        settings.save()

    def _load_visible_thumbs(self):
        viewport = ensure(self._grid.viewport()).rect()
        for i in _visible_range(self._grid):
            item = ensure(self._grid.item(i))
            if item is None or not self._grid.visualItemRect(item).intersects(viewport):
                continue
            entry: Entry = item.data(Qt.ItemDataRole.UserRole)
            url = entry.thumb_url
            if not url or url in self._thumbs or url in self._pending:
                continue
            self._pending.add(url)
            eventloop.run(self._load_thumb(entry, self.source, self.mode))

    async def _load_thumb(self, entry: Entry, source: str, mode: str):
        if source == _SOURCE_OFFLINE:
            data = await self._catalog.thumbnail(mode, entry)
        else:
            data = await animadex.fetch_thumbnail(entry.thumb_url)
        self._pending.discard(entry.thumb_url)
        pixmap = QPixmap()
        if not data or not pixmap.loadFromData(data):
            host = animadex.host_of(entry.thumb_url)
            if host in animadex.unreachable_hosts and not self._net_hint.isVisible():
                self._net_hint.setText(animadex.connection_hint(entry.thumb_url))
                self._net_hint.setVisible(True)
            return  # missing, or no WebP decoder in this Qt build - keep the blank tile
        self._net_hint.setVisible(False)
        self._thumbs[entry.thumb_url] = pixmap
        for i in _visible_range(self._grid):
            item = ensure(self._grid.item(i))
            if item is not None and item.data(Qt.ItemDataRole.UserRole) is entry:
                item.setIcon(self._tile_icon(entry))
                break

    # ── changes ──

    def _on_mode_changed(self):
        settings.animadex_browser_mode = self.mode
        settings.save()
        self._rebuild_sort_combo()
        self._update_catalog_label()
        self._rebuild_facets()

    def _on_source_changed(self):
        settings.animadex_browser_source = self._source_combo.currentData()
        settings.save()
        self._rebuild_sort_combo()
        self._update_catalog_label()
        self._rebuild_facets()

    def _on_sort_changed(self):
        settings.animadex_browser_sort = self._sort_combo.currentData()
        settings.save()
        if self._sort_combo.currentData() == "random":
            self._seed = random.randint(1, 1_000_000)  # re-picking Random reshuffles
        self._new_search()

    def _on_with_tags_changed(self, value: bool):
        settings.animadex_with_tags = value
        settings.save()
        self._on_selection_changed()

    # ── selection / insertion ──

    def _selected(self) -> list[Entry]:
        return [i.data(Qt.ItemDataRole.UserRole) for i in self._grid.selectedItems()]

    def _is_wildcard_block(self, entries: list[Entry]) -> bool:
        return len(entries) > 1 and self._format_combo.currentData() != _FORMAT_SEPARATE

    def _prompt_text(self, entries: list[Entry]) -> str:
        with_tags = self._with_tags.isChecked()
        lines = [e.prompt_with_tags if with_tags else e.prompt for e in entries]
        if not self._is_wildcard_block(entries):
            return ", ".join(lines)
        # One option per line, like the LoRA browser. Triggers contain commas
        # ("hatsune miku, vocaloid"), which is fine: options only split at "|".
        return format_option_group(lines, self._format_combo.currentData() == _FORMAT_SEQUENTIAL)

    def _on_format_changed(self):
        settings.animadex_multi_format = self._format_combo.currentData()
        settings.save()
        self._on_selection_changed()

    def _on_selection_changed(self):
        entries = self._selected()
        self._add_btn.setEnabled(bool(entries))
        self._format_combo.setVisible(len(entries) > 1)
        if entries:
            count = _("{count} selected").format(count=len(entries))
            preview = self._prompt_text(entries).replace("\n|\n", " | ").replace("\n", "")
            self._selected_label.setText(f"<b>{count}</b> · {html.escape(preview)}")
            self._wildcard_btn.setText(_("Save Selected as Wildcard…"))
        else:
            self._selected_label.setText(_("Nothing selected"))
            self._wildcard_btn.setText(_("Save as Wildcard…"))

    def _add_to_prompt(self):
        entries = self._selected()
        model = root.active_model
        if not entries or model is None:
            return
        text = self._prompt_text(entries)
        position = self._position_combo.currentData()
        if position == _POS_CURSOR and self._insert_at_cursor(text):
            return
        region = model.regions.active_or_root
        current = region.positive.strip()
        block = self._is_wildcard_block(entries)  # a multi-line group goes on its own line
        if not current:
            region.positive = text
        elif position == _POS_START:
            region.positive = f"{text}\n{current}" if block else f"{text}, {current}"
        elif block:
            region.positive = f"{current}\n{text}"
        else:
            separator = " " if current.endswith(",") else ", "
            region.positive = current + separator + text

    def _insert_at_cursor(self, text: str) -> bool:
        # the prompt widget is an ancestor, not the direct parent: that is the dialog
        widget = None
        node = self.parent()
        while node is not None and widget is None:
            widget = getattr(node, "positive", None)
            node = node.parent()
        if widget is None or not hasattr(widget, "textCursor"):
            return False
        cursor = widget.textCursor()
        cursor.insertText(text)
        widget.setTextCursor(cursor)
        return True

    def _show_context_menu(self, pos):
        item = self._grid.itemAt(pos)
        if item is None:
            return
        entry: Entry = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.addAction(_("Copy Trigger"), lambda: self._copy(entry.prompt))
        if entry.tags:
            menu.addAction(_("Copy Trigger + Tags"), lambda: self._copy(entry.prompt_with_tags))
        if entry.source_url.startswith("https://"):
            menu.addAction(_("Open on Danbooru"), lambda: self._open(entry.source_url))
        for lora in entry.loras:
            if lora.url.startswith("https://"):
                menu.addAction(_("LoRA:") + f" {lora.name}", lambda u=lora.url: self._open(u))
        menu.exec(self._grid.mapToGlobal(pos))

    def _copy(self, text: str):
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)

    def _open(self, url: str):
        QDesktopServices.openUrl(QUrl(url))

    # ── wildcards ──

    def _save_as_wildcard(self):
        # A selection is what the user means; only without one does "all" apply.
        entries = self._selected()
        if entries:
            scope = _("{count} selected").format(count=len(entries))
        elif self.source == _SOURCE_OFFLINE:
            entries = self._catalog.all_matching(
                self.mode, self._search.text().strip(), self._filters()
            )
            entries = [e for e in entries if not e.is_hidden]
            scope = _("all {count} results").format(count=len(entries))
        else:
            entries = list(self._entries)
            scope = _("{count} loaded results").format(count=len(entries))
        with_tags = self._with_tags.isChecked()
        lines = wildcard_lines(entries, with_tags)
        if not lines:
            QMessageBox.information(self, _("Save as Wildcard"), _("The search has no results."))
            return
        default = (
            "animadex/" + (self._search.text().strip().replace(",", " ").split() or [self.mode])[0]
        )
        name, ok = QInputDialog.getText(
            self,
            _("Save as Wildcard"),
            _("{scope}, {count} lines{tags}. File name (may include a folder/ prefix):").format(
                scope=scope,
                count=len(lines),
                tags=_(" with tags") if with_tags else "",
            ),
            QLineEdit.EchoMode.Normal,
            default,
        )
        if not ok or not name.strip():
            return
        library = WildcardLibrary.instance()
        if library.create(name, lines):
            QMessageBox.information(
                self,
                _("Save as Wildcard"),
                _("Saved - use it in the prompt as") + f" __{name.strip().strip('/').lower()}__",
            )
        else:
            QMessageBox.warning(
                self,
                _("Save as Wildcard"),
                _("Could not create the wildcard file - a file with that name may already exist."),
            )

    # ── import ──

    def _import_with_token(self):
        token, ok = QInputDialog.getText(
            self,
            _("Import AnimaDex Catalogue"),
            _(
                "Export token from animadex.net → Account → Offline dataset export.\n"
                "It is only used for this import and not saved. Thumbnails are loaded\n"
                "as you browse, so the import itself is only a few MB."
            ),
            QLineEdit.EchoMode.Password,
        )
        if ok and token.strip():
            self._run_import(token.strip())

    def _run_import(self, token: str):
        if self._importing:
            return
        self._importing = True
        self._update_catalog_label()
        eventloop.run(self._do_import(token))

    async def _do_import(self, token: str):
        try:
            summary = await self._catalog.update(
                token=token, host=self.host, progress=self._catalog_label.setText
            )
        except ImportFailed as e:
            self._importing = False
            self._update_catalog_label()
            if e.kind == "token":
                answer = QMessageBox.question(
                    self,
                    _("Import AnimaDex Catalogue"),
                    str(e) + "\n\n" + _("Enter a new export token?"),
                )
                if answer == QMessageBox.StandardButton.Yes:
                    self._import_with_token()
            else:
                QMessageBox.warning(self, _("Import AnimaDex Catalogue"), str(e))
            return
        except Exception as e:
            log.exception("AnimaDex import failed")
            self._importing = False
            self._update_catalog_label()
            QMessageBox.warning(self, _("Import AnimaDex Catalogue"), str(e))
            return
        self._importing = False
        self._facets = {k: v for k, v in self._facets.items() if k[0] != _SOURCE_OFFLINE}
        self._update_catalog_label()
        self._status.setText(
            _("Imported {characters} characters and {artists} artists ({changed} changed)").format(
                characters=f"{summary.characters:,}",
                artists=f"{summary.artists:,}",
                changed=f"{summary.changed:,}",
            )
        )
        if self.source == _SOURCE_OFFLINE:
            self._rebuild_sort_combo()
            self._rebuild_facets()


class AnimadexPickerDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(_("AnimaDex"))
        self.setMinimumSize(560, 420)
        self.resize(900, 680)
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        self.browser = AnimadexBrowser(self)
        close_btn = QPushButton(_("Close"), self)
        close_btn.clicked.connect(self.close)  # type: ignore (slot returns bool)
        bottom = QHBoxLayout()
        link = QLabel("<a href='https://animadex.net'>animadex.net</a>", self)
        link.setOpenExternalLinks(True)
        bottom.addWidget(link)
        bottom.addStretch(1)
        bottom.addWidget(close_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.browser, 1)
        layout.addLayout(bottom)
        self.setLayout(layout)

    def showEvent(self, a0):
        self.browser.update_arch_hint()  # the model may have changed while hidden
        super().showEvent(a0)
