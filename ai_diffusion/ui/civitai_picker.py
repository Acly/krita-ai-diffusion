from __future__ import annotations

import asyncio

from PyQt5.QtCore import QRect, QSize, Qt, QTimer, QUrl
from PyQt5.QtGui import (
    QColor,
    QDesktopServices,
    QGuiApplication,
    QIcon,
    QPainter,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..backend import civitai
from ..backend.civitai import CivitaiModel, CivitaiVersion
from ..backend.lora_manager import (
    arch_for_base_model,
    cancel_download,
    clear_lora_cache,
    fetch_download_progress,
    fetch_folders,
    fetch_installed_hashes,
    fetch_model_roots,
    load_cached_loras,
    new_download_id,
    start_download,
)
from ..localization import translate as _
from ..model.root import root
from ..settings import settings
from ..util import ensure
from . import theme
from .lora_picker import (
    _ARCH_LABELS,
    _NSFW_ALL,
    _NSFW_HIDE_EXPLICIT,
    _NSFW_SAFE,
    _visible_range,
    _with_badges,
)

_PREVIEW_SIZE_DEFAULT = 128
_PREVIEW_SIZE_MIN = 64
_PREVIEW_SIZE_MAX = 384
_ARCH_ANY = "__any__"
_PAGE_SIZE = 50
# CivitAI applies baseModels (and our own rating/library filters run on top) after
# paging, so a page of 50 can leave a handful of tiles - keep pulling pages until
# the grid holds at least this many, or the result set runs out.
_MIN_RESULTS = 24
_MAX_AUTO_PAGES = 10

_KIND_LORA = "lora"
_KIND_CHECKPOINT = "checkpoint"

# state of a search result relative to the local library
_STATE_NEW = ""
_STATE_INSTALLED = "installed"
_STATE_UPDATE = "update"

_STATE_COLORS = {
    _STATE_INSTALLED: QColor(60, 170, 75),
    _STATE_UPDATE: QColor(60, 130, 210),
}
_STATE_SYMBOLS = {_STATE_INSTALLED: "✓", _STATE_UPDATE: "↑"}
_STATE_LABELS = {_STATE_INSTALLED: _("installed"), _STATE_UPDATE: _("update")}

_SHOW_ALL = "all"
_SHOW_NEW = "new"
_SHOW_INSTALLED = "installed"


def _with_state_marker(pixmap: QPixmap, state: str) -> QPixmap:
    """Banner across the top of the tile: green "installed" for this exact version,
    blue "update" when another version of the same model is in the library.

    A small corner badge was too easy to miss against a busy preview image, so the
    whole tile is dimmed and labelled instead - the point is to see at a glance
    which results are worth looking at."""
    if state == _STATE_NEW:
        return pixmap
    result = QPixmap(pixmap)
    w, h = result.width(), result.height()
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 90))  # dim the preview

    bar = max(14, h // 6)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_STATE_COLORS[state])
    painter.drawRect(0, 0, w, bar)

    font = painter.font()
    font.setBold(True)
    font.setPixelSize(max(8, int(bar * 0.62)))
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))
    label = f"{_STATE_SYMBOLS[state]} {_STATE_LABELS[state]}"
    metrics = painter.fontMetrics()
    elided = metrics.elidedText(label, Qt.TextElideMode.ElideRight, w - 6)
    painter.drawText(QRect(0, 0, w, bar), Qt.AlignmentFlag.AlignCenter, elided)
    painter.end()
    return result


def _nsfw_request_flag(mode: str) -> bool | None:
    """What to ask CivitAI for. Explicitly requesting nsfw=true widens the result set
    beyond the default; None leaves the choice to CivitAI, and the rating filter is
    then applied client-side on top."""
    if mode == _NSFW_SAFE:
        return False
    if mode == _NSFW_ALL:
        return True
    return None


def _distinguisher(model: CivitaiModel, version: CivitaiVersion, others: list) -> str:
    """Something to tell two results with the same title apart.

    CivitAI does host distinct models under identical names, including several by
    the same uploader, so a tile that looks like a duplicate usually is not one.
    Prefer the uploader, fall back to the version and finally the model id."""
    same_name = [m for m, _v in others if m.name == model.name]
    if model.creator and any(m.creator != model.creator for m in same_name):
        return model.creator
    if version.name:
        return version.name
    return f"#{model.id}"


def _size_label(mb: float) -> str:
    if mb <= 0:
        return ""
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


class CivitaiBrowser(QWidget):
    """Search civitai.com and download models straight into the local library.

    Lives as a tab next to the local browsers rather than as a dialog of its own,
    so remote and local models of the same kind sit in one place. `kind` fixes which
    of the two it searches for.

    The plugin only searches; the download itself is handed to ComfyUI-Lora-Manager,
    which knows the folder layout and writes metadata and previews alongside the file.
    """

    def __init__(
        self, kind: str = _KIND_LORA, current_arch: str = "", parent: QWidget | None = None
    ):
        super().__init__(parent)
        self._kind = kind
        self._models: list[CivitaiModel] = []
        self._cursor = ""
        self._loading = False
        self._preview_size = _PREVIEW_SIZE_DEFAULT
        self._preview_cache: dict[int, QPixmap] = {}
        self._pending_previews: set[int] = set()
        self._pending_details: set[int] = set()
        self._installed_hashes: set[str] = set()
        self._installed_models: set[int] = set()
        self._download_id = ""
        self._download_running = False
        self._search_generation = 0
        self._known_tags: list[str] = []
        self._checked_hashes: set[str] = set()
        self._batch_cancelled = False
        self._batch_label = ""
        self._seen_ids: set[int] = set()
        self._auto_pages = 0

        # ── search row ──
        self._search = QLineEdit(self)
        self._search.setPlaceholderText(_("Search CivitAI…"))
        self._search.setClearButtonEnabled(True)
        self._search.returnPressed.connect(self._start_search)

        search_btn = QPushButton(_("Search"), self)
        search_btn.clicked.connect(self._start_search)

        self._refresh_btn = QToolButton(self)
        self._refresh_btn.setIcon(theme.icon("reset"))
        self._refresh_btn.setToolTip(
            _(
                "Run the search again and reload everything cached here:"
                " previews, the tag list, the download folders, and which models are"
                " already in your library."
            )
        )
        self._refresh_btn.setAutoRaise(True)
        self._refresh_btn.clicked.connect(self._refresh_all)

        self._sort_combo = QComboBox(self)
        for option in civitai.sort_options:
            self._sort_combo.addItem(_(option), option)
        self._sort_combo.currentIndexChanged.connect(self._start_search)

        self._period_combo = QComboBox(self)
        for option in civitai.period_options:
            self._period_combo.addItem(_(option), option)
        self._period_combo.currentIndexChanged.connect(self._start_search)

        row1 = QHBoxLayout()
        row1.addWidget(self._search, 1)
        row1.addWidget(search_btn)
        row1.addWidget(self._refresh_btn)
        row1.addWidget(self._sort_combo)
        row1.addWidget(self._period_combo)

        # ── filter row ──
        self._arch_combo = QComboBox(self)
        self._arch_combo.addItem(_("Any base model"), _ARCH_ANY)
        for arch in sorted(civitai.civitai_base_models, key=lambda a: _ARCH_LABELS.get(a, a)):
            self._arch_combo.addItem(_ARCH_LABELS.get(arch, arch), arch)
        index = self._arch_combo.findData(current_arch)
        self._arch_combo.setCurrentIndex(max(index, 0))
        self._arch_combo.currentIndexChanged.connect(self._start_search)

        self._tag_combo = QComboBox(self)
        self._tag_combo.setEditable(True)
        self._tag_combo.setMinimumWidth(150)
        self._tag_combo.addItem(_("Any tag"), "")
        self._tag_combo.setToolTip(
            _(
                "Filter by a CivitAI tag (character, style, clothing, ...). The list is"
                " the site's own tag vocabulary, most used first. A tag CivitAI does"
                " not know is ignored by the search, so unknown ones are flagged here."
            )
        )
        self._tag_combo.currentIndexChanged.connect(self._start_search)
        ensure(self._tag_combo.lineEdit()).returnPressed.connect(self._start_search)

        self._nsfw_combo = QComboBox(self)
        self._nsfw_combo.addItem(_("Safe only"), _NSFW_SAFE)
        self._nsfw_combo.addItem(_("Hide explicit"), _NSFW_HIDE_EXPLICIT)
        self._nsfw_combo.addItem(_("All ratings"), _NSFW_ALL)
        nsfw_index = self._nsfw_combo.findData(settings.civitai_nsfw_filter)
        self._nsfw_combo.setCurrentIndex(max(nsfw_index, 0))
        self._nsfw_combo.setToolTip(
            _(
                "Content rating limit for CivitAI results. The starting value comes from"
                " Settings - Interface, and changing it here updates that setting."
            )
        )
        self._nsfw_combo.currentIndexChanged.connect(self._save_nsfw_setting)
        self._nsfw_combo.currentIndexChanged.connect(self._start_search)

        self._show_combo = QComboBox(self)
        self._show_combo.addItem(_("All results"), _SHOW_ALL)
        self._show_combo.addItem(_("Not in library"), _SHOW_NEW)
        self._show_combo.addItem(_("In library"), _SHOW_INSTALLED)
        self._show_combo.setToolTip(
            _(
                "Hide models you already have (or show only those). Counts a model as"
                " in your library when the exact version is installed, or another"
                " version of it is."
            )
        )
        self._show_combo.currentIndexChanged.connect(self._apply_filter)

        self._commercial_check = QCheckBox(_("Sellable images only"), self)
        self._commercial_check.setToolTip(
            _(
                "Only show models whose CivitAI license allows selling generated images"
                " (allowCommercialUse contains 'Image').\nFilters the results already"
                " loaded - CivitAI cannot filter this server-side."
            )
        )
        self._commercial_check.stateChanged.connect(self._apply_filter)

        size_label = QLabel(_("Size:"), self)
        self._size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._size_slider.setMinimum(_PREVIEW_SIZE_MIN)
        self._size_slider.setMaximum(_PREVIEW_SIZE_MAX)
        self._size_slider.setValue(self._preview_size)
        self._size_slider.setFixedWidth(90)
        self._size_slider.valueChanged.connect(self._on_preview_size_changed)

        row2 = QHBoxLayout()
        row2.addWidget(self._arch_combo)
        row2.addWidget(self._tag_combo)
        row2.addWidget(self._nsfw_combo)
        row2.addWidget(self._show_combo)
        row2.addWidget(self._commercial_check)
        row2.addStretch(1)
        row2.addWidget(size_label)
        row2.addWidget(self._size_slider)

        # ── result grid ──
        self._grid = QListWidget(self)
        self._grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(QSize(self._preview_size, self._preview_size))
        self._grid.setGridSize(QSize(self._preview_size + 16, self._preview_size + 44))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setMovement(QListWidget.Movement.Static)
        self._grid.setWordWrap(True)
        self._grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._grid.customContextMenuRequested.connect(self._show_context_menu)
        self._grid.itemSelectionChanged.connect(self._on_selection_changed)
        self._grid.itemDoubleClicked.connect(lambda _item: self._download_selected())
        ensure(self._grid.verticalScrollBar()).valueChanged.connect(self._on_scrolled)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._load_visible_previews)

        # ── download row (hidden while idle) ──
        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress_label = QLabel("", self)
        self._cancel_btn = QToolButton(self)
        self._cancel_btn.setIcon(theme.icon("cancel"))
        self._cancel_btn.setToolTip(_("Cancel the running download"))
        self._cancel_btn.setAutoRaise(True)
        self._cancel_btn.clicked.connect(self._cancel_download)
        self._download_row = QWidget(self)
        download_layout = QHBoxLayout()
        download_layout.setContentsMargins(0, 0, 0, 0)
        download_layout.addWidget(self._progress_label, 1)
        download_layout.addWidget(self._progress, 1)
        download_layout.addWidget(self._cancel_btn)
        self._download_row.setLayout(download_layout)
        self._download_row.setVisible(False)

        # ── download location ──
        self._root_combo = QComboBox(self)
        self._root_combo.addItem(_("Lora Manager default"), "")
        self._root_combo.setToolTip(
            _("Model root new downloads are written to, as configured in Lora Manager")
        )
        self._root_combo.currentIndexChanged.connect(self._save_location_setting)

        self._folder_combo = QComboBox(self)
        self._folder_combo.setEditable(True)
        self._folder_combo.setMinimumWidth(200)
        self._folder_combo.addItem(_("(root folder)"), "")
        self._folder_combo.setToolTip(
            _(
                "Subfolder below that root. Pick an existing one or type a new path -"
                " Lora Manager creates it on download."
            )
        )
        self._folder_combo.currentIndexChanged.connect(self._save_location_setting)
        ensure(self._folder_combo.lineEdit()).editingFinished.connect(self._save_location_setting)

        location_row = QHBoxLayout()
        location_row.addWidget(QLabel(_("Save to:"), self))
        location_row.addWidget(self._root_combo, 1)
        location_row.addWidget(self._folder_combo, 1)

        # ── bottom bar ──
        self._selected_label = QLabel(_("No model selected"), self)
        self._selected_label.setWordWrap(True)
        self._version_combo = QComboBox(self)
        self._version_combo.setMinimumWidth(140)
        self._version_combo.setToolTip(_("Model version to download"))
        self._version_combo.currentIndexChanged.connect(self._on_version_changed)
        self._download_btn = QPushButton(theme.icon("web-connection"), _("Download"), self)
        self._download_btn.setEnabled(False)
        self._download_btn.clicked.connect(self._download_selected)
        bottom = QHBoxLayout()
        bottom.addWidget(self._selected_label, 1)
        bottom.addWidget(self._version_combo)
        bottom.addWidget(self._download_btn)

        self._status = QLabel("", self)
        self._status.setStyleSheet(f"color: {theme.grey}; font-style: italic;")

        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self._grid, 1)
        layout.addWidget(self._status)
        layout.addWidget(self._download_row)
        layout.addLayout(location_row)
        layout.addLayout(bottom)
        self.setLayout(layout)

        self._refresh_installed()
        self._load_tags()
        self._load_locations()
        self._start_search()

    def _refresh_all(self):
        """Everything in this dialog is cached one way or another - previews, the tag
        vocabulary, the folder list and the installed-model set. Reload the lot, so
        one button covers "I downloaded something elsewhere" as well as stale results."""
        self._checked_hashes.clear()
        self._refresh_installed()
        self._load_tags()
        self._load_locations()
        self._start_search()

    def _api_key(self) -> str:
        return settings.civitai_api_key.strip()

    def _save_nsfw_setting(self):
        settings.civitai_nsfw_filter = self._nsfw_combo.currentData()
        settings.save()

    def _current_tag(self) -> str:
        # editable combo: the placeholder entry shows its label as text, which would
        # otherwise be sent as a literal tag named "any tag"
        text = self._tag_combo.currentText().strip().lower()
        return "" if text == _("Any tag").lower() else text

    def _load_tags(self):
        eventloop.run(self._fetch_tags())

    async def _fetch_tags(self):
        tags = await civitai.fetch_tags(api_key=self._api_key())
        if not tags:
            return
        self._known_tags = tags
        current = self._tag_combo.currentText()
        with theme.SignalBlocker(self._tag_combo):
            self._tag_combo.clear()
            self._tag_combo.addItem(_("Any tag"), "")
            for tag in sorted(tags):
                self._tag_combo.addItem(tag, tag)
            self._tag_combo.setCurrentText(current)

    # ── download location ──

    def _save_location_setting(self):
        settings.civitai_download_root = self._root_combo.currentData() or ""
        settings.civitai_download_subfolder = self._folder_combo.currentText().strip()
        settings.save()

    def _load_locations(self):
        eventloop.run(self._fetch_locations())

    async def _fetch_locations(self):
        client = root.connection.client_if_connected
        if client is None:
            return
        kind = self._library_kind()
        roots = await fetch_model_roots(client._requests, client.url, kind)
        folders = await fetch_folders(client._requests, client.url, kind)
        with theme.SignalBlocker(self._root_combo):
            self._root_combo.clear()
            self._root_combo.addItem(_("Lora Manager default"), "")
            for entry in roots:
                self._root_combo.addItem(entry, entry)
            index = self._root_combo.findData(settings.civitai_download_root)
            self._root_combo.setCurrentIndex(max(index, 0))
        with theme.SignalBlocker(self._folder_combo):
            self._folder_combo.clear()
            self._folder_combo.addItem(_("(root folder)"), "")
            for entry in sorted(f for f in folders if f):
                self._folder_combo.addItem(entry, entry)
            self._folder_combo.setCurrentText(settings.civitai_download_subfolder)

    # ── local library state ──

    def _refresh_installed(self):
        """Hashes and model ids of what is already in the library, so results can be
        marked as installed. Uses the LoRA browser's cache (checkpoints are not
        cached, so those stay unmarked) - a miss only means a tile is not marked,
        and Lora Manager still refuses a duplicate download."""
        client = root.connection.client_if_connected
        if client is None:
            return
        cached = load_cached_loras(client.url)
        if not cached:
            # a download of our own drops the cache, and it is only rebuilt when the
            # LoRA browser next runs - keep what we know rather than losing the marks
            return
        self._installed_hashes = {lora.sha256.upper() for lora in cached if lora.sha256}
        self._installed_models = {
            int(lora.civitai_model_id) for lora in cached if lora.civitai_model_id
        }

    async def _check_installed(self, models: list[CivitaiModel]):
        """Ask Lora Manager which of these versions are on disk.

        The cached LoRA list is only a hint - it expires, and it is empty until the
        LoRA browser has been opened once. Lora Manager can filter its list by a batch
        of hashes, so one request per page of results gives an authoritative answer.
        """
        client = root.connection.client_if_connected
        if client is None:
            return
        hashes = []
        for model in models:
            for version in model.versions:
                file = version.primary_file
                if file and file.sha256 and file.sha256.upper() not in self._checked_hashes:
                    hashes.append(file.sha256.upper())
        if not hashes:
            return
        self._checked_hashes.update(hashes)
        found = await fetch_installed_hashes(
            client._requests, client.url, hashes, self._library_kind()
        )
        if not found:
            return
        self._installed_hashes.update(found)
        self._apply_filter()

    def _state_of(self, model: CivitaiModel, version: CivitaiVersion) -> str:
        file = version.primary_file
        if file and file.sha256 and file.sha256.upper() in self._installed_hashes:
            return _STATE_INSTALLED
        if model.id in self._installed_models:
            return _STATE_UPDATE
        return _STATE_NEW

    # ── search ──

    def _current_types(self) -> list[str]:
        if self._kind == _KIND_CHECKPOINT:
            return civitai.checkpoint_types
        return civitai.lora_types

    def _library_kind(self) -> str:
        """Lora Manager's url prefix for this browser's model kind."""
        return "checkpoints" if self._kind == _KIND_CHECKPOINT else "loras"

    def _current_base_models(self) -> list[str]:
        arch = self._arch_combo.currentData()
        if arch == _ARCH_ANY:
            return []
        return civitai.civitai_base_models.get(arch, [])

    def _start_search(self):
        # abandon a request that is still in flight - its results belong to the
        # previous query and must not be appended to this one
        self._search_generation += 1
        self._loading = False
        self._models = []
        self._seen_ids = set()
        self._auto_pages = 0
        self._cursor = ""
        self._grid.clear()
        self._preview_cache.clear()
        self._pending_previews.clear()
        self._pending_details.clear()
        self._fetch_page()

    def _fetch_page(self):
        if self._loading:
            return
        self._loading = True
        self._status.setText(_("Searching CivitAI…"))
        # the generation is captured here, not inside the coroutine: _start_search may
        # run again before the coroutine starts, and results of the older query would
        # then be counted as belonging to the newer one
        eventloop.run(self._fetch(self._search_generation))

    async def _fetch(self, generation: int):
        nsfw_mode = self._nsfw_combo.currentData()
        try:
            models, cursor = await civitai.search_models(
                query=self._search.text().strip(),
                types=self._current_types(),
                tag=self._current_tag(),
                base_models=self._current_base_models(),
                sort=self._sort_combo.currentData(),
                period=self._period_combo.currentData(),
                nsfw=_nsfw_request_flag(nsfw_mode),
                limit=_PAGE_SIZE,
                cursor=self._cursor,
                api_key=self._api_key(),
            )
        except Exception as e:
            if generation == self._search_generation:
                self._loading = False
                self._status.setText(_("CivitAI search failed") + f": {e}")
            return
        if generation != self._search_generation:
            return  # a newer search replaced this one
        self._loading = False
        self._cursor = cursor
        # the same model can come back on a later page when the query is re-ranked
        fresh = [m for m in models if m.id not in self._seen_ids]
        self._seen_ids.update(m.id for m in fresh)
        self._models.extend(fresh)
        self._apply_filter()
        eventloop.run(self._check_installed(fresh))
        # a page of 50 raw results can survive as a handful of tiles - keep going
        # instead of leaving a nearly empty grid that has no scrollbar to trigger
        # loading the rest
        if (
            self._cursor
            and self._grid.count() < _MIN_RESULTS
            and self._auto_pages < _MAX_AUTO_PAGES
        ):
            self._auto_pages += 1
            self._fetch_page()
        tag = self._current_tag()
        if tag and self._known_tags and tag not in self._known_tags:
            # CivitAI ignores an unknown tag instead of returning nothing, so results
            # would silently be unfiltered - say so rather than let it look like a hit
            self._status.setText(
                self._status.text() + f" – {_('unknown tag')} '{tag}', {_('filter ignored')}"
            )

    def _on_scrolled(self, value: int):
        self._schedule_visible_previews()
        scroll = ensure(self._grid.verticalScrollBar())
        near_bottom = value >= scroll.maximum() - scroll.pageStep() // 2
        if near_bottom and self._cursor and not self._loading:
            self._fetch_page()

    # ── filtering / grid ──

    def _visible_models(self) -> list[tuple[CivitaiModel, CivitaiVersion]]:
        nsfw_mode = self._nsfw_combo.currentData()
        arch = self._arch_combo.currentData()
        # only needed where CivitAI has no usable baseModels label (see civitai.py)
        client_side_arch = arch != _ARCH_ANY and not civitai.civitai_base_models.get(arch)
        result = []
        for model in self._models:
            version = model.versions[0] if model.versions else None
            if version is None:
                continue
            if self._commercial_check.isChecked() and model.commercial != "yes":
                continue
            # A model's nsfwLevel is the union of the ratings of ALL its images, so
            # one spicy example sets the R bit on an otherwise tame model - filtering
            # on it dropped over half of even an nsfw=false result set. Judge the
            # preview we actually show instead, like the local LoRA browser does.
            preview_level = version.preview_nsfw_level
            if nsfw_mode == _NSFW_SAFE and preview_level >= 8:
                continue
            if nsfw_mode == _NSFW_HIDE_EXPLICIT and preview_level >= 16:
                continue
            if client_side_arch and arch_for_base_model(version.base_model) != arch:
                continue
            show = self._show_combo.currentData()
            installed = self._state_of(model, version) != _STATE_NEW
            if show == _SHOW_NEW and installed:
                continue
            if show == _SHOW_INSTALLED and not installed:
                continue
            result.append((model, version))
        return result

    def _apply_filter(self):
        selected_ids = {
            item.data(Qt.ItemDataRole.UserRole).id for item in self._grid.selectedItems()
        }
        # appending a page rebuilds the grid - keep the viewport where it was, or
        # loading more results would yank the user back to the top
        scroll = ensure(self._grid.verticalScrollBar()).value()
        # rebuilding re-emits selection changes, which would reset the version combo
        visible = self._visible_models()
        names = [model.name for model, _version in visible]
        with theme.SignalBlocker(self._grid):
            self._grid.clear()
            for model, version in visible:
                label = model.name
                if names.count(model.name) > 1:
                    label = f"{model.name} · {_distinguisher(model, version, visible)}"
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, model)
                item.setSizeHint(self._grid.gridSize())
                item.setToolTip(self._tooltip(model, version))
                self._set_tile_icon(item, model, version)
                self._grid.addItem(item)
                if model.id in selected_ids:
                    item.setSelected(True)
        # the grid lays out asynchronously, so the scroll position only sticks once
        # the new items have been arranged
        QTimer.singleShot(0, lambda: ensure(self._grid.verticalScrollBar()).setValue(scroll))
        total = len(self._models)
        shown = self._grid.count()
        if self._loading:
            more = _("loading more…")
        elif self._cursor:
            more = _("scroll for more")
        else:
            more = _("end of results")
        self._status.setText(f"{shown} / {total} {_('results')} – {more}")
        self._schedule_visible_previews()

    def _tooltip(self, model: CivitaiModel, version: CivitaiVersion) -> str:
        lines = [model.name]
        if model.creator:
            lines.append(f"{_('By')}: {model.creator}")
        lines.append(f"{_('Type')}: {model.type} · {version.base_model or '?'}")
        if size := _size_label(version.size_mb):
            lines.append(f"{_('Version')}: {version.name or '?'} · {size}")
        lines.append(f"{_('Downloads')}: {model.downloads}")
        verdict = {"yes": _("images may be sold"), "no": _("images may not be sold")}
        lines.append(
            f"{_('License')}: {model.license_summary}"
            f" ({verdict.get(model.commercial, _('unstated'))})"
        )
        if version.trained_words:
            lines.append(f"{_('Triggers')}: {' | '.join(version.trained_words[:4])}")
        if model.poi:
            lines.append(_("Depicts a real person"))
        if version.early_access:
            lines.append(_("Early access - requires a purchase on CivitAI"))
        state = self._state_of(model, version)
        if state == _STATE_INSTALLED:
            lines.append(_("Already in your library"))
        elif state == _STATE_UPDATE:
            lines.append(_("Another version of this model is in your library"))
        return "\n".join(lines)

    def _tile_base(self, version: CivitaiVersion) -> QPixmap:
        size = self._preview_size
        if version.id in self._preview_cache:
            return self._preview_cache[version.id].scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        if version.preview_url and civitai.is_video_url(version.preview_url):
            return theme.icon("play").pixmap(size, size)
        blank = QPixmap(size, size)
        blank.fill(Qt.GlobalColor.transparent)
        return blank

    def _set_tile_icon(self, item: QListWidgetItem, model: CivitaiModel, version: CivitaiVersion):
        pixmap = _with_badges(self._tile_base(version), False, model.commercial, version.base_model)
        item.setIcon(QIcon(_with_state_marker(pixmap, self._state_of(model, version))))

    def _on_preview_size_changed(self, value: int):
        self._preview_size = value
        self._grid.setIconSize(QSize(value, value))
        self._grid.setGridSize(QSize(value + 16, value + 44))
        for i in range(self._grid.count()):
            item = ensure(self._grid.item(i))
            model: CivitaiModel = item.data(Qt.ItemDataRole.UserRole)
            item.setSizeHint(self._grid.gridSize())
            if model.versions:
                self._set_tile_icon(item, model, model.versions[0])
        self._schedule_visible_previews()

    # ── lazy previews ──

    def _schedule_visible_previews(self):
        self._preview_timer.start()

    def _load_visible_previews(self):
        viewport = ensure(self._grid.viewport()).rect()
        for i in _visible_range(self._grid):
            item = ensure(self._grid.item(i))
            if item is None or not self._grid.visualItemRect(item).intersects(viewport):
                continue
            model: CivitaiModel = item.data(Qt.ItemDataRole.UserRole)
            if not model.versions:
                continue
            version = model.versions[0]
            if version.id in self._preview_cache or version.id in self._pending_previews:
                continue
            if not version.preview_url:
                # the search endpoint omits images for models rated R and above; the
                # detail endpoint still has them, so fetch one for this tile only
                if model.id not in self._pending_details:
                    self._pending_details.add(model.id)
                    eventloop.run(self._load_missing_preview(model, version, item))
                continue
            self._pending_previews.add(version.id)
            eventloop.run(self._load_preview(model, version, item))

    async def _load_missing_preview(
        self, model: CivitaiModel, version: CivitaiVersion, item: QListWidgetItem
    ):
        url, nsfw_level = await civitai.fetch_model_preview(model.id, self._api_key())
        if not url:
            return
        version.preview_url = url
        version.preview_nsfw_level = nsfw_level
        if version.id in self._pending_previews:
            return
        self._pending_previews.add(version.id)
        await self._load_preview(model, version, item)

    async def _load_preview(self, model: CivitaiModel, version: CivitaiVersion, item):
        # CivitAI's CDN renders a still frame for animated previews (anim=false), so
        # even mp4 previews arrive as a small JPEG - no local video decoding needed
        is_video = civitai.is_video_url(version.preview_url)
        url = civitai.preview_thumbnail_url(version.preview_url, _PREVIEW_SIZE_MAX)
        data = await civitai.fetch_image(url)
        if not data:
            return
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        if pixmap.isNull():
            return
        if is_video:  # mark it as coming from an animated preview
            badge = theme.icon("play").pixmap(24, 24)
            painter = QPainter(pixmap)
            painter.drawPixmap(pixmap.width() - 28, pixmap.height() - 28, badge)
            painter.end()
        self._preview_cache[version.id] = pixmap
        try:
            self._set_tile_icon(item, model, version)
        except RuntimeError:
            pass  # item removed while the image was loading

    # ── selection ──

    def _selected_model(self) -> tuple[CivitaiModel, CivitaiVersion] | None:
        items = self._grid.selectedItems()
        if not items:
            return None
        model: CivitaiModel = items[0].data(Qt.ItemDataRole.UserRole)
        if not model.versions:
            return None
        index = self._version_combo.currentIndex()
        versions = model.versions
        version = versions[index] if 0 <= index < len(versions) else versions[0]
        return model, version

    def _on_selection_changed(self):
        items = self._grid.selectedItems()
        self._version_combo.setEnabled(len(items) == 1)
        if len(items) == 1:
            model: CivitaiModel = items[0].data(Qt.ItemDataRole.UserRole)
            with theme.SignalBlocker(self._version_combo):
                self._version_combo.clear()
                for version in model.versions:
                    size = _size_label(version.size_mb)
                    label = version.name or str(version.id)
                    self._version_combo.addItem(f"{label} ({size})" if size else label, version.id)
                self._version_combo.setCurrentIndex(0)
        elif not items:
            self._version_combo.clear()
        self._update_selection_label()

    def _on_version_changed(self):
        self._update_selection_label()

    def _update_selection_label(self):
        downloads = self._selected_downloads()
        if not downloads:
            self._selected_label.setText(_("No model selected"))
            self._download_btn.setText(_("Download"))
            self._download_btn.setEnabled(False)
            return
        pending = [
            (m, v)
            for m, v in downloads
            if self._state_of(m, v) != _STATE_INSTALLED and not v.early_access
        ]
        if len(downloads) == 1:
            model, version = downloads[0]
            state = self._state_of(model, version)
            parts = [model.name, version.base_model or "?"]
            if size := _size_label(version.size_mb):
                parts.append(size)
            if state == _STATE_INSTALLED:
                parts.append(_("already in library"))
            self._selected_label.setText(" · ".join(parts))
            if version.early_access:
                self._download_btn.setToolTip(_("Early access - requires a purchase on CivitAI"))
            elif state == _STATE_INSTALLED:
                self._download_btn.setToolTip(_("This version is already in your library"))
            else:
                self._download_btn.setToolTip(_("Download into the local model library"))
            self._download_btn.setText(_("Download"))
        else:
            total = sum(v.size_mb for _model, v in pending)
            summary = f"{len(downloads)} {_('models selected')}"
            if skipped := len(downloads) - len(pending):
                summary += f" ({skipped} {_('already in library')})"
            if size := _size_label(total):
                summary += f" · {size}"
            self._selected_label.setText(summary)
            self._download_btn.setText(f"{_('Download')} ({len(pending)})")
            self._download_btn.setToolTip(
                _("Download the selected models one after another into the local library")
            )
        self._download_btn.setEnabled(not self._download_running and bool(pending))

    def _show_context_menu(self, pos):
        item = self._grid.itemAt(pos)
        if item is None:
            return
        model: CivitaiModel = item.data(Qt.ItemDataRole.UserRole)
        version = model.versions[0] if model.versions else None
        menu = QMenu(self)
        menu.addAction(
            _("Open on CivitAI"),
            lambda: QDesktopServices.openUrl(
                QUrl(civitai.model_page_url(model.id, version.id if version else 0))
            ),
        )
        if version and version.trained_words:
            triggers = ", ".join(version.trained_words)
            menu.addAction(
                _("Copy trigger words"),
                lambda: ensure(QGuiApplication.clipboard()).setText(triggers),
            )
        menu.exec(self._grid.mapToGlobal(pos))

    # ── download ──

    def _selected_downloads(self) -> list[tuple[CivitaiModel, CivitaiVersion]]:
        """Model/version pairs to download for the current selection.

        With a single tile selected the version combo decides; with several, each
        model contributes its newest version - picking versions per model would need
        a dialog per model, which defeats the point of selecting a batch."""
        items = self._grid.selectedItems()
        if len(items) == 1:
            selection = self._selected_model()
            return [selection] if selection else []
        result = []
        for item in items:
            model: CivitaiModel = item.data(Qt.ItemDataRole.UserRole)
            if model.versions:
                result.append((model, model.versions[0]))
        return result

    def _download_selected(self):
        if self._download_running:
            return
        client = root.connection.client_if_connected
        if client is None:
            self._status.setText(_("Not connected to ComfyUI"))
            return
        queue = []
        skipped = 0
        for model, version in self._selected_downloads():
            if self._state_of(model, version) == _STATE_INSTALLED or version.early_access:
                skipped += 1
                continue
            queue.append((model, version))
        if not queue:
            self._status.setText(_("Nothing to download - everything selected is in your library"))
            return
        self._download_running = True
        self._batch_cancelled = False
        self._download_btn.setEnabled(False)
        self._progress.setValue(0)
        self._download_row.setVisible(True)
        eventloop.run(self._run_batch(client, queue, skipped))

    async def _run_batch(self, client, queue: list, skipped: int):
        """Downloads run one after another on purpose: Lora Manager has no server-side
        queue worker (its queue endpoints only record entries in a table, nothing pops
        them), and parallel transfers would just split the same bandwidth."""
        done = failed = 0
        errors: list[str] = []
        for index, (model, version) in enumerate(queue, start=1):
            if self._batch_cancelled:
                break
            self._batch_label = f"({index}/{len(queue)}) {model.name}"
            self._progress.setValue(0)
            self._progress_label.setText(f"{_('Downloading')} {self._batch_label}…")
            result = await self._run_download(client, model, version)
            if result.get("success"):
                done += 1
                self._after_download(client, model, version)
            elif not self._batch_cancelled:
                failed += 1
                errors.append(f"{model.name}: {result.get('error') or _('unknown error')}")
        self._download_running = False
        self._download_row.setVisible(False)
        self._status.setText(self._batch_summary(done, failed, skipped, errors))
        self._update_selection_label()

    def _batch_summary(self, done: int, failed: int, skipped: int, errors: list[str]) -> str:
        parts = [f"{done} {_('downloaded')}"]
        if failed:
            parts.append(f"{failed} {_('failed')}")
        if skipped:
            parts.append(f"{skipped} {_('skipped')}")
        if self._batch_cancelled:
            parts.append(_("cancelled"))
        summary = ", ".join(parts)
        return f"{summary} – {errors[0]}" if errors else summary

    async def _run_download(self, client, model: CivitaiModel, version: CivitaiVersion) -> dict:
        self._download_id = new_download_id()
        download_id = self._download_id
        task = asyncio.ensure_future(
            start_download(
                client._requests,
                client.url,
                model.id,
                version.id,
                download_id=download_id,
                model_root=self._root_combo.currentData() or "",
                relative_path=self._folder_combo.currentText().strip(),
            )
        )
        try:
            while not task.done():
                await asyncio.sleep(1.0)
                if download_id != self._download_id:
                    break  # cancelled - stop polling, the task resolves on its own
                progress = await fetch_download_progress(client._requests, client.url, download_id)
                self._show_progress(progress)
            return await task
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _show_progress(self, progress: dict):
        prefix = (
            f"{_('Downloading')} {self._batch_label}" if self._batch_label else _("Downloading")
        )
        if not progress:
            self._progress_label.setText(f"{prefix}…")
            return
        percent = int(progress.get("progress") or 0)
        self._progress.setValue(percent)
        speed = float(progress.get("bytes_per_second") or 0.0) / (1024 * 1024)
        total = float(progress.get("total_bytes") or 0.0) / (1024 * 1024)
        done = float(progress.get("bytes_downloaded") or 0.0) / (1024 * 1024)
        detail = f"{done:.0f} / {total:.0f} MB" if total else ""
        if speed:
            detail = f"{detail} – {speed:.1f} MB/s" if detail else f"{speed:.1f} MB/s"
        message = progress.get("message")
        self._progress_label.setText(message or f"{prefix} {detail}".strip())

    def _cancel_download(self):
        if not self._download_running:
            return
        client = root.connection.client_if_connected
        self._batch_cancelled = True  # also stops the remaining items of a batch
        download_id, self._download_id = self._download_id, ""
        self._download_row.setVisible(False)
        self._status.setText(_("Cancelling download…"))
        if client is not None and download_id:
            eventloop.run(cancel_download(client._requests, client.url, download_id))

    def _after_download(self, client, model: CivitaiModel, version: CivitaiVersion):
        # the new file exists on the server but neither ComfyUI nor the LoRA browser
        # know about it yet: drop the cached list and trigger a model refresh
        clear_lora_cache(client.url)
        root.connection.refresh()
        # mark it installed right away - the cache we would read that from was just
        # dropped, and rebuilding it costs a full server round trip
        file = version.primary_file
        if file and file.sha256:
            self._installed_hashes.add(file.sha256.upper())
        self._installed_models.add(model.id)
        self._apply_filter()

    def hideEvent(self, a0):
        self._preview_timer.stop()  # nothing to load while the tab is not visible
        super().hideEvent(a0)

    def showEvent(self, a0):
        super().showEvent(a0)
        self._schedule_visible_previews()  # tab came back - fill in what is on screen
