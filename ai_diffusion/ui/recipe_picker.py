from __future__ import annotations

from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..backend.lora_manager import RecipeInfo, fetch_preview_bytes, fetch_recipes
from ..localization import translate as _
from ..model.root import root
from ..settings import settings
from ..util import ensure
from . import theme
from .lora_picker import _visible_range

_PREVIEW_SIZE_DEFAULT = 128
_PREVIEW_SIZE_MIN = 64
_PREVIEW_SIZE_MAX = 256
_ARCH_ANY = "__any__"
_SORT_NAME = "name"
_SORT_DATE = "date"


class RecipePickerDialog(QDialog):
    recipe_applied = pyqtSignal(str)  # recipe id

    def __init__(self, current_arch: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._current_arch = current_arch
        self._all_recipes: list[RecipeInfo] = []
        self._filtered: list[RecipeInfo] = []
        self._preview_cache: dict[str, QPixmap] = {}  # original, unscaled
        self._pending_previews: set[str] = set()
        self._loading = False
        self._preview_size = settings.recipe_browser_size

        self.setWindowTitle(_("Recipe Browser"))
        self.setMinimumSize(640, 480)
        self.resize(900, 640)
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        # ── row 1: search ──
        self._search = QLineEdit(self)
        self._search.setPlaceholderText(_("Search recipes…"))
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)

        self._refresh_btn = QToolButton(self)
        self._refresh_btn.setIcon(theme.icon("reset"))
        self._refresh_btn.setToolTip(_("Reload recipe list from server"))
        self._refresh_btn.clicked.connect(self._load_recipes)

        row1 = QHBoxLayout()
        row1.addWidget(self._search, 1)
        row1.addWidget(self._refresh_btn)

        # ── row 2: filters ──
        arch_label = QLabel(_("Base Model:"), self)
        self._arch_combo = QComboBox(self)
        self._arch_combo.addItem(_("Any"), _ARCH_ANY)
        self._arch_combo.currentIndexChanged.connect(self._apply_filter)

        self._favorites_only = QCheckBox(_("Favorites"), self)
        self._favorites_only.toggled.connect(self._apply_filter)

        sort_label = QLabel(_("Sort:"), self)
        self._sort_combo = QComboBox(self)
        self._sort_combo.addItem(_("Name"), _SORT_NAME)
        self._sort_combo.addItem(_("Date Added"), _SORT_DATE)
        idx = self._sort_combo.findData(settings.recipe_browser_sort)
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
        row2.addWidget(self._favorites_only)
        row2.addWidget(sort_label)
        row2.addWidget(self._sort_combo)
        row2.addWidget(size_label)
        row2.addWidget(self._size_slider)

        # ── grid ──
        self._grid = QListWidget(self)
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(QSize(self._preview_size, self._preview_size))
        self._grid.setGridSize(QSize(self._preview_size + 16, self._preview_size + 40))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setMovement(QListWidget.Movement.Static)
        self._grid.setWordWrap(True)
        self._grid.setSpacing(4)
        self._grid.itemSelectionChanged.connect(self._on_selection_changed)
        self._grid.itemDoubleClicked.connect(lambda item: self._apply_recipe(replace=True))

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._load_visible_previews)
        ensure(self._grid.verticalScrollBar()).valueChanged.connect(self._schedule_visible_previews)

        # ── bottom bar ──
        self._selected_label = QLabel(_("No recipe selected"), self)
        self._selected_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._selected_label.setWordWrap(True)

        self._add_btn = QPushButton(_("Add to Prompt"), self)
        self._add_btn.setEnabled(False)
        self._add_btn.setToolTip(
            _("Append the recipe's prompt and LoRAs to the current prompt, on a new line")
        )
        self._add_btn.clicked.connect(lambda: self._apply_recipe(replace=False))

        self._replace_btn = QPushButton(_("Replace Prompt"), self)
        self._replace_btn.setEnabled(False)
        self._replace_btn.setToolTip(
            _("Replace the current prompt and negative prompt with the recipe's")
        )
        self._replace_btn.clicked.connect(lambda: self._apply_recipe(replace=True))

        close_btn = QPushButton(_("Close"), self)
        close_btn.clicked.connect(self.close)  # type: ignore (slot returns bool)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self._selected_label, 1)
        bottom_layout.addWidget(self._add_btn)
        bottom_layout.addWidget(self._replace_btn)
        bottom_layout.addWidget(close_btn)

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

        self._load_recipes()

    # ── data loading ──

    def _load_recipes(self):
        if self._loading:
            return
        client = root.connection.client_if_connected
        if client is None:
            self._status.setText(_("Not connected to ComfyUI"))
            return
        self._status.setText(_("Loading…"))
        self._grid.clear()
        self._all_recipes = []
        self._loading = True
        eventloop.run(self._fetch(client))

    async def _fetch(self, client):
        try:
            self._all_recipes = await fetch_recipes(client._requests, client.url)
        finally:
            self._loading = False
        if not self._all_recipes:
            self._status.setText(_("No recipes found (requires ComfyUI-Lora-Manager)"))
            return
        self._rebuild_filters()
        self._apply_filter()

    # ── filtering ──

    def _rebuild_filters(self):
        base_models = sorted({r.base_model for r in self._all_recipes if r.base_model})
        current = self._arch_combo.currentData()
        self._arch_combo.blockSignals(True)
        self._arch_combo.clear()
        self._arch_combo.addItem(_("Any"), _ARCH_ANY)
        for bm in base_models:
            self._arch_combo.addItem(bm, bm)
        idx = self._arch_combo.findData(current)
        self._arch_combo.setCurrentIndex(max(idx, 0))
        self._arch_combo.blockSignals(False)

    def _apply_filter(self):
        search = self._search.text().strip().lower()
        base_model = self._arch_combo.currentData()
        base_model = "" if base_model in (None, _ARCH_ANY) else base_model
        favorites_only = self._favorites_only.isChecked()

        def matches(r: RecipeInfo) -> bool:
            if favorites_only and not r.favorite:
                return False
            if base_model and r.base_model != base_model:
                return False
            if search:
                haystack = (
                    r.title + " " + r.prompt + " " + " ".join(l.name for l in r.loras)
                ).lower()
                if search not in haystack:
                    return False
            return True

        self._filtered = [r for r in self._all_recipes if matches(r)]
        if self._sort_combo.currentData() == _SORT_DATE:
            self._filtered.sort(key=lambda r: r.created_date, reverse=True)
        else:
            self._filtered.sort(key=lambda r: (r.title or r.id).lower())
        self._populate_grid()

    def _populate_grid(self):
        self._grid.clear()
        cell_size = self._grid.gridSize()
        for recipe in self._filtered:
            fav = "★ " if recipe.favorite else ""
            item = QListWidgetItem(recipe.title or recipe.id)
            item.setSizeHint(cell_size)
            item.setData(Qt.ItemDataRole.UserRole, recipe)
            loras_text = "\n".join(
                f"  {l.name} ({l.strength})" + ("" if l.available else " [missing]")
                for l in recipe.loras
            )
            item.setToolTip(
                f"{fav}{recipe.title}\nBase: {recipe.base_model or '?'}\n"
                f"Checkpoint: {recipe.checkpoint or '?'}\nLoRAs:\n{loras_text}"
            )
            if recipe.id in self._preview_cache:
                item.setIcon(self._scaled_icon(recipe.id))
            self._grid.addItem(item)
        if not self._loading:
            self._status.setText(f"{len(self._filtered)} / {len(self._all_recipes)} recipes")
        self._schedule_visible_previews()

    def _scaled_icon(self, recipe_id: str) -> QIcon:
        pixmap = self._preview_cache[recipe_id]
        scaled = pixmap.scaled(
            self._preview_size,
            self._preview_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QIcon(scaled)

    def _on_preview_size_changed(self, value: int):
        self._preview_size = value
        self._grid.setIconSize(QSize(value, value))
        self._grid.setGridSize(QSize(value + 16, value + 40))
        for i in range(self._grid.count()):
            item = ensure(self._grid.item(i))
            recipe: RecipeInfo = item.data(Qt.ItemDataRole.UserRole)
            item.setSizeHint(self._grid.gridSize())
            if recipe.id in self._preview_cache:
                item.setIcon(self._scaled_icon(recipe.id))
        self._schedule_visible_previews()

    def _save_size_setting(self):
        settings.recipe_browser_size = self._preview_size
        settings.save()

    def _save_sort_setting(self):
        settings.recipe_browser_sort = self._sort_combo.currentData()
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
            rect = self._grid.visualItemRect(item)
            if not rect.intersects(viewport_rect):
                continue
            recipe: RecipeInfo = item.data(Qt.ItemDataRole.UserRole)
            if not recipe.preview_url or recipe.id in self._preview_cache:
                continue
            if recipe.id in self._pending_previews:
                continue
            self._pending_previews.add(recipe.id)
            eventloop.run(self._load_preview(recipe, item))

    async def _load_preview(self, recipe: RecipeInfo, item: QListWidgetItem):
        client = root.connection.client_if_connected
        if client is None:
            return
        data = await fetch_preview_bytes(client._requests, recipe.preview_url)
        if data:
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            if not pixmap.isNull():
                self._preview_cache[recipe.id] = pixmap
                item.setIcon(self._scaled_icon(recipe.id))

    # ── selection / apply ──

    def _selected_recipe(self) -> RecipeInfo | None:
        items = self._grid.selectedItems()
        if items:
            return items[0].data(Qt.ItemDataRole.UserRole)
        return None

    def _on_selection_changed(self):
        recipe = self._selected_recipe()
        if recipe is None:
            self._selected_label.setText(_("No recipe selected"))
            self._add_btn.setEnabled(False)
            self._replace_btn.setEnabled(False)
            return
        missing = [l.name for l in recipe.loras if not l.available]
        text = f"{recipe.title}  [{recipe.base_model or '?'}]  ({len(recipe.loras)} LoRAs)"
        if missing:
            text += "  ⚠ " + _("missing:") + " " + ", ".join(missing)
        self._selected_label.setText(text)
        self._add_btn.setEnabled(True)
        self._replace_btn.setEnabled(True)

    def _apply_recipe(self, replace: bool):
        recipe = self._selected_recipe()
        if recipe is None:
            return
        model = root.active_model
        if model is None:
            return
        prompt = recipe.prompt
        # append tags for available LoRAs that the prompt text doesn't already contain
        extra_tags = [
            f"<lora:{l.name}:{l.strength:g}>"
            for l in recipe.loras
            if l.available and f"<lora:{l.name}:" not in prompt
        ]
        if extra_tags:
            prompt = prompt.rstrip("\n") + "\n" + " ".join(extra_tags)

        region = model.regions
        if replace:
            region.positive = prompt
            if recipe.negative_prompt:
                region.negative = recipe.negative_prompt
        else:
            current = region.positive
            region.positive = current.rstrip("\n") + "\n" + prompt if current.strip() else prompt
            if recipe.negative_prompt:
                current_neg = region.negative
                region.negative = (
                    current_neg.rstrip("\n") + "\n" + recipe.negative_prompt
                    if current_neg.strip()
                    else recipe.negative_prompt
                )
        self.recipe_applied.emit(recipe.id)
        self.close()
