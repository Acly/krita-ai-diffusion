from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
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
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..localization import translate as _
from ..model.root import root
from ..prompt_library import PromptEntry, PromptLibrary
from ..settings import settings
from ..util import ensure
from . import theme

_POS_END = "end"
_POS_START = "start"
_POS_CURSOR = "cursor"
_POS_REPLACE = "replace"
_SORT_NAME = "name"
_SORT_CATEGORY = "category"
_SORT_RECENT = "recent"
_SORT_CREATED = "created"
_CATEGORY_ANY = "__any__"
_CATEGORY_NONE = "__none__"


class PromptBrowser(QWidget):
    """Browse a local library of saved prompt snippets and insert one into the active
    region's prompt - a plain local text library, unlike Recipe (checkpoint + LoRA
    stack + prompt fetched from the ComfyUI-Lora-Manager server)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._library = PromptLibrary.instance()
        self._current_id: str | None = None
        self._dirty = False
        self._loading = False

        self._search = QLineEdit(self)
        self._search.setPlaceholderText(_("Search prompts…"))
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)

        self._category_combo = QComboBox(self)
        self._category_combo.currentIndexChanged.connect(self._apply_filter)

        self._favorites_check = QCheckBox(_("Favorites"), self)
        self._favorites_check.toggled.connect(self._apply_filter)

        sort_label = QLabel(_("Sort:"), self)
        self._sort_combo = QComboBox(self)
        self._sort_combo.addItem(_("Name"), _SORT_NAME)
        self._sort_combo.addItem(_("Category"), _SORT_CATEGORY)
        self._sort_combo.addItem(_("Recently Used"), _SORT_RECENT)
        self._sort_combo.addItem(_("Date Added"), _SORT_CREATED)
        idx = self._sort_combo.findData(settings.prompt_browser_sort)
        self._sort_combo.setCurrentIndex(max(idx, 0))
        self._sort_combo.currentIndexChanged.connect(self._apply_filter)
        self._sort_combo.currentIndexChanged.connect(self._save_sort_setting)

        self._new_btn = QToolButton(self)
        self._new_btn.setText(_("New…"))
        self._new_btn.setToolTip(_("Add a new saved prompt"))
        self._new_btn.clicked.connect(self._create_entry)

        row1 = QHBoxLayout()
        row1.addWidget(self._search, 1)
        row1.addWidget(self._category_combo)
        row1.addWidget(self._favorites_check)
        row1.addWidget(sort_label)
        row1.addWidget(self._sort_combo)
        row1.addWidget(self._new_btn)

        self._list = QListWidget(self)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setIconSize(QSize(32, 32))
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemDoubleClicked.connect(lambda _item: self._insert_to_prompt())

        self._title_edit = QLineEdit(self)
        self._title_edit.setPlaceholderText(_("Title"))
        self._title_edit.textChanged.connect(self._on_edited)

        self._category_edit = QComboBox(self)
        self._category_edit.setEditable(True)
        ensure(self._category_edit.lineEdit()).setPlaceholderText(_("Category (optional)"))
        self._category_edit.editTextChanged.connect(self._on_edited)

        self._favorite_check = QCheckBox(_("Favorite"), self)
        self._favorite_check.toggled.connect(self._on_edited)

        self._preview_label = QLabel(self)
        self._preview_label.setFixedSize(64, 64)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet(f"border: 1px solid {theme.grey};")

        header = QHBoxLayout()
        header.addWidget(self._preview_label)
        fields = QVBoxLayout()
        fields.addWidget(self._title_edit)
        fields.addWidget(self._category_edit)
        header.addLayout(fields, 1)
        header.addWidget(self._favorite_check)

        self._text_edit = QPlainTextEdit(self)
        self._text_edit.setPlaceholderText(_("Prompt text"))
        self._text_edit.textChanged.connect(self._on_edited)

        self._negative_edit = QPlainTextEdit(self)
        self._negative_edit.setPlaceholderText(_("Negative prompt (optional)"))
        self._negative_edit.setFixedHeight(60)
        self._negative_edit.textChanged.connect(self._on_edited)

        self._save_btn = QPushButton(_("Save"), self)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_entry)

        self._delete_btn = QPushButton(_("Delete"), self)
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._delete_entry)

        edit_buttons = QHBoxLayout()
        edit_buttons.addStretch(1)
        edit_buttons.addWidget(self._save_btn)
        edit_buttons.addWidget(self._delete_btn)

        edit_layout = QVBoxLayout()
        edit_layout.addLayout(header)
        edit_layout.addWidget(self._text_edit, 1)
        edit_layout.addWidget(self._negative_edit)
        edit_layout.addLayout(edit_buttons)
        edit_widget = QWidget(self)
        edit_widget.setLayout(edit_layout)
        edit_widget.setEnabled(False)
        self._edit_widget = edit_widget

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.addWidget(self._list)
        self._splitter.addWidget(edit_widget)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)

        self._hint = QLabel(self)
        self._hint.setStyleSheet(f"color: {theme.grey}; font-style: italic;")

        position_label = QLabel(_("Insert:"), self)
        self._position_combo = QComboBox(self)
        self._position_combo.addItem(_("at Cursor"), _POS_CURSOR)
        self._position_combo.addItem(_("at End"), _POS_END)
        self._position_combo.addItem(_("at Start"), _POS_START)
        self._position_combo.addItem(_("Replace Prompt"), _POS_REPLACE)

        self._selected_label = QLabel(_("No prompt selected"), self)
        self._selected_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._insert_btn = QPushButton(_("Insert to Prompt"), self)
        self._insert_btn.setEnabled(False)
        self._insert_btn.clicked.connect(self._insert_to_prompt)

        self._save_current_btn = QPushButton(_("Save Current Prompt…"), self)
        self._save_current_btn.setToolTip(
            _("Save the active region's current prompt text as a new library entry")
        )
        self._save_current_btn.clicked.connect(self._save_current_prompt)

        bottom = QHBoxLayout()
        bottom.addWidget(self._selected_label, 1)
        bottom.addWidget(position_label)
        bottom.addWidget(self._position_combo)
        bottom.addWidget(self._insert_btn)

        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addWidget(self._splitter, 1)
        layout.addWidget(self._hint)
        layout.addLayout(bottom)
        layout.addWidget(self._save_current_btn)
        self.setLayout(layout)

        self._library.changed.connect(self._reload)
        self._reload()

    # -- data --

    def _reload(self):
        self._refresh_categories()
        self._apply_filter()

    def _refresh_categories(self):
        current = self._category_combo.currentData()
        self._category_combo.blockSignals(True)
        self._category_combo.clear()
        self._category_combo.addItem(_("All Categories"), _CATEGORY_ANY)
        self._category_combo.addItem(_("No Category"), _CATEGORY_NONE)
        for category in self._library.categories():
            self._category_combo.addItem(category, category)
        idx = self._category_combo.findData(current)
        self._category_combo.setCurrentIndex(max(idx, 0))
        self._category_combo.blockSignals(False)

        # the edit-panel category field: same list, but editable to allow a new one
        current_text = self._category_edit.currentText()
        self._category_edit.blockSignals(True)
        self._category_edit.clear()
        self._category_edit.addItem("")
        self._category_edit.addItems(self._library.categories())
        self._category_edit.setCurrentText(current_text)
        self._category_edit.blockSignals(False)

    def _sort_key(self, entry: PromptEntry):
        sort = self._sort_combo.currentData()
        if sort == _SORT_CATEGORY:
            return (entry.category.lower(), entry.title.lower())
        if sort == _SORT_RECENT:
            return -entry.last_used
        if sort == _SORT_CREATED:
            return -entry.created
        return entry.title.lower()

    def _save_sort_setting(self):
        settings.prompt_browser_sort = self._sort_combo.currentData()
        settings.save()

    def _apply_filter(self):
        search = self._search.text().strip().lower()
        category = self._category_combo.currentData()
        favorites_only = self._favorites_check.isChecked()
        entries = sorted(self._library.entries(), key=self._sort_key)

        self._list.blockSignals(True)
        try:
            self._list.clear()
            for entry in entries:
                if favorites_only and not entry.favorite:
                    continue
                if category == _CATEGORY_NONE and entry.category:
                    continue
                if category not in (_CATEGORY_ANY, _CATEGORY_NONE) and entry.category != category:
                    continue
                haystack = f"{entry.title} {entry.text} {entry.category}".lower()
                if search and search not in haystack:
                    continue
                label = f"★ {entry.title}" if entry.favorite else entry.title
                if entry.category:
                    label += f"  [{entry.category}]"
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, entry.id)
                preview = entry.text[:200]
                item.setToolTip(preview)
                if self._library.has_preview(entry.id):
                    pixmap = QPixmap(str(self._library.preview_path(entry.id)))
                    if not pixmap.isNull():
                        item.setIcon(QIcon(pixmap))
                self._list.addItem(item)
            self._select_item_by_id(self._current_id)
        finally:
            self._list.blockSignals(False)

        total = len(self._library.entries())
        if total == 0:
            self._hint.setText(_("No saved prompts yet. Use 'New…' or 'Save Current Prompt…'."))
        else:
            self._hint.setText(f"{self._list.count()} / {total} " + _("prompts"))

    # -- selection / editing --

    def _selected_id(self) -> str | None:
        items = self._list.selectedItems()
        return items[0].data(Qt.ItemDataRole.UserRole) if items else None

    def _select_item_by_id(self, id: str | None):
        for i in range(self._list.count()):
            item = ensure(self._list.item(i))
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == id:
                self._list.setCurrentItem(item)
                return
        self._list.clearSelection()

    def _on_selection_changed(self):
        id = self._selected_id()
        if id == self._current_id:
            return
        if self._dirty and not self._confirm_discard():
            self._select_item_by_id(self._current_id)
            return
        self._current_id = id
        self._dirty = False
        self._insert_btn.setEnabled(id is not None)
        self._delete_btn.setEnabled(id is not None)
        entry = self._library.get(id) if id else None
        self._selected_label.setText(entry.title if entry else _("No prompt selected"))
        self._load_entry(entry)

    def _load_entry(self, entry: PromptEntry | None):
        self._loading = True
        try:
            self._edit_widget.setEnabled(entry is not None)
            self._save_btn.setEnabled(False)
            if entry is None:
                self._title_edit.setText("")
                self._category_edit.setCurrentText("")
                self._text_edit.setPlainText("")
                self._negative_edit.setPlainText("")
                self._favorite_check.setChecked(False)
                self._preview_label.clear()
                return
            self._title_edit.setText(entry.title)
            self._category_edit.setCurrentText(entry.category)
            self._text_edit.setPlainText(entry.text)
            self._negative_edit.setPlainText(entry.negative)
            self._favorite_check.setChecked(entry.favorite)
            self._update_preview_label(entry.id)
        finally:
            self._loading = False

    def _update_preview_label(self, id: str):
        if self._library.has_preview(id):
            pixmap = QPixmap(str(self._library.preview_path(id)))
            if not pixmap.isNull():
                self._preview_label.setPixmap(
                    pixmap.scaled(
                        64,
                        64,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                return
        self._preview_label.clear()

    def _on_edited(self):
        if self._loading:
            return
        self._dirty = True
        self._save_btn.setEnabled(True)

    def _confirm_discard(self) -> bool:
        result = QMessageBox.question(
            self,
            _("Unsaved Changes"),
            _("Discard unsaved changes to this prompt?"),
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return result == QMessageBox.StandardButton.Discard

    def _save_entry(self):
        if self._current_id is None:
            return
        title = self._title_edit.text().strip()
        if not title:
            QMessageBox.warning(self, _("Save Failed"), _("A saved prompt needs a title."))
            return
        self._library.update(
            self._current_id,
            title=title,
            category=self._category_edit.currentText().strip(),
            text=self._text_edit.toPlainText(),
            negative=self._negative_edit.toPlainText(),
            favorite=self._favorite_check.isChecked(),
        )
        self._dirty = False
        self._save_btn.setEnabled(False)

    def _delete_entry(self):
        if self._current_id is None:
            return
        entry = self._library.get(self._current_id)
        title = entry.title if entry else ""
        if (
            QMessageBox.question(
                self,
                _("Delete Prompt"),
                _("Delete") + f' "{title}"?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._library.remove(self._current_id)
        self._current_id = None
        self._dirty = False
        # the reload triggered by remove() above reselects nothing under blockSignals,
        # so _on_selection_changed never runs - clear the form explicitly
        self._insert_btn.setEnabled(False)
        self._delete_btn.setEnabled(False)
        self._selected_label.setText(_("No prompt selected"))
        self._load_entry(None)

    def _create_entry(self):
        entry = self._library.add(_("New Prompt"), "")
        self._apply_filter()
        self._select_item_by_id(entry.id)
        self._title_edit.setFocus()
        self._title_edit.selectAll()

    def _save_current_prompt(self):
        model = root.active_model
        if model is None:
            return
        region = model.regions.active_or_root
        if not region.positive.strip():
            QMessageBox.information(self, _("Nothing to Save"), _("The active prompt is empty."))
            return
        entry = self._library.add(_("New Prompt"), region.positive, negative=model.regions.negative)
        self._apply_filter()
        self._select_item_by_id(entry.id)
        self._title_edit.setFocus()
        self._title_edit.selectAll()

    def create_from_image(self, text: str, negative: str, image) -> PromptEntry:
        """Called from the history panel's "Save as Prompt" - text/negative/preview
        arrive pre-filled, only the title is left for the user to type over."""
        entry = self._library.add(_("New Prompt"), text, negative=negative)
        self._library.save_preview(entry.id, image)
        self._apply_filter()
        self._select_item_by_id(entry.id)
        self._title_edit.setFocus()
        self._title_edit.selectAll()
        return entry

    # -- insertion --

    def _insert_to_prompt(self):
        id = self._selected_id()
        entry = self._library.get(id) if id else None
        model = root.active_model
        if entry is None or model is None:
            return
        self._library.mark_used(entry.id)

        position = self._position_combo.currentData()
        if position == _POS_CURSOR and self._insert_at_cursor(entry.text):
            return

        region = model.regions.active_or_root
        current = region.positive
        if position == _POS_REPLACE:
            region.positive = entry.text
        elif position == _POS_START:
            region.positive = entry.text + "\n" + current.lstrip("\n")
        else:
            region.positive = (
                current.rstrip("\n") + "\n" + entry.text if current.strip() else entry.text
            )

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


class PromptPickerDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(_("Prompts"))
        self.setMinimumSize(480, 360)
        self.resize(620, 440)
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        self.browser = PromptBrowser(self)

        close_btn = QPushButton(_("Close"), self)
        close_btn.clicked.connect(self.close)  # type: ignore (slot returns bool)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(close_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.browser, 1)
        layout.addLayout(bottom)
        self.setLayout(layout)

    def closeEvent(self, a0):
        if self.browser._dirty and not self.browser._confirm_discard():
            ensure(a0).ignore()
            return
        super().closeEvent(a0)
