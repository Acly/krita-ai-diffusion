from __future__ import annotations

from krita import Krita
from PyQt5.QtCore import QFile, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..localization import translate as _
from ..model.root import root
from ..settings import settings
from ..text import format_option_group
from ..util import ensure
from ..wildcards import WildcardLibrary
from . import theme

_POS_END = "end"
_POS_START = "start"
_POS_CURSOR = "cursor"
_MODE_NORMAL = "normal"
_MODE_RANDOM = "random"
_MODE_SEQUENTIAL = "sequential"
_MODE_SEQUENTIAL_FILE = "sequential_file"
_SORT_NAME = "name"
_SORT_DATE = "date"


def _move_to_trash(path: str) -> bool:
    # Qt < 5.15 has no trash support - the library then deletes outright. The static
    # PyQt5 overload returns (ok, path_in_trash), a tuple that is always truthy.
    move = getattr(QFile, "moveToTrash", None)
    if move is None:
        return False
    try:
        result = move(path)
    except Exception:
        return False
    return bool(result[0] if isinstance(result, tuple) else result)


class WildcardBrowser(QWidget):
    """Browse file-based wildcards (<user data>/wildcards/*.txt, one option per line,
    __name__ or __folder/name__ in the prompt) and insert a reference tag."""

    wildcard_selected = pyqtSignal(str)  # name

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._library = WildcardLibrary.instance()
        self._current_name: str | None = None
        self._dirty = False
        self._loading_preview = False

        self._search = QLineEdit(self)
        self._search.setPlaceholderText(_("Search wildcards…"))
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)

        self._new_btn = QToolButton(self)
        self._new_btn.setText(_("New…"))
        self._new_btn.setToolTip(_("Create a new wildcard file"))
        self._new_btn.clicked.connect(self._create_wildcard)

        self._reload_btn = QToolButton(self)
        self._reload_btn.setIcon(theme.icon("reset"))
        self._reload_btn.setToolTip(_("Rescan the wildcards folder for new/changed files"))
        self._reload_btn.clicked.connect(self._reload)

        self._folder_btn = QToolButton(self)
        self._folder_btn.setIcon(Krita.instance().icon("document-open"))
        self._folder_btn.setToolTip(_("Open the wildcards folder"))
        self._folder_btn.clicked.connect(self._open_folder)

        sort_label = QLabel(_("Sort:"), self)
        self._sort_combo = QComboBox(self)
        self._sort_combo.addItem(_("Name"), _SORT_NAME)
        self._sort_combo.addItem(_("Date Added"), _SORT_DATE)
        idx = self._sort_combo.findData(settings.wildcard_browser_sort)
        self._sort_combo.setCurrentIndex(max(idx, 0))
        self._sort_combo.currentIndexChanged.connect(self._apply_filter)
        self._sort_combo.currentIndexChanged.connect(self._save_sort_setting)

        row1 = QHBoxLayout()
        row1.addWidget(self._search, 1)
        row1.addWidget(sort_label)
        row1.addWidget(self._sort_combo)
        row1.addWidget(self._new_btn)
        row1.addWidget(self._reload_btn)
        row1.addWidget(self._folder_btn)

        self._list = QListWidget(self)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemDoubleClicked.connect(lambda _item: self._add_to_prompt())

        self._preview_label = QLabel(_("Select a wildcard to preview its content."), self)
        self._preview_label.setStyleSheet(f"color: {theme.grey}; font-style: italic;")

        self._rename_btn = QToolButton(self)
        self._rename_btn.setText(_("Rename…"))
        self._rename_btn.setToolTip(_("Rename or move this wildcard file"))
        self._rename_btn.setEnabled(False)
        self._rename_btn.clicked.connect(self._rename_selected)

        self._delete_btn = QToolButton(self)
        self._delete_btn.setText(_("Delete…"))
        self._delete_btn.setToolTip(_("Move this wildcard file to the trash (Del)"))
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._delete_selected)
        delete_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self._list)
        delete_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        delete_shortcut.activated.connect(self._delete_selected)

        self._save_btn = QToolButton(self)
        self._save_btn.setText(_("Save"))
        self._save_btn.setToolTip(_("Write the edited content back to the file"))
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_preview)

        preview_header = QHBoxLayout()
        preview_header.addWidget(self._preview_label, 1)
        preview_header.addWidget(self._rename_btn)
        preview_header.addWidget(self._delete_btn)
        preview_header.addWidget(self._save_btn)

        self._preview = QPlainTextEdit(self)
        self._preview.setPlaceholderText(_("Select a wildcard to preview and edit its content…"))
        self._preview.setEnabled(False)
        self._preview.textChanged.connect(self._on_preview_edited)
        preview_font = self._preview.font()
        preview_font.setFamily("monospace")
        self._preview.setFont(preview_font)

        preview_layout = QVBoxLayout()
        preview_layout.addLayout(preview_header)
        preview_layout.addWidget(self._preview, 1)
        preview_widget = QWidget(self)
        preview_widget.setLayout(preview_layout)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.addWidget(self._list)
        self._splitter.addWidget(preview_widget)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)

        self._hint = QLabel(self)
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color: {theme.grey}; font-style: italic;")

        mode_label = QLabel(_("Mode:"), self)
        self._mode_combo = QComboBox(self)
        self._mode_combo.addItem(_("Normal __name__"), _MODE_NORMAL)
        self._mode_combo.addItem(_("Random {a|b}"), _MODE_RANDOM)
        self._mode_combo.addItem(_("Sequential (snapshot) [[a|b]]"), _MODE_SEQUENTIAL)
        self._mode_combo.addItem(_("Sequential (live file) __seq:name__"), _MODE_SEQUENTIAL_FILE)
        self._mode_combo.setToolTip(
            _(
                "Normal: live reference to the file, picks one line at random per image.\n"
                "Random: expands the file's lines directly into the prompt as a wildcard group"
                " (a frozen snapshot, not a live reference) - picks randomly per image.\n"
                "Sequential (snapshot): same frozen expansion, but cycles through every line"
                " across the batch instead of picking randomly.\n"
                "Sequential (live file): stays a short live reference to the file (like Normal)"
                " but cycles through its lines across the batch - use the batch-count button"
                " next to the batch slider to size the batch to the file's line count."
            )
        )

        position_label = QLabel(_("Insert:"), self)
        self._position_combo = QComboBox(self)
        self._position_combo.addItem(_("at End"), _POS_END)
        self._position_combo.addItem(_("at Start"), _POS_START)
        self._position_combo.addItem(_("at Cursor"), _POS_CURSOR)

        self._selected_label = QLabel(_("No wildcard selected"), self)
        self._selected_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._add_btn = QPushButton(_("Add to Prompt"), self)
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._add_to_prompt)

        bottom = QHBoxLayout()
        bottom.addWidget(self._selected_label, 1)
        bottom.addWidget(mode_label)
        bottom.addWidget(self._mode_combo)
        bottom.addWidget(position_label)
        bottom.addWidget(self._position_combo)
        bottom.addWidget(self._add_btn)

        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addWidget(self._splitter, 1)
        layout.addWidget(self._hint)
        layout.addLayout(bottom)
        self.setLayout(layout)

        self._reload()

    # ── data ──

    def _reload(self):
        self._library.reload()
        self._apply_filter()

    def _sort_key(self, name: str):
        if self._sort_combo.currentData() == _SORT_DATE:
            path = self._library.path_for(name)
            try:
                return -path.stat().st_ctime if path else 0.0
            except OSError:
                return 0.0
        return name

    def _save_sort_setting(self):
        settings.wildcard_browser_sort = self._sort_combo.currentData()
        settings.save()

    def _apply_filter(self):
        search = self._search.text().strip().lower()
        names = sorted(self._library.names(), key=self._sort_key)
        # rebuilding the list is not a user-driven selection change (e.g. it also runs
        # on every keystroke in the search box) - block signals so it never triggers
        # the unsaved-changes prompt, then restore the previous selection silently
        self._list.blockSignals(True)
        try:
            self._list.clear()
            for name in names:
                if search and search not in name:
                    continue
                options = self._library.get(name) or []
                item = QListWidgetItem(f"{name}  ({len(options)})")
                item.setData(Qt.ItemDataRole.UserRole, name)
                preview = "\n".join(options[:8])
                more = f"\n… +{len(options) - 8} more" if len(options) > 8 else ""
                item.setToolTip(f"__{name}__\n\n{preview}{more}")
                self._list.addItem(item)
            self._select_item_by_name(self._current_name)
        finally:
            self._list.blockSignals(False)

        if not names:
            self._hint.setText(
                _("No wildcard files found. Put .txt files (one option per line) in:")
                + f"\n{self._library.folder}"
            )
        else:
            self._hint.setText(f"{self._list.count()} / {len(names)} " + _("wildcards"))

    def _open_folder(self):
        self._library.folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._library.folder)))

    # ── selection / insertion ──

    def _selected_name(self) -> str | None:
        items = self._list.selectedItems()
        return items[0].data(Qt.ItemDataRole.UserRole) if items else None

    def _on_selection_changed(self):
        name = self._selected_name()
        if name == self._current_name:
            return  # no actual change (e.g. reverted below) - keep any unsaved edits as-is
        if self._dirty and not self._confirm_discard():
            self._select_item_by_name(self._current_name)  # revert, re-enters this method
            return
        self._dirty = False
        self._current_name = name
        self._add_btn.setEnabled(name is not None)
        self._selected_label.setText(f"__{name}__" if name else _("No wildcard selected"))
        self._update_preview(name)

    def _update_preview(self, name: str | None):
        self._loading_preview = True
        try:
            self._preview.setEnabled(name is not None)
            self._rename_btn.setEnabled(name is not None)
            self._delete_btn.setEnabled(name is not None)
            self._save_btn.setEnabled(False)
            if name is None:
                self._preview_label.setText(_("Select a wildcard to preview its content."))
                self._preview.setPlainText("")
                return
            options = self._library.get(name) or []
            self._preview_label.setText(f"__{name}__  ({len(options)})")
            self._preview.setPlainText("\n".join(options))
        finally:
            self._loading_preview = False

    def _on_preview_edited(self):
        if self._loading_preview:
            return
        self._dirty = True
        self._save_btn.setEnabled(True)
        if self._current_name is not None:
            self._preview_label.setText(f"__{self._current_name}__ *")

    def _confirm_discard(self) -> bool:
        result = QMessageBox.question(
            self,
            _("Unsaved Changes"),
            _("Discard unsaved changes to") + f" __{self._current_name}__?",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return result == QMessageBox.StandardButton.Discard

    def _select_item_by_name(self, name: str | None):
        for i in range(self._list.count()):
            item = ensure(self._list.item(i))
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == name:
                self._list.setCurrentItem(item)
                return
        self._list.clearSelection()

    def _save_preview(self):
        name = self._current_name
        if name is None:
            return
        text = self._preview.toPlainText()
        lines = text.splitlines()
        if not text.strip():
            QMessageBox.warning(
                self, _("Save Failed"), _("A wildcard file needs at least one option line.")
            )
            return
        if self._library.save(name, lines):
            self._dirty = False
            self._apply_filter()  # refreshes the item's line count in the list
            self._update_preview(name)  # reflects the file as re-parsed from disk
        else:
            QMessageBox.warning(
                self,
                _("Save Failed"),
                _("Could not write the wildcard file. See the log for details."),
            )

    def _create_wildcard(self):
        name, ok = QInputDialog.getText(
            self, _("New Wildcard"), _("File name (may include a folder/ prefix):")
        )
        if not ok or not name.strip():
            return
        if self._library.create(name, [_("option 1"), _("option 2")]):
            self._apply_filter()
            self._select_item_by_name(name.strip().strip("/\\").lower())
            self._preview.setFocus()
        else:
            QMessageBox.warning(
                self,
                _("Create Failed"),
                _("Could not create the wildcard file - a file with that name may already exist."),
            )

    def _rename_selected(self):
        name = self._current_name
        if name is None:
            return
        if self._dirty:
            QMessageBox.information(
                self,
                _("Unsaved Changes"),
                _("Save or discard the edited content before renaming this file."),
            )
            return
        new_name, ok = QInputDialog.getText(
            self,
            _("Rename Wildcard"),
            _("New name (may include a folder/ prefix to move the file):"),
            QLineEdit.EchoMode.Normal,
            name,
        )
        if not ok or not new_name.strip() or new_name.strip().lower() == name:
            return
        if self._library.rename(name, new_name):
            self._apply_filter()
            self._select_item_by_name(new_name.strip().strip("/\\").lower())
        else:
            QMessageBox.warning(
                self,
                _("Rename Failed"),
                _("Could not rename the file - the target name may already exist or be invalid."),
            )

    def _delete_selected(self):
        name = self._current_name
        if name is None:
            return
        count = len(self._library.get(name) or [])
        answer = QMessageBox.question(
            self,
            _("Delete Wildcard"),
            _("Move") + f" __{name}__ ({count} " + _("lines") + ") " + _("to the trash?"),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self._library.delete(name, _move_to_trash):
            self._dirty = False  # unsaved edits of a deleted file have nothing to go to
            self._current_name = None
            self._apply_filter()
            self._add_btn.setEnabled(False)
            self._selected_label.setText(_("No wildcard selected"))
            self._update_preview(None)
        else:
            QMessageBox.warning(
                self,
                _("Delete Failed"),
                _("Could not delete the wildcard file. See the log for details."),
            )

    def may_close(self) -> bool:
        """False when unsaved edits should keep the window open."""
        return not self._dirty or self._confirm_discard()

    def select_wildcard(self, name: str):
        """Show a file that was just created elsewhere - the generator tab uses this."""
        self._reload()
        self._select_item_by_name(name)

    def _build_tag(self, name: str) -> str:
        mode = self._mode_combo.currentData()
        if mode == _MODE_NORMAL:
            return f"__{name}__"
        if mode == _MODE_SEQUENTIAL_FILE:
            return f"__seq:{name}__"
        options = self._library.get(name) or []
        return format_option_group(options, mode == _MODE_SEQUENTIAL)

    def _add_to_prompt(self):
        name = self._selected_name()
        model = root.active_model
        if name is None or model is None:
            return
        tag = self._build_tag(name)
        self.wildcard_selected.emit(name)

        position = self._position_combo.currentData()
        if position == _POS_CURSOR and self._insert_at_cursor(tag):
            return

        region = model.regions.active_or_root
        current = region.positive
        if position == _POS_START:
            region.positive = tag + "\n" + current.lstrip("\n")
        else:
            region.positive = current.rstrip("\n") + "\n" + tag

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


class WildcardPickerDialog(QDialog):
    """The wildcard library and the generator that fills it, as two tabs.

    Generating a term list produces a file for the library, so it belongs next to it
    rather than in the prompt batch dialog, where it had to switch off half the form
    it was sitting in.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        from .wildcard_generate import WildcardGenerator

        self.setWindowTitle(_("Wildcards"))
        self.setMinimumSize(480, 360)
        self.resize(620, 480)
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        self.library = WildcardBrowser(self)
        self.generator = WildcardGenerator(self)
        self.generator.wildcard_created.connect(self._on_wildcard_created)

        self._tabs = QTabWidget(self)
        self._tabs.addTab(self.library, _("Library"))
        self._tabs.addTab(self.generator, _("Generate"))

        close_btn = QPushButton(_("Close"), self)
        close_btn.clicked.connect(self.close)  # type: ignore (slot returns bool)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(close_btn)

        layout = QVBoxLayout()
        layout.addWidget(self._tabs, 1)
        layout.addLayout(bottom)
        self.setLayout(layout)

    def _on_wildcard_created(self, name: str):
        # switch to the new file rather than leaving it invisible behind the tab
        self.library.select_wildcard(name)
        self._tabs.setCurrentWidget(self.library)

    def closeEvent(self, a0):
        if not self.library.may_close():
            ensure(a0).ignore()
            return
        self.generator.shutdown()
        super().closeEvent(a0)
