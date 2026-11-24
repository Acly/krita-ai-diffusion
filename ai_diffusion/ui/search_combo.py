"""An editable choice list with provisional, case-insensitive substring search."""

from PyQt6.QtCore import QEvent, QModelIndex, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QComboBox, QCompleter, QWidget


class SearchableComboBox(QComboBox):
    selection_committed = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._searching = False
        self._completion_guard = False
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        # Attach directly to the editor: QComboBox's text-based completion would
        # otherwise select the first duplicate label instead of its filename.
        self.setCompleter(None)
        editor = self.lineEdit()
        assert editor is not None
        self._editor = editor
        model = self.model()
        assert model is not None
        self._completion = QCompleter(model, self)
        popup = self._completion.popup()
        assert popup is not None
        self._popup = popup
        self._completion.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completion.setFilterMode(Qt.MatchFlag.MatchContains)
        self._completion.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._editor.setCompleter(self._completion)
        self._editor.setClearButtonEnabled(True)
        self._editor.textEdited.connect(self._search_edited)
        self.installEventFilter(self)
        self._editor.installEventFilter(self)
        self._popup.installEventFilter(self)
        self._completion.activated[QModelIndex].connect(self._commit_completion)
        self.activated.connect(self._commit_dropdown)
        app = QApplication.instance()
        assert isinstance(app, QApplication)
        app.focusChanged.connect(self._focus_changed)

    def _focus_changed(self, previous: QWidget | None, current: QWidget | None):
        if not self._searching or previous not in (self, self._editor):
            return
        popup = self._popup
        if current in (self, self._editor, popup) or (current and popup.isAncestorOf(current)):
            return
        if current is not None or not popup.isVisible():
            QTimer.singleShot(0, self.cancel_search)

    def _search_edited(self, text: str):
        self._searching = True

    def _commit_dropdown(self, index: int):
        if not self._searching and not self._completion_guard and self.itemData(index) is not None:
            self.selection_committed.emit(index)

    def _commit_completion(self, index: QModelIndex):
        key = index.data(Qt.ItemDataRole.UserRole)
        row = self.findData(key) if key is not None else -1
        if row < 0:
            return
        self._completion_guard = True
        self.setCurrentIndex(row)
        self.setEditText(self.itemText(row))
        self._searching = False
        self._popup.hide()
        self.selection_committed.emit(row)
        QTimer.singleShot(0, self._release_completion_guard)

    def _release_completion_guard(self):
        self._completion_guard = False
        # QCompleter updates the editor after activated slots return. A slot can
        # rebuild/reorder recent items, so restore the current item's final label.
        if not self._searching:
            self.setEditText(self.itemText(self.currentIndex()))

    def cancel_search(self):
        self._popup.hide()
        self.setEditText(self.itemText(self.currentIndex()))
        self._searching = False

    def showPopup(self):
        self.cancel_search()
        super().showPopup()

    def eventFilter(self, a0: QObject | None, a1: QEvent | None):
        watched, event = a0, a1
        if event is None:
            return False
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up) and self._searching:
                completion = self._completion
                model = completion.completionModel()
                if model is not None and model.rowCount():
                    completion.complete()
                    popup = self._popup
                    row = popup.currentIndex().row()
                    step = 1 if event.key() == Qt.Key.Key_Down else -1
                    row = (
                        (row + step) % model.rowCount()
                        if row >= 0
                        else (0 if step > 0 else model.rowCount() - 1)
                    )
                    popup.setCurrentIndex(model.index(row, 0))
                return True
            if event.key() == Qt.Key.Key_Escape and self._searching:
                self.cancel_search()
                return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._searching:
                completion = self._completion
                index = self._popup.currentIndex()
                if not index.isValid():
                    model = completion.completionModel()
                    index = model.index(0, 0) if model is not None else QModelIndex()
                if index.isValid():
                    self._commit_completion(index)
                else:
                    self.cancel_search()
                return True
        elif watched is self or watched is self._editor:
            if event.type() == QEvent.Type.FocusIn:
                QTimer.singleShot(0, self._editor.selectAll)
        return super().eventFilter(watched, event)
