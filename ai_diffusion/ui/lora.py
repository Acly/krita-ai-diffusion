from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
)

from ..backend.resources import Arch
from ..files import File, FileSource
from ..localization import translate as _
from ..model.model import DocumentModel
from ..model.root import root
from . import theme
from .style import LoraList, populate_lora_folders


def lora_pixmap(file: File | None, size: int):
    path = root.files.loras.image_path(file) if file else None
    pixmap = QPixmap(str(path)) if path else QPixmap()
    return theme.icon("puzzle-bg").pixmap(size, size) if pixmap.isNull() else pixmap


class LoraSelectionDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter | None, option: QStyleOptionViewItem, index):
        if painter is None:
            return
        background = QStyleOptionViewItem(option)
        self.initStyleOption(background, index)
        background.text = ""
        background.icon = theme.icon("puzzle-bg")
        background.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration
        style = option.widget.style() if option.widget else QApplication.style()
        assert style is not None
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, background, painter, option.widget)
        file = root.files.loras.find(index.data(Qt.ItemDataRole.UserRole))
        if file is None:
            return
        painter.save()
        size = 40
        rect = option.rect.adjusted(5, 4, -5, -4)
        pixmap = lora_pixmap(file, size)
        target = QRect(rect.topLeft(), QSize(size, size))
        painter.drawPixmap(target, pixmap, pixmap.rect())
        rect.setLeft(rect.left() + size + 8)
        rect.setHeight(rect.height() // 2)
        selected = option.state & QStyle.StateFlag.State_Selected
        role = (
            option.palette.ColorRole.HighlightedText if selected else option.palette.ColorRole.Text
        )
        painter.setPen(option.palette.color(role))
        name = option.fontMetrics.elidedText(
            file.full_name, Qt.TextElideMode.ElideRight, rect.width()
        )
        painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter, name)
        rect.translate(0, rect.height())
        if not selected:
            painter.setPen(QColor(theme.grey))
        id = option.fontMetrics.elidedText(file.id, Qt.TextElideMode.ElideMiddle, rect.width())
        painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter, id)
        painter.restore()


class LoraSelectionDialog(QDialog):
    selected = pyqtSignal(str)

    def __init__(self, model: DocumentModel, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._model = model

        self.search = QLineEdit(self)
        self.search.setPlaceholderText(_("Search LoRA"))
        self.search.installEventFilter(self)
        self.folders = QComboBox(self)
        self.folders.setMinimumWidth(100)
        self.items = QListWidget(self)
        self.items.setItemDelegate(LoraSelectionDelegate(self.items))
        self.items.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.items.setMinimumHeight(100)
        self.items.setMaximumHeight(360)
        self.items.itemClicked.connect(self._select)
        self.items.itemActivated.connect(self._select)
        self.search.returnPressed.connect(self._select_current)
        self.search.textChanged.connect(self._filter)
        self.folders.currentIndexChanged.connect(self._filter)

        header = QHBoxLayout()
        header.addWidget(self.search, 1)
        header.addWidget(self.folders)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(header)
        layout.addWidget(self.items)
        self.resize(480, 410)

        root.files.loras.rowsInserted.connect(self._collect_folders)
        root.files.loras.rowsRemoved.connect(self._collect_folders)
        root.files.loras.dataChanged.connect(self._filter)
        model.active_style_changed.connect(self._filter)

        self._collect_folders()

    def open_at(self, pos: QPoint):
        self.move(pos)
        self.show()
        if screen := self.screen():
            area = screen.availableGeometry()
            self.move(
                max(area.left(), min(pos.x(), area.right() - self.width() + 1)),
                max(area.top(), min(pos.y(), area.bottom() - self.height() + 1)),
            )
        self.search.setFocus(Qt.FocusReason.PopupFocusReason)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None):
        if a0 is self.search and isinstance(a1, QKeyEvent) and a1.type() is QEvent.Type.KeyPress:
            if a1.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                if self.items.count():
                    step = -1 if a1.key() == Qt.Key.Key_Up else 1
                    row = max(0, min(self.items.currentRow() + step, self.items.count() - 1))
                    self.items.setCurrentRow(row)
                    if item := self.items.currentItem():
                        self.items.scrollToItem(item)
                return True
        return super().eventFilter(a0, a1)

    def _collect_folders(self):
        populate_lora_folders(self.folders, LoraList.last_filter)
        self._filter()

    def _filter(self):
        query = self.search.text().casefold()
        folder = self.folders.currentText()
        LoraList.last_filter = folder
        prefix = folder + "/" if folder and folder != "All" else ""
        arch = self._model.arch
        special = set()
        if client := root.connection.client_if_connected:
            special = {
                file
                for res, file in client.models.resources.items()
                if file is not None and res.startswith("lora-")
            }
        self.items.clear()
        for file in root.files.loras:
            if file.source is FileSource.unavailable or file.id in special:
                continue
            if prefix and not file.id.replace("\\", "/").startswith(prefix):
                continue
            if query not in file.display_name.casefold() and query not in file.id.casefold():
                continue
            if (
                file.arch not in (None, Arch.auto, Arch.all)
                and arch not in (Arch.auto, Arch.all)
                and not Arch.is_compatible(file.arch, arch)
            ):
                continue
            item = QListWidgetItem(file.display_name)
            item.setData(Qt.ItemDataRole.UserRole, file.id)
            item.setToolTip(file.id)
            item.setSizeHint(QSize(0, 48))
            self.items.addItem(item)
        if self.items.count():
            self.items.setCurrentRow(0)

    def _select_current(self):
        if item := self.items.currentItem():
            self._select(item)

    def _select(self, item: QListWidgetItem):
        self.selected.emit(item.data(Qt.ItemDataRole.UserRole))
        self.close()
