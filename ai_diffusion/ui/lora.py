from __future__ import annotations

from html import escape as html_escape
from pathlib import Path

from krita import Krita
from PyQt6.QtCore import (
    QEvent,
    QModelIndex,
    QObject,
    QPoint,
    QRect,
    QRectF,
    QSize,
    QSortFilterProxyModel,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QPainter, QPainterPath, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..backend.resources import Arch
from ..document import Document
from ..files import File, FileCollection, FileFilter, FileFormat, FileSource
from ..localization import translate as _
from ..model.model import DocumentModel
from ..model.root import root
from ..settings import ServerMode, Setting, settings
from . import theme
from .settings_widgets import WarningIcon
from .theme import SignalBlocker, add_header, icon


def lora_pixmap(file: File | None, size: int):
    path = root.files.loras.image_path(file) if file else None
    pixmap = QPixmap(str(path)) if path else QPixmap()
    return theme.icon("puzzle-bg").pixmap(size, size) if pixmap.isNull() else pixmap


class LoraThumbnail(QLabel):
    def paintEvent(self, a0):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 4, 4)
        painter.setClipPath(path)
        painter.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Base))
        pixmap = self.pixmap()
        if not pixmap.isNull():
            painter.drawPixmap(self.rect(), pixmap, pixmap.rect())
        if not self.isEnabled():
            painter.fillRect(self.rect(), QColor(128, 128, 128, 160))


class LoraPreviewMenu(QMenu):
    def __init__(self, file: File | None, parent=None, document: Document | None = None):
        super().__init__(parent)
        self._file = file
        self._document = document or root.active_model.document
        canvas = self.addAction(_("Use current canvas"))
        assert canvas is not None
        canvas.setEnabled(file is not None and self._document.is_valid)
        canvas.triggered.connect(self._set_image_from_canvas)
        open_file = self.addAction(_("Open file..."))
        assert open_file is not None
        open_file.setEnabled(file is not None)
        open_file.triggered.connect(self._load_image_from_file)

    def _set_image_from_canvas(self):
        if self._file is None:
            return
        try:
            image = self._document.get_image().to_pixmap().toImage()
            root.files.loras.set_image(self._file, image)
        except Exception as e:
            QMessageBox.warning(self, _("Preview image"), str(e))

    def _load_image_from_file(self):
        if self._file is None:
            return
        filename, _filter = QFileDialog.getOpenFileName(
            self, _("Select preview image"), "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not filename:
            return
        try:
            root.files.loras.set_image(self._file, QImage(filename))
        except Exception as e:
            QMessageBox.warning(self, _("Preview image"), str(e))


class LoraStrengthSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(-400, 400)
        self.setSingleStep(5)
        self.setValue(100)
        self.setSuffix("%")
        self.setMinimumWidth(80)


class LoraItem(QWidget):
    changed = pyqtSignal()
    removed = pyqtSignal(QWidget)

    def __init__(self, loras: FileFilter, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        self._loras = loras
        self._current: File | None = None
        self._is_active = True
        self._style_arch = Arch.auto
        self._edit_triggers = False

        completer = QCompleter(self._loras)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)

        self._select = QComboBox(self)
        self._select.setEditable(True)
        self._select.setModel(self._loras)
        self._select.setCompleter(completer)
        self._select.setMaxVisibleItems(20)
        self._select.setMinimumWidth(200)
        self._select.currentIndexChanged.connect(self._select_lora)

        self._warning_icon = WarningIcon(self)

        self._enabled = QCheckBox(self)
        self._enabled.setChecked(True)
        self._enabled.toggled.connect(self._notify_changed)

        self._strength = LoraStrengthSpinBox(self)
        self._strength.setPrefix(_("Strength") + ": ")
        self._strength.valueChanged.connect(self._notify_changed)

        self._remove = QToolButton(self)
        self._remove.setIcon(icon("discard"))
        self._remove.clicked.connect(self.remove)

        self._more = QToolButton(self)
        self._more.setIcon(icon("more"))
        self._more.setToolTip(_("More options"))
        self._more.clicked.connect(self._more_menu)

        self._thumbnail = LoraThumbnail(self)
        self._thumbnail.setFixedSize(56, 56)
        self._thumbnail.setToolTip(_("Right click to set a preview image"))
        self._thumbnail.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._thumbnail.customContextMenuRequested.connect(self._preview_menu)

        self._summary = QLabel(self)
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        self._summary.setMinimumWidth(0)
        self._summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self._trigger_edit = QLineEdit(self)
        self._trigger_edit.setPlaceholderText(
            _("Optional text which is added to the prompt when the LoRA is used")
        )
        self._trigger_edit.textChanged.connect(self._set_triggers)
        self._trigger_edit.setVisible(False)

        strength_layout = QHBoxLayout()
        strength_layout.setContentsMargins(0, 0, 0, 0)
        strength_layout.addWidget(self._strength)

        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.addWidget(self._summary, 1)
        action_layout.addWidget(self._trigger_edit, 1)
        action_layout.addWidget(self._warning_icon)
        action_layout.addWidget(self._more)

        item_layout = QGridLayout()
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.addWidget(self._enabled, 0, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        item_layout.addWidget(self._thumbnail, 0, 1, 2, 1)
        item_layout.addWidget(self._select, 0, 2)
        item_layout.addLayout(strength_layout, 0, 3)
        item_layout.addWidget(self._remove, 0, 4, Qt.AlignmentFlag.AlignRight)
        item_layout.addLayout(action_layout, 1, 2, 1, 3)
        item_layout.setColumnStretch(2, 3)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(item_layout)
        self.setLayout(layout)

        root.files.loras.dataChanged.connect(self._update)
        self._update_preview()
        if self._loras.rowCount() > 0:
            self._select_lora()

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self._update_summary()

    def _update_summary(self):
        name = ""
        detail = ""
        if file := self._current:
            name = file.full_name
            if file.arch:
                detail += f" | {file.arch.value}"
            if triggers := file.meta("lora_triggers", ""):
                detail += " | " + _("Trigger words") + f": {triggers}"
        self._summary.setToolTip(name + detail)
        metrics = self._summary.fontMetrics()
        available = self._summary.width()
        detail_width = metrics.horizontalAdvance(detail.strip())
        if detail_width >= available:
            name, detail = (
                "",
                metrics.elidedText(detail.strip(), Qt.TextElideMode.ElideRight, available),
            )
        elif name:
            name = metrics.elidedText(name, Qt.TextElideMode.ElideRight, available - detail_width)
        self._summary.setText(
            f"{html_escape(name)}<span style='color:{theme.grey}'>{html_escape(detail)}</span>"
        )

    def _update_preview(self):
        path = root.files.loras.image_path(self._current) if self._current else None
        pixmap = QPixmap(str(path)) if path else QPixmap()
        if pixmap.isNull():
            pixmap = icon("puzzle-bg").pixmap(56, 56)
        self._thumbnail.setPixmap(
            pixmap.scaled(
                56,
                56,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _preview_menu(self, position):
        if not self._current:
            return
        menu = LoraPreviewMenu(self._current, self)
        menu.exec(self._thumbnail.mapToGlobal(position))

    def _more_menu(self):
        menu = self._create_more_menu()
        menu.exec(self._more.mapToGlobal(self._more.rect().bottomLeft()))

    def _create_more_menu(self):
        menu = LoraPreviewMenu(self._current, self)
        has_lora = self._current is not None
        menu.addSeparator()

        default = self._default_strength
        strength = self._strength.value()
        if default == strength:
            set_default = menu.addAction(_("Set default strength ({strength}%)", strength=strength))
        else:
            set_default = menu.addAction(
                _(
                    "Set default strength {default}% -> {strength}%",
                    default=default,
                    strength=strength,
                )
            )
        assert set_default is not None
        set_default.setEnabled(has_lora and default != strength)
        set_default.triggered.connect(lambda: self._set_default_strength())

        edit_triggers = menu.addAction(_("Edit trigger words"))
        assert edit_triggers is not None
        edit_triggers.setCheckable(True)
        edit_triggers.setChecked(self._edit_triggers)
        edit_triggers.triggered.connect(self._set_edit_triggers)

        return menu

    def _set_edit_triggers(self, enabled: bool):
        self._edit_triggers = enabled
        self._summary.setVisible(not enabled)
        self._trigger_edit.setVisible(enabled)
        if enabled:
            self._trigger_edit.setFocus()

    def _notify_changed(self):
        self._update()
        self.changed.emit()

    def _update(self):
        enabled = self._enabled.isChecked()
        self._select.setEnabled(enabled)
        self._strength.setEnabled(enabled)
        self._thumbnail.setEnabled(enabled)
        self._update_summary()
        self._update_preview()
        if self._current:
            with SignalBlocker(self._trigger_edit):
                self._trigger_edit.setText(self._current.meta("lora_triggers", ""))
            self._show_lora_warnings(self._current)

    def _select_lora(self):
        id = self._select.currentData()
        file = root.files.loras.find(id)
        if file and file != self._current:
            self._current = file
            default_strength = int(file.meta("lora_strength", 1.0) * 100)
            if default_strength != self._strength.value():
                self._strength.setValue(default_strength)
            if trigger_words := file.meta("lora_triggers", ""):
                self._trigger_edit.setText(trigger_words)
            self._notify_changed()

    def _set_triggers(self):
        value = self._trigger_edit.text()
        if self._current and self._current.meta("lora_triggers") != value:
            root.files.loras.set_meta(self._current, "lora_triggers", value)

    def _set_default_strength(self):
        if self._current and self._default_strength != self._strength.value():
            root.files.loras.set_meta(self._current, "lora_strength", self.strength)

    @property
    def _default_strength(self):
        if self._current is None:
            return 100
        strength = self._current.meta("lora_strength")
        return 100 if strength is None else int(strength * 100)

    def remove(self):
        self._set_edit_triggers(False)
        self.removed.emit(self)

    @property
    def strength(self):
        return self._strength.value() / 100

    @strength.setter
    def strength(self, value: float):
        value_int = int(value * 100)
        if value_int != self._strength.value():
            self._strength.setValue(value_int)

    @property
    def value(self):
        if self._current is None:
            return {"name": "", "strength": 1.0, "enabled": True}
        return {
            "name": self._current.id,
            "strength": self.strength,
            "enabled": self._enabled.isChecked(),
        }

    @value.setter
    def value(self, v: dict):
        new_value = root.files.loras.find(v["name"]) or File.remote(v["name"])
        ui_name = self._select.currentText()
        if (
            self._current is None
            or new_value.id != self._current.id
            or ui_name != new_value.display_name
        ):
            self._current = new_value
            index = self._select.findData(new_value.id)
            if index >= 0:
                self._select.setCurrentIndex(index)
            else:
                self._select.setEditText(self._current.display_name)
        self.strength = v["strength"]
        self._enabled.setChecked(v.get("enabled", True))
        self._update()

    @property
    def is_active(self):
        """False if item is kept alive for reuse but not displayed."""
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool):
        self._is_active = value
        self.setVisible(value)  # isVisible() can be False depending on UI state, even if active

    def start_apply_filter(self):
        self._select.blockSignals(True)

    def apply_filter(self):
        # filter available items _without_ changing current selection
        if self._current and self._current.id != self._select.currentData():
            self._select.setEditText(self._current.display_name)
        self._select.blockSignals(False)

    def reset(self):
        self._current = None
        self.strength = 1.0
        self._enabled.setChecked(True)
        self._select.setCurrentIndex(0)
        if self._loras.rowCount() > 0:
            self._select_lora()

    def set_style_arch(self, arch: Arch):
        self._style_arch = arch
        if self._current:
            self._show_lora_warnings(self._current)

    def _show_lora_warnings(self, lora: File):
        warnings = []
        if (
            lora.arch is not None
            and lora.arch not in {Arch.auto, Arch.all}
            and self._style_arch not in {Arch.auto, Arch.all}
            and not Arch.is_compatible(lora.arch, self._style_arch)
        ):
            warnings.append(
                _(
                    "This LoRA is not compatible with the Style's Model Checkpoint, and will fail to apply. LoRA: {lora_arch} | Diffusion model: {style_arch}",
                    lora_arch=lora.arch.value,
                    style_arch=self._style_arch.value,
                )
            )
        if client := root.connection.client_if_connected:
            special_loras = [
                file
                for res, file in client.models.resources.items()
                if file is not None and res.startswith("lora-")
            ]
            file_ref = root.files.loras.find(lora.id)
            if file_ref is None or file_ref.source is FileSource.unavailable:
                warnings.append(_lora_not_installed_warning)
            elif lora.id in special_loras:
                warnings.append(_special_lora_warning)
        message = "\n".join(warnings)
        if message:
            self._warning_icon.show_message(message)
        else:
            self._warning_icon.hide()


_lora_not_installed_warning = _("The LoRA file is not installed on the server.")
_special_lora_warning = _(
    "This LoRA is usually added automatically by a Sampler or Control Layer when needed.\nIt is not required to add it manually here."
)


def populate_lora_folders(combo: QComboBox, selected: str):
    with SignalBlocker(combo):
        combo.clear()
        combo.addItem(icon("filter"), "All")
        folders = set()
        for lora in root.files.loras:
            if lora.source is not FileSource.unavailable:
                parts = lora.id.replace("\\", "/").split("/")
                folders.update("/".join(parts[:i]) for i in range(1, len(parts)))
        folder_icon = Krita.instance().icon("document-open")
        for folder in sorted(folders, key=str.lower):
            combo.addItem(folder_icon, folder)
    combo.setCurrentText(selected)


class LoraList(QWidget):
    value_changed = pyqtSignal()

    open_folder_button: QToolButton | None = None
    last_filter = "All"

    def __init__(self, setting: Setting, parent=None):
        super().__init__(parent)
        self._items: list[LoraItem] = []
        self._style_arch = Arch.auto
        self._loras = FileFilter(root.files.loras)
        self._loras.available_only = True

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_text_layout = QVBoxLayout()
        add_header(header_text_layout, setting)
        header_layout.addLayout(header_text_layout, 5)

        self._add_button = QPushButton(_("Add"), self)
        self._add_button.setMinimumWidth(100)
        self._add_button.clicked.connect(self._add_item)
        header_layout.addWidget(self._add_button, 1)

        self._upload_button = QPushButton(icon("upload"), "  " + _("Upload"), self)
        self._upload_button.setToolTip(_("Import a LoRA file from your local system"))
        self._upload_button.clicked.connect(self._upload_lora)
        header_layout.addWidget(self._upload_button, 1)

        self._filter_combo = QComboBox(self)
        self._filter_combo.currentIndexChanged.connect(self._set_filtered_names)
        header_layout.addWidget(self._filter_combo, 2)

        self._refresh_button = QToolButton(self)
        self._refresh_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._refresh_button.setIcon(Krita.instance().icon("reload-preset"))
        self._refresh_button.setToolTip(_("Look for new LoRA files"))
        self._refresh_button.clicked.connect(root.connection.refresh)
        header_layout.addWidget(self._refresh_button, 0)

        if settings.server_mode is ServerMode.managed:
            open_folder = self.open_folder_button = QToolButton(self)
            open_folder.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            open_folder.setIcon(Krita.instance().icon("document-open"))
            open_folder.setToolTip(_("Open folder containing LoRA files"))
            header_layout.addWidget(open_folder, 0)

        self._layout.addLayout(header_layout)

        self._item_list = QVBoxLayout()
        self._item_list.setContentsMargins(0, 0, 0, 0)
        self._layout.addLayout(self._item_list)

        root.files.loras.rowsInserted.connect(self._collect_filters)
        root.files.loras.rowsRemoved.connect(self._collect_filters)
        self._collect_filters()

    def _add_item(self, lora: dict | File | None = None):
        assert self._item_list is not None
        for i in self._items:
            if not i.is_active:
                item = i  # reuse existing item to avoid cost of creating new ones
                item.is_active = True
                break
        else:
            item = LoraItem(self._loras, parent=self)
            item.changed.connect(self._update_item)
            item.removed.connect(self._remove_item)
            self._items.append(item)

        item.set_style_arch(self._style_arch)
        if isinstance(lora, dict):
            item.value = lora
        elif isinstance(lora, File):
            item.value = {"name": lora.id, "strength": 1.0}
        else:
            item.reset()
        self._item_list.addWidget(item)
        self.value_changed.emit()

    def set_style_arch(self, arch: Arch):
        self._style_arch = arch
        for item in self._items:
            item.set_style_arch(arch)

    def _remove_item(self, item: LoraItem):
        # removing and creating items is slow, hiding allows reuse
        item.is_active = False
        self._item_list.removeWidget(item)
        self.value_changed.emit()

    def _update_item(self):
        self.value_changed.emit()

    def _collect_filters(self):
        populate_lora_folders(self._filter_combo, LoraList.last_filter)
        self._add_button.setEnabled(root.files.loras.rowCount() > 0)

    def _set_filtered_names(self):
        LoraList.last_filter = self.filter
        for item in self._items:
            item.start_apply_filter()
        self._loras.name_prefix = self.filter_prefix
        for item in self._items:
            item.apply_filter()

    def _upload_lora(self):
        filepath = QFileDialog.getOpenFileName(
            self, _("Select LoRA file"), None, "LoRA files (*.safetensors)"
        )
        if filepath[0]:
            path = Path(filepath[0])
            if client := root.connection.client_if_connected:
                max_size = client.features.max_upload_size
                if max_size and path.stat().st_size > max_size:
                    _show_file_too_large_warning(max_size, self)
                    return
            file = File.local(path, FileFormat.lora)
            root.files.loras.add(file)
            self._add_item(file)

    @property
    def filter(self):
        return self._filter_combo.currentText()

    @property
    def filter_prefix(self):
        return self.filter if self.filter != "All" else ""

    @property
    def value(self):
        return [item.value for item in self._items if item.is_active]

    @value.setter
    def value(self, v: list[dict | File]):
        for item in self._items:
            self._remove_item(item)
        for lora in v:
            self._add_item(lora)


def _show_file_too_large_warning(max_size: int, parent=None):
    msg = _("The file is too large to be uploaded. Files up to a size of {size} MB are supported.")
    QMessageBox.warning(parent, _("File too large"), msg.format(size=max_size / (1024**2)))


class LoraSelectionDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return QSize(0, 48)

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
        file = index.data(FileCollection.file_role)
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
        self._loras = FileFilter(root.files.loras, self)
        self._loras.available_only = True
        self._search_filter = QSortFilterProxyModel(self)
        self._search_filter.setSourceModel(self._loras)
        self._search_filter.setFilterRole(FileCollection.search_role)
        self._search_filter.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        self.search = QLineEdit(self)
        self.search.setPlaceholderText(_("Search LoRA"))
        self.search.installEventFilter(self)
        self.folders = QComboBox(self)
        self.folders.setMinimumWidth(100)
        self.items = QListView(self)
        self.items.setModel(self._search_filter)
        self.items.setUniformItemSizes(True)
        self.items.setItemDelegate(LoraSelectionDelegate(self.items))
        self.items.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.items.setMinimumHeight(100)
        self.items.setMaximumHeight(360)
        self.items.clicked.connect(self._select)
        self.items.activated.connect(self._select)
        self.search.returnPressed.connect(self._select_current)
        self.search.textChanged.connect(self._filter_text)
        self.folders.currentIndexChanged.connect(self._filter_folder)

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
        root.connection.models_changed.connect(self._filter_models)
        model.active_style_changed.connect(self._filter_models)

        self._filter_models()
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
                if count := self._search_filter.rowCount():
                    step = -1 if a1.key() == Qt.Key.Key_Up else 1
                    row = max(0, min(self.items.currentIndex().row() + step, count - 1))
                    index = self._search_filter.index(row, 0)
                    self.items.setCurrentIndex(index)
                    self.items.scrollTo(index)
                return True
        return super().eventFilter(a0, a1)

    def _collect_folders(self):
        populate_lora_folders(self.folders, LoraList.last_filter)
        self._filter_folder()

    def _filter_text(self, text: str):
        self._search_filter.setFilterFixedString(text)
        self._select_first()

    def _filter_folder(self):
        folder = self.folders.currentText()
        LoraList.last_filter = folder
        self._loras.name_prefix = folder if folder != "All" else ""
        self._select_first()

    def _filter_models(self):
        self._loras.arch = self._model.arch
        special = set()
        if client := root.connection.client_if_connected:
            special = {
                file
                for res, file in client.models.resources.items()
                if file is not None and res.startswith("lora-")
            }
        self._loras.excluded_files = special
        self._select_first()

    def _select_first(self):
        self.items.setCurrentIndex(self._search_filter.index(0, 0))

    def _select_current(self):
        self._select(self.items.currentIndex())

    def _select(self, index: QModelIndex):
        if index.isValid():
            self.selected.emit(index.data(Qt.ItemDataRole.UserRole))
            self.close()
