from __future__ import annotations

import json
from enum import Enum
from math import pi, sin
from textwrap import wrap as wrap_text
from typing import cast

from krita import Krita
from PyQt6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QItemSelectionModel,
    QMetaObject,
    QPoint,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QTimer,
    pyqtProperty,  # type: ignore
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QEnterEvent,
    QFocusEvent,
    QGuiApplication,
    QHideEvent,
    QIcon,
    QKeyEvent,
    QKeySequence,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPalette,
    QPen,
    QPixmap,
    QPolygon,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from ..backend.api import WorkflowKind
from ..image import Bounds, Extent, Image
from ..localization import translate as _
from ..model.jobs import Job, JobKind, JobParams, JobQueue, JobState
from ..model.model import DocumentModel, Workspace
from ..model.properties import Binding
from ..model.region import RootRegion
from ..model.root import root
from ..settings import ApplyBehavior, settings
from ..style import Styles
from ..util import ensure, flatten, sequence_equal
from . import theme

_job_info_translations: dict[str, str] = {
    "prompt": _("Prompt"),
    "prompt_eval": _("Prompt (Evaluated)"),
    "negative_prompt": _("Negative Prompt"),
    "negative_prompt_eval": _("Negative Prompt (Evaluated)"),
    "style": _("Style"),
    "strength": _("Strength"),
    "checkpoint": _("Model"),
    "loras": _("LoRA"),
    "sampler": _("Sampler"),
    "seed": _("Seed"),
    "steps": _("Sampler Steps"),
    "guidance": _("Guidance Strength (CFG Scale)"),
    "control": _("Control Layers"),
}


def job_info_text(params: JobParams, hint: str | None = None):
    if hint is None:
        hint = _("Click to toggle preview, double-click to apply.")
    title = params.name if params.name != "" else "<no prompt>"
    if len(title) > 70:
        title = title[:66] + "..."
    if params.strength != 1.0:
        title = f"{title} @ {params.strength * 100:.0f}%"
    style = Styles.list().find(params.style)
    strings: list[str | list[str]] = [title + "\n", hint, ""]
    for key, value in params.metadata.items():
        if key not in _job_info_translations:
            continue
        if key == "style" and style:
            value = style.name
        if isinstance(value, list) and len(value) == 0:
            continue
        if key == "loras" and isinstance(value, list) and isinstance(value[0], dict):
            value = " | ".join(
                f"{v.get('name')} ({v.get('weight', v.get('strength', '?'))})"
                for v in value
                if v.get("enabled", True)
            )
        if key == "control" and isinstance(value, list) and isinstance(value[0], dict):
            control_text = []
            for v in value:
                t = f"{v.get('mode')}: {v.get('image', '')[:30]} @{v.get('strength', '?')}"
                control_text.append(t)
            value = " | ".join(control_text)
        s = f"{_job_info_translations.get(key, key)}: {value}"
        s = wrap_text(s, 80, subsequent_indent=" ")
        strings.append(s)
    strings.append(_("Seed") + f": {params.seed}")
    return "\n".join(flatten(strings))


def copy_job_prompt(model: DocumentModel, job: Job, evaluated=False):
    positive = "prompt_eval" if evaluated else "prompt"
    prompt = job.params.metadata.get(positive, job.params.prompt)
    active = model.active_regions.active_or_root
    active.positive = prompt
    if isinstance(active, RootRegion):
        negative = "negative_prompt_eval" if evaluated else "negative_prompt"
        active.negative = job.params.metadata.get(
            negative, job.params.metadata.get("negative_prompt", "")
        )

    if clipboard := QGuiApplication.clipboard():
        clipboard.setText(prompt)

    if model.workspace is Workspace.custom and model.document.is_active:
        model.custom.try_set_params(job.params.metadata)


def copy_job_strength(model: DocumentModel, job: Job):
    model.strength = job.params.strength


def copy_job_style(model: DocumentModel, job: Job):
    if style := Styles.list().find(job.params.style):
        model.style = style


def copy_job_seed(model: DocumentModel, job: Job):
    model.fixed_seed = True
    model.seed = job.params.seed


def copy_job_info(job: Job):
    if clipboard := QGuiApplication.clipboard():
        style = Styles.list().find(job.params.style)
        data = job.params.metadata.copy()
        if style:
            data["style"] = f"{style.name} ({style.filename})"
        text = json.dumps(data, indent=2)
        clipboard.setText(text)


def discard_all_results(model: DocumentModel, parent: QWidget):
    reply = QMessageBox.warning(
        parent,
        _("Clear History"),
        _("Are you sure you want to discard all generated images?"),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply == QMessageBox.StandardButton.Yes:
        model.jobs.clear()
        model.hide_preview(delete_layer=True)
        return True
    return False


class HistoryWidget(QListWidget):
    item_activated = pyqtSignal(QListWidgetItem)

    _thumb_size = 96
    _applied_icon = Image.load(theme.icon_path / "star.png")
    _list_css = f"""
        QListWidget {{ background-color: transparent; }}
        QListWidget::item:selected {{ border: 1px solid {theme.grey}; }}
    """
    _button_css = f"""
        QPushButton {{
            border: 1px solid {theme.grey};
            background: {"rgba(64, 64, 64, 170)" if theme.is_dark else "rgba(240, 240, 240, 160)"};
            padding: 2px;
        }}
        QPushButton:hover {{
            background: {"rgba(72, 72, 72, 210)" if theme.is_dark else "rgba(240, 240, 240, 200)"};
        }}
    """

    def __init__(self, parent: QWidget | None):
        super().__init__(parent)
        self._model = root.active_model
        self._connections: list[QMetaObject.Connection] = []
        self._sort_descending = True
        self._star_filter = False

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setIconSize(theme.screen_scale(self, QSize(self._thumb_size, self._thumb_size)))
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setStyleSheet(self._list_css)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setDragEnabled(False)
        self.itemClicked.connect(self.handle_preview_click)
        self.itemDoubleClicked.connect(self.item_activated)
        self.itemSelectionChanged.connect(self.select_item)

        self._top_bar = QWidget(self)
        self._top_bar.setObjectName("historyTopBar")
        top_layout = QHBoxLayout(self._top_bar)
        top_layout.setContentsMargins(4, 2, 4, 2)
        top_layout.setSpacing(4)

        self._search = QLineEdit(self._top_bar)
        self._search.setPlaceholderText(_("Search history..."))
        self._search.setClearButtonEnabled(True)
        self._search.addAction(theme.icon("search"), QLineEdit.ActionPosition.LeadingPosition)
        self._search.textChanged.connect(self._update_filter)

        self._star_button = QToolButton(self._top_bar)
        self._star_button.setIcon(theme.icon("star"))
        self._star_button.setCheckable(True)
        self._star_button.setChecked(False)
        self._star_button.setToolTip(_("Show applied images only"))
        self._star_button.toggled.connect(self._set_star_filter)

        self._sort_button = QToolButton(self._top_bar)
        self._sort_button.setIcon(theme.icon("sort-descending"))
        self._sort_button.setToolTip(_("Sort: newest last"))
        self._sort_button.clicked.connect(self._toggle_sort)

        top_layout.addWidget(self._search, 1)
        top_layout.addWidget(self._star_button)
        top_layout.addWidget(self._sort_button)

        self._bar_height = max(self._top_bar.sizeHint().height(), self.fontMetrics().height() + 8)
        self.setViewportMargins(0, self._bar_height, 0, 0)
        self._layout_top_bar()

        self._apply_button = QPushButton(theme.icon("apply"), _("Apply"), self)
        self._apply_button.setStyleSheet(self._button_css)
        self._apply_button.setVisible(False)
        self._apply_button.clicked.connect(self._activate_selection)

        self._context_button = QPushButton(theme.icon("context"), "", self)
        self._context_button.setStyleSheet(self._button_css)
        self._context_button.setVisible(False)
        self._context_button.clicked.connect(self._show_context_menu_dropdown)

        f = self.fontMetrics()
        self._apply_button.setFixedHeight(f.height() + 8)
        self._context_button.setFixedWidth(f.height() + 8)
        if scrollbar := self.verticalScrollBar():
            scrollbar.valueChanged.connect(self.update_apply_button)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    @property
    def model_(self):
        return self._model

    @model_.setter
    def model_(self, model: DocumentModel):
        Binding.disconnect_all(self._connections)
        self._model = model
        jobs = model.jobs
        self._connections = [
            jobs.selection_changed.connect(self.update_selection),
            jobs.job_finished.connect(self.add),
            jobs.job_discarded.connect(self.remove),
            jobs.result_used.connect(self.update_image_thumbnail),
            jobs.result_discarded.connect(self.remove_image),
        ]
        self.rebuild()
        self.update_selection()

    def add(self, job: Job):
        if not self.is_finished(job):
            return  # Only finished diffusion/animation jobs have images to show

        indices = self._visible_indices(job)
        if not indices:
            return

        scrollbar = self.verticalScrollBar()
        if self._sort_descending:
            scroll_to_edge = scrollbar and scrollbar.value() >= scrollbar.maximum() - 4
            prev_job = self._job_at_edge(top=False)
            if prev_job is None or not JobParams.equal_ignore_seed(prev_job.params, job.params):
                self.addItem(self._make_header(job))
            for item in self._make_job_items(job, indices):
                self.addItem(item)
            if scroll_to_edge:
                self.scrollToBottom()
        else:
            scroll_to_edge = scrollbar and scrollbar.value() <= 4
            next_job = self._job_at_edge(top=True)
            items = self._make_job_items(job, indices)
            if next_job is None or not JobParams.equal_ignore_seed(job.params, next_job.params):
                items.insert(0, self._make_header(job))
                for item in reversed(items):
                    self.insertItem(0, item)
            else:
                if self.count() > 0 and self._is_header(ensure(self.item(0))):
                    self.takeItem(0)
                self.insertItem(0, self._make_header(job))
                for item in reversed(items):
                    self.insertItem(1, item)
            if scroll_to_edge:
                self.scrollToTop()
        self._sync_headers()

    def _make_header(self, job: Job):
        prompt = job.params.name if job.params.name != "" else "<no prompt>"
        strength = job.params.metadata.get("strength", 1.0)
        strength = f"{strength * 100:.0f}% - " if strength != 1.0 else ""

        header = QListWidgetItem(f"{job.timestamp.astimezone():%H:%M} - {strength}{prompt}")
        header.setFlags(Qt.ItemFlag.NoItemFlags)
        header.setData(Qt.ItemDataRole.UserRole, job.id)
        header.setData(Qt.ItemDataRole.ToolTipRole, job.params.prompt)
        header.setSizeHint(QSize(9999, self.fontMetrics().lineSpacing() + 4))
        header.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
        return header

    def _make_job_items(self, job: Job, indices: list[int]):
        return [self._make_job_item(job, index) for index in indices]

    def _make_job_item(self, job: Job, index: int):
        if job.kind is JobKind.diffusion:
            if job.params.is_layered:
                item = QListWidgetItem(self._image_thumbnail(job, 0), None)
            else:
                item = QListWidgetItem(self._image_thumbnail(job, index), None)
        elif job.kind is JobKind.animation:
            item = AnimatedListItem([
                self._image_thumbnail(job, i) for i in range(len(job.results))
            ])
        else:
            raise AssertionError(f"Unexpected job kind {job.kind}")
        item.setData(Qt.ItemDataRole.UserRole, job.id)
        item.setData(Qt.ItemDataRole.UserRole + 1, index)
        item.setData(Qt.ItemDataRole.ToolTipRole, job_info_text(job.params))
        return item

    def remove(self, job: Job):
        self._remove_items(ensure(job.id))

    def remove_image(self, id: JobQueue.Item):
        self._remove_items(id.job, id.image)

    def _remove_items(self, job_id: str, image_index: int = -1):
        def _job_id(item: QListWidgetItem | None):
            return item.data(Qt.ItemDataRole.UserRole) if item else None

        item_was_selected = False
        with theme.SignalBlocker(self):
            # Remove all the job's items before triggering potential selection changes
            current = next((i for i in range(self.count()) if _job_id(self.item(i)) == job_id), -1)
            if current >= 0:
                item = self.item(current)
                while item and _job_id(item) == job_id:
                    _, index = self.item_info(item)
                    if image_index == index or (index is not None and image_index == -1):
                        item_was_selected = item_was_selected or item.isSelected()
                        self.takeItem(current)
                    else:
                        if index and index > image_index:
                            item.setData(Qt.ItemDataRole.UserRole + 1, index - 1)
                        current += 1
                    item = self.item(current)
            self._sync_headers()

        if item_was_selected:
            self._model.jobs.selection = []
        else:
            self.update_apply_button()  # selection may have moved

    def update_selection(self):
        current = [self._item_data(i) for i in self.selectedItems()]
        changed = not sequence_equal(self._model.jobs.selection, current)

        with theme.SignalBlocker(self):
            for i in range(self.count()):
                item = self.item(i)
                if item and item.type() == QListWidgetItem.ItemType.UserType:
                    cast(AnimatedListItem, item).stop_animation()

            if changed:  # don't mess with widget's state if it already matches
                self.clearSelection()

            for selection in self._model.jobs.selection:
                if item := self._find(selection):
                    if changed:
                        item.setSelected(True)
                    if item.type() == QListWidgetItem.ItemType.UserType:
                        cast(AnimatedListItem, item).start_animation()

        self.update_apply_button()

    def update_apply_button(self):
        selected = self.selectedItems()
        if len(selected) > 0:
            rect = self.visualItemRect(selected[0])
            rect.translate(ensure(self.viewport()).pos())
            font = self._apply_button.fontMetrics()
            context_visible = rect.width() >= 0.6 * self.iconSize().width()
            apply_text_visible = font.horizontalAdvance(_("Apply")) < 0.35 * rect.width()
            apply_pos = QPoint(rect.left() + 3, rect.bottom() - self._apply_button.height() - 2)
            if context_visible:
                cw = self._context_button.width()
                context_pos = QPoint(rect.right() - cw - 2, apply_pos.y())
                context_size = QSize(cw, self._apply_button.height())
            else:
                context_pos = QPoint(rect.right(), apply_pos.y())
                context_size = QSize(0, 0)
            apply_size = QSize(context_pos.x() - rect.left() - 5, self._apply_button.height())
            self._apply_button.setVisible(True)
            self._apply_button.move(apply_pos)
            self._apply_button.resize(apply_size)
            self._apply_button.setText(_("Apply") if apply_text_visible else "")
            self._context_button.setVisible(context_visible)
            if context_visible:
                self._context_button.move(context_pos)
                self._context_button.resize(context_size)
        else:
            self._apply_button.setVisible(False)
            self._context_button.setVisible(False)

    def update_image_thumbnail(self, id: JobQueue.Item):
        if item := self._find(id):
            job = ensure(self._model.jobs.find(id.job))
            item.setIcon(self._image_thumbnail(job, id.image))
        elif self._star_filter:
            self._rebuild()

    def select_item(self):
        self._model.jobs.selection = [self._item_data(i) for i in self.selectedItems()]

    def _toggle_selection(self):
        self._model.jobs.toggle_selection()

    def _activate_selection(self):
        items = self.selectedItems()
        if len(items) > 0:
            self.item_activated.emit(items[0])

    def is_finished(self, job: Job):
        return job.kind in [JobKind.diffusion, JobKind.animation] and job.state is JobState.finished

    def rebuild(self):
        self._rebuild()
        if self._sort_descending:
            self.scrollToBottom()
        else:
            self.scrollToTop()

    def _rebuild(self):
        cached = self._collect_image_items()
        with theme.SignalBlocker(self):
            while self.count():
                self.takeItem(0)
            last_params = None
            for job in self._display_jobs():
                indices = self._visible_indices(job)
                if not indices:
                    continue
                if not JobParams.equal_ignore_seed(last_params, job.params):
                    self.addItem(self._make_header(job))
                    last_params = job.params
                for index in indices:
                    item = cached.get((job.id, index))
                    if item is None:
                        item = self._make_job_item(job, index)
                    self.addItem(item)
        self.update_selection()

    def _display_jobs(self):
        jobs = [job for job in self._model.jobs if self.is_finished(job)]
        if not self._sort_descending:
            jobs.reverse()
        return jobs

    def _job_matches_search(self, job: Job):
        text = self._search.text().strip().lower()
        if not text:
            return True
        return text in job.params.prompt.lower() or text in job.params.name.lower()

    def _visible_indices(self, job: Job):
        if not self._job_matches_search(job):
            return []
        if job.kind is JobKind.animation or job.params.is_layered:
            indices = [0]
        else:
            indices = list(range(len(job.results)))
        if self._star_filter:
            indices = [i for i in indices if job.result_was_used(i)]
        return indices

    def _job_at_edge(self, top: bool):
        if self.count() == 0:
            return None
        item = ensure(self.item(0 if top else self.count() - 1))
        return self._job_for_item(item)

    def _job_for_item(self, item: QListWidgetItem):
        job_id = item.data(Qt.ItemDataRole.UserRole)
        return self._model.jobs.find(job_id) if job_id else None

    def _is_header(self, item: QListWidgetItem):
        return item.data(Qt.ItemDataRole.UserRole + 1) is None

    def _collect_image_items(self):
        cached = {}
        for i in range(self.count()):
            item = ensure(self.item(i))
            index = item.data(Qt.ItemDataRole.UserRole + 1)
            if index is not None:
                cached[(item.data(Qt.ItemDataRole.UserRole), index)] = item
        return cached

    def _sync_headers(self):
        prev_params: JobParams | None = None
        i = 0
        while i < self.count():
            item = ensure(self.item(i))
            job = self._job_for_item(item)
            if job is None:
                self.takeItem(i)
                continue
            if self._is_header(item):
                if prev_params is not None and JobParams.equal_ignore_seed(prev_params, job.params):
                    self.takeItem(i)
                    continue
                i += 1
                continue
            if prev_params is None or not JobParams.equal_ignore_seed(prev_params, job.params):
                prev_item = self.item(i - 1) if i > 0 else None
                prev_job = self._job_for_item(prev_item) if prev_item is not None else None
                if not (
                    prev_item is not None
                    and self._is_header(prev_item)
                    and prev_job is not None
                    and JobParams.equal_ignore_seed(prev_job.params, job.params)
                ):
                    self.insertItem(i, self._make_header(job))
                    i += 1
            prev_params = job.params
            i += 1

    def _update_filter(self):
        self._rebuild()
        self.scrollToTop()

    def _set_star_filter(self, checked: bool):
        self._star_filter = checked
        self._star_button.setToolTip(
            _("Show all images") if checked else _("Show applied images only")
        )
        self._update_filter()

    def _toggle_sort(self):
        self._sort_descending = not self._sort_descending
        self._sort_button.setIcon(
            theme.icon("sort-descending" if self._sort_descending else "sort-ascending")
        )
        self._sort_button.setToolTip(
            _("Sort: newest last") if self._sort_descending else _("Sort: newest first")
        )
        self._rebuild()
        if self._sort_descending:
            self.scrollToBottom()
        else:
            self.scrollToTop()

    def _layout_top_bar(self):
        viewport = ensure(self.viewport())
        self._top_bar.setGeometry(viewport.x(), 0, viewport.width(), self._bar_height)

    def item_info(self, item: QListWidgetItem) -> tuple[str, int]:  # job id, image index
        return item.data(Qt.ItemDataRole.UserRole), item.data(Qt.ItemDataRole.UserRole + 1)

    @property
    def selected_job(self) -> Job | None:
        items = self.selectedItems()
        if len(items) > 0:
            job_id, _ = self.item_info(items[0])
            return self._model.jobs.find(job_id)
        return None

    def handle_preview_click(self, item: QListWidgetItem):
        if item.text() != "" and item.text() != "<no prompt>":
            if clipboard := QGuiApplication.clipboard():
                prompt = item.data(Qt.ItemDataRole.ToolTipRole)
                clipboard.setText(prompt)

    def mousePressEvent(self, e: QMouseEvent | None):
        if (  # make single click deselect current item (usually requires Ctrl+click)
            e is not None
            and e.button() == Qt.MouseButton.LeftButton
            and e.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            item = self.itemAt(e.pos())
            if item is not None and item.isSelected():
                self.clearSelection()
                e.accept()
                return
        super().mousePressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout_top_bar()
        self.update_apply_button()

    def event(self, e: QEvent | None):
        assert e is not None
        # Disambiguate shortcut events which Krita overrides
        if e.type() == QEvent.Type.ShortcutOverride:
            assert isinstance(e, QKeyEvent)
            if e.matches(QKeySequence.StandardKey.Delete):
                self._discard_image(confirm=False)
                e.accept()
            elif e.key() == Qt.Key.Key_Space:
                self._toggle_selection()
                e.accept()
        return super().event(e)

    def _find(self, id: JobQueue.Item):
        items = (ensure(self.item(i)) for i in range(self.count()))
        return next((item for item in items if self._item_data(item) == id), None)

    def _item_data(self, item: QListWidgetItem):
        return JobQueue.Item(
            item.data(Qt.ItemDataRole.UserRole), item.data(Qt.ItemDataRole.UserRole + 1)
        )

    def _image_thumbnail(self, job: Job, index: int):
        image = job.results[index]
        # Use 2x thumb size for good quality on high-DPI screens
        thumb = Image.scale_to_fit(image, Extent(self._thumb_size * 2, self._thumb_size * 2))
        min_height = min(4 * self._apply_button.height(), 2 * self._thumb_size)
        if thumb.extent.height < min_height:
            thumb = Image.crop(thumb, Bounds(0, 0, thumb.extent.width, min_height))
        if job.result_was_used(index):  # add tiny star icon to mark used results
            thumb.draw_image(self._applied_icon, offset=(thumb.extent.width - 28, 4))
        return thumb.to_icon()

    def _show_context_menu(self, pos: QPoint):
        item = self.itemAt(pos)
        if item is not None:
            job = self._model.jobs.find(self._item_data(item).job)
            menu = QMenu(self)
            menu.addAction(_("Copy Prompt"), self._copy_prompt)
            menu.addAction(_("Copy Prompt (Evaluated)"), self._copy_prompt_evaluated)
            menu.addAction(_("Copy Strength"), self._copy_strength)
            style_action = ensure(menu.addAction(_("Copy Style"), self._copy_style))
            if job is None or Styles.list().find(job.params.style) is None:
                style_action.setEnabled(False)
            menu.addAction(_("Copy Seed"), self._copy_seed)
            menu.addAction(_("Info to Clipboard"), self._info_to_clipboard)
            menu.addSeparator()
            save_action = ensure(menu.addAction(_("Save Image"), self._save_image))
            if self._model.document.filename == "":
                tt = _(
                    "Save as separate image to the same folder as the document.\nMust save the document first!"
                )
                save_action.setEnabled(False)
                save_action.setToolTip(tt)
                menu.setToolTipsVisible(True)
            menu.addAction(_("Discard Image"), self._discard_image)
            menu.addSeparator()
            menu.addAction(_("Clear History"), self._clear_all)
            menu.exec(ensure(self.viewport()).mapToGlobal(pos))

    def _show_context_menu_dropdown(self):
        pos = self._context_button.pos()
        pos.setY(pos.y() + self._context_button.height())
        self._show_context_menu(ensure(self.viewport()).mapFrom(self, pos))

    def _copy_prompt(self, evaluated=False):
        if job := self.selected_job:
            copy_job_prompt(self._model, job, evaluated)

    def _copy_prompt_evaluated(self):
        self._copy_prompt(evaluated=True)

    def _copy_strength(self):
        if job := self.selected_job:
            copy_job_strength(self._model, job)

    def _copy_style(self):
        if job := self.selected_job:
            copy_job_style(self._model, job)

    def _copy_seed(self):
        if job := self.selected_job:
            copy_job_seed(self._model, job)

    def _info_to_clipboard(self):
        if job := self.selected_job:
            copy_job_info(job)

    def _save_image(self):
        items = self.selectedItems()
        for item in items:
            job_id, image_index = self.item_info(item)
            self._model.save_result(job_id, image_index)

    def _discard_image(self, confirm=True):
        confirm = confirm and settings.confirm_discard_image
        reply = QMessageBox.StandardButton.Yes
        if confirm:
            reply = QMessageBox.warning(
                self,
                _("Discard Image"),
                _("Are you sure you want to discard the selected images?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
        if reply == QMessageBox.StandardButton.Yes:
            items = self.selectedItems()
            next_item = self.row(items[0]) if len(items) > 0 else -1
            for item in items:
                job_id, image_index = self.item_info(item)
                self._model.jobs.discard(job_id, image_index)
            if next_item >= 0:
                self.setCurrentRow(next_item, QItemSelectionModel.SelectionFlag.Current)

    def _clear_all(self):
        if discard_all_results(self._model, self):
            self.clear()


class AnimatedListItem(QListWidgetItem):
    def __init__(self, images: list[QIcon]):
        super().__init__(images[0], None, type=QListWidgetItem.ItemType.UserType)
        self._images = images
        self._current = 0
        self._is_running = False
        self._timer = QTimer()
        self._timer.setSingleShot(False)
        self._timer.timeout.connect(self._next_frame)

    def start_animation(self):
        if not self._is_running:
            self._is_running = True
            self._timer.start(40)

    def stop_animation(self):
        if self._is_running:
            self._timer.stop()
            self._is_running = False
            self._current = 0
            self.setIcon(self._images[self._current])

    def _next_frame(self):
        self._current = (self._current + 1) % len(self._images)
        self.setIcon(self._images[self._current])


class PreviewReelInfo(QLabel):
    """Single-line tooltip overlaying the bottom part of the PreviewReel."""

    def __init__(self, parent: QWidget):
        # a child widget (not a window) that is transparent for mouse events: Qt skips
        # it during hit-testing, so the reel keeps hover state while the label is visible
        # (Qt.WindowType.WindowTransparentForInput is not supported on all platforms)
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # allows alpha background
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._background = QGuiApplication.palette().color(QPalette.ColorRole.Base)
        self._background.setAlphaF(0.85)
        self.setStyleSheet("QLabel { padding: 2px 4px; }")

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._background)
        painter.end()
        super().paintEvent(a0)

    def show_below(self, widget: QWidget, text: str):
        theme.set_text_clipped(self, text.replace("\n", " "), 14)
        self.setFixedWidth(widget.width() - 4)
        self.move(1, widget.height() - self.height() - 2)
        self.show()


class PreviewReelItem:
    """One square in the PreviewReel: a job result thumbnail, or a placeholder
    for a job that is still queued or executing."""

    class Kind(Enum):
        result = 0
        executing = 1
        queued = 2

    def __init__(self, kind: PreviewReelItem.Kind, job: Job, index=0):
        self.kind = kind
        self.job = job
        self.index = index  # image index within the job (results only)
        self.frames: list[QPixmap] = []  # multiple frames for animation results
        self.current_frame = 0
        self.icon_name = ""  # placeholders only
        self.input: QPixmap | None = None  # input image thumbnail, placeholders only


class QueueCountOverlay(QWidget):
    """Shows the number of queued jobs in the top-left corner of the PreviewReel."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._count = 0
        self._color = QColor(theme.highlight)
        self._background_color = QGuiApplication.palette().color(QPalette.ColorRole.Base)
        self._background_color.setAlphaF(0.9)
        font = self.font()
        font.setBold(True)
        self.setFont(font)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        metrics = self.fontMetrics()
        self._height = metrics.height() + 4
        self._icon_size = self._height + 2
        self._width = self._icon_size + 6 + metrics.horizontalAdvance("999")
        self._pixmap = theme.icon("queue-inactive").pixmap(self._icon_size, self._icon_size)
        self.setFixedSize(self._width, self._height)
        self.hide()

    def set_count(self, count: int):
        if count != self._count:
            metrics = self.fontMetrics()
            self._count = count
            self._width = self._icon_size + 6 + metrics.horizontalAdvance(str(self._count))
            self.setVisible(count > 0)
            self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self._width, self._height, 4, 4)
        # painter.setBrush(QBrush(self._background_color))
        painter.fillPath(path, QBrush(self._background_color))
        painter.drawPixmap(0, 0, self._pixmap)
        painter.setPen(self._color)
        painter.drawText(
            QRect(self._icon_size + 2, 1, self.width() - self._icon_size, self._height),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            str(self._count),
        )
        painter.end()


class ReelNavButton(QWidget):
    """Overlay navigation button for the PreviewReel."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget, direction: int):
        super().__init__(parent)
        self._direction = direction  # -1 = left, +1 = right
        base = QGuiApplication.palette().color(QPalette.ColorRole.Base)
        self._gradient_start_hover = QColor(base)
        self._gradient_start = theme.relative_color(self._gradient_start_hover, 120)
        self._gradient_end = QColor(self._gradient_start)
        self._gradient_start.setAlphaF(0.6)
        self._gradient_start_hover.setAlphaF(0.8)
        self._icon_color_hover = QGuiApplication.palette().color(QPalette.ColorRole.Text)
        self._icon_color = theme.relative_color(self._icon_color_hover, 120)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._direction < 0:  # right-to-left: opaque at the outer (left) edge
            gradient = QLinearGradient(self.width(), 0, 0, 0)
        else:  # left-to-right: opaque at the outer (right) edge
            gradient = QLinearGradient(0, 0, self.width(), 0)
        hover = self.underMouse()
        gradient.setColorAt(0, self._gradient_start_hover if hover else self._gradient_start)
        gradient.setColorAt(1, self._gradient_end)
        rect = self.rect()
        painter.fillRect(rect, QBrush(gradient))
        w, h = rect.width(), rect.height()
        tw, th = int(w * 0.35), int(h * 0.4)
        cx, cy = rect.center().x(), rect.center().y()
        if self._direction < 0:
            points = [
                QPoint(cx - tw // 2, cy),
                QPoint(cx + tw // 2, cy - th // 2),
                QPoint(cx + tw // 2, cy + th // 2),
            ]
        else:
            points = [
                QPoint(cx + tw // 2, cy),
                QPoint(cx - tw // 2, cy - th // 2),
                QPoint(cx - tw // 2, cy + th // 2),
            ]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._icon_color_hover if hover else self._icon_color)
        painter.drawPolygon(QPolygon(points))
        painter.end()

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is not None and a0.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            a0.accept()

    def enterEvent(self, event: QEnterEvent | None) -> None:
        self.update()
        return super().enterEvent(event)

    def leaveEvent(self, a0: QEvent | None) -> None:
        self.update()
        return super().leaveEvent(a0)


class PreviewReel(QWidget):
    """Horizontal reel showing queued and in-progress jobs followed by the latest results.

    Hovering a result image previews it on the canvas, clicking applies it. Content is
    scrolled by grabbing with the mouse or via the wheel (one wheel step per item)."""

    thumb_size = 72
    _nav_button_width = 24
    _star = HistoryWidget._applied_icon
    _background_top = QColor(theme.base).darker(120)
    _background_bottom = QColor(theme.base).lighter(120)
    _overlay = QColor(0, 0, 0, 128)  # 50% black on top of placeholder input images
    _active_border = QColor(255, 255, 255) if theme.is_dark else QColor(0, 0, 0)
    _active_overlay = QColor(theme.active)
    _active_overlay.setAlphaF(0.3)
    _cancel_color = QColor(theme.red)
    _kinds_with_input = (WorkflowKind.inpaint, WorkflowKind.refine, WorkflowKind.refine_region)
    _handled_keys = (
        Qt.Key.Key_Left,
        Qt.Key.Key_Right,
        Qt.Key.Key_Return,
        Qt.Key.Key_Enter,
        Qt.Key.Key_Space,
    )

    def __init__(self, parent: QWidget | None):
        super().__init__(parent)
        self._model = root.active_model
        self._connections: list[QMetaObject.Connection] = []
        self._items: list[PreviewReelItem] = []
        self._active: PreviewReelItem | None = None
        self._hover_item: PreviewReelItem | None = None  # item under the mouse cursor
        self._space_toggle_item: PreviewReelItem | None = None  # restored on Space press
        self._info: PreviewReelInfo | None = None
        self._info_item: PreviewReelItem | None = None  # item whose info tooltip is shown
        self._preview_owned = False  # whether canvas preview was triggered by this widget
        self._offset = 0.0  # horizontal scroll position in pixels
        self._press_pos: QPoint | None = None
        self._press_offset = 0.0
        self._dragging = False
        self._pulse_phase = 0.0
        self._icon_cache: dict[tuple[str, int, int], QPixmap] = {}

        scale = theme.screen_scale(self, QSize(self.thumb_size, self.thumb_size))
        self._thumb = scale.width()
        self._pad = 2

        gradient = QLinearGradient(0, 1 + self._pad, 0, 1 + self._pad + self._thumb)
        gradient.setColorAt(0, self._background_top)
        gradient.setColorAt(1, self._background_bottom)
        self._item_background = QBrush(gradient)

        self._scroll_anim = QPropertyAnimation(self, b"scroll_offset", self)
        self._scroll_anim.setDuration(120)
        self._scroll_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(50)
        self._pulse_timer.timeout.connect(self._pulse)
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(40)
        self._frame_timer.timeout.connect(self._next_frame)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self._thumb + 2 * self._pad + 2)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        self._nav_left = ReelNavButton(self, -1)
        self._nav_right = ReelNavButton(self, 1)
        self._nav_left.clicked.connect(self._nav_left_clicked)
        self._nav_right.clicked.connect(self._nav_right_clicked)
        self._nav_left.hide()
        self._nav_right.hide()
        self._queue_count = QueueCountOverlay(self)
        self._queue_count.move(self._nav_button_width + 3, 1)

    def sizeHint(self):
        # the widget keeps its final height even when it is empty
        return QSize(4 * self._stride, self._thumb + 2 * self._pad + 2)

    @pyqtProperty(float)
    def scroll_offset(self):  # type: ignore
        return self._offset

    @scroll_offset.setter  # type: ignore
    def scroll_offset(self, value: float):
        self._set_offset(value)

    @property
    def model_(self):
        return self._model

    @model_.setter
    def model_(self, model: DocumentModel):
        Binding.disconnect_all(self._connections)
        self._model = model
        jobs = model.jobs
        self._connections = [
            jobs.count_changed.connect(self._sync_placeholders),
            jobs.job_finished.connect(self._on_job_finished),
            jobs.job_discarded.connect(self._on_job_discarded),
            jobs.result_used.connect(self._on_result_used),
            jobs.result_discarded.connect(self._on_result_discarded),
            jobs.selection_changed.connect(self._on_selection_changed),
        ]
        self.rebuild()

    def rebuild(self):
        self._scroll_anim.stop()
        self._items.clear()
        self._active = None
        self._hover_item = None
        self._space_toggle_item = None
        self._hide_info()
        self._preview_owned = False
        self._frame_timer.stop()
        # both placeholders and results are ordered newest first (from the left)
        placeholders: list[PreviewReelItem] = []
        results: list[PreviewReelItem] = []
        for job in self._model.jobs:
            if not self._accepts(job):
                continue
            if job.state in (JobState.queued, JobState.executing):
                placeholders[0:0] = self._make_placeholders(job)
            elif job.state is JobState.finished:
                results[0:0] = self._make_result_items(job)
        self._items = placeholders + results
        self._offset = 0
        self._update_pulse_timer()
        self._update_nav_buttons()
        self.update()

    @staticmethod
    def _accepts(job: Job):
        return job.kind in (JobKind.diffusion, JobKind.animation)

    def _make_placeholders(self, job: Job):
        if job.state is JobState.executing:
            kind = PreviewReelItem.Kind.executing
        else:
            kind = PreviewReelItem.Kind.queued
        input_image = None
        if job.params.workflow_kind in self._kinds_with_input:
            try:
                image = self._model.document.get_image(job.params.bounds)
                input_image = Image.scale_to_fit(
                    image, Extent(2 * self._thumb, 2 * self._thumb)
                ).to_pixmap()
            except Exception:
                input_image = None  # no visible layers: fall back to white
        count = (
            1 if job.kind is JobKind.animation or job.params.is_layered else job.params.result_count
        )
        items = []
        for index in range(count):
            item = PreviewReelItem(kind, job, index)
            item.icon_name = (
                "queue-waiting" if kind is PreviewReelItem.Kind.queued else self._job_icon_name(job)
            )
            item.input = input_image
            items.append(item)
        return items

    def _make_result_items(self, job: Job):
        if job.kind is JobKind.animation:
            frames = [self._result_thumb(job, i) for i in range(len(job.results))]
            if not frames:
                return []
            item = PreviewReelItem(PreviewReelItem.Kind.result, job)
            item.frames = frames
            return [item]
        if job.params.is_layered:
            item = PreviewReelItem(PreviewReelItem.Kind.result, job)
            item.frames = [self._result_thumb(job, 0)]
            return [item]
        items = []
        for i in range(len(job.results)):
            item = PreviewReelItem(PreviewReelItem.Kind.result, job, i)
            item.frames = [self._result_thumb(job, i)]
            items.append(item)
        return items

    def _result_thumb(self, job: Job, index: int):
        # Use 2x thumb size for good quality on high-DPI screens
        size = 2 * self._thumb
        thumb = Image.scale_to_fit(job.results[index], Extent(size, size))
        if job.result_was_used(index):  # add tiny star icon to mark used results
            star_size = max(24, thumb.extent.width // 8)
            star = Image.scale(self._star, Extent(star_size, star_size))
            thumb.draw_image(star, offset=(thumb.extent.width - star_size - 4, 4))
        return thumb.to_pixmap()

    @staticmethod
    def _job_icon_name(job: Job):
        if job.kind is JobKind.animation:
            return "workspace-animation"
        kind = job.params.workflow_kind
        if kind in (WorkflowKind.refine, WorkflowKind.refine_region):
            return "refine"
        elif kind is WorkflowKind.custom:
            return "workspace-custom"
        return "workspace-generation"

    # -- job queue events ---------------------------------------------------

    def _sync_placeholders(self):
        open_states = (JobState.queued, JobState.executing)
        open_jobs = [j for j in self._model.jobs if self._accepts(j) and j.state in open_states]
        changed = False

        for item in list(self._items):
            if item.kind is PreviewReelItem.Kind.result:
                continue
            if item.job not in open_jobs:
                if self._active is item:
                    self._set_active(None)
                if self._info_item is item:
                    self._hide_info()
                if self._hover_item is item:
                    self._hover_item = None
                if self._space_toggle_item is item:
                    self._space_toggle_item = None
                self._items.remove(item)
                changed = True
            else:
                kind = PreviewReelItem.Kind.queued
                if item.job.state is JobState.executing:
                    kind = PreviewReelItem.Kind.executing
                if item.kind is not kind:  # job started executing
                    item.kind = kind
                    item.icon_name = self._job_icon_name(item.job)
                    changed = True

        missing = [j for j in open_jobs if all(item.job is not j for item in self._items)]
        if missing:
            # Insert in chronological order so that the newest job ends up leftmost.
            new_items = [item for job in missing for item in self._make_placeholders(job)]
            self._items[0:0] = new_items
            # New items appear on the left. Keep the view scrolled all the way left if it
            # already is, otherwise shift the offset so existing items don't move.
            self._scroll_anim.stop()
            if self._offset > 1:
                self._set_offset(self._offset + len(new_items) * self._stride)
            changed = True

        if changed:
            self._set_offset(self._offset)  # clamp to valid range
            self._update_pulse_timer()
            self._refresh_hover()
            self.update()

    def _on_job_finished(self, job: Job):
        if not self._accepts(job):
            return
        pos = next((i for i, item in enumerate(self._items) if item.job is job), -1)
        if pos < 0:
            return
        # Replace every placeholder with the job's results in place.
        placeholders = []
        while pos + len(placeholders) < len(self._items):
            item = self._items[pos + len(placeholders)]
            if item.job is not job or item.kind is PreviewReelItem.Kind.result:
                break
            placeholders.append(item)
        if self._hover_item in placeholders:
            self._hover_item = None
        if self._info_item in placeholders:
            self._hide_info()
        if self._space_toggle_item in placeholders:
            self._space_toggle_item = None
        self._items[pos : pos + len(placeholders)] = self._make_result_items(job)
        self._set_offset(self._offset)
        self._update_pulse_timer()
        self._refresh_hover()
        self.update()

    def _on_job_discarded(self, job: Job):
        if not any(item.job is job for item in self._items):
            return
        if self._active is not None and self._active.job is job:
            self._set_active(None)
        if self._info_item is not None and self._info_item.job is job:
            self._hide_info()
        if self._hover_item is not None and self._hover_item.job is job:
            self._hover_item = None
        if self._space_toggle_item is not None and self._space_toggle_item.job is job:
            self._space_toggle_item = None
        self._items = [item for item in self._items if item.job is not job]
        self._set_offset(self._offset)
        self._update_pulse_timer()
        self.update()

    def _on_result_used(self, id: JobQueue.Item):
        for item in self._items:
            if item.kind is PreviewReelItem.Kind.result and item.job.id == id.job:
                if len(item.frames) > 1:
                    item.frames[id.image] = self._result_thumb(item.job, id.image)
                elif item.index == id.image:
                    item.frames = [self._result_thumb(item.job, id.image)]
        self.update()

    def _on_result_discarded(self, id: JobQueue.Item):
        for item in list(self._items):
            if item.kind is not PreviewReelItem.Kind.result or item.job.id != id.job:
                continue
            if item.index == id.image:
                if self._active is item:
                    self._set_active(None)
                if self._info_item is item:
                    self._hide_info()
                if self._hover_item is item:
                    self._hover_item = None
                if self._space_toggle_item is item:
                    self._space_toggle_item = None
                self._items.remove(item)
            elif item.index > id.image:
                item.index -= 1
        self._set_offset(self._offset)
        self.update()

    def _on_selection_changed(self):
        selection = self._model.jobs.selection
        if self._active is not None and self._active.job.id is not None:
            if selection == [JobQueue.Item(self._active.job.id, self._active.index)]:
                return  # selection matches the active item (usually triggered by us)
        self._active = None
        self._preview_owned = False
        self._frame_timer.stop()
        if len(selection) == 1:  # adopt external selection (eg. preview of a finished job)
            match = next(
                (
                    item
                    for item in self._items
                    if item.kind is PreviewReelItem.Kind.result
                    and item.job.id == selection[0].job
                    and item.index == selection[0].image
                ),
                None,
            )
            if match is not None:
                self._active = match
                self._space_toggle_item = None
                self._preview_owned = True
                match.current_frame = 0
                if len(match.frames) > 1:
                    self._frame_timer.start()
        self.update()

    # -- active item / preview -----------------------------------------------

    def _set_active(self, item: PreviewReelItem | None):
        if item is self._active:
            return
        self._active = item
        if item is not None:
            self._space_toggle_item = None  # active item changed, reset Space toggle state
            item.current_frame = 0
            if len(item.frames) > 1:
                self._frame_timer.start()
            self._model.jobs.selection = [JobQueue.Item(ensure(item.job.id), item.index)]
            self._preview_owned = True
        else:
            self._frame_timer.stop()
            if self._preview_owned:
                self._preview_owned = False
                if self._model.jobs.selection:
                    self._model.jobs.selection = []
        self.update()

    def _update_hover(self, pos: QPoint):
        item = self._item_at(pos, self._hover_offset)
        if item is None:
            return  # keep the previous item active between items to avoid flicker
        self._hover_item = item
        self._show_info(item)
        if item.kind is PreviewReelItem.Kind.result:
            self._set_active(item)
        else:  # placeholders cannot be active
            self._set_active(None)

    def _show_info(self, item: PreviewReelItem):
        if item is self._info_item:
            return
        self._info_item = item
        # single line above the widget: "{timestamp} - {strength%} - {prompt}..."
        job = item.job
        strength = job.params.strength
        strength_text = f"{strength * 100:.0f}% - " if strength != 1.0 else ""
        prompt = job.params.name if job.params.name != "" else _("<no prompt>")
        text = f"{job.timestamp.astimezone():%H:%M} - {strength_text}{prompt}"
        if self._info is None:
            self._info = PreviewReelInfo(self)
            self._nav_left.raise_()
            self._nav_right.raise_()
            self._queue_count.raise_()
        self._info.show_below(self, text)

    def _hide_info(self):
        self._info_item = None
        if self._info is not None:
            self._info.hide()

    def _refresh_hover(self):
        if self.underMouse() and self._press_pos is None:
            pos = self.mapFromGlobal(QCursor.pos())
            if self.childAt(pos) is not None:
                return
            if self._item_at(pos, self._hover_offset) is None:
                self._hide_info()
            self._update_hover(pos)

    # -- geometry --------------------------------------------------------------

    @property
    def _stride(self):
        return self._thumb + self._pad

    @property
    def _items_per_view(self):
        return max(1, (self.width() - 2) // self._stride)

    def _max_offset(self):
        content = len(self._items) * self._stride + self._pad
        return max(0.0, content - (self.width() - 2))

    @property
    def _hover_offset(self):
        # while a scroll animation is running, hover targets its final position
        if self._scroll_anim.state() is QAbstractAnimation.State.Running:
            return float(self._scroll_anim.endValue())
        return self._offset

    def _item_rect(self, i: int, offset: float | None = None):
        offset = self._offset if offset is None else offset
        x = 1 + self._pad + i * self._stride - round(offset)
        return QRect(x, 1 + self._pad, self._thumb, self._thumb)

    def _item_at(self, pos: QPoint, offset: float | None = None):
        for i, item in enumerate(self._items):
            if self._item_rect(i, offset).contains(pos):
                return item
        return None

    def _is_item_visible(self, i: int):
        # an item is considered visible if more than a 1px sliver is in the content area
        rect = self._item_rect(i)
        return rect.right() > 1 and rect.left() < self.width() - 1

    def _queued_count(self):
        return sum(1 for item in self._items if item.kind is PreviewReelItem.Kind.queued)

    def _update_nav_buttons(self):
        btn_w = self._nav_button_width
        y = self._pad
        self._nav_left.setGeometry(1, y, btn_w, self._thumb + 1)
        self._nav_right.setGeometry(self.width() - btn_w - 1, y, btn_w, self._thumb + 1)
        left_visible = self._offset > 0
        right_visible = self._offset < self._max_offset()
        if self._nav_left.isHidden() == left_visible:
            self._nav_left.setVisible(left_visible)
            self._nav_left.raise_()
            self._queue_count.raise_()
        if self._nav_right.isHidden() == right_visible:
            self._nav_right.setVisible(right_visible)
            self._nav_right.raise_()
            self._queue_count.raise_()
        self._queue_count.set_count(self._queued_count())

    def _nav_left_clicked(self):
        first = next(
            (
                i
                for i, item in enumerate(self._items)
                if item.kind is not PreviewReelItem.Kind.queued
            ),
            None,
        )
        if first is not None and self._item_rect(first).right() <= 1:
            self._animate_to(first * self._stride)
        else:
            self._animate_to(0)

    def _nav_right_clicked(self):
        first = next(
            (
                i
                for i, item in enumerate(self._items)
                if item.kind is not PreviewReelItem.Kind.queued
            ),
            None,
        )
        if first is not None and any(
            item.kind is PreviewReelItem.Kind.queued and self._is_item_visible(i)
            for i, item in enumerate(self._items)
        ):
            self._animate_to(first * self._stride)
        else:
            step = max(1, self._items_per_view - 1) * self._stride
            self._animate_to(self._offset + step)

    def _set_offset(self, value: float):
        value = min(max(value, 0.0), self._max_offset())
        if value != self._offset:
            self._offset = value
            self._update_nav_buttons()
            self._refresh_hover()  # items may have moved under a stationary cursor
            self.update()
        else:
            self._update_nav_buttons()

    def _animate_to(self, target: float):
        target = min(max(target, 0.0), self._max_offset())
        self._scroll_anim.stop()
        self._scroll_anim.setStartValue(self._offset)
        self._scroll_anim.setEndValue(target)
        self._scroll_anim.start()

    # -- painting ---------------------------------------------------------------

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setPen(QColor(theme.grey))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        content = self.rect().adjusted(1, 1, -2, -2)
        if not self._items:
            painter.drawText(content, Qt.AlignmentFlag.AlignCenter, _("No generated images yet."))
        else:
            painter.setClipRect(content)
            for i, item in enumerate(self._items):
                rect = self._item_rect(i)
                if rect.right() >= 0 and rect.left() <= self.width():
                    self._paint_item(painter, item, rect)
        painter.end()

    def _paint_item(self, painter: QPainter, item: PreviewReelItem, rect: QRect):
        painter.fillRect(rect, self._item_background)
        if item.kind is PreviewReelItem.Kind.result:
            frame = item.frames[item.current_frame]
            painter.drawPixmap(self._fit_rect(frame.size(), rect), frame, frame.rect())
        else:
            if item.input is not None:
                target = self._fit_rect(item.input.size(), rect)
                painter.drawPixmap(target, item.input, item.input.rect())
                painter.fillRect(rect, self._overlay)
            self._paint_icon(painter, item, rect)
        if item is self._active:
            painter.fillRect(rect, self._active_overlay)
            painter.setPen(QPen(self._active_border, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect.adjusted(0, 0, -1, -1))

    def _paint_icon(self, painter: QPainter, item: PreviewReelItem, rect: QRect):
        if item is self._hover_item:
            self._paint_cancel_icon(painter, rect)
            return
        size = int(rect.width() * 0.4)
        if item.kind is PreviewReelItem.Kind.queued:
            pixmap = theme.icon(item.icon_name).pixmap(QSize(size, size))
        else:
            t = (sin(self._pulse_phase) + 1) * 0.5
            base = QColor(theme.grey)
            color = QColor(
                base.red() + round((255 - base.red()) * t),
                base.green() + round((255 - base.green()) * t),
                base.blue() + round((255 - base.blue()) * t),
            )
            pixmap = self._tinted_icon(item.icon_name, color, size)
        pos = rect.center() - QPoint(size // 2, size // 2)
        painter.drawPixmap(pos, pixmap)

    def _paint_cancel_icon(self, painter: QPainter, rect: QRect):
        size = int(rect.width() * 0.4)
        pen = QPen(self._cancel_color, max(2, size // 8))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        center = rect.center()
        half = size // 2
        painter.drawLine(center.x() - half, center.y() - half, center.x() + half, center.y() + half)
        painter.drawLine(center.x() + half, center.y() - half, center.x() - half, center.y() + half)

    def _tinted_icon(self, name: str, color: QColor, size: int):
        key = (name, size, color.red() // 16)  # tint color is always a shade of grey
        if key not in self._icon_cache:
            image = theme.icon(name).pixmap(QSize(size, size)).toImage()
            painter = QPainter(image)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(image.rect(), color)
            painter.end()
            self._icon_cache[key] = QPixmap.fromImage(image)
        return self._icon_cache[key]

    @staticmethod
    def _fit_rect(size: QSize, rect: QRect):
        scaled = size.scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = rect.x() + (rect.width() - scaled.width()) // 2
        y = rect.y() + (rect.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

    # -- animations ----------------------------------------------------------------

    def _pulse(self):
        self._pulse_phase = (self._pulse_phase + 0.12) % (2 * pi)
        self.update()

    def _update_pulse_timer(self):
        has_executing = any(item.kind is PreviewReelItem.Kind.executing for item in self._items)
        if has_executing and not self._pulse_timer.isActive():
            self._pulse_timer.start()
        elif not has_executing and self._pulse_timer.isActive():
            self._pulse_timer.stop()

    def _next_frame(self):
        item = self._active
        if item is not None and len(item.frames) > 1:
            item.current_frame = (item.current_frame + 1) % len(item.frames)
            self.update()
        else:
            self._frame_timer.stop()

    # -- input -----------------------------------------------------------------------

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is not None and a0.button() == Qt.MouseButton.LeftButton:
            self._press_pos = a0.pos()
            self._press_offset = self._offset
            self._dragging = False
            self._scroll_anim.stop()

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is None:
            return
        if self._press_pos is not None and a0.buttons() & Qt.MouseButton.LeftButton:
            delta = a0.pos().x() - self._press_pos.x()
            if self._dragging or abs(delta) > 4:
                self._dragging = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self._set_offset(self._press_offset - delta)
        else:
            self._update_hover(a0.pos())

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is None or a0.button() != Qt.MouseButton.LeftButton or self._press_pos is None:
            return
        self._press_pos = None
        was_dragging = self._dragging
        self._dragging = False
        self.unsetCursor()
        if was_dragging:
            self._animate_to(round(self._offset / self._stride) * self._stride)
        else:
            self._click(self._item_at(a0.pos()), a0.modifiers())
        self._update_hover(a0.pos())

    def wheelEvent(self, a0: QWheelEvent | None) -> None:
        if a0 is None or not self._items:
            return
        delta = a0.angleDelta().y()
        if delta == 0:
            a0.accept()
            return
        direction = -1 if delta > 0 else 1
        if self._scroll_by(direction):
            # activate the item the cursor will hover after scrolling, without
            # waiting for the scroll animation to finish
            self._update_hover(a0.position().toPoint())
        a0.accept()

    def enterEvent(self, event: QEnterEvent | None) -> None:
        if self.isVisible():
            self.grabKeyboard()
        super().enterEvent(event)

    def leaveEvent(self, a0: QEvent | None) -> None:
        self._hover_item = None
        self._set_active(None)
        self._hide_info()
        self.releaseKeyboard()
        super().leaveEvent(a0)

    def focusOutEvent(self, a0: QFocusEvent | None) -> None:
        self._set_active(None)
        self._hide_info()
        super().focusOutEvent(a0)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        self._set_offset(self._offset)  # clamp to valid range
        self._hide_info()
        super().resizeEvent(a0)

    def hideEvent(self, a0: QHideEvent | None) -> None:
        self._hide_info()
        self.releaseKeyboard()
        super().hideEvent(a0)

    def event(self, a0: QEvent | None) -> bool:
        assert a0 is not None
        if a0.type() == QEvent.Type.ShortcutOverride:
            assert isinstance(a0, QKeyEvent)
            if a0.key() in self._handled_keys:
                a0.accept()
        return super().event(a0)

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 is None:
            return
        if a0.key() == Qt.Key.Key_Left:
            self._scroll_by(-1)
            a0.accept()
        elif a0.key() == Qt.Key.Key_Right:
            self._scroll_by(1)
            a0.accept()
        elif a0.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._click(self._current_item, a0.modifiers())
            a0.accept()
        elif a0.key() == Qt.Key.Key_Space:
            self._toggle_active()
            a0.accept()
        else:
            super().keyPressEvent(a0)

    # -- actions --------------------------------------------------------------------

    def _scroll_by(self, direction: int) -> bool:
        """Scroll by one item in the given direction (+1 right, -1 left).

        Returns True if a scroll was started, False if the reel was already at the
        edge and the adjacent item was activated instead."""
        if not self._items:
            return False
        if self._scroll_anim.state() is QAbstractAnimation.State.Running:
            base = float(self._scroll_anim.endValue())
        else:
            base = self._offset
        max_offset = self._max_offset()
        if direction > 0 and base >= max_offset - 0.5:
            self._activate_adjacent(1)
            return False
        if direction < 0 and base <= 0.5:
            self._activate_adjacent(-1)
            return False
        item = round(base / self._stride)  # one step moves exactly one item
        self._animate_to((item + direction) * self._stride)
        return True

    def _activate_adjacent(self, direction: int):
        if self._active is None:
            return
        index = next((i for i, item in enumerate(self._items) if item is self._active), -1)
        if index < 0:
            return
        i = index + direction
        while 0 <= i < len(self._items):
            item = self._items[i]
            if item.kind is PreviewReelItem.Kind.result:
                self._set_active(item)
                return
            i += direction

    @property
    def _current_item(self):
        if self._active is not None:
            return self._active
        return self._hover_item

    def _toggle_active(self):
        if self._active is not None:
            self._space_toggle_item = self._active
            self._set_active(None)
        elif self._space_toggle_item is not None and self._space_toggle_item in self._items:
            item = self._space_toggle_item
            self._space_toggle_item = None
            self._set_active(item)

    def _click(self, item: PreviewReelItem | None, modifiers: Qt.KeyboardModifier):
        if item is None:
            self._set_active(None)
            self._hide_info()
            return
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if item.kind is PreviewReelItem.Kind.result:
            self._apply(item, shift)
        elif item.kind is PreviewReelItem.Kind.executing:
            if shift:
                self._model.cancel(active=True)
            else:
                self._cancel_job(item.job)
        elif shift:
            self._model.cancel(queued=True)
        else:
            self._cancel_job(item.job)

    def _apply(self, item: PreviewReelItem, alternate: bool):
        behavior = None
        if alternate:  # use replace if the primary behavior is a new layer, and vice versa
            primary = settings.apply_behavior
            if primary is ApplyBehavior.replace:
                behavior = ApplyBehavior.layer
            else:
                behavior = ApplyBehavior.replace
        if item.job.id is not None:
            self._model.apply_generated_result(item.job.id, item.index, behavior)

    def _cancel_job(self, job: Job):
        if job.state is JobState.queued:
            if job.id is not None and root.connection.client_if_connected is not None:
                root.connection.cancel([job.id])
            self._model.jobs.remove(job)
        elif job.state is JobState.executing:
            self._model.cancel(active=True)

    def _show_context_menu(self, pos: QPoint):
        item = self._item_at(pos)
        if item is None:
            return
        menu = QMenu(self)
        if item.kind is PreviewReelItem.Kind.result:
            self._build_result_menu(menu, item)
        else:
            self._build_placeholder_menu(menu, item)
        menu.exec(self.mapToGlobal(pos))

    def _build_result_menu(self, menu: QMenu, item: PreviewReelItem):
        job = item.job
        menu.addAction(_("Copy Prompt"), lambda: copy_job_prompt(self._model, job))
        menu.addAction(
            _("Copy Prompt (Evaluated)"),
            lambda: copy_job_prompt(self._model, job, evaluated=True),
        )
        menu.addAction(_("Copy Strength"), lambda: copy_job_strength(self._model, job))
        style_action = ensure(
            menu.addAction(_("Copy Style"), lambda: copy_job_style(self._model, job))
        )
        if Styles.list().find(job.params.style) is None:
            style_action.setEnabled(False)
        menu.addAction(_("Copy Seed"), lambda: copy_job_seed(self._model, job))
        menu.addAction(_("Info to Clipboard"), lambda: copy_job_info(job))
        menu.addSeparator()
        save_action = ensure(menu.addAction(_("Save Image"), lambda: self._save_image(item)))
        if self._model.document.filename == "":
            tt = _(
                "Save as separate image to the same folder as the document.\nMust save the document first!"
            )
            save_action.setEnabled(False)
            save_action.setToolTip(tt)
            menu.setToolTipsVisible(True)
        menu.addAction(_("Discard Image"), lambda: self._discard_image(item))
        menu.addSeparator()
        menu.addAction(_("Open History"), self._open_history)
        menu.addAction(_("Clear History"), self._clear_all)

    def _build_placeholder_menu(self, menu: QMenu, item: PreviewReelItem):
        menu.addAction(_("Info to Clipboard"), lambda: copy_job_info(item.job))
        menu.addSeparator()
        menu.addAction(_("Cancel"), lambda: self._cancel_job(item.job))
        queued = ensure(menu.addAction(_("Cancel queued"), lambda: self._model.cancel(queued=True)))
        queued.setEnabled(self._model.jobs.count(JobState.queued) > 0)
        all_ = ensure(
            menu.addAction(_("Cancel all"), lambda: self._model.cancel(active=True, queued=True))
        )
        all_.setEnabled(
            self._model.jobs.any_executing() or self._model.jobs.count(JobState.queued) > 0
        )

    def _save_image(self, item: PreviewReelItem):
        self._model.save_result(ensure(item.job.id), item.index)

    def _discard_image(self, item: PreviewReelItem):
        reply = QMessageBox.StandardButton.Yes
        if settings.confirm_discard_image:
            reply = QMessageBox.warning(
                self,
                _("Discard Image"),
                _("Are you sure you want to discard the selected images?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
        if reply == QMessageBox.StandardButton.Yes:
            self._model.jobs.discard(ensure(item.job.id), item.index)

    def _clear_all(self):
        discard_all_results(self._model, self)

    def _open_history(self):
        for docker in Krita.instance().dockers():
            if docker.objectName() == "aiImageHistory":
                docker.setVisible(True)
                docker.raise_()
                return
