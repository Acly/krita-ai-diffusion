from __future__ import annotations

import json
import time
from textwrap import wrap as wrap_text
from typing import ClassVar, cast

from krita import Krita
from PyQt5.QtCore import (
    QEvent,
    QItemSelectionModel,
    QMetaObject,
    QPoint,
    QSize,
    Qt,
    QTimer,
    QUuid,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QColor,
    QGuiApplication,
    QIcon,
    QImage,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPalette,
)
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..backend.api import InpaintContext, SegmentMode
from ..backend.resources import Arch
from ..backend.workflow import FillMode, InpaintMode
from ..image import Bounds, Extent, Image
from ..localization import translate as _
from ..model.jobs import Job, JobKind, JobParams, JobQueue, JobState
from ..model.model import DocumentModel, ProgressKind, Workspace
from ..model.properties import Bind, Binding, bind, bind_combo, bind_toggle
from ..model.region import RootRegion
from ..model.root import root
from ..settings import settings
from ..style import Styles
from ..text import create_mode_label
from ..util import client_logger as log
from ..util import ensure, flatten, sequence_equal
from . import actions, theme
from .region import RegionPromptWidget
from .widget import (
    ErrorBox,
    GenerateButton,
    LayerCountWidget,
    QueueButton,
    StrengthWidget,
    StyleSelectWidget,
    WorkspaceSelectWidget,
    create_wide_tool_button,
    wildcard_combination_count,
)


def _format_duration(seconds: float):
    if seconds < 10:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes} min {seconds:02} s" if minutes else f"{seconds} s"


def _format_bytes(size: int):
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / 1024:.0f} KB"


def _tint_image(image: Image, color: QColor) -> Image:
    """Recolor the opaque pixels of an icon-like image, keeping its alpha shape."""
    qimg = image._qimage.copy()
    painter = QPainter(qimg)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(qimg.rect(), color)
    painter.end()
    return Image(qimg)


def _tint_icon(name: str, color: QColor, size: int = 32) -> QIcon:
    """A themed icon recolored to `color` (e.g. to highlight an active button)."""
    pixmap = theme.icon(name).pixmap(size, size)
    painter = QPainter(pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), color)
    painter.end()
    return QIcon(pixmap)


def _make_badge(icon_name: str, disc_color: QColor, size: int = 30) -> Image:
    """Build a prominent circular badge: a solid colored disc with a white icon
    glyph on top, so it stays clearly visible over any thumbnail."""
    qimg = QImage(size, size, QImage.Format.Format_ARGB32)
    qimg.fill(Qt.GlobalColor.transparent)
    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(255, 255, 255))
    painter.drawEllipse(0, 0, size - 1, size - 1)  # white outline ring
    painter.setBrush(disc_color)
    painter.drawEllipse(2, 2, size - 5, size - 5)
    glyph = theme.icon(icon_name).pixmap(size - 12, size - 12).toImage()
    gp = QPainter(glyph)
    gp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    gp.fillRect(glyph.rect(), QColor(255, 255, 255))
    gp.end()
    off = (size - glyph.width()) // 2
    painter.drawImage(off, off, glyph)
    painter.end()
    return Image(qimg)


_SEARCH_SCOPE_ALL = "all"
_SEARCH_SCOPE_RAW = "raw"
_SEARCH_SCOPE_EVAL = "eval"


class HistoryWidget(QListWidget):
    _model: DocumentModel
    _connections: list[QMetaObject.Connection]
    _last_job_params: JobParams | None = None

    item_activated = pyqtSignal(QListWidgetItem)
    save_as_prompt_requested = pyqtSignal(str, str, object)  # prompt, negative, Image

    _thumb_size = 96
    _applied_icon = Image.load(theme.icon_path / "star.png")
    _favorite_icon = _tint_image(_applied_icon, QColor(255, 255, 255))
    _rating_star = Image.scale(_applied_icon, Extent(14, 14))  # native yellow
    _eagle_icon = _make_badge("upload", QColor(30, 130, 230), size=30)
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
        self._connections = []
        self._search_text = ""
        self._search_scope = _SEARCH_SCOPE_ALL
        self._favorites_only = False
        self._applied_only = False
        self._rating_filter = 0
        self._current_header: QListWidgetItem | None = None
        self._collapsed_batches: set[str] = set()
        self._active_header: QListWidgetItem | None = None  # last clicked batch header

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setResizeMode(QListView.Adjust)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFlow(QListView.LeftToRight)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(theme.screen_scale(self, QSize(self._thumb_size, self._thumb_size)))
        self.setFrameStyle(QListWidget.NoFrame)
        self.setStyleSheet(self._list_css)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setDragEnabled(False)
        self.itemClicked.connect(self.handle_preview_click)
        self.itemDoubleClicked.connect(self.item_activated)
        self.itemSelectionChanged.connect(self.select_item)

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
            jobs.favorite_changed.connect(self.update_image_thumbnail),
            jobs.rating_changed.connect(self.update_image_thumbnail),
            jobs.eagle_changed.connect(self.update_image_thumbnail),
            jobs.results_stored.connect(self._update_stored_sizes),
        ]
        self.rebuild()
        self.update_selection()

    def _same_batch(self, a: JobParams | None, b: JobParams) -> bool:
        if a is None:
            return False
        if a.batch_id and b.batch_id:
            # jobs from the same enqueue_jobs() call always share one batch, even
            # if wildcards make their evaluated prompt differ per item
            return a.batch_id == b.batch_id
        return JobParams.equal_ignore_seed(a, b)  # fallback for pre-batch_id history

    def add(self, job: Job):
        if not self.is_finished(job):
            return  # Only finished diffusion/animation jobs have images to show
        if any(self.item_info(ensure(self.item(i)))[0] == job.id for i in range(self.count())):
            return  # already shown, a second copy would outlive the job's discard

        # Thumbnails first: if a result can't be shown, the batch must not get a header
        # on its own - it could never be selected, discarded or collapsed.
        items: list[tuple[QListWidgetItem, int]] = []
        try:
            if job.kind is JobKind.diffusion:
                count = min(len(job.results), 1) if job.params.is_layered else len(job.results)
                for i in range(count):
                    items.append((QListWidgetItem(self._image_thumbnail(job, i), None), i))
            elif job.kind is JobKind.animation and len(job.results) > 0:
                thumbs = [self._image_thumbnail(job, i) for i in range(len(job.results))]
                items.append((AnimatedListItem(thumbs), 0))
        except Exception as e:
            log.exception(f"Failed to show history result {job.params.name}: {e}")
            return
        if len(items) == 0:
            return

        scrollbar = self.verticalScrollBar()
        scroll_to_bottom = scrollbar and scrollbar.value() >= scrollbar.maximum() - 4

        if not self._same_batch(self._last_job_params, job.params):
            self._last_job_params = job.params
            prompt = job.params.name if job.params.name != "" else "<no prompt>"
            strength = job.params.metadata.get("strength", 1.0)
            strength = f"{strength * 100:.0f}% - " if strength != 1.0 else ""

            header = QListWidgetItem(f"{job.timestamp.astimezone():%H:%M} - {strength}{prompt}")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setData(Qt.ItemDataRole.UserRole, job.id)
            header.setData(Qt.ItemDataRole.UserRole + 2, [job.id])  # ids of jobs in this batch
            header.setData(Qt.ItemDataRole.ToolTipRole, job.params.prompt)
            header.setSizeHint(QSize(9999, self.fontMetrics().lineSpacing() + 4))
            header.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            self.addItem(header)
            self._current_header = header
        elif self._current_header is not None:
            group = self._current_header.data(Qt.ItemDataRole.UserRole + 2) or []
            self._current_header.setData(Qt.ItemDataRole.UserRole + 2, [*group, job.id])

        for item, index in items:
            self._add_item(job, item, index)

        if scroll_to_bottom:
            self.scrollToBottom()
        self._apply_filter()

    def _add_item(self, job: Job, item: QListWidgetItem, index=0):
        item.setData(Qt.ItemDataRole.UserRole, job.id)
        item.setData(Qt.ItemDataRole.UserRole + 1, index)
        item.setData(Qt.ItemDataRole.ToolTipRole, self._item_tooltip(job, index))
        self.addItem(item)

    def _item_tooltip(self, job: Job, index: int):
        size = job.results[index].extent if index < len(job.results) else None
        return self._job_info(job.params, size, job.stored_sizes.get(index))

    def _update_stored_sizes(self, job: Job):
        # Stored sizes arrive after the item was added, once the result is encoded.
        for i in range(self.count()):
            item = self.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == job.id:
                if item.flags() != Qt.ItemFlag.NoItemFlags:  # not a batch header
                    _, index = self.item_info(item)
                    tooltip = self._item_tooltip(job, index or 0)
                    item.setData(Qt.ItemDataRole.ToolTipRole, tooltip)

    _job_info_translations: ClassVar[dict[str, str]] = {
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
        "duration": _("Generation Time"),
        "custom_inpaint": _("Custom Inpaint Settings"),
    }

    def _job_info(
        self, params: JobParams, size: Extent | None = None, stored_bytes: int | None = None
    ):
        title = params.name if params.name != "" else "<no prompt>"
        if len(title) > 70:
            title = title[:66] + "..."
        if params.strength != 1.0:
            title = f"{title} @ {params.strength * 100:.0f}%"
        style = Styles.list().find(params.style)
        strings: list[str | list[str]] = [
            title + "\n",
            _("Click to toggle preview, double-click to apply."),
            "",
        ]
        if mode := create_mode_label(params):
            strings.append(_("Mode") + f": {mode}")
        if size := size or params.bounds.extent:  # the result image, placed at the bounds
            strings.append(_("Size") + f": {size.width} × {size.height} px")
        if stored_bytes is not None:
            strings.append(_("File Size (in document)") + f": {_format_bytes(stored_bytes)}")
        if isinstance(dlss5 := params.metadata.get("dlss5"), dict):
            parts = [str(dlss5.get("area", "")).capitalize(), str(dlss5.get("style", ""))]
            for key, label in (("intensity", _("intensity")), ("tone", _("tone")),
                               ("structure", _("structure")), ("skin", _("skin")),
                               ("model_preset", _("model"))):  # fmt: skip
                if key in dlss5:
                    parts.append(f"{label} {dlss5[key]}")
            if tiles := dlss5.get("tiles"):
                parts.append(_("{count} tiles").format(count=tiles))
            strings.append(wrap_text("DLSS5: " + ", ".join(parts), 80, subsequent_indent=" "))
        for key, value in params.metadata.items():
            if key not in self._job_info_translations:
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
            if key == "duration" and isinstance(value, (int, float)):
                value = _format_duration(value)
            if key == "custom_inpaint" and isinstance(value, dict):
                parts = []
                if value.get("seamless"):
                    parts.append(_("Seamless"))
                if value.get("focus"):
                    parts.append(_("Focus"))
                if value.get("edit"):
                    parts.append(_("Edit"))
                fill = value.get("fill")
                if fill and fill != "none":
                    parts.append(f"{_('Fill')}: {fill.capitalize()}")
                if context := value.get("context"):
                    parts.append(f"{_('Context')}: {context}")
                if not parts:
                    continue
                value = " | ".join(parts)
            s = f"{self._job_info_translations.get(key, key)}: {value}"
            s = wrap_text(s, 80, subsequent_indent=" ")
            strings.append(s)
        strings.append(_("Seed") + f": {params.seed}")
        return "\n".join(flatten(strings))

    def remove(self, job: Job):
        self._remove_items(ensure(job.id))

    def remove_image(self, id: JobQueue.Item):
        self._remove_items(id.job, id.image)

    def _remove_items(self, job_id: str, image_index: int = -1):
        item_was_selected = False
        with theme.SignalBlocker(self):
            # Remove all the job's items before triggering potential selection changes.
            # The whole list is scanned: stopping at the first run of the job's items left
            # any other copy behind, as a thumbnail that could never be selected or discarded.
            for row in reversed(range(self.count())):
                item = ensure(self.item(row))
                if self._is_header(item) or item.data(Qt.ItemDataRole.UserRole) != job_id:
                    continue
                _, index = self.item_info(item)
                if image_index == -1 or index == image_index:
                    item_was_selected = item_was_selected or item.isSelected()
                    self.takeItem(row)
                elif index is not None and index > image_index:
                    item.setData(Qt.ItemDataRole.UserRole + 1, index - 1)
            item_was_selected = self._drop_orphans() or item_was_selected

        if item_was_selected:
            self._model.jobs.selection = []
        else:
            self.update_apply_button()  # selection may have moved

    def _is_header(self, item: QListWidgetItem):
        return item.flags() == Qt.ItemFlag.NoItemFlags

    def _remove_orphans(self):
        with theme.SignalBlocker(self):
            was_selected = self._drop_orphans()
        if was_selected:
            self._model.jobs.selection = [self._item_data(i) for i in self.selectedItems()]
        self.update_apply_button()

    def _drop_orphans(self) -> bool:
        """Removes thumbnails whose result is no longer in the job queue, and batch headers
        left without thumbnails. Returns True if a selected item was removed."""
        results = {job.id: len(job.results) for job in self._model.jobs if self.is_finished(job)}
        was_selected = False
        shown: set[str] = set()
        for row in reversed(range(self.count())):
            item = ensure(self.item(row))
            if self._is_header(item):
                continue
            job_id, index = self.item_info(item)
            if (index or 0) < results.get(job_id, 0):
                shown.add(job_id)
            else:
                was_selected = was_selected or item.isSelected()
                self.takeItem(row)
        for row in reversed(range(self.count())):
            item = ensure(self.item(row))
            if not self._is_header(item):
                continue
            group = [id for id in item.data(Qt.ItemDataRole.UserRole + 2) or [] if id in shown]
            if len(group) > 0:
                item.setData(Qt.ItemDataRole.UserRole + 2, group)
                continue
            self.takeItem(row)
            if item is self._active_header:
                self._active_header = None
            if item is self._current_header:
                # the next result must open a new header, not join the removed one
                self._current_header = None
                self._last_job_params = None
        return was_selected

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
            font = self._apply_button.fontMetrics()
            context_visible = rect.width() >= 0.6 * self.iconSize().width()
            apply_text_visible = font.width(_("Apply")) < 0.35 * rect.width()
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
        if self._favorites_only or self._applied_only or self._rating_filter > 0:
            self._apply_filter()

    def select_item(self):
        selection = [self._item_data(i) for i in self.selectedItems()]
        if not all(self._model.jobs.has_item(s) for s in selection):
            self._remove_orphans()  # stale thumbnail, can't be previewed or applied
            selection = [self._item_data(i) for i in self.selectedItems()]
        self._model.jobs.selection = selection

    def _toggle_selection(self):
        self._model.jobs.toggle_selection()

    def _activate_selection(self):
        items = self.selectedItems()
        if len(items) > 0:
            self.item_activated.emit(items[0])

    def is_finished(self, job: Job):
        return job.kind in [JobKind.diffusion, JobKind.animation] and job.state is JobState.finished

    def rebuild(self):
        self.clear()
        self._active_header = None
        self._current_header = None
        self._last_job_params = None
        for job in filter(self.is_finished, self._model.jobs):
            self.add(job)
        self.scrollToBottom()
        self._apply_filter()

    def set_search_filter(self, text: str):
        self._search_text = text.strip().lower()
        self._apply_filter()

    def set_search_scope(self, scope: str):
        self._search_scope = scope
        self._apply_filter()

    def set_thumb_size(self, value: int):
        self._thumb_size = value
        self.setIconSize(theme.screen_scale(self, QSize(value, value)))
        self.rebuild()

    def set_favorites_only(self, value: bool):
        self._favorites_only = value
        self._apply_filter()

    def set_applied_only(self, value: bool):
        self._applied_only = value
        self._apply_filter()

    def set_rating_filter(self, value: int):
        self._rating_filter = value
        self._apply_filter()

    def _item_matches_filter(self, item: QListWidgetItem) -> bool:
        job_id, index = self.item_info(item)
        job = self._model.jobs.find(job_id)
        if job is None:
            return False
        if self._favorites_only and not job.is_favorite(index or 0):
            return False
        if self._applied_only and not job.result_was_used(index or 0):
            return False
        if self._rating_filter > 0 and job.rating(index or 0) != self._rating_filter:
            return False
        if self._search_text:
            if self._search_scope == _SEARCH_SCOPE_RAW:
                keys = ("prompt", "negative_prompt")
            elif self._search_scope == _SEARCH_SCOPE_EVAL:
                keys = ("prompt_eval", "negative_prompt_eval")
            else:
                keys = ("prompt", "prompt_eval", "negative_prompt", "negative_prompt_eval")
            haystack = " ".join(str(job.params.metadata.get(k, "")) for k in keys).lower()
            if self._search_scope != _SEARCH_SCOPE_RAW:
                haystack += " " + job.params.name.lower()
            if self._search_text not in haystack:
                return False
        return True

    def _apply_filter(self):
        collapsed_job_ids: set[str] = set()
        for i in range(self.count()):
            it = ensure(self.item(i))
            if it.flags() == Qt.ItemFlag.NoItemFlags:
                if it.data(Qt.ItemDataRole.UserRole) in self._collapsed_batches:
                    collapsed_job_ids.update(it.data(Qt.ItemDataRole.UserRole + 2) or [])

        job_has_visible: dict[str, bool] = {}
        for i in range(self.count()):
            item = ensure(self.item(i))
            if item.flags() == Qt.ItemFlag.NoItemFlags:
                continue  # header item, handled below
            job_id, __ = self.item_info(item)
            # a batch matching the filter stays visible via its header even
            # while collapsed - only the individual thumbnails get hidden
            matches = self._item_matches_filter(item)
            if matches:
                job_has_visible[job_id] = True
            item.setHidden(not matches or job_id in collapsed_job_ids)
        for i in range(self.count()):
            item = ensure(self.item(i))
            if item.flags() == Qt.ItemFlag.NoItemFlags:
                # any job of the batch counts, the first one may have been discarded
                group = item.data(Qt.ItemDataRole.UserRole + 2) or []
                item.setHidden(not any(job_has_visible.get(id, False) for id in group))

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
        if item.flags() == Qt.ItemFlag.NoItemFlags:  # header - select the whole batch
            self._select_batch(item)

    def _select_batch(self, header_item: QListWidgetItem):
        group = set(header_item.data(Qt.ItemDataRole.UserRole + 2) or [])
        with theme.SignalBlocker(self):
            self.clearSelection()
            for i in range(self.count()):
                it = ensure(self.item(i))
                if it.flags() == Qt.ItemFlag.NoItemFlags or it.isHidden():
                    continue
                job_id, __ = self.item_info(it)
                if job_id in group:
                    it.setSelected(True)
        self.select_item()

    def _toggle_batch_collapse(self, header_item: QListWidgetItem):
        key = header_item.data(Qt.ItemDataRole.UserRole)
        if key in self._collapsed_batches:
            self._collapsed_batches.discard(key)
        else:
            self._collapsed_batches.add(key)
        self._apply_filter()

    def mousePressEvent(self, e: QMouseEvent | None):
        if e is not None:  # headers can't be selected, remember the one Del should act on
            clicked = self.itemAt(e.pos())
            is_header = clicked is not None and clicked.flags() == Qt.ItemFlag.NoItemFlags
            self._active_header = clicked if is_header else None
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
        self.update_apply_button()

    def event(self, e: QEvent | None):
        assert e is not None
        # Disambiguate shortcut events which Krita overrides. Everything reachable
        # from a key is also in the context menu, which shows the key next to it -
        # these are not QAction shortcuts, so the menu is the only place to find them.
        if e.type() == QEvent.Type.ShortcutOverride:
            assert isinstance(e, QKeyEvent)
            ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
            shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if e.matches(QKeySequence.StandardKey.Delete):
                self._discard_image(confirm=False)
                e.accept()
            elif ctrl and shift and e.key() == Qt.Key.Key_C:
                self._copy_prompt_evaluated_to_clipboard()
                e.accept()
            elif ctrl and e.key() == Qt.Key.Key_C:
                self._copy_image_to_clipboard()
                e.accept()
            elif ctrl and e.key() == Qt.Key.Key_S:
                self._save_image()
                e.accept()
            elif ctrl:
                return super().event(e)  # leave other Ctrl combinations to Krita
            elif e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._activate_selection()
                e.accept()
            elif e.key() == Qt.Key.Key_Space:
                self._toggle_selection()
                e.accept()
            elif e.key() == Qt.Key.Key_F:
                self._toggle_favorite()
                e.accept()
            elif e.key() == Qt.Key.Key_S:
                self._apply_style()
                e.accept()
            elif e.key() == Qt.Key.Key_P:
                self._apply_prompt_to_field()
                e.accept()
            elif e.key() == Qt.Key.Key_R:
                self._save_as_recipe()
                e.accept()
            elif e.key() == Qt.Key.Key_E:
                self._save_to_eagle()
                e.accept()
            elif Qt.Key.Key_0 <= e.key() <= Qt.Key.Key_5:
                self._set_selected_rating(e.key() - Qt.Key.Key_0)
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
        if job.result_was_used(index):  # add tiny star icon top-right to mark used results
            thumb.draw_image(self._applied_icon, offset=(thumb.extent.width - 28, 4))
        if job.is_favorite(index):  # add tiny star icon top-left to mark favorites
            thumb.draw_image(self._favorite_icon, offset=(4, 4))
        rating = job.rating(index)
        if rating > 0:  # row of yellow stars bottom-left to show the rating
            star_w = self._rating_star.extent.width
            y = thumb.extent.height - self._rating_star.extent.height - 4
            for i in range(rating):
                thumb.draw_image(self._rating_star, offset=(4 + i * (star_w - 3), y))
        if job.sent_to_eagle(index):  # blue upload badge bottom-right for Eagle exports
            badge = self._eagle_icon
            thumb.draw_image(
                badge,
                offset=(
                    thumb.extent.width - badge.extent.width - 4,
                    thumb.extent.height - badge.extent.height - 4,
                ),
            )
        return thumb.to_icon()

    def _show_batch_context_menu(self, header_item: QListWidgetItem, pos: QPoint):
        key = header_item.data(Qt.ItemDataRole.UserRole)
        menu = theme.style_menu(QMenu(self))
        label = _("Expand Batch") if key in self._collapsed_batches else _("Collapse Batch")
        menu.addAction(label, lambda: self._toggle_batch_collapse(header_item))
        menu.addAction(_("Select Batch"), lambda: self._select_batch(header_item))
        menu.addSeparator()
        menu.addAction(_("Discard Batch") + "\tDel", lambda: self._discard_batch(header_item))
        menu.exec(self.mapToGlobal(pos))

    def _show_context_menu(self, pos: QPoint):
        item = self.itemAt(pos)
        if item is not None and item.flags() == Qt.ItemFlag.NoItemFlags:
            self._show_batch_context_menu(item, pos)
        elif item is not None:
            job = self._model.jobs.find(self._item_data(item).job)
            menu = theme.style_menu(QMenu(self))
            # Shortcut hints go into every entry that has one (after a tab, which Qt
            # right-aligns): the keys are handled in event() rather than by QAction,
            # so without the hint there is nothing to discover them from.
            menu.addAction(_("Apply to Canvas") + "\tEnter", self._activate_selection)
            menu.addAction(_("Apply Prompt to Field") + "\tP", self._apply_prompt_to_field)
            menu.addAction(
                _("Apply Prompt (Evaluated) to Field"), self._apply_prompt_evaluated_to_field
            )
            # "Apply" here means the current settings, not the clipboard - these three
            # set the style, strength and seed the image was generated with
            style_action = ensure(menu.addAction(_("Apply Style") + "\tS", self._apply_style))
            style_action.setToolTip(_("Select the style this image was generated with"))
            if job is None or Styles.list().find(job.params.style) is None:
                style_action.setEnabled(False)
                style_action.setToolTip(_("The style of this image no longer exists"))
            style = Styles.list().find(job.params.style) if job is not None else None
            is_fav_style = style is not None and style.filename in settings.favorite_styles
            fav_style_label = (
                _("Remove Style from Favorites") if is_fav_style else _("Mark Style as Favorite")
            )
            fav_style_action = ensure(
                menu.addAction(fav_style_label, lambda: self._toggle_style_favorite())
            )
            fav_style_action.setEnabled(style is not None)
            menu.addAction(_("Apply Strength"), self._apply_strength)
            menu.addAction(_("Apply Seed"), self._apply_seed)
            menu.addSeparator()
            menu.addAction(_("Copy Image to Clipboard") + "\tCtrl+C", self._copy_image_to_clipboard)
            menu.addAction(_("Copy Prompt to Clipboard"), self._copy_prompt_to_clipboard)
            menu.addAction(
                _("Copy Prompt (Evaluated) to Clipboard") + "	Ctrl+Shift+C",
                self._copy_prompt_evaluated_to_clipboard,
            )
            menu.addAction(_("Info to Clipboard"), self._info_to_clipboard)
            menu.addSeparator()
            if self._model.dlss5_available:  # only with the optional DLSS5 node pack
                dlss5_menu = ensure(menu.addMenu(_("DLSS5 Enhance")))
                for style in ("Cinematic", "Default", "Natural"):
                    dlss5_menu.addAction(_(style), self._make_dlss5_enhancer(style))
                menu.addSeparator()
            __, index = self.item_info(item)
            is_fav = job is not None and job.is_favorite(index or 0)
            fav_label = _("Remove from Favorites") if is_fav else _("Mark as Favorite")
            menu.addAction(fav_label + "\tF", self._toggle_favorite)
            current_rating = job.rating(index or 0) if job is not None else 0
            rating_menu = ensure(menu.addMenu(_("Set Rating")))
            for stars in range(1, 6):
                label = "★" * stars + "☆" * (5 - stars)
                action = ensure(
                    rating_menu.addAction(label + f"\t{stars}", self._make_rating_setter(stars))
                )
                action.setCheckable(True)
                action.setChecked(stars == current_rating)
            rating_menu.addSeparator()
            clear_rating = ensure(
                rating_menu.addAction(_("Clear Rating") + "\t0", self._make_rating_setter(0))
            )
            clear_rating.setEnabled(current_rating > 0)
            menu.addSeparator()
            menu.addAction(_("Save to Eagle") + "\tE", self._save_to_eagle)
            menu.addAction(_("Save as Recipe") + "\tR", self._save_as_recipe)
            menu.addAction(_("Save as Prompt"), self._save_as_prompt)
            save_action = ensure(menu.addAction(_("Save Image") + "\tCtrl+S", self._save_image))
            if self._model.document.filename == "":
                tt = _(
                    "Save as separate image to the same folder as the document.\nMust save the document first!"
                )
                save_action.setEnabled(False)
                save_action.setToolTip(tt)
                menu.setToolTipsVisible(True)
            # via lambda: QAction.triggered passes its checked flag as the first
            # argument, which would silently turn the confirmation prompt off
            menu.addAction(_("Discard Image") + "\tDel", lambda: self._discard_image())
            menu.addSeparator()
            menu.addAction(_("Toggle Selection") + "\tSpace", self._toggle_selection)
            menu.addAction(_("Clear History"), self._clear_all)
            menu.exec(self.mapToGlobal(pos))

    def _show_context_menu_dropdown(self):
        pos = self._context_button.pos()
        pos.setY(pos.y() + self._context_button.height())
        self._show_context_menu(pos)

    def _apply_prompt_to_field(self, evaluated=False):
        if job := self.selected_job:
            positive = "prompt_eval" if evaluated else "prompt"
            prompt = job.params.metadata.get(positive, job.params.prompt)
            active = self._model.active_regions.active_or_root
            active.positive = prompt
            if isinstance(active, RootRegion):
                negative = "negative_prompt_eval" if evaluated else "negative_prompt"
                active.negative = job.params.metadata.get(
                    negative, job.params.metadata.get("negative_prompt", "")
                )
            if self._model.workspace is Workspace.custom and self._model.document.is_active:
                self._model.custom.try_set_params(job.params.metadata)

    def _apply_prompt_evaluated_to_field(self):
        self._apply_prompt_to_field(evaluated=True)

    def _copy_prompt_to_clipboard(self, evaluated=False):
        if job := self.selected_job:
            positive = "prompt_eval" if evaluated else "prompt"
            prompt = job.params.metadata.get(positive, job.params.prompt)
            if clipboard := QGuiApplication.clipboard():
                clipboard.setText(prompt)

    def _copy_prompt_evaluated_to_clipboard(self):
        self._copy_prompt_to_clipboard(evaluated=True)

    def _apply_strength(self):
        if job := self.selected_job:
            self._model.strength = job.params.strength

    def _apply_style(self):
        if (job := self.selected_job) and (style := Styles.list().find(job.params.style)):
            self._model.style = style

    def _toggle_style_favorite(self):
        if (job := self.selected_job) and (style := Styles.list().find(job.params.style)):
            favorites = list(settings.favorite_styles)
            if style.filename in favorites:
                favorites.remove(style.filename)
            else:
                favorites.append(style.filename)
            settings.favorite_styles = favorites

    def _apply_seed(self):
        if job := self.selected_job:
            self._model.fixed_seed = True
            self._model.seed = job.params.seed

    def _copy_image_to_clipboard(self):
        """The generated image itself, at full resolution - for pasting into another
        application. The clipboard holds one image, so a multi-selection copies the
        first of them."""
        items = self.selectedItems()
        if not items:
            return
        job_id, image_index = self.item_info(items[0])
        job = self._model.jobs.find(job_id)
        index = image_index or 0
        if job is None or len(job.results) <= index:
            return
        if clipboard := QGuiApplication.clipboard():
            clipboard.setPixmap(job.results[index].to_pixmap())

    def _info_to_clipboard(self):
        if (job := self.selected_job) and (clipboard := QGuiApplication.clipboard()):
            style = Styles.list().find(job.params.style)
            data = job.params.metadata.copy()
            if style:
                data["style"] = f"{style.name} ({style.filename})"
            text = json.dumps(data, indent=2)
            clipboard.setText(text)

    def _save_image(self):
        if not self._model.document.filename:
            QMessageBox.information(
                self,
                _("Save Image"),
                _("Please save the Krita document first, then save the generated image."),
            )
            if action := Krita.instance().action("file_save_as"):
                action.trigger()
            return
        items = self.selectedItems()
        for item in items:
            job_id, image_index = self.item_info(item)
            self._model.save_result(job_id, image_index)

    def _save_to_eagle(self):
        items = self.selectedItems()
        for item in items:
            job_id, image_index = self.item_info(item)
            self._model.send_result_to_eagle(job_id, image_index)

    def _make_dlss5_enhancer(self, style: str):
        def enhance():
            for item in self.selectedItems():
                job_id, image_index = self.item_info(item)
                self._model.enhance_result_dlss5(job_id, image_index or 0, style)

        return enhance

    def _save_as_recipe(self):
        items = self.selectedItems()
        for item in items:
            job_id, image_index = self.item_info(item)
            self._model.send_result_to_recipe(job_id, image_index)

    def _save_as_prompt(self):
        # unlike Recipe (checkpoint + LoRA stack fetched via the Lora Manager server),
        # this is a plain local prompt snippet with a thumbnail - see prompt_library.py.
        # job.params.name is the full (evaluated) prompt text, unusable as a title, so
        # this hands off to the Prompt Picker itself rather than a plain input dialog -
        # text/negative/preview arrive pre-filled, only the name needs typing.
        items = self.selectedItems()
        for item in items:
            job_id, image_index = self.item_info(item)
            job = self._model.jobs.find(job_id)
            index = image_index or 0
            if job is None or index >= len(job.results):
                continue
            prompt = job.params.prompt
            negative = job.params.metadata.get("negative_prompt", "")
            self.save_as_prompt_requested.emit(prompt, negative, job.results[index])

    def _toggle_favorite(self):
        items = self.selectedItems()
        for item in items:
            job_id, image_index = self.item_info(item)
            self._model.jobs.toggle_favorite(job_id, image_index or 0)

    def _set_selected_rating(self, rating: int):
        items = self.selectedItems()
        for item in items:
            job_id, image_index = self.item_info(item)
            self._model.jobs.set_rating(job_id, image_index or 0, rating)

    def _make_rating_setter(self, rating: int):
        return lambda: self._set_selected_rating(rating)

    def _discard_image(self, confirm=True):
        confirm = confirm and settings.confirm_discard_image
        reply = QMessageBox.Yes
        if confirm:
            reply = QMessageBox.warning(
                self,
                _("Discard Image"),
                _("Are you sure you want to discard the selected images?"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.StandardButton.Yes,
            )
        if reply == QMessageBox.Yes:
            items = self.selectedItems()
            if len(items) == 0 and self._active_header is not None:
                self._discard_batch(self._active_header, confirm=False)
                return
            next_item = self.row(items[0]) if len(items) > 0 else -1
            for item in items:
                job_id, image_index = self.item_info(item)
                if self._model.jobs.has_item(JobQueue.Item(job_id, image_index or 0)):
                    self._model.jobs.discard(job_id, image_index or 0)
            # a thumbnail whose result is already gone has nothing to discard in the queue,
            # it only has to leave the list
            self._remove_orphans()
            if next_item >= 0:
                self.setCurrentRow(next_item, QItemSelectionModel.SelectionFlag.Current)

    def _discard_batch(self, header_item: QListWidgetItem, confirm=True):
        if confirm and settings.confirm_discard_image:
            reply = QMessageBox.warning(
                self,
                _("Discard Batch"),
                _("Are you sure you want to discard all images of this batch?"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.Yes:
                return
        self._active_header = None
        for job_id in header_item.data(Qt.ItemDataRole.UserRole + 2) or []:
            self._model.jobs.discard_job(job_id)
        # Discarding the jobs usually takes the header along. One whose jobs are gone
        # already, or never had an image to show, has to be removed here.
        row = self.row(header_item)
        if row >= 0:
            self.takeItem(row)
        if header_item is self._current_header:
            self._current_header = None
            self._last_job_params = None

    def _clear_all(self):
        reply = QMessageBox.warning(
            self,
            _("Clear History"),
            _("Are you sure you want to discard all generated images?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._model.jobs.clear()
            self.clear()
            # the headers are gone, the next result must not try to join one
            self._current_header = None
            self._last_job_params = None
            self._active_header = None
            self._model.hide_preview(delete_layer=True)


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


class CustomInpaintWidget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._model = root.active_model
        self._model_bindings = []

        self.use_inpaint_button = QCheckBox(self)
        self.use_inpaint_button.setText(_("Seamless"))
        self.use_inpaint_button.setToolTip(_("Generate content which blends into the surroundings"))

        self.use_prompt_focus_button = QCheckBox(self)
        self.use_prompt_focus_button.setText(_("Focus"))
        self.use_prompt_focus_button.setToolTip(
            _(
                "Use the text prompt to describe the selected region rather than the context area / Use only one regional prompt"
            )
        )

        self.edit_mode_switch = QCheckBox(self)
        self.edit_mode_switch.setText(_("Edit"))
        self.edit_mode_switch.setToolTip(_("Edit canvas with text instructions"))

        self.fill_mode_combo = QComboBox(self)
        fill_icon = theme.icon("fill")
        self.fill_mode_combo.addItem(theme.icon("fill-empty"), _("None"), FillMode.none)
        self.fill_mode_combo.addItem(fill_icon, _("Neutral"), FillMode.neutral)
        self.fill_mode_combo.addItem(fill_icon, _("Blur"), FillMode.blur)
        self.fill_mode_combo.addItem(fill_icon, _("Border"), FillMode.border)
        self.fill_mode_combo.addItem(fill_icon, _("Inpaint"), FillMode.inpaint)
        self.fill_mode_combo.setStyleSheet(theme.flat_combo_stylesheet)
        self.fill_mode_combo.setToolTip(_("Pre-fill the selected region before diffusion"))

        def ctx_icon(name):
            return theme.icon(f"context-{name}")

        self.context_combo = QComboBox(self)
        self.context_combo.addItem(
            ctx_icon("automatic"), _("Automatic Context"), InpaintContext.automatic
        )
        self.context_combo.addItem(
            ctx_icon("mask"), _("Selection Bounds"), InpaintContext.mask_bounds
        )
        self.context_combo.addItem(
            ctx_icon("image"), _("Entire Image"), InpaintContext.entire_image
        )
        self.context_combo.setStyleSheet(theme.flat_combo_stylesheet)
        self.context_combo.setToolTip(
            _("Part of the image around the selection which is used as context.")
        )
        self.context_combo.setMinimumContentsLength(20)
        self.context_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLength
        )
        self.context_combo.currentIndexChanged.connect(self.set_context)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.use_inpaint_button)
        layout.addWidget(self.use_prompt_focus_button)
        layout.addWidget(self.edit_mode_switch)
        layout.addWidget(self.fill_mode_combo, 1)
        layout.addWidget(self.context_combo, 1)
        self.setLayout(layout)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            self._model_bindings = [
                bind_combo(model.inpaint, "fill", self.fill_mode_combo),
                bind_toggle(model.inpaint, "use_inpaint", self.use_inpaint_button),
                bind_toggle(model.inpaint, "use_prompt_focus", self.use_prompt_focus_button),
                bind_toggle(model, "edit_mode", self.edit_mode_switch),
                model.style_changed.connect(self.update_widgets_enabled),
                model.strength_changed.connect(self.update_widgets_enabled),
                model.layers.changed.connect(self.update_context_layers),
                model.edit_mode_changed.connect(self.update_widgets_enabled),
            ]
            self.update_widgets_enabled()
            self.update_context_layers()
            self.update_context()

    def update_widgets_enabled(self):
        arch = self._model.arch
        self.fill_mode_combo.setEnabled(self.model.strength == 1.0 and not self.model.is_editing)
        self.use_inpaint_button.setEnabled(arch.is_sdxl_like or arch.has_controlnet_inpaint)
        self.use_prompt_focus_button.setVisible(arch is Arch.sd15 or arch.is_sdxl_like)
        self.edit_mode_switch.setEnabled(self.model.can_toggle_edit)

    def update_context_layers(self):
        current = self.context_combo.currentData()
        with theme.SignalBlocker(self.context_combo):
            while self.context_combo.count() > 3:
                self.context_combo.removeItem(self.context_combo.count() - 1)
            icon = theme.icon("context-layer")
            for layer in self._model.layers.masks:
                self.context_combo.addItem(icon, f"{layer.name}", layer.id)
        current_index = self.context_combo.findData(current)
        if current_index >= 0:
            self.context_combo.setCurrentIndex(current_index)

    def update_context(self):
        if self._model.inpaint.context == InpaintContext.layer_bounds:
            i = self.context_combo.findData(self._model.inpaint.context_layer_id)
            self.context_combo.setCurrentIndex(i)
        else:
            i = self.context_combo.findData(self._model.inpaint.context)
            self.context_combo.setCurrentIndex(i)

    def set_context(self):
        data = self.context_combo.currentData()
        if isinstance(data, QUuid):
            self._model.inpaint.context = InpaintContext.layer_bounds
            self._model.inpaint.context_layer_id = data
        elif isinstance(data, InpaintContext):
            self._model.inpaint.context = data


class ProgressBar(QProgressBar):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._model = root.active_model
        self._model_bindings: list[QMetaObject.Connection] = []
        self._palette = self.palette()
        self.setMinimum(0)
        self.setMaximum(1000)
        self.setTextVisible(False)
        self.setFixedHeight(6)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            self._model_bindings = [
                self._model.progress_changed.connect(self._update_progress),
                self._model.progress_kind_changed.connect(self._update_progress_kind),
            ]

    def _update_progress_kind(self):
        palette = self._palette
        if self._model.progress_kind is ProgressKind.upload:
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.Highlight, QColor(theme.progress_alt))
        self.setPalette(palette)

    def _update_progress(self):
        if self._model.progress >= 0:
            self.setValue(int(self._model.progress * 1000))
        else:
            if self.value() >= 100:
                self.reset()
            self.setValue(min(99, self.value() + 2))


class ProgressStatus(QLabel):
    """One line under the progress bar: what the server is doing and for how long.
    Hidden while idle so it takes no space."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._model = root.active_model
        self._model_bindings: list[QMetaObject.Connection] = []
        self._started = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_text)
        self.setStyleSheet(f"font-style: italic; color: {theme.grey};")
        self.setVisible(False)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            self._model_bindings = [
                model.progress_status_changed.connect(self._update_status),
            ]
            self._update_status(model.progress_status)

    def _update_status(self, status: str):
        if status and not self._timer.isActive():
            self._started = time.monotonic()
            self._timer.start()
        elif not status:
            self._timer.stop()
        self.setVisible(bool(status))
        self._update_text()

    def _update_text(self):
        status = self._model.progress_status
        elapsed = int(time.monotonic() - self._started)
        theme.set_text_clipped(self, f"{status} · {elapsed // 60}:{elapsed % 60:02d}")


class GenerationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._model: DocumentModel = root.active_model
        self._model_bindings: list[QMetaObject.Connection | Binding] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 2, 0)
        self.setLayout(layout)

        self.workspace_select = WorkspaceSelectWidget(self)
        self.style_select = StyleSelectWidget(self)

        style_layout = QHBoxLayout()
        style_layout.addWidget(self.workspace_select)
        style_layout.addWidget(self.style_select)
        layout.addLayout(style_layout)

        self.region_prompt = RegionPromptWidget(self)
        layout.addWidget(self.region_prompt)

        self.strength_slider = StrengthWidget(parent=self)
        self.layer_count_widget = LayerCountWidget(self)
        self.layer_count_widget.setVisible(False)
        self.add_region_button = create_wide_tool_button("region-add", _("Add Region"), self)
        self.add_control_button = create_wide_tool_button(
            "control-add", _("Add Control Layer"), self
        )
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_slider)
        self.sweep_button = QToolButton(self)
        self.sweep_button.setText("⇅")
        self.sweep_button.setToolTip(
            _("Strength Sweep: one image per strength value, e.g. 40% to 75% in steps of 5")
        )
        self.sweep_button.clicked.connect(self._open_strength_sweep)
        strength_layout.addWidget(self.sweep_button)
        strength_layout.addWidget(self.layer_count_widget)
        strength_layout.addWidget(self.add_control_button)
        strength_layout.addWidget(self.add_region_button)
        layout.addLayout(strength_layout)

        self.custom_inpaint = CustomInpaintWidget(self)
        layout.addWidget(self.custom_inpaint)

        self.generate_button = GenerateButton(JobKind.diffusion, self)

        self.inpaint_mode_button = QToolButton(self)
        self.inpaint_mode_button.setArrowType(Qt.ArrowType.DownArrow)
        self.inpaint_mode_button.setFixedHeight(self.generate_button.minimumSizeHint().height() - 3)
        self.inpaint_mode_button.clicked.connect(self.show_inpaint_menu)
        self.generate_menu = self._create_generate_menu()
        self.inpaint_menu = self._create_inpaint_menu()
        self.refine_menu = self._create_refine_menu()
        self.refine_selection_menu = self._create_refine_selection_menu()
        self.generate_region_menu = self._create_generate_region_menu()
        self.refine_region_menu = self._create_refine_region_menu()
        self.edit_menu = self._create_edit_menu()
        self.dlss5_canvas_menu = self._create_dlss5_menu(_("DLSS5 Enhance Canvas"), False)
        self.dlss5_selection_menu = self._create_dlss5_menu(_("DLSS5 Enhance Selection"), True)
        self.ai_select_menu = self._create_ai_select_menu()

        self.region_mask_button = QToolButton(self)
        self.region_mask_button.setIcon(theme.icon("region-alpha"))
        self.region_mask_button.setCheckable(True)
        self.region_mask_button.setFixedHeight(self.generate_button.height() - 2)
        self.region_mask_button.setToolTip(
            _("Generate the active layer region only (use layer transparency as mask)")
        )

        generate_layout = QHBoxLayout()
        generate_layout.setSpacing(0)
        generate_layout.addWidget(self.generate_button)
        generate_layout.addWidget(self.inpaint_mode_button)
        generate_layout.addWidget(self.region_mask_button)

        self.queue_button = QueueButton(parent=self)
        self.queue_button.setFixedHeight(self.generate_button.height() - 2)

        self.loop_button = QToolButton(self)
        self._loop_icon = theme.icon("reset")
        self._loop_icon_active = _tint_icon("reset", QColor(theme.strong_highlight))
        self.loop_button.setIcon(self._loop_icon)
        self.loop_button.setCheckable(True)
        self.loop_button.setFixedHeight(self.generate_button.height() - 2)
        self.loop_button.setToolTip(_("Generate continuously until stopped - click again to stop"))
        # make the active state clearly visible (barely showed in the dark theme)
        self.loop_button.setStyleSheet(
            f"QToolButton:checked {{ background-color: {theme.active};"
            f" border: 1px solid {theme.strong_highlight}; border-radius: 3px; }}"
        )
        self.loop_button.toggled.connect(
            lambda on: self.loop_button.setIcon(self._loop_icon_active if on else self._loop_icon)
        )

        self.pause_button = QToolButton(self)
        self._pause_icon = theme.icon("pause")
        self._pause_icon_active = _tint_icon("pause", QColor(theme.strong_highlight))
        self.pause_button.setIcon(self._pause_icon)
        self.pause_button.setCheckable(True)
        self.pause_button.setFixedHeight(self.generate_button.height() - 2)
        self.pause_button.setToolTip(
            _(
                "Pause the queue: the running generation finishes, queued jobs wait until resumed"
                " - click again to resume"
            )
        )
        self.pause_button.setStyleSheet(
            f"QToolButton:checked {{ background-color: {theme.active};"
            f" border: 1px solid {theme.strong_highlight}; border-radius: 3px; }}"
        )
        self.pause_button.toggled.connect(self._toggle_pause)
        root.connection.paused_changed.connect(self._update_pause_button)

        self.preview_button = QToolButton(self)
        self._preview_icon = theme.icon("eye")
        self._preview_icon_active = _tint_icon("eye", QColor(theme.strong_highlight))
        self.preview_button.setCheckable(True)
        self.preview_button.setFixedHeight(self.generate_button.height() - 2)
        self.preview_button.setToolTip(
            _("Show intermediate results on the canvas while generating - click to toggle")
        )
        self.preview_button.setStyleSheet(
            f"QToolButton:checked {{ background-color: {theme.active};"
            f" border: 1px solid {theme.strong_highlight}; border-radius: 3px; }}"
        )
        self.preview_button.setChecked(settings.show_generation_preview)
        self._update_preview_button(settings.show_generation_preview)
        self.preview_button.toggled.connect(self._toggle_preview)

        self.combinations_button = QToolButton(self)
        self.combinations_button.setIcon(theme.icon("combinations"))
        self.combinations_button.setFixedHeight(self.generate_button.height() - 2)
        self.combinations_button.setToolTip(
            _(
                "Set batch count to the number of sequential wildcard combinations in the prompt and generate"
            )
        )
        self.combinations_button.clicked.connect(self._generate_combinations)

        self.skip_button = QToolButton(self)
        self.skip_button.setIcon(theme.icon("cancel"))
        self.skip_button.setFixedHeight(self.generate_button.height() - 2)
        self.skip_button.setToolTip(
            _("Cancel the current generation - queued jobs continue with the next one")
        )
        self.skip_button.setEnabled(False)
        self.skip_button.clicked.connect(lambda: self.model.cancel(active=True))

        actions_layout = QHBoxLayout()
        actions_layout.addLayout(generate_layout)
        # grouped: what gets generated | canvas view | queue control
        actions_layout.addWidget(self.combinations_button)
        actions_layout.addWidget(self.loop_button)
        actions_layout.addSpacing(6)
        actions_layout.addWidget(self.preview_button)
        actions_layout.addSpacing(6)
        actions_layout.addWidget(self.pause_button)
        actions_layout.addWidget(self.skip_button)
        actions_layout.addWidget(self.queue_button)
        layout.addLayout(actions_layout)

        self.progress_bar = ProgressBar(self)
        layout.addWidget(self.progress_bar)
        self.progress_status = ProgressStatus(self)
        layout.addWidget(self.progress_status)

        self.error_box = ErrorBox(self)
        layout.addWidget(self.error_box)

        self.history_search = QLineEdit(self)
        self.history_search.setPlaceholderText(_("Search prompts…"))
        self.history_search.setClearButtonEnabled(True)

        self.history_search_scope = QComboBox(self)
        self.history_search_scope.addItem(_("All Prompts"), _SEARCH_SCOPE_ALL)
        self.history_search_scope.addItem(_("Raw Only"), _SEARCH_SCOPE_RAW)
        self.history_search_scope.addItem(_("Evaluated Only"), _SEARCH_SCOPE_EVAL)
        self.history_search_scope.setToolTip(
            _("Search the raw prompt text, the evaluated text (wildcards resolved), or both")
        )

        # make the active state clearly visible (barely showed in the dark theme),
        # same treatment as the loop button
        _filter_checked_style = (
            f"QToolButton:checked {{ background-color: {theme.active};"
            f" border: 1px solid {theme.strong_highlight}; border-radius: 3px; }}"
        )

        self.history_favorites_only = QToolButton(self)
        self.history_favorites_only.setIcon(HistoryWidget._favorite_icon.to_icon())
        self.history_favorites_only.setCheckable(True)
        self.history_favorites_only.setToolTip(_("Show only favorite images (F to toggle)"))
        self.history_favorites_only.setStyleSheet(_filter_checked_style)

        self.history_applied_only = QToolButton(self)
        self.history_applied_only.setIcon(theme.icon("apply"))
        self.history_applied_only.setCheckable(True)
        self.history_applied_only.setToolTip(_("Show only images that were applied to the canvas"))
        self.history_applied_only.setStyleSheet(_filter_checked_style)

        self.history_min_rating = QComboBox(self)
        self.history_min_rating.addItem(_("Any Rating"), 0)
        for stars in range(1, 6):
            self.history_min_rating.addItem("★" * stars, stars)
        self.history_min_rating.setToolTip(
            _("Show only images with exactly this rating (1-5 to rate)")
        )

        history_filter_layout = QHBoxLayout()
        history_filter_layout.addWidget(self.history_search, 1)
        history_filter_layout.addWidget(self.history_search_scope)
        history_filter_layout.addWidget(self.history_favorites_only)
        history_filter_layout.addWidget(self.history_applied_only)
        history_filter_layout.addWidget(self.history_min_rating)
        layout.addLayout(history_filter_layout)

        history_size_label = QLabel(_("Size:"), self)
        self.history_size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.history_size_slider.setMinimum(48)
        self.history_size_slider.setMaximum(256)
        self.history_size_slider.setValue(96)
        self.history_size_slider.setToolTip(_("Preview image size"))
        history_size_layout = QHBoxLayout()
        history_size_layout.addWidget(history_size_label)
        history_size_layout.addWidget(self.history_size_slider, 1)
        layout.addLayout(history_size_layout)

        self.history = HistoryWidget(self)
        self.history.item_activated.connect(self.apply_result)
        self.history.save_as_prompt_requested.connect(self._save_as_prompt_entry)
        self.history_search.textChanged.connect(self.history.set_search_filter)
        self.history_search_scope.currentIndexChanged.connect(
            lambda: self.history.set_search_scope(self.history_search_scope.currentData())
        )
        self.history_favorites_only.toggled.connect(self.history.set_favorites_only)
        self.history_applied_only.toggled.connect(self.history.set_applied_only)
        self.history_min_rating.currentIndexChanged.connect(
            lambda: self.history.set_rating_filter(self.history_min_rating.currentData())
        )
        self.history_size_slider.valueChanged.connect(self.history.set_thumb_size)
        layout.addWidget(self.history)

        self.update_generate_options()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            self._model_bindings = [
                bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
                bind(model, "style", self.style_select, "value"),
                bind(model, "strength", self.strength_slider, "value"),
                bind(model, "layer_count", self.layer_count_widget, "value"),
                bind(model, "error", self.error_box, "error", Bind.one_way),
                bind_toggle(model, "region_only", self.region_mask_button),
                model.inpaint.mode_changed.connect(self.update_generate_options),
                model.strength_changed.connect(self.update_generate_options),
                model.batch_count_changed.connect(self.update_generate_options),
                model.document.selection_bounds_changed.connect(self.update_generate_options),
                # DLSS5 availability (and with it the arrow menu) depends on server nodes
                root.connection.models_changed.connect(self.update_generate_options),
                model.document.layers.active_changed.connect(self.update_generate_options),
                model.regions.active_changed.connect(self.update_generate_options),
                model.region_only_changed.connect(self.update_generate_options),
                model.style_changed.connect(self.update_generate_options),
                model.edit_mode_changed.connect(self.update_generate_options),
                self.add_control_button.clicked.connect(self.add_control),
                self.add_region_button.clicked.connect(self.add_region),
                self.region_prompt.activated.connect(model.generate),
                self.generate_button.clicked.connect(model.generate),
                self.generate_button.ctrl_clicked.connect(model.generate_replace),
                bind_toggle(model, "loop_generate", self.loop_button),
                model.jobs.count_changed.connect(self._update_skip_button),
            ]
            self.region_prompt.regions = model.active_regions
            self.custom_inpaint.model = model
            self.generate_button.model = model
            self.queue_button.model = model
            self.progress_bar.model = model
            self.progress_status.model = model
            self.strength_slider.model = model
            self.history.model_ = model
            self._update_skip_button()
            self.update_generate_options()

    def _open_strength_sweep(self):
        from .strength_sweep import StrengthSweepDialog

        StrengthSweepDialog(self.model, self).exec_()

    def _toggle_preview(self, on: bool):
        settings.show_generation_preview = on
        settings.save()
        self._update_preview_button(on)

    def _update_preview_button(self, on: bool):
        self.preview_button.setIcon(self._preview_icon_active if on else self._preview_icon)

    def _update_skip_button(self):
        self.skip_button.setEnabled(self.model.jobs.any_executing())

    def _toggle_pause(self, on: bool):
        root.connection.paused = on
        if on and not root.connection.paused:  # client doesn't support it
            self.pause_button.setChecked(False)

    def _update_pause_button(self, paused: bool):
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(paused)
        self.pause_button.blockSignals(False)
        self.pause_button.setIcon(self._pause_icon_active if paused else self._pause_icon)

    def _generate_combinations(self):
        if count := wildcard_combination_count(self.model):
            self.model.batch_count = count
            self.model.generate()

    def apply_result(self, item: QListWidgetItem):
        job_id, index = self.history.item_info(item)
        self.model.apply_generated_result(job_id, index)

    def _save_as_prompt_entry(self, prompt: str, negative: str, image):
        self.region_prompt.open_prompt_picker_with(prompt, negative, image)

    _inpaint_text: ClassVar[dict[InpaintMode, str]] = {
        InpaintMode.automatic: _("Default (Auto-detect)"),
        InpaintMode.fill: _("Fill"),
        InpaintMode.expand: _("Expand"),
        InpaintMode.add_object: _("Add Content"),
        InpaintMode.remove_object: _("Remove Content"),
        InpaintMode.replace_background: _("Replace Background"),
        InpaintMode.custom: _("Generate (Custom)"),
    }

    def _mk_action(self, mode: InpaintMode, text: str, icon: str, is_edit: bool | None = False):
        action = QAction(text, self)
        action.setIcon(theme.icon(icon))
        action.setIconVisibleInMenu(True)
        action.triggered.connect(lambda: self.change_inpaint_mode(mode, is_edit))
        return action

    def _create_generate_menu(self):
        menu = theme.style_menu(QMenu(self))
        menu.addAction(
            self._mk_action(InpaintMode.automatic, _("Generate"), "workspace-generation")
        )
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Edit"), "edit", is_edit=True))
        return menu

    def _create_inpaint_menu(self):
        menu = theme.style_menu(QMenu(self))

        def add(mode: InpaintMode, text: str, icon: str, is_edit: bool | None = False):
            text = text or self._inpaint_text[mode]
            menu.addAction(self._mk_action(mode, text, icon, is_edit))

        add(InpaintMode.automatic, "", "inpaint-automatic")
        add(InpaintMode.fill, "", "inpaint-fill")
        add(InpaintMode.expand, "", "inpaint-expand")
        add(InpaintMode.add_object, "", "inpaint-add_object")
        add(InpaintMode.remove_object, "", "inpaint-remove_object")
        add(InpaintMode.replace_background, "", "inpaint-replace_background")
        add(InpaintMode.add_object, _("Edit"), "edit", is_edit=True)
        add(InpaintMode.custom, "", "inpaint-custom", is_edit=None)
        return menu

    def _create_generate_region_menu(self):
        menu = theme.style_menu(QMenu(self))
        menu.addAction(
            self._mk_action(InpaintMode.automatic, _("Generate Region"), "generate-region")
        )
        menu.addAction(
            self._mk_action(
                InpaintMode.custom, _("Generate Region (Custom)"), "inpaint-custom", is_edit=None
            )
        )
        return menu

    def _create_refine_menu(self):
        menu = theme.style_menu(QMenu(self))
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Refine"), "refine"))
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Edit"), "edit", is_edit=True))
        return menu

    def _create_refine_selection_menu(self):
        menu = theme.style_menu(QMenu(self))
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Refine"), "refine"))
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Edit"), "edit", is_edit=True))
        menu.addAction(
            self._mk_action(
                InpaintMode.custom, _("Refine (Custom)"), "inpaint-custom", is_edit=None
            )
        )
        return menu

    def _create_refine_region_menu(self):
        menu = theme.style_menu(QMenu(self))
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Refine Region"), "refine-region"))
        menu.addAction(
            self._mk_action(InpaintMode.custom, _("Refine Region (Custom)"), "inpaint-custom")
        )
        return menu

    def _create_edit_menu(self):
        menu = theme.style_menu(QMenu(self))
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Edit"), "edit"))
        menu.addAction(self._mk_action(InpaintMode.custom, _("Edit (Custom)"), "inpaint-custom"))
        return menu

    def _create_dlss5_menu(self, title: str, use_selection: bool):
        menu = theme.style_menu(QMenu(title, self))
        menu.setIcon(theme.icon("workspace-upscaling"))
        for style in ("Cinematic", "Default", "Natural"):
            # triggered passes its checked flag first, so it must not land in `s`
            menu.addAction(
                _(style),
                lambda checked=False, s=style: self.model.enhance_canvas_dlss5(s, use_selection),
            )
        return menu

    def _create_ai_select_menu(self):
        menu = theme.style_menu(QMenu(_("AI Select"), self))
        menu.setIcon(theme.icon("control-segmentation"))
        subject = ensure(menu.addAction(_("Select Subject"), lambda: self.model.select_subject()))
        text = ensure(menu.addAction(_("Select by Text..."), actions.select_by_text))
        menu.addSeparator()
        remove = ensure(
            menu.addAction(_("Remove Background"), lambda: self.model.remove_background())
        )
        remove.setToolTip(
            _(
                "Hide the background of the active layer with a transparency mask - nothing is erased"
            )
        )
        menu.setToolTipsVisible(True)
        self._ai_select_actions = {SegmentMode.subject: [subject, remove], SegmentMode.text: [text]}
        return menu

    def _update_ai_select_menu(self):
        any_available = False
        for mode, entries in self._ai_select_actions.items():
            available = self.model.segment_available(mode)
            any_available = any_available or available
            for action in entries:
                action.setEnabled(available)
        return any_available

    def show_inpaint_menu(self):
        width = self.generate_button.width() + self.inpaint_mode_button.width()
        pos = QPoint(0, self.generate_button.height())
        if not self.model.edit_mode and self.model.arch.is_edit:
            menu = self.edit_menu
        elif self.model.strength == 1.0:
            if self.model.region_only:
                menu = self.generate_region_menu
            elif self.model.document.selection_bounds:
                menu = self.inpaint_menu
                menu.actions()[-2].setEnabled(self.model.can_edit)
            else:
                menu = self.generate_menu
                menu.actions()[1].setEnabled(self.model.can_edit)
        else:
            if self.model.region_only:
                menu = self.refine_region_menu
            elif self.model.document.selection_bounds:
                menu = self.refine_selection_menu
                menu.actions()[1].setEnabled(self.model.can_edit)
            else:
                menu = self.refine_menu
                menu.actions()[1].setEnabled(self.model.can_edit)

        # Appended only for the duration of the popup: the mode menus above are indexed
        # by position, and DLSS5 depends on the optional node pack on the server.
        extra = []
        if self.model.dlss5_available:
            extra = [menu.addSeparator(), menu.addMenu(self.dlss5_canvas_menu)]
            if self.model.document.selection_bounds:
                extra.append(menu.addMenu(self.dlss5_selection_menu))
        if self._update_ai_select_menu():  # only with segmentation nodes on the server
            extra += [menu.addSeparator(), menu.addMenu(self.ai_select_menu)]

        menu.setFixedWidth(width)
        menu.exec_(self.generate_button.mapToGlobal(pos))
        for action in extra:
            menu.removeAction(action)

    def change_inpaint_mode(self, mode: InpaintMode, is_edit: bool | None):
        self.model.inpaint.mode = mode
        if is_edit is not None:
            self.model.edit_mode = is_edit

    def toggle_region_only(self, checked: bool):
        self.model.region_only = checked

    def add_region(self):
        self.model.active_regions.create_region_group()

    def add_control(self):
        self.model.active_regions.add_control()

    def update_generate_options(self):
        if not self.model.has_document:
            return

        arch = self.model.arch
        self.strength_slider.setVisible(arch is not Arch.qwen_l)
        self.layer_count_widget.setVisible(arch is Arch.qwen_l)

        regions = self.model.active_regions
        self.region_prompt.regions = regions

        has_regions = len(regions) > 0
        has_active_region = regions.is_linked(self.model.layers.active)
        is_region_only = has_regions and has_active_region and self.model.region_only
        is_edit = self.model.is_editing
        self.region_mask_button.setVisible(has_regions)
        self.region_mask_button.setEnabled(has_active_region)
        self.region_mask_button.setIcon(_region_mask_button_icons[is_region_only])

        if self.model.document.selection_bounds is None and not is_region_only:
            # the menu also holds DLSS5 Enhance Canvas, which needs no selection
            has_menu = self.model.can_toggle_edit or self.model.dlss5_available
            self.inpaint_mode_button.setVisible(has_menu)
            self.custom_inpaint.setVisible(False)
            if is_edit:
                icon = "edit"
                text = _("Edit")
            elif self.model.strength == 1.0:
                icon = "workspace-generation"
                text = _("Generate")
            else:
                icon = "refine"
                text = _("Refine")
        else:
            self.inpaint_mode_button.setVisible(True)
            self.custom_inpaint.setVisible(self.model.inpaint.mode is InpaintMode.custom)
            mode = self.model.resolve_inpaint_mode()
            text = _("Generate")
            if is_edit:
                text = _("Edit")
            elif self.model.strength < 1:
                text = _("Refine")
            if is_region_only:
                text += " " + _("Region")
            if mode is InpaintMode.custom:
                text += " " + _("(Custom)")
            if self.model.strength == 1.0 and not is_edit:
                if mode is InpaintMode.custom:
                    icon = "inpaint-custom"
                elif is_region_only:
                    icon = "generate-region"
                else:
                    icon = f"inpaint-{mode.name}"
                    text = self._inpaint_text[mode]
            elif not is_edit:
                if mode is InpaintMode.custom:
                    icon = "inpaint-custom"
                elif is_region_only:
                    icon = "refine-region"
                else:
                    icon = "refine"
            else:
                if mode is InpaintMode.custom:
                    icon = "inpaint-custom"
                else:
                    icon = "edit"

        text += f" ({self.model.batch_count})"
        self.generate_button.operation = text
        self.generate_button.setIcon(theme.icon(icon))


_region_mask_button_icons = {
    True: theme.icon("region-alpha-active"),
    False: theme.icon("region-alpha"),
}
