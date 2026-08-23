from __future__ import annotations

import json
from textwrap import wrap as wrap_text
from typing import ClassVar, cast

from PyQt6.QtCore import (
    QEvent,
    QItemSelectionModel,
    QMetaObject,
    QPoint,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QGuiApplication,
    QIcon,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
)
from PyQt6.QtWidgets import (
    QFrame,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ..image import Bounds, Extent, Image
from ..localization import translate as _
from ..model.jobs import Job, JobKind, JobParams, JobQueue, JobState
from ..model.model import DocumentModel, Workspace
from ..model.properties import Binding
from ..model.region import RootRegion
from ..model.root import root
from ..settings import settings
from ..style import Styles
from ..util import ensure, flatten, sequence_equal
from . import theme


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
        self._last_job_params: JobParams | None = None

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

        scrollbar = self.verticalScrollBar()
        scroll_to_bottom = scrollbar and scrollbar.value() >= scrollbar.maximum() - 4

        if not JobParams.equal_ignore_seed(self._last_job_params, job.params):
            self._last_job_params = job.params
            prompt = job.params.name if job.params.name != "" else "<no prompt>"
            strength = job.params.metadata.get("strength", 1.0)
            strength = f"{strength * 100:.0f}% - " if strength != 1.0 else ""

            header = QListWidgetItem(f"{job.timestamp.astimezone():%H:%M} - {strength}{prompt}")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setData(Qt.ItemDataRole.UserRole, job.id)
            header.setData(Qt.ItemDataRole.ToolTipRole, job.params.prompt)
            header.setSizeHint(QSize(9999, self.fontMetrics().lineSpacing() + 4))
            header.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            self.addItem(header)

        if job.kind is JobKind.diffusion:
            if job.params.is_layered:
                self._add_item(job, QListWidgetItem(self._image_thumbnail(job, 0), None))
            else:
                for i, img in enumerate(job.results):
                    self._add_item(job, QListWidgetItem(self._image_thumbnail(job, i), None), i)

        if job.kind is JobKind.animation:
            item = AnimatedListItem([
                self._image_thumbnail(job, i) for i in range(len(job.results))
            ])
            self._add_item(job, item)

        if scroll_to_bottom:
            self.scrollToBottom()

    def _add_item(self, job: Job, item: QListWidgetItem, index=0):
        item.setData(Qt.ItemDataRole.UserRole, job.id)
        item.setData(Qt.ItemDataRole.UserRole + 1, index)
        item.setData(Qt.ItemDataRole.ToolTipRole, self._job_info(job.params))
        self.addItem(item)

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
    }

    def _job_info(self, params: JobParams):
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

        if item_was_selected:
            self._model.jobs.selection = []
        else:
            self.update_apply_button()  # selection may have moved

        for i in range(self.count()):
            item = self.item(i)
            next_item = self.item(i + 1)
            if item and item.text() != "" and next_item and next_item.text() != "":
                self.takeItem(i)

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
        self.clear()
        for job in filter(self.is_finished, self._model.jobs):
            self.add(job)
        self.scrollToBottom()

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
            menu.exec(self.mapToGlobal(pos))

    def _show_context_menu_dropdown(self):
        pos = self._context_button.pos()
        pos.setY(pos.y() + self._context_button.height())
        self._show_context_menu(pos)

    def _copy_prompt(self, evaluated=False):
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

            if clipboard := QGuiApplication.clipboard():
                clipboard.setText(prompt)

            if self._model.workspace is Workspace.custom and self._model.document.is_active:
                self._model.custom.try_set_params(job.params.metadata)

    def _copy_prompt_evaluated(self):
        self._copy_prompt(evaluated=True)

    def _copy_strength(self):
        if job := self.selected_job:
            self._model.strength = job.params.strength

    def _copy_style(self):
        if (job := self.selected_job) and (style := Styles.list().find(job.params.style)):
            self._model.style = style

    def _copy_seed(self):
        if job := self.selected_job:
            self._model.fixed_seed = True
            self._model.seed = job.params.seed

    def _info_to_clipboard(self):
        if (job := self.selected_job) and (clipboard := QGuiApplication.clipboard()):
            style = Styles.list().find(job.params.style)
            data = job.params.metadata.copy()
            if style:
                data["style"] = f"{style.name} ({style.filename})"
            text = json.dumps(data, indent=2)
            clipboard.setText(text)

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
        reply = QMessageBox.warning(
            self,
            _("Clear History"),
            _("Are you sure you want to discard all generated images?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._model.jobs.clear()
            self.clear()
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
