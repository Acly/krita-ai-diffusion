from __future__ import annotations
from PyQt5.QtCore import Qt, QMetaObject, QSize, QPoint, pyqtSignal
from PyQt5.QtGui import QGuiApplication, QMouseEvent
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QProgressBar,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QListView,
    QSizePolicy,
    QMenu,
)

from ..properties import Binding, bind, Bind
from ..image import Bounds, Extent, Image
from ..jobs import Job, JobQueue, JobState, JobKind, JobParams
from ..model import Model
from ..root import root
from ..settings import settings
from ..util import ensure
from . import theme
from .widget import (
    WorkspaceSelectWidget,
    StyleSelectWidget,
    TextPromptWidget,
    StrengthWidget,
    ControlLayerButton,
    QueueButton,
    ControlListWidget,
)


class HistoryWidget(QListWidget):
    _model: Model
    _connections: list[QMetaObject.Connection]
    _last_job_params: JobParams | None = None

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

    def __init__(self, parent):
        super().__init__(parent)
        self._model = root.active_model
        self._connections = []

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setResizeMode(QListView.Adjust)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFlow(QListView.LeftToRight)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(theme.screen_scale(self, QSize(self._thumb_size, self._thumb_size)))
        self.setFrameStyle(QListWidget.NoFrame)
        self.setStyleSheet(self._list_css)
        self.setDragEnabled(False)
        self.itemClicked.connect(self.handle_preview_click)
        self.itemDoubleClicked.connect(self.item_activated)

        self._apply_button = QPushButton(theme.icon("apply"), "Apply", self)
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
    def model_(self, model: Model):
        Binding.disconnect_all(self._connections)
        self._model = model
        jobs = model.jobs
        self._connections = [
            jobs.selection_changed.connect(self.update_selection),
            self.itemSelectionChanged.connect(self.select_item),
            jobs.job_finished.connect(self.add),
            jobs.job_discarded.connect(self.remove),
            jobs.result_used.connect(self.update_image_thumbnail),
        ]
        self.rebuild()
        self.update_selection()

    def add(self, job: Job):
        if job.state is not JobState.finished or job.kind is not JobKind.diffusion:
            return  # Only finished diffusion jobs have images to show

        scrollbar = self.verticalScrollBar()
        scroll_to_bottom = (
            scrollbar and scrollbar.isVisible() and scrollbar.value() >= scrollbar.maximum() - 4
        )
        prompt = job.params.prompt if job.params.prompt != "" else "<no prompt>"

        if not JobParams.equal_ignore_seed(self._last_job_params, job.params):
            self._last_job_params = job.params
            strength = f"{job.params.strength*100:.0f}% - " if job.params.strength != 1.0 else ""

            header = QListWidgetItem(f"{job.timestamp:%H:%M} - {strength}{prompt}")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setData(Qt.ItemDataRole.UserRole, job.id)
            header.setData(Qt.ItemDataRole.ToolTipRole, job.params.prompt)
            header.setSizeHint(QSize(9999, self.fontMetrics().lineSpacing() + 4))
            header.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            self.addItem(header)

        for i, img in enumerate(job.results):
            item = QListWidgetItem(self._image_thumbnail(job, i), None)  # type: ignore (text can be None)
            item.setData(Qt.ItemDataRole.UserRole, job.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, i)
            item.setData(
                Qt.ItemDataRole.ToolTipRole,
                f"{prompt} @ {job.params.strength*100:.0f}% strength\n"
                "Click to toggle preview, double-click to apply.",
            )
            self.addItem(item)

        if scroll_to_bottom:
            self.scrollToBottom()

    def remove(self, job: Job):
        item_was_selected = False
        with theme.SignalBlocker(self):
            # Remove all the job's items before triggering potential selection changes
            for i in range(self.count()):
                item = self.item(i)
                while item and item.data(Qt.ItemDataRole.UserRole) == job.id:
                    item_was_selected = item_was_selected or item.isSelected()
                    self.takeItem(i)
                    item = self.item(i)
                break
        if item_was_selected:
            self._model.jobs.selection = None
        else:
            self.update_apply_button()  # selection may have moved

    def update_selection(self):
        selection = self._model.jobs.selection
        if selection is None:
            self.clearSelection()
        elif selection:
            item = self._find(selection)
            if item is not None and not item.isSelected():
                item.setSelected(True)
        self.update_apply_button()

    def update_apply_button(self):
        selected = self.selectedItems()
        if len(selected) > 0:
            rect = self.visualItemRect(selected[0])
            context_visible = rect.width() >= 0.6 * self.iconSize().width()
            apply_text_visible = rect.width() >= 0.85 * self.iconSize().width()
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
            self._apply_button.setText("Apply" if apply_text_visible else "")
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
        items = self.selectedItems()
        if len(items) > 0:
            self._model.jobs.selection = self._item_data(items[0])
        else:
            self._model.jobs.selection = None

    def _activate_selection(self):
        items = self.selectedItems()
        if len(items) > 0:
            self.item_activated.emit(items[0])

    def is_finished(self, job: Job):
        return job.kind is JobKind.diffusion and job.state is JobState.finished

    def rebuild(self):
        self.clear()
        for job in filter(self.is_finished, self._model.jobs):
            self.add(job)

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

    def mousePressEvent(self, e: QMouseEvent | None) -> None:
        # make single click deselect current item (usually requires Ctrl+click)
        if e is not None and e.button() == Qt.MouseButton.LeftButton:
            mods = e.modifiers()
            mods |= Qt.KeyboardModifier.ControlModifier
            e = QMouseEvent(
                e.type(),
                e.localPos(),
                e.windowPos(),
                e.screenPos(),
                e.button(),
                e.buttons(),
                mods,
                e.source(),
            )
        return super().mousePressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update_apply_button()

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
        if thumb.extent.height < self._thumb_size:
            thumb = Image.crop(thumb, Bounds(0, 0, thumb.extent.width, self._thumb_size))
        if job.result_was_used(index):  # add tiny star icon to mark used results
            thumb.draw_image(self._applied_icon, offset=(-28, 4))
        return thumb.to_icon()

    def _show_context_menu(self, pos: QPoint):
        item = self.itemAt(pos)
        if item is not None:
            menu = QMenu(self)
            menu.addAction("Copy Prompt", self._copy_prompt)
            menu.addAction("Copy Strength", self._copy_strength)
            menu.addAction("Copy Seed", self._copy_seed)
            menu.addSeparator()
            save_action = ensure(menu.addAction("Save Image", self._save_image))
            if self._model.document.filename == "":
                save_action.setEnabled(False)
                save_action.setToolTip(
                    "Save as separate image to the same folder as the document.\nMust save the"
                    " document first!"
                )
                menu.setToolTipsVisible(True)
            menu.exec(self.mapToGlobal(pos))

    def _show_context_menu_dropdown(self):
        pos = self._context_button.pos()
        pos.setY(pos.y() + self._context_button.height())
        self._show_context_menu(pos)

    def _copy_prompt(self):
        if job := self.selected_job:
            self._model.prompt = job.params.prompt
            self._model.negative_prompt = job.params.negative_prompt

    def _copy_strength(self):
        if job := self.selected_job:
            self._model.strength = job.params.strength

    def _copy_seed(self):
        if job := self.selected_job:
            self._model.fixed_seed = True
            self._model.seed = job.params.seed

    def _save_image(self):
        items = self.selectedItems()
        if len(items) > 0:
            job_id, image_index = self.item_info(items[0])
            self._model.save_result(job_id, image_index)


class GenerationWidget(QWidget):
    _model: Model
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        settings.changed.connect(self.update_settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 2, 0)
        self.setLayout(layout)

        self.workspace_select = WorkspaceSelectWidget(self)
        self.style_select = StyleSelectWidget(self)

        style_layout = QHBoxLayout()
        style_layout.addWidget(self.workspace_select)
        style_layout.addWidget(self.style_select)
        layout.addLayout(style_layout)

        self.prompt_textbox = TextPromptWidget(parent=self)
        self.prompt_textbox.line_count = settings.prompt_line_count

        self.negative_textbox = TextPromptWidget(line_count=1, is_negative=True, parent=self)
        self.negative_textbox.setVisible(settings.show_negative_prompt)

        prompt_layout = QVBoxLayout()
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(2)
        prompt_layout.addWidget(self.prompt_textbox)
        prompt_layout.addWidget(self.negative_textbox)
        layout.addLayout(prompt_layout)

        self.control_list = ControlListWidget(self)
        layout.addWidget(self.control_list)

        self.strength_slider = StrengthWidget(parent=self)
        self.add_control_button = ControlLayerButton(self)
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.add_control_button)
        layout.addLayout(strength_layout)

        self.generate_button = QPushButton("Generate", self)
        self.generate_button.setMinimumHeight(int(self.generate_button.sizeHint().height() * 1.2))
        self.queue_button = QueueButton(parent=self)
        self.queue_button.setMinimumHeight(self.generate_button.minimumHeight())
        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.generate_button)
        actions_layout.addWidget(self.queue_button)
        layout.addLayout(actions_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        self.error_text = QLabel(self)
        self.error_text.setStyleSheet("font-weight: bold; color: red;")
        self.error_text.setWordWrap(True)
        self.error_text.setVisible(False)
        layout.addWidget(self.error_text)

        self.history = HistoryWidget(self)
        self.history.item_activated.connect(self.apply_result)
        layout.addWidget(self.history)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            self._model_bindings = [
                bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
                bind(model, "style", self.style_select, "value"),
                bind(model, "prompt", self.prompt_textbox, "text"),
                bind(model, "negative_prompt", self.negative_textbox, "text"),
                bind(model, "strength", self.strength_slider, "value"),
                model.progress_changed.connect(self.update_progress),
                model.error_changed.connect(self.error_text.setText),
                model.has_error_changed.connect(self.error_text.setVisible),
                self.add_control_button.clicked.connect(model.control.add),
                self.prompt_textbox.activated.connect(model.generate),
                self.negative_textbox.activated.connect(model.generate),
                self.generate_button.clicked.connect(model.generate),
            ]
            self.control_list.model = model
            self.queue_button.model = model
            self.history.model_ = model

    def update_progress(self):
        self.progress_bar.setValue(int(self.model.progress * 100))

    def update_settings(self, key: str, value):
        if key == "prompt_line_count":
            self.prompt_textbox.line_count = value
        elif key == "show_negative_prompt":
            self.negative_textbox.text = ""
            self.negative_textbox.setVisible(value)

    def apply_result(self, item: QListWidgetItem):
        job_id, index = self.history.item_info(item)
        self.model.apply_result(job_id, index)
