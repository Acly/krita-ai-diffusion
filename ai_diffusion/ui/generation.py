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
)

from ..properties import Binding, bind, Bind
from ..image import Bounds, Extent, Image
from ..jobs import Job, JobQueue, JobState, JobKind
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
    QueueWidget,
    ControlListWidget,
)


class HistoryWidget(QListWidget):
    _jobs: JobQueue
    _connections: list[QMetaObject.Connection]
    _last_prompt: str | None = None
    _last_bounds: Bounds | None = None

    item_activated = pyqtSignal(QListWidgetItem)

    _applied_icon = Image.load(theme.icon_path / "star.png")
    _list_css = f"QListWidget::item:selected {{ border: 1px solid {theme.grey}; }}"
    _button_css = f"""
        QPushButton {{
            border: 1px solid {theme.grey};
            background: {"rgba(96, 96, 96, 180)" if theme.is_dark else "rgba(192, 192, 192, 180)"};
            padding: 2px;
        }}
        QPushButton:hover {{
            background: {"rgba(96, 96, 96, 220)" if theme.is_dark else "rgba(192, 192, 192, 220)"};
        }}
    """

    def __init__(self, parent):
        super().__init__(parent)
        self._jobs = JobQueue()
        self._connections = []

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setResizeMode(QListView.Adjust)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFlow(QListView.LeftToRight)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(96, 96))
        self.setFrameStyle(QListWidget.NoFrame)
        self.setStyleSheet(self._list_css)
        self.itemClicked.connect(self.handle_preview_click)
        self.itemDoubleClicked.connect(self.item_activated)

        self._apply_button = QPushButton(theme.icon("apply"), "Apply", self)
        self._apply_button.setStyleSheet(self._button_css)
        self._apply_button.setVisible(False)
        self._apply_button.clicked.connect(self._activate_selection)
        if scrollbar := self.verticalScrollBar():
            scrollbar.valueChanged.connect(self.update_apply_button)

    @property
    def jobs(self):
        return self._jobs

    @jobs.setter
    def jobs(self, jobs: JobQueue):
        Binding.disconnect_all(self._connections)
        self._jobs = jobs
        self._connections = [
            jobs.selection_changed.connect(self.update_selection),
            self.itemSelectionChanged.connect(self.select_item),
            jobs.job_finished.connect(self.add),
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

        if self._last_prompt != job.prompt or self._last_bounds != job.bounds:
            self._last_prompt = job.prompt
            self._last_bounds = job.bounds
            prompt = job.prompt if job.prompt != "" else "<no prompt>"

            header = QListWidgetItem(f"{job.timestamp:%H:%M} - {prompt}")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setData(Qt.ItemDataRole.UserRole, job.id)
            header.setData(Qt.ItemDataRole.ToolTipRole, job.prompt)
            header.setSizeHint(QSize(9999, self.fontMetrics().lineSpacing() + 4))
            header.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            self.addItem(header)

        for i, img in enumerate(job.results):
            item = QListWidgetItem(self._image_thumbnail(job, i), None)  # type: ignore (text can be None)
            item.setData(Qt.ItemDataRole.UserRole, job.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, i)
            item.setData(
                Qt.ItemDataRole.ToolTipRole,
                f"{job.prompt}\nClick to toggle preview, double-click to apply.",
            )
            self.addItem(item)

        if scroll_to_bottom:
            self.scrollToBottom()

    def update_selection(self):
        selection = self._jobs.selection
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
            self._apply_button.setVisible(True)
            self._apply_button.move(
                QPoint(rect.left() + 3, rect.bottom() - self._apply_button.height() - 2)
            )
            self._apply_button.resize(QSize(rect.width() - 6, self._apply_button.height()))
            if rect.width() < 56:
                self._apply_button.setText("")
            else:
                self._apply_button.setText("Apply")
        else:
            self._apply_button.setVisible(False)

    def update_image_thumbnail(self, id: JobQueue.Item):
        if item := self._find(id):
            job = ensure(self._jobs.find(id.job))
            item.setIcon(self._image_thumbnail(job, id.image))

    def select_item(self):
        items = self.selectedItems()
        if len(items) > 0:
            self._jobs.selection = self._item_data(items[0])
        else:
            self._jobs.selection = None

    def _activate_selection(self):
        items = self.selectedItems()
        if len(items) > 0:
            self.item_activated.emit(items[0])

    def is_finished(self, job: Job):
        return job.kind is JobKind.diffusion and job.state is JobState.finished

    def prune(self, jobs: JobQueue):
        first_id = next((job.id for job in jobs if self.is_finished(job)), None)
        while self.count() > 0 and ensure(self.item(0)).data(Qt.ItemDataRole.UserRole) != first_id:
            self.takeItem(0)

    def rebuild(self):
        self.clear()
        for job in filter(self.is_finished, self._jobs):
            self.add(job)

    def item_info(self, item: QListWidgetItem) -> tuple[str, int]:  # job id, image index
        return item.data(Qt.ItemDataRole.UserRole), item.data(Qt.ItemDataRole.UserRole + 1)

    def handle_preview_click(self, item: QListWidgetItem):
        if item.text() != "" and item.text() != "<no prompt>":
            if clipboard := QGuiApplication.clipboard():
                prompt = item.data(Qt.ItemDataRole.ToolTipRole)
                clipboard.setText(prompt)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        # make single click deselect current item (usually requires Ctrl+click)
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
        if job.result_was_used(index):  # add tiny star icon to mark used results
            thumb = Image.scale_to_fit(image, Extent(96 * 2, 96 * 2))
            thumb.draw_image(self._applied_icon, offset=(-28, 4))
            return thumb.to_icon()
        return image.to_icon()


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
        self.queue_button = QueueWidget(self)
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
            self.queue_button.jobs = model.jobs
            self.history.jobs = model.jobs

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
