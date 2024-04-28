from __future__ import annotations
from PyQt5.QtCore import Qt, QMetaObject, QSize, QPoint, QUuid, pyqtSignal
from PyQt5.QtGui import QGuiApplication, QMouseEvent
from PyQt5.QtWidgets import (
    QAction,
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
    QToolButton,
    QComboBox,
    QCheckBox,
    QMenu,
    QShortcut,
)

from ..properties import Binding, Bind, bind, bind_combo, bind_toggle
from ..image import Bounds, Extent, Image
from ..jobs import Job, JobQueue, JobState, JobKind, JobParams
from ..model import Model, InpaintContext
from ..root import root
from ..workflow import InpaintMode, FillMode
from ..settings import settings
from ..util import ensure
from . import theme
from .control import ControlLayerButton, ControlListWidget
from .widget import (
    WorkspaceSelectWidget,
    StyleSelectWidget,
    StrengthWidget,
    QueueButton,
    RegionPromptWidget,
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
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
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

        widget_context = Qt.ShortcutContext.WidgetShortcut
        QShortcut(Qt.Key.Key_Delete, self, self._discard_image, self._discard_image, widget_context)
        QShortcut(
            Qt.Key.Key_Space, self, self._toggle_selection, self._toggle_selection, widget_context
        )

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
            jobs.result_discarded.connect(self.remove_image),
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
        self._remove_items(ensure(job.id))

    def remove_image(self, id: JobQueue.Item):
        self._remove_items(id.job, id.image)

    def _remove_items(self, job_id: str, image_index: int = -1):
        def _item_job_id(item: QListWidgetItem | None):
            return item.data(Qt.ItemDataRole.UserRole) if item else None

        item_was_selected = False
        with theme.SignalBlocker(self):
            # Remove all the job's items before triggering potential selection changes
            current = next(i for i in range(self.count()) if _item_job_id(self.item(i)) == job_id)
            item = self.item(current)
            while item and _item_job_id(item) == job_id:
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
            font = self._apply_button.fontMetrics()
            context_visible = rect.width() >= 0.6 * self.iconSize().width()
            apply_text_visible = font.width("Apply") < 0.35 * rect.width()
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

    def _toggle_selection(self):
        self._model.jobs.toggle_selection()

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
                return
        super().mousePressEvent(e)

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
        min_height = min(4 * self._apply_button.height(), 2 * self._thumb_size)
        if thumb.extent.height < min_height:
            thumb = Image.crop(thumb, Bounds(0, 0, thumb.extent.width, min_height))
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
            menu.addAction("Discard Image", self._discard_image)
            menu.exec(self.mapToGlobal(pos))

    def _show_context_menu_dropdown(self):
        pos = self._context_button.pos()
        pos.setY(pos.y() + self._context_button.height())
        self._show_context_menu(pos)

    def _copy_prompt(self):
        if job := self.selected_job:
            self._model.regions.active.prompt = job.params.prompt
            self._model.regions.active.negative_prompt = job.params.negative_prompt

    def _copy_strength(self):
        if job := self.selected_job:
            self._model.strength = job.params.strength

    def _copy_seed(self):
        if job := self.selected_job:
            self._model.fixed_seed = True
            self._model.seed = job.params.seed

    def _save_image(self):
        items = self.selectedItems()
        for item in items:
            job_id, image_index = self.item_info(item)
            self._model.save_result(job_id, image_index)

    def _discard_image(self):
        items = self.selectedItems()
        for item in items:
            job_id, image_index = self.item_info(item)
            self._model.jobs.discard(job_id, image_index)


class CustomInpaintWidget(QWidget):
    _model: Model
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._model = root.active_model
        self._model_bindings = []

        self.use_inpaint_button = QCheckBox(self)
        self.use_inpaint_button.setText("Seamless")
        self.use_inpaint_button.setToolTip("Generate content which blends into the surroundings")

        self.use_prompt_focus_button = QCheckBox(self)
        self.use_prompt_focus_button.setText("Focus")
        self.use_prompt_focus_button.setToolTip(
            "Use the text prompt to describe the selected region rather than the context area"
        )

        self.fill_mode_combo = QComboBox(self)
        fill_icon = theme.icon("fill")
        self.fill_mode_combo.addItem(theme.icon("fill-empty"), "None", FillMode.none)
        self.fill_mode_combo.addItem(fill_icon, "Neutral", FillMode.neutral)
        self.fill_mode_combo.addItem(fill_icon, "Blur", FillMode.blur)
        self.fill_mode_combo.addItem(fill_icon, "Border", FillMode.border)
        self.fill_mode_combo.addItem(fill_icon, "Inpaint", FillMode.inpaint)
        self.fill_mode_combo.setStyleSheet(theme.flat_combo_stylesheet)
        self.fill_mode_combo.setToolTip("Pre-fill the selected region before diffusion")

        self.context_combo = QComboBox(self)
        ctx_icon = lambda name: theme.icon(f"context-{name}")
        self.context_combo.addItem(
            ctx_icon("automatic"), "Automatic Context", InpaintContext.automatic
        )
        self.context_combo.addItem(ctx_icon("mask"), "Selection Bounds", InpaintContext.mask_bounds)
        self.context_combo.addItem(ctx_icon("image"), "Entire Image", InpaintContext.entire_image)
        self.context_combo.setStyleSheet(theme.flat_combo_stylesheet)
        self.context_combo.setToolTip(
            "Part of the image around the selection which is used as context."
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
        layout.addWidget(self.fill_mode_combo)
        layout.addWidget(self.context_combo, 1)
        self.setLayout(layout)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            self._model_bindings = [
                bind_combo(model.inpaint, "fill", self.fill_mode_combo),
                bind_toggle(model.inpaint, "use_inpaint", self.use_inpaint_button),
                bind_toggle(model.inpaint, "use_prompt_focus", self.use_prompt_focus_button),
                model.layers.changed.connect(self.update_context_layers),
                model.strength_changed.connect(self.update_fill_enabled),
            ]
            self.update_fill_enabled()
            self.update_context_layers()
            self.update_context()

    def update_fill_enabled(self):
        self.fill_mode_combo.setEnabled(self.model.strength == 1.0)

    def update_context_layers(self):
        current = self.context_combo.currentData()
        with theme.SignalBlocker(self.context_combo):
            while self.context_combo.count() > 3:
                self.context_combo.removeItem(self.context_combo.count() - 1)
            icon = theme.icon("context-layer")
            for layer in self._model.layers.masks:
                self.context_combo.addItem(icon, f"{layer.name()}", layer.uniqueId())
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


class GenerationWidget(QWidget):
    _model: Model
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []

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
        self.add_control_button = ControlLayerButton(self)
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.add_control_button)
        layout.addLayout(strength_layout)

        self.custom_inpaint = CustomInpaintWidget(self)
        layout.addWidget(self.custom_inpaint)

        self.generate_button = QPushButton(self)
        self.generate_button.setMinimumHeight(int(self.generate_button.sizeHint().height() * 1.2))

        self.inpaint_mode_button = QToolButton(self)
        self.inpaint_mode_button.setArrowType(Qt.ArrowType.DownArrow)
        self.inpaint_mode_button.setMinimumHeight(self.generate_button.minimumHeight())
        self.inpaint_mode_button.clicked.connect(self.show_inpaint_menu)
        self.inpaint_menu = self._create_inpaint_menu()
        self.refine_menu = self._create_refine_menu()

        generate_layout = QHBoxLayout()
        generate_layout.setSpacing(0)
        generate_layout.addWidget(self.generate_button)
        generate_layout.addWidget(self.inpaint_mode_button)

        self.queue_button = QueueButton(parent=self)
        self.queue_button.setMinimumHeight(self.generate_button.minimumHeight())

        actions_layout = QHBoxLayout()
        actions_layout.addLayout(generate_layout)
        actions_layout.addWidget(self.queue_button)
        layout.addLayout(actions_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1000)
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

        self.update_generate_button()

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
                bind(model, "strength", self.strength_slider, "value"),
                model.inpaint.mode_changed.connect(self.update_generate_button),
                model.strength_changed.connect(self.update_generate_button),
                model.document.selection_bounds_changed.connect(self.update_generate_button),
                model.progress_changed.connect(self.update_progress),
                model.error_changed.connect(self.error_text.setText),
                model.has_error_changed.connect(self.error_text.setVisible),
                self.add_control_button.clicked.connect(model.regions.add_control),
                self.region_prompt.activated.connect(model.generate),
                self.generate_button.clicked.connect(model.generate),
            ]
            self.region_prompt.regions = model.regions
            self.custom_inpaint.model = model
            self.queue_button.model = model
            self.history.model_ = model
            self.update_generate_button()

    def update_progress(self):
        if self.model.progress >= 0:
            self.progress_bar.setValue(int(self.model.progress * 1000))
        else:
            if self.progress_bar.value() >= 100:
                self.progress_bar.reset()
            self.progress_bar.setValue(min(99, self.progress_bar.value() + 2))

    def apply_result(self, item: QListWidgetItem):
        job_id, index = self.history.item_info(item)
        self.model.apply_result(job_id, index)

    _inpaint_text = {
        InpaintMode.automatic: "Default (Auto-detect)",
        InpaintMode.fill: "Fill",
        InpaintMode.expand: "Expand",
        InpaintMode.add_object: "Add Content",
        InpaintMode.remove_object: "Remove Content",
        InpaintMode.replace_background: "Replace Background",
        InpaintMode.custom: "Generate (Custom)",
    }

    def _create_inpaint_action(self, mode: InpaintMode, text: str, icon: str):
        action = QAction(text, self)
        action.setIcon(theme.icon(icon))
        action.setIconVisibleInMenu(True)
        action.triggered.connect(lambda: self.change_inpaint_mode(mode))
        return action

    def _create_inpaint_menu(self):
        menu = QMenu(self)
        for mode in InpaintMode:
            text = self._inpaint_text[mode]
            menu.addAction(self._create_inpaint_action(mode, text, f"inpaint-{mode.name}"))
        return menu

    def _create_refine_menu(self):
        menu = QMenu(self)
        menu.addAction(self._create_inpaint_action(InpaintMode.automatic, "Refine", "refine"))
        menu.addAction(
            self._create_inpaint_action(InpaintMode.custom, "Refine (Custom)", "inpaint-custom")
        )
        return menu

    def show_inpaint_menu(self):
        width = self.generate_button.width() + self.inpaint_mode_button.width()
        pos = QPoint(0, self.generate_button.height())
        menu = self.inpaint_menu if self.model.strength == 1.0 else self.refine_menu
        menu.setFixedWidth(width)
        menu.exec_(self.generate_button.mapToGlobal(pos))

    def change_inpaint_mode(self, mode: InpaintMode):
        self.model.inpaint.mode = mode

    def update_generate_button(self):
        if self.model.document.selection_bounds is None:
            self.inpaint_mode_button.setVisible(False)
            self.custom_inpaint.setVisible(False)
            if self.model.strength == 1.0:
                self.generate_button.setIcon(theme.icon("workspace-generation"))
                self.generate_button.setText("Generate")
            else:
                self.generate_button.setIcon(theme.icon("refine"))
                self.generate_button.setText("Refine")
        else:
            self.inpaint_mode_button.setVisible(True)
            self.custom_inpaint.setVisible(self.model.inpaint.mode is InpaintMode.custom)
            if self.model.strength == 1.0:
                mode = self.model.resolve_inpaint_mode()
                self.generate_button.setIcon(theme.icon(f"inpaint-{mode.name}"))
                self.generate_button.setText(self._inpaint_text[mode])
            else:
                is_custom = self.model.inpaint.mode is InpaintMode.custom
                self.generate_button.setIcon(
                    theme.icon("inpaint-custom" if is_custom else "refine")
                )
                self.generate_button.setText("Refine (Custom)" if is_custom else "Refine")
        self.generate_button.setText(" " + self.generate_button.text())
