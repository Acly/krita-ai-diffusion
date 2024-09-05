from __future__ import annotations
from PyQt5.QtCore import Qt, QMetaObject, QSize, QPoint, QUuid, pyqtSignal
from PyQt5.QtGui import QGuiApplication, QMouseEvent, QPalette, QColor
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
    QMessageBox,
)

from ..properties import Binding, Bind, bind, bind_combo, bind_toggle
from ..image import Bounds, Extent, Image
from ..jobs import Job, JobQueue, JobState, JobKind, JobParams
from ..model import Model, InpaintContext, RootRegion, ProgressKind
from ..root import root
from ..workflow import InpaintMode, FillMode
from ..localization import translate as _
from ..util import ensure
from .widget import WorkspaceSelectWidget, StyleSelectWidget, StrengthWidget, QueueButton
from .widget import GenerateButton, create_wide_tool_button
from .region import RegionPromptWidget
from . import theme


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
                + _("Click to toggle preview, double-click to apply."),
            )
            self.addItem(item)

        if scroll_to_bottom:
            self.scrollToBottom()

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
            thumb.draw_image(self._applied_icon, offset=(thumb.extent.width - 28, 4))
        return thumb.to_icon()

    def _show_context_menu(self, pos: QPoint):
        item = self.itemAt(pos)
        if item is not None:
            menu = QMenu(self)
            menu.addAction(_("Copy Prompt"), self._copy_prompt)
            menu.addAction(_("Copy Strength"), self._copy_strength)
            menu.addAction(_("Copy Seed"), self._copy_seed)
            menu.addSeparator()
            save_action = ensure(menu.addAction(_("Save Image"), self._save_image))
            if self._model.document.filename == "":
                save_action.setEnabled(False)
                save_action.setToolTip(
                    _(
                        "Save as separate image to the same folder as the document.\nMust save the document first!"
                    )
                )
                menu.setToolTipsVisible(True)
            menu.addAction(_("Discard Image"), self._discard_image)
            menu.addSeparator()
            menu.addAction(_("Clear History"), self._clear_all)
            menu.exec(self.mapToGlobal(pos))

    def _show_context_menu_dropdown(self):
        pos = self._context_button.pos()
        pos.setY(pos.y() + self._context_button.height())
        self._show_context_menu(pos)

    def _copy_prompt(self):
        if job := self.selected_job:
            active = self._model.regions.active_or_root
            active.positive = job.params.prompt
            if isinstance(active, RootRegion):
                active.negative = job.params.negative_prompt

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


class CustomInpaintWidget(QWidget):
    _model: Model
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

        self.fill_mode_combo = QComboBox(self)
        fill_icon = theme.icon("fill")
        self.fill_mode_combo.addItem(theme.icon("fill-empty"), _("None"), FillMode.none)
        self.fill_mode_combo.addItem(fill_icon, _("Neutral"), FillMode.neutral)
        self.fill_mode_combo.addItem(fill_icon, _("Blur"), FillMode.blur)
        self.fill_mode_combo.addItem(fill_icon, _("Border"), FillMode.border)
        self.fill_mode_combo.addItem(fill_icon, _("Inpaint"), FillMode.inpaint)
        self.fill_mode_combo.setStyleSheet(theme.flat_combo_stylesheet)
        self.fill_mode_combo.setToolTip(_("Pre-fill the selected region before diffusion"))

        self.context_combo = QComboBox(self)
        ctx_icon = lambda name: theme.icon(f"context-{name}")
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
        self.add_region_button = create_wide_tool_button("region-add", _("Add Region"), self)
        self.add_control_button = create_wide_tool_button(
            "control-add", _("Add Control Layer"), self
        )
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.add_control_button)
        strength_layout.addWidget(self.add_region_button)
        layout.addLayout(strength_layout)

        self.custom_inpaint = CustomInpaintWidget(self)
        layout.addWidget(self.custom_inpaint)

        self.generate_button = GenerateButton(JobKind.diffusion, self)

        self.inpaint_mode_button = QToolButton(self)
        self.inpaint_mode_button.setArrowType(Qt.ArrowType.DownArrow)
        self.inpaint_mode_button.setFixedHeight(self.generate_button.height() - 2)
        self.inpaint_mode_button.clicked.connect(self.show_inpaint_menu)
        self.inpaint_menu = self._create_inpaint_menu()
        self.refine_menu = self._create_refine_menu()
        self.generate_region_menu = self._create_generate_region_menu()
        self.refine_region_menu = self._create_refine_region_menu()

        self.region_mask_button = QToolButton(self)
        self.region_mask_button.setIcon(theme.icon("region-alpha"))
        self.region_mask_button.setCheckable(True)
        self.region_mask_button.setFixedHeight(self.generate_button.height() - 2)
        self.region_mask_button.setToolTip(
            _("Generate the active layer region only (use layer transparency as mask)")
        )

        self.use_transparency_button = QCheckBox(self)
        self.use_transparency_button.setText(_("Transparency"))
        self.use_transparency_button.setToolTip(_("Generate content on transparent background"))

        options_layout = QHBoxLayout(self)
        options_layout.setContentsMargins(5, 0, 5, 0)
        options_layout.addWidget(self.use_transparency_button)
        layout.addLayout(options_layout)

        generate_layout = QHBoxLayout()
        generate_layout.setSpacing(0)
        generate_layout.addWidget(self.generate_button)
        generate_layout.addWidget(self.inpaint_mode_button)
        generate_layout.addWidget(self.region_mask_button)

        self.queue_button = QueueButton(parent=self)
        self.queue_button.setFixedHeight(self.generate_button.height() - 2)

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
                bind_toggle(model, "use_transparency", self.use_transparency_button),
                bind_toggle(model, "region_only", self.region_mask_button),
                model.inpaint.mode_changed.connect(self.update_generate_button),
                model.strength_changed.connect(self.update_generate_button),
                model.document.selection_bounds_changed.connect(self.update_generate_button),
                model.document.layers.active_changed.connect(self.update_generate_button),
                model.regions.active_changed.connect(self.update_generate_button),
                model.region_only_changed.connect(self.update_generate_button),
                model.progress_changed.connect(self.update_progress),
                model.progress_kind_changed.connect(self.update_progress_kind),
                model.error_changed.connect(self.error_text.setText),
                model.has_error_changed.connect(self.error_text.setVisible),
                self.add_control_button.clicked.connect(model.regions.add_control),
                self.add_region_button.clicked.connect(model.regions.create_region_group),
                self.region_prompt.activated.connect(model.generate),
                self.generate_button.clicked.connect(model.generate),
            ]
            self.region_prompt.regions = model.regions
            self.custom_inpaint.model = model
            self.generate_button.model = model
            self.queue_button.model = model
            self.strength_slider.model = model
            self.history.model_ = model
            self.update_generate_button()

    def update_progress_kind(self):
        palette = self.palette()
        if self.model.progress_kind is ProgressKind.upload:
            palette.setColor(QPalette.ColorRole.Highlight, QColor(theme.progress_alt))
        self.progress_bar.setPalette(palette)

    def update_progress(self):
        if self.model.progress >= 0:
            self.progress_bar.setValue(int(self.model.progress * 1000))
        else:
            if self.progress_bar.value() >= 100:
                self.progress_bar.reset()
            self.progress_bar.setValue(min(99, self.progress_bar.value() + 2))

    def apply_result(self, item: QListWidgetItem):
        job_id, index = self.history.item_info(item)
        self.model.apply_generated_result(job_id, index)

    _inpaint_text = {
        InpaintMode.automatic: _("Default (Auto-detect)"),
        InpaintMode.fill: _("Fill"),
        InpaintMode.expand: _("Expand"),
        InpaintMode.add_object: _("Add Content"),
        InpaintMode.remove_object: _("Remove Content"),
        InpaintMode.replace_background: _("Replace Background"),
        InpaintMode.custom: _("Generate (Custom)"),
    }

    def _mk_action(self, mode: InpaintMode, text: str, icon: str):
        action = QAction(text, self)
        action.setIcon(theme.icon(icon))
        action.setIconVisibleInMenu(True)
        action.triggered.connect(lambda: self.change_inpaint_mode(mode))
        return action

    def _create_inpaint_menu(self):
        menu = QMenu(self)
        for mode in InpaintMode:
            text = self._inpaint_text[mode]
            menu.addAction(self._mk_action(mode, text, f"inpaint-{mode.name}"))
        return menu

    def _create_generate_region_menu(self):
        menu = QMenu(self)
        menu.addAction(
            self._mk_action(InpaintMode.automatic, _("Generate Region"), "generate-region")
        )
        menu.addAction(
            self._mk_action(InpaintMode.custom, _("Generate Region (Custom)"), "inpaint-custom")
        )
        return menu

    def _create_refine_menu(self):
        menu = QMenu(self)
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Refine"), "refine"))
        menu.addAction(self._mk_action(InpaintMode.custom, _("Refine (Custom)"), "inpaint-custom"))
        return menu

    def _create_refine_region_menu(self):
        menu = QMenu(self)
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Refine Region"), "refine-region"))
        menu.addAction(
            self._mk_action(InpaintMode.custom, _("Refine Region (Custom)"), "inpaint-custom")
        )
        return menu

    def show_inpaint_menu(self):
        width = self.generate_button.width() + self.inpaint_mode_button.width()
        pos = QPoint(0, self.generate_button.height())
        if self.model.strength == 1.0:
            if self.model.region_only:
                menu = self.generate_region_menu
            else:
                menu = self.inpaint_menu
        else:
            if self.model.region_only:
                menu = self.refine_region_menu
            else:
                menu = self.refine_menu
        menu.setFixedWidth(width)
        menu.exec_(self.generate_button.mapToGlobal(pos))

    def change_inpaint_mode(self, mode: InpaintMode):
        self.model.inpaint.mode = mode

    def toggle_region_only(self, checked: bool):
        self.model.region_only = checked

    def update_generate_button(self):
        if not self.model.has_document:
            return
        has_regions = len(self.model.regions) > 0
        has_active_region = self.model.regions.is_linked(self.model.layers.active)
        is_region_only = has_regions and has_active_region and self.model.region_only
        self.region_mask_button.setVisible(has_regions)
        self.region_mask_button.setEnabled(has_active_region)
        self.region_mask_button.setIcon(_region_mask_button_icons[is_region_only])

        if self.model.document.selection_bounds is None and not is_region_only:
            self.inpaint_mode_button.setVisible(False)
            self.custom_inpaint.setVisible(False)
            if self.model.strength == 1.0:
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
            if self.model.strength < 1:
                text = _("Refine")
            if is_region_only:
                text += " " + _("Region")
            if mode is InpaintMode.custom:
                text += " " + _("(Custom)")
            if self.model.strength == 1.0:
                if mode is InpaintMode.custom:
                    icon = "inpaint-custom"
                elif is_region_only:
                    icon = "generate-region"
                else:
                    icon = f"inpaint-{mode.name}"
                    text = self._inpaint_text[mode]
            else:
                if mode is InpaintMode.custom:
                    icon = "inpaint-custom"
                elif is_region_only:
                    icon = "refine-region"
                else:
                    icon = "refine"

        self.generate_button.operation = text
        self.generate_button.setIcon(theme.icon(icon))


_region_mask_button_icons = {
    True: theme.icon("region-alpha-active"),
    False: theme.icon("region-alpha"),
}
