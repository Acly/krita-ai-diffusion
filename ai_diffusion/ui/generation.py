from __future__ import annotations

from typing import ClassVar

from PyQt6.QtCore import (
    QMetaObject,
    QPoint,
    Qt,
    QUuid,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QPalette,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..backend.api import InpaintContext
from ..backend.resources import Arch
from ..backend.workflow import FillMode, InpaintMode
from ..localization import translate as _
from ..model.jobs import JobKind
from ..model.model import DocumentModel, ProgressKind
from ..model.properties import Bind, Binding, bind, bind_combo, bind_toggle
from ..model.root import root
from . import theme
from .history import HistoryWidget, PreviewReel
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
)


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
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
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
        self.fill_mode_combo.setEnabled(is_one(self.model.strength) and not self.model.is_editing)
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

        self.strength_slider = StrengthWidget()
        self.layer_count_widget = LayerCountWidget(self)
        self.layer_count_widget.setVisible(False)
        self.add_region_button = create_wide_tool_button("region-add", _("Add Region"), self)
        self.add_control_button = create_wide_tool_button(
            "control-add", _("Add Control Layer"), self
        )
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_slider.widget())
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

        actions_layout = QHBoxLayout()
        actions_layout.addLayout(generate_layout)
        actions_layout.addWidget(self.queue_button)
        layout.addLayout(actions_layout)

        self.progress_bar = ProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.preview_reel = PreviewReel(self)
        layout.addWidget(self.preview_reel)

        self.error_box = ErrorBox(self)
        layout.addWidget(self.error_box)

        self.history = HistoryWidget(self)
        self.history.item_activated.connect(self.apply_result)
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
                model.document.selection_bounds_changed.connect(self.update_generate_options),
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
            ]
            self.region_prompt.regions = model.active_regions
            self.custom_inpaint.model = model
            self.generate_button.model = model
            self.queue_button.model = model
            self.progress_bar.model = model
            self.strength_slider.model = model
            self.preview_reel.model_ = model
            self.history.model_ = model
            self.update_generate_options()

    def apply_result(self, item: QListWidgetItem):
        job_id, index = self.history.item_info(item)
        self.model.apply_generated_result(job_id, index)

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
        menu = QMenu(self)
        menu.addAction(
            self._mk_action(InpaintMode.automatic, _("Generate"), "workspace-generation")
        )
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Edit"), "edit", is_edit=True))
        return menu

    def _create_inpaint_menu(self):
        menu = QMenu(self)

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
        menu = QMenu(self)
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
        menu = QMenu(self)
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Refine"), "refine"))
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Edit"), "edit", is_edit=True))
        return menu

    def _create_refine_selection_menu(self):
        menu = QMenu(self)
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Refine"), "refine"))
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Edit"), "edit", is_edit=True))
        menu.addAction(
            self._mk_action(
                InpaintMode.custom, _("Refine (Custom)"), "inpaint-custom", is_edit=None
            )
        )
        return menu

    def _create_refine_region_menu(self):
        menu = QMenu(self)
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Refine Region"), "refine-region"))
        menu.addAction(
            self._mk_action(InpaintMode.custom, _("Refine Region (Custom)"), "inpaint-custom")
        )
        return menu

    def _create_edit_menu(self):
        menu = QMenu(self)
        menu.addAction(self._mk_action(InpaintMode.automatic, _("Edit"), "edit"))
        menu.addAction(self._mk_action(InpaintMode.custom, _("Edit (Custom)"), "inpaint-custom"))
        return menu

    def show_inpaint_menu(self):
        width = self.generate_button.width() + self.inpaint_mode_button.width()
        pos = QPoint(0, self.generate_button.height())
        if not self.model.edit_mode and self.model.arch.is_edit:
            menu = self.edit_menu
        elif is_one(self.model.strength):
            if self.model.region_only:
                menu = self.generate_region_menu
            elif self.model.document.selection_bounds:
                menu = self.inpaint_menu
                menu.actions()[-2].setEnabled(self.model.can_edit)
            else:
                menu = self.generate_menu
        else:
            if self.model.region_only:
                menu = self.refine_region_menu
            elif self.model.document.selection_bounds:
                menu = self.refine_selection_menu
                menu.actions()[1].setEnabled(self.model.can_edit)
            else:
                menu = self.refine_menu
                menu.actions()[1].setEnabled(self.model.can_edit)

        menu.setFixedWidth(width)
        menu.exec(self.generate_button.mapToGlobal(pos))

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
            self.inpaint_mode_button.setVisible(self.model.can_toggle_edit)
            self.custom_inpaint.setVisible(False)
            if is_edit:
                icon = "edit"
                text = _("Edit")
            elif is_one(self.model.strength):
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
            if is_one(self.model.strength) and not is_edit:
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

        self.generate_button.operation = text
        self.generate_button.setIcon(theme.icon(icon))


_region_mask_button_icons = {
    True: theme.icon("region-alpha-active"),
    False: theme.icon("region-alpha"),
}
