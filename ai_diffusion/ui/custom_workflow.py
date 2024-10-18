from functools import wraps
from pathlib import Path
from typing import Any, Callable

from PyQt5.QtCore import Qt, pyqtSignal, QMetaObject, QUuid, QUrl, QPoint
from PyQt5.QtGui import QFontMetrics, QIcon, QDesktopServices
from PyQt5.QtWidgets import QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QMenu
from PyQt5.QtWidgets import QLabel, QLineEdit, QListWidgetItem, QMessageBox, QSpinBox, QAction
from PyQt5.QtWidgets import QToolButton, QVBoxLayout, QWidget, QSlider, QDoubleSpinBox

from ..custom_workflow import CustomParam, ParamKind, SortedWorkflows, WorkflowSource
from ..custom_workflow import CustomGenerationMode
from ..jobs import JobKind
from ..model import Model, ApplyBehavior
from ..properties import Binding, Bind, bind, bind_combo
from ..style import Styles
from ..root import root
from ..settings import settings
from ..localization import translate as _
from ..util import ensure, clamp, base_type_match
from .generation import GenerateButton, ProgressBar, QueueButton, HistoryWidget, create_error_label
from .live import LivePreviewArea
from .switch import SwitchWidget
from .widget import TextPromptWidget, WorkspaceSelectWidget, StyleSelectWidget
from . import theme


class LayerSelect(QComboBox):
    value_changed = pyqtSignal()

    def __init__(self, filter: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.filter = filter

        self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumContentsLength(20)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLength)
        self.currentIndexChanged.connect(lambda _: self.value_changed.emit())

        self._update()
        root.active_model.layers.changed.connect(self._update)

    def _update(self):
        if self.filter is None:
            layers = root.active_model.layers.all
        elif self.filter == "image":
            layers = root.active_model.layers.images
        elif self.filter == "mask":
            layers = root.active_model.layers.masks
        else:
            assert False, f"Unknown filter: {self.filter}"

        for l in layers:
            index = self.findData(l.id)
            if index == -1:
                self.addItem(l.name, l.id)
            elif self.itemText(index) != l.name:
                self.setItemText(index, l.name)
        i = 0
        while i < self.count():
            if self.itemData(i) not in (l.id for l in layers):
                self.removeItem(i)
            else:
                i += 1

    @property
    def value(self) -> str:
        if self.currentIndex() == -1:
            return ""
        return self.currentData().toString()

    @value.setter
    def value(self, value: str):
        i = self.findData(QUuid(value))
        if i != -1 and i != self.currentIndex():
            self.setCurrentIndex(i)


class IntParamWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, param: CustomParam, parent: QWidget | None = None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        assert param.min is not None and param.max is not None and param.default is not None
        if param.max - param.min <= 200:
            self._widget = QSlider(Qt.Orientation.Horizontal, parent)
            self._widget.setMinimumHeight(self._widget.minimumSizeHint().height() + 4)
            self._widget.valueChanged.connect(self._notify)
            self._label = QLabel(self)
            self._label.setFixedWidth(32)
            self._label.setAlignment(Qt.AlignmentFlag.AlignRight)
            layout.addWidget(self._widget)
            layout.addWidget(self._label)
        else:
            self._widget = QSpinBox(parent)
            self._widget.valueChanged.connect(self._notify)
            self._label = None
            layout.addWidget(self._widget)

        min_range = clamp(int(param.min), -(2**31), 2**31 - 1)
        max_range = clamp(int(param.max), -(2**31), 2**31 - 1)
        self._widget.setRange(min_range, max_range)

        self.value = param.default

    def _notify(self):
        if self._label:
            self._label.setText(str(self._widget.value()))
        self.value_changed.emit()

    @property
    def value(self):
        return self._widget.value()

    @value.setter
    def value(self, value: int | float):
        self._widget.setValue(int(value))


class FloatParamWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, param: CustomParam, parent: QWidget | None = None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        assert param.min is not None and param.max is not None and param.default is not None
        if param.max - param.min <= 100:
            self._widget = QSlider(Qt.Orientation.Horizontal, parent)
            self._widget.setRange(round(param.min * 100), round(param.max * 100))
            self._widget.setMinimumHeight(self._widget.minimumSizeHint().height() + 4)
            self._widget.valueChanged.connect(self._notify)
            self._label = QLabel(self)
            self._label.setFixedWidth(32)
            self._label.setAlignment(Qt.AlignmentFlag.AlignRight)
            layout.addWidget(self._widget)
            layout.addWidget(self._label)
        else:
            self._widget = QDoubleSpinBox(parent)
            self._widget.setRange(param.min, param.max)
            self._widget.valueChanged.connect(self._notify)
            self._label = None
            layout.addWidget(self._widget)

        self.value = param.default

    def _notify(self):
        if self._label:
            self._label.setText(f"{self.value:.2f}")
        self.value_changed.emit()

    @property
    def value(self):
        if isinstance(self._widget, QSlider):
            return self._widget.value() / 100
        else:
            return self._widget.value()

    @value.setter
    def value(self, value: float | int):
        if isinstance(self._widget, QSlider):
            self._widget.setValue(round(value * 100))
        else:
            self._widget.setValue(float(value))


class BoolParamWidget(QWidget):
    value_changed = pyqtSignal()

    _true_text = _("On")
    _false_text = _("Off")

    def __init__(self, param: CustomParam, parent: QWidget | None = None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        fm = QFontMetrics(self.font())
        self._label = QLabel(self)
        self._label.setMinimumWidth(max(fm.width(self._true_text), fm.width(self._false_text)) + 4)
        self._widget = SwitchWidget(parent)
        self._widget.toggled.connect(self._notify)
        layout.addWidget(self._widget)
        layout.addWidget(self._label)

        assert isinstance(param.default, bool)
        self.value = param.default

    def _notify(self):
        self._label.setText(self._true_text if self.value else self._false_text)
        self.value_changed.emit()

    @property
    def value(self):
        return self._widget.isChecked()

    @value.setter
    def value(self, value: bool):
        self._widget.setChecked(value)


class TextParamWidget(QLineEdit):
    value_changed = pyqtSignal()

    def __init__(self, param: CustomParam, parent: QWidget | None = None):
        super().__init__(parent)
        assert isinstance(param.default, str)

        self.value = param.default
        self.textChanged.connect(self._notify)

    def _notify(self):
        self.value_changed.emit()

    @property
    def value(self):
        return self.text()

    @value.setter
    def value(self, value: str):
        self.setText(value)


class PromptParamWidget(TextPromptWidget):
    value_changed = pyqtSignal()

    def __init__(self, param: CustomParam, parent: QWidget | None = None):
        super().__init__(is_negative=param.kind is ParamKind.prompt_negative, parent=parent)
        assert isinstance(param.default, str)

        self.setObjectName("PromptParam")
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"QFrame#PromptParam {{ background-color: {theme.base}; border: 1px solid {theme.line_base}; }}"
        )
        self.text = param.default
        self.text_changed.connect(self.value_changed)

    @property
    def value(self):
        return self.text

    @value.setter
    def value(self, value: str):
        self.text = value


class ChoiceParamWidget(QComboBox):
    value_changed = pyqtSignal()

    def __init__(self, param: CustomParam, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumContentsLength(20)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLength)

        if param.choices:
            self.addItems(param.choices)
        self.currentIndexChanged.connect(lambda _: self.value_changed.emit())

        if param.default is not None:
            self.value = param.default

    @property
    def value(self) -> str:
        if self.currentIndex() == -1:
            return ""
        return self.currentText()

    @value.setter
    def value(self, value: str):
        i = self.findText(value)
        if i != -1 and i != self.currentIndex():
            self.setCurrentIndex(i)


class StyleParamWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._style_select = StyleSelectWidget(self)
        self._style_select.value_changed.connect(self._notify)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._style_select)
        self.setLayout(layout)

    def _notify(self):
        self.value_changed.emit()

    @property
    def value(self):
        return self._style_select.value.filename

    @value.setter
    def value(self, value: str):
        if style := Styles.list().find(value):
            self._style_select.value = style


CustomParamWidget = (
    LayerSelect
    | IntParamWidget
    | FloatParamWidget
    | BoolParamWidget
    | TextParamWidget
    | PromptParamWidget
    | ChoiceParamWidget
    | StyleParamWidget
)


def _create_param_widget(param: CustomParam, parent: QWidget) -> CustomParamWidget:
    match param.kind:
        case ParamKind.image_layer:
            return LayerSelect("image", parent)
        case ParamKind.mask_layer:
            return LayerSelect("mask", parent)
        case ParamKind.number_int:
            return IntParamWidget(param, parent)
        case ParamKind.number_float:
            return FloatParamWidget(param, parent)
        case ParamKind.toggle:
            return BoolParamWidget(param, parent)
        case ParamKind.text:
            return TextParamWidget(param, parent)
        case ParamKind.prompt_positive | ParamKind.prompt_negative:
            return PromptParamWidget(param, parent)
        case ParamKind.choice:
            return ChoiceParamWidget(param, parent)
        case ParamKind.style:
            return StyleParamWidget(parent)
        case _:
            assert False, f"Unknown param kind: {param.kind}"


class WorkflowParamsWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, params: list[CustomParam], parent: QWidget):
        super().__init__(parent)
        self._widgets: dict[str, CustomParamWidget] = {}

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setColumnMinimumWidth(1, 10)
        layout.setColumnStretch(2, 1)
        self.setLayout(layout)

        for p in params:
            label = QLabel(p.name, self)
            widget = _create_param_widget(p, self)
            widget.value_changed.connect(self._notify)
            row = len(self._widgets)
            layout.addWidget(label, row, 0, Qt.AlignmentFlag.AlignBaseline)
            layout.addWidget(widget, row, 2)
            self._widgets[p.name] = widget

    def _notify(self):
        self.value_changed.emit()

    @property
    def value(self):
        return {name: widget.value for name, widget in self._widgets.items()}

    @value.setter
    def value(self, values: dict[str, Any]):
        for name, value in values.items():
            if widget := self._widgets.get(name):
                if base_type_match(widget.value, value):
                    widget.value = value


def popup_on_error(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            QMessageBox.critical(self, _("Error"), str(e))

    return wrapper


def _create_tool_button(parent: QWidget, icon: QIcon, tooltip: str, handler: Callable[..., None]):
    button = QToolButton(parent)
    button.setIcon(icon)
    button.setToolTip(tooltip)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    button.setAutoRaise(True)
    button.clicked.connect(handler)
    return button


class CustomWorkflowWidget(QWidget):
    def __init__(self):
        super().__init__()

        self._model = root.active_model
        self._model_bindings: list[QMetaObject.Connection | Binding] = []

        self._workspace_select = WorkspaceSelectWidget(self)

        self._workflow_select_widgets = QWidget(self)

        self._workflow_select = QComboBox(self._workflow_select_widgets)
        self._workflow_select.setModel(SortedWorkflows(root.workflows))
        self._workflow_select.currentIndexChanged.connect(self._change_workflow)

        self._import_workflow_button = _create_tool_button(
            self._workflow_select_widgets,
            theme.icon("import"),
            _("Import workflow from file"),
            self._import_workflow,
        )
        self._save_workflow_button = _create_tool_button(
            self._workflow_select_widgets,
            theme.icon("save"),
            _("Save workflow to file"),
            self._save_workflow,
        )
        self._delete_workflow_button = _create_tool_button(
            self._workflow_select_widgets,
            theme.icon("discard"),
            _("Delete the currently selected workflow"),
            self._delete_workflow,
        )
        self._open_webui_button = _create_tool_button(
            self._workflow_select_widgets,
            theme.icon("comfyui"),
            _("Open Web UI to create custom workflows"),
            self._open_webui,
        )

        self._workflow_edit_widgets = QWidget(self)
        self._workflow_edit_widgets.setVisible(False)

        self._workflow_name_edit = QLineEdit(self._workflow_edit_widgets)
        self._workflow_name_edit.textEdited.connect(self._edit_name)
        self._workflow_name_edit.returnPressed.connect(self._accept_name)

        self._accept_name_button = _create_tool_button(
            self._workflow_edit_widgets, theme.icon("apply"), _("Apply"), self._accept_name
        )
        self._cancel_name_button = _create_tool_button(
            self._workflow_edit_widgets, theme.icon("cancel"), _("Cancel"), self._cancel_name
        )

        self._params_widget = WorkflowParamsWidget([], self)

        self._generate_button = GenerateButton(JobKind.diffusion, self)
        self._generate_button.clicked.connect(self._generate)

        self._apply_button = QToolButton(self)
        self._apply_button.setIcon(theme.icon("apply"))
        self._apply_button.setFixedHeight(self._generate_button.height() - 2)
        self._apply_button.setToolTip(_("Create a new layer with the current result"))
        self._apply_button.clicked.connect(self.apply_live_result)

        self._mode_button = QToolButton(self)
        self._mode_button.setArrowType(Qt.ArrowType.DownArrow)
        self._mode_button.setFixedHeight(self._generate_button.height() - 2)
        self._mode_button.clicked.connect(self._show_generate_menu)
        menu = QMenu(self)
        menu.addAction(self._mk_action(CustomGenerationMode.regular, _("Generate"), "generate"))
        menu.addAction(
            self._mk_action(CustomGenerationMode.live, _("Generate Live"), "workspace-live")
        )
        self._generate_menu = menu

        self._queue_button = QueueButton(parent=self)
        self._queue_button.setFixedHeight(self._generate_button.height() - 2)

        self._progress_bar = ProgressBar(self)
        self._error_text = create_error_label(self)

        self._history = HistoryWidget(self)
        self._history.item_activated.connect(self.apply_result)

        self._live_preview = LivePreviewArea(self)

        self._layout = QVBoxLayout()
        select_layout = QHBoxLayout()
        select_layout.setContentsMargins(0, 0, 0, 0)
        select_layout.setSpacing(2)
        select_layout.addWidget(self._workflow_select)
        select_layout.addWidget(self._import_workflow_button)
        select_layout.addWidget(self._save_workflow_button)
        select_layout.addWidget(self._delete_workflow_button)
        select_layout.addWidget(self._open_webui_button)
        self._workflow_select_widgets.setLayout(select_layout)
        edit_layout = QHBoxLayout()
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.setSpacing(2)
        edit_layout.addWidget(self._workflow_name_edit)
        edit_layout.addWidget(self._accept_name_button)
        edit_layout.addWidget(self._cancel_name_button)
        self._workflow_edit_widgets.setLayout(edit_layout)
        header_layout = QHBoxLayout()
        header_layout.addWidget(self._workspace_select)
        header_layout.addWidget(self._workflow_select_widgets)
        header_layout.addWidget(self._workflow_edit_widgets)
        self._layout.addLayout(header_layout)
        self._layout.addWidget(self._params_widget)
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(0)
        actions_layout.addWidget(self._generate_button)
        actions_layout.addWidget(self._apply_button)
        actions_layout.addWidget(self._mode_button)
        actions_layout.addSpacing(4)
        actions_layout.addWidget(self._queue_button)
        self._layout.addLayout(actions_layout)
        self._layout.addWidget(self._progress_bar)
        self._layout.addWidget(self._error_text)
        self._layout.addWidget(self._history)
        self._layout.addWidget(self._live_preview)
        self.setLayout(self._layout)

        self._update_ui()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            self._model_bindings = [
                bind(model, "workspace", self._workspace_select, "value", Bind.one_way),
                bind_combo(model.custom, "workflow_id", self._workflow_select, Bind.one_way),
                model.workspace_changed.connect(self._cancel_name),
                model.custom.graph_changed.connect(self._update_current_workflow),
                model.error_changed.connect(self._error_text.setText),
                model.has_error_changed.connect(self._error_text.setVisible),
                model.custom.mode_changed.connect(self._update_ui),
                model.custom.is_live_changed.connect(self._update_ui),
                model.custom.result_available.connect(self._live_preview.show_image),
                model.custom.has_result_changed.connect(self._apply_button.setEnabled),
            ]
            self._queue_button.model = model
            self._progress_bar.model = model
            self._history.model_ = model
            self._update_current_workflow()
            self._update_ui()

    def _mk_action(self, mode: CustomGenerationMode, text: str, icon: str):
        action = QAction(text, self)
        action.setIcon(theme.icon(icon))
        action.setIconVisibleInMenu(True)
        action.triggered.connect(lambda: self._change_mode(mode))
        return action

    def _change_mode(self, mode: CustomGenerationMode):
        self.model.custom.mode = mode

    def _show_generate_menu(self):
        width = self._generate_button.width() + self._mode_button.width()
        pos = QPoint(0, self._generate_button.height())
        self._generate_menu.setFixedWidth(width)
        self._generate_menu.exec_(self._generate_button.mapToGlobal(pos))

    def _update_ui(self):
        is_live_mode = self.model.custom.mode is CustomGenerationMode.live
        self._history.setVisible(not is_live_mode)
        self._live_preview.setVisible(is_live_mode)
        self._apply_button.setVisible(is_live_mode)
        self._apply_button.setEnabled(self.model.custom.has_result)

        if not is_live_mode:
            text = _("Generate")
            icon = "generate"
        elif not self.model.custom.is_live:
            text = _("Start Generating")
            icon = "play"
        else:
            text = _("Stop Generating")
            icon = "pause"
        self._generate_button.operation = text
        self._generate_button.setIcon(theme.icon(icon))

    def _generate(self):
        if self.model.custom.mode is CustomGenerationMode.regular:
            self.model.custom.generate()
        else:
            self.model.custom.is_live = not self.model.custom.is_live

    def _update_current_workflow(self):
        if not self.model.custom.workflow:
            self._save_workflow_button.setEnabled(False)
            self._delete_workflow_button.setEnabled(False)
            return
        self._save_workflow_button.setEnabled(True)
        self._delete_workflow_button.setEnabled(
            self.model.custom.workflow.source is WorkflowSource.local
        )

        self._params_widget.deleteLater()
        self._params_widget = WorkflowParamsWidget(self.model.custom.metadata, self)
        self._params_widget.value = self.model.custom.params
        self._layout.insertWidget(1, self._params_widget)
        self._params_widget.value_changed.connect(self._change_params)

    def _change_workflow(self):
        self.model.custom.workflow_id = self._workflow_select.currentData()

    def _change_params(self):
        self.model.custom.params = self._params_widget.value

    def apply_result(self, item: QListWidgetItem):
        job_id, index = self._history.item_info(item)
        self.model.apply_generated_result(job_id, index)

    def apply_live_result(self):
        image, params = self.model.custom.live_result
        self.model.apply_result(image, params, ApplyBehavior.layer)
        if settings.new_seed_after_apply:
            self.model.generate_seed()

    @popup_on_error
    def _import_workflow(self, *args):
        filename, __ = QFileDialog.getOpenFileName(
            self,
            _("Import Workflow"),
            str(Path.home()),
            "Workflow Files (*.json);;All Files (*)",
        )
        if filename:
            self.model.custom.import_file(Path(filename))

    def _save_workflow(self):
        self.is_edit_mode = True

    def _delete_workflow(self):
        filepath = ensure(self.model.custom.workflow).path
        q = QMessageBox.question(
            self,
            _("Delete Workflow"),
            _("Are you sure you want to delete the current workflow?") + f"\n{filepath}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.StandardButton.No,
        )
        if q == QMessageBox.StandardButton.Yes:
            self.model.custom.remove_workflow()

    def _open_webui(self):
        if client := root.connection.client_if_connected:
            QDesktopServices.openUrl(QUrl(client.url))

    @property
    def is_edit_mode(self):
        return self._workflow_edit_widgets.isVisible()

    @is_edit_mode.setter
    def is_edit_mode(self, value: bool):
        if value == self.is_edit_mode:
            return
        self._workflow_select_widgets.setVisible(not value)
        self._workflow_edit_widgets.setVisible(value)
        if value:
            self._workflow_name_edit.setText(self.model.custom.workflow_id)
            self._workflow_name_edit.selectAll()
            self._workflow_name_edit.setFocus()

    def _edit_name(self):
        self._accept_name_button.setEnabled(self._workflow_name_edit.text().strip() != "")

    @popup_on_error
    def _accept_name(self, *args):
        self.model.custom.save_as(self._workflow_name_edit.text())
        self.is_edit_mode = False

    def _cancel_name(self):
        self.is_edit_mode = False


class CustomWorkflowPlaceholder(QWidget):
    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._connections = []

        self._workspace_select = WorkspaceSelectWidget(self)
        note = QLabel("<i>" + _("Custom workflows are not available on Cloud.") + "</i>", self)

        self._layout = QVBoxLayout()
        self._layout.addWidget(self._workspace_select)
        self._layout.addSpacing(50)
        self._layout.addWidget(note, 0, Qt.AlignmentFlag.AlignCenter)
        self._layout.addStretch()
        self.setLayout(self._layout)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model != model:
            Binding.disconnect_all(self._connections)
            self._model = model
            self._connections = [
                bind(model, "workspace", self._workspace_select, "value", Bind.one_way)
            ]
