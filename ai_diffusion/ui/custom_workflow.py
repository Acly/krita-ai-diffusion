from functools import wraps
from pathlib import Path
from typing import Any, Callable

from krita import Krita
from PyQt5.QtCore import Qt, pyqtSignal, QMetaObject, QUuid, QUrl, QPoint, QSize
from PyQt5.QtGui import QFontMetrics, QIcon, QDesktopServices, QPalette
from PyQt5.QtWidgets import QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QMenu
from PyQt5.QtWidgets import QLabel, QLineEdit, QListWidgetItem, QMessageBox, QSpinBox, QAction
from PyQt5.QtWidgets import QToolButton, QVBoxLayout, QWidget, QSlider, QDoubleSpinBox
from PyQt5.QtWidgets import QScrollArea, QTextEdit, QSplitter

from ..custom_workflow import CustomParam, ParamKind, SortedWorkflows, WorkflowSource
from ..custom_workflow import CustomGenerationMode
from ..client import TextOutput
from ..jobs import JobKind
from ..model import Model
from ..properties import Binding, Bind, bind, bind_combo
from ..style import Styles
from ..root import root
from ..settings import settings
from ..localization import translate as _
from ..util import ensure, clamp, base_type_match
from .generation import GenerateButton, ProgressBar, QueueButton, HistoryWidget
from .live import LivePreviewArea
from .switch import SwitchWidget
from .widget import TextPromptWidget, WorkspaceSelectWidget, StyleSelectWidget, ErrorBox
from .settings_widgets import ExpanderButton
from . import theme
from .theme import SignalBlocker


class LayerSelect(QComboBox):
    value_changed = pyqtSignal()

    def __init__(self, filter: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.param = None
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
        self.param = param
        self.setContentsMargins(0, 0, 0, 0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._slider: QSlider | None = None
        assert param.min is not None and param.max is not None and param.default is not None
        if param.max - param.min <= 200:
            self._slider = QSlider(Qt.Orientation.Horizontal, parent)
            self._slider.setMinimumHeight(self._slider.minimumSizeHint().height() + 4)
            self._slider.valueChanged.connect(self._slider_changed)
            self._widget = QSpinBox(parent)
            self._widget.valueChanged.connect(self._input_changed)
            layout.addWidget(self._slider)
            layout.addWidget(self._widget)
        else:
            self._widget = QSpinBox(parent)
            self._widget.valueChanged.connect(self._notify)
            layout.addWidget(self._widget)

        min_range = clamp(int(param.min), -(2**31), 2**31 - 1)
        max_range = clamp(int(param.max), -(2**31), 2**31 - 1)
        self._widget.setRange(min_range, max_range)
        if self._slider is not None:
            self._slider.setRange(min_range, max_range)

        self.value = param.default

    def _slider_changed(self, value: int):
        with SignalBlocker(self._widget):
            self._widget.setValue(value)
        self._notify()

    def _input_changed(self, value: int):
        if self._slider is not None:
            with SignalBlocker(self._slider):
                self._slider.setValue(value)
        self._notify()

    def _notify(self):
        self.value_changed.emit()

    @property
    def value(self):
        return self._widget.value()

    @value.setter
    def value(self, value: int | float):
        v = int(value)
        v = max(self._widget.minimum(), min(self._widget.maximum(), v))
        with SignalBlocker(self._widget):
            self._widget.setValue(v)
        if self._slider is not None:
            with SignalBlocker(self._slider):
                self._slider.setValue(v)


class FloatParamWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, param: CustomParam, parent: QWidget | None = None):
        super().__init__(parent)
        self.param = param
        self.setContentsMargins(0, 0, 0, 0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._slider: QSlider | None = None
        assert param.min is not None and param.max is not None and param.default is not None
        if param.max - param.min <= 100:
            self._slider = QSlider(Qt.Orientation.Horizontal, parent)
            self._slider.setRange(round(param.min * 100), round(param.max * 100))
            self._slider.setMinimumHeight(self._slider.minimumSizeHint().height() + 4)
            self._slider.valueChanged.connect(self._slider_changed)
            self._widget = QDoubleSpinBox(parent)
            self._widget.setRange(param.min, param.max)
            self._widget.setDecimals(2)
            self._widget.valueChanged.connect(self._input_changed)
            layout.addWidget(self._slider)
            layout.addWidget(self._widget)
        else:
            self._widget = QDoubleSpinBox(parent)
            self._widget.setRange(param.min, param.max)
            self._widget.valueChanged.connect(self._notify)
            layout.addWidget(self._widget)

        self.value = param.default

    def _slider_changed(self, value: int):
        v = value / 100.0
        with SignalBlocker(self._widget):
            self._widget.setValue(v)
        self._notify()

    def _input_changed(self, value: float):
        if self._slider is not None:
            with SignalBlocker(self._slider):
                self._slider.setValue(round(value * 100))
        self._notify()

    def _notify(self):
        self.value_changed.emit()

    @property
    def value(self):
        return float(self._widget.value())

    @value.setter
    def value(self, value: float | int):
        v = float(value)
        v = max(self._widget.minimum(), min(self._widget.maximum(), v))
        with SignalBlocker(self._widget):
            self._widget.setValue(v)
        if self._slider is not None:
            with SignalBlocker(self._slider):
                self._slider.setValue(round(v * 100))


class BoolParamWidget(QWidget):
    value_changed = pyqtSignal()

    _true_text = _("On")
    _false_text = _("Off")

    def __init__(self, param: CustomParam, parent: QWidget | None = None):
        super().__init__(parent)
        self.param = param
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
        self.param = param

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
        line_count = settings.prompt_line_count if param.kind is ParamKind.prompt_positive else 2
        super().__init__(
            is_negative=param.kind is ParamKind.prompt_negative,
            line_count=line_count,
            parent=parent,
        )
        assert isinstance(param.default, str)
        self.param = param

        base = self.palette().color(QPalette.ColorRole.Base)
        if param.kind is ParamKind.prompt_negative:
            base.setRed(base.red() + 15)
        self.setObjectName("PromptParam")
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"QFrame#PromptParam {{ background-color: {base.name()}; border: 1px solid {theme.line_base}; }}"
        )
        self.text = param.default
        self.text_changed.connect(self.value_changed)
        settings.changed.connect(self.update_settings)

        self.is_resizable = True
        self.handle_dragged.connect(self._handle_dragging)

    def sizeHint(self):
        fm = QFontMetrics(ensure(self.document()).defaultFont())
        return QSize(200, fm.lineSpacing() * self.line_count + 8)

    @property
    def value(self):
        return self.text

    @value.setter
    def value(self, value: str):
        self.text = value

    def update_settings(self, key: str, value):
        if key == "prompt_line_count" and self.param.kind is ParamKind.prompt_positive:
            self.line_count = value

    def _handle_dragging(self, y_pos: int):
        fm = QFontMetrics(ensure(self.document()).defaultFont())
        new_line_count = round((y_pos - 5) / fm.lineSpacing())
        if 1 <= new_line_count <= 10:
            settings.prompt_line_count = new_line_count
            self.line_count = new_line_count


class ChoiceParamWidget(QComboBox):
    value_changed = pyqtSignal()

    def __init__(self, param: CustomParam, parent: QWidget | None = None):
        super().__init__(parent)
        self.param = param
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
        self.param = None
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
        if value != self.value:
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


def _create_param_widget(param: CustomParam, parent: "WorkflowParamsWidget") -> CustomParamWidget:
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
            w = PromptParamWidget(param, parent)
            w.activated.connect(parent.activated)
            return w
        case ParamKind.choice:
            return ChoiceParamWidget(param, parent)
        case ParamKind.style:
            return StyleParamWidget(parent)
        case _:
            assert False, f"Unknown param kind: {param.kind}"


class GroupHeader(QWidget):
    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._group_widgets: list[CustomParamWidget] = []

        self._expander = ExpanderButton(text, self)
        self._expander.toggled.connect(self._show_group)

        fh = self.fontMetrics().height()
        self._reset_button = QToolButton(self)
        self._reset_button.setFixedSize(fh + 2, fh + 2)
        self._reset_button.setIcon(theme.icon("reset"))
        self._reset_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._reset_button.setAutoRaise(True)
        self._reset_button.setToolTip(_("Reset all parameters in this group"))
        self._reset_button.clicked.connect(self._reset_group)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._expander, stretch=1, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._reset_button, alignment=Qt.AlignmentFlag.AlignRight)

    def set_group_widgets(self, widgets: list[CustomParamWidget], show_group: bool):
        self._group_widgets = widgets
        self._expander.setChecked(show_group)
        self._show_group(show_group)

    def _show_group(self, checked: bool):
        for w in self._group_widgets:
            w.setVisible(checked)
        self._reset_button.setVisible(checked)

    def _reset_group(self):
        for w in self._group_widgets:
            if not isinstance(w, QLabel) and w.param is not None and w.param.default is not None:
                w.value = w.param.default


class WorkflowParamsWidget(QWidget):
    value_changed = pyqtSignal()
    activated = pyqtSignal()

    def __init__(self, params: list[CustomParam], parent: QWidget | None = None):
        super().__init__(parent)
        self._widgets: dict[str, CustomParamWidget] = {}
        self._max_group_height = 0

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 2, 0)
        layout.setColumnMinimumWidth(0, 10)
        layout.setColumnMinimumWidth(2, 10)
        layout.setColumnStretch(3, 1)
        self.setLayout(layout)

        params = sorted(params)
        current_group: tuple[str, GroupHeader | None, list[CustomParamWidget]] = ("", None, [])

        for p in params:
            group, expander, group_widgets = current_group
            if p.group != group:
                self._create_group(expander, group_widgets)
                expander = GroupHeader(p.group, self)
                group_widgets = []
                current_group = (p.group, expander, group_widgets)
                layout.addWidget(expander, layout.rowCount(), 0, 1, 4)
            label = QLabel(p.display_name, self)
            widget = _create_param_widget(p, self)
            widget.value_changed.connect(self._notify)
            row = layout.rowCount()
            col, col_span = (0, 2) if p.group == "" else (1, 1)
            layout.addWidget(label, row, col, 1, col_span, Qt.AlignmentFlag.AlignBaseline)
            layout.addWidget(widget, row, 3)
            self._widgets[p.name] = widget
            group_widgets.extend((label, widget))

        self._create_group(current_group[1], current_group[2])
        layout.setRowStretch(layout.rowCount(), 1)

    def _notify(self):
        self.value_changed.emit()

    def _create_group(self, expander: GroupHeader | None, widgets: list[CustomParamWidget]):
        display_height = sum(w.sizeHint().height() for w in widgets if not isinstance(w, QLabel))
        display_height += 2 * len(widgets)  # spacing
        if expander is not None:
            expander.set_group_widgets(widgets, show_group=len(self._widgets) < 7)
            display_height += expander.sizeHint().height()
        self._max_group_height = max(self._max_group_height, display_height + 4)

    @property
    def value(self):
        return {name: widget.value for name, widget in self._widgets.items()}

    @value.setter
    def value(self, values: dict[str, Any]):
        for name, value in values.items():
            if widget := self._widgets.get(name):
                if base_type_match(widget.value, value):
                    widget.value = value

    @property
    def min_size(self):
        return self._max_group_height


class WorkflowOutputsWidget(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._value: dict[str, TextOutput] = {}

        self._scroll_area = QScrollArea(self)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.expander = ExpanderButton(_("Text Output"), self)
        self.expander.setStyleSheet("QToolButton { border: none; }")
        self.expander.setChecked(True)
        self.expander.toggled.connect(self._scroll_area.setVisible)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.expander)
        layout.addWidget(self._scroll_area)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: dict[str, TextOutput]):
        self._value = value
        self._update()

    @property
    def is_visible(self):
        return self._scroll_area.isVisible()

    def _update(self):
        if len(self._value) == 0:
            self.expander.hide()
            self._scroll_area.hide()
            return
        elif not self.expander.isVisible():
            self.expander.show()
            self._scroll_area.setVisible(self.expander.isChecked())

        widget = QWidget(self._scroll_area)
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setColumnMinimumWidth(1, 8)
        layout.setColumnStretch(2, 1)
        widget.setLayout(layout)

        line = 0
        text_areas: list[QTextEdit] = []
        for output in self._value.values():
            label = QLabel(output.name, widget)
            if (not output.mime or output.mime == "text/plain") and len(output.text) < 40:
                value = QLabel(output.text, widget)
                value.setWordWrap(True)
                value.setMinimumWidth(40)
                layout.addWidget(label, line, 0)
                layout.addWidget(value, line, 2)
                line += 1
            else:
                value = QTextEdit(widget)
                value.setFrameShape(QFrame.Shape.StyledPanel)
                value.setStyleSheet(
                    "QTextEdit { background: transparent; border-left: 1px solid %s; padding-left: 2px; }"
                    % theme.line
                )
                value.setReadOnly(True)
                match output.mime:
                    case "" | "text/plain":
                        value.setPlainText(output.text)
                    case "text/html":
                        value.setHtml(output.text)
                    case "text/markdown":
                        value.setMarkdown(output.text)
                layout.addWidget(label, line, 0, 1, 3)
                layout.addWidget(value, line + 1, 0, 1, 3)
                text_areas.append(value)
                line += 2

        layout.setRowStretch(line, 1)
        widget.setFixedWidth(self._scroll_area.width() - 8)
        self._scroll_area.setWidget(widget)
        if self.expander.isChecked():
            widget.show()

        for w in text_areas:
            size = ensure(w.document()).size().toSize()
            w.setFixedHeight(max(size.height() + 2, self.fontMetrics().height() + 6))
        widget.adjustSize()


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
        self._open_settings_button = _create_tool_button(
            self._workflow_select_widgets,
            theme.icon("settings"),
            _("Open settings"),
            self._show_settings,
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

        self._params_widget: WorkflowParamsWidget | None = None
        self._params_scroll = QScrollArea(self)
        self._params_scroll.setWidgetResizable(True)
        self._params_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._bottom = QWidget(self)

        self._generate_button = GenerateButton(JobKind.diffusion, self._bottom)
        self._generate_button.clicked.connect(self._generate)

        self._apply_button = QToolButton(self._bottom)
        self._apply_button.setIcon(theme.icon("apply"))
        self._apply_button.setFixedHeight(self._generate_button.minimumSizeHint().height() - 3)
        self._apply_button.setToolTip(_("Create a new layer with the current result"))
        self._apply_button.clicked.connect(self.apply_live_result)

        self._mode_button = QToolButton(self._bottom)
        self._mode_button.setArrowType(Qt.ArrowType.DownArrow)
        self._mode_button.setFixedHeight(self._generate_button.minimumSizeHint().height() - 3)
        self._mode_button.clicked.connect(self._show_generate_menu)
        menu = QMenu(self)
        menu.addAction(self._mk_action(CustomGenerationMode.regular, _("Generate"), "generate"))
        menu.addAction(
            self._mk_action(CustomGenerationMode.live, _("Generate Live"), "workspace-live")
        )
        menu.addAction(
            self._mk_action(
                CustomGenerationMode.animation, _("Generate Animation"), "workspace-animation"
            )
        )
        self._generate_menu = menu

        self._queue_button = QueueButton(parent=self._bottom)
        self._queue_button.setFixedHeight(self._generate_button.height() - 2)

        self._progress_bar = ProgressBar(self._bottom)
        self._error_box = ErrorBox(self._bottom)

        self._outputs = WorkflowOutputsWidget(self._bottom)
        self._outputs.expander.toggled.connect(self._update_layout)

        self._history = HistoryWidget(self._bottom)
        self._history.item_activated.connect(self.apply_result)

        self._live_preview = LivePreviewArea(self._bottom)

        self._splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._splitter.addWidget(self._params_scroll)
        self._splitter.addWidget(self._bottom)
        self._splitter.setCollapsible(1, False)
        self._splitter.splitterMoved.connect(self._splitter_moved)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 2, 2, 0)
        select_layout = QHBoxLayout()
        select_layout.setContentsMargins(0, 0, 0, 0)
        select_layout.setSpacing(2)
        select_layout.addWidget(self._workflow_select)
        select_layout.addWidget(self._import_workflow_button)
        select_layout.addWidget(self._save_workflow_button)
        select_layout.addWidget(self._delete_workflow_button)
        select_layout.addWidget(self._open_webui_button)
        select_layout.addWidget(self._open_settings_button)
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
        layout.addLayout(header_layout)
        layout.addWidget(self._splitter)
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(0)
        actions_layout.addWidget(self._generate_button)
        actions_layout.addWidget(self._apply_button)
        actions_layout.addWidget(self._mode_button)
        actions_layout.addSpacing(4)
        actions_layout.addWidget(self._queue_button)
        self._bottom_layout = QVBoxLayout(self._bottom)
        self._bottom_layout.addLayout(actions_layout)
        self._bottom_layout.addWidget(self._progress_bar)
        self._bottom_layout.addWidget(self._error_box)
        self._bottom_layout.addWidget(self._outputs, stretch=0)
        self._bottom_layout.addWidget(self._history, stretch=3)
        self._bottom_layout.addWidget(self._live_preview, stretch=5)
        self.setLayout(layout)

        self._update_ui()

    def _update_layout(self):
        stretch = 1 if self._outputs.is_visible else 0
        self._bottom_layout.setStretchFactor(self._outputs, stretch)

    def _show_settings(self):
        Krita.instance().action("ai_diffusion_settings").trigger()

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
                bind(model, "error", self._error_box, "error", Bind.one_way),
                bind_combo(model.custom, "workflow_id", self._workflow_select, Bind.one_way),
                bind(model.custom, "outputs", self._outputs, "value", Bind.one_way),
                model.custom.outputs_changed.connect(self._update_layout),
                model.workspace_changed.connect(self._cancel_name),
                model.custom.graph_changed.connect(self._update_current_workflow),
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
            self._set_params_height(model.custom.params_ui_height)

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

        if self.model.custom.mode is CustomGenerationMode.regular:
            text = _("Generate")
            icon = "generate"
        elif self.model.custom.mode is CustomGenerationMode.animation:
            text = _("Generate Animation")
            icon = "workspace-animation"
        elif not self.model.custom.is_live:
            text = _("Start Generating")
            icon = "play"
        else:
            text = _("Stop Generating")
            icon = "pause"
        self._generate_button.operation = text
        self._generate_button.setIcon(theme.icon(icon))

    def _generate(self):
        if self.model.custom.mode is CustomGenerationMode.live:
            self.model.custom.is_live = not self.model.custom.is_live
        else:
            self.model.custom.generate()

    def _update_current_workflow(self):
        if not self.model.custom.workflow:
            self._save_workflow_button.setEnabled(False)
            self._delete_workflow_button.setEnabled(False)
            return
        self._save_workflow_button.setEnabled(True)
        self._delete_workflow_button.setEnabled(
            self.model.custom.workflow.source is WorkflowSource.local
        )

        if self._params_widget:
            self._params_scroll.setWidget(None)
            self._params_widget.deleteLater()
            self._params_widget = None
        if len(self.model.custom.metadata) > 0:
            self._params_widget = WorkflowParamsWidget(self.model.custom.metadata, self)
            self._params_widget.value = self.model.custom.params  # set default values from model
            self.model.custom.params = self._params_widget.value  # set default values from widgets
            self._params_widget.value_changed.connect(self._change_params)
            self._params_widget.activated.connect(self._generate)

            self._params_scroll.setWidget(self._params_widget)
            params_size = min(self.height() // 2, self._params_widget.min_size)
            self._set_params_height(max(self.model.custom.params_ui_height, params_size))
        else:
            self._set_params_height(0)

    def _change_workflow(self):
        self.model.custom.workflow_id = self._workflow_select.currentData()

    def _change_params(self):
        if self._params_widget:
            self.model.custom.params = self._params_widget.value

    def _set_params_height(self, height: int):
        self._splitter.setSizes([height, self._splitter.height() - height])

    def _splitter_moved(self, pos: int, index: int):
        self.model.custom.params_ui_height = self._splitter.sizes()[0]

    def apply_result(self, item: QListWidgetItem):
        job_id, index = self._history.item_info(item)
        self.model.apply_generated_result(job_id, index)

    def apply_live_result(self):
        image, params = self.model.custom.live_result
        self.model.apply_result(image, params)
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
            self.model.custom.switch_to_web_workflow()

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
        name = self._workflow_name_edit.text().strip()
        workspace = self.model.custom
        overwrite = False

        current = workspace.workflow
        existing = workspace.workflows.find(name)
        if (
            current is not None
            and current.source is WorkflowSource.remote
            and existing is not None
            and existing.source is WorkflowSource.local
        ):
            details = f"\n{existing.path}" if existing.path is not None else ""
            q = QMessageBox.question(
                self,
                _("Overwrite Workflow"),
                _("A workflow named '{name}' already exists. Do you want to overwrite it?").format(
                    name=name
                )
                + details,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.StandardButton.No,
            )
            if q == QMessageBox.StandardButton.Yes:
                overwrite = True

        workspace.save_as(name, overwrite=overwrite)
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
        self._layout.setContentsMargins(0, 2, 2, 0)
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
