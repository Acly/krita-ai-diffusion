from __future__ import annotations
from typing import Callable, Optional

from PyQt5.QtWidgets import (
    QAction,
    QSlider,
    QWidget,
    QPlainTextEdit,
    QLabel,
    QLineEdit,
    QMenu,
    QSpinBox,
    QDoubleSpinBox,
    QToolButton,
    QComboBox,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
)
from PyQt5.QtGui import QColor, QFontMetrics, QKeyEvent, QPalette, QTextCursor
from PyQt5.QtCore import Qt, QMetaObject, QSize, pyqtSignal
import krita

from ..style import Style, Styles
from ..resources import ControlMode
from ..root import root
from ..client import filter_supported_styles, resolve_sd_version
from ..properties import Binding, bind, bind_combo
from ..jobs import JobState, JobQueue
from ..model import Model, Workspace, ControlLayer
from ..attention_edit import edit_attention, select_on_cursor_pos
from ..util import ensure
from .settings import SettingsDialog
from .theme import SignalBlocker
from . import actions, theme


class QueueWidget(QToolButton):
    _style = """
        QToolButton {{ border: none; border-radius: 6px; background-color: {color}; color: white; }}
        QToolButton::menu-indicator {{ width: 0px; }}"""

    _jobs: JobQueue

    def __init__(self, parent):
        super().__init__(parent)
        self._jobs = JobQueue()
        self._jobs.count_changed.connect(self._update)

        queue_menu = QMenu(self)
        queue_menu.addAction(self._create_action("Cancel active", actions.cancel_active))
        queue_menu.addAction(self._create_action("Cancel queued", actions.cancel_queued))
        queue_menu.addAction(self._create_action("Cancel all", actions.cancel_all))
        self.setMenu(queue_menu)

        self.setStyleSheet(self._style.format(color=theme.background_inactive))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setArrowType(Qt.ArrowType.NoArrow)
        self._update()

    @property
    def jobs(self):
        return self._jobs

    @jobs.setter
    def jobs(self, jobs: JobQueue):
        self._jobs.count_changed.disconnect(self._update)
        self._jobs = jobs
        self._jobs.count_changed.connect(self._update)

    def _update(self):
        count = self._jobs.count(JobState.queued)
        if self._jobs.any_executing():
            self.setStyleSheet(self._style.format(color=theme.background_active))
            if count > 0:
                self.setToolTip(f"Generating image. {count} jobs queued - click to cancel.")
            else:
                self.setToolTip(f"Generating image. Click to cancel.")
        else:
            self.setStyleSheet(self._style.format(color=theme.background_inactive))
            self.setToolTip("Idle.")
        self.setText(f"+{count} ")

    def _create_action(self, name: str, func: Callable[[], None]):
        action = QAction(name, self)
        action.triggered.connect(func)
        return action


class ControlWidget(QWidget):
    _model: Model
    _control: ControlLayer

    def __init__(self, model: Model, control: ControlLayer, parent: ControlListWidget):
        super().__init__(parent)
        self._model = model
        self._control = control

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.mode_select = QComboBox(self)
        self.mode_select.setStyleSheet(
            "QComboBox { border:none; background-color:transparent; padding: 1px 12px 1px 2px;}"
        )
        for mode in (m for m in ControlMode if m is not ControlMode.inpaint):
            icon = theme.icon(f"control-{mode.name}")
            self.mode_select.addItem(icon, mode.text, mode)
        bind_combo(control, "mode", self.mode_select)

        self.layer_select = QComboBox(self)
        self.layer_select.setMinimumContentsLength(20)
        self.layer_select.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLength
        )
        self._update_layers()
        bind_combo(control, "layer_id", self.layer_select)
        self._model.image_layers.changed.connect(self._update_layers)
        control.has_active_job_changed.connect(lambda x: self.layer_select.setEnabled(not x))

        self.generate_button = QToolButton(self)
        self.generate_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.generate_button.setIcon(theme.icon("control-generate"))
        self.generate_button.setToolTip("Generate control layer from current image")
        self.generate_button.clicked.connect(control.generate)
        control.has_active_job_changed.connect(lambda x: self.generate_button.setEnabled(not x))

        self.add_pose_button = QToolButton(self)
        self.add_pose_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.add_pose_button.setIcon(theme.icon("add-pose"))
        self.add_pose_button.setToolTip("Add new character pose to selected layer")
        self.add_pose_button.clicked.connect(self._add_pose_character)

        self.strength_spin = QSpinBox(self)
        self.strength_spin.setRange(0, 100)
        self.strength_spin.setValue(int(control.strength * 100))
        self.strength_spin.setSuffix("%")
        self.strength_spin.setSingleStep(10)
        self.strength_spin.setToolTip("Control strength")
        bind(control, "strength", self.strength_spin, "value")

        self.end_spin = QDoubleSpinBox(self)
        self.end_spin.setRange(0.0, 1.0)
        self.end_spin.setValue(control.end)
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setToolTip("Control ending step ratio")
        bind(control, "end", self.end_spin, "value")

        self.error_text = QLabel(self)
        self.error_text.setStyleSheet(f"color: {theme.red};")
        self.error_text.setVisible(not control.is_supported)
        control.error_text_changed.connect(self._set_error)
        self._set_error(control.error_text)

        self.remove_button = QToolButton(self)
        self.remove_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.remove_button.setIcon(theme.icon("remove"))
        self.remove_button.setToolTip("Remove control layer")
        button_height = self.remove_button.iconSize().height()
        self.remove_button.setIconSize(QSize(int(button_height * 1.25), button_height))
        self.remove_button.setAutoRaise(True)
        self.remove_button.clicked.connect(self.remove)

        layout.addWidget(self.mode_select)
        layout.addWidget(self.layer_select, 1)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.add_pose_button)
        layout.addWidget(self.strength_spin)
        layout.addWidget(self.end_spin)
        layout.addWidget(self.error_text, 1)
        layout.addWidget(self.remove_button)

        control.is_supported_changed.connect(self._update_visibility)
        control.can_generate_changed.connect(self._update_visibility)
        control.show_end_changed.connect(self._update_visibility)
        control.is_pose_vector_changed.connect(self._update_visibility)
        self._update_visibility()

    def _update_layers(self):
        layers: reversed[krita.Node] = reversed(self._model.image_layers)
        with SignalBlocker(self.layer_select):
            self.layer_select.clear()
            index = -1
            for layer in layers:
                self.layer_select.addItem(layer.name(), layer.uniqueId())
                if layer.uniqueId() == self._control.layer_id:
                    index = self.layer_select.count() - 1
            if index == -1 and self._control in self._model.control:
                self.remove()
            else:
                self.layer_select.setCurrentIndex(index)

    def remove(self):
        self._model.control.remove(self._control)

    def _add_pose_character(self):
        self._model.document.add_pose_character(self._control.layer)

    def _update_visibility(self):
        def controls():
            self.layer_select.setVisible(self._control.is_supported)
            self.generate_button.setVisible(self._control.can_generate)
            self.add_pose_button.setVisible(self._control.is_pose_vector)
            self.strength_spin.setVisible(self._control.is_supported)
            self.end_spin.setVisible(self._control.show_end)

        def error():
            self.error_text.setVisible(not self._control.is_supported)

        if self._control.is_supported:
            error()
            controls()
        else:  # always hide things to hide first to make space in the layout
            controls()
            error()

    def _set_error(self, error: str):
        parts = error.split("[", 2)
        self.error_text.setText(parts[0])
        if len(parts) > 1:
            self.error_text.setToolTip(f"Missing one of the following models: {parts[1][:-1]}")


class ControlListWidget(QWidget):
    _controls: list[ControlWidget]
    _model: Model
    _model_connections: list[QMetaObject.Connection]

    changed = pyqtSignal()

    def __init__(self, model: Model, parent=None):
        super().__init__(parent)
        self._model = model
        self._controls = []
        self._model_connections = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model != model:
            Binding.disconnect_all(self._model_connections)
            self._model = model
            while len(self._controls) > 0:
                self._remove_widget(self._controls[0])
            for control in self._model.control:
                self._add_widget(control)
            self._model_connections = [
                model.control.added.connect(self._add_widget),
                model.control.removed.connect(self._remove_widget),
            ]

    def _add_widget(self, control: ControlLayer):
        widget = ControlWidget(self._model, control, self)
        self._controls.append(widget)
        self._layout.addWidget(widget)

    def _remove_widget(self, widget: ControlWidget | ControlLayer):
        if isinstance(widget, ControlLayer):
            widget = next(w for w in self._controls if w._control == widget)
        self._controls.remove(widget)
        widget.deleteLater()


class ControlLayerButton(QToolButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setIcon(theme.icon("control-add"))
        self.setToolTip("Add control layer")
        self.setAutoRaise(True)
        icon_height = self.iconSize().height()
        self.setIconSize(QSize(int(icon_height * 1.25), icon_height))


class StyleSelectWidget(QWidget):
    _value: Style
    _styles: list[Style]

    value_changed = pyqtSignal(Style)

    def __init__(self, parent):
        super().__init__(parent)
        self._value = Styles.list().default

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._combo = QComboBox(self)
        self.update_styles()
        self._combo.currentIndexChanged.connect(self.change_style)
        layout.addWidget(self._combo)

        settings = QToolButton(self)
        settings.setIcon(theme.icon("settings"))
        settings.setAutoRaise(True)
        settings.clicked.connect(self.show_settings)
        layout.addWidget(settings)

        Styles.list().changed.connect(self.update_styles)
        Styles.list().name_changed.connect(self.update_styles)
        root.connection.state_changed.connect(self.update_styles)

    def update_styles(self):
        comfy = root.connection.client_if_connected
        self._styles = filter_supported_styles(Styles.list(), comfy)
        with SignalBlocker(self._combo):
            self._combo.clear()
            for style in self._styles:
                icon = theme.sd_version_icon(resolve_sd_version(style, comfy))
                self._combo.addItem(icon, style.name, style.filename)
            if self._value in self._styles:
                self._combo.setCurrentText(self._value.name)
            elif len(self._styles) > 0:
                self._value = self._styles[0]
                self._combo.setCurrentIndex(0)
                self.value_changed.emit(self._value)

    def change_style(self):
        style = self._styles[self._combo.currentIndex()]
        if style != self._value:
            self._value = style
            self.value_changed.emit(style)

    def show_settings(self):
        SettingsDialog.instance().show(self._value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, style: Style):
        if style != self._value:
            self._value = style
            self._combo.setCurrentText(style.name)


def handle_weight_adjustment(
    self: MultiLineTextPromptWidget | SingleLineTextPromptWidget, event: QKeyEvent
):
    """Handles Ctrl + (arrow key up / arrow key down) attention weight adjustment."""
    if event.key() in [Qt.Key.Key_Up, Qt.Key.Key_Down] and (event.modifiers() & Qt.Modifier.CTRL):
        if self.hasSelectedText():
            start = self.selectionStart()
            end = self.selectionEnd()
        else:
            start, end = select_on_cursor_pos(self.text(), self.cursorPosition())

        text = self.text()
        target_text = text[start:end]
        text_after_edit = edit_attention(target_text, event.key() == Qt.Key.Key_Up)
        self.setText(text[:start] + text_after_edit + text[end:])
        if isinstance(self, MultiLineTextPromptWidget):
            self.setSelection(start, start + len(text_after_edit))
        else:
            # Note: setSelection has some wield bug in `SingleLineTextPromptWidget`
            # that the end range will be set to end of text. So set cursor instead
            # as compromise.
            self.setCursorPosition(start + len(text_after_edit) - 2)


class MultiLineTextPromptWidget(QPlainTextEdit):
    activated = pyqtSignal()

    _line_count = 2

    def __init__(self, parent):
        super().__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabChangesFocus(True)
        self.line_count = 2
        self.is_negative = False

    def keyPressEvent(self, event: QKeyEvent):
        handle_weight_adjustment(self, event)

        if (
            event.key() == Qt.Key.Key_Return
            and event.modifiers() == Qt.KeyboardModifier.ShiftModifier
        ):
            self.activated.emit()
        else:
            super().keyPressEvent(event)

    @property
    def line_count(self):
        return self._line_count

    @line_count.setter
    def line_count(self, value: int):
        self._line_count = value
        fm = QFontMetrics(ensure(self.document()).defaultFont())
        self.setFixedHeight(fm.lineSpacing() * value + 6)

    def hasSelectedText(self) -> bool:
        return self.textCursor().hasSelection()

    def selectionStart(self) -> int:
        return self.textCursor().selectionStart()

    def selectionEnd(self) -> int:
        return self.textCursor().selectionEnd()

    def cursorPosition(self) -> int:
        return self.textCursor().position()

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: str):
        self.setPlainText(text)

    def setSelection(self, start: int, end: int):
        new_cursor = self.textCursor()
        new_cursor.setPosition(min(end, len(self.toPlainText())))
        new_cursor.setPosition(min(start, len(self.toPlainText())), QTextCursor.KeepAnchor)
        self.setTextCursor(new_cursor)


class SingleLineTextPromptWidget(QLineEdit):
    def keyPressEvent(self, event: QKeyEvent):
        handle_weight_adjustment(self, event)
        super().keyPressEvent(event)


class TextPromptWidget(QWidget):
    """Wraps a single or multi-line text widget, with ability to switch between them.
    Using QPlainTextEdit set to a single line doesn't work properly because it still
    scrolls to the next line when eg. selecting and then looks like it's empty."""

    activated = pyqtSignal()
    text_changed = pyqtSignal(str)

    _multi: MultiLineTextPromptWidget
    _single: QLineEdit
    _line_count = 2
    _is_negative = False
    _base_color: QColor

    def __init__(self, line_count=2, is_negative=False, parent=None):
        super().__init__(parent)
        self._line_count = line_count
        self._is_negative = is_negative
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._multi = MultiLineTextPromptWidget(self)
        self._multi.line_count = self._line_count
        self._multi.activated.connect(self.notify_activated)
        self._multi.textChanged.connect(self.notify_text_changed)
        self._multi.setVisible(self._line_count > 1)

        self._single = SingleLineTextPromptWidget(self)
        self._single.textChanged.connect(self.notify_text_changed)
        self._single.returnPressed.connect(self.notify_activated)
        self._single.setVisible(self._line_count == 1)

        self._layout.addWidget(self._multi)
        self._layout.addWidget(self._single)

        palette: QPalette = self._multi.palette()
        self._base_color = palette.color(QPalette.ColorRole.Base)
        self.is_negative = self._is_negative

    def notify_text_changed(self):
        self.text_changed.emit(self.text)

    def notify_activated(self):
        self.activated.emit()

    @property
    def text(self):
        return self._multi.toPlainText() if self._line_count > 1 else self._single.text()

    @text.setter
    def text(self, value: str):
        if value == self.text:
            return
        if self._line_count > 1:
            self._multi.setPlainText(value)
        else:
            self._single.setText(value)

    @property
    def line_count(self):
        return self._line_count

    @line_count.setter
    def line_count(self, value: int):
        text = self.text
        self._line_count = value
        self.text = text
        self._multi.setVisible(self._line_count > 1)
        self._single.setVisible(self._line_count == 1)
        if self._line_count > 1:
            self._multi.line_count = self._line_count

    @property
    def is_negative(self):
        return self._is_negative

    @is_negative.setter
    def is_negative(self, value: bool):
        self._is_negative = value
        for w in [self._multi, self._single]:
            palette: QPalette = w.palette()
            color = self._base_color
            if not value:
                w.setPlaceholderText("Describe the content you want to see, or leave empty.")
            else:
                w.setPlaceholderText("Describe content you want to avoid.")
                color = QColor(color.red(), color.green() - 8, color.blue() - 8)
            palette.setColor(QPalette.ColorRole.Base, color)
            w.setPalette(palette)


class StrengthWidget(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, slider_range: tuple[int, int] = (1, 100), parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setMinimum(slider_range[0])
        self._slider.setMaximum(slider_range[1])
        self._slider.setSingleStep(5)
        self._slider.valueChanged.connect(self.notify_changed)

        self._input = QSpinBox(self)
        self._input.setMinimum(1)
        self._input.setMaximum(100)
        self._input.setSingleStep(5)
        self._input.setPrefix("Strength: ")
        self._input.setSuffix("%")
        self._input.valueChanged.connect(self.notify_changed)

        self._layout.addWidget(self._slider)
        self._layout.addWidget(self._input)

    def notify_changed(self, value: int):
        if self._slider.value() != value:
            self._slider.setValue(value)
        if self._input.value() != value:
            self._input.setValue(value)
        self.value_changed.emit(self.value)

    @property
    def value(self):
        return self._slider.value() / 100

    @value.setter
    def value(self, value: float):
        if value == self.value:
            return
        self._slider.setValue(int(value * 100))
        self._input.setValue(int(value * 100))


class WorkspaceSelectWidget(QToolButton):
    _icons = {
        Workspace.generation: theme.icon("workspace-generation"),
        Workspace.upscaling: theme.icon("workspace-upscaling"),
        Workspace.live: theme.icon("workspace-live"),
    }

    _value = Workspace.generation

    def __init__(self, parent):
        super().__init__(parent)

        menu = QMenu(self)
        menu.addAction(self._create_action("Generate", Workspace.generation))
        menu.addAction(self._create_action("Upscale", Workspace.upscaling))
        menu.addAction(self._create_action("Live", Workspace.live))

        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setMenu(menu)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setAutoRaise(True)
        self.setToolTip("Switch between workspaces: image generation, upscaling, live preview")
        self.setMinimumWidth(int(self.sizeHint().width() * 1.4))
        self.value = Workspace.generation

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, workspace: Workspace):
        self._value = workspace
        self.setIcon(self._icons[workspace])

    def _create_action(self, name: str, workspace: Workspace):
        action = QAction(name, self)
        action.setIcon(self._icons[workspace])
        action.setIconVisibleInMenu(True)
        action.triggered.connect(actions.set_workspace(workspace))
        return action
