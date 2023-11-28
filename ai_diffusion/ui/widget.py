from __future__ import annotations
from copy import copy
import random
from typing import Callable, Iterable, List, Optional

from PyQt5.QtWidgets import (
    QAction,
    QSlider,
    QPushButton,
    QWidget,
    QPlainTextEdit,
    QGroupBox,
    QLabel,
    QLineEdit,
    QProgressBar,
    QSizePolicy,
    QListWidget,
    QListView,
    QListWidgetItem,
    QMenu,
    QSpinBox,
    QDoubleSpinBox,
    QStackedWidget,
    QToolButton,
    QComboBox,
    QHBoxLayout,
    QVBoxLayout,
)
from PyQt5.QtGui import QColor, QFontMetrics, QGuiApplication, QKeyEvent, QMouseEvent, QPalette
from PyQt5.QtCore import Qt, QSize, QUuid, pyqtSignal
from krita import Krita, DockWidget
import krita

from .. import Control, ControlMode, Server, Style, Styles, Bounds, client
from . import actions, EventSuppression, SettingsDialog, theme
from .model import Model, ModelRegistry, Job, JobKind, JobQueue, State, Workspace
from .connection import Connection, ConnectionState
from ..image import Extent, Image
from ..resources import UpscalerName
from ..settings import ServerMode, settings
from ..util import ensure


class QueueWidget(QToolButton):
    _style = """
        QToolButton {{ border: none; border-radius: 6px; background-color: {color}; color: white; }}
        QToolButton::menu-indicator {{ width: 0px; }}"""

    def __init__(self, parent):
        super().__init__(parent)

        queue_menu = QMenu(self)
        queue_menu.addAction(self._create_action("Cancel active", actions.cancel_active))
        queue_menu.addAction(self._create_action("Cancel queued", actions.cancel_queued))
        queue_menu.addAction(self._create_action("Cancel all", actions.cancel_all))
        self.setMenu(queue_menu)

        self.setStyleSheet(self._style.format(color=theme.background_inactive))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setArrowType(Qt.ArrowType.NoArrow)

    def update(self, jobs: JobQueue):
        count = jobs.count(State.queued)
        if jobs.any_executing():
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
    changed = pyqtSignal()

    _model: Model
    _control: Control

    def __init__(self, parent=None):
        super().__init__(parent)
        model = Model.active()
        assert model
        self._model = model
        self._control = Control(ControlMode.image, self._model.document.active_layer)  # type: ignore (CTRLLAYER)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.mode_select = QComboBox(self)
        self.mode_select.setStyleSheet(
            "QComboBox { border:none; background-color:transparent; padding: 1px 12px 1px 2px;}"
        )
        for mode in (m for m in ControlMode if m is not ControlMode.inpaint):
            icon = theme.icon(f"control-{mode.name}")
            self.mode_select.addItem(icon, mode.text, mode.value)
        self.mode_select.currentIndexChanged.connect(self._notify)
        self.mode_select.currentIndexChanged.connect(self.update_installed_packages)

        self.layer_select = QComboBox(self)
        self.layer_select.currentIndexChanged.connect(self._notify)
        self.layer_select.setMinimumContentsLength(20)
        self.layer_select.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLength
        )

        self.generate_button = QToolButton(self)
        self.generate_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.generate_button.setIcon(theme.icon("control-generate"))
        self.generate_button.setToolTip("Generate control layer from current image")
        self.generate_button.clicked.connect(self.generate)

        self.add_pose_button = QToolButton(self)
        self.add_pose_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.add_pose_button.setIcon(theme.icon("add-pose"))
        self.add_pose_button.setToolTip("Add new character pose to selected layer")
        self.add_pose_button.setVisible(False)
        self.add_pose_button.clicked.connect(self._add_pose_character)

        self.strength_spin = QSpinBox(self)
        self.strength_spin.setRange(0, 100)
        self.strength_spin.setValue(100)
        self.strength_spin.setSuffix("%")
        self.strength_spin.setSingleStep(10)
        self.strength_spin.setToolTip("Control strength")
        self.strength_spin.valueChanged.connect(self._notify)

        self.end_spin = QDoubleSpinBox(self)
        self.end_spin.setRange(0.0, 1.0)
        self.end_spin.setValue(1.0)
        self.end_spin.setSuffix("")
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setToolTip("Control ending step ratio")
        self.end_spin.valueChanged.connect(self._notify)
        self.end_spin.setVisible(settings.show_control_end)

        self.error_text = QLabel(self)
        self.error_text.setText("ControlNet not installed")
        self.error_text.setStyleSheet(f"color: {theme.red};")
        self.error_text.setVisible(False)

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

        self.value = Control(ControlMode.scribble, self._model.document.active_layer, 1)  # type: ignore (CTRLLAYER)

        # non-exhaustive list of actions that create/remove layers
        Krita.instance().action("add_new_paint_layer").triggered.connect(self.update_layers)
        Krita.instance().action("duplicatelayer").triggered.connect(self.update_layers)
        Krita.instance().action("remove_layer").triggered.connect(self.update_layers)

    _suppress_changes = EventSuppression()

    def _notify(self):
        if not self._suppress_changes:
            self._control.mode = ControlMode(self.mode_select.currentData())
            id = self.layer_select.currentData()
            self._control.image = self._model.document.find_layer(id)  # type: ignore (CTRLLAYER)
            self._control.strength = self.strength_spin.value() / 100
            self._control.end = self.end_spin.value()
            self.changed.emit()

    def update_and_select_layer(self, id: QUuid):
        layers = reversed(self._model.document.image_layers)
        self.layer_select.clear()
        index = -1
        for layer in layers:
            self.layer_select.addItem(layer.name(), layer.uniqueId())
            if layer.uniqueId() == id:
                index = self.layer_select.count() - 1
        if index == -1 and self.value in self._model.control:
            self.remove()
        else:
            self.layer_select.setCurrentIndex(index)

    def update_layers(self):
        with self._suppress_changes:
            self.update_and_select_layer(self.layer_select.currentData())

    def generate(self):
        self._model.generate_control_layer(self.value)
        self.generate_button.setEnabled(False)
        self.layer_select.setEnabled(False)

    def remove(self):
        self._model.remove_control_layer(self.value)

    def _add_pose_character(self):
        self._model.document.add_pose_character(self.value.image)  # type: ignore (CTRLLAYER)

    @property
    def value(self):
        return self._control

    @value.setter
    def value(self, control: Control):
        changed = self._control != control
        self._control = copy(control)
        with self._suppress_changes:
            if changed:
                self.update_and_select_layer(control.image.uniqueId())  # type: ignore (CTRLLAYER)
                self.mode_select.setCurrentIndex(self.mode_select.findData(control.mode.value))
                self.strength_spin.setValue(int(control.strength * 100))
                self.end_spin.setValue(float(control.end))
            if self._check_is_installed():
                active_job = self._model.jobs.find(control)
                has_active_job = active_job and active_job.state is not State.finished
                self.generate_button.setEnabled(not has_active_job)
                self.layer_select.setEnabled(not has_active_job)

    def _check_is_installed(self):
        connection = Connection.instance()
        is_installed = True
        mode = ControlMode(self.mode_select.currentData())
        if connection.state is ConnectionState.connected:
            sdver = client.resolve_sd_version(self._model.style, connection.client)
            if mode is ControlMode.image:
                if connection.client.ip_adapter_model[sdver] is None:
                    self.error_text.setToolTip(f"The server is missing ip-adapter_sdxl_vit-h.bin")
                    is_installed = False
            elif connection.client.control_model[mode][sdver] is None:
                filenames = mode.filenames(sdver)
                if filenames:
                    self.error_text.setToolTip(f"The server is missing {filenames}")
                else:
                    self.error_text.setText(f"Not supported for {sdver.value}")
                is_installed = False
        self.error_text.setVisible(False)  # Avoid layout resize
        self.layer_select.setVisible(is_installed)
        self.generate_button.setVisible(
            is_installed and mode not in [ControlMode.image, ControlMode.stencil]
        )
        self.add_pose_button.setVisible(is_installed and mode is ControlMode.pose)
        self.add_pose_button.setEnabled(self._is_vector_layer())
        self.strength_spin.setVisible(is_installed)
        self.strength_spin.setEnabled(self._is_first_image_mode())
        self.end_spin.setVisible(is_installed and settings.show_control_end)
        self.end_spin.setEnabled(self._is_first_image_mode())
        self.error_text.setVisible(not is_installed)
        return is_installed

    def update_installed_packages(self):
        _ = self._check_is_installed()

    def _is_first_image_mode(self):
        return self._control.mode is not ControlMode.image or self._control == next(
            (c for c in self._model.control if c.mode is ControlMode.image), None
        )

    def _is_vector_layer(self):
        return isinstance(self.value.image, krita.Node) and self.value.image.type() == "vectorlayer"


class ControlListWidget(QWidget):
    _controls: List[ControlWidget]

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        self._controls = []

    def add(self):
        model = Model.active()
        assert model
        model.control.append(Control(ControlMode.image, model.document.active_layer))  # type: ignore (CTRLLAYER)
        self.value = model.control

    @property
    def value(self):
        # Filter out controls whose layer has been deleted
        result, removed = [], []
        for control in self._controls:
            c = control.value
            removed.append(control) if c.image is None else result.append(c)
        for control in removed:
            self._remove_widget(control)
        return result

    @value.setter
    def value(self, controls: List[Control]):
        with self._suppress_changes:
            if len(controls) != len(self._controls):
                while len(self._controls) > len(controls):
                    self._remove_widget(self._controls[0])
                while len(self._controls) < len(controls):
                    self._add_widget()
            for control, widget in zip(controls, self._controls):
                widget.value = control

    def notify_style_changed(self):
        for control in self._controls:
            control.update_installed_packages()

    _suppress_changes = EventSuppression()

    def _notify(self):
        if not self._suppress_changes:
            self.changed.emit()

    def _add_widget(self):
        control = ControlWidget(self)
        control.changed.connect(self._notify)
        self._controls.append(control)
        self._layout.addWidget(control)
        return control

    def _remove_widget(self, control: ControlWidget):
        self._controls.remove(control)
        control.deleteLater()

    def update_control_field(self, name, function):
        for control in self._controls:
            setting = getattr(control, name, None)
            if setting is not None:
                function(setting)


class ControlLayerButton(QToolButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setIcon(theme.icon("control-add"))
        self.setToolTip("Add control layer")
        self.setAutoRaise(True)
        icon_height = self.iconSize().height()
        self.setIconSize(QSize(int(icon_height * 1.25), icon_height))


class HistoryWidget(QListWidget):
    _last_prompt: Optional[str] = None
    _last_bounds: Optional[Bounds] = None

    def __init__(self, parent):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setResizeMode(QListView.Adjust)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFlow(QListView.LeftToRight)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(96, 96))
        self.itemClicked.connect(self.handle_preview_click)

    def add(self, job: Job):
        if self._last_prompt != job.prompt or self._last_bounds != job.bounds:
            self._last_prompt = job.prompt
            self._last_bounds = job.bounds
            prompt = job.prompt if job.prompt != "" else "<no prompt>"

            header = QListWidgetItem(f"{job.timestamp:%H:%M} - {prompt}")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setData(Qt.ItemDataRole.UserRole, job.id)
            header.setData(Qt.ItemDataRole.ToolTipRole, job.prompt)
            header.setSizeHint(QSize(800, self.fontMetrics().lineSpacing() + 4))
            header.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            self.addItem(header)

        for i, img in enumerate(job.results):
            item = QListWidgetItem(img.to_icon(), None)  # type: ignore (text can be None)
            item.setData(Qt.ItemDataRole.UserRole, job.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, i)
            item.setData(
                Qt.ItemDataRole.ToolTipRole,
                f"{job.prompt}\nClick to toggle preview, double-click to apply.",
            )
            self.addItem(item)

        scrollbar = self.verticalScrollBar()
        if scrollbar.isVisible() and scrollbar.value() >= scrollbar.maximum() - 4:
            self.scrollToBottom()

    def is_finished(self, job: Job):
        return job.kind is JobKind.diffusion and job.state is State.finished

    def prune(self, jobs: JobQueue):
        first_id = next((job.id for job in jobs if self.is_finished(job)), None)
        while self.count() > 0 and self.item(0).data(Qt.ItemDataRole.UserRole) != first_id:
            self.takeItem(0)

    def rebuild(self, jobs: Iterable[Job]):
        self.clear()
        for job in filter(self.is_finished, jobs):
            self.add(job)

    def item_info(self, item: QListWidgetItem):
        return item.data(Qt.ItemDataRole.UserRole), item.data(Qt.ItemDataRole.UserRole + 1)

    def handle_preview_click(self, item: QListWidgetItem):
        if item.text() != "" and item.text() != "<no prompt>":
            prompt = item.data(Qt.ItemDataRole.ToolTipRole)
            QGuiApplication.clipboard().setText(prompt)

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


class StyleSelectWidget(QWidget):
    _value: Style
    _styles: list[Style]

    changed = pyqtSignal()

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
        Connection.instance().changed.connect(self.update_styles)

    def update_styles(self):
        comfy = Connection.instance().client_if_connected
        self._styles = client.filter_supported_styles(Styles.list(), comfy)
        self._combo.blockSignals(True)
        self._combo.clear()
        for style in self._styles:
            icon = theme.sd_version_icon(client.resolve_sd_version(style, comfy))
            self._combo.addItem(icon, style.name, style.filename)
        if self._value in self._styles:
            self._combo.setCurrentText(self._value.name)
        elif len(self._styles) > 0:
            self._value = self._styles[0]
            self._combo.setCurrentIndex(0)
            self.changed.emit()
        self._combo.blockSignals(False)

    def change_style(self):
        style = self._styles[self._combo.currentIndex()]
        if style != self._value:
            self._value = style
            self.changed.emit()

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
        fm = QFontMetrics(self.document().defaultFont())
        self.setFixedHeight(fm.lineSpacing() * value + 6)


class TextPromptWidget(QWidget):
    """Wraps a single or multi-line text widget, with ability to switch between them.
    Using QPlainTextEdit set to a single line doesn't work properly because it still
    scrolls to the next line when eg. selecting and then looks like it's empty."""

    activated = pyqtSignal()
    changed = pyqtSignal()

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

        self._single = QLineEdit(self)
        self._single.textChanged.connect(self.notify_text_changed)
        self._single.returnPressed.connect(self.notify_activated)
        self._single.setVisible(self._line_count == 1)

        self._layout.addWidget(self._multi)
        self._layout.addWidget(self._single)

        palette: QPalette = self._multi.palette()
        self._base_color = palette.color(QPalette.ColorRole.Base)
        self.is_negative = self._is_negative

    def notify_text_changed(self):
        self.changed.emit()

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


class GenerationWidget(QWidget):
    _model: Optional[Model] = None

    def __init__(self):
        super().__init__()
        settings.changed.connect(self.update_settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 2, 0)
        self.setLayout(layout)

        self.workspace_select = WorkspaceSelectWidget(self)

        self.style_select = StyleSelectWidget(self)
        self.style_select.changed.connect(self.change_style)

        style_layout = QHBoxLayout()
        style_layout.addWidget(self.workspace_select)
        style_layout.addWidget(self.style_select)
        layout.addLayout(style_layout)

        self.prompt_textbox = TextPromptWidget(parent=self)
        self.prompt_textbox.line_count = settings.prompt_line_count
        self.prompt_textbox.changed.connect(self.change_prompt)
        self.prompt_textbox.activated.connect(self.generate)

        self.negative_textbox = TextPromptWidget(line_count=1, is_negative=True, parent=self)
        self.negative_textbox.setVisible(settings.show_negative_prompt)
        self.negative_textbox.changed.connect(self.change_negative_prompt)
        self.negative_textbox.activated.connect(self.generate)

        prompt_layout = QVBoxLayout()
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(2)
        prompt_layout.addWidget(self.prompt_textbox)
        prompt_layout.addWidget(self.negative_textbox)
        layout.addLayout(prompt_layout)

        self.control_list = ControlListWidget(self)
        self.control_list.changed.connect(self.change_control)
        layout.addWidget(self.control_list)

        self.strength_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.strength_slider.setMinimum(1)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setSingleStep(5)
        self.strength_slider.valueChanged.connect(self.change_strength)

        self.strength_input = QSpinBox(self)
        self.strength_input.setMinimum(1)
        self.strength_input.setMaximum(100)
        self.strength_input.setSingleStep(5)
        self.strength_input.setPrefix("Strength: ")
        self.strength_input.setSuffix("%")
        self.strength_input.valueChanged.connect(self.change_strength)

        self.add_control_button = ControlLayerButton(self)
        self.add_control_button.clicked.connect(self.control_list.add)

        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_input)
        strength_layout.addWidget(self.add_control_button)
        layout.addLayout(strength_layout)

        self.generate_button = QPushButton("Generate", self)
        self.generate_button.setMinimumHeight(int(self.generate_button.sizeHint().height() * 1.2))
        self.generate_button.clicked.connect(self.generate)

        self.queue_button = QueueWidget(self)

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
        self.history.itemSelectionChanged.connect(self.select_preview)
        self.history.itemDoubleClicked.connect(self.apply_result)
        layout.addWidget(self.history)

        self.apply_button = QPushButton(theme.icon("apply"), "Apply", self)
        self.apply_button.clicked.connect(self.apply_selected_result)
        layout.addWidget(self.apply_button)

    @property
    def model(self):
        assert self._model is not None
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model != model:
            self.history.rebuild(model.history)
            self._model = model

    def update(self):
        model = self.model
        self.workspace_select.value = model.workspace
        self.style_select.value = model.style
        if self.style_select.value != model.style:
            self.change_style()  # Model style is not in style list (filtered out)
        self.prompt_textbox.text = model.prompt
        self.negative_textbox.text = model.negative_prompt
        self.control_list.value = model.control
        self.strength_input.setValue(int(model.strength * 100))
        self.error_text.setText(model.error)
        self.error_text.setVisible(model.error != "")
        self.apply_button.setEnabled(model.can_apply_result)
        self.update_progress()

    def update_progress(self):
        self.progress_bar.setValue(int(self.model.progress * 100))
        self.queue_button.update(self.model.jobs)

    def update_settings(self, key: str, value):
        if key == "prompt_line_count":
            self.prompt_textbox.line_count = value
        elif key == "show_negative_prompt":
            self.negative_textbox.text = ""
            self.negative_textbox.setVisible(value)
        elif key == "show_control_end":
            self.control_list.update_control_field("end_spin", lambda x: x.setVisible(value))
            self.control_list.update_control_field("end_spin", lambda x: x.setValue(1.0))

    def show_results(self, job: Job):
        if job.kind is JobKind.diffusion:
            self.history.prune(self.model.jobs)
            self.history.add(job)

    def generate(self):
        self.model.generate()
        self.update()

    def change_style(self):
        if self._model:
            self.model.style = self.style_select.value
            self.control_list.notify_style_changed()

    def change_prompt(self):
        self.model.prompt = self.prompt_textbox.text

    def change_negative_prompt(self):
        self.model.negative_prompt = self.negative_textbox.text

    def change_strength(self, value: int):
        self.model.strength = value / 100
        if self.strength_input.value() != value:
            self.strength_input.setValue(value)
        if self.strength_slider.value() != value:
            self.strength_slider.setValue(value)

    def change_control(self):
        self.model.control = self.control_list.value

    def show_preview(self, item: QListWidgetItem):
        job_id, index = self.history.item_info(item)
        self.model.show_preview(job_id, index)

    def select_preview(self):
        items = self.history.selectedItems()
        if len(items) > 0:
            self.show_preview(items[0])
        else:
            self.model.hide_preview()

    def apply_selected_result(self):
        self.model.apply_current_result()

    def apply_result(self, item: QListWidgetItem):
        self.show_preview(item)
        self.apply_selected_result()


class UpscaleWidget(QWidget):
    _model: Optional[Model] = None

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 0)
        self.setLayout(layout)

        self.workspace_select = WorkspaceSelectWidget(self)

        self.model_select = QComboBox(self)
        self.model_select.currentIndexChanged.connect(self.change_model)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.workspace_select)
        model_layout.addWidget(self.model_select)
        layout.addLayout(model_layout)

        self.factor_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.factor_slider.setMinimum(100)
        self.factor_slider.setMaximum(400)
        self.factor_slider.setTickInterval(50)
        self.factor_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.factor_slider.setSingleStep(50)
        self.factor_slider.setPageStep(50)
        self.factor_slider.valueChanged.connect(self.change_factor_slider)

        self.factor_input = QDoubleSpinBox(self)
        self.factor_input.setMinimum(1.0)
        self.factor_input.setMaximum(4.0)
        self.factor_input.setSingleStep(0.5)
        self.factor_input.setPrefix("Scale: ")
        self.factor_input.setSuffix("x")
        self.factor_input.setDecimals(2)
        self.factor_input.valueChanged.connect(self.change_factor)

        factor_layout = QHBoxLayout()
        factor_layout.addWidget(self.factor_slider)
        factor_layout.addWidget(self.factor_input)
        layout.addLayout(factor_layout)

        self.target_label = QLabel("Target size:", self)
        layout.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addSpacing(6)

        self.refinement_checkbox = QGroupBox("Refine upscaled image", self)
        self.refinement_checkbox.setCheckable(True)
        self.refinement_checkbox.toggled.connect(self.change_refinement)

        self.style_select = StyleSelectWidget(self)
        self.style_select.changed.connect(self.change_style)

        self.strength_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.strength_slider.setMinimum(20)
        self.strength_slider.setMaximum(50)
        self.strength_slider.setSingleStep(5)
        self.strength_slider.valueChanged.connect(self.change_strength)

        self.strength_input = QSpinBox(self)
        self.strength_input.setMinimum(0)
        self.strength_input.setMaximum(100)
        self.strength_input.setSingleStep(5)
        self.strength_input.setPrefix("Strength: ")
        self.strength_input.setSuffix("%")
        self.strength_input.valueChanged.connect(self.change_strength)

        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_input)

        group_layout = QVBoxLayout(self.refinement_checkbox)
        group_layout.addWidget(self.style_select)
        group_layout.addLayout(strength_layout)
        self.refinement_checkbox.setLayout(group_layout)
        layout.addWidget(self.refinement_checkbox)
        self.factor_input.setMinimumWidth(self.strength_input.width() + 10)

        self.upscale_button = QPushButton("Upscale", self)
        self.upscale_button.setMinimumHeight(int(self.upscale_button.sizeHint().height() * 1.2))
        self.upscale_button.clicked.connect(self.upscale)

        self.queue_button = QueueWidget(self)

        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.upscale_button)
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

        layout.addStretch()

    @property
    def model(self):
        assert self._model is not None
        return self._model

    @model.setter
    def model(self, model: Model):
        self._model = model

    def update(self):
        params = self.model.upscale
        self.workspace_select.value = self.model.workspace
        self.update_models()
        self.factor_slider.setValue(int(params.factor * 100))
        self.factor_input.setValue(params.factor)
        self.update_target_extent()
        self.refinement_checkbox.setChecked(params.use_diffusion)
        self.style_select.value = self.model.style
        self.strength_slider.setValue(int(params.strength * 100))
        self.strength_input.setValue(int(params.strength * 100))
        self.error_text.setText(self.model.error)
        self.error_text.setVisible(self.model.error != "")
        self.update_progress()

    def update_models(self):
        client = Connection.instance().client
        self.model_select.blockSignals(True)
        self.model_select.clear()
        for file in client.upscalers:
            if file == UpscalerName.default.value:
                name = f"Default ({file.removesuffix('.pth')})"
                self.model_select.insertItem(0, name, file)
            elif file == UpscalerName.quality.value:
                name = f"Quality ({file.removesuffix('.pth')})"
                self.model_select.insertItem(1, name, file)
            elif file == UpscalerName.sharp.value:
                name = f"Sharp ({file.removesuffix('.pth')})"
                self.model_select.insertItem(2, name, file)
            else:
                self.model_select.addItem(file, file)
        selected = self.model_select.findData(self.model.upscale.upscaler)
        self.model_select.setCurrentIndex(max(selected, 0))
        self.model_select.blockSignals(False)

    def update_progress(self):
        self.progress_bar.setValue(int(self.model.progress * 100))
        self.queue_button.update(self.model.jobs)

    def upscale(self):
        self.model.upscale_image()

    def change_model(self):
        self.model.upscale.upscaler = self.model_select.currentData()

    def change_factor_slider(self, value: int | float):
        value = round(value / 50) * 50
        if self.factor_slider.value() != value:
            self.factor_slider.setValue(value)
        else:
            value_float = value / 100
            self.model.upscale.factor = value_float
            if self.factor_input.value() != value_float:
                self.factor_input.setValue(value_float)
            self.update_target_extent()

    def change_factor(self, value: float):
        self.model.upscale.factor = value
        value_int = round(value * 100)
        if self.factor_slider.value() != value_int:
            self.factor_slider.blockSignals(True)
            self.factor_slider.setValue(value_int)
            self.factor_slider.blockSignals(False)
        self.update_target_extent()

    def update_target_extent(self):
        e = self.model.upscale.target_extent
        self.target_label.setText(f"Target size: {e.width} x {e.height}")

    def change_refinement(self):
        self.model.upscale.use_diffusion = self.refinement_checkbox.isChecked()
        self.update()

    def change_style(self):
        if self._model is not None:
            self.model.style = self.style_select.value

    def change_strength(self, value: int):
        self.model.upscale.strength = value / 100
        if self.strength_input.value() != value:
            self.strength_input.setValue(value)
        if self.strength_slider.value() != value:
            self.strength_slider.setValue(value)


class LiveWidget(QWidget):
    _play_icon = theme.icon("play")
    _pause_icon = theme.icon("pause")

    _model: Optional[Model] = None

    def __init__(self):
        super().__init__()
        settings.changed.connect(self.update_settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 0)
        self.setLayout(layout)

        self.workspace_select = WorkspaceSelectWidget(self)

        self.active_button = QToolButton(self)
        self.active_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.active_button.setIcon(self._play_icon)
        self.active_button.setAutoRaise(True)
        self.active_button.setToolTip("Start/stop live preview")
        self.active_button.clicked.connect(self.toggle_active)

        self.apply_button = QToolButton(self)
        self.apply_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.apply_button.setIcon(theme.icon("copy-image"))
        self.apply_button.setAutoRaise(True)
        self.apply_button.setEnabled(False)
        self.apply_button.setToolTip("Copy the current result to the image as a new layer")
        self.apply_button.clicked.connect(self.apply_result)

        self.style_select = StyleSelectWidget(self)
        self.style_select.changed.connect(self.change_style)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.workspace_select)
        controls_layout.addWidget(self.active_button)
        controls_layout.addWidget(self.apply_button)
        controls_layout.addWidget(self.style_select)
        layout.addLayout(controls_layout)

        self.strength_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.strength_slider.setMinimum(1)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setSingleStep(5)
        self.strength_slider.valueChanged.connect(self.change_strength)

        self.strength_input = QSpinBox(self)
        self.strength_input.setMinimum(1)
        self.strength_input.setMaximum(100)
        self.strength_input.setSingleStep(5)
        self.strength_input.setPrefix("Strength: ")
        self.strength_input.setSuffix("%")
        self.strength_input.valueChanged.connect(self.change_strength)

        self.seed_input = QSpinBox(self)
        self.seed_input.setMinimum(0)
        self.seed_input.setMaximum(2**31 - 1)
        self.seed_input.setPrefix("Seed: ")
        self.seed_input.setToolTip(
            "The seed controls the random part of the output. The same seed value will always"
            " produce the same result."
        )
        self.seed_input.valueChanged.connect(self.change_seed)

        self.random_seed_button = QToolButton(self)
        self.random_seed_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.random_seed_button.setIcon(theme.icon("random"))
        self.random_seed_button.setAutoRaise(True)
        self.random_seed_button.setToolTip(
            "Generate a random seed value to get a variation of the image."
        )
        self.random_seed_button.clicked.connect(self.randomize_seed)

        params_layout = QHBoxLayout()
        params_layout.addWidget(self.strength_slider)
        params_layout.addWidget(self.strength_input)
        params_layout.addWidget(self.seed_input)
        params_layout.addWidget(self.random_seed_button)
        layout.addLayout(params_layout)

        self.control_list = ControlListWidget(self)
        self.control_list.changed.connect(self.change_control)

        self.add_control_button = ControlLayerButton(self)
        self.add_control_button.clicked.connect(self.control_list.add)

        self.prompt_textbox = TextPromptWidget(line_count=1, parent=self)
        self.prompt_textbox.changed.connect(self.change_prompt)

        self.negative_textbox = TextPromptWidget(line_count=1, is_negative=True, parent=self)
        self.negative_textbox.setVisible(settings.show_negative_prompt)
        self.negative_textbox.changed.connect(self.change_negative_prompt)

        prompt_layout = QVBoxLayout()
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(2)
        prompt_layout.addWidget(self.prompt_textbox)
        prompt_layout.addWidget(self.negative_textbox)
        cond_layout = QHBoxLayout()
        cond_layout.addLayout(prompt_layout)
        cond_layout.addWidget(self.add_control_button)
        layout.addLayout(cond_layout)
        layout.addWidget(self.control_list)

        self.error_text = QLabel(self)
        self.error_text.setStyleSheet("font-weight: bold; color: red;")
        self.error_text.setWordWrap(True)
        self.error_text.setVisible(False)
        layout.addWidget(self.error_text)

        self.preview_area = QLabel(self)
        self.preview_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_area.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.preview_area)

    @property
    def model(self):
        assert self._model is not None
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model:
            self._model.job_finished.disconnect(self.handle_job_finished)
        self._model = model
        self._model.job_finished.connect(self.handle_job_finished)

    def update(self):
        self.workspace_select.value = self.model.workspace
        self.active_button.setIcon(
            self._pause_icon if self.model.live.is_active else self._play_icon
        )
        self.apply_button.setEnabled(self.model.has_live_result)
        self.style_select.value = self.model.style
        self.strength_input.setValue(int(self.model.live.strength * 100))
        self.strength_slider.setValue(int(self.model.live.strength * 100))
        self.seed_input.setValue(self.model.live.seed)
        self.prompt_textbox.text = self.model.prompt
        self.negative_textbox.text = self.model.negative_prompt
        self.control_list.value = self.model.control
        self.error_text.setText(self.model.error)
        self.error_text.setVisible(self.model.error != "")

    def update_settings(self, key: str, value):
        if key == "show_negative_prompt":
            self.negative_textbox.text = ""
            self.negative_textbox.setVisible(value)

    def toggle_active(self):
        self.model.live.is_active = not self.model.live.is_active
        self.update()
        if self.model.live.is_active:
            self.model.generate_live()

    def apply_result(self):
        if self.model.has_live_result:
            self.model.add_live_layer()

    def change_style(self):
        if self._model is not None:
            self.model.style = self.style_select.value
            self.control_list.notify_style_changed()

    def change_strength(self, value: int):
        self.model.live.strength = value / 100
        if self.strength_input.value() != value:
            self.strength_input.setValue(value)
        if self.strength_slider.value() != value:
            self.strength_slider.setValue(value)

    def change_seed(self, value: int):
        self.model.live.seed = value

    def randomize_seed(self):
        self.seed_input.setValue(random.randint(0, 2**31 - 1))

    def change_prompt(self):
        self.model.prompt = self.prompt_textbox.text

    def change_negative_prompt(self):
        self.model.negative_prompt = self.negative_textbox.text

    def change_control(self):
        self.model.control = self.control_list.value

    def handle_job_finished(self, job: Job):
        if job.kind is JobKind.live_preview:
            if len(job.results) > 0:  # no results if input didn't change!
                target = Extent.from_qsize(self.preview_area.size())
                img = Image.scale_to_fit(job.results[0], target)
                self.preview_area.setPixmap(img.to_pixmap())
                self.preview_area.setMinimumSize(256, 256)
            if self.model.workspace is Workspace.live and self.model.live.is_active:
                self.model.generate_live()


class WelcomeWidget(QWidget):
    _server: Server

    def __init__(self, server: Server):
        super().__init__()
        self._server = server

        layout = QVBoxLayout()
        self.setLayout(layout)

        header_layout = QHBoxLayout()
        header_logo = QLabel(self)
        header_logo.setPixmap(theme.logo().scaled(64, 64))
        header_logo.setMaximumSize(64, 64)
        header_text = QLabel("AI Image\nGeneration", self)
        header_text.setStyleSheet("font-size: 12pt")
        header_layout.addWidget(header_logo)
        header_layout.addWidget(header_text)
        layout.addLayout(header_layout)
        layout.addSpacing(12)

        self._connect_status = QLabel("Not connected to server.", self)
        layout.addWidget(self._connect_status)
        layout.addSpacing(6)

        self._connect_error = QLabel(self)
        self._connect_error.setVisible(False)
        self._connect_error.setWordWrap(True)
        self._connect_error.setStyleSheet(f"color: {theme.yellow};")
        layout.addWidget(self._connect_error)

        self._settings_button = QPushButton(theme.icon("settings"), "Configure", self)
        self._settings_button.setMinimumHeight(32)
        self._settings_button.clicked.connect(self.show_settings)
        layout.addWidget(self._settings_button)

        layout.addStretch()

        Connection.instance().changed.connect(self.update)
        self.update()

    def update(self):
        connection = Connection.instance()
        if connection.state in [ConnectionState.disconnected, ConnectionState.error]:
            self._connect_status.setText("Not connected to server.")
        if connection.state is ConnectionState.error:
            self._connect_error.setText(
                "Connection attempt failed! Click below to configure and reconnect."
            )
            self._connect_error.setVisible(True)
        if (
            connection.state is ConnectionState.disconnected
            and settings.server_mode is ServerMode.managed
        ):
            if self._server.upgrade_required:
                self._connect_error.setText("Server version is outdated. Click below to upgrade.")
            else:
                self._connect_error.setText(
                    "Server is not installed or not running. Click below to start."
                )
            self._connect_error.setVisible(True)
        if connection.state is ConnectionState.connecting:
            self._connect_status.setText(f"Connecting to server...")
        if connection.state is ConnectionState.connected:
            self._connect_status.setText(
                f"Connected to server at {connection.client.url}.\n\nCreate"
                " a new document or open an existing image to start!"
            )
            self._connect_error.setVisible(False)

    def show_settings(self):
        Krita.instance().action("ai_diffusion_settings").trigger()


class ImageDiffusionWidget(DockWidget):
    _server: Server = ...  # type: ignore (injected in extension.py)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Generation")
        self._welcome = WelcomeWidget(self._server)
        self._generation = GenerationWidget()
        self._upscaling = UpscaleWidget()
        self._live = LiveWidget()
        self._frame = QStackedWidget(self)
        self._frame.addWidget(self._welcome)
        self._frame.addWidget(self._generation)
        self._frame.addWidget(self._upscaling)
        self._frame.addWidget(self._live)
        self.setWidget(self._frame)

        Connection.instance().changed.connect(self.update)
        ModelRegistry.instance().created.connect(self.register_model)

    def canvasChanged(self, canvas):
        self.update()

    def register_model(self, model):
        model.changed.connect(self.update)
        model.job_finished.connect(self._generation.show_results)
        model.progress_changed.connect(self.update_progress)

    def update(self):
        model = Model.active()
        connection = Connection.instance()
        if model is None or connection.state in [
            ConnectionState.disconnected,
            ConnectionState.connecting,
            ConnectionState.error,
        ]:
            self._frame.setCurrentWidget(self._welcome)
        elif model.workspace is Workspace.generation:
            self._generation.model = model
            self._generation.update()
            self._frame.setCurrentWidget(self._generation)
        elif model.workspace is Workspace.upscaling:
            self._upscaling.model = model
            self._upscaling.update()
            self._frame.setCurrentWidget(self._upscaling)
        elif model.workspace is Workspace.live:
            self._live.model = model
            self._live.update()
            self._frame.setCurrentWidget(self._live)

    def update_progress(self):
        if model := Model.active():
            if model.workspace is Workspace.generation:
                self._generation.update_progress()
            elif model.workspace is Workspace.upscaling:
                self._upscaling.update_progress()
