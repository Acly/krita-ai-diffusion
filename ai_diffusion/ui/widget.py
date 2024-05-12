from __future__ import annotations
from typing import Callable, cast

from PyQt5.QtWidgets import (
    QAction,
    QSlider,
    QWidget,
    QPlainTextEdit,
    QLabel,
    QLineEdit,
    QMenu,
    QSpinBox,
    QToolButton,
    QComboBox,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QWidgetAction,
    QCheckBox,
    QGridLayout,
    QCompleter,
    QFrame,
)
from PyQt5.QtGui import (
    QColor,
    QFontMetrics,
    QKeyEvent,
    QMouseEvent,
    QPalette,
    QResizeEvent,
    QTextCursor,
    QPainter,
    QImage,
    QPixmap,
)
from PyQt5.QtCore import QObject, QEvent, Qt, QMetaObject, QSize, QStringListModel, pyqtSignal

from ..style import Style, Styles
from ..root import root
from ..client import filter_supported_styles, resolve_sd_version
from ..properties import Binding, Bind, bind, bind_combo
from ..jobs import JobState
from ..model import Model, Workspace, SamplingQuality, Region, RegionTree
from ..text import LoraId, edit_attention, select_on_cursor_pos
from ..util import ensure
from .control import ControlListWidget
from .settings import SettingsDialog, settings
from .theme import SignalBlocker
from . import actions, theme


class QueuePopup(QMenu):
    _model: Model
    _connections: list[QMetaObject.Connection]

    def __init__(self, supports_batch=True, parent: QWidget | None = None):
        super().__init__(parent)
        self._connections = []

        palette = self.palette()
        self.setObjectName("QueuePopup")
        self.setStyleSheet(
            f"""
            QWidget#QueuePopup {{
                background-color: {palette.window().color().name()}; 
                border: 1px solid {palette.dark().color().name()};
            }}"""
        )

        self._layout = QGridLayout()
        self.setLayout(self._layout)

        batch_label = QLabel("Batches", self)
        batch_label.setVisible(supports_batch)
        self._layout.addWidget(batch_label, 0, 0)
        batch_layout = QHBoxLayout()
        self._batch_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._batch_slider.setMinimum(1)
        self._batch_slider.setMaximum(10)
        self._batch_slider.setSingleStep(1)
        self._batch_slider.setPageStep(1)
        self._batch_slider.setVisible(supports_batch)
        self._batch_slider.setToolTip("Number of jobs to enqueue at once")
        self._batch_label = QLabel("1", self)
        self._batch_label.setVisible(supports_batch)
        batch_layout.addWidget(self._batch_slider)
        batch_layout.addWidget(self._batch_label)
        self._layout.addLayout(batch_layout, 0, 1)

        self._seed_label = QLabel("Seed", self)
        self._layout.addWidget(self._seed_label, 1, 0)
        self._seed_input = QSpinBox(self)
        self._seed_check = QCheckBox(self)
        self._seed_check.setText("Fixed")
        self._seed_input.setMinimum(0)
        self._seed_input.setMaximum(2**31 - 1)
        self._seed_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._seed_input.setToolTip(
            "The seed controls the random part of the output. A fixed seed value will always"
            " produce the same result for the same inputs."
        )
        self._randomize_seed = QToolButton(self)
        self._randomize_seed.setIcon(theme.icon("random"))
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(self._seed_check)
        seed_layout.addWidget(self._seed_input)
        seed_layout.addWidget(self._randomize_seed)
        self._layout.addLayout(seed_layout, 1, 1)

        enqueue_label = QLabel("Enqueue", self)
        self._queue_front_combo = QComboBox(self)
        self._queue_front_combo.addItem("in Front (new jobs first)", True)
        self._queue_front_combo.addItem("at the Back", False)
        self._layout.addWidget(enqueue_label, 2, 0)
        self._layout.addWidget(self._queue_front_combo, 2, 1)

        cancel_label = QLabel("Cancel", self)
        self._layout.addWidget(cancel_label, 3, 0)
        self._cancel_active = self._create_cancel_button("Active", actions.cancel_active)
        self._cancel_queued = self._create_cancel_button("Queued", actions.cancel_queued)
        self._cancel_all = self._create_cancel_button("All", actions.cancel_all)
        cancel_layout = QHBoxLayout()
        cancel_layout.addWidget(self._cancel_active)
        cancel_layout.addWidget(self._cancel_queued)
        cancel_layout.addWidget(self._cancel_all)
        self._layout.addLayout(cancel_layout, 3, 1)

        self._model = root.active_model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        Binding.disconnect_all(self._connections)
        self._model = model
        self._randomize_seed.setEnabled(self._model.fixed_seed)
        self._seed_input.setEnabled(self._model.fixed_seed)
        self._batch_label.setText(str(self._model.batch_count))
        self._connections = [
            bind(self._model, "batch_count", self._batch_slider, "value"),
            model.batch_count_changed.connect(lambda v: self._batch_label.setText(str(v))),
            bind(self._model, "seed", self._seed_input, "value"),
            bind(self._model, "fixed_seed", self._seed_check, "checked", Bind.one_way),
            self._seed_check.toggled.connect(lambda v: setattr(self._model, "fixed_seed", v)),
            self._model.fixed_seed_changed.connect(self._seed_input.setEnabled),
            self._model.fixed_seed_changed.connect(self._randomize_seed.setEnabled),
            self._randomize_seed.clicked.connect(self._model.generate_seed),
            bind_combo(self._model, "queue_front", self._queue_front_combo),
            model.jobs.count_changed.connect(self._update_cancel_buttons),
        ]

    def _create_cancel_button(self, name: str, action: Callable[[], None]):
        button = QToolButton(self)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setText(name)
        button.setIcon(theme.icon("cancel"))
        button.setEnabled(False)
        button.clicked.connect(action)
        return button

    def _update_cancel_buttons(self):
        has_active = self._model.jobs.any_executing()
        has_queued = self._model.jobs.count(JobState.queued) > 0
        self._cancel_active.setEnabled(has_active)
        self._cancel_queued.setEnabled(has_queued)
        self._cancel_all.setEnabled(has_active or has_queued)

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        if parent := cast(QWidget, self.parent()):
            parent.close()
        return super().mouseReleaseEvent(a0)


class QueueButton(QToolButton):
    _model: Model
    _popup: QueuePopup

    def __init__(self, supports_batch=True, parent: QWidget | None = None):
        super().__init__(parent)
        self._model = root.active_model
        self._model.jobs.count_changed.connect(self._update)

        self._popup = QueuePopup(supports_batch)
        popup_action = QWidgetAction(self)
        popup_action.setDefaultWidget(self._popup)
        self.addAction(popup_action)

        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._update()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        self._model.jobs.count_changed.disconnect(self._update)
        self._model = model
        self._popup.model = model
        self._model.jobs.count_changed.connect(self._update)

    def _update(self):
        count = self._model.jobs.count(JobState.queued)
        if self._model.jobs.any_executing():
            self.setIcon(theme.icon("queue-active"))
            if count > 0:
                self.setToolTip(f"Generating image. {count} jobs queued - click to cancel.")
            else:
                self.setToolTip(f"Generating image. Click to cancel.")
            count += 1
        else:
            self.setIcon(theme.icon("queue-inactive"))
            self.setToolTip("Idle.")
        self.setText(f"{count} ")

    def sizeHint(self) -> QSize:
        original = super().sizeHint()
        width = original.height() * 0.75 + self.fontMetrics().width(" 99 ") + 20
        return QSize(int(width), original.height())

    def paintEvent(self, a0):
        _paint_tool_drop_down(self, self.text())


class StyleSelectWidget(QWidget):
    _value: Style
    _styles: list[Style]

    value_changed = pyqtSignal(Style)
    quality_changed = pyqtSignal(SamplingQuality)

    def __init__(self, parent, show_quality=False):
        super().__init__(parent)
        self._value = Styles.list().default

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._combo = QComboBox(self)
        self.update_styles()
        self._combo.currentIndexChanged.connect(self.change_style)
        layout.addWidget(self._combo, 3)

        if show_quality:
            self._quality_combo = QComboBox(self)
            self._quality_combo.addItem("Fast", SamplingQuality.fast.value)
            self._quality_combo.addItem("Quality", SamplingQuality.quality.value)
            self._quality_combo.currentIndexChanged.connect(self.change_quality)
            layout.addWidget(self._quality_combo, 1)

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
        self._styles = filter_supported_styles(Styles.list().filtered(), comfy)
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

    def change_style(self):
        style = self._styles[self._combo.currentIndex()]
        if style != self._value:
            self._value = style
            self.value_changed.emit(style)

    def change_quality(self):
        quality = SamplingQuality(self._quality_combo.currentData())
        self.quality_changed.emit(quality)

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


class PromptAutoComplete:
    _completer: QCompleter

    def __init__(self, widget: QLineEdit):
        self._widget = widget
        self._completer = QCompleter()
        self._completer.activated.connect(self._insert_completion)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setWidget(widget)
        self._popup = ensure(self._completer.popup())

        self._refresh_loras()
        root.connection.state_changed.connect(self._refresh_loras)

    def _refresh_loras(self):
        if client := root.connection.client_if_connected:
            loras = [LoraId.normalize(lora).name for lora in client.models.loras]
            self._completer.setModel(QStringListModel(loras))

    def _current_text(self) -> str:
        text = self._widget.text()
        start = pos = self._widget.cursorPosition()
        while pos > 0 and text[pos - 1] not in " >\n":
            pos -= 1
        return text[pos:start]

    def check_completion(self):
        prefix = self._current_text()
        name = prefix.removeprefix("<lora:")
        if len(prefix) == len(name):
            self._popup.hide()
            return

        self._completer.setCompletionPrefix(name)
        rect = self._widget.cursorRect()
        self._popup.setCurrentIndex(ensure(self._completer.completionModel()).index(0, 0))
        scrollbar = ensure(self._popup.verticalScrollBar())
        rect.setWidth(self._popup.sizeHintForColumn(0) + scrollbar.sizeHint().width())
        self._completer.complete(rect)

    def _insert_completion(self, completion):
        text = self._widget.text()
        pos = self._widget.cursorPosition()
        prefix_len = len(self._completer.completionPrefix())
        text = text[: pos - prefix_len] + completion + ">" + text[pos:]
        self._widget.setText(text)
        self._widget.setCursorPosition(pos - prefix_len + len(completion) + 1)

    @property
    def is_active(self):
        return self._popup.isVisible()

    action_keys = [
        Qt.Key.Key_Enter,
        Qt.Key.Key_Return,
        Qt.Key.Key_Up,
        Qt.Key.Key_Down,
        Qt.Key.Key_Tab,
        Qt.Key.Key_Backtab,
    ]


class MultiLineTextPromptWidget(QPlainTextEdit):
    activated = pyqtSignal()

    _line_count = 2

    def __init__(self, parent):
        super().__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabChangesFocus(True)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.line_count = 2

        self._completer = PromptAutoComplete(self)
        self.textChanged.connect(self._completer.check_completion)

    def keyPressEvent(self, e: QKeyEvent | None):
        assert e is not None
        if self._completer.is_active and e.key() in PromptAutoComplete.action_keys:
            e.ignore()
            return

        handle_weight_adjustment(self, e)

        if e.key() == Qt.Key.Key_Return and e.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            self.activated.emit()
        else:
            super().keyPressEvent(e)

    @property
    def line_count(self):
        return self._line_count

    @line_count.setter
    def line_count(self, value: int):
        self._line_count = value
        fm = QFontMetrics(ensure(self.document()).defaultFont())
        self.setFixedHeight(fm.lineSpacing() * value + 8)

    def hasSelectedText(self) -> bool:
        return self.textCursor().hasSelection()

    def selectionStart(self) -> int:
        return self.textCursor().selectionStart()

    def selectionEnd(self) -> int:
        return self.textCursor().selectionEnd()

    def cursorPosition(self) -> int:
        return self.textCursor().position()

    def setCursorPosition(self, pos: int):
        cursor = self.textCursor()
        cursor.setPosition(pos)
        self.setTextCursor(cursor)

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: str):
        self.setPlainText(text)

    def setSelection(self, start: int, end: int):
        new_cursor = self.textCursor()
        new_cursor.setPosition(min(end, len(self.text())))
        new_cursor.setPosition(min(start, len(self.text())), QTextCursor.KeepAnchor)
        self.setTextCursor(new_cursor)


class SingleLineTextPromptWidget(QLineEdit):

    _completer: PromptAutoComplete

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._completer = PromptAutoComplete(self)
        self.textChanged.connect(self._completer.check_completion)
        self.setFrame(False)
        self.setStyleSheet(f"QLineEdit {{ background: transparent; }}")

    def keyPressEvent(self, a0: QKeyEvent | None):
        assert a0 is not None
        handle_weight_adjustment(self, a0)
        super().keyPressEvent(a0)


class TextPromptWidget(QFrame):
    """Wraps a single or multi-line text widget, with ability to switch between them.
    Using QPlainTextEdit set to a single line doesn't work properly because it still
    scrolls to the next line when eg. selecting and then looks like it's empty."""

    activated = pyqtSignal()
    text_changed = pyqtSignal(str)

    _line_count = 2
    _is_negative = False

    def __init__(self, line_count=2, is_negative=False, parent=None):
        super().__init__(parent)
        self._line_count = line_count
        self._is_negative = is_negative
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

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
        return self._multi.text() if self._line_count > 1 else self._single.text()

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
        for w in (self._multi, self._single):
            if not value:
                w.setPlaceholderText("Describe the content you want to see, or leave empty.")
            else:
                w.setPlaceholderText("Describe content you want to avoid.")

        if value:
            self.setContentsMargins(0, 2, 0, 2)
            self.setFrameStyle(QFrame.Shape.StyledPanel)
            self.setStyleSheet(f"QFrame {{ background: rgba(255, 0, 0, 15); }}")
        else:
            self.setFrameStyle(QFrame.Shape.NoFrame)

    @property
    def has_focus(self):
        return self._multi.hasFocus() or self._single.hasFocus()

    @has_focus.setter
    def has_focus(self, value: bool):
        if value:
            if self._line_count > 1:
                self._multi.setFocus()
            else:
                self._single.setFocus()

    def install_event_filter(self, obj: QObject):
        self._multi.installEventFilter(obj)
        self._single.installEventFilter(obj)

    def move_cursor_to_end(self):
        if self._line_count > 1:
            cursor = self._multi.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._multi.setTextCursor(cursor)
        else:
            self._single.setCursorPosition(len(self._single.text()))


class RegionThumbnailWidget(QLabel):
    _scale: float = 1.0

    def __init__(self, region: Region, parent: QWidget, scale=1.0):
        super().__init__(parent)
        self._scale = scale
        self.set_region(region)

    def set_region(self, region: Region):
        icon_size = int(self._scale * self.fontMetrics().height())
        if layer := region.layer:
            icon_image = QPixmap.fromImage(layer.thumbnail(icon_size, icon_size))
            icon_text = f"Text for region {layer.name()}"
        else:
            icon_image = theme.icon("root").pixmap(icon_size, icon_size)
            icon_text = "Text which is common to all regions."

        self.setPixmap(icon_image)
        self.setToolTip(icon_text)


class InactiveRegionWidget(QFrame):
    activated = pyqtSignal(Region)

    region: Region

    _text: str

    def __init__(self, region: Region, parent: QWidget):
        super().__init__(parent)
        self.region = region
        self._text = self.region.prompt.replace("\n", " ")

        self.setObjectName("InactiveRegionWidget")
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"QFrame#InactiveRegionWidget {{ background-color: {theme.base} }}")

        scale = 1.2 if region.is_root else 1.5
        icon = RegionThumbnailWidget(region, self, scale=scale)

        self._prompt = QLabel(self)
        self._prompt.setCursor(Qt.CursorShape.IBeamCursor)
        if self._text == "":
            if layer := region.layer:
                self._prompt.setText(f"{layer.name()} - click to add regional text")
            else:
                self._prompt.setText("Common text prompt - click to add content")
            self._prompt.setStyleSheet(f"QLabel {{ font-style: italic; color: {theme.grey}; }}")

        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(icon)
        layout.addWidget(self._prompt, 1)
        self.setLayout(layout)

        icon_size = int(1.2 * self.fontMetrics().height())
        for c in region.control:
            icon = theme.icon(f"control-{c.mode.name}")
            label = QLabel(self)
            label.setPixmap(icon.pixmap(icon_size, icon_size))
            layout.addWidget(label)

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        self.activated.emit(self.region)
        return super().mousePressEvent(a0)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        if self._text != "":
            theme.set_text_clipped(self._prompt, self.region.prompt.replace("\n", " "))
        return super().resizeEvent(a0)


class ActiveRegionWidget(QFrame):
    _style_base = f"QFrame#ActiveRegionWidget {{ background-color: {theme.base}; border: 1px solid {theme.line_base}; }}"
    _style_focus = f"QFrame#ActiveRegionWidget {{ background-color: {theme.base}; border: 1px solid {theme.active}; }}"

    _region: Region
    _bindings: list[QMetaObject.Connection]
    _max_lines: int

    def __init__(self, region: Region, parent: QWidget, max_lines=99):
        super().__init__(parent)
        self._region = region
        self._bindings = []
        self._max_lines = max_lines

        self.setObjectName("ActiveRegionWidget")
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(self._style_base)

        self._header_icon = RegionThumbnailWidget(region, self, scale=1.2)
        self._header_label = QLabel(self)
        self._header_label.setStyleSheet(f"font-style: italic; color: {theme.grey};")

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(2, 2, 2, 0)
        header_layout.addWidget(self._header_icon)
        header_layout.addWidget(self._header_label, 1)

        self._header = QWidget(self)
        self._header.setLayout(header_layout)

        self.positive = TextPromptWidget(parent=self)
        self.positive.line_count = min(settings.prompt_line_count, self._max_lines)
        self.positive.install_event_filter(self)

        self.negative = TextPromptWidget(line_count=1, is_negative=True, parent=self)
        self.negative.install_event_filter(self)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(self.positive)
        layout.addWidget(self.negative)
        self.setLayout(layout)

        self._setup_bindings(region)
        settings.changed.connect(self.update_settings)

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region: Region):
        if region != self._region:
            self._region = region
            self._setup_bindings(region)

    def _setup_bindings(self, region: Region):
        Binding.disconnect_all(self._bindings)
        self._bindings = [
            bind(region, "prompt", self.positive, "text"),
            bind(region, "negative_prompt", self.negative, "text"),
        ]
        self._header_icon.set_region(region)
        if layer := region.layer:
            self._header_label.setText(f"{layer.name()} - Regional text prompt")
        else:
            self._header_label.setText("Text prompt common to all regions")
        self.positive.move_cursor_to_end()
        self.negative.setVisible(region.is_root and settings.show_negative_prompt)

    def focus(self):
        if not (self.positive.has_focus or self.negative.has_focus):
            self.positive.has_focus = True

    @property
    def has_header(self):
        return self._header.isVisible()

    @has_header.setter
    def has_header(self, value: bool):
        self._header.setVisible(value)

    def update_settings(self, key: str, value):
        if key == "prompt_line_count":
            self.positive.line_count = min(value, self._max_lines)
        elif key == "show_negative_prompt":
            self.negative.text = ""
            self.negative.setVisible(value and self._region.is_root)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 and a1.type() == QEvent.Type.FocusIn:
            self.setStyleSheet(self._style_focus)
        elif a1 and a1.type() == QEvent.Type.FocusOut:
            self.setStyleSheet(self._style_base)
        return False


class RegionPromptWidget(QWidget):
    _regions: RegionTree
    _inactive_regions: list[InactiveRegionWidget]

    activated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._regions = root.active_model.regions
        self._inactive_regions = []

        self._prompt = ActiveRegionWidget(self._regions.active, self)
        self._prompt.positive.activated.connect(self.activated)
        self._prompt.negative.activated.connect(self.activated)
        self._prompt.has_header = False

        self._control = ControlListWidget(self._regions.active.control, parent=self)
        self._regions_above = QVBoxLayout()
        self._regions_below = QVBoxLayout()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(self._regions_above)
        layout.addWidget(self._prompt)
        layout.addLayout(self._regions_below)
        layout.addSpacing(4)
        layout.addWidget(self._control)
        self.setLayout(layout)

        self._update_active()

    @property
    def regions(self):
        return self._regions

    @regions.setter
    def regions(self, regions: RegionTree):
        if regions == self._regions:
            return
        self._regions = regions
        self._update_active()
        regions.active_changed.connect(self._setup_bindings)
        regions.added.connect(self._show_inactive_regions)
        regions.removed.connect(self._show_inactive_regions)

    def _update_active(self):
        self._setup_bindings(self._regions.active)

    def _setup_bindings(self, region: Region):
        self._prompt.region = region
        self._prompt.has_header = len(self._regions) > 0
        self._control.model = region.control
        self._show_inactive_regions()

    def _add_inactive_region(self, region: Region, layout: QVBoxLayout):
        widget = InactiveRegionWidget(region, self)
        widget.activated.connect(self._activate_region)
        self._inactive_regions.append(widget)
        layout.addWidget(widget)

    def _show_inactive_regions(self):
        active = self._regions.active

        for widget in self._inactive_regions:
            widget.deleteLater()
        self._inactive_regions.clear()

        below, above = active.siblings  # sorted from bottom to top
        for region in reversed(above):
            self._add_inactive_region(region, self._regions_above)
        for region in reversed(below):
            self._add_inactive_region(region, self._regions_below)
        if active is not self._regions.root:
            self._add_inactive_region(self._regions.root, self._regions_below)

    def _activate_region(self, region: Region):
        self._regions.active = region
        self._prompt.focus()


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
        Workspace.animation: theme.icon("workspace-animation"),
    }

    _value = Workspace.generation

    def __init__(self, parent):
        super().__init__(parent)

        menu = QMenu(self)
        menu.addAction(self._create_action("Generate", Workspace.generation))
        menu.addAction(self._create_action("Upscale", Workspace.upscaling))
        menu.addAction(self._create_action("Live", Workspace.live))
        menu.addAction(self._create_action("Animation", Workspace.animation))

        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setMenu(menu)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setToolTip(
            "Switch between workspaces: image generation, upscaling, live preview and animation."
        )
        self.setMinimumWidth(int(self.sizeHint().width() * 1.6))
        self.value = Workspace.generation

    def paintEvent(self, a0):
        _paint_tool_drop_down(self)

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


def create_wide_tool_button(icon_name: str, text: str, parent=None):
    button = QToolButton(parent)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    button.setIcon(theme.icon(icon_name))
    button.setToolTip(text)
    button.setAutoRaise(True)
    icon_height = button.iconSize().height()
    button.setIconSize(QSize(int(icon_height * 1.25), icon_height))
    return button


def _paint_tool_drop_down(widget: QToolButton, text: str | None = None):
    opt = QStyleOption()
    opt.initFrom(widget)
    painter = QPainter(widget)
    style = ensure(widget.style())
    rect = widget.rect()
    pixmap = widget.icon().pixmap(int(rect.height() * 0.75))
    element = QStyle.PrimitiveElement.PE_Widget
    if int(opt.state) & QStyle.StateFlag.State_MouseOver:
        element = QStyle.PrimitiveElement.PE_PanelButtonCommand
    style.drawPrimitive(element, opt, painter, widget)
    style.drawItemPixmap(
        painter,
        rect.adjusted(4, 0, 0, 0),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        pixmap,
    )
    if text:
        text_rect = rect.adjusted(pixmap.width() + 4, 0, 0, 0)
        style.drawItemText(
            painter, text_rect, Qt.AlignmentFlag.AlignVCenter, widget.palette(), True, text
        )
    painter.translate(int(0.5 * rect.width() - 10), 0)
    style.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorArrowDown, opt, painter)
