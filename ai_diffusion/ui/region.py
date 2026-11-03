from __future__ import annotations

import asyncio
import time
from enum import Enum
from functools import partial

from PyQt5.QtCore import QEvent, QMetaObject, QObject, QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QImage,
    QMouseEvent,
    QPainter,
    QPixmap,
    QResizeEvent,
    QTextCursor,
)
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..backend import ollama
from ..backend.client import Client
from ..backend.network import NetworkError
from ..backend.ollama import EnhanceTask
from ..document import LayerType
from ..image import Bounds, Extent, Image
from ..localization import translate as _
from ..model.properties import Binding, bind
from ..model.region import Region, RegionLink, RootRegion, translate_prompt
from ..model.root import root
from ..text import char16_index_to_str_index, format_option_group, str_index_to_char16_index
from ..util import client_logger as log
from ..util import ensure
from . import theme
from .control import ControlListWidget
from .lora_picker import LoraPickerDialog
from .settings import settings
from .widget import TextPromptWidget


class InactiveRegionWidget(QFrame):
    activated = pyqtSignal(Region)

    region: RootRegion | Region

    _text: str

    def __init__(self, region: RootRegion | Region, parent: QWidget):
        super().__init__(parent)
        self.region = region
        self._text = self.region.positive.replace("\n", " ")

        self.setObjectName("InactiveRegionWidget")
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"QFrame#InactiveRegionWidget {{ background-color: {theme.base} }}")

        thumbnail = RegionThumbnailWidget(region, self)

        self._prompt = QLabel(self)
        self._prompt.setCursor(Qt.CursorShape.IBeamCursor)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 5, 0)
        layout.addWidget(thumbnail, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._prompt, 1)
        self.setLayout(layout)

        icon_size = int(1.2 * self.fontMetrics().height())
        for c in region.control:
            icon = theme.icon(f"control-{c.mode.name}")
            label = QLabel(self)
            label.setPixmap(icon.pixmap(icon_size, icon_size))
            layout.addWidget(label)

        if self._text == "":
            self._prompt.setStyleSheet(f"QLabel {{ font-style: italic; color: {theme.grey}; }}")
            if isinstance(region, Region):
                self._text = f"{region.name} - click to add regional text"
            else:
                self._text = _("Common text prompt - click to add content")

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        self.activated.emit(self.region)
        return super().mousePressEvent(a0)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        theme.set_text_clipped(self._prompt, self._text)
        return super().resizeEvent(a0)


# Vision models see a fixed number of image tokens anyway, and a full 4k canvas would
# blow up the base64 payload for no gain.
describe_max_extent = Extent(1024, 1024)


def _canvas_image(model, require_selection: bool = False) -> str | None:
    """The current canvas as base64 PNG, cropped to the selection if there is one.
    With require_selection, returns None instead of silently falling back to the
    whole canvas when nothing is selected."""
    doc = model.document
    bounds = doc.selection_bounds
    if require_selection and bounds is None:
        return None
    image = doc.get_image(bounds)
    if image.extent.width > describe_max_extent.width or (
        image.extent.height > describe_max_extent.height
    ):
        image = Image.scale_to_fit(image, describe_max_extent)
    return image.to_base64()


class PromptHeader(Enum):
    none = 0
    icon = 1
    full = 2


class ActiveRegionWidget(QFrame):
    _style_base = f"QFrame#ActiveRegionWidget {{ background-color: {theme.base}; border: 1px solid {theme.line_base}; }}"
    _style_focus = f"QFrame#ActiveRegionWidget {{ background-color: {theme.base}; border: 1px solid {theme.active}; }}"

    focused = pyqtSignal()

    def __init__(self, root: RootRegion, parent: QWidget, header=PromptHeader.full):
        super().__init__(parent)
        self._root = root
        self._region: RootRegion | Region | None = root
        self._bindings: list[QMetaObject.Connection] = []
        self._header_style = header
        self._translation_enabled = True
        self._is_slim = False

        self.setObjectName("ActiveRegionWidget")
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(self._style_base)

        self._header_icon = RegionThumbnailWidget(self._region, self)
        self._header_label = QLabel(self)
        self._header_label.setStyleSheet(f"font-style: italic; color: {theme.grey};")

        self._link_button = QToolButton(self)
        self._link_button.setIcon(theme.icon("link"))
        self._link_button.setAutoRaise(True)

        self._remove_button = QToolButton(self)
        self._remove_button.setIcon(theme.icon("remove"))
        self._remove_button.setAutoRaise(True)
        self._remove_button.setToolTip(_("Remove this region"))

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 2, 0)
        header_layout.setSpacing(0)
        header_layout.addWidget(self._header_icon)
        header_layout.addSpacing(5)
        header_layout.addWidget(self._header_label, 1)
        header_layout.addWidget(self._link_button)
        header_layout.addWidget(self._remove_button)

        self._header = QWidget(self)
        self._header.setLayout(header_layout)

        self.positive = TextPromptWidget(parent=self)
        self.positive.handle_dragged.connect(self._handle_positive_dragging)
        self.positive.installEventFilter(self)

        self.negative = TextPromptWidget(line_count=1, is_negative=True, parent=self)
        self.negative.handle_dragged.connect(self._handle_negative_dragging)
        self.negative.installEventFilter(self)

        self._no_region = QWidget(self)
        self._no_region.setVisible(False)

        self._no_region_label = QLabel(_("Active layer is not linked to a region"), self._no_region)
        self._no_region_label.setStyleSheet(f"font-style: italic; color: {theme.grey};")

        self._new_region_button = QToolButton(self._no_region)
        self._new_region_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._new_region_button.setIcon(theme.icon("region-add"))
        self._new_region_button.setText(_("New region"))

        self._link_region_button = QToolButton(self._no_region)
        self._link_region_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._link_region_button.setIcon(theme.icon("link"))
        self._link_region_button.setText(_("Link region"))
        self._link_region_button.clicked.connect(self._show_link_menu)

        no_region_layout = QHBoxLayout()
        no_region_layout.setContentsMargins(4, 1, 4, 1)
        no_region_layout.addWidget(self._no_region_label, 1)
        no_region_layout.addWidget(self._new_region_button)
        no_region_layout.addWidget(self._link_region_button)
        self._no_region.setLayout(no_region_layout)

        self._lora_browse_button = QToolButton(self)
        self._lora_browse_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._lora_browse_button.setIcon(theme.icon("lora"))
        self._lora_browse_button.setText(_("LoRA"))
        self._lora_browse_button.setToolTip(_("Browse LoRAs and search CivitAI"))
        self._lora_browse_button.setAutoRaise(True)
        self._lora_browse_button.clicked.connect(self._open_lora_picker)
        self._lora_dialog: LoraPickerDialog | None = None

        self._recipe_browse_button = QToolButton(self)
        self._recipe_browse_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._recipe_browse_button.setIcon(theme.icon("recipe"))
        self._recipe_browse_button.setText(_("Recipe"))
        self._recipe_browse_button.setToolTip(_("Browse and apply recipes (Lora Manager)"))
        self._recipe_browse_button.setAutoRaise(True)
        self._recipe_browse_button.clicked.connect(self._open_recipe_picker)
        self._recipe_dialog = None

        self._wildcard_browse_button = QToolButton(self)
        self._wildcard_browse_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._wildcard_browse_button.setIcon(theme.icon("random"))
        self._wildcard_browse_button.setText(_("Wildcards"))
        self._wildcard_browse_button.setToolTip(_("Browse file-based wildcards (__name__)"))
        self._wildcard_browse_button.setAutoRaise(True)
        self._wildcard_browse_button.clicked.connect(self._open_wildcard_picker)
        self._wildcard_dialog = None

        self._animadex_button = QToolButton(self)
        self._animadex_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._animadex_button.setIcon(theme.icon("sd-version-anima"))
        self._animadex_button.setText(_("AnimaDex"))
        self._animadex_button.setToolTip(_("Browse AnimaDex characters and artists for Anima"))
        self._animadex_button.setAutoRaise(True)
        self._animadex_button.clicked.connect(self._open_animadex_picker)
        self._animadex_button.setVisible(False)  # see _update_animadex_visibility
        self._animadex_dialog = None

        self._prompt_browse_button = QToolButton(self)
        self._prompt_browse_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._prompt_browse_button.setIcon(theme.icon("region-prompt"))
        self._prompt_browse_button.setText(_("Prompts"))
        self._prompt_browse_button.setToolTip(_("Browse and reuse saved prompt snippets"))
        self._prompt_browse_button.setAutoRaise(True)
        self._prompt_browse_button.clicked.connect(self._open_prompt_picker)
        self._prompt_dialog = None

        self._enhance_button = QToolButton(self)
        self._enhance_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._enhance_button.setIcon(theme.icon("enhance"))
        self._enhance_button.setText(_("AI"))
        self._enhance_button.setAutoRaise(True)
        self._enhance_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._enhance_button.setMenu(self._create_enhance_menu())
        self._enhance_button.clicked.connect(self._enhance_clicked)
        self._enhance_button.setVisible(settings.ollama_enabled)
        self._enhance_backup: list[str] = []
        self._enhance_backup_limit = 20
        self._last_instruction = ""
        self._instruction_to_negative = False
        self._enhance_running = False
        self._enhance_job: ollama.Generation | None = None
        self._enhance_task = EnhanceTask.enhance
        self._enhance_started = 0.0
        self._update_enhance_tooltip()

        self._enhance_progress = QLabel(self)
        self._enhance_progress.setVisible(False)
        self._enhance_progress.setStyleSheet(f"font-style: italic; color: {theme.grey};")
        self._enhance_timer = QTimer(self)
        self._enhance_timer.setInterval(200)
        self._enhance_timer.timeout.connect(self._update_enhance_progress)

        # left-aligned, grouped by what they do: insert saved/dynamic content first
        # (Prompts -> Wildcards -> Recipe -> LoRA, increasing in scope from a single
        # snippet to a whole checkpoint+LoRA+prompt bundle), AI last since it acts on
        # whatever text is already in the field rather than inserting new content
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(2)
        buttons_layout.addWidget(self._prompt_browse_button)
        buttons_layout.addWidget(self._wildcard_browse_button)
        buttons_layout.addWidget(self._animadex_button)
        buttons_layout.addWidget(self._recipe_browse_button)
        buttons_layout.addWidget(self._lora_browse_button)
        buttons_layout.addWidget(self._enhance_button)
        buttons_layout.addStretch(1)

        prompt_tools_layout = QVBoxLayout()
        prompt_tools_layout.setContentsMargins(0, 0, 0, 0)
        prompt_tools_layout.setSpacing(0)
        prompt_tools_layout.addLayout(buttons_layout)
        prompt_tools_layout.addWidget(self._enhance_progress)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if header is PromptHeader.icon:
            self._header.setVisible(False)
            layout.addLayout(prompt_tools_layout)
            positive_layout = QHBoxLayout()
            positive_layout.addWidget(self._header_icon, alignment=Qt.AlignmentFlag.AlignTop)
            positive_layout.addWidget(self.positive, 1)
            layout.addLayout(positive_layout)
        else:
            layout.addWidget(self._header)
            layout.addLayout(prompt_tools_layout)
            layout.addWidget(self.positive)
        layout.addWidget(self.negative)
        layout.addWidget(self._no_region)
        self.setLayout(layout)

        font_size = self.font().pointSize()
        self._language_button = QToolButton(self)
        self._language_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._language_button.setText(settings.prompt_translation.upper())
        self._language_button.setStyleSheet(
            f"QToolButton {{ font-size: {max(6, font_size - 2)}pt; background: #40808080;"
            " border: 1px solid #60808080; border-radius: 2px; }"
        )
        self._language_button.clicked.connect(self._toggle_translation_enabled)

        font_height = QFontMetrics(self.font()).height()
        self._negative_warning = QLabel(self)
        self._negative_warning.setPixmap(theme.icon("alert").pixmap(font_height, font_height))
        self._negative_warning.setToolTip(_("The selected Style does not use the negative prompt."))
        self._negative_warning.setVisible(False)

        self._setup_bindings(self._region)
        settings.changed.connect(self.update_settings)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root: RootRegion):
        self._root = root

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region: RootRegion | Region | None):
        if region != self._region:
            self._region = region
            self._setup_bindings(region)
            # an undo stack from a different region's text makes no sense here
            self._enhance_backup.clear()
            if self._revert_action is not None:
                self._revert_action.setEnabled(False)

    def _setup_bindings(self, region: RootRegion | Region | None):
        Binding.disconnect_all(self._bindings)
        is_root_region = isinstance(region, RootRegion)
        if is_root_region:
            self._root = region
            self._bindings = [
                bind(region, "positive", self.positive, "text"),
                bind(region, "negative", self.negative, "text"),
            ]
            if self.is_slim:
                evt = region.negative_enabled_live_changed
            else:
                evt = region.negative_enabled_changed
            self._bindings.append(evt.connect(self._show_negative_warning))
            self._show_negative_warning()
        elif isinstance(region, Region):
            self._root = region.root
            self._bindings = [
                bind(region, "positive", self.positive, "text"),
                region.layer_ids_changed.connect(self._update_links),
                self._link_button.clicked.connect(region.toggle_active_link),
                self._remove_button.clicked.connect(region.remove),
            ]
        else:  # Active layer is not linked to a region
            self._bindings = [self._root.layers.active_changed.connect(self._update_actions)]
            self._update_actions()
        self._bindings += [
            self._root.active_layer_changed.connect(self._update_links),
            self._new_region_button.clicked.connect(self._root.create_region_layer),
            self._root._model.translation_enabled_changed.connect(self._update_language),
            self._root._model.style_changed.connect(self._update_animadex_visibility),
            self._root._model.edit_mode_changed.connect(self._update_animadex_visibility),
            self._root._model.workspace_changed.connect(self._update_animadex_visibility),
            root.connection.models_changed.connect(self._update_animadex_visibility),
        ]
        self._update_animadex_visibility()
        self._update_header()
        self._update_links()
        self._update_language()
        self._update_prompt_widgets()
        self.positive.move_cursor_to_end()
        self._link_button.setVisible(not is_root_region)
        self._remove_button.setVisible(not is_root_region)
        self.positive.setVisible(region is not None)
        self._no_region.setVisible(region is None)

    def focus(self):
        if not (self.positive.has_focus or self.negative.has_focus):
            self.positive.has_focus = True

    @property
    def header_style(self):
        return self._header_style

    @header_style.setter
    def header_style(self, value: PromptHeader):
        if value is self._header_style:
            return
        self._header_style = value
        self._update_header()

    def _update_header(self):
        style = self._header_style
        self._header.setVisible(len(self._root) > 0 and style is PromptHeader.full)
        self._header_icon.setVisible(self.region is not None and style is not PromptHeader.none)

    def _update_links(self):
        if isinstance(self._region, RootRegion):
            self._header_label.setText(_("Text prompt common to all regions"))
            self._header_icon.set_region(self._region)
        elif isinstance(self._region, Region):
            theme.set_text_clipped(
                self._header_label, f"{self._region.name} - " + _("Regional text prompt")
            )
            active_layer = self._root.layers.active
            link_enabled = False
            if self._region.is_linked(active_layer, RegionLink.direct):
                icon = "link-active"
                desc = _("Active layer is linked to this region - click to unlink")
                link_enabled = True
            elif self._root.is_linked(active_layer, RegionLink.indirect):
                icon = "link"
                desc = _("Active layer is linked to this region via a group layer")
            elif active_layer.type not in [LayerType.paint, LayerType.group]:
                icon = "link-disabled"
                desc = _("Only paint layers and groups can be linked to regions")
            elif self._root.is_linked(active_layer, RegionLink.direct):
                icon = "link-disabled"
                desc = _("Active layer is already linked to another region")
            elif Region.link_target(active_layer) is not active_layer:
                icon = "link-disabled"
                desc = _("Active layer is part of a group - select the group layer to link it")
            else:
                icon = "link-off"
                desc = _("Active layer is not linked - click to link it to this region")
                link_enabled = True
            self._link_button.setIcon(theme.icon(icon))
            self._link_button.setEnabled(link_enabled)
            self._link_button.setToolTip(desc)
            self._header_icon.set_region(self._region)

    def _update_actions(self):
        active_layer = self._root.layers.active
        can_link = active_layer.type in [LayerType.paint, LayerType.group]
        self._new_region_button.setEnabled(can_link)
        self._link_region_button.setEnabled(can_link)
        if can_link:
            self._no_region_label.setText(_("Active layer is not linked to a region"))
        else:
            self._no_region_label.setText(_("Active layer cannot be linked to a region"))

    def _show_link_menu(self):
        active_layer = self._root.layers.active
        menu = QMenu()
        for region in self._root:
            if region is not self._region:
                name = region.positive.replace("\n", " ")
                if name == "":
                    name = _("<No text prompt>")
                if len(name) > 20:
                    name = name[:17] + "..."

                def link(r: Region):
                    r.link(active_layer)
                    self.region = r

                action = ensure(menu.addAction(name))
                action.triggered.connect(partial(link, region))

        pos = self._link_region_button.rect().bottomLeft()
        menu.exec_(self._link_region_button.mapToGlobal(pos))

    @property
    def is_slim(self):
        return self._is_slim

    @is_slim.setter
    def is_slim(self, value: bool):
        if value == self._is_slim:
            return
        self._is_slim = value
        self._update_prompt_widgets()

    @property
    def has_negative(self):
        return settings.show_negative_prompt and isinstance(self._region, RootRegion)

    def update_settings(self, key: str, value):
        if key in {
            "prompt_line_count",
            "prompt_line_count_live",
            "negative_prompt_line_count",
            "negative_prompt_line_count_live",
        }:
            self._update_prompt_widgets()
        elif key == "show_negative_prompt":
            self.negative.text = ""
            self._update_prompt_widgets()
        elif key == "prompt_translation":
            self._update_language()
        elif key == "ollama_enabled":
            self._enhance_button.setVisible(value)
        elif key == "animadex_enabled":
            self._update_animadex_visibility()
        elif key == "ollama_model":
            self._update_enhance_tooltip()

    async def _replace_with_translation(self, client: Client):
        region = self.region
        if region is None:
            return
        if positive := region.positive:
            translated = await client.translate(positive, settings.prompt_translation)
            if self.region is region and positive == region.positive:
                region.positive = translated
        if isinstance(region, RootRegion) and region.negative:
            negative = region.negative
            translated = await client.translate(negative, settings.prompt_translation)
            if self.region is region and negative == region.negative:
                region.negative = translated

    def _toggle_translation_enabled(self):
        model = self._root._model
        ctrl_down = QGuiApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier
        if model.translation_enabled and bool(ctrl_down) and self.region is not None:
            translate_prompt(self.region)
        model.translation_enabled = not model.translation_enabled

    _lang_help_enabled = _(
        "Prompt translation is active! Click to disable and switch to original input."
    )
    _lang_help_disabled = _(
        "Translation is disabled. Click to enable prompt translation from your language to English"
    )
    _lang_help_translate = _("Use Ctrl+Click to replace the text with a translation immediately.")

    def _update_language(self):
        self._language_button.setVisible(bool(settings.prompt_translation))
        if settings.prompt_translation:
            enabled = self._root._model.translation_enabled
            lang = settings.prompt_translation if enabled else "en"
            self._language_button.setText(lang.upper())
            if enabled:
                text = self._lang_help_enabled
                if (client := root.connection.client_if_connected) and client.features.translation:
                    text += "\n" + self._lang_help_translate
            else:
                text = self._lang_help_disabled
            self._language_button.setToolTip(text)

    def _update_prompt_widgets(self):
        if self.has_negative:
            self.negative.line_count = (
                settings.negative_prompt_line_count_live
                if self.is_slim
                else settings.negative_prompt_line_count
            )
        if not self.is_slim:
            self.positive.line_count = settings.prompt_line_count
        elif isinstance(self._region, Region):
            self.positive.line_count = 1
        elif self.has_negative:
            self.positive.line_count = max(
                1, settings.prompt_line_count_live - self.negative.line_count
            )
        else:
            self.positive.line_count = settings.prompt_line_count_live
        self.negative.setVisible(self.has_negative)
        self._layout_language_button()
        self._setup_resize_handle()
        self._show_negative_warning()

    def _layout_language_button(self):
        if settings.prompt_translation:
            pos = self.positive.geometry().bottomRight()
            if self.has_negative:
                pos = self.negative.geometry().bottomRight()
            s = QSize(self.fontMetrics().width("EN"), self.fontMetrics().height())
            self._language_button.move(pos.x() - s.width() - 2, pos.y() - s.height() - 2)
            self._language_button.resize(s)

        if self.has_negative:
            pos = self.negative.geometry().bottomRight()
            if settings.prompt_translation:
                pos = pos - QPoint(self._language_button.width() + 4, 0)
            s = self.fontMetrics().height() + 2
            self._negative_warning.move(pos.x() - s - 2, pos.y() - s - 2)
            self._negative_warning.resize(QSize(s, s))

    def _show_negative_warning(self):
        if isinstance(self._region, RootRegion):
            r = self._region
            enabled = r.negative_enabled_live if self.is_slim else r.negative_enabled
            self._negative_warning.setVisible(settings.show_negative_prompt and not enabled)
        else:
            self._negative_warning.setVisible(False)

    def _setup_resize_handle(self):
        can_resize = not (isinstance(self._region, Region) and self.is_slim)
        self.positive.is_resizable = can_resize
        self.negative.is_resizable = self.has_negative and can_resize

    def _handle_positive_dragging(self, y_pos: int):
        new_line_count = self._drag_to_line_count(self.positive, y_pos)
        if new_line_count is not None:
            if self.is_slim:
                settings.prompt_line_count_live = new_line_count
            else:
                settings.prompt_line_count = new_line_count
            self._update_prompt_widgets()

    def _handle_negative_dragging(self, y_pos: int):
        new_line_count = self._drag_to_line_count(self.negative, y_pos)
        if new_line_count is not None:
            if self.is_slim:
                settings.negative_prompt_line_count_live = new_line_count
            else:
                settings.negative_prompt_line_count = new_line_count
            self._update_prompt_widgets()

    def _drag_to_line_count(self, widget: TextPromptWidget, y_pos: int):
        new_height = y_pos - 5
        fm = QFontMetrics(ensure(widget.document()).defaultFont())
        new_line_count = round(new_height / fm.lineSpacing())
        if 1 <= new_line_count <= theme.prompt_max_line_count:
            return new_line_count
        return None

    def _create_enhance_menu(self):
        menu = theme.style_menu(QMenu(self))
        menu.addAction(_("Enhance"), partial(self._enhance, EnhanceTask.enhance))
        self._enhance_selection_action = ensure(
            menu.addAction(
                _("Enhance selection"), partial(self._enhance, EnhanceTask.enhance, "", True)
            )
        )
        self._enhance_selection_action.setVisible(False)
        # Grouped by kind - prompt rewrites, image captions, dialogs, undo - and alphabetical
        # within a group. Enhance stays on top (it is what the button does) and Revert last.
        menu.addSeparator()
        menu.addAction(_("Add detail only"), partial(self._enhance, EnhanceTask.detail))
        menu.addAction(_("Rewrite from scratch"), partial(self._enhance, EnhanceTask.rewrite))
        menu.addAction(
            _("Split into sections (character, clothing, ...)"),
            partial(self._enhance, EnhanceTask.structure),
        )
        menu.addAction(
            _("Variations as sequential wildcard"),
            partial(self._enhance, EnhanceTask.variations),
        )
        menu.addSeparator()
        menu.addAction(_("Describe the image"), partial(self._enhance, EnhanceTask.describe))
        self._describe_selection_action = ensure(
            menu.addAction(
                _("Describe current selection"),
                partial(self._enhance, EnhanceTask.describe, "", False, True),
            )
        )
        self._describe_selection_action.setVisible(False)
        menu.addSeparator()
        menu.addAction(_("Modify with instruction..."), self._ask_instruction)
        menu.addAction(_("Prompt batch for generation..."), self._open_prompt_batch)
        menu.addSeparator()
        self._revert_action = ensure(menu.addAction(_("Revert"), self._revert_enhance))
        self._revert_action.setEnabled(False)
        menu.aboutToShow.connect(self._update_enhance_selection_visibility)
        return menu

    def _update_enhance_selection_visibility(self):
        # only shown when there's something to send instead of the whole prompt/canvas
        self._enhance_selection_action.setVisible(self.positive.textCursor().hasSelection())
        model = root.active_model
        has_canvas_selection = model is not None and model.document.selection_bounds is not None
        self._describe_selection_action.setVisible(has_canvas_selection)

    def _update_enhance_tooltip(self):
        model = settings.ollama_model or _("not configured")
        self._enhance_button.setToolTip(
            _("Rewrite the prompt with a local language model") + f" ({model})"
        )

    def _revert_enhance(self):
        if self._enhance_backup and self.region is not None:
            self.region.positive = self._enhance_backup.pop()
            self._revert_action.setEnabled(bool(self._enhance_backup))

    def _open_prompt_batch(self):
        """Build a pool of prompts before generating, so a batch or a generate loop
        runs through many different prompts instead of repeating one."""
        from .prompt_batch import PromptBatchDialog

        if self._enhance_running or self.region is None:
            return
        dialog = PromptBatchDialog(self.region, self)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _ask_instruction(self):
        """Ask for a free-form change to apply to the current prompt, eg. "make it
        night", "add rain, remove the hat"."""
        if self._enhance_running:
            return
        dialog = InstructionDialog(
            self._last_instruction, self, self.has_negative, self._instruction_to_negative
        )
        if dialog.exec_() == QDialog.DialogCode.Accepted and dialog.text:
            self._last_instruction = dialog.text
            if self.has_negative:
                self._instruction_to_negative = dialog.to_negative
            self._enhance(
                EnhanceTask.instruct,
                self._last_instruction,
                to_negative=self.has_negative and dialog.to_negative,
            )

    def _enhance_clicked(self):
        if self._enhance_running:
            self._cancel_enhance()
        else:
            self._enhance(EnhanceTask.enhance)

    def _cancel_enhance(self):
        if self._enhance_job is not None:
            self._enhance_job.cancel()

    def _begin_enhance_progress(self, task: EnhanceTask):
        self._enhance_started = time.monotonic()
        self._enhance_task = task
        self._enhance_button.setIcon(theme.icon("cancel"))
        self._enhance_button.setText(_("Stop"))
        self._enhance_button.setToolTip(_("Stop the running prompt generation"))
        self._enhance_progress.setVisible(True)
        self._enhance_timer.start()
        self._update_enhance_progress()

    def _end_enhance_progress(self):
        self._enhance_timer.stop()
        self._enhance_progress.setVisible(False)
        self._enhance_button.setIcon(theme.icon("enhance"))
        self._enhance_button.setText(_("AI"))
        self._update_enhance_tooltip()

    def _update_enhance_progress(self):
        elapsed = time.monotonic() - self._enhance_started
        words = len(self._enhance_job.text.split()) if self._enhance_job else 0
        if self._enhance_task is EnhanceTask.describe:
            text = _("Looking at the image...") + f" {elapsed:.0f}s"
        else:
            text = _("Writing prompt...") + f" {elapsed:.0f}s"
        if words > 0:
            text += f" - {words} " + _("words")
        elif elapsed > 3:  # nothing yet: the model is most likely still loading into VRAM
            text += " - " + _("loading model")
        theme.set_text_clipped(self._enhance_progress, text)

    def _enhance(
        self,
        task: EnhanceTask,
        instruction: str = "",
        use_selection: bool = False,
        canvas_selection: bool = False,
        to_negative: bool = False,
    ):
        if self._enhance_running:
            return
        if not settings.ollama_model:
            self._report_error(
                _("No language model selected. Configure one in Settings -> Prompt AI.")
            )
            return
        eventloop.run(
            self._run_enhance(task, instruction, use_selection, canvas_selection, to_negative)
        )

    def _report_error(self, message: str):
        if model := root.active_model:
            model.report_error(message)
        else:
            log.error(message)

    async def _run_enhance(
        self,
        task: EnhanceTask,
        instruction: str = "",
        use_selection: bool = False,
        canvas_selection: bool = False,
        to_negative: bool = False,
    ):
        region = self.region
        model = root.active_model
        if region is None or model is None:
            return
        if canvas_selection and model.document.selection_bounds is None:
            self._report_error(_("No selection on the canvas"))
            return

        original = region.positive
        selection: tuple[int, int] | None = None
        if use_selection:
            cursor = self.positive.textCursor()
            if not cursor.hasSelection():
                return
            start = char16_index_to_str_index(original, cursor.selectionStart())
            end = char16_index_to_str_index(original, cursor.selectionEnd())
            selection = (start, end)
            source = original[start:end]
        else:
            source = original

        # Like Generate: a new attempt replaces the last error instead of leaving it
        # standing (and pushing the history down) after the problem is gone.
        model.clear_error()
        self._enhance_running = True
        self._begin_enhance_progress(task)
        try:
            family = model.active_style.effective_family(model.arch)
            profile = ollama.Profiles.instance().for_family(family)
            if profile is None:
                self._report_error(_("No prompt profile found for") + f" {family}")
                return

            protected = ollama.protect(source)
            count = max(2, settings.ollama_variation_count)
            to_negative = to_negative and isinstance(region, RootRegion)
            request = ollama.build_prompt(task, protected.text, count, instruction, to_negative)
            images = None
            if task is EnhanceTask.describe:  # selection presence was checked on entry
                images = [ensure(_canvas_image(model, canvas_selection))]

            if settings.ollama_free_comfy_vram:
                if client := root.connection.client_if_connected:
                    await ollama.free_comfy_vram(client)

            self._enhance_job = ollama.Generation()
            response = await self._enhance_job.run(
                request,
                system=(
                    ollama.structure_system_prompt()
                    if task is EnhanceTask.structure
                    else profile.system_for(task)
                ),
                model=profile.model_for(task),
                temperature=ollama.temperature_for(task),
                images=images,
            )
            if not response:
                self._report_error(_("The language model returned an empty response"))
                return

            negative_terms = ""
            if to_negative:
                response, negative_terms = ollama.split_negative(response)
                if not response:
                    self._report_error(_("The language model returned an empty response"))
                    return

            if task is EnhanceTask.variations:
                lines = [line.strip(" -*\t") for line in response.splitlines()]
                lines = [line for line in lines if line][:count]
                if len(lines) < 2:
                    self._report_error(_("The language model did not return variations"))
                    return
                result = format_option_group(lines, sequential=True)
            elif task is EnhanceTask.structure:
                result = ollama.format_sections(response, protected.text, protected.tokens)
                protected = ollama.ProtectedPrompt(result, [])  # already in their sections
            elif task is EnhanceTask.detail and not selection:
                result = f"{source.rstrip(' ,')}, {response}" if source.strip() else response
                protected = ollama.ProtectedPrompt(result, [])  # tokens are still in `source`
            else:
                result = response
            result = protected.restore(result)

            if self.region is not region or region.positive != original:
                return  # the prompt changed underneath us while the model was writing

            if selection:
                start, end = selection
                new_text = original[:start] + result + original[end:]
                region.positive = new_text
                cursor = self.positive.textCursor()
                cursor.setPosition(str_index_to_char16_index(new_text, start))
                cursor.setPosition(
                    str_index_to_char16_index(new_text, start + len(result)),
                    QTextCursor.MoveMode.KeepAnchor,
                )
                self.positive.setTextCursor(cursor)
            else:
                region.positive = result
            if negative_terms and isinstance(region, RootRegion):
                region.negative = ollama.merge_negative(region.negative, negative_terms)

            if len(self._enhance_backup) >= self._enhance_backup_limit:
                self._enhance_backup.pop(0)
            self._enhance_backup.append(original)
            self._revert_action.setEnabled(True)
        except asyncio.CancelledError:
            pass  # stopped by the user
        except NetworkError as e:
            self._report_error(_("Could not reach Ollama") + f" ({ollama.url()}): {e}")
        except Exception as e:
            log.exception("Prompt enhancement failed")
            self._report_error(_("Prompt enhancement failed") + f": {e}")
        finally:
            self._enhance_running = False
            self._enhance_job = None
            self._end_enhance_progress()

    def _open_lora_picker(self):
        if self._lora_dialog is None:
            model = root.active_model
            arch = model.arch.name if model else ""
            self._lora_dialog = LoraPickerDialog(arch, parent=self)
        self._lora_dialog.show()
        self._lora_dialog.raise_()
        self._lora_dialog.activateWindow()

    def _open_recipe_picker(self):
        from .recipe_picker import RecipePickerDialog

        if self._recipe_dialog is None:
            model = root.active_model
            arch = model.arch.name if model else ""
            self._recipe_dialog = RecipePickerDialog(arch, parent=self)
        self._recipe_dialog.show()
        self._recipe_dialog.raise_()
        self._recipe_dialog.activateWindow()

    def _open_wildcard_picker(self):
        from .wildcard_picker import WildcardPickerDialog

        if self._wildcard_dialog is None:
            self._wildcard_dialog = WildcardPickerDialog(parent=self)
        self._wildcard_dialog.show()
        self._wildcard_dialog.raise_()
        self._wildcard_dialog.activateWindow()

    def _update_animadex_visibility(self, *_args):
        # AnimaDex triggers are made for Anima; with any other style they're noise.
        # The arch comes from the style's checkpoint, which the server's model list
        # can change, hence the connection signal too.
        try:
            is_anima = self._root._model.arch.name == "anima"
        except Exception:
            is_anima = False
        self._animadex_button.setVisible(settings.animadex_enabled and is_anima)

    def _open_animadex_picker(self):
        from .animadex_picker import AnimadexPickerDialog

        if self._animadex_dialog is None:
            self._animadex_dialog = AnimadexPickerDialog(parent=self)
        self._animadex_dialog.show()
        self._animadex_dialog.raise_()
        self._animadex_dialog.activateWindow()

    def _open_prompt_picker(self):
        dialog = self._ensure_prompt_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _ensure_prompt_dialog(self):
        from .prompt_picker import PromptPickerDialog

        if self._prompt_dialog is None:
            self._prompt_dialog = PromptPickerDialog(parent=self)
        return self._prompt_dialog

    def open_prompt_picker_with(self, text: str, negative: str, image):
        """Open the Prompt Picker with a new entry pre-filled from a history image -
        only the title needs typing. See HistoryWidget.save_as_prompt_requested."""
        dialog = self._ensure_prompt_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        dialog.browser.create_from_image(text, negative, image)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self._layout_language_button()

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 and a1.type() == QEvent.Type.FocusIn:
            self.setStyleSheet(self._style_focus)
            self.focused.emit()
        elif a1 and a1.type() == QEvent.Type.FocusOut:
            self.setStyleSheet(self._style_base)
        return False


class InstructionDialog(QDialog):
    """Small prompt for a free-form modification of the current text prompt. Kept
    deliberately compact: it is opened often and holds one short sentence."""

    def __init__(
        self,
        text: str,
        parent: QWidget,
        offer_negative: bool = False,
        to_negative: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle(_("Modify Prompt"))
        self.setModal(True)

        label = QLabel(_("Change to apply:"), self)

        self._negative_check = QCheckBox(_("Move exclusions to negative prompt"), self)
        self._negative_check.setToolTip(
            _('Things the change removes or avoids ("no hat") are added to the negative prompt')
        )
        self._negative_check.setChecked(to_negative)
        self._negative_check.setVisible(offer_negative)

        self._edit = QPlainTextEdit(text, self)
        self._edit.setTabChangesFocus(True)
        line_height = QFontMetrics(self._edit.font()).lineSpacing()
        self._edit.setFixedHeight(2 * line_height + 10)
        self._edit.selectAll()

        hint = QLabel(_("Ctrl+Enter"), self)
        hint.setStyleSheet(f"font-style: italic; color: {theme.grey};")

        # plain buttons rather than QDialogButtonBox: the box reserves a wide minimum
        # for its button area, which is what kept this dialog from opening small
        ok_button = QPushButton(_("Apply"), self)
        ok_button.setDefault(True)
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton(_("Cancel"), self)
        cancel_button.clicked.connect(self.reject)

        button_row = QHBoxLayout()
        button_row.setSpacing(4)
        button_row.addWidget(hint)
        button_row.addStretch()
        button_row.addWidget(ok_button)
        button_row.addWidget(cancel_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(label)
        layout.addWidget(self._edit)
        layout.addWidget(self._negative_check)
        layout.addLayout(button_row)
        self.setLayout(layout)

        layout.activate()
        self.adjustSize()
        self.resize(QSize(min(self.width(), 300), min(self.height(), 130)))
        self._center_on_parent(parent)

    def _center_on_parent(self, parent: QWidget):
        window = parent.window()
        if window is not None:
            center = window.frameGeometry().center()
            self.move(center - QPoint(self.width() // 2, self.height() // 2))

    def keyPressEvent(self, a0):
        ctrl = a0 is not None and bool(a0.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl and a0 is not None and a0.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
        else:
            super().keyPressEvent(a0)

    @property
    def text(self):
        return self._edit.toPlainText().strip()

    @property
    def to_negative(self):
        return self._negative_check.isChecked()


class RegionPromptWidget(QWidget):
    activated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._regions: RootRegion = root.active_model.regions
        self._inactive_regions: list[InactiveRegionWidget] = []
        self._bindings: list[QMetaObject.Connection] = []

        self._prompt = ActiveRegionWidget(self._regions, self)
        self._prompt.positive.activated.connect(self.activated)
        self._prompt.negative.activated.connect(self.activated)

        self._control = ControlListWidget(self._regions.active_or_root.control, parent=self)
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
    def regions(self, regions: RootRegion):
        if regions == self._regions:
            return
        self._regions = regions
        self._setup_bindings()

    def open_prompt_picker_with(self, text: str, negative: str, image):
        self._prompt.open_prompt_picker_with(text, negative, image)

    def _setup_bindings(self):
        Binding.disconnect_all(self._bindings)
        regions = self._regions
        self._bindings = [
            regions.active_changed.connect(self._setup_region_bindings),
            regions.added.connect(self._show_inactive_regions),
            regions.removed.connect(self._show_inactive_regions),
        ]
        self._update_active()

    def _update_active(self):
        self._setup_region_bindings(self._regions.active_or_root)

    def _setup_region_bindings(self, region: RootRegion | Region | None):
        region = region or self._regions
        self._prompt.region = region
        self._control.model = region.control
        self._show_inactive_regions()

    def _add_inactive_region(self, region: RootRegion | Region, layout: QVBoxLayout):
        widget = InactiveRegionWidget(region, self)
        widget.activated.connect(self._activate_region)
        self._inactive_regions.append(widget)
        layout.addWidget(widget)

    def _show_inactive_regions(self):
        active = self._regions.active_or_root

        for widget in self._inactive_regions:
            widget.deleteLater()
        self._inactive_regions.clear()

        below, above = active.siblings  # sorted from bottom to top
        for region in (r for r in self._regions if r != active and not r.has_links):
            self._add_inactive_region(region, self._regions_above)
        for region in reversed(above):
            self._add_inactive_region(region, self._regions_above)
        for region in reversed(below):
            self._add_inactive_region(region, self._regions_below)
        if not isinstance(active, RootRegion):
            self._add_inactive_region(self._regions, self._regions_below)

    def _activate_region(self, region: RootRegion | Region):
        self._regions.active = region
        self._prompt.focus()


class RegionThumbnailWidget(QLabel):
    def __init__(self, region: RootRegion | Region, parent: QWidget):
        super().__init__(parent)
        self.set_region(region)

    def set_region(self, region: RootRegion | Region):
        font_height = self.fontMetrics().height()
        icon_size = int(1.5 * font_height + 6)
        if isinstance(region, Region):
            if layer := region.first_layer:
                parent_bounds = layer.parent_layer.bounds if layer.parent_layer else layer.bounds
                parent_bounds = Bounds.at_least(parent_bounds, icon_size)
                layer_bounds = layer.bounds.relative_to(parent_bounds)
                scale = icon_size / parent_bounds.height
                canvas_extent = parent_bounds.extent * scale
                thumb_bounds = Bounds.scale(layer_bounds, scale)
                thumb_bounds = Bounds.minimum_size(thumb_bounds, 4, canvas_extent)
                thumb_bounds = thumb_bounds or Bounds(0, 0, *canvas_extent)
                thumb = layer.thumbnail(thumb_bounds.extent)
                image = QImage(*canvas_extent, QImage.Format.Format_ARGB32)
                painter = QPainter(image)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
                painter.fillRect(image.rect(), Qt.GlobalColor.transparent)
                painter.drawImage(*thumb_bounds.offset, thumb)
                painter.end()
                icon_image = QPixmap.fromImage(image)
            else:
                icon_image = theme.icon("region-prompt")
            self.setToolTip(_("Text prompt for region") + f" {region.name}")
        else:
            icon_image = theme.icon("root")
            self.setToolTip(_("Text which is common to all regions"))
        if isinstance(icon_image, QIcon):
            size = int(1.2 * font_height)
            offset = (icon_size - size) // 2
            image = QImage(icon_size, icon_size, QImage.Format.Format_ARGB32)
            painter = QPainter(image)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(image.rect(), Qt.GlobalColor.transparent)
            painter.drawPixmap(offset, offset, icon_image.pixmap(size, size))
            painter.end()
            icon_image = QPixmap.fromImage(image)
        self.setPixmap(icon_image)
