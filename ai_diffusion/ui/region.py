from __future__ import annotations
from enum import Enum
from PyQt5.QtWidgets import QWidget, QLabel, QToolButton, QHBoxLayout, QVBoxLayout, QFrame, QMenu
from PyQt5.QtGui import (
    QGuiApplication,
    QMouseEvent,
    QResizeEvent,
    QPixmap,
    QImage,
    QPainter,
    QIcon,
    QFontMetrics,
)
from PyQt5.QtCore import QObject, QEvent, Qt, QMetaObject, QSize, pyqtSignal

from ..root import root
from ..client import Client
from ..image import Bounds
from ..properties import Binding, bind
from ..document import LayerType
from ..region import Region, RootRegion, RegionLink, translate_prompt
from ..localization import translate as _
from ..util import ensure
from .control import ControlListWidget
from .widget import TextPromptWidget
from .settings import settings
from . import theme


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


class PromptHeader(Enum):
    none = 0
    icon = 1
    full = 2


class ActiveRegionWidget(QFrame):
    _style_base = f"QFrame#ActiveRegionWidget {{ background-color: {theme.base}; border: 1px solid {theme.line_base}; }}"
    _style_focus = f"QFrame#ActiveRegionWidget {{ background-color: {theme.base}; border: 1px solid {theme.active}; }}"

    focused = pyqtSignal()

    _root: RootRegion
    _region: RootRegion | Region | None
    _bindings: list[QMetaObject.Connection]
    _header_style: PromptHeader
    _translation_enabled: bool = True

    def __init__(self, root: RootRegion, parent: QWidget, header=PromptHeader.full):
        super().__init__(parent)
        self._root = root
        self._region = root
        self._bindings = []
        self._header_style = header
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
        self.positive.handle_dragged.connect(self._handle_dragging)
        self.positive.installEventFilter(self)

        self.negative = TextPromptWidget(line_count=1, is_negative=True, parent=self)
        self.negative.handle_dragged.connect(self._handle_dragging)
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

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if header is PromptHeader.icon:
            self._header.setVisible(False)
            positive_layout = QHBoxLayout()
            positive_layout.addWidget(self._header_icon, alignment=Qt.AlignmentFlag.AlignTop)
            positive_layout.addWidget(self.positive, 1)
            layout.addLayout(positive_layout)
        else:
            layout.addWidget(self._header)
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

    def _setup_bindings(self, region: RootRegion | Region | None):
        Binding.disconnect_all(self._bindings)
        is_root_region = isinstance(region, RootRegion)
        if is_root_region:
            self._root = region
            self._bindings = [
                bind(region, "positive", self.positive, "text"),
                bind(region, "negative", self.negative, "text"),
            ]
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
        ]
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

                def link():
                    region.link(active_layer)
                    self.region = region

                action = ensure(menu.addAction(name))
                action.triggered.connect(link)

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
        if key == "prompt_line_count" or key == "prompt_line_count_live":
            self._update_prompt_widgets()
        elif key == "show_negative_prompt":
            self.negative.text = ""
            self._update_prompt_widgets()
        elif key == "prompt_translation":
            self._update_language()

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
                if client := root.connection.client_if_connected:
                    if client.features.translation:
                        text += "\n" + self._lang_help_translate
            else:
                text = self._lang_help_disabled
            self._language_button.setToolTip(text)

    def _update_prompt_widgets(self):
        if not self.is_slim:
            self.positive.line_count = settings.prompt_line_count
        elif isinstance(self._region, Region):
            self.positive.line_count = 1
        elif self.has_negative:
            self.positive.line_count = max(1, settings.prompt_line_count_live - 1)
        else:
            self.positive.line_count = settings.prompt_line_count_live
        self.negative.setVisible(self.has_negative)
        self._layout_language_button()
        self._setup_resize_handle()

    def _layout_language_button(self):
        if settings.prompt_translation:
            pos = self.positive.geometry().bottomRight()
            if self.has_negative:
                pos = self.negative.geometry().bottomRight()
            s = QSize(self.fontMetrics().width("EN"), self.fontMetrics().height())
            self._language_button.move(pos.x() - s.width() - 2, pos.y() - s.height() - 2)
            self._language_button.resize(s)

    def _setup_resize_handle(self):
        can_resize = not (isinstance(self._region, Region) and self.is_slim)
        self.positive.is_resizable = not self.has_negative and can_resize
        self.negative.is_resizable = self.has_negative and can_resize

    def _handle_dragging(self, y_pos: int):
        # math determined experimentally, sorry :(
        if self.has_negative:
            pos_height = self.positive.contentsRect().height()
            neg_height = self.negative.contentsRect().height()
            new_height = y_pos - neg_height + pos_height - 10
        else:
            new_height = y_pos - 5
        fm = QFontMetrics(ensure(self.positive.document()).defaultFont())
        new_line_count = round(new_height / fm.lineSpacing())
        if 1 <= new_line_count <= 10:
            if self.is_slim:
                settings.prompt_line_count_live = new_line_count
            else:
                settings.prompt_line_count = new_line_count
            self._update_prompt_widgets()

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


class RegionPromptWidget(QWidget):
    _regions: RootRegion
    _inactive_regions: list[InactiveRegionWidget]
    _bindings: list[QMetaObject.Connection]

    activated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._regions = root.active_model.regions
        self._inactive_regions = []
        self._bindings = []

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
