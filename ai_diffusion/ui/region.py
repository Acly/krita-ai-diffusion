from __future__ import annotations
from PyQt5.QtWidgets import QWidget, QLabel, QToolButton, QHBoxLayout, QVBoxLayout, QFrame
from PyQt5.QtGui import QMouseEvent, QResizeEvent, QPixmap
from PyQt5.QtCore import QObject, QEvent, Qt, QMetaObject, pyqtSignal

from ..root import root
from ..image import Extent
from ..properties import Binding, bind
from ..document import LayerType
from ..model import Region, RootRegion, RegionLink
from .control import ControlListWidget
from .widget import TextPromptWidget
from .settings import settings
from . import theme


class RegionThumbnailWidget(QLabel):
    _scale: float = 1.0

    def __init__(self, region: RootRegion | Region, parent: QWidget, scale=1.0):
        super().__init__(parent)
        self._scale = scale
        self.set_region(region)

    def set_region(self, region: RootRegion | Region):
        icon_size = int(self._scale * self.fontMetrics().height())
        if isinstance(region, Region):
            if layer := region.first_layer:
                icon_image = QPixmap.fromImage(layer.thumbnail(Extent(icon_size, icon_size)))
            else:
                icon_image = theme.icon("region-prompt").pixmap(icon_size, icon_size)
        else:
            icon_image = theme.icon("root").pixmap(icon_size, icon_size)
        self.setPixmap(icon_image)


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

        scale = 1.2 if isinstance(region, RootRegion) else 1.5
        icon = RegionThumbnailWidget(region, self, scale=scale)

        self._prompt = QLabel(self)
        self._prompt.setCursor(Qt.CursorShape.IBeamCursor)

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

        if self._text == "":
            self._prompt.setStyleSheet(f"QLabel {{ font-style: italic; color: {theme.grey}; }}")
            if isinstance(region, Region):
                self._text = f"{region.name} - click to add regional text"
            else:
                self._text = "Common text prompt - click to add content"

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        self.activated.emit(self.region)
        return super().mousePressEvent(a0)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        theme.set_text_clipped(self._prompt, self._text)
        return super().resizeEvent(a0)


class ActiveRegionWidget(QFrame):
    _style_base = f"QFrame#ActiveRegionWidget {{ background-color: {theme.base}; border: 1px solid {theme.line_base}; }}"
    _style_focus = f"QFrame#ActiveRegionWidget {{ background-color: {theme.base}; border: 1px solid {theme.active}; }}"

    _region: RootRegion | Region
    _bindings: list[QMetaObject.Connection]
    _max_lines: int

    def __init__(self, root: RootRegion, parent: QWidget, max_lines=99):
        super().__init__(parent)
        self._region = root
        self._bindings = []
        self._max_lines = max_lines

        self.setObjectName("ActiveRegionWidget")
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(self._style_base)

        self._header_icon = RegionThumbnailWidget(self._region, self, scale=1.2)
        self._header_label = QLabel(self)
        self._header_label.setStyleSheet(f"font-style: italic; color: {theme.grey};")

        self._link_button = QToolButton(self)
        self._link_button.setIcon(theme.icon("link"))
        self._link_button.setAutoRaise(True)

        self._remove_button = QToolButton(self)
        self._remove_button.setIcon(theme.icon("remove"))
        self._remove_button.setAutoRaise(True)
        self._remove_button.setToolTip("Remove this region")

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(2, 2, 2, 0)
        header_layout.setSpacing(0)
        header_layout.addWidget(self._header_icon)
        header_layout.addSpacing(5)
        header_layout.addWidget(self._header_label, 1)
        header_layout.addWidget(self._link_button)
        header_layout.addWidget(self._remove_button)

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

        self._setup_bindings(self._region)
        settings.changed.connect(self.update_settings)

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region: RootRegion | Region):
        if region != self._region:
            self._region = region
            self._setup_bindings(region)

    @property
    def root(self):
        return self._region.root if isinstance(self._region, Region) else self._region

    def _setup_bindings(self, region: RootRegion | Region):
        Binding.disconnect_all(self._bindings)
        is_root_region = isinstance(region, RootRegion)
        if is_root_region:
            self._bindings = [
                bind(region, "positive", self.positive, "text"),
                bind(region, "negative", self.negative, "text"),
            ]
        else:
            self._bindings = [
                bind(region, "positive", self.positive, "text"),
                region.layer_ids_changed.connect(self._update_links),
                self._link_button.clicked.connect(region.toggle_active_link),
                self._remove_button.clicked.connect(region.remove),
            ]
        self._bindings.append(self.root.active_layer_changed.connect(self._update_links))
        self._update_links()
        self.positive.move_cursor_to_end()
        self.negative.setVisible(is_root_region and settings.show_negative_prompt)
        self._link_button.setVisible(not is_root_region)
        self._remove_button.setVisible(not is_root_region)

    def focus(self):
        if not (self.positive.has_focus or self.negative.has_focus):
            self.positive.has_focus = True

    @property
    def has_header(self):
        return self._header.isVisible()

    @has_header.setter
    def has_header(self, value: bool):
        self._header.setVisible(value)

    def _update_links(self):
        if isinstance(self._region, RootRegion):
            self._header_label.setText("Text prompt common to all regions")
        else:
            theme.set_text_clipped(
                self._header_label, f"{self._region.name} - Regional text prompt"
            )
            active_layer = self.root.layers.active
            link_enabled = False
            if self._region.is_linked(active_layer, RegionLink.direct):
                icon = "link-active"
                desc = "Active layer is linked to this region - click to unlink"
                link_enabled = True
            elif self.root.is_linked(active_layer, RegionLink.indirect):
                icon = "link"
                desc = "Active layer is linked to this region via a group layer"
            elif active_layer.type not in [LayerType.paint, LayerType.group]:
                icon = "link-disabled"
                desc = "Only paint layers and groups and be linked to regions"
            elif self.root.is_linked(active_layer, RegionLink.direct):
                icon = "link-disabled"
                desc = "Active layer is already linked to another region"
            elif Region.link_target(active_layer) is not active_layer:
                icon = "link-disabled"
                desc = "Active layer is part of a group - select the group layer to link it"
            else:
                icon = "link-off"
                desc = "Active layer is not linked - click to link it to this region"
                link_enabled = True
            self._link_button.setIcon(theme.icon(icon))
            self._link_button.setEnabled(link_enabled)
            self._link_button.setToolTip(desc)

        self._header_icon.set_region(self._region)

    def update_settings(self, key: str, value):
        if key == "prompt_line_count":
            self.positive.line_count = min(value, self._max_lines)
        elif key == "show_negative_prompt":
            self.negative.text = ""
            self.negative.setVisible(value and self._region is None)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 and a1.type() == QEvent.Type.FocusIn:
            self.setStyleSheet(self._style_focus)
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
        self._prompt.has_header = False

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
        self._prompt.has_header = len(self._regions) > 0
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

    def _activate_region(self, region: Region):
        self._regions.active = region
        self._prompt.focus()
