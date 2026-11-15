from __future__ import annotations

from contextlib import contextmanager
from functools import partial

from PyQt6.QtCore import QEvent, QMetaObject, QPoint, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QEnterEvent,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPalette,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from ..backend.api import LoraInput
from ..backend.resources import ControlMode
from ..image import Extent
from ..localization import translate as _
from ..model.control import ControlLayer, ControlLayerList
from ..model.model import DocumentModel
from ..model.properties import Binding, bind_combo
from ..model.region import Region
from ..model.root import root
from . import theme
from .interval_slider import IntervalSlider
from .lora import LoraSelectionDialog, lora_pixmap
from .style import LoraPreviewMenu, LoraStrengthSpinBox, LoraThumbnail
from .theme import SignalBlocker

THUMBNAIL_HEIGHT = 64
THUMBNAIL_WIDTH = THUMBNAIL_HEIGHT * 3 // 2
CONTROL_HEIGHT = 20
ITEM_HEIGHT = THUMBNAIL_HEIGHT + CONTROL_HEIGHT
ITEM_SPACING = 6


class FlowLayout(QLayout):
    def __init__(self, parent=None, spacing=ITEM_SPACING):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, a0: QLayoutItem | None):
        if a0 is not None:
            self._items.append(a0)

    def addWidget(self, w: QWidget | None):
        if w is not None:
            self.addChildWidget(w)
            self.addItem(QWidgetItem(w))

    def insertWidget(self, index: int, widget: QWidget):
        self.addChildWidget(widget)
        self._items.insert(index, QWidgetItem(widget))
        self.invalidate()

    def takeWidget(self, widget: QWidget):
        index = next((i for i, item in enumerate(self._items) if item.widget() is widget), -1)
        if index >= 0:
            return self.takeAt(index)
        return None

    def count(self):
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, a0: int):
        return self._arrange(QRect(0, 0, a0, 0), test_only=True)

    def setGeometry(self, a0: QRect):
        super().setGeometry(a0)
        self._arrange(a0, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        margins = self.contentsMargins()
        return QSize(
            THUMBNAIL_WIDTH + margins.left() + margins.right(),
            ITEM_HEIGHT + margins.top() + margins.bottom(),
        )

    def _arrange(self, rect: QRect, test_only: bool):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y = area.x(), area.y()
        line_height = 0
        for item in self._items:
            size = item.sizeHint()
            if x > area.x() and x + size.width() > area.right() + 1:
                x = area.x()
                y += line_height + self.spacing()
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), size))
            x += size.width() + self.spacing()
            line_height = max(line_height, size.height())
        return y + line_height - rect.y() + margins.bottom()


class AddConditionButton(QWidget):
    clicked = pyqtSignal()

    def __init__(self, icon: str, tooltip: str, parent=None):
        super().__init__(parent)
        self._icon = icon
        self._hovered = False
        self.setToolTip(tooltip)
        self.setFixedSize(THUMBNAIL_HEIGHT, (ITEM_HEIGHT - ITEM_SPACING) // 2)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, a0: QPaintEvent | None):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        background = QColor(self.palette().color(QPalette.ColorRole.Base))
        background.setAlpha(125 if self._hovered else 70)
        painter.setBrush(background)
        painter.setPen(QPen(QColor(theme.grey), 1, Qt.PenStyle.DashLine))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 2, 2)
        size = 26
        icon_rect = QRect(0, 0, size, size)
        icon_rect.moveCenter(self.rect().center())
        theme.icon(self._icon).paint(painter, icon_rect)

    def enterEvent(self, event: QEnterEvent | None):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, a0: QEvent | None):
        self._hovered = False
        self.update()
        super().leaveEvent(a0)

    def mouseReleaseEvent(self, a0: QMouseEvent | None):
        if a0 and a0.button() is Qt.MouseButton.LeftButton and self.rect().contains(a0.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(a0)


class CompactButton(QPushButton):
    def __init__(self, icon: str, text: str, tooltip: str, parent=None):
        super().__init__(theme.icon(icon), text, parent)
        self.setToolTip(tooltip)
        self._text = text

    def set_compact(self, compact: bool):
        self.setText("" if compact else self._text)

    @property
    def label(self):
        return self._text


class WarningLabel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        size = int(1.2 * self.fontMetrics().height())
        self.icon = QLabel(self)
        self.icon.setPixmap(theme.icon("warning").pixmap(size, size))
        self.text = QLabel(self)
        self.text.setStyleSheet(f"color: {theme.yellow}")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self.icon)
        layout.addWidget(self.text)
        self.hide()

    def show_message(self, summary: str, full: str):
        if summary:
            self.text.setText(summary)
            for widget in (self, self.icon, self.text):
                widget.setToolTip(full)
        else:
            self.text.setText(full)
        self.show()


class ControlAdvancedPanel(QFrame):
    def __init__(self, owner: ControlWidget):
        super().__init__(owner.list_widget)
        self.owner = owner
        self.control = owner.control
        self._connections: list[QMetaObject.Connection | Binding] = []
        self._compact = False
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setLineWidth(1)

        self.mode_combo = QComboBox(self)
        self.mode_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.mode_combo.setMinimumContentsLength(12)
        for mode in (mode for mode in ControlMode if not mode.is_internal):
            self.mode_combo.addItem(theme.icon(f"control-{mode.name}"), mode.text, mode)
        self.mode_combo.setToolTip(_("Select control layer content"))

        self.warning = WarningLabel(self)

        self.link_button = QToolButton(self)
        self.link_button.setAutoRaise(True)
        self.link_button.clicked.connect(self._toggle_region_link)

        close_button = QToolButton(self)
        close_button.setIcon(theme.icon("remove"))
        close_button.setAutoRaise(True)
        close_button.setToolTip(_("Close"))
        close_button.clicked.connect(owner.list_widget.close_advanced)

        self.layer_button = QPushButton(theme.icon("layer-replace"), "", self)
        self.layer_button.setMaximumWidth(180)
        self.layer_button.clicked.connect(self._show_layer_menu)

        self.generate_button = _create_generate_button(self)
        self.generate_button.clicked.connect(self.control.generate)
        self.generate_regions_button = _create_generate_regions_button(self)
        self.generate_regions_button.clicked.connect(self.control.generate_segmentation)
        self.add_pose_button = _create_add_pose_button(self)
        self.add_pose_button.clicked.connect(self._add_pose_character)
        self._action_buttons = [
            self.generate_button,
            self.generate_regions_button,
            self.add_pose_button,
        ]

        self.strength_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.strength_slider.setRange(0, 75)
        self.strength_slider.valueChanged.connect(self._set_strength)
        self.strength_label = QLabel(self)
        self.range_label = QLabel(_("Range") + ":", self)
        self.range_slider = IntervalSlider(0, 20, 0, 20, self)
        self.range_start_label = QLabel(self)
        self.range_end_label = QLabel(self)
        self.range_slider.intervalChanged.connect(self._set_range)

        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(4)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(self.warning)
        mode_layout.addStretch()
        mode_layout.addWidget(self.link_button)

        layer_layout = QHBoxLayout()
        layer_layout.setSpacing(4)
        layer_layout.addWidget(self.layer_button)
        layer_layout.addWidget(self.generate_button)
        layer_layout.addWidget(self.generate_regions_button)
        layer_layout.addWidget(self.add_pose_button)
        layer_layout.addStretch()

        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        self._grid = layout = QGridLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(2)
        layout.addWidget(QLabel(_("Mode") + ":", self), 0, 0)
        layout.addLayout(mode_layout, 0, 1, 1, 2)
        layout.addWidget(close_button, 0, 3, right)
        self.layer_label = QLabel(_("Layer") + ":", self)
        layout.addWidget(self.layer_label, 1, 0)
        layout.addLayout(layer_layout, 1, 1, 1, 3)
        layout.addWidget(QLabel(_("Strength") + ":", self), 2, 0)
        layout.addWidget(self.strength_slider, 2, 1, 1, 2)
        layout.addWidget(self.strength_label, 2, 3, right)
        layout.addWidget(self.range_label, 3, 0)
        layout.addWidget(self.range_start_label, 3, 1)
        layout.addWidget(self.range_slider, 3, 2)
        layout.addWidget(self.range_end_label, 3, 3, right)
        layout.setColumnStretch(2, 1)

        row_height = max(
            self.mode_combo.sizeHint().height(),
            self.generate_button.sizeHint().height(),
            self.generate_regions_button.sizeHint().height(),
            self.add_pose_button.sizeHint().height(),
        )
        self.strength_slider.setMinimumHeight(row_height)
        self.range_slider.setMinimumHeight(row_height)

        self._connections = [
            bind_combo(self.control, "mode", self.mode_combo),
            self.control.region_changed.connect(self._update_region_link),
            owner.control_list.changed.connect(self._update_region_link),
            self.control.layer_id_changed.connect(self._update_layer),
            root.active_model.layers.changed.connect(self._update_layer),
            self.control.strength_changed.connect(self._update_strength),
            self.control.start_changed.connect(self._update_range),
            self.control.end_changed.connect(self._update_range),
            self.control.has_range_changed.connect(self._update_visibility),
            self.control.can_generate_changed.connect(self._update_visibility),
            self.control.mode_changed.connect(self._update_visibility),
            self.control.is_pose_vector_changed.connect(self._update_visibility),
            self.control.is_supported_changed.connect(self._update_visibility),
            self.control.error_text_changed.connect(self._update_visibility),
            self.control.has_active_job_changed.connect(self._job_changed),
        ]
        self._update_layer()
        self._update_strength()
        self._update_range()
        self._update_visibility()

    def disconnect_all(self):
        Binding.disconnect_all(self._connections)

    def _show_layer_menu(self):
        self.owner.show_layer_menu(
            self.layer_button.mapToGlobal(self.layer_button.rect().bottomLeft())
        )

    def _toggle_region_link(self):
        self.control.region = None if self.control.region else self.owner.control_list.active_region

    def _update_region_link(self):
        self.link_button.setVisible(self.control.mode.is_ip_adapter)
        region = self.control.region
        active = self.owner.control_list.active_region
        can_link = self.control.mode.is_ip_adapter and isinstance(active, Region)
        icon = "link-active" if region else "link-off" if can_link else "link-disabled"
        self.link_button.setIcon(theme.icon(icon))
        self.link_button.setEnabled(region is not None or can_link)
        tooltip = (
            _("Linked to {region}").format(region=region.name)
            if region
            else _("Not linked to any region.")
        )
        if region:
            tooltip += "\n" + _("Click to unlink.")
        elif can_link and isinstance(active, Region):
            tooltip += "\n" + _("Click to link to {region}").format(region=active.name)
        self.link_button.setToolTip(tooltip)

    def _update_layer(self):
        self._update_region_link()
        name = self.control.layer.name
        self.layer_button.setText(
            self.layer_button.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, 140)
        )
        self.layer_button.setToolTip(_("Select a different layer as input") + f"\n{name}")
        self._update_compact()

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self._update_compact()

    def _update_compact(self):
        if self.control.mode not in (ControlMode.segmentation, ControlMode.pose):
            return
        fm = self.fontMetrics()
        width = 72  # fixed offset
        width += fm.horizontalAdvance(self.layer_button.text())
        width += fm.horizontalAdvance(self.layer_label.text())
        for button in self._action_buttons:
            width += fm.horizontalAdvance(button.label)
        compact = self.width() < width
        if compact != self._compact:
            self._compact = compact
            for button in self._action_buttons:
                button.set_compact(compact)

    def _set_strength(self, value: int):
        self.control.use_custom_strength = True
        self.control.strength = value

    def _set_range(self, low: int, high: int):
        self.control.use_custom_strength = True
        self.control.start = low / 20
        self.control.end = high / 20

    def _update_range(self):
        self.range_slider.blockSignals(True)
        self.range_slider.setInterval(int(self.control.start * 20), int(self.control.end * 20))
        self.range_slider.blockSignals(False)
        self.range_start_label.setText(f"{self.control.start:.2f}")
        self.range_end_label.setText(f"{self.control.end:.2f}")

    def _update_strength(self):
        self.strength_slider.blockSignals(True)
        self.strength_slider.setValue(self.control.strength)
        self.strength_slider.blockSignals(False)
        self.strength_label.setText(
            f"{self.control.strength / ControlLayer.strength_multiplier:.2f}"
        )

    def _update_visibility(self):
        self._update_region_link()
        self._update_compact()
        supported = self.control.is_supported
        mode = self.control.mode
        if supported:
            self.warning.hide()
        else:
            self.warning.show_message(self.control.error_text_short, self.control.error_text)
        self.generate_button.setVisible(mode.has_preprocessor)
        for button in (self.generate_regions_button, self.add_pose_button):
            button.setVisible(False)  # Hide first to avoid order-dependent panel resizing
        self.generate_regions_button.setVisible(mode is ControlMode.segmentation)
        self.add_pose_button.setVisible(mode is ControlMode.pose)
        for button in (self.generate_button, self.generate_regions_button, self.add_pose_button):
            button.setEnabled(supported and self.control.can_generate)
        self.strength_slider.setEnabled(supported)
        self.range_slider.setEnabled(supported)
        for widget in (
            self.range_label,
            self.range_slider,
            self.range_start_label,
            self.range_end_label,
        ):
            widget.setVisible(self.control.has_range)
        self.updateGeometry()

    def _job_changed(self, active: bool):
        if active:
            self.owner.list_widget.close_advanced()

    def _add_pose_character(self):
        root.active_model.document.add_pose_character(self.control.layer)


class ControlThumbnail(QWidget):
    def __init__(self, owner: ControlWidget):
        super().__init__(owner)
        self.owner = owner
        self.control = owner.control
        self._pixmap = QPixmap()
        self.setFixedSize(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_button = self._create_badge(2, _("Select control layer content"))
        self.mode_button.clicked.connect(self._show_mode_menu)
        self.remove_button = self._create_badge(
            self.width() - 22,
            _("Stop using this layer for control (does not remove the layer)"),
            remove=True,
        )
        self.remove_button.setIcon(theme.icon("remove"))
        self.remove_button.clicked.connect(owner.remove)
        self.refresh()

    def _create_badge(self, x: int, tooltip: str, remove=False):
        button = QToolButton(self)
        button.setGeometry(x, 2, 20, 20)
        button.setIconSize(QSize(14, 14))
        button.setToolTip(tooltip)
        base = self.palette().color(QPalette.ColorRole.Base)
        hover = QColor(170, 45, 45) if remove else base.lighter(160)
        button.setStyleSheet(
            f"QToolButton {{background: rgba({base.red()}, {base.green()}, {base.blue()}, 204);"
            " border: none; border-radius: 2px; padding: 3px;}"
            f"QToolButton:hover {{background: {hover.name()};}}"
        )
        return button

    def _show_mode_menu(self):
        self.owner.show_mode_menu(
            self.mode_button.mapToGlobal(self.mode_button.rect().bottomLeft())
        )

    def refresh(self):
        try:
            layer = self.control.layer
            size = layer.bounds.extent.at_least(1).scale_keep_aspect(
                Extent(THUMBNAIL_WIDTH * 2, THUMBNAIL_HEIGHT * 2)
            )
            image = layer.thumbnail(size.at_least(1))
            self._pixmap = QPixmap.fromImage(image)
        except Exception:
            self._pixmap = QPixmap()
        self.update()

    def paintEvent(self, a0: QPaintEvent | None):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 2, 2)
        painter.setClipPath(path)
        background = (
            QColor(Qt.GlobalColor.white)
            if self.control.mode.is_lines
            else self.palette().color(QPalette.ColorRole.Base)
        )
        painter.fillRect(self.rect(), background)

        if self.control.has_active_job:
            gradient = QLinearGradient(0, 0, 0, self.height())
            base = QColor(theme.base)
            gradient.setColorAt(0, base.darker(120))
            gradient.setColorAt(1, base.lighter(120))
            painter.fillRect(self.rect(), gradient)
            self._paint_icon(painter, "control-generate", self.rect(), 28)
            return

        if not self._pixmap.isNull():
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            size = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRect(QPoint(), size)
            target.moveCenter(self.rect().center())
            painter.drawPixmap(target, self._pixmap, self._pixmap.rect())

        if self.control.strength == 0:
            painter.save()
            painter.setPen(QPen(QColor(120, 120, 120, 135), 6))
            for x in range(-self.height(), self.width() + self.height(), 12):
                painter.drawLine(x, self.height(), x + self.height(), 0)
            painter.restore()

        if not self.control.is_supported:
            overlay = QColor(self.palette().color(QPalette.ColorRole.Base))
            overlay.setAlphaF(0.8)
            painter.fillRect(self.rect(), overlay)
            self._paint_icon(painter, "warning", self.rect(), 28)

        if self.owner.is_advanced_open:
            painter.setClipping(False)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Highlight), 3))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5), 2, 2)

    def _paint_icon(self, painter: QPainter, icon: str, rect: QRect, size=16):
        target = QRect(0, 0, size, size)
        target.moveCenter(rect.center())
        theme.icon(icon).paint(painter, target)

    def enterEvent(self, event: QEnterEvent | None):
        self.refresh()
        super().enterEvent(event)

    def focusInEvent(self, a0):
        self.refresh()
        super().focusInEvent(a0)

    def mouseReleaseEvent(self, a0: QMouseEvent | None):
        if not a0 or a0.button() is not Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(a0)
        if self.control.has_active_job:
            return
        if self.rect().contains(a0.pos()):
            self.owner.list_widget.toggle_advanced(self.owner)
        super().mouseReleaseEvent(a0)


class ControlWidget(QWidget):
    def __init__(
        self, control_list: ControlLayerList, control: ControlLayer, parent: ControlListWidget
    ):
        super().__init__(parent)
        self.control_list = control_list
        self.list_widget: ControlListWidget = parent
        self.control = control
        self._connections: list[QMetaObject.Connection | Binding] = []
        self.setFixedSize(THUMBNAIL_WIDTH, ITEM_HEIGHT)

        self.thumbnail = ControlThumbnail(self)
        self.preset_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.preset_slider.setRange(0, control.max_preset_value + 1)
        self.preset_slider.setSingleStep(1)
        self.preset_slider.setPageStep(1)
        self.preset_slider.setTickInterval(1)
        self.preset_slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.preset_slider.setFixedHeight(CONTROL_HEIGHT)
        self.preset_slider.setToolTip(_("Guidance strength of the control image"))
        self.preset_slider.valueChanged.connect(self._set_preset)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.thumbnail)
        layout.addWidget(self.preset_slider)

        self._connections = [
            control.mode_changed.connect(self._update_mode),
            control.layer_id_changed.connect(self.thumbnail.refresh),
            control.preset_value_changed.connect(self._update_preset),
            control.strength_changed.connect(self._update_preset),
            control.is_supported_changed.connect(self._update_support),
            control.error_text_changed.connect(self._update_support),
            control.has_active_job_changed.connect(self._update_job),
            root.active_model.layers.changed.connect(self._layers_changed),
        ]
        self._update_mode()
        self._update_preset()
        self._update_support()
        self._update_job()

    @property
    def is_advanced_open(self):
        return self.list_widget.advanced_owner is self

    def disconnect_all(self):
        Binding.disconnect_all(self._connections)
        if self.is_advanced_open:
            self.list_widget.close_advanced()

    def remove(self):
        self.control_list.remove(self.control)

    def show_mode_menu(self, pos: QPoint):
        menu = QMenu(self)
        for mode in (mode for mode in ControlMode if not mode.is_internal):
            action = QAction(theme.icon(f"control-{mode.name}"), mode.text, menu)
            action.setIconVisibleInMenu(True)
            action.setCheckable(True)
            action.setChecked(mode is self.control.mode)
            action.triggered.connect(partial(self._set_mode, mode))
            menu.addAction(action)
        menu.exec(pos)

    def _set_mode(self, mode: ControlMode):
        self.control.mode = mode

    def show_layer_menu(self, pos: QPoint):
        menu = QMenu(self)
        active = root.active_model.layers.active
        active_action = QAction(_("Active layer"), menu)
        active_action.setEnabled(active.type.is_image)
        active_action.triggered.connect(partial(self._set_layer, active.id))
        menu.addAction(active_action)
        menu.addSeparator()
        for layer in reversed(root.active_model.layers.images):
            action = QAction(layer.name, menu)
            action.setCheckable(True)
            action.setChecked(layer.id == self.control.layer_id)
            action.triggered.connect(partial(self._set_layer, layer.id))
            menu.addAction(action)
        menu.exec(pos)

    def _set_layer(self, layer_id):
        self.control.layer_id = layer_id

    def _set_preset(self, value: int):
        self.control.use_custom_strength = True
        self.control.preset_value = max(0, value - 1)
        self.control.use_custom_strength = False
        if value == 0:
            self.control.strength = 0

    def _update_preset(self):
        self.preset_slider.blockSignals(True)
        value = 0 if self.control.strength == 0 else self.control.preset_value + 1
        self.preset_slider.setValue(value)
        self.preset_slider.blockSignals(False)
        self.thumbnail.update()

    def _update_mode(self):
        self.thumbnail.mode_button.setIcon(theme.icon(f"control-{self.control.mode.name}"))
        self.thumbnail.update()

    def _update_support(self):
        self.preset_slider.setEnabled(self.control.is_supported and not self.control.has_active_job)
        self.thumbnail.setToolTip(
            _("Open advanced options to configure strength and step range")
            if self.control.is_supported
            else self.control.error_text
        )
        self.thumbnail.update()

    def _update_job(self):
        self.thumbnail.mode_button.setVisible(not self.control.has_active_job)
        self.thumbnail.remove_button.setVisible(not self.control.has_active_job)
        self.preset_slider.setEnabled(self.control.is_supported and not self.control.has_active_job)
        if self.control.has_active_job:
            self.thumbnail.update()
        else:
            self.thumbnail.refresh()

    def _layers_changed(self):
        if root.active_model.layers.find(self.control.layer_id) is None:
            self.remove()


@contextmanager
def _silent_layout_update(widget: QWidget):
    """Apply pending layout changes without painting intermediate passes.

    Inserting or removing the advanced panel changes the height of the control list.
    That takes two layout passes: the list changes its size hint, then its parent
    resizes it. Painting the in-between state makes other controls sharing the layout
    (like the prompt box) visibly jump around.
    """
    top = widget
    while (parent := top.parentWidget()) is not None and not parent.isWindow():
        top = parent
    updates_enabled = top.updatesEnabled()
    top.setUpdatesEnabled(False)
    try:
        yield
        widget.updateGeometry()
        QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
    finally:
        top.setUpdatesEnabled(updates_enabled)


class ControlListWidget(QWidget):
    def __init__(self, model: DocumentModel, parent=None):
        super().__init__(parent)
        self._model = model.control
        self._document_model = model
        self._lora_widgets: list[DocumentLoraWidget] = []
        self._widgets: list[ControlWidget] = []
        self._model_connections: list[QMetaObject.Connection] = []
        self._advanced_panel: ControlAdvancedPanel | LoraAdvancedPanel | None = None

        self._items = QWidget(self)
        self._flow_layout = FlowLayout(self._items)
        self._items.setLayout(self._flow_layout)
        self._items.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._details_layout = QVBoxLayout()
        self._details_layout.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._items)
        layout.addLayout(self._details_layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.model = model

    @property
    def model(self):
        return self._document_model

    @model.setter
    def model(self, document_model: DocumentModel):
        model = document_model.control
        if document_model is self._document_model and self._model_connections:
            return
        self.close_advanced()
        if self._model_connections:
            Binding.disconnect_all(self._model_connections)
        for widget in self._widgets:
            self._flow_layout.takeWidget(widget)
            widget.disconnect_all()
            widget.deleteLater()
        self._widgets.clear()
        for widget in self._lora_widgets:
            self._flow_layout.takeWidget(widget)
            widget.disconnect_all()
            widget.deleteLater()
        self._lora_widgets.clear()
        old_add = getattr(self, "_add_button", None)
        if old_add is not None:
            self._flow_layout.takeWidget(old_add)
            old_add.deleteLater()

        self._document_model = document_model
        self._model = model
        self._add_button = None
        self._refresh_loras()
        for control in model:
            self._add_widget(control)
        self._add_button = QWidget(self)
        buttons = QVBoxLayout(self._add_button)
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(ITEM_SPACING)
        self._add_control_button = AddConditionButton("control-add", _("Add Control Layer"), self)
        self._add_control_button.clicked.connect(model.add)
        self._add_lora_button = AddConditionButton("lora-add", _("Add LoRA"), self)
        self._add_lora_button.clicked.connect(self._select_lora)
        buttons.addWidget(self._add_control_button)
        buttons.addWidget(self._add_lora_button)
        self._flow_layout.addWidget(self._add_button)
        self._model_connections = [
            model.added.connect(self._add_widget),
            model.removed.connect(self._remove_widget),
            model.changed.connect(self._refresh),
            document_model.loras_changed.connect(self._refresh_loras),
        ]
        self._flow_layout.invalidate()

    @property
    def advanced_owner(self):
        return self._advanced_panel.owner if self._advanced_panel else None

    def toggle_advanced(self, owner: ControlWidget | DocumentLoraWidget):
        if self.advanced_owner is owner:
            self.close_advanced()
            return
        self.close_advanced()
        with _silent_layout_update(self):
            self._advanced_panel = (
                LoraAdvancedPanel(owner)
                if isinstance(owner, DocumentLoraWidget)
                else ControlAdvancedPanel(owner)
            )
            self._details_layout.addWidget(self._advanced_panel)
            self._advanced_panel.show()
            owner.thumbnail.update()

    def close_advanced(self):
        panel = self._advanced_panel
        if panel is None:
            return
        owner = panel.owner
        self._advanced_panel = None
        with _silent_layout_update(self):
            panel.disconnect_all()
            panel.hide()
            self._details_layout.removeWidget(panel)
            panel.deleteLater()
            owner.thumbnail.update()

    def _refresh(self):
        controls = list(self._model)
        for widget in list(self._widgets):
            if widget.control not in controls:
                self._remove_widget(widget.control)
        for control in controls:
            if not any(w.control is control for w in self._widgets):
                self._add_widget(control)
        self._widgets.sort(key=lambda w: controls.index(w.control))
        for widget in self._widgets:
            self._flow_layout.takeWidget(widget)
        for i, widget in enumerate(self._widgets):
            self._flow_layout.insertWidget(len(self._lora_widgets) + i, widget)
        self._flow_layout.invalidate()

    def _select_lora(self):
        dialog = LoraSelectionDialog(self._document_model, self)
        dialog.selected.connect(self._add_lora)
        dialog.open_at(self._add_lora_button.mapToGlobal(QPoint()))

    def _add_lora(self, id: str):
        if not any(l.name == id for l in self._document_model.loras):
            file = root.files.loras.find(id)
            strength = file.meta("lora_strength", 1.0) if file else 1.0
            self._document_model.loras = [*self._document_model.loras, LoraInput(id, strength)]

    def _refresh_loras(self):
        loras = self._document_model.loras
        with _silent_layout_update(self):
            for widget in list(self._lora_widgets):
                if not any(l.name == widget.id for l in loras):
                    self._lora_widgets.remove(widget)
                    self._flow_layout.takeWidget(widget)
                    widget.disconnect_all()
                    widget.deleteLater()
            for index, lora in enumerate(loras):
                widget = next((w for w in self._lora_widgets if w.id == lora.name), None)
                if widget is None:
                    widget = DocumentLoraWidget(lora.name, self)
                    self._lora_widgets.append(widget)
                self._flow_layout.takeWidget(widget)
                self._flow_layout.insertWidget(index, widget)
                widget.refresh()
            self._flow_layout.invalidate()

    def _add_widget(self, control: ControlLayer):
        if control not in self._model:
            return
        widget = ControlWidget(self._model, control, self)
        self._widgets.append(widget)
        index = self._flow_layout.count()
        if self._add_button is not None:
            index -= 1
        with _silent_layout_update(self):
            self._flow_layout.insertWidget(index, widget)
            self._flow_layout.invalidate()

    def _remove_widget(self, control: ControlLayer):
        widget = next((w for w in self._widgets if w.control is control), None)
        if widget is None:
            return
        self._widgets.remove(widget)
        with _silent_layout_update(self):
            self._flow_layout.takeWidget(widget)
            widget.disconnect_all()
            widget.deleteLater()
            self._flow_layout.invalidate()


def _create_generate_button(parent):
    return CompactButton(
        "control-generate", _("From Image"), _("Generate control layer from current image"), parent
    )


def _create_generate_regions_button(parent):
    return CompactButton(
        "region-prompt",
        _("From Regions"),
        _("Generate segmentation control layer from current regions"),
        parent,
    )


def _create_add_pose_button(parent):
    return CompactButton(
        "add-pose", _("Add Skeleton"), _("Add new character pose to selected layer"), parent
    )


class DocumentLoraThumbnail(LoraThumbnail):
    def __init__(self, owner: DocumentLoraWidget):
        super().__init__(owner)
        self.owner = owner
        self.setFixedSize(THUMBNAIL_HEIGHT, THUMBNAIL_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_button = QToolButton(self)
        self.remove_button.setGeometry(self.width() - 22, 2, 20, 20)
        self.remove_button.setIconSize(QSize(14, 14))
        self.remove_button.setIcon(theme.icon("remove"))
        self.remove_button.setToolTip(_("Remove LoRA"))
        base = self.palette().color(QPalette.ColorRole.Base)
        self.remove_button.setStyleSheet(
            f"QToolButton {{background: rgba({base.red()}, {base.green()}, {base.blue()}, 204);"
            " border: none; border-radius: 2px; padding: 3px;}"
            "QToolButton:hover {background: rgb(170, 45, 45);}"
        )
        self.remove_button.clicked.connect(owner.remove)

    def paintEvent(self, a0):
        super().paintEvent(a0)
        painter = QPainter(self)
        if self.owner.strength == 0:
            painter.setPen(QPen(QColor(120, 120, 120, 135), 6))
            for x in range(-self.height(), self.width() + self.height(), 12):
                painter.drawLine(x, self.height(), x + self.height(), 0)
        if self.owner.is_advanced_open:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Highlight), 3))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5), 2, 2)

    def mouseReleaseEvent(self, ev: QMouseEvent | None):
        if ev and ev.button() is Qt.MouseButton.LeftButton and self.rect().contains(ev.pos()):
            self.owner.list_widget.toggle_advanced(self.owner)
        super().mouseReleaseEvent(ev)


class DocumentLoraWidget(QWidget):
    def __init__(self, id: str, parent: ControlListWidget):
        super().__init__(parent)
        self.id = id
        self.list_widget = parent
        self.model = parent.model
        self.setFixedSize(THUMBNAIL_HEIGHT, ITEM_HEIGHT)
        self.thumbnail = DocumentLoraThumbnail(self)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 20)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setFixedHeight(CONTROL_HEIGHT)
        self.slider.setToolTip(_("LoRA strength"))
        self.slider.valueChanged.connect(lambda value: self.set_strength(value / 20))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.thumbnail)
        layout.addWidget(self.slider)
        self._connections = [root.files.loras.dataChanged.connect(self.refresh)]
        self.refresh()

    @property
    def strength(self):
        return next((l.strength for l in self.model.loras if l.name == self.id), 0.0)

    @property
    def is_advanced_open(self):
        return self.list_widget.advanced_owner is self

    def set_strength(self, value: float):
        self.model.loras = [
            LoraInput(l.name, value) if l.name == self.id else l for l in self.model.loras
        ]

    def remove(self):
        self.model.loras = [l for l in self.model.loras if l.name != self.id]

    def refresh(self):
        file = root.files.loras.find(self.id)
        self.thumbnail.setPixmap(lora_pixmap(file, THUMBNAIL_HEIGHT))
        self.thumbnail.setToolTip(file.full_name if file else self.id)
        with SignalBlocker(self.slider):
            self.slider.setValue(round(self.strength * 20))
        self.thumbnail.update()

    def disconnect_all(self):
        Binding.disconnect_all(self._connections)
        if self.is_advanced_open:
            self.list_widget.close_advanced()


class LoraAdvancedPanel(QFrame):
    def __init__(self, owner: DocumentLoraWidget):
        super().__init__(owner.list_widget)
        self.owner = owner

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setLineWidth(1)

        self.name = QLabel(self)
        self.name.setTextFormat(Qt.TextFormat.PlainText)
        self.name.setMinimumWidth(0)
        self.name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.preview_button = QToolButton(self)
        self.preview_button.setIcon(theme.icon("control-reference"))
        self.preview_button.setAutoRaise(True)
        self.preview_button.setToolTip(_("Preview image"))
        self.preview_button.clicked.connect(self._preview_menu)
        close_button = QToolButton(self)
        close_button.setIcon(theme.icon("remove"))
        close_button.setAutoRaise(True)
        close_button.setToolTip(_("Close"))
        close_button.clicked.connect(owner.list_widget.close_advanced)
        self.strength = LoraStrengthSpinBox(self)
        self.strength.valueChanged.connect(lambda value: owner.set_strength(value / 100))
        self.triggers = QLineEdit(self)
        self.triggers.setPlaceholderText(
            _("Optional text which is added to the prompt when the LoRA is used")
        )
        self.triggers.textChanged.connect(self._set_triggers)

        layout = QGridLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(2)
        buttons = QHBoxLayout()
        buttons.setSpacing(2)
        buttons.addWidget(self.preview_button)
        buttons.addWidget(close_button)
        layout.addWidget(self.name, 0, 0, 1, 2)
        layout.addLayout(buttons, 0, 2, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(QLabel(_("Strength") + ":", self), 1, 0)
        layout.addWidget(self.strength, 1, 1, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(QLabel(_("Triggers") + ":", self), 2, 0)
        layout.addWidget(self.triggers, 2, 1, 1, 2)
        layout.setColumnStretch(1, 1)

        self._connections = [
            owner.model.loras_changed.connect(self._update),
            root.files.loras.dataChanged.connect(self._update),
        ]
        self._update()

    def _update(self):
        file = root.files.loras.find(self.owner.id)
        self.name.setText(file.full_name if file else self.owner.id)
        self.name.setToolTip(self.name.text())
        with SignalBlocker(self.strength), SignalBlocker(self.triggers):
            self.strength.setValue(round(self.owner.strength * 100))
            triggers = file.meta("lora_triggers", "") if file else ""
            if self.triggers.text() != triggers:
                self.triggers.setText(triggers)

    def _preview_menu(self):
        file = root.files.loras.find(self.owner.id)
        menu = LoraPreviewMenu(file, self, self.owner.model.document)
        menu.exec(self.preview_button.mapToGlobal(self.preview_button.rect().bottomLeft()))

    def _set_triggers(self, text: str):
        if file := root.files.loras.find(self.owner.id):
            if file.meta("lora_triggers", "") != text:
                root.files.loras.set_meta(file, "lora_triggers", text)

    def disconnect_all(self):
        Binding.disconnect_all(self._connections)
