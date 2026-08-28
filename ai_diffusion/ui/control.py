from __future__ import annotations

from functools import partial

from PyQt6.QtCore import QEvent, QMetaObject, QPoint, QRect, QRectF, QSize, Qt
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
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from ..backend.resources import ControlMode
from ..image import Extent
from ..localization import translate as _
from ..model.control import ControlLayer, ControlLayerList
from ..model.properties import Binding, bind, bind_toggle
from ..model.root import root
from . import theme
from .interval_slider import IntervalSlider

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


class AddControlButton(QWidget):
    def __init__(self, model: ControlLayerList, parent: ControlListWidget):
        super().__init__(parent)
        self.model = model
        self.list_widget = parent
        self._hovered = False
        self.setFixedSize(THUMBNAIL_HEIGHT, THUMBNAIL_HEIGHT)
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
        theme.icon("control-add").paint(painter, icon_rect)

    def enterEvent(self, event: QEnterEvent | None):
        self._hovered = True
        self.list_widget.show_tooltip(self, _("Add Control Layer"))
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, a0: QEvent | None):
        self._hovered = False
        self.list_widget.clear_tooltip(self)
        self.update()
        super().leaveEvent(a0)

    def mouseReleaseEvent(self, a0: QMouseEvent | None):
        if a0 and a0.button() is Qt.MouseButton.LeftButton and self.rect().contains(a0.pos()):
            self.model.add()
        super().mouseReleaseEvent(a0)


class ControlAdvancedPanel(QFrame):
    def __init__(self, owner: ControlWidget):
        super().__init__(owner.list_widget)
        self.owner = owner
        self.control = owner.control
        self._connections: list[QMetaObject.Connection | Binding] = []
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setLineWidth(1)

        self.custom_checkbox = QToolButton(self)
        self.custom_checkbox.setText(_("Custom values"))
        self.custom_checkbox.setCheckable(True)

        close_button = QToolButton(self)
        close_button.setIcon(theme.icon("remove"))
        close_button.setAutoRaise(True)
        close_button.setToolTip(_("Close"))
        close_button.clicked.connect(owner.list_widget.close_advanced)

        self.generate_button = _create_generate_button(self)
        self.generate_button.clicked.connect(self.control.generate)
        self.generate_regions_button = _create_generate_regions_button(self)
        self.generate_regions_button.clicked.connect(self.control.generate_segmentation)
        self.add_pose_button = _create_add_pose_button(self)
        self.add_pose_button.clicked.connect(self._add_pose_character)
        actions = QHBoxLayout()
        actions.addWidget(self.custom_checkbox)
        actions.addWidget(self.generate_button)
        actions.addWidget(self.generate_regions_button)
        actions.addWidget(self.add_pose_button)
        actions.addStretch()
        actions.addWidget(close_button)

        self.strength_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.strength_slider.setRange(0, 75)
        self.strength_label = QLabel(self)
        self.range_label = QLabel(_("Range") + ":", self)
        self.range_slider = IntervalSlider(0, 20, 0, 20, self)
        self.range_start_label = QLabel(self)
        self.range_end_label = QLabel(self)
        self.range_slider.intervalChanged.connect(self._set_range)

        sliders = QGridLayout()
        sliders.addWidget(QLabel(_("Strength") + ":", self), 0, 0)
        sliders.addWidget(self.strength_slider, 0, 1, 1, 2)
        sliders.addWidget(self.strength_label, 0, 3)
        sliders.addWidget(self.range_label, 1, 0)
        sliders.addWidget(self.range_start_label, 1, 1)
        sliders.addWidget(self.range_slider, 1, 2)
        sliders.addWidget(self.range_end_label, 1, 3)

        layout = QVBoxLayout(self)
        layout.addLayout(actions)
        layout.addLayout(sliders)

        self._connections = [
            bind_toggle(self.control, "use_custom_strength", self.custom_checkbox),
            bind(self.control, "strength", self.strength_slider, "value"),
            self.control.strength_changed.connect(self._update_strength),
            self.control.start_changed.connect(self._update_range),
            self.control.end_changed.connect(self._update_range),
            self.control.use_custom_strength_changed.connect(self._update_enabled),
            self.control.has_range_changed.connect(self._update_visibility),
            self.control.can_generate_changed.connect(self._update_visibility),
            self.control.mode_changed.connect(self._update_visibility),
            self.control.is_pose_vector_changed.connect(self._update_visibility),
            self.control.is_supported_changed.connect(self._support_changed),
            self.control.has_active_job_changed.connect(self._job_changed),
        ]
        self._update_strength()
        self._update_range()
        self._update_enabled()
        self._update_visibility()

    def disconnect_all(self):
        Binding.disconnect_all(self._connections)

    def _set_range(self, low: int, high: int):
        self.control.start = low / 20
        self.control.end = high / 20

    def _update_range(self):
        self.range_slider.setInterval(int(self.control.start * 20), int(self.control.end * 20))
        self.range_start_label.setText(f"{self.control.start:.2f}")
        self.range_end_label.setText(f"{self.control.end:.2f}")

    def _update_strength(self):
        self.strength_label.setText(
            f"{self.control.strength / ControlLayer.strength_multiplier:.2f}"
        )

    def _update_enabled(self):
        enabled = self.control.use_custom_strength and self.control.is_supported
        self.strength_slider.setEnabled(enabled)
        self.range_slider.setEnabled(enabled)

    def _update_visibility(self):
        supported = self.control.is_supported
        is_segmentation = self.control.mode is ControlMode.segmentation
        is_pose = self.control.mode is ControlMode.pose
        self.generate_button.setVisible(supported and self.control.can_generate)
        self.generate_regions_button.setVisible(supported and is_segmentation)
        self.add_pose_button.setVisible(supported and is_pose)
        for widget in (
            self.range_label,
            self.range_slider,
            self.range_start_label,
            self.range_end_label,
        ):
            widget.setVisible(self.control.has_range)
        self._update_enabled()
        self.adjustSize()

    def _support_changed(self, supported: bool):
        if not supported:
            self.owner.list_widget.close_advanced()
        else:
            self._update_visibility()

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
        self._hovered = False
        self._hover_section = ""
        self.setFixedSize(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.refresh()

    def refresh(self):
        try:
            image = self.control.layer.thumbnail(Extent(THUMBNAIL_WIDTH * 2, THUMBNAIL_HEIGHT * 2))
            pixmap = QPixmap.fromImage(image)
            pixmap = pixmap.scaled(
                self.size() * 2,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = max(0, (pixmap.width() - THUMBNAIL_WIDTH * 2) // 2)
            y = max(0, (pixmap.height() - THUMBNAIL_HEIGHT * 2) // 2)
            self._pixmap = pixmap.copy(x, y, THUMBNAIL_WIDTH * 2, THUMBNAIL_HEIGHT * 2)
        except Exception:
            self._pixmap = QPixmap()
        self.update()

    def paintEvent(self, a0: QPaintEvent | None):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 2, 2)
        painter.setClipPath(path)
        painter.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Base))

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
            painter.drawPixmap(self.rect(), self._pixmap, self._pixmap.rect())

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

        if self._hovered or self.owner.is_advanced_open:
            self._paint_grid(painter)
            if not self.control.is_supported:
                self._paint_icon(painter, "warning", self.rect(), 28)
        else:
            self._paint_badge(painter, f"control-{self.control.mode.name}", QRect(2, 2, 20, 20))
            self._paint_badge(painter, "remove", QRect(self.width() - 22, 2, 20, 20))

    def _paint_badge(self, painter: QPainter, icon: str, rect: QRect):
        background = QColor(self.palette().color(QPalette.ColorRole.Base))
        background.setAlphaF(0.8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 2, 2)
        theme.icon(icon).paint(painter, rect.adjusted(3, 3, -3, -3))

    def _paint_icon(self, painter: QPainter, icon: str, rect: QRect, size=16):
        target = QRect(0, 0, size, size)
        target.moveCenter(rect.center())
        theme.icon(icon).paint(painter, target)

    def _paint_grid(self, painter: QPainter):
        base = QColor(self.palette().color(QPalette.ColorRole.Base))
        base.setAlphaF(0.6)
        painter.fillRect(self.rect(), base)
        sections = self._section_rects()
        for name, rect in sections.items():
            selected = name == self._hover_section or (
                name == "advanced" and self.owner.is_advanced_open
            )
            if selected:
                color = QColor(170, 45, 45, 175) if name == "remove" else QColor(255, 255, 255, 45)
                painter.fillRect(rect, color)

        line_color = QColor(self.palette().color(QPalette.ColorRole.Text))
        painter.setPen(QPen(line_color, 1))
        y1, y2 = THUMBNAIL_HEIGHT // 3, THUMBNAIL_HEIGHT * 2 // 3
        painter.drawLine(0, y1, self.width(), y1)
        painter.drawLine(self.width() // 2, 0, self.width() // 2, y1)

        self._paint_icon(painter, f"control-{self.control.mode.name}", sections["mode"])
        self._paint_icon(painter, "remove", sections["remove"])
        if self.control.is_supported:
            painter.drawLine(0, y2, self.width(), y2)
            painter.drawLine(self.width() // 2, y2, self.width() // 2, self.height())
            self._paint_icon(painter, "layer-replace", sections["layer"])
            if self.control.can_generate:
                self._paint_icon(painter, "control-generate", sections["generate"])
            self._paint_icon(painter, "more", sections["advanced"])

    def _section_rects(self):
        half = self.width() // 2
        third = self.height() // 3
        sections = {
            "mode": QRect(0, 0, half, third),
            "remove": QRect(half, 0, self.width() - half, third),
        }
        if self.control.is_supported:
            sections.update({
                "layer": QRect(0, third, self.width(), third),
                "generate": QRect(0, third * 2, half, self.height() - third * 2),
                "advanced": QRect(half, third * 2, self.width() - half, self.height() - third * 2),
            })
        return sections

    def _section_at(self, pos: QPoint):
        return next(
            (name for name, rect in self._section_rects().items() if rect.contains(pos)), ""
        )

    def enterEvent(self, event: QEnterEvent | None):
        self.refresh()
        if not self.control.has_active_job:
            self._hovered = True
            if event:
                self._hover_section = self._section_at(event.position().toPoint())
        self._update_tooltip()
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, a0: QEvent | None):
        self._hovered = False
        self._hover_section = ""
        self.owner.list_widget.clear_tooltip(self)
        self.update()
        super().leaveEvent(a0)

    def focusInEvent(self, a0):
        self.refresh()
        super().focusInEvent(a0)

    def mouseMoveEvent(self, a0: QMouseEvent | None):
        if a0 and self._hovered:
            section = self._section_at(a0.pos())
            if section != self._hover_section:
                self._hover_section = section
                self._update_tooltip()
                self.update()
        super().mouseMoveEvent(a0)

    def _update_tooltip(self):
        if not self.control.is_supported:
            text = self.control.error_text
        else:
            tooltips = {
                "mode": _("Select control layer content"),
                "remove": _("Stop using this layer for control (does not remove the layer)"),
                "layer": _("Select a different layer as input"),
                "generate": _("Generate new {mode} layer from visible canvas").format(
                    mode=self.control.mode.text
                ),
                "advanced": _("Open advanced options to configure strength and step range"),
            }
            text = tooltips.get(self._hover_section, "")
        self.owner.list_widget.show_tooltip(self, text)

    def mouseReleaseEvent(self, a0: QMouseEvent | None):
        if not a0 or a0.button() is not Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(a0)
        if self.control.has_active_job:
            return
        section = self._section_at(a0.pos())
        if section == "mode":
            self.owner.show_mode_menu(self.mapToGlobal(a0.pos()))
        elif section == "remove":
            self.owner.remove()
        elif section == "layer":
            self.owner.show_layer_menu(self.mapToGlobal(a0.pos()))
        elif section == "generate" and self.control.is_supported and self.control.can_generate:
            self.control.generate()
        elif section == "advanced" and self.control.is_supported:
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
        self.preset_slider.installEventFilter(self)
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

    def eventFilter(self, a0, a1):
        if a0 is self.preset_slider and a1:
            if a1.type() == QEvent.Type.Enter:
                self.list_widget.show_tooltip(
                    self.preset_slider, _("Guidance strength of the control image")
                )
            elif a1.type() == QEvent.Type.Leave:
                self.list_widget.clear_tooltip(self.preset_slider)
        return super().eventFilter(a0, a1)

    def remove(self):
        self.list_widget.clear_tooltip(self.thumbnail)
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
        if value == 0:
            self.control.use_custom_strength = True
            self.control.strength = 0
        else:
            self.control.preset_value = value - 1
            self.control.use_custom_strength = False

    def _update_preset(self):
        self.preset_slider.blockSignals(True)
        value = 0 if self.control.strength == 0 else self.control.preset_value + 1
        self.preset_slider.setValue(value)
        self.preset_slider.blockSignals(False)
        self.thumbnail.update()

    def _update_mode(self):
        self.thumbnail.update()

    def _update_support(self):
        self.preset_slider.setEnabled(self.control.is_supported and not self.control.has_active_job)
        if self.thumbnail.underMouse():
            self.thumbnail._update_tooltip()
        self.thumbnail.update()

    def _update_job(self):
        self.preset_slider.setEnabled(self.control.is_supported and not self.control.has_active_job)
        if self.control.has_active_job:
            self.thumbnail.update()
        else:
            self.thumbnail.refresh()

    def _layers_changed(self):
        if root.active_model.layers.find(self.control.layer_id) is None:
            self.remove()


class ControlListWidget(QWidget):
    def __init__(self, model: ControlLayerList, parent=None):
        super().__init__(parent)
        self._model = model
        self._widgets: list[ControlWidget] = []
        self._model_connections: list[QMetaObject.Connection] = []
        self._tooltip_source: QWidget | None = None
        self._advanced_panel: ControlAdvancedPanel | None = None

        self._items = QWidget(self)
        self._flow_layout = FlowLayout(self._items)
        self._items.setLayout(self._flow_layout)
        self._items.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self._tooltip_label = QLabel(self, Qt.WindowType.ToolTip)
        self._tooltip_label.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._tooltip_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._tooltip_label.setWordWrap(True)
        self._tooltip_label.setStyleSheet(
            f"QLabel {{padding: 4px; background: {theme.base}; color: {theme.grey}; border: 1px solid {theme.line};}}"
        )
        self._tooltip_label.hide()

        self._details_layout = QVBoxLayout()
        self._details_layout.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._items)
        layout.addLayout(self._details_layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: ControlLayerList):
        self.close_advanced()
        self._tooltip_source = None
        self._tooltip_label.hide()
        if self._model_connections:
            Binding.disconnect_all(self._model_connections)
        for widget in self._widgets:
            self._flow_layout.takeWidget(widget)
            widget.disconnect_all()
            widget.deleteLater()
        self._widgets.clear()
        old_add = getattr(self, "_add_button", None)
        if old_add is not None:
            self._flow_layout.takeWidget(old_add)
            old_add.deleteLater()

        self._model = model
        self._add_button = None
        for control in model:
            self._add_widget(control)
        self._add_button = AddControlButton(model, self)
        self._flow_layout.addWidget(self._add_button)
        self._model_connections = [
            model.added.connect(self._add_widget),
            model.removed.connect(self._remove_widget),
        ]
        self._flow_layout.invalidate()

    @property
    def advanced_owner(self):
        return self._advanced_panel.owner if self._advanced_panel else None

    def show_tooltip(self, source: QWidget, text: str):
        self._tooltip_source = source
        self._tooltip_label.setText(text)
        if text:
            self._position_tooltip()
            self._tooltip_label.show()
            self._tooltip_label.raise_()
        else:
            self._tooltip_label.hide()

    def clear_tooltip(self, source: QWidget):
        if self._tooltip_source is source:
            self._tooltip_source = None
            self._tooltip_label.hide()
            self._tooltip_label.clear()

    def _position_tooltip(self):
        width = self.width()
        height = self._tooltip_label.heightForWidth(width)
        pos = self.mapToGlobal(QPoint(0, self._items.geometry().bottom() + 4))
        self._tooltip_label.setGeometry(pos.x(), pos.y(), width, height)

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        if self._tooltip_label.isVisible():
            self._position_tooltip()

    def moveEvent(self, a0):
        super().moveEvent(a0)
        if self._tooltip_label.isVisible():
            self._position_tooltip()

    def hideEvent(self, a0):
        self._tooltip_label.hide()
        super().hideEvent(a0)

    def toggle_advanced(self, owner: ControlWidget):
        if self.advanced_owner is owner:
            self.close_advanced()
            return
        self.close_advanced()
        self._advanced_panel = ControlAdvancedPanel(owner)
        self._details_layout.addWidget(self._advanced_panel)
        self._advanced_panel.show()
        owner.thumbnail.update()
        self.updateGeometry()

    def close_advanced(self):
        panel = self._advanced_panel
        if panel is None:
            return
        owner = panel.owner
        self._advanced_panel = None
        panel.disconnect_all()
        self._details_layout.removeWidget(panel)
        panel.deleteLater()
        owner.thumbnail.update()
        self.updateGeometry()

    def _add_widget(self, control: ControlLayer):
        widget = ControlWidget(self._model, control, self)
        self._widgets.append(widget)
        index = self._flow_layout.count()
        if self._add_button is not None:
            index -= 1
        self._flow_layout.insertWidget(index, widget)
        self._flow_layout.invalidate()
        self.updateGeometry()

    def _remove_widget(self, control: ControlLayer):
        widget = next(widget for widget in self._widgets if widget.control is control)
        if self._tooltip_source in (widget.thumbnail, widget.preset_slider):
            self.clear_tooltip(self._tooltip_source)
        self._widgets.remove(widget)
        self._flow_layout.takeWidget(widget)
        widget.disconnect_all()
        widget.deleteLater()
        self._flow_layout.invalidate()
        self.updateGeometry()


def _create_generate_button(parent):
    button = QPushButton(theme.icon("control-generate"), _("From Image"), parent)
    button.setToolTip(_("Generate control layer from current image"))
    return button


def _create_generate_regions_button(parent):
    button = QPushButton(theme.icon("region-prompt"), _("From Regions"), parent)
    button.setToolTip(_("Generate segmentation control layer from current regions"))
    return button


def _create_add_pose_button(parent):
    button = QPushButton(theme.icon("add-pose"), _("Add Skeleton"), parent)
    button.setToolTip(_("Add new character pose to selected layer"))
    return button
