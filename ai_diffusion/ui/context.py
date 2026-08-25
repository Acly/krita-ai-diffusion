from __future__ import annotations

from typing import Any, cast

from krita import DoubleSliderSpinBox, Selection
from PyQt6.QtCore import (
    QByteArray,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QUuid,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QImage,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..backend.api import InpaintContext, InpaintParams
from ..image import Bounds, Extent, Image
from ..localization import translate as _
from ..model.model import DocumentModel, MaskSource
from ..model.properties import Binding
from ..model.root import root
from ..settings import settings
from ..util import client_logger as log
from . import theme


class ContextPreview(QWidget):
    _background_top = QColor(theme.base).darker(120)
    _background_bottom = QColor(theme.base).lighter(120)

    def __init__(self, parent=None, interactive=True):
        super().__init__(parent)
        self._interactive = interactive
        self._hovered = False
        self._pixmap = QPixmap()
        self._aspect = 1.0
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus if interactive else Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        return QSize(180, 120)

    def set_preview(self, pixmap: QPixmap, aspect: float):
        self._pixmap = pixmap
        self._aspect = aspect
        self.update()

    def paintEvent(self, a0: QPaintEvent | None):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, self._background_top)
        gradient.setColorAt(1, self._background_bottom)
        painter.fillRect(self.rect(), gradient)

        target = _fit_aspect(self.rect(), self._aspect)
        painter.fillRect(target, Qt.GlobalColor.black)
        if not self._pixmap.isNull():
            painter.drawPixmap(target, self._pixmap, self._pixmap.rect())

        if self._interactive and self._hovered:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 110))
            outer = self.rect().adjusted(16, 16, -16, -16)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawRect(outer)
            painter.setPen(QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine))
            painter.drawRect(outer.adjusted(16, 16, -16, -16))

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, a0):
        self._hovered = False
        self.update()
        super().leaveEvent(a0)


class ContextPopup(QFrame):
    def __init__(self, owner: ContextWidget):
        super().__init__(None, Qt.WindowType.Popup)
        self.owner = owner
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self.context_combo = QComboBox(self)
        self.context_combo.setMinimumContentsLength(18)
        self.context_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.context_combo.currentIndexChanged.connect(owner._set_context)

        self.padding = _slider(_("Pad"), 0, 25, "%", settings.selection_padding, self)
        self.feather = _slider(_("Feather"), 0, 25, "%", settings.selection_feather, self)
        self.blend = _slider(_("Blend"), 0, 100, " px", settings.selection_blend, self)
        self.padding.widget().valueChanged.connect(
            lambda value: owner._set_setting("selection_padding", value)
        )
        self.feather.widget().valueChanged.connect(
            lambda value: owner._set_setting("selection_feather", value)
        )
        self.blend.widget().valueChanged.connect(
            lambda value: owner._set_setting("selection_blend", value)
        )

        self.target_group = QButtonGroup(self)
        target_layout = QHBoxLayout()
        target_layout.setContentsMargins(0, 0, 0, 0)
        for text, source in (
            (_("Canvas"), MaskSource.none),
            (_("Selection"), MaskSource.selection),
            (_("Region"), MaskSource.region),
        ):
            button = QRadioButton(text, self)
            self.target_group.addButton(button, source.value)
            target_layout.addWidget(button)
        self.target_group.idClicked.connect(owner._set_mask_source)

        controls = QVBoxLayout()
        controls.addWidget(QLabel(_("Context"), self))
        controls.addWidget(self.context_combo)
        controls.addWidget(self.padding.widget())
        controls.addWidget(theme.horizontal_line(self))
        controls.addWidget(QLabel(_("Target"), self))
        controls.addLayout(target_layout)
        controls.addWidget(self.feather.widget())
        controls.addWidget(self.blend.widget())
        controls.addStretch()

        self.preview = ContextPreview(self, interactive=False)
        layout = QHBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.preview, alignment=Qt.AlignmentFlag.AlignTop)
        self.setLayout(layout)

    def sync(self):
        owner = self.owner
        owner._update_context_layers(self.context_combo)
        context = owner.model.inpaint.context
        data = (
            owner.model.inpaint.context_layer_id
            if context is InpaintContext.layer_bounds
            else context
        )
        index = self.context_combo.findData(data)
        if index >= 0:
            self.context_combo.setCurrentIndex(index)
        self.padding.widget().setValue(settings.selection_padding)
        self.feather.widget().setValue(settings.selection_feather)
        self.blend.widget().setValue(settings.selection_blend)
        if button := self.target_group.button(owner.model.mask_source.value):
            button.setChecked(True)
        self.padding.widget().setEnabled(context is InpaintContext.automatic)
        has_selection = owner.model.document.selection_bounds is not None
        enabled = owner.model.mask_source is not MaskSource.none
        enabled = enabled and not (
            owner.model.mask_source is MaskSource.selection and not has_selection
        )
        self.feather.widget().setEnabled(enabled)
        self.blend.widget().setEnabled(enabled)
        self.preview.set_preview(owner._pixmap, owner._aspect)

    def resize_preview(self):
        layout = self.layout()
        assert layout is not None
        controls_item = layout.itemAt(0)
        assert controls_item is not None
        controls_height = controls_item.sizeHint().height()
        max_height = max(320, controls_height)
        aspect = self.owner._aspect
        width = min(320, round(max_height * aspect))
        height = min(max_height, round(width / max(aspect, 0.001)))
        self.preview.setFixedSize(width, height)


class _ResizeSignals(QObject):
    finished = pyqtSignal(int, object, object)
    failed = pyqtSignal(int, object)


class _ResizeJob(QRunnable):
    def __init__(
        self,
        request: int,
        image_data: QByteArray,
        image_extent: Extent,
        mask_data: QByteArray | QImage | None,
        mask_bounds: Bounds | None,
        maximum: QSize,
    ):
        super().__init__()
        self.setAutoDelete(False)
        self.request = request
        self.image_data = image_data
        self.image_extent = image_extent
        self.mask_data = mask_data
        self.mask_bounds = mask_bounds
        self.maximum = maximum
        self.signals = _ResizeSignals()

    def run(self):
        try:
            preview, original_mask = _resize_preview_bytes(
                self.image_data,
                self.image_extent,
                self.mask_data,
                self.mask_bounds,
                self.maximum,
            )
            self.image_data = QByteArray()
            self.mask_data = None
            self.signals.finished.emit(self.request, preview, original_mask)
        except Exception as error:
            self.image_data = QByteArray()
            self.mask_data = None
            self.signals.failed.emit(self.request, error)


class ContextWidget(ContextPreview):
    def __init__(self, parent=None):
        super().__init__(parent, interactive=True)
        self._model = root.active_model
        self._connections: list[QMetaObject.Connection | Binding] = []
        self._popup: ContextPopup | None = None
        self._resize_pool = QThreadPool(self)
        self._resize_pool.setMaxThreadCount(1)
        self._resize_request = 0
        self._resize_running: tuple[_ResizeJob, tuple] | None = None
        self._refresh_pending = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        self._selection_timer = QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.setInterval(400)
        self._selection_timer.timeout.connect(self._selection_changed)
        self.setToolTip(_("Preview and tune the image context used for generation"))
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        settings.changed.connect(self._settings_changed)
        self.model = self._model
        self.refresh()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._connections:
            Binding.disconnect_all(self._connections)
        self._model = model
        self._connections = [
            model.mask_source_changed.connect(self._sub_setting_changed),
            model.strength_changed.connect(self._sub_setting_changed),
            model.inpaint.context_changed.connect(self._sub_setting_changed),
            model.inpaint.context_layer_id_changed.connect(self._sub_setting_changed),
            model.layers.changed.connect(self._layers_changed),
            model.document.selection_bounds_changed.connect(self._selection_update_requested),
            model.regions.active_changed.connect(self._target_changed),
        ]
        self.refresh()

    def showEvent(self, a0):
        super().showEvent(a0)
        if self._refresh_pending or self._pixmap.isNull():
            self.refresh()

    def resizeEvent(self, a0):
        width = round(self.height() * 1.5)
        if self.width() != width:
            self.setFixedWidth(width)
        super().resizeEvent(a0)

    def enterEvent(self, event):
        self.refresh()
        super().enterEvent(event)

    def focusInEvent(self, a0):
        self.refresh()
        super().focusInEvent(a0)

    def mouseReleaseEvent(self, a0: QMouseEvent | None):
        if a0 and a0.button() is Qt.MouseButton.LeftButton:
            self.show_popup()
        super().mouseReleaseEvent(a0)

    def refresh(self):
        self._selection_timer.stop()
        self._refresh_timer.stop()
        if not self.isVisible() or self._resize_running is not None:
            self._refresh_pending = True
            return

        self._refresh_pending = False
        self._resize_request += 1
        request = self._resize_request
        try:
            image_data, bounds, mask_data, mask_bounds, params = (
                self.model.get_generation_context_bytes()
            )
            self._aspect = bounds.width / max(1, bounds.height)
            job = _ResizeJob(
                request,
                image_data,
                bounds.extent,
                mask_data,
                mask_bounds,
                QSize(320, 320),
            )
            job.signals.finished.connect(self._resize_finished)
            job.signals.failed.connect(self._resize_failed)
            context = (bounds.extent, params)
            self._resize_running = (job, context)
            self._resize_pool.start(job)
        except Exception as error:
            log.warning(f"Failed to update context preview: {error}")
            self._pixmap = QPixmap()
            extent = self.model.document.extent
            self._aspect = extent.width / max(1, extent.height)
            self.update()

    def _resize_finished(self, request: int, preview: QImage, original_mask: QImage | None):
        context = self._complete_resize(request)
        if request != self._resize_request or context is None:
            return
        source_extent, params = context
        self._pixmap = _compose_preview(preview, original_mask, source_extent, params)
        self.update()
        if self._popup and self._popup.isVisible():
            self._popup.sync()
            self._popup.resize_preview()
        self._schedule_pending_refresh()

    def _resize_failed(self, request: int, error: object):
        context = self._complete_resize(request)
        if context is not None:
            log.warning(f"Failed to resize context preview: {error}")
            self._schedule_pending_refresh()

    def _complete_resize(self, request: int):
        if self._resize_running is None or self._resize_running[0].request != request:
            return None
        _, context = self._resize_running
        self._resize_running = None
        return context

    def _schedule_pending_refresh(self):
        if self._refresh_pending and self.isVisible():
            self._refresh_pending = False
            self._refresh_timer.start(0)

    def show_popup(self):
        if self._popup is None:
            self._popup = ContextPopup(self)
        self._popup.sync()
        self._popup.resize_preview()
        self._popup.adjustSize()
        pos = self.mapToGlobal(QPoint(0, self.height()))
        screen_object = self.screen()
        assert screen_object is not None
        screen = screen_object.availableGeometry()
        if pos.y() + self._popup.height() > screen.bottom():
            pos.setY(self.mapToGlobal(QPoint(0, 0)).y() - self._popup.height())
        pos.setX(min(pos.x(), screen.right() - self._popup.width()))
        self._popup.move(pos)
        self._popup.show()

    def _set_context(self):
        if self._popup is None:
            return
        data = self._popup.context_combo.currentData()
        if isinstance(data, QUuid):
            self.model.inpaint.context = InpaintContext.layer_bounds
            self.model.inpaint.context_layer_id = data
        elif isinstance(data, InpaintContext):
            self.model.inpaint.context = data

    def _set_mask_source(self, value: int):
        self.model.mask_source = MaskSource(value)

    def _set_setting(self, name: str, value: float):
        setattr(settings, name, round(value))

    def _settings_changed(self, name: str, value: object):
        if name in {"selection_padding", "selection_feather", "selection_blend"}:
            self.refresh()

    def _sub_setting_changed(self, value=None):
        self.refresh()

    def _target_changed(self, value=None):
        if self._popup:
            self._popup.sync()
        self.refresh()

    def _selection_update_requested(self):
        if self._popup:
            self._popup.sync()
        self._selection_timer.start()

    def _selection_changed(self):
        self.refresh()

    def _layers_changed(self):
        if self._popup:
            self._update_context_layers(self._popup.context_combo)

    def _update_context_layers(self, combo: QComboBox):
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(theme.icon("context-automatic"), _("Automatic"), InpaintContext.automatic)
        combo.addItem(theme.icon("context-mask"), _("Selection"), InpaintContext.mask_bounds)
        combo.addItem(theme.icon("context-image"), _("Entire Canvas"), InpaintContext.entire_image)
        for layer in self.model.layers.masks:
            combo.addItem(theme.icon("context-layer"), layer.name, layer.id)
        index = combo.findData(current)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)


def _slider(label: str, minimum: int, maximum: int, suffix: str, value: int, parent):
    slider = DoubleSliderSpinBox()
    slider.setRange(minimum, maximum, 0)
    spin = slider.widget()
    spin.setPrefix(label + ": ")
    spin.setSuffix(suffix)
    spin.setValue(value)
    return slider


def _fit_aspect(rect: QRect, aspect: float):
    width = rect.width()
    height = round(width / max(aspect, 0.001))
    if height > rect.height():
        height = rect.height()
        width = round(height * aspect)
    result = QRect(0, 0, width, height)
    result.moveCenter(rect.center())
    return result


def _resize_preview_bytes(
    image_data: QByteArray,
    image_extent: Extent,
    mask_data: QByteArray | QImage | None,
    mask_bounds: Bounds | None,
    maximum: QSize,
):
    source = QImage(
        cast(bytes, memoryview(cast(Any, image_data))),
        image_extent.width,
        image_extent.height,
        image_extent.width * 4,
        QImage.Format.Format_ARGB32,
    )
    extent = image_extent.scale_keep_aspect(Extent(maximum.width(), maximum.height()))
    preview = source.scaled(
        extent.width,
        extent.height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )

    original_mask = None
    if mask_data is not None and mask_bounds is not None:
        if isinstance(mask_data, QByteArray):
            mask_source = QImage(
                cast(bytes, memoryview(cast(Any, mask_data))),
                mask_bounds.width,
                mask_bounds.height,
                mask_bounds.width,
                QImage.Format.Format_Grayscale8,
            )
        else:
            mask_source = mask_data
        scale_x = extent.width / image_extent.width
        scale_y = extent.height / image_extent.height
        target = Bounds(
            round(mask_bounds.x * scale_x),
            round(mask_bounds.y * scale_y),
            max(1, round(mask_bounds.width * scale_x)),
            max(1, round(mask_bounds.height * scale_y)),
        )
        scaled_mask = mask_source.scaled(
            target.width,
            target.height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        original_mask = QImage(extent.width, extent.height, QImage.Format.Format_Grayscale8)
        original_mask.fill(0)
        painter = QPainter(original_mask)
        painter.drawImage(target.x, target.y, scaled_mask)
        painter.end()
    return preview, original_mask


def _compose_preview(
    preview: QImage,
    mask: QImage | None,
    source_extent: Extent,
    params: InpaintParams | None,
):
    if mask is None:
        return QPixmap.fromImage(preview)

    original_mask = Image(mask)
    transition_mask = _transition_mask(original_mask, source_extent, params)
    painter = QPainter(preview)
    painter.fillRect(preview.rect(), QColor(0, 0, 0, 170))
    if transition_mask is not None:
        col = _masked_color(transition_mask, QColor(255, 210, 32, 105))
        painter.drawImage(0, 0, col)
    painter.drawImage(0, 0, _masked_color(original_mask, QColor(255, 255, 255, 145)))
    painter.end()
    return QPixmap.fromImage(preview)


def _transition_mask(original: Image, source_extent: Extent, params: InpaintParams | None):
    if params is None:
        return None
    scale = original.width / max(1, source_extent.width)
    grow = round(max(0, params.feather - params.blend // 2) * scale)
    feather = round(params.blend * scale)
    if grow == 0 and feather == 0:
        return None

    selection = Selection()
    if not hasattr(selection, "grow") or not hasattr(selection, "feather"):
        return None
    selection.setPixelData(original.to_packed_bytes(), 0, 0, original.width, original.height)
    if grow > 0:
        selection.grow(grow, grow)
    if feather > 0:
        selection.feather(feather)
    processed = Image.from_packed_bytes(
        selection.pixelData(0, 0, original.width, original.height),
        original.extent,
        channels=1,
    )
    return Image.mask_subtract(processed, original)


def _masked_color(mask: Image, color: QColor):
    overlay = QImage(mask.width, mask.height, QImage.Format.Format_ARGB32_Premultiplied)
    overlay.fill(color)
    alpha = mask._qimage.copy()
    alpha.reinterpretAsFormat(QImage.Format.Format_Alpha8)
    painter = QPainter(overlay)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
    painter.drawImage(0, 0, alpha)
    painter.end()
    return overlay
