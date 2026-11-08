from __future__ import annotations

from math import atan2, degrees, hypot

from krita import Canvas, DockWidget, Krita, View
from PyQt6.QtCore import QMetaObject, QPointF, QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QLinearGradient,
    QPainter,
    QPalette,
    QPen,
    QTransform,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..image import Image
from ..localization import translate as _
from ..model.model import DocumentModel
from ..model.properties import Binding, bind_toggle
from ..model.root import root
from ..settings import settings
from . import theme
from .history import PreviewReel
from .widget import SeedWidget


class SpinnerWidget(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setFixedSize(48, 20)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.progress = 0
        self.rotation = 0
        self.brightness = 0
        self.brightness_v = 5

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_rotation)
        self.timer.setInterval(50)

    def start_animation(self):
        if not self.timer.isActive():
            self.brightness = 0
            self.brightness_v = 5
            self.timer.start()
            self.show()

    def stop_animation(self):
        self.timer.stop()
        self.hide()

    def set_progress(self, progress: float):
        self.progress = progress
        self.update()

    def _update_rotation(self):
        self.rotation = (self.rotation + 20) % 360
        self.brightness = self.brightness + self.brightness_v
        if self.brightness >= 40:
            self.brightness_v = -5
        elif self.brightness <= -40:
            self.brightness_v = 5
        self.update()

    def paintEvent(self, a0):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(theme.grey).lighter(100 + self.brightness)
        painter.setPen(QPen(color, 2))
        painter.drawArc(QRectF(30, 2, 16, 16), self.rotation * 16, 120 * 16)
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        text = f"{int(self.progress * 100)}%"
        painter.drawText(
            2, 2, 24, 16, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text
        )


class LiveThumbnail(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._image: Image | None = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(_("Open AI Live Preview"))
        settings.changed.connect(self._settings_changed)
        self._update_size()

    def show_image(self, image: Image | None):
        self._image = image
        self._update_size()
        self.update()

    def _settings_changed(self, name: str, value: object):
        if name == "preview_reel_size":
            self._update_size()

    def _update_size(self):
        height = theme.screen_scale(
            self, QSize(settings.preview_reel_size, settings.preview_reel_size)
        ).height()
        width = (
            max(height, round(height * self._image.width / self._image.height))
            if self._image
            else height
        )
        self.setFixedSize(width + 6, height + 6)

    def paintEvent(self, a0):
        painter = QPainter(self)
        rect = QRectF(self.rect()).adjusted(3, 3, -3, -3)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0, PreviewReel._background_top)
        gradient.setColorAt(1, PreviewReel._background_bottom)
        painter.fillRect(rect, gradient)
        if self._image:
            size = QSize(self._image.width, self._image.height).scaled(
                rect.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio
            )
            target = QRectF(QPointF(), size.toSizeF())
            target.moveCenter(rect.center())
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(target, self._image._qimage)
        if self.underMouse():
            painter.fillRect(rect, PreviewReel._active_overlay)
            painter.setPen(QPen(PreviewReel._active_border, 2))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))

    def enterEvent(self, event):
        self.update()

    def leaveEvent(self, a0):
        self.update()

    def mouseReleaseEvent(self, a0):
        if a0 and a0.button() == Qt.MouseButton.LeftButton and self.rect().contains(a0.pos()):
            if window := Krita.instance().activeWindow():
                for docker in window.dockers():
                    if docker.objectName() == "aiLivePreview":
                        docker.show()
                        docker.raise_()
                        if isinstance(area := docker.widget(), LivePreviewArea):
                            QTimer.singleShot(0, area.fit)
                        break


class LiveControls(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._model: DocumentModel | None = None
        self._bindings: list[QMetaObject.Connection | Binding] = []
        self.thumbnail = LiveThumbnail(self)
        self.apply_button = QToolButton(self)
        self.apply_button.setText(_("Apply"))
        self.apply_button.setIcon(theme.icon("apply"))
        self.apply_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.apply_button.setToolTip(_("Copy the current result to the active layer"))
        self.apply_layer_button = QToolButton(self)
        self.apply_layer_button.setText(_("Apply as layer"))
        self.apply_layer_button.setIcon(theme.icon("apply-layer"))
        self.apply_layer_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.apply_layer_button.setToolTip(_("Create a new layer with the current result"))
        self.seed_controls = SeedWidget(self)
        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.apply_layer_button)
        buttons.addStretch()
        controls = QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addLayout(buttons)
        controls.addWidget(self.seed_controls)
        controls.addStretch()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.thumbnail, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(controls, 1)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        Binding.disconnect_all(self._bindings)
        self._model = model
        self.seed_controls.model = model
        self._bindings = [
            model.live.result_available.connect(self.thumbnail.show_image),
            model.live.has_result_changed.connect(self.apply_button.setEnabled),
            model.live.has_result_changed.connect(self.apply_layer_button.setEnabled),
            self.apply_button.clicked.connect(lambda: model.live.apply_result()),
            self.apply_layer_button.clicked.connect(
                lambda: model.live.apply_result(layer_only=True)
            ),
        ]
        self.apply_button.setEnabled(model.live.has_result)
        self.apply_layer_button.setEnabled(model.live.has_result)
        self.thumbnail.show_image(model.live.result_composition)


class ImagePreviewArea(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._image: Image | None = None
        self.setMinimumSize(256, 256)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def show_image(self, image: Image):
        self._image = image
        self.update()

    def paintEvent(self, a0):
        if self._image:
            zoom = min(self.width() / self._image.width, self.height() / self._image.height)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.scale(zoom, zoom)
            painter.drawImage(0, 0, self._image._qimage)


class LivePreviewArea(QWidget):
    zoom_steps = (
        0.125,
        0.167,
        0.25,
        0.33,
        0.5,
        0.67,
        1.0,
        1.33,
        2.0,
        2.67,
        4.0,
        5.33,
        8.0,
        10.67,
        16.0,
        21.33,
        32.0,
        42.67,
        64.0,
    )

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._model = root.active_model
        self._bindings: list[QMetaObject.Connection | Binding] = []
        self._image: Image | None = None
        self._space = False
        self._drag: str | None = None
        self._last_pos = QPointF()
        self._tracked: QTransform | None = None
        self._tracking = False
        self._view: View | None = None

        self.setMinimumSize(256, 256)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self.controls = QWidget(self)
        row = QHBoxLayout(self.controls)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(2)

        self.zoom_out = self._button("−", lambda: self.step_zoom(-1))
        self.zoom_label = self._button(_("Zoom: 100%"), lambda: self.set_zoom(1.0))
        self.zoom_in = self._button("+", lambda: self.step_zoom(1))

        self.fit_button = self._button("", self.fit)
        self.fit_button.setIcon(theme.icon("zoom-fit"))
        self.fit_button.setToolTip(_("Fit full image"))

        self.track_check = QCheckBox(_("Track"), self.controls)
        self.track_check.setToolTip(_("Track Krita's current view"))

        for control in (
            self.zoom_out,
            self.zoom_label,
            self.zoom_in,
            self.fit_button,
            self.track_check,
        ):
            row.addWidget(control)
        base = self.palette().color(QPalette.ColorRole.Base)
        self.controls.setStyleSheet(
            "QToolButton, QCheckBox { background-color: "
            f"rgba({base.red()}, {base.green()}, {base.blue()}, 230);"
            "padding: 3px; border-radius: 3px; }"
            f"QToolButton:hover {{ color: {theme.highlight}; }}"
        )
        self.controls.hide()

        self.spinner = SpinnerWidget(self)
        self.spinner.hide()

        self.setMinimumWidth(max(256, self.controls.sizeHint().width() + self.spinner.width() + 8))
        self._poller = QTimer(self)
        self._poller.setInterval(20)
        self._poller.timeout.connect(self._track_view)

        self.model = self._model

    def _button(self, text: str, callback):
        button = QToolButton(self.controls)
        button.setText(text)
        button.clicked.connect(callback)
        return button

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        Binding.disconnect_all(self._bindings)
        self._model = model
        self._bindings = [
            model.live.result_available.connect(self.show_image),
            model.progress_changed.connect(self._update_progress),
            model.live.is_active_changed.connect(self._update_progress),
            model.live.zoom_changed.connect(self._view_changed),
            model.live.pan_x_changed.connect(self._view_changed),
            model.live.pan_y_changed.connect(self._view_changed),
            model.live.rotation_changed.connect(self._view_changed),
            model.live.mirrored_changed.connect(self._view_changed),
            model.live.track_changed.connect(self._track_changed),
            bind_toggle(model.live, "track", self.track_check),
        ]
        self._image = model.live.result_composition
        self._tracked = None
        self._view_changed()
        self._update_progress()

    def show_image(self, image: Image):
        self._image = image
        self.spinner.stop_animation()
        self.update()

    def _update_progress(self):
        if self._model.live.is_active and self._model.progress < 1:
            self.spinner.set_progress(max(0, self._model.progress))
            self.spinner.start_animation()
        else:
            self.spinner.stop_animation()

    def _view_changed(self):
        if self._model.live.track and self.isVisible() and self._view is not None:
            if not self._poller.isActive():
                self._poller.start()
        else:
            self._poller.stop()
        self._update_zoom_label()
        self.update()

    def _track_changed(self, active: bool):
        live = self._model.live
        if active:
            self._tracked = self._read_view_transform()
            live.zoom = 1.0
            live.pan_x = 0.0
            live.pan_y = 0.0
            live.rotation = 0.0
            live.mirrored = False
            self.fit()
        else:
            combined = self._transform()
            self._tracked = None
            self._set_transform(combined)
        self._view_changed()

    def _update_zoom_label(self):
        text = _("Zoom: {0}%").format(round(self._effective_zoom() * 100))
        if self.zoom_label.text() != text:
            self.zoom_label.setText(text)
            self.controls.adjustSize()

    def _effective_zoom(self):
        transform = self._transform()
        return hypot(transform.m11(), transform.m12())

    def _image_center(self):
        extent = self._image.extent if self._image else self._model.document.extent
        return QPointF(extent.width / 2, extent.height / 2)

    def _base_transform(self):
        if self._tracked is not None:
            return self._tracked
        center = self._image_center()
        return QTransform.fromTranslate(
            self.width() / 2 - center.x(), self.height() / 2 - center.y()
        )

    def _manual_transform(self):
        live = self._model.live
        transform = QTransform()
        transform.translate(self.width() / 2, self.height() / 2)
        transform.scale(live.zoom, live.zoom)
        transform.rotate(live.rotation)
        transform.scale(1, -1 if live.mirrored else 1)
        transform.translate(live.pan_x - self.width() / 2, live.pan_y - self.height() / 2)
        return transform

    def _transform(self):
        # QTransform multiplication applies the left-hand transform first.
        return self._base_transform() * self._manual_transform()

    def _set_transform(self, combined: QTransform):
        inverse, valid = self._base_transform().inverted()
        if not valid:
            return
        manual = inverse * combined
        linear = QTransform(manual.m11(), manual.m12(), manual.m21(), manual.m22(), 0, 0)
        inverse_linear, valid = linear.inverted()
        if not valid:
            return
        center = QPointF(self.width() / 2, self.height() / 2)
        pan = inverse_linear.map(manual.map(center) - center)
        live = self._model.live
        live.zoom = hypot(manual.m11(), manual.m12())
        live.rotation = degrees(atan2(manual.m12(), manual.m11()))
        live.mirrored = manual.determinant() < 0
        live.pan_x = pan.x()
        live.pan_y = pan.y()

    def _track_view(self):
        model = self._model
        if self._tracking or not self.isVisible() or not model.live.track:
            return
        # Krita can run a nested event loop during a view update. Never sample
        # the view recursively, or synchronously from a Qt resize handler.
        self._tracking = True
        try:
            transform = self._read_view_transform()
            if self._model is not model or not model.live.track:
                return
            if transform is not None and transform != self._tracked:
                self._tracked = transform
                self._update_zoom_label()
                self.update()
        finally:
            self._tracking = False

    def set_canvas(self, canvas: Canvas | None):
        self._poller.stop()
        self._view = canvas.view() if canvas else None
        self._tracked = None
        self._view_changed()

    def _read_view_transform(self) -> QTransform | None:
        view = self._view
        if view is None:
            return
        image_to_flake, valid = view.flakeToImageTransform().inverted()
        if not valid:
            return
        transform = image_to_flake * view.flakeToCanvasTransform()
        return transform if transform.isInvertible() else None

    def set_zoom(self, zoom: float, anchor: QPointF | None = None):
        live = self._model.live
        zoom = max(0.0001, min(self.zoom_steps[-1], zoom))
        base = self._base_transform()
        zoom /= hypot(base.m11(), base.m12())
        if anchor is not None:
            inverse, valid = self._manual_transform().inverted()
            if valid:
                center = QPointF(self.width() / 2, self.height() / 2)
                offset = inverse.map(anchor) - inverse.map(center)
                offset *= live.zoom / zoom - 1
                live.pan_x += offset.x()
                live.pan_y += offset.y()
        live.zoom = zoom

    def step_zoom(self, direction: int, anchor: QPointF | None = None):
        zoom = self._effective_zoom()
        steps = self.zoom_steps if direction > 0 else reversed(self.zoom_steps)
        target = next((s for s in steps if (s - zoom) * direction > 0.001), zoom)
        self.set_zoom(target, anchor)

    def fit(self):
        if self._image:
            zoom = min(self.width() / self._image.width, self.height() / self._image.height)
            center = self._image_center()
            transform = QTransform.fromScale(zoom, zoom)
            transform *= QTransform.fromTranslate(
                self.width() / 2 - zoom * center.x(), self.height() / 2 - zoom * center.y()
            )
            self._set_transform(transform)

    def paintEvent(self, a0):
        if self._image:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.setTransform(self._transform())
            painter.drawImage(0, 0, self._image._qimage)

    def resizeEvent(self, a0):
        self.controls.move(0, 0)
        self.spinner.move(self.width() - self.spinner.width() - 4, 4)
        super().resizeEvent(a0)

    def showEvent(self, a0):
        self._view_changed()
        super().showEvent(a0)

    def hideEvent(self, a0):
        self._poller.stop()
        self._space = False
        self._drag = None
        super().hideEvent(a0)

    def enterEvent(self, event):
        self.controls.show()
        self.controls.adjustSize()
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().enterEvent(event)

    def leaveEvent(self, a0):
        if not self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            self.controls.hide()
        super().leaveEvent(a0)

    def keyPressEvent(self, a0):
        if a0 and a0.key() == Qt.Key.Key_Space:
            self._space = True
            a0.accept()
        else:
            super().keyPressEvent(a0)

    def keyReleaseEvent(self, a0):
        if a0 and a0.key() == Qt.Key.Key_Space and self._space:
            self._space = False
            a0.accept()
        else:
            super().keyReleaseEvent(a0)

    def focusOutEvent(self, a0):
        self._space = False
        self._drag = None
        self.unsetCursor()
        super().focusOutEvent(a0)

    def wheelEvent(self, a0):
        if a0 is None:
            return
        if a0.angleDelta().y():
            self.step_zoom(1 if a0.angleDelta().y() > 0 else -1, a0.position())
        a0.accept()

    def mousePressEvent(self, a0):
        if a0 is None:
            return
        if a0.button() == Qt.MouseButton.MiddleButton:
            self._drag = "pan"
        elif a0.button() == Qt.MouseButton.LeftButton and self._space:
            self._drag = "zoom" if a0.modifiers() & Qt.KeyboardModifier.ControlModifier else "pan"
        self._last_pos = a0.position()
        if self._drag:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            a0.accept()
        else:
            super().mousePressEvent(a0)

    def mouseMoveEvent(self, a0):
        if a0 is None:
            return
        if self._drag:
            delta = a0.position() - self._last_pos
            live = self._model.live
            if self._drag == "pan":
                inverse, valid = self._manual_transform().inverted()
                if valid:
                    delta = inverse.map(delta) - inverse.map(QPointF())
                    live.pan_x += delta.x()
                    live.pan_y += delta.y()
            else:
                self.set_zoom(self._effective_zoom() * 2 ** (-delta.y() / 150), self._last_pos)
            self._last_pos = a0.position()
            a0.accept()
        else:
            super().mouseMoveEvent(a0)

    def mouseReleaseEvent(self, a0):
        self._drag = None
        self.unsetCursor()
        super().mouseReleaseEvent(a0)


class LivePreviewDocker(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("AI Live Preview"))
        self.preview_area = LivePreviewArea(self)
        self.setWidget(self.preview_area)
        root.model_created.connect(self._register_model)
        self.visibilityChanged.connect(self._update_model)

    def _register_model(self, model: DocumentModel):
        self._update_model()

    def _update_model(self):
        model = root.active_model
        if self.preview_area.model is not model:
            self.preview_area.model = model

    def canvasChanged(self, canvas: Canvas | None):
        # Krita owns this wrapper and may delete it when the callback returns.
        self.preview_area.set_canvas(canvas)
        QTimer.singleShot(0, self._update_model)
