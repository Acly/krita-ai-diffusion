"""Toggle switch widget
from https://stackoverflow.com/a/51825815
"""

from PyQt5.QtCore import QPropertyAnimation, QSize, Qt, pyqtProperty  # type: ignore
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QAbstractButton, QSizePolicy


class SwitchWidget(QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._thumb_radius = self.fontMetrics().height() // 2
        self._track_radius = self._thumb_radius + 2

        self._margin = max(0, self._thumb_radius - self._track_radius)
        self._base_offset = max(self._thumb_radius, self._track_radius)
        self._end_offset = {
            True: lambda: self.width() - self._base_offset,
            False: lambda: self._base_offset,
        }
        self._offset = self._base_offset

        palette = self.palette()
        self._thumb_color = {
            True: palette.text(),
            False: palette.light(),
        }
        self._track_color = {
            True: palette.highlight(),
            False: palette.dark(),
        }
        self._text_color = {
            True: palette.highlight().color(),
            False: palette.dark().color(),
        }
        self._track_opacity = 1

    @pyqtProperty(int)
    def offset(self):  # type: ignore
        return self._offset

    @offset.setter  # type: ignore
    def offset(self, value):
        self._offset = value
        self.update()

    def sizeHint(self):
        return QSize(
            4 * self._track_radius + 2 * self._margin,
            2 * self._track_radius + 2 * self._margin,
        )

    @property
    def is_checked(self):
        return self.isChecked()

    @is_checked.setter
    def is_checked(self, checked):
        super().setChecked(checked)
        self.offset = self._end_offset[checked]()

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self.offset = self._end_offset[self.isChecked()]()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        track_opacity = self._track_opacity
        thumb_opacity = 1.0
        if self.isEnabled():
            track_brush = self._track_color[self.isChecked()]
            thumb_brush = self._thumb_color[self.isChecked()]
        else:
            track_opacity *= 0.8
            track_brush = self.palette().shadow()
            thumb_brush = self.palette().mid()

        p.setBrush(track_brush)
        p.setOpacity(track_opacity)
        p.drawRoundedRect(
            self._margin,
            self._margin,
            self.width() - 2 * self._margin,
            self.height() - 2 * self._margin,
            self._track_radius,
            self._track_radius,
        )
        p.setBrush(thumb_brush)
        p.setOpacity(thumb_opacity)
        p.drawEllipse(
            self.offset - self._thumb_radius,
            self._base_offset - self._thumb_radius,
            2 * self._thumb_radius,
            2 * self._thumb_radius,
        )

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if e and e.button() == Qt.MouseButton.LeftButton:
            anim = QPropertyAnimation(self, b"offset", self)
            anim.setDuration(120)
            anim.setStartValue(self.offset)
            anim.setEndValue(self._end_offset[self.isChecked()]())
            anim.start()
