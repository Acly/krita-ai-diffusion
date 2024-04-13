# From https://github.com/biolab/orange3
#
# Copyright (c) 2016 Bioinformatics Laboratory, University of Ljubljana,
#   Faculty of Computer and Information Science
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT ANY WARRANTY WHATSOEVER.
#
# If you use or redistribute this software, you are permitted to do so
# under the terms of GNU [GPL-3.0]+ license.
#
# [GPL-3.0]: https://www.gnu.org/licenses/gpl-3.0.en.html

from typing import Tuple

from PyQt5.QtCore import pyqtSignal as Signal

from PyQt5.QtWidgets import QWidget, QStyleOptionSlider, QSizePolicy, QStyle, QSlider
from PyQt5.QtGui import QPainter, QMouseEvent, QPalette, QBrush
from PyQt5.QtCore import QRect, Qt, QSize


# Based on idea and in part the code from
# https://stackoverflow.com/questions/47342158/porting-range-slider-widget-to-pyqt5


class IntervalSlider(QWidget):
    """
    Slider with two handles for setting an interval of values.

    Only horizontal orientation is supported.

    Signals:

    rangeChanged(minimum: int, maximum: int):
        minumum or maximum or both have changed

    intervalChanged(low: int, high: int):
        One or both boundaries have been changed.The tracking determines
        whether this signal is emitted during user interaction or only when
        the mouse is released

    sliderPressed(id: int)
        The user started dragging.
        Id is IntervarlSlider.LowHandle or IntervalSlider.HighHandle

    sliderMoved(id: int, value: int)
        the user drags the slider
        Id is IntervarlSlider.LowHandle or IntervalSlider.HighHandle

    sliderReleased(id: int)
        The user finished dragging.
        Id is IntervarlSlider.LowHandle or IntervalSlider.HighHandle

    """

    NoHandle, LowHandle, HighHandle = 0, 1, 2

    rangeChanged = Signal((int, int))  # notifier for setMinimum/setMaximum
    intervalChanged = Signal((int, int))  # setRange, slider move (when tracking)
    sliderPressed = Signal(int)  # argument is handle id (see above)
    sliderMoved = Signal(int, int)  #
    sliderReleased = Signal(int)

    def __init__(self, low=1, high=8, minimum=0, maximum=10, parent: QWidget | None = None, **args):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed, QSizePolicy.Slider)
        )

        self._dragged = self.NoHandle
        self._start_drag_value = None

        self.opt = QStyleOptionSlider()
        self._interval = self._pos = (low, high)
        self.opt.minimum = minimum
        self.opt.maximum = maximum
        self._tracking = True
        for opt, value in args.items():
            getattr(self, f"set{opt[0].upper()}{opt[1:]}")(value)

    # Properties

    def orientation(self):
        """Orientation. Always Qt.Horizontal"""
        return Qt.Orientation.Horizontal

    def setOrientation(self, orientation):
        """Set orientation (to Qt.Horizontal)"""
        if orientation != Qt.Orientation.Horizontal:
            raise ValueError("IntervalSlider supports only horizontal direction")

    def sliderPosition(self) -> Tuple[int, int]:
        """
        Current position of sliders.
        This is the same as `interval` if tracking is enabled.
        """
        return self._pos

    def setSliderPosition(self, low: int, high: int) -> None:
        """
        Set position of sliders.

        This also changes `interval` if tracking is enabled.
        """
        self._pos = (low, high)
        self.update()
        if self._tracking:
            self._interval = self._pos
            self.intervalChanged.emit(low, high)
        self.sliderMoved.emit(self.LowHandle, low)
        self.sliderMoved.emit(self.HighHandle, high)

    def interval(self) -> Tuple[int, int]:
        """Current interval"""
        return self._interval

    def setInterval(self, low: int, high: int) -> None:
        """Set the current interval"""
        self._interval = (low, high)
        self.setSliderPosition(low, high)

    def low(self) -> int:
        """The lower bound of the interavl"""
        return self._interval[0]

    def setLow(self, low: int) -> None:
        """Set the lower bound"""
        self.setInterval(low, self.high())

    def high(self) -> int:
        """The higher bound of the interval"""
        return self._interval[1]

    def setHigh(self, high: int) -> None:
        """Set the higher bound of the interval"""
        self.setInterval(self.low(), high)

    def setMinimum(self, minimum: int) -> None:
        """Set the minimal value of the lower bound"""
        self.opt.minimum = minimum
        self.rangeChanged.emit(minimum, self.opt.maximum)

    def setMaximum(self, maximum: int) -> None:
        """Set the maximal value of the higher bound"""
        self.opt.maximum = maximum
        self.rangeChanged.emit(self.opt.minimum, maximum)

    def setTickPosition(self, position: QSlider.TickPosition) -> None:
        """See QSlider.setTickPosition"""
        self.opt.tickPosition = position

    def setTickInterval(self, ti: int) -> None:
        """See QSlider.setInterval"""
        self.opt.tickInterval = ti

    def setTracking(self, enabled: bool) -> None:
        """
        If enabled, interval is changed during user interaction and the
        notifier signal is emitted immediately.
        """
        self._tracking = enabled

    def hasTracking(self) -> bool:
        return self._tracking

    # Mouse events

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        style = self.style()
        assert style is not None and a0 is not None
        self._dragged = self.NoHandle
        args = (QStyle.ComplexControl.CC_Slider, self.opt, a0.pos(), self)
        self.opt.sliderPosition = self._pos[0]
        if style.hitTestComplexControl(*args) == QStyle.SubControl.SC_SliderHandle:
            self._dragged |= self.LowHandle
        self.opt.sliderPosition = self._pos[1]
        if style.hitTestComplexControl(*args) == QStyle.SubControl.SC_SliderHandle:
            self._dragged |= self.HighHandle
        if self._dragged != self.NoHandle:
            self.sliderPressed.emit(self._dragged)

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        style = self.style()
        assert style is not None and a0 is not None
        distance = self.opt.maximum - self.opt.minimum
        pos = style.sliderValueFromPosition(0, distance, a0.pos().x(), self.rect().width())
        low, high = self._pos

        # If handles overlap, determine which one is dragged
        if self._dragged == self.LowHandle | self.HighHandle:
            if pos < high:
                self._dragged = self.LowHandle
            elif pos > low:
                self._dragged = self.HighHandle

        if self._dragged == self.LowHandle:
            low = min(pos, high)
        elif self._dragged == self.HighHandle:
            high = max(pos, low)

        if self._pos == (low, high):
            return

        self._pos = (low, high)
        self.update()
        if self._tracking:
            self._interval = self._pos
            self.intervalChanged.emit(*self._interval)

    def mouseReleaseEvent(self, a0) -> None:
        if self._dragged == self.NoHandle:
            return

        self.sliderReleased.emit(self._dragged)
        if self._interval != self._pos:
            self._interval = self._pos
            self.intervalChanged.emit(*self._interval)

    # Paint

    def paintEvent(self, a0) -> None:
        painter = QPainter(self)
        style = self.style()
        assert style is not None

        # Draw groove
        self.opt.initFrom(self)
        self.opt.rect = self.rect()
        self.opt.sliderPosition = 0
        self.opt.subControls = (
            QStyle.SubControl.SC_SliderGroove | QStyle.SubControl.SC_SliderTickmarks
        )
        style.drawComplexControl(QStyle.ComplexControl.CC_Slider, self.opt, painter)

        # Draw interval
        # Interval has an arbitrary width of 4; I don't know how to get the
        # actual groove thickness.
        color = self.palette().color(QPalette.Highlight)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)

        args = QStyle.ComplexControl.CC_Slider, self.opt, QStyle.SubControl.SC_SliderHandle
        self.opt.sliderPosition = self._pos[0]
        x_left_handle = style.subControlRect(*args).right()
        self.opt.sliderPosition = self._pos[1]
        x_right_handle = style.subControlRect(*args).left()

        groove_rect = style.subControlRect(
            QStyle.ComplexControl.CC_Slider, self.opt, QStyle.SubControl.SC_SliderGroove, self
        )
        selection = QRect(
            x_left_handle,
            groove_rect.y() + groove_rect.height() // 2 - 2,
            x_right_handle - x_left_handle,
            4,
        )
        painter.drawRect(selection)

        # Draw handles
        self.opt.subControls = QStyle.SubControl.SC_SliderHandle
        for self.opt.sliderPosition in self._pos:
            style.drawComplexControl(QStyle.ComplexControl.CC_Slider, self.opt, painter)

    def sizeHint(self) -> QSize:
        style = self.style()
        assert style is not None

        SliderLength = 84
        TickSpace = 5

        w = SliderLength
        h = style.pixelMetric(QStyle.PixelMetric.PM_SliderThickness, self.opt, self)
        if self.opt.tickPosition != QSlider.NoTicks:
            h += TickSpace

        return style.sizeFromContents(QStyle.ContentsType.CT_Slider, self.opt, QSize(w, h))
