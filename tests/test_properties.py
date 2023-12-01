from enum import Enum
from ai_diffusion.properties import Property, PropertyMeta, bind
from PyQt5.QtCore import QObject, pyqtBoundSignal, pyqtSignal


class Piong(Enum):
    a = 1
    b = 2


class ObjectWithProperties(QObject, metaclass=PropertyMeta):
    inty = Property(0)
    stringy = Property("")
    enumy = Property(Piong.a)
    custom = Property(3, getter="get_custom", setter="set_custom")
    _qtstyle = 99

    inty_changed = pyqtSignal(int)
    stringy_changed = pyqtSignal(str)
    enumy_changed = pyqtSignal(Piong)
    custom_changed = pyqtSignal(int)
    qtstyleChanged = pyqtSignal(int)

    def __init__(self):
        super().__init__()

    def get_custom(self):
        return self._custom + 10

    def set_custom(self, value: int):
        self._custom = value + 1
        self.custom_changed.emit(self._custom)

    def qtstyle(self):
        return self._qtstyle

    def setQtstyle(self, value: int):
        self._qtstyle = value
        self.qtstyleChanged.emit(self._qtstyle)


def test_property():
    called = []

    def callback(x):
        called.append(x)

    t = ObjectWithProperties()
    t.inty_changed.connect(callback)
    t.inty = 42
    assert t.inty == 42

    t.stringy_changed.connect(callback)
    t.stringy = "hello"
    assert t.stringy == "hello"

    t.enumy_changed.connect(callback)
    t.enumy = Piong.b
    assert t.enumy == Piong.b

    assert t.custom == 13
    t.custom_changed.connect(callback)
    t.custom = 4
    assert t.custom == 15

    assert t.qtstyle() == 99
    t.qtstyleChanged.connect(callback)
    t.setQtstyle(55)
    assert t.qtstyle() == 55

    assert called == [42, "hello", Piong.b, 5, 55]


def test_bind():
    a = ObjectWithProperties()
    b = ObjectWithProperties()

    bind(a, "inty", b, "inty")
    a.inty = 5
    assert a.inty == b.inty
    assert b.inty == 5
    b.inty = 99
    assert a.inty == b.inty
    assert a.inty == 99


def test_bind_property_to_qt():
    a = ObjectWithProperties()
    b = ObjectWithProperties()

    bind(a, "inty", b, "qtstyle")
    a.inty = 5
    assert a.inty == b.qtstyle()
    assert b.qtstyle() == 5
    b.setQtstyle(99)
    assert a.inty == b.qtstyle()
    assert a.inty == 99


def test_bind_qt_to_property():
    a = ObjectWithProperties()
    b = ObjectWithProperties()

    bind(a, "qtstyle", b, "inty")
    a.setQtstyle(5)
    assert a.qtstyle() == b.inty
    assert b.inty == 5
    b.inty = 99
    assert a.qtstyle() == b.inty
    assert a.qtstyle() == 99
