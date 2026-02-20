from enum import Enum

import pytest
from PyQt5.QtCore import QObject, pyqtSignal

from ai_diffusion.properties import ObservableProperties, Property, bind, deserialize, serialize


class Piong(Enum):
    a = 1
    b = 2


class ObjectWithProperties(QObject, ObservableProperties):
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


def test_multiple():
    a = ObjectWithProperties()
    b = ObjectWithProperties()

    a.inty = 5
    b.inty = 99
    assert a.inty != b.inty


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


class PersistentObject(QObject, ObservableProperties):
    inty = Property(0, persist=True)
    stringy = Property("", persist=True)
    enumy = Property(Piong.a, persist=True)
    not_persistent = Property(99)
    not_property = "dummy"

    inty_changed = pyqtSignal(int)
    stringy_changed = pyqtSignal(str)
    enumy_changed = pyqtSignal(Piong)
    not_persistent_changed = pyqtSignal(int)
    modified = pyqtSignal(QObject, str)


def test_serialize():
    a = PersistentObject()
    a.inty = 5
    a.enumy = Piong.b

    state = serialize(a)
    assert state == {"inty": 5, "stringy": "", "enumy": 2}

    a.stringy = "hello"
    state = serialize(a)
    assert state == {"inty": 5, "stringy": "hello", "enumy": 2}

    b = PersistentObject()
    b.inty = 827
    deserialize(b, state)
    assert b.inty == 5
    assert b.stringy == "hello"
    assert b.enumy is Piong.b


def test_deserialize_non_existing():
    """Loading state with non-existing properties should not fail to
    support previous versions after properties have been removed.
    """
    a = PersistentObject()
    a.inty = 5

    state = {"inty": 99, "doesnt_exist": 28}
    deserialize(a, state)
    assert a.inty == 99


def test_deserialize_non_property():
    a = PersistentObject()
    a.inty = 5

    state = {"inty": 99, "not_property": 28}
    deserialize(a, state)
    assert a.inty == 99
    assert a.not_property == "dummy"


def test_serialize_custom():
    a = PersistentObject()
    a.stringy = "hello"

    def _serializer(obj):
        if isinstance(obj, str):
            return obj.upper()
        return obj

    def _deserializer(type, value):
        if type is str:
            return value.lower()
        return value

    state = serialize(a, _serializer)
    assert state == {"inty": 0, "stringy": "HELLO", "enumy": 1}

    b = PersistentObject()
    deserialize(b, state, _deserializer)
    assert b.stringy == "hello"


def test_type_mismatch():
    a = PersistentObject()
    state = {"inty": "hello"}
    with pytest.raises(TypeError):
        deserialize(a, state)


def test_modified():
    called = []

    def callback(obj, prop):
        called.append((obj, prop))

    a = PersistentObject()
    a.modified.connect(callback)
    a.inty = 5
    a.not_persistent = 5
    assert called == [(a, "inty")]
