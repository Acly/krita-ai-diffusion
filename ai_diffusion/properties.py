from enum import Enum
from typing import NamedTuple, Sequence, TypeVar, Generic

from PyQt5.QtCore import QObject, QMetaObject, pyqtSignal, pyqtBoundSignal, pyqtProperty  # type: ignore
from PyQt5.QtWidgets import QComboBox


T = TypeVar("T")


class PropertyMeta(type(QObject)):
    """Provides default implementations for properties (get, set, signal)."""

    def __new__(cls, name, bases, attrs):
        for key in list(attrs.keys()):
            attr = attrs[key]
            if not isinstance(attr, Property):
                continue

            attrs[f"_{key}"] = attr._default_value
            getter, setter = None, None
            if attr._getter is not None:
                getter = attrs[attr._getter]
            if attr._setter is not None:
                setter = attrs[attr._setter]
            attrs[key] = PropertyImpl(key, getter, setter)

        return super().__new__(cls, name, bases, attrs)


class Property(Generic[T]):
    """Property definition. Will be replaced with with PropertyImpl at instance creation."""

    _default_value: T
    _getter = None
    _setter = None

    def __init__(self, default_value: T, getter=None, setter=None):
        self._default_value = default_value
        self._getter = getter
        self._setter = setter

    def __get__(self, instance, owner) -> T: ...
    def __set__(self, instance, value: T): ...
    def __delete__(self, instance): ...


class PropertyImpl(property):
    """Property implementation: gets, sets, and notifies of change."""

    name: str

    def __init__(self, name: str, getter=None, setter=None):
        super().__init__(getter or self.getter, setter or self.setter)
        self.name = name

    def getter(self, instance):
        return getattr(instance, f"_{self.name}")

    def setter(self, instance, value):
        previous = getattr(instance, f"_{self.name}")
        if previous == value:
            return

        setattr(instance, f"_{self.name}", value)
        signal = getattr(instance, f"{self.name}_changed")
        signal.emit(value)


class Binding(NamedTuple):
    model_connection: QMetaObject.Connection
    widget_connection: QMetaObject.Connection

    def disconnect(self):
        QObject.disconnect(self.model_connection)
        QObject.disconnect(self.widget_connection)

    @staticmethod
    def disconnect_all(bindings: Sequence["QMetaObject.Connection | Binding"]):
        for binding in bindings:
            if isinstance(binding, Binding):
                binding.disconnect()
            else:
                QObject.disconnect(binding)


class Bind(Enum):
    one_way = 1
    two_way = 2


def bind(model, model_property: str, widget, widget_property: str, mode=Bind.two_way):
    # model change -> update widget
    widget_setter = _setter(widget, widget_property)
    model_to_widget = _signal(model, model_property).connect(widget_setter)

    # set initial value from model
    widget_setter(getattr(model, model_property))

    if mode is Bind.one_way:
        return model_to_widget
    else:
        # widget change -> update model
        widget_to_model = _signal(widget, widget_property).connect(_setter(model, model_property))
        return Binding(model_to_widget, widget_to_model)


def bind_combo(model, model_property: str, combo: QComboBox, mode=Bind.two_way):
    def set_combo(value):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def set_model(index):
        setattr(model, model_property, combo.currentData())

    model_to_widget = _signal(model, model_property).connect(set_combo)
    set_combo(getattr(model, model_property))
    if mode is Bind.one_way:
        return model_to_widget
    else:
        widget_to_model = combo.currentIndexChanged.connect(set_model)
        return Binding(model_to_widget, widget_to_model)


def _signal(inst, property: str) -> pyqtBoundSignal:
    if hasattr(inst, f"{property}_changed"):
        return getattr(inst, f"{property}_changed")
    else:
        return getattr(inst, f"{property}Changed")


def _setter(inst, property: str):
    def set_py(value):
        setattr(inst, property, value)

    qt_setter_name = f"set{property.capitalize()}"
    if hasattr(inst, qt_setter_name):
        return getattr(inst, qt_setter_name)
    else:
        return set_py
