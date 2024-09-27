from copy import copy
from enum import Enum
from typing import Any, NamedTuple, Sequence, TypeVar, Generic

from PyQt5.QtCore import QObject, QMetaObject, QUuid, pyqtBoundSignal
from PyQt5.QtWidgets import QComboBox


T = TypeVar("T")


class ObservableProperties:
    """Provides default implementations for properties (get, set, signal) to sub-classes."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        properties = {
            name: attr for name, attr in cls.__dict__.items() if isinstance(attr, Property)
        }
        for name, property in properties.items():
            setattr(cls, f"_{name}", _copy_reference_types(property.default_value))
            getter, setter = None, None
            if property.getter is not None:
                getter = getattr(cls, property.getter)
            if property.setter is not None:
                setter = getattr(cls, property.setter)
            setattr(cls, name, PropertyImpl(name, getter, setter, property.persist))


class Property(Generic[T]):
    """Property definition. Will be replaced with with PropertyImpl at instance creation."""

    default_value: T
    getter = None
    setter = None
    persist = False

    def __init__(self, default_value: T, getter=None, setter=None, persist=False):
        self.default_value = default_value
        self.getter = getter
        self.setter = setter
        self.persist = persist

    def __get__(self, instance, owner) -> T: ...
    def __set__(self, instance, value: T): ...
    def __delete__(self, instance): ...


class PropertyImpl(property):
    """Property implementation: gets/sets a value, and emits a signal when it changes."""

    name: str
    persist: bool

    def __init__(self, name: str, getter, setter, persist: bool):
        super().__init__(getter or self._getter, setter or self._setter)
        self.name = name
        self.persist = persist

    def _getter(self, instance):
        return getattr(instance, f"_{self.name}")

    def _setter(self, instance, value):
        previous = getattr(instance, f"_{self.name}")
        if previous == value:
            return

        setattr(instance, f"_{self.name}", value)
        signal = getattr(instance, f"{self.name}_changed")
        signal.emit(value)

        if self.persist:
            if modified_signal := getattr(instance, "modified", None):
                modified_signal.emit(instance, self.name)


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
        if combo.count() > 0:
            setattr(model, model_property, combo.currentData())

    model_to_widget = _signal(model, model_property).connect(set_combo)
    set_combo(getattr(model, model_property))
    if mode is Bind.one_way:
        return model_to_widget
    else:
        widget_to_model = combo.currentIndexChanged.connect(set_model)
        return Binding(model_to_widget, widget_to_model)


def bind_toggle(model, model_property: str, widget, mode=Bind.two_way):
    widget_setter = _setter(widget, "checked")
    model_to_widget = _signal(model, model_property).connect(widget_setter)
    widget_setter(getattr(model, model_property))
    if mode is Bind.one_way:
        return model_to_widget
    else:
        widget_to_model = widget.toggled.connect(_setter(model, model_property))
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


def is_persistent(obj: QObject, name: str):
    if prop := obj.__class__.__dict__.get(name, None):
        return isinstance(prop, PropertyImpl) and prop.persist
    return False


def _default_serializer(value):
    return value


def _default_deserializer(type, value):
    return value


def serialize(obj: QObject, converter=_default_serializer):
    def _serialize(propt: str):
        value = getattr(obj, propt)
        if isinstance(value, Enum):
            return value.value
        elif isinstance(value, QUuid):
            return value.toString()
        return converter(value)

    return {name: _serialize(name) for name in dir(obj.__class__) if is_persistent(obj, name)}


def deserialize(obj: QObject, data: dict[str, Any], converter=_default_deserializer):
    for name, value in data.items():
        if is_persistent(obj, name):
            current = getattr(obj, name, None)
            if isinstance(current, Enum):
                value = current.__class__(value)
            elif isinstance(current, QUuid):
                value = QUuid(value)
            value = converter(type(current), value)
            if not isinstance(value, type(current)):
                raise TypeError(f"{name} was '{value}', but expected {type(current)}")
            setattr(obj, name, value)


def _copy_reference_types(object):
    if isinstance(object, (list, dict)):
        return copy(object)
    return object
