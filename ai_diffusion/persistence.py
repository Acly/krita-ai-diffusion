import json
from typing import Any
from PyQt5.QtCore import QObject, QByteArray
from PyQt5.QtWidgets import QMessageBox

from .model import Model
from .style import Style, Styles
from .properties import serialize, deserialize
from .util import client_logger as log

# Version of the persistence format, increment when there are breaking changes
version = 1


def _serialize(obj: QObject):
    def converter(obj):
        if isinstance(obj, Style):
            return obj.filename
        return obj

    return serialize(obj, converter)


def _deserialize(obj: QObject, data: dict[str, Any]):
    def converter(type, value):
        if type is Style:
            style = Styles.list().find(value)[0]
            return style or Styles.list().default
        return value

    return deserialize(obj, data, converter)


def save(model: Model, _: str = ""):
    state = _serialize(model)
    state["version"] = version
    state["upscale"] = _serialize(model.upscale)
    state["live"] = _serialize(model.live)
    state["control"] = [_serialize(c) for c in model.control]
    state_str = json.dumps(state, indent=2)
    state_bytes = QByteArray(state_str.encode("utf-8"))
    model.document.annotate("ui", state_bytes)


def load(model: Model):
    if state_bytes := model.document.find_annotation("ui"):
        try:
            state = json.loads(state_bytes.data().decode("utf-8"))
            _deserialize(model, state)
            _deserialize(model.upscale, state.get("upscale", {}))
            _deserialize(model.live, state.get("live", {}))
            for control_state in state.get("control", []):
                control = model.control.add()
                _deserialize(control, control_state)
        except Exception as e:
            msg = f"Failed to load state from {model.document.filename}: {e}"
            log.warning(msg)
            QMessageBox.warning(None, "AI Diffusion Plugin", msg)


def track(model: Model):
    def save_model():
        save(model)

    def track_control(control):
        save(model)
        control.modified.connect(save_model)

    model.modified.connect(save)
    model.upscale.modified.connect(save_model)
    model.live.modified.connect(save_model)
    model.control.added.connect(track_control)
    model.control.removed.connect(save_model)
    for control in model.control:
        track_control(control)
