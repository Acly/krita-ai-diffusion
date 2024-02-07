from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Any
from PyQt5.QtCore import QObject, QByteArray, QBuffer
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMessageBox

from .image import Bounds, Image, ImageCollection, ImageFileFormat
from .model import Model
from .control import ControlLayer
from .jobs import Job, JobKind, JobParams
from .style import Style, Styles
from .properties import serialize, deserialize
from .settings import settings
from .util import client_logger as log

# Version of the persistence format, increment when there are breaking changes
version = 1


@dataclass
class _HistoryResult:
    id: str
    slot: int  # annotation slot where images are stored
    offsets: list[int]  # offsets in bytes for result images
    params: JobParams

    @staticmethod
    def from_dict(data: dict[str, Any]):
        data["params"]["bounds"] = Bounds(*data["params"]["bounds"])
        data["params"] = JobParams(**data["params"])
        return _HistoryResult(**data)


class ModelSync:
    """Synchronizes the model with the document's annotations."""

    _model: Model
    _history: list[_HistoryResult]
    _memory_used: dict[int, int]  # slot -> memory used for images in bytes
    _slot_index = 0

    def __init__(self, model: Model):
        self._model = model
        self._history = []
        self._memory_used = {}
        if state_bytes := _find_annotation(model.document, "ui.json"):
            try:
                self._load(model, state_bytes.data())
            except Exception as e:
                msg = f"Failed to load state from {model.document.filename}: {e}"
                log.exception(msg)
                QMessageBox.warning(None, "AI Diffusion Plugin", msg)
        self._track(model)

    def _save(self):
        model = self._model
        state = _serialize(model)
        state["version"] = version
        state["inpaint"] = _serialize(model.inpaint)
        state["upscale"] = _serialize(model.upscale)
        state["live"] = _serialize(model.live)
        state["control"] = [_serialize(c) for c in model.control]
        state["history"] = [asdict(h) for h in self._history]
        state_str = json.dumps(state, indent=2)
        state_bytes = QByteArray(state_str.encode("utf-8"))
        model.document.annotate("ui.json", state_bytes)

    def _load(self, model: Model, state_bytes: bytes):
        state = json.loads(state_bytes.decode("utf-8"))
        _deserialize(model, state)
        _deserialize(model.inpaint, state.get("inpaint", {}))
        _deserialize(model.upscale, state.get("upscale", {}))
        _deserialize(model.live, state.get("live", {}))

        for control_state in state.get("control", []):
            model.control.add()
            _deserialize(model.control[-1], control_state)

        for result in state.get("history", []):
            item = _HistoryResult.from_dict(result)
            if images_bytes := _find_annotation(model.document, f"result{item.slot}.webp"):
                job = model.jobs.add_job(Job(item.id, JobKind.diffusion, item.params))
                results = _deserialize_images(images_bytes, item.offsets, item.slot)
                model.jobs.set_results(job, results)
                model.jobs.notify_finished(job)
                self._history.append(item)
                self._memory_used[item.slot] = images_bytes.size()
                self._slot_index = max(self._slot_index, item.slot + 1)

    def _track(self, model: Model):
        model.modified.connect(self._save)
        model.inpaint.modified.connect(self._save)
        model.upscale.modified.connect(self._save)
        model.live.modified.connect(self._save)
        model.control.added.connect(self._track_control)
        model.control.removed.connect(self._save)
        for control in model.control:
            self._track_control(control)
        model.jobs.job_finished.connect(self._save_results)

    def _track_control(self, control: ControlLayer):
        self._save()
        control.modified.connect(self._save)

    def _save_results(self, job: Job):
        if job.kind is JobKind.diffusion and len(job.results) > 0:
            slot = self._slot_index
            self._slot_index += 1
            image_data, image_offsets = _serialize_images(job.results)
            self._model.document.annotate(f"result{slot}.webp", image_data)
            self._history.append(_HistoryResult(job.id or "", slot, image_offsets, job.params))
            self._memory_used[slot] = image_data.size()
            self._prune()
            self._save()

    @property
    def memory_used(self):
        return sum(self._memory_used.values())

    def _prune(self):
        limit = settings.history_storage * 1024 * 1024
        used = self.memory_used
        while used > limit and len(self._history) > 0:
            slot = self._history.pop(0).slot
            self._model.document.remove_annotation(f"result{slot}.webp")
            used -= self._memory_used.pop(slot, 0)


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


def _serialize_images(images: ImageCollection):
    offsets = []
    data = QByteArray()
    result = QBuffer(data)
    result.open(QBuffer.OpenModeFlag.WriteOnly)
    for img in images:
        offsets.append(result.pos())
        img.write(result, ImageFileFormat.webp)
    result.close()
    return data, offsets


def _deserialize_images(data: QByteArray, offsets: list[int], slot: int):
    images = ImageCollection()
    buffer = QBuffer(data)
    buffer.open(QBuffer.OpenModeFlag.ReadOnly)
    for i, offset in enumerate(offsets):
        buffer.seek(offset)
        img = QImage()
        if img.load(buffer, "WEBP"):
            img.convertTo(QImage.Format.Format_ARGB32)
            images.append(Image(img))
        else:
            raise Exception(f"Failed to load image {i} in slot {slot} from buffer")
    buffer.close()
    return images


def _find_annotation(document, name: str):
    if result := document.find_annotation(name):
        return result
    without_ext = name.rsplit(".", 1)[0]
    if result := document.find_annotation(without_ext):
        return result
    return None
