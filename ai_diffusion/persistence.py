from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Any
from PyQt5.QtCore import QObject, QByteArray, QBuffer
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMessageBox

from .image import Bounds, Image, ImageCollection, ImageFileFormat
from .model import Model
from .control import ControlLayer, ControlLayerList
from .region import RootRegion, Region
from .jobs import Job, JobKind, JobParams, JobQueue
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
        data["params"] = JobParams.from_dict(data["params"])
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
        state["animation"] = _serialize(model.animation)
        state["history"] = [asdict(h) for h in self._history]
        state["root"] = _serialize(model.regions)
        state["control"] = [_serialize(c) for c in model.regions.control]
        state["regions"] = []
        for region in model.regions:
            state["regions"].append(_serialize(region))
            state["regions"][-1]["control"] = [_serialize(c) for c in region.control]
        state_str = json.dumps(state, indent=2)
        state_bytes = QByteArray(state_str.encode("utf-8"))
        model.document.annotate("ui.json", state_bytes)

    def _load(self, model: Model, state_bytes: bytes):
        state = json.loads(state_bytes.decode("utf-8"))
        _deserialize(model, state)
        _deserialize(model.inpaint, state.get("inpaint", {}))
        _deserialize(model.upscale, state.get("upscale", {}))
        _deserialize(model.live, state.get("live", {}))
        _deserialize(model.animation, state.get("animation", {}))
        _deserialize(model.regions, state.get("root", {}))
        for control_state in state.get("control", []):
            _deserialize(model.regions.control.emplace(), control_state)
        for region_state in state.get("regions", []):
            region = model.regions.emplace()
            _deserialize(region, region_state)
            for control_state in region_state.get("control", []):
                _deserialize(region.control.emplace(), control_state)

        for result in state.get("history", []):
            item = _HistoryResult.from_dict(result)
            if images_bytes := _find_annotation(model.document, f"result{item.slot}.webp"):
                job = model.jobs.add_job(Job(item.id, JobKind.diffusion, item.params))
                results = ImageCollection.from_bytes(images_bytes, item.offsets)
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
        model.animation.modified.connect(self._save)
        model.jobs.job_finished.connect(self._save_results)
        model.jobs.job_discarded.connect(self._remove_results)
        model.jobs.result_discarded.connect(self._remove_image)
        self._track_regions(model.regions)

    def _track_control(self, control: ControlLayer):
        self._save()
        control.modified.connect(self._save)

    def _track_control_layers(self, control_layers: ControlLayerList):
        control_layers.added.connect(self._track_control)
        control_layers.removed.connect(self._save)
        for control in control_layers:
            self._track_control(control)

    def _track_region(self, region: Region):
        region.modified.connect(self._save)
        self._track_control_layers(region.control)

    def _track_regions(self, root_region: RootRegion):
        root_region.added.connect(self._track_region)
        root_region.removed.connect(self._save)
        root_region.modified.connect(self._save)
        self._track_control_layers(root_region.control)
        for region in root_region:
            self._track_region(region)

    def _save_results(self, job: Job):
        if job.kind is JobKind.diffusion and len(job.results) > 0:
            slot = self._slot_index
            self._slot_index += 1
            image_data, image_offsets = job.results.to_bytes()
            self._model.document.annotate(f"result{slot}.webp", image_data)
            self._history.append(_HistoryResult(job.id or "", slot, image_offsets, job.params))
            self._memory_used[slot] = image_data.size()
            self._prune()
            self._save()

    def _remove_results(self, job: Job):
        index = next((i for i, h in enumerate(self._history) if h.id == job.id), None)
        if index is not None:
            item = self._history.pop(index)
            self._model.document.remove_annotation(f"result{item.slot}.webp")
            self._memory_used.pop(item.slot, None)

    def _remove_image(self, item: JobQueue.Item):
        if history := next((h for h in self._history if h.id == item.job), None):
            if job := self._model.jobs.find(item.job):
                image_data, history.offsets = job.results.to_bytes()
                self._model.document.annotate(f"result{history.slot}.webp", image_data)
                self._memory_used[history.slot] = image_data.size()
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
            style = Styles.list().find(value)
            return style or Styles.list().default
        return value

    return deserialize(obj, data, converter)


def _find_annotation(document, name: str):
    if result := document.find_annotation(name):
        return result
    without_ext = name.rsplit(".", 1)[0]
    if result := document.find_annotation(without_ext):
        return result
    return None
