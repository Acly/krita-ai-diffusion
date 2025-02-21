from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any
from PyQt5.QtCore import QObject, QByteArray
from PyQt5.QtGui import QImageReader
from PyQt5.QtWidgets import QMessageBox

from .api import InpaintMode, FillMode
from .image import ImageCollection
from .model import Model, InpaintContext
from .custom_workflow import CustomWorkspace
from .control import ControlLayer, ControlLayerList
from .region import RootRegion, Region
from .jobs import Job, JobKind, JobParams, JobQueue
from .style import Style, Styles
from .properties import serialize, deserialize
from .settings import settings
from .localization import translate as _
from .util import client_logger as log, encode_json

# Version of the persistence format, increment when there are breaking changes
version = 1


@dataclass
class RecentlyUsedSync:
    """Stores the most recently used parameters for various settings across all models.
    This is used to initialize new models with the last used parameters if they are
    created from scratch (not opening an existing .kra with stored settings).
    """

    style: str = ""
    batch_count: int = 1
    translation_enabled: bool = True
    inpaint_mode: str = "automatic"
    inpaint_fill: str = "neutral"
    inpaint_use_model: bool = True
    inpaint_use_prompt_focus: bool = False
    inpaint_context: str = "automatic"
    upscale_model: str = ""

    @staticmethod
    def from_settings():
        try:
            return RecentlyUsedSync(**settings.document_defaults)
        except Exception as e:
            log.warning(f"Failed to load default document settings: {type(e)} {e}")
            return RecentlyUsedSync()

    def track(self, model: Model):
        try:
            if _find_annotation(model.document, "ui.json") is None:
                model.style = Styles.list().find(self.style) or Styles.list().default
                model.batch_count = self.batch_count
                model.translation_enabled = self.translation_enabled
                model.inpaint.mode = InpaintMode[self.inpaint_mode]
                model.inpaint.fill = FillMode[self.inpaint_fill]
                model.inpaint.use_inpaint = self.inpaint_use_model
                model.inpaint.use_prompt_focus = self.inpaint_use_prompt_focus
                model.upscale.upscaler = self.upscale_model
                if self.inpaint_context != InpaintContext.layer_bounds.name:
                    model.inpaint.context = InpaintContext[self.inpaint_context]
        except Exception as e:
            log.warning(f"Failed to apply default settings to new document: {type(e)} {e}")

        model.style_changed.connect(self._set("style"))
        model.batch_count_changed.connect(self._set("batch_count"))
        model.translation_enabled_changed.connect(self._set("translation_enabled"))
        model.inpaint.mode_changed.connect(self._set("inpaint_mode"))
        model.inpaint.fill_changed.connect(self._set("inpaint_fill"))
        model.inpaint.use_inpaint_changed.connect(self._set("inpaint_use_model"))
        model.inpaint.use_prompt_focus_changed.connect(self._set("inpaint_use_prompt_focus"))
        model.inpaint.context_changed.connect(self._set("inpaint_context"))
        model.upscale.upscaler_changed.connect(self._set("upscale_model"))

    def _set(self, key):
        def setter(value):
            if isinstance(value, Style):
                value = value.filename
            if isinstance(value, Enum):
                value = value.name
            setattr(self, key, value)
            self._save()

        return setter

    def _save(self):
        settings.document_defaults = asdict(self)
        settings.save()


@dataclass
class _HistoryResult:
    id: str
    slot: int  # annotation slot where images are stored
    offsets: list[int]  # offsets in bytes for result images
    params: JobParams
    kind: JobKind = JobKind.diffusion
    in_use: dict[int, bool] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict[str, Any]):
        data["params"] = JobParams.from_dict(data["params"])
        data["kind"] = JobKind[data.get("kind", "diffusion")]
        data["in_use"] = {int(k): v for k, v in data.get("in_use", {}).items()}
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
                msg = _("Failed to load state from") + f" {model.document.filename}: {e}"
                log.exception(msg)
                QMessageBox.warning(None, "AI Diffusion Plugin", msg)
        self._track(model)

    def _save(self):
        model = self._model
        state = _serialize(model)
        state["version"] = version
        state["preview_layer"] = model.preview_layer_id
        state["inpaint"] = _serialize(model.inpaint)
        state["upscale"] = _serialize(model.upscale)
        state["live"] = _serialize(model.live)
        state["animation"] = _serialize(model.animation)
        state["custom"] = _serialize_custom(model.custom)
        state["history"] = [asdict(h) for h in self._history]
        state["root"] = _serialize(model.regions)
        state["control"] = [_serialize(c) for c in model.regions.control]
        state["regions"] = []
        for region in model.regions:
            state["regions"].append(_serialize(region))
            state["regions"][-1]["control"] = [_serialize(c) for c in region.control]
        state_str = json.dumps(state, indent=2, default=encode_json)
        state_bytes = QByteArray(state_str.encode("utf-8"))
        model.document.annotate("ui.json", state_bytes)

    def _load(self, model: Model, state_bytes: bytes):
        state = json.loads(state_bytes.decode("utf-8"))
        model.try_set_preview_layer(state.get("preview_layer", ""))
        _deserialize(model, state)
        _deserialize(model.inpaint, state.get("inpaint", {}))
        _deserialize(model.upscale, state.get("upscale", {}))
        _deserialize(model.live, state.get("live", {}))
        _deserialize(model.animation, state.get("animation", {}))
        _deserialize_custom(model.custom, state.get("custom", {}), model.name)
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
                job = model.jobs.add_job(Job(item.id, item.kind, item.params))
                job.in_use = item.in_use
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
        model.custom.modified.connect(self._save)
        model.jobs.job_finished.connect(self._save_results)
        model.jobs.job_discarded.connect(self._remove_results)
        model.jobs.result_discarded.connect(self._remove_image)
        model.jobs.result_used.connect(self._save)
        model.jobs.selection_changed.connect(self._save)
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
        if job.kind in [JobKind.diffusion, JobKind.animation] and len(job.results) > 0:
            slot = self._slot_index
            self._slot_index += 1
            image_data, image_offsets = job.results.to_bytes()
            self._model.document.annotate(f"result{slot}.webp", image_data)
            self._history.append(
                _HistoryResult(job.id or "", slot, image_offsets, job.params, job.kind, job.in_use)
            )
            self._memory_used[slot] = image_data.size()
            self._prune()
            self._save()

    def _remove_results(self, job: Job):
        index = next((i for i, h in enumerate(self._history) if h.id == job.id), None)
        if index is not None:
            item = self._history.pop(index)
            self._model.document.remove_annotation(f"result{item.slot}.webp")
            self._memory_used.pop(item.slot, None)
        self._save()

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

    if "unblur_strength" in data and not isinstance(data["unblur_strength"], float):
        data["unblur_strength"] = 0.5

    return deserialize(obj, data, converter)


def _serialize_custom(custom: CustomWorkspace):
    result = _serialize(custom)
    result["workflow_id"] = custom.workflow_id
    result["graph"] = custom.graph.root if custom.graph else None
    return result


def _deserialize_custom(custom: CustomWorkspace, data: dict[str, Any], document_name: str):
    _deserialize(custom, data)
    workflow_id = data.get("workflow_id", "")
    graph = data.get("graph", None)
    if workflow_id and graph:
        custom.set_graph(workflow_id, graph, document_name)


def _find_annotation(document, name: str):
    if result := document.find_annotation(name):
        return result
    without_ext = name.rsplit(".", 1)[0]
    if result := document.find_annotation(without_ext):
        return result
    return None


def import_prompt_from_file(model: Model):
    exts = (".png", ".jpg", ".jpeg", ".webp")
    filename = model.document.filename
    if model.regions.positive == "" and model.regions.negative == "" and filename.endswith(exts):
        try:
            reader = QImageReader(filename)
            # A1111
            if text := reader.text("parameters"):
                if "Negative prompt:" in text:
                    positive, negative = text.split("Negative prompt:", 1)
                    model.regions.positive = positive.strip()
                    model.regions.negative = negative.split("Steps:", 1)[0].strip()
            # ComfyUI
            elif text := reader.text("prompt"):
                prompt: dict[str, dict] = json.loads(text)
                for node in prompt.values():
                    if node["class_type"] in _comfy_sampler_types:
                        inputs = node["inputs"]
                        model.regions.positive = _find_text_prompt(prompt, inputs["positive"][0])
                        model.regions.negative = _find_text_prompt(prompt, inputs["negative"][0])

        except Exception as e:
            log.warning(f"Failed to read PNG metadata from {filename}: {e}")


_comfy_sampler_types = ["KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"]


def _find_text_prompt(workflow: dict[str, dict], node_key: str):
    if node := workflow.get(node_key):
        if node["class_type"] == "CLIPTextEncode":
            text = node.get("inputs", {}).get("text", "")
            return text if isinstance(text, str) else ""
        for input in node.get("inputs", {}).values():
            if isinstance(input, list):
                return _find_text_prompt(workflow, input[0])
    return ""
