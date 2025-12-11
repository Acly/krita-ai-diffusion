from __future__ import annotations
from collections import deque
from dataclasses import dataclass, fields, field
from datetime import datetime
from enum import Enum, Flag
from typing import Any, NamedTuple, TYPE_CHECKING
from PyQt5.QtCore import QObject, pyqtSignal

from .image import Bounds, ImageCollection
from .settings import settings
from .style import Style
from .util import ensure

if TYPE_CHECKING:
    from . import control


class JobState(Flag):
    queued = 0
    executing = 1
    finished = 2
    cancelled = 3


class JobKind(Enum):
    diffusion = 0
    control_layer = 1
    upscaling = 2
    live_preview = 3
    animation_batch = 4  # single frame as part of an animation batch
    animation_frame = 5  # just a single frame
    animation = 6  # full animation in one job


@dataclass
class JobRegion:
    layer_id: str
    prompt: str
    bounds: Bounds
    is_background: bool = False

    @staticmethod
    def from_dict(data: dict[str, Any]):
        data["bounds"] = Bounds(*data["bounds"])
        return JobRegion(**data)


@dataclass
class JobParams:
    bounds: Bounds
    name: str  # used eg. as name for new layers created from this job
    regions: list[JobRegion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    has_mask: bool = False
    frame: tuple[int, int, int] = (0, 0, 0)
    animation_id: str = ""
    resize_canvas: bool = False

    @staticmethod
    def from_dict(data: dict[str, Any]):
        data["bounds"] = Bounds(*data["bounds"])
        data["regions"] = [JobRegion.from_dict(r) for r in data.get("regions", [])]
        if "metadata" not in data:  # older documents before version 1.26.0
            data["name"] = data.get("prompt", "")
            data["metadata"] = {}
            _move_field(data, "prompt", data["metadata"])
            _move_field(data, "negative_prompt", data["metadata"])
            _move_field(data, "strength", data["metadata"])
            _move_field(data, "style", data["metadata"])
            _move_field(data, "sampler", data["metadata"])
            _move_field(data, "checkpoint", data["metadata"])
        return JobParams(**data)

    @classmethod
    def equal_ignore_seed(cls, a: JobParams | None, b: JobParams | None):
        if a is None or b is None:
            return a is b
        field_names = (f.name for f in fields(cls) if not f.name == "seed")
        return all(getattr(a, name) == getattr(b, name) for name in field_names)

    def set_style(self, style: Style, checkpoint: str):
        self.metadata["style"] = style.filename
        self.metadata["checkpoint"] = checkpoint
        self.metadata["sampler"] = style.sampler
        self.metadata["steps"] = style.sampler_steps
        self.metadata["guidance"] = style.cfg_scale

    def set_control(self, control: "control.ControlLayerList"):
        self.metadata["control"] = [
            {
                "mode": c.mode.text,
                "strength": c.strength / c.strength_multiplier,
                "image": c.layer.name,
                "start": c.start,
                "end": c.end,
            }
            for c in control
        ]

    @property
    def prompt(self):
        return self.metadata.get("prompt", "")

    @property
    def style(self):
        return self.metadata.get("style", "")

    @property
    def strength(self):
        return self.metadata.get("strength", 1.0)


class Job:
    id: str | None
    kind: JobKind
    state = JobState.queued
    params: JobParams
    control: "control.ControlLayer | None" = None
    timestamp: datetime
    results: ImageCollection
    in_use: dict[int, bool]

    def __init__(self, id: str | None, kind: JobKind, params: JobParams):
        self.id = id
        self.kind = kind
        self.params = params
        self.timestamp = datetime.now()
        self.results = ImageCollection()
        self.in_use = {}

    def result_was_used(self, index: int):
        return self.in_use.get(index, False)


class JobQueue(QObject):
    """Queue of waiting, ongoing and finished jobs for one document."""

    class Item(NamedTuple):
        job: str
        image: int

    count_changed = pyqtSignal()
    selection_changed = pyqtSignal()
    job_finished = pyqtSignal(Job)
    job_discarded = pyqtSignal(Job)
    result_used = pyqtSignal(Item)
    result_discarded = pyqtSignal(Item)

    def __init__(self):
        super().__init__()
        self._entries: deque[Job] = deque()
        self._selection: list[JobQueue.Item] = []
        self._previous_selection: JobQueue.Item | None = None
        self._memory_usage = 0  # in MB

    def add(self, kind: JobKind, params: JobParams):
        return self.add_job(Job(None, kind, params))

    def add_control(self, control: "control.ControlLayer", bounds: Bounds):
        job = Job(None, JobKind.control_layer, JobParams(bounds, f"[Control] {control.mode.text}"))
        job.control = control
        return self.add_job(job)

    def add_job(self, job: Job):
        self._entries.append(job)
        self.count_changed.emit()
        return job

    def remove(self, job: Job):
        # Diffusion/Animation jobs: kept for history, pruned according to meomry usage
        # Other jobs: removed immediately once finished
        self._entries.remove(job)
        self.count_changed.emit()

    def find(self, id: str):
        return next((j for j in self._entries if j.id == id), None)

    def count(self, state: JobState):
        return sum(1 for j in self._entries if j.state is state)

    def has_item(self, item: Item):
        job = self.find(item.job)
        return job is not None and item.image < len(job.results)

    def set_results(self, job: Job, results: ImageCollection):
        job.results = results
        if job.kind in [JobKind.diffusion, JobKind.animation]:
            self._memory_usage += results.size / (1024**2)
            self.prune(keep=job)

    def notify_started(self, job: Job):
        if job.state is not JobState.executing:
            job.state = JobState.executing
            self.count_changed.emit()

    def notify_finished(self, job: Job):
        job.state = JobState.finished
        self.job_finished.emit(job)
        self._cancel_earlier_jobs(job)
        self.count_changed.emit()

        if job.kind not in [JobKind.diffusion, JobKind.animation]:
            self.remove(job)

    def notify_cancelled(self, job: Job):
        job.state = JobState.cancelled
        self._cancel_earlier_jobs(job)
        self.count_changed.emit()

    def notify_used(self, job_id: str, index: int):
        job = ensure(self.find(job_id))
        job.in_use[index] = True
        self.result_used.emit(self.Item(job_id, index))

    def select(self, job_id: str, index: int):
        self.selection = [self.Item(job_id, index)]

    def toggle_selection(self):
        if self._selection:
            self._previous_selection = self._selection[0]
            self.selection = []
        elif self._previous_selection is not None and self.has_item(self._previous_selection):
            self.selection = [self._previous_selection]

    def _discard_job(self, job: Job):
        self._entries.remove(job)
        self._memory_usage -= job.results.size / (1024**2)
        self.job_discarded.emit(job)

    def prune(self, keep: Job):
        while self._memory_usage > settings.history_size and self._entries[0] != keep:
            self._discard_job(self._entries[0])

    def discard(self, job_id: str, index: int):
        job = ensure(self.find(job_id))
        if len(job.results) <= 1 or job.kind is JobKind.animation:
            self._discard_job(job)
            return
        for i in range(index, len(job.results) - 1):
            job.in_use[i] = job.in_use.get(i + 1, False)
        img = job.results.remove(index)
        self._memory_usage -= img.size / (1024**2)
        self.result_discarded.emit(self.Item(job_id, index))

    def clear(self):
        jobs_to_discard = [
            job
            for job in self._entries
            if job.kind in (JobKind.diffusion, JobKind.animation) and job.state is JobState.finished
        ]
        for job in jobs_to_discard:
            self._discard_job(job)

    def any_executing(self):
        return any(j.state is JobState.executing for j in self._entries)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, i):
        return self._entries[i]

    def __iter__(self):
        return iter(self._entries)

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, value: list[Item]):
        if self._selection != value:
            self._selection = value
            self.selection_changed.emit()

    @property
    def memory_usage(self):
        return self._memory_usage

    def _cancel_earlier_jobs(self, job: Job):
        # Clear jobs that should have been completed before, but may not have completed
        # (still queued or executing state) due to sporadic server disconnect
        for j in self._entries:
            if j is job:
                break
            if j.state in [JobState.queued, JobState.executing]:
                j.state = JobState.cancelled


def _move_field(src: dict[str, Any], field: str, dest: dict[str, Any]):
    if field in src:
        dest[field] = src[field]
        del src[field]
