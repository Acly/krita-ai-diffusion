from __future__ import annotations
from collections import deque
from datetime import datetime
from enum import Enum, Flag
from typing import Deque, NamedTuple
from PyQt5.QtCore import QObject, pyqtSignal

from .image import Bounds, ImageCollection
from .settings import settings
from .util import ensure
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


class Job:
    id: str | None
    kind: JobKind
    state = JobState.queued
    prompt: str
    bounds: Bounds
    control: "control.ControlLayer | None" = None
    timestamp: datetime
    results: ImageCollection
    _in_use: dict[int, bool]

    def __init__(self, id: str | None, kind: JobKind, prompt: str, bounds: Bounds):
        self.id = id
        self.kind = kind
        self.prompt = prompt
        self.bounds = bounds
        self.timestamp = datetime.now()
        self.results = ImageCollection()
        self._in_use = {}

    def result_was_used(self, index: int):
        return self._in_use.get(index, False)


class JobQueue(QObject):
    """Queue of waiting, ongoing and finished jobs for one document."""

    class Item(NamedTuple):
        job: str
        image: int

    count_changed = pyqtSignal()
    selection_changed = pyqtSignal()
    job_finished = pyqtSignal(Job)
    result_used = pyqtSignal(Item)

    _entries: Deque[Job]
    _selection: Item | None = None
    _memory_usage = 0  # in MB

    def __init__(self):
        super().__init__()
        self._entries = deque()

    def add(self, kind: JobKind, id: str, prompt: str, bounds: Bounds):
        self._add(Job(id, kind, prompt, bounds))

    def add_control(self, control: "control.ControlLayer", bounds: Bounds):
        job = Job(None, JobKind.control_layer, f"[Control] {control.mode.text}", bounds)
        job.control = control
        return self._add(job)

    def add_upscale(self, bounds: Bounds):
        job = Job(None, JobKind.upscaling, f"[Upscale] {bounds.width}x{bounds.height}", bounds)
        return self._add(job)

    def _add(self, job: Job):
        self._entries.append(job)
        self.count_changed.emit()
        return job

    def remove(self, job: Job):
        # Diffusion jobs: kept for history, pruned according to meomry usage
        # Control layer jobs: removed immediately once finished
        self._entries.remove(job)
        self.count_changed.emit()

    def find(self, id: str):
        return next((j for j in self._entries if j.id == id), None)

    def count(self, state: JobState):
        return sum(1 for j in self._entries if j.state is state)

    def set_results(self, job: Job, results: ImageCollection):
        job.results = results
        if job.kind is JobKind.diffusion:
            self._memory_usage += results.size / (1024**2)
            self.prune(keep=job)

    def notify_started(self, job: Job):
        job.state = JobState.executing
        self.count_changed.emit()

    def notify_finished(self, job: Job):
        job.state = JobState.finished
        self.job_finished.emit(job)
        self.count_changed.emit()

    def notify_used(self, job_id: str, index: int):
        job = ensure(self.find(job_id))
        job._in_use[index] = True
        self.result_used.emit(self.Item(job_id, index))

    def prune(self, keep: Job):
        while self._memory_usage > settings.history_size and self._entries[0] != keep:
            discarded = self._entries.popleft()
            self._memory_usage -= discarded.results.size / (1024**2)

    def select(self, job_id: str, index: int):
        self.selection = self.Item(job_id, index)

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
    def selection(self, value: Item | None):
        self._selection = value
        self.selection_changed.emit()

    @property
    def memory_usage(self):
        return self._memory_usage
