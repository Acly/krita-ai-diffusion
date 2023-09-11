import asyncio
from collections import deque
from datetime import datetime
from enum import Flag
from typing import Deque, Sequence, NamedTuple, Optional, Callable
from PyQt5.QtCore import QObject, pyqtSignal
from .. import (
    eventloop,
    ClientMessage,
    ClientEvent,
    Document,
    Image,
    Mask,
    Extent,
    Bounds,
    ImageCollection,
    workflow,
    NetworkError,
    Style,
    Styles,
    settings,
    util,
)
from .connection import Connection, ConnectionState
import krita


async def _report_errors(parent, coro):
    try:
        return await coro
    except NetworkError as e:
        parent.report_error(f"{util.log_error(e)} [url={e.url}, code={e.code}]")
    except Exception as e:
        parent.report_error(util.log_error(e))


class State(Flag):
    queued = 0
    executing = 1
    finished = 2
    cancelled = 3


class Job:
    id: str
    state = State.queued
    prompt: str
    bounds: Bounds
    timestamp: datetime
    _results: ImageCollection

    def __init__(self, id, prompt, bounds):
        self.id = id
        self.prompt = prompt
        self.bounds = bounds
        self.timestamp = datetime.now()
        self._results = ImageCollection()

    @property
    def results(self):
        return self._results


class JobQueue:
    _entries: Deque[Job]
    _memory_usage = 0  # in MB

    def __init__(self):
        self._entries = deque()

    def add(self, id: str, prompt: str, bounds: Bounds):
        self._entries.append(Job(id, prompt, bounds))

    def find(self, id: str):
        return next((j for j in self._entries if j.id == id), None)

    def count(self, state: State):
        return sum(1 for j in self._entries if j.state is state)

    def set_results(self, job: Job, results: ImageCollection):
        job._results = results
        self._memory_usage += results.size / (1024**2)
        while self._memory_usage > settings.history_size and self._entries[0] != job:
            discarded = self._entries.popleft()
            self._memory_usage -= discarded._results.size / (1024**2)

    def any_executing(self):
        return any(j.state is State.executing for j in self._entries)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, i):
        return self._entries[i]

    def __iter__(self):
        return iter(self._entries)

    @property
    def memory_usage(self):
        return self._memory_usage


class Model(QObject):
    """View-model for diffusion workflows on a Krita document. Stores all inputs related to
    image generation. Launches generation jobs. Listens to server messages and keeps a
    list of finished, currently running and enqueued jobs.
    """

    _doc: Document
    _layer: Optional[krita.Node] = None

    changed = pyqtSignal()
    job_finished = pyqtSignal(Job)
    progress_changed = pyqtSignal()

    style: Style
    prompt = ""
    strength = 1.0
    progress = 0.0
    jobs: JobQueue
    error = ""
    task: Optional[asyncio.Task] = None

    def __init__(self, document: Document):
        super().__init__()
        self._doc = document
        self.style = Styles.list().default
        self.jobs = JobQueue()

    @staticmethod
    def active():
        """Return the model for the currently active document."""
        return ModelRegistry.instance().model_for_active_document()

    def generate(self):
        """Enqueue image generation for the current setup."""
        image = None
        # For 100% strength inpainting, pad the mask if it is small to get additional context
        # For img2img inpainting, use a smaller mask area and scale it to get more detail
        pad_min_size = 512 if self.strength == 1 else 0
        mask = self._doc.create_mask_from_selection(pad_min_size)

        if mask is not None or self.strength < 1.0:
            image = self._doc.get_image(exclude_layer=self._layer)
            bounds = mask.bounds if mask else Bounds(0, 0, *image.extent)
        else:
            bounds = Bounds(0, 0, *self._doc.extent)

        self.clear_error()
        self.task = eventloop.run(_report_errors(self, self._generate(bounds, image, mask)))

    async def _generate(self, bounds: Bounds, image: Optional[Image], mask: Optional[Mask]):
        assert Connection.instance().state is ConnectionState.connected

        client = Connection.instance().client
        style, prompt = self.style, self.prompt
        if not self.jobs.any_executing():
            self.progress = 0.0
            self.changed.emit()

        if image is None and mask is None:
            assert self.strength == 1
            job = workflow.generate(client, style, bounds.extent, prompt)
        elif mask is None and self.strength < 1:
            assert image is not None
            job = workflow.refine(client, style, image, prompt, self.strength)
        elif self.strength == 1:
            assert image is not None and mask is not None
            job = workflow.inpaint(client, style, image, mask, self.prompt)
        else:
            assert image is not None and mask is not None and self.strength < 1
            job = workflow.refine_region(client, style, image, mask, prompt, self.strength)

        job_id = await client.enqueue(job)
        self.jobs.add(job_id, prompt, bounds)
        self.changed.emit()

    def cancel(self):
        Connection.instance().interrupt()

    def report_progress(self, value):
        self.progress = value
        self.progress_changed.emit()

    def report_error(self, message: str):
        self.error = message
        self.changed.emit()

    def clear_error(self):
        if self.error != "":
            self.error = ""
            self.changed.emit()

    def handle_message(self, message: ClientMessage):
        job = self.jobs.find(message.job_id)
        assert job is not None, "Received message for unknown job."

        if message.event is ClientEvent.progress:
            job.state = State.executing
            self.report_progress(message.progress)
        elif message.event is ClientEvent.finished:
            job.state = State.finished
            self.jobs.set_results(job, message.images)
            self.progress = 1
            if self._layer is None:
                self.show_preview(message.job_id, 0)
            self.job_finished.emit(job)
            self.changed.emit()
        elif message.event is ClientEvent.interrupted:
            job.state = State.cancelled
            self.report_progress(0)
        elif message.event is ClientEvent.error:
            job.state = State.cancelled
            self.report_error(f"Server execution error: {message.error}")

    def show_preview(self, job_id: str, index: int):
        job = self.jobs.find(job_id)
        name = f"[Preview] {job.prompt}"
        if self._layer and self._layer.parentNode() is None:
            self._layer = None
        if self._layer is not None:
            self._layer.setName(name)
            self._doc.set_layer_content(self._layer, job.results[index], job.bounds)
        else:
            self._layer = self._doc.insert_layer(name, job.results[index], job.bounds)
        self.changed.emit()

    def hide_preview(self):
        if self._layer is not None:
            self._doc.hide_layer(self._layer)
            self.changed.emit()

    def apply_current_result(self):
        """Promote the preview layer to a user layer."""
        assert self.can_apply_result
        self._layer.setLocked(False)
        self._layer.setName(self._layer.name().replace("[Preview]", "[Generated]"))
        self._layer = None
        self.changed.emit()

    @property
    def history(self):
        return (job for job in self.jobs if job.state is State.finished)

    @property
    def can_apply_result(self):
        return self._layer is not None and self._layer.visible()

    @property
    def is_active(self):
        return self._doc.is_active

    @property
    def is_valid(self):
        return self._doc.is_valid


class ModelRegistry(QObject):
    """Singleton that keeps track of all models (one per open image document) and notifies
    widgets when new ones are created."""

    _instance = None
    _models = []
    _task: Optional[asyncio.Task] = None

    created = pyqtSignal(Model)

    def __init__(self):
        super().__init__()
        connection = Connection.instance()

        def handle_messages():
            if self._task is None and connection.state is ConnectionState.connected:
                self._task = eventloop._loop.create_task(self._handle_messages())
            elif self._task and connection.state is ConnectionState.disconnected:
                self._task.cancel()
                self._task = None

        connection.changed.connect(handle_messages)

    def __del__(self):
        if self._task is not None:
            self._task.cancel()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    def model_for_active_document(self):
        # Remove models for documents that have been closed
        self._models = [m for m in self._models if m.is_valid]

        # Find or create model for active document
        if Document.active() is not None:
            model = next((m for m in self._models if m.is_active), None)
            if model is None:
                model = Model(Document.active())
                self._models.append(model)
                self.created.emit(model)
            return model

        return None

    def report_error(self, message: str):
        for m in self._models:
            m.report_error(message)

    def _find_model(self, job_id: str):
        return next((m for m in self._models if m.jobs.find(job_id)), None)

    async def _handle_messages_impl(self):
        assert Connection.instance().state is ConnectionState.connected
        client = Connection.instance().client

        async for msg in client.listen():
            model = self._find_model(msg.job_id)
            if model is not None:
                model.handle_message(msg)

    async def _handle_messages(self):
        try:
            # TODO: maybe use async for websockets.connect which is meant for this
            while True:
                # Run inner loop
                await _report_errors(self, self._handle_messages_impl())
                # After error or unexpected disconnect, wait a bit before reconnecting
                await asyncio.sleep(5)

        except asyncio.CancelledError:
            pass  # shutdown
