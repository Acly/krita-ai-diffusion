import asyncio
from datetime import datetime
import sys
import traceback
from enum import Flag
from typing import Sequence, NamedTuple, Optional, Callable
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
    Interrupted,
    NetworkError,
)
from .connection import Connection, ConnectionState
import krita


async def _report_errors(parent, coro):
    try:
        return await coro
    except NetworkError as e:
        parent.report_error(e.message, f"[url={e.url}, code={e.code}]")
    except AssertionError as e:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)
        parent.report_error(f"Error: Internal assertion failed [{str(e)}]")
    except Exception as e:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)
        parent.report_error(f"Error: {str(e)}")


class State(Flag):
    queued = 0
    executing = 1
    finished = 2


class Job:
    id: str
    state = State.queued
    prompt: str
    bounds: Bounds
    timestamp: datetime
    results: ImageCollection

    def __init__(self, id, prompt, bounds):
        self.id = id
        self.prompt = prompt
        self.bounds = bounds
        self.timestamp = datetime.now()
        self.results = ImageCollection()


class JobQueue:
    _entries: Sequence[Job]

    def __init__(self):
        self._entries = []

    def add(self, id: str, prompt: str, bounds: Bounds):
        self._entries.append(Job(id, prompt, bounds))

    def find(self, id: str):
        return next((j for j in self._entries if j.id == id), None)

    def any_executing(self):
        return any(j.state is State.executing for j in self._entries)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, i):
        return self._entries[i]

    def __iter__(self):
        return iter(self._entries)


class Model(QObject):
    """ViewModel for diffusion workflows on a Krita document. Stores all inputs related to
    image generation. Launches generation jobs. Listens to server messages and keeps a
    list of finished, currently running and enqueued jobs.
    """

    _doc: Document
    _layer: Optional[krita.Node] = None
    _image: Optional[Image] = None
    _mask: Optional[Mask] = None
    _extent: Optional[Extent] = None
    _bounds: Optional[Bounds] = None

    changed = pyqtSignal()
    progress_changed = pyqtSignal()

    # state = State.setup
    prompt = ""
    strength = 1.0
    progress = 0.0
    jobs: JobQueue
    error = ""
    task: Optional[asyncio.Task] = None

    def __init__(self, document: Document):
        super().__init__()
        self._doc = document
        self.jobs = JobQueue()

    @staticmethod
    def active():
        """Return the model for the currently active document."""
        return ModelRegistry.instance().model_for_active_document()

    def setup(self):
        """Retrieve the current image and selection mask as inputs for the next generation(s)."""
        self._mask = self._doc.create_mask_from_selection()
        if self._mask is not None or self.strength < 1.0:
            self._image = self._doc.get_image()
            self._bounds = self._mask.bounds if self._mask else Bounds(0, 0, *self._image.extent)
        else:
            self._extent = self._doc.extent
            self._bounds = Bounds(0, 0, *self._extent)

    async def _generate(self):
        # assert State.generating not in self.state
        assert Connection.instance().state is ConnectionState.connected

        client = Connection.instance().client
        image, mask = self._image, self._mask
        prompt, bounds = self.prompt, self._bounds
        # self.state = self.state | State.generating
        if not self.jobs.any_executing():
            self.progress = 0.0
            self.changed.emit()

        if image is None and mask is None:
            assert self._extent is not None and self.strength == 1
            job = workflow.generate(client, self._extent, prompt)
        elif mask is None and self.strength < 1:
            assert image is not None
            job = workflow.refine(client, image, self.prompt, self.strength)
        elif self.strength == 1:
            assert False, "Not implemented"
        #     assert image is not None and mask is not None
        #     generator = workflow.inpaint(client, image, mask, self.prompt, progress)
        else:
            assert image is not None and mask is not None and self.strength < 1
            job = workflow.refine_region(client, image, mask, self.prompt, self.strength)

        prompt_id = await job
        self.jobs.add(prompt_id, prompt, bounds)

    def generate(self):
        self.task = eventloop.run(_report_errors(self, self._generate()))

    def cancel(self):
        Connection.instance().interrupt()

    def report_progress(self, value):
        self.progress = value
        self.progress_changed.emit()

    def report_error(self, message: str, details: Optional[str] = None):
        self.error = message
        self.changed.emit()

    def handle_message(self, message: ClientMessage):
        if message.event is ClientEvent.progress:
            self.report_progress(message.progress)
        elif message.event is ClientEvent.finished:
            job = self.jobs.find(message.prompt_id)
            assert job is not None, 'Received "finished" message for unknown prompt ID.'
            job.state = State.finished
            job.results = message.images
            self.changed.emit()

    def show_preview(self, prompt_id: str, index: int):
        job = self.jobs.find(prompt_id)
        if self._layer is None:
            self._layer = self._doc.insert_layer()
        self._layer.setName(f"[Preview] {self.prompt}")
        self._doc.set_layer_pixels(self._layer, job.results[index], job.bounds)

    def apply_current_result(self):
        """Apply selected result by duplicating the preview layer and inserting it below.
        This allows to apply multiple results (eg. to combine them afterwards by erasing parts).
        """
        new_layer = self._layer
        self._layer = self._layer.duplicate()
        parent = new_layer.parentNode()
        parent.addChildNode(self._layer, new_layer)
        new_layer.setLocked(False)
        new_layer.setName(new_layer.name().replace("[Preview]", "[Generated]"))

    @property
    def can_apply_result(self):
        return self._layer and self._layer.visible()

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

    def report_error(self, message: str, details: Optional[str] = None):
        for m in self._models:
            m.report_error(message, details)

    def _find_model(self, prompt_id: str):
        return next((m for m in self._models if m.jobs.find(prompt_id)), None)

    async def _handle_messages_impl(self):
        assert Connection.instance().state is ConnectionState.connected
        client = Connection.instance().client

        async for msg in client.listen():
            model = self._find_model(msg.prompt_id)
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
