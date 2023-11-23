from __future__ import annotations
import asyncio
from collections import deque
from copy import copy
from datetime import datetime
from enum import Enum, Flag
from typing import Deque, Optional, cast
from PyQt5.QtCore import Qt, QObject, pyqtSignal

from .. import eventloop, Document, workflow, NetworkError, client, settings, util
from ..image import Image, ImageCollection, Mask, Bounds
from ..client import ClientMessage, ClientEvent
from ..pose import Pose
from ..style import Style, Styles
from ..workflow import Control, ControlMode, Conditioning, LiveParams
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


class JobKind(Enum):
    diffusion = 0
    control_layer = 1
    upscaling = 2
    live_preview = 3


class Job:
    id: Optional[str]
    kind: JobKind
    state = State.queued
    prompt: str
    bounds: Bounds
    control: Optional[Control] = None
    timestamp: datetime
    _results: ImageCollection

    def __init__(self, id: Optional[str], kind: JobKind, prompt: str, bounds: Bounds):
        self.id = id
        self.kind = kind
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
        self._entries.append(Job(id, JobKind.diffusion, prompt, bounds))

    def add_control(self, control: Control, bounds: Bounds):
        job = Job(None, JobKind.control_layer, f"[Control] {control.mode.text}", bounds)
        job.control = control
        self._entries.append(job)
        return job

    def add_upscale(self, bounds: Bounds):
        job = Job(None, JobKind.upscaling, f"[Upscale] {bounds.width}x{bounds.height}", bounds)
        self._entries.append(job)
        return job

    def add_live(self, prompt: str, bounds: Bounds):
        job = Job(None, JobKind.live_preview, prompt, bounds)
        self._entries.append(job)
        return job

    def remove(self, job: Job):
        # Diffusion jobs: kept for history, pruned according to meomry usage
        # Control layer jobs: removed immediately once finished
        self._entries.remove(job)

    def find(self, id: str | Control):
        if isinstance(id, str):
            return next((j for j in self._entries if j.id == id), None)
        elif isinstance(id, Control):
            return next((j for j in self._entries if j.control is id), None)
        assert False, "Invalid job id"

    def count(self, state: State):
        return sum(1 for j in self._entries if j.state is state)

    def set_results(self, job: Job, results: ImageCollection):
        job._results = results
        if job.kind is JobKind.diffusion:
            self._memory_usage += results.size / (1024**2)
            self.prune(keep=job)

    def prune(self, keep: Job):
        while self._memory_usage > settings.history_size and self._entries[0] != keep:
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


class Workspace(Enum):
    generation = 0
    upscaling = 1
    live = 2


class Model(QObject):
    """View-model for diffusion workflows on a Krita document. Stores all inputs related to
    image generation. Launches generation jobs. Listens to server messages and keeps a
    list of finished, currently running and enqueued jobs.
    """

    _doc: Document
    _layer: Optional[krita.Node] = None
    _live_result: Optional[Image] = None

    changed = pyqtSignal()
    job_finished = pyqtSignal(Job)
    progress_changed = pyqtSignal()

    workspace = Workspace.generation
    style: Style
    prompt = ""
    negative_prompt = ""
    control: list[Control]
    strength = 1.0
    upscale: UpscaleParams
    live: LiveParams
    progress = 0.0
    jobs: JobQueue
    error = ""
    task: Optional[asyncio.Task] = None

    def __init__(self, document: Document):
        super().__init__()
        self._doc = document
        styles = client.filter_supported_styles(
            Styles.list(), Connection.instance().client_if_connected
        )
        self.style = styles[0] if len(styles) > 0 else Styles.list().default
        self.control = []
        self.upscale = UpscaleParams(self)
        self.live = LiveParams()
        self.jobs = JobQueue()

    @staticmethod
    def active():
        """Return the model for the currently active document."""
        return ModelRegistry.instance().model_for_active_document()

    def generate(self):
        """Enqueue image generation for the current setup."""
        ok, msg = self._doc.check_color_mode()
        if not ok and msg:
            self.report_error(msg)
            return

        image = None
        extent = self._doc.extent

        mask, selection_bounds = self._doc.create_mask_from_selection(
            grow=settings.selection_grow / 100,
            feather=settings.selection_feather / 100,
            padding=settings.selection_padding / 100,
        )
        image_bounds = workflow.compute_bounds(extent, mask.bounds if mask else None, self.strength)
        if mask is not None or self.strength < 1.0:
            image = self._get_current_image(image_bounds)
        if selection_bounds is not None:
            selection_bounds = Bounds.apply_crop(selection_bounds, image_bounds)
            selection_bounds = Bounds.minimum_size(selection_bounds, 64, image_bounds.extent)

        control = [self._get_control_image(c, image_bounds) for c in self.control]
        conditioning = Conditioning(self.prompt, self.negative_prompt, control)
        conditioning.area = selection_bounds if self.strength == 1.0 else None

        self.clear_error()
        self.task = eventloop.run(
            _report_errors(self, self._generate(image_bounds, conditioning, image, mask))
        )

    async def _generate(
        self,
        bounds: Bounds,
        conditioning: Conditioning,
        image: Optional[Image],
        mask: Optional[Mask],
    ):
        client = Connection.instance().client
        style, strength = self.style, self.strength
        if not self.jobs.any_executing():
            self.progress = 0.0
            self.changed.emit()

        if mask is not None:
            mask_bounds_rel = Bounds(  # mask bounds relative to cropped image
                mask.bounds.x - bounds.x, mask.bounds.y - bounds.y, *mask.bounds.extent
            )
            bounds = mask.bounds  # absolute mask bounds, required to insert result image
            mask.bounds = mask_bounds_rel

        if image is None and mask is None:
            assert strength == 1
            job = workflow.generate(client, style, bounds.extent, conditioning)
        elif mask is None and strength < 1:
            assert image is not None
            job = workflow.refine(client, style, image, conditioning, strength)
        elif strength == 1:
            assert image is not None and mask is not None
            job = workflow.inpaint(client, style, image, mask, conditioning)
        else:
            assert image is not None and mask is not None and strength < 1
            job = workflow.refine_region(client, style, image, mask, conditioning, strength)

        job_id = await client.enqueue(job)
        self.jobs.add(job_id, conditioning.prompt, bounds)
        self.changed.emit()

    def upscale_image(self):
        image = self._doc.get_image(Bounds(0, 0, *self._doc.extent))
        job = self.jobs.add_upscale(Bounds(0, 0, *self.upscale.target_extent))
        self.clear_error()
        self.task = eventloop.run(
            _report_errors(self, self._upscale_image(job, image, copy(self.upscale)))
        )

    async def _upscale_image(self, job: Job, image: Image, params: UpscaleParams):
        client = Connection.instance().client
        if params.upscaler == "":
            params.upscaler = client.default_upscaler
        if params.use_diffusion:
            work = workflow.upscale_tiled(
                client, image, params.upscaler, params.factor, self.style, params.strength
            )
        else:
            work = workflow.upscale_simple(client, image, params.upscaler, params.factor)
        job.id = await client.enqueue(work)
        self._doc.resize(params.target_extent)
        self.changed.emit()

    def generate_live(self):
        bounds = Bounds(0, 0, *self._doc.extent)
        image = None
        if self.live.strength < 1:
            image = self._get_current_image(bounds)
        control = [self._get_control_image(c, bounds) for c in self.control]
        cond = Conditioning(self.prompt, self.negative_prompt, control)
        job = self.jobs.add_live(self.prompt, bounds)
        self.clear_error()
        self.task = eventloop.run(
            _report_errors(self, self._generate_live(job, image, self.style, cond))
        )

    async def _generate_live(self, job: Job, image: Image | None, style: Style, cond: Conditioning):
        client = Connection.instance().client
        if image:
            work = workflow.refine(client, style, image, cond, self.live.strength, self.live)
        else:
            work = workflow.generate(client, style, self._doc.extent, cond, self.live)
        job.id = await client.enqueue(work)
        self.changed.emit()

    def _get_current_image(self, bounds: Bounds):
        exclude = [  # exclude control layers from projection
            cast(krita.Node, c.image)
            for c in self.control
            if c.mode not in [ControlMode.image, ControlMode.blur]
        ]
        if self._layer:  # exclude preview layer
            exclude.append(self._layer)
        return self._doc.get_image(bounds, exclude_layers=exclude)

    def _get_control_image(self, control: Control, bounds: Optional[Bounds]):
        layer = cast(krita.Node, control.image)
        if control.mode is ControlMode.image and not layer.bounds().isEmpty():
            bounds = None  # ignore mask bounds, use layer bounds
        image = self._doc.get_layer_image(layer, bounds)
        if control.mode.is_lines:
            image.make_opaque(background=Qt.GlobalColor.white)
        return Control(control.mode, image, strength=control.strength, end=control.end)

    def generate_control_layer(self, control: Control):
        ok, msg = self._doc.check_color_mode()
        if not ok and msg:
            self.report_error(msg)
            return

        image = self._doc.get_image(Bounds(0, 0, *self._doc.extent))
        job = self.jobs.add_control(control, Bounds(0, 0, *image.extent))
        self.clear_error()
        self.task = eventloop.run(
            _report_errors(self, self._generate_control_layer(job, image, control.mode))
        )

    async def _generate_control_layer(self, job: Job, image: Image, mode: ControlMode):
        client = Connection.instance().client
        work = workflow.create_control_image(image, mode)
        job.id = await client.enqueue(work)
        self.changed.emit()

    def remove_control_layer(self, control: Control):
        self.control.remove(control)
        self.changed.emit()

    def cancel(self, active=False, queued=False):
        if queued:
            to_remove = [job for job in self.jobs if job.state is State.queued]
            if len(to_remove) > 0:
                Connection.instance().clear_queue()
                for job in to_remove:
                    self.jobs.remove(job)
        if active and self.jobs.any_executing():
            Connection.instance().interrupt()

    def report_progress(self, value):
        self.progress = value
        self.progress_changed.emit()

    def report_error(self, message: str):
        self.error = message
        self.live.is_active = False
        self.changed.emit()

    def clear_error(self):
        if self.error != "":
            self.error = ""
            self.changed.emit()

    def handle_message(self, message: ClientMessage):
        job = self.jobs.find(message.job_id)
        if job is None:
            util.client_logger.error(f"Received message {message} for unknown job.")
            return

        if message.event is ClientEvent.progress:
            job.state = State.executing
            self.report_progress(message.progress)
        elif message.event is ClientEvent.finished:
            job.state = State.finished
            self.progress = 1
            if message.images:
                self.jobs.set_results(job, message.images)
            if job.kind is JobKind.diffusion and self._layer is None and job.id:
                self.show_preview(job.id, 0)
            elif job.kind is JobKind.control_layer:
                job.control.image = self.add_control_layer(job, message.result)  # type: ignore
            elif job.kind is JobKind.upscaling:
                self.add_upscale_layer(job)
            elif job.kind is JobKind.live_preview and len(job.results) > 0:
                self._live_result = job.results[0]
            if job.kind is not JobKind.diffusion:
                self.jobs.remove(job)
            self.job_finished.emit(job)
            self.changed.emit()
        elif message.event is ClientEvent.interrupted:
            job.state = State.cancelled
            self.report_progress(0)
        elif message.event is ClientEvent.error:
            job.state = State.cancelled
            self.report_error(f"Server execution error: {message.error}")

    def show_preview(self, job_id: str, index: int, name_prefix="Preview"):
        job = self.jobs.find(job_id)
        assert job is not None, "Cannot show preview, invalid job id"
        name = f"[{name_prefix}] {job.prompt}"
        if self._layer and self._layer.parentNode() is None:
            self._layer = None
        if self._layer is not None:
            self._layer.setName(name)
            self._doc.set_layer_content(self._layer, job.results[index], job.bounds)
        else:
            self._layer = self._doc.insert_layer(name, job.results[index], job.bounds)
            self._layer.setLocked(True)
        self.changed.emit()

    def hide_preview(self):
        if self._layer is not None:
            self._doc.hide_layer(self._layer)
            self.changed.emit()

    def apply_current_result(self):
        """Promote the preview layer to a user layer."""
        assert self._layer and self.can_apply_result
        self._layer.setLocked(False)
        self._layer.setName(self._layer.name().replace("[Preview]", "[Generated]"))
        self._layer = None
        self.changed.emit()

    def add_control_layer(self, job: Job, result: Optional[dict]):
        assert job.kind is JobKind.control_layer and job.control
        if job.control.mode is ControlMode.pose and result is not None:
            pose = Pose.from_open_pose_json(result)
            pose.scale(job.bounds.extent)
            return self._doc.insert_vector_layer(job.prompt, pose.to_svg(), below=self._layer)
        elif len(job.results) > 0:
            return self._doc.insert_layer(job.prompt, job.results[0], job.bounds, below=self._layer)
        return self.document.active_layer  # Execution was cached and no image was produced

    def add_upscale_layer(self, job: Job):
        assert job.kind is JobKind.upscaling
        assert len(job.results) > 0, "Upscaling job did not produce an image"
        if self._layer:
            self._layer.remove()
            self._layer = None
        self._doc.insert_layer(job.prompt, job.results[0], job.bounds)

    def add_live_layer(self):
        assert self._live_result is not None
        self._doc.insert_layer(
            f"[Live] {self.prompt}", self._live_result, Bounds(0, 0, *self._doc.extent)
        )

    @property
    def history(self):
        return (job for job in self.jobs if job.state is State.finished)

    @property
    def can_apply_result(self):
        return self._layer is not None and self._layer.visible()

    @property
    def has_live_result(self):
        return self._live_result is not None

    @property
    def document(self):
        return self._doc

    @property
    def is_active(self):
        return self._doc.is_active

    @property
    def is_valid(self):
        return self._doc.is_valid


class UpscaleParams:
    upscaler: str
    factor = 2.0
    use_diffusion = True
    strength = 0.3

    _model: Model

    def __init__(self, model: Model):
        self._model = model
        if client := Connection.instance().client_if_connected:
            self.upscaler = client.default_upscaler
        else:
            self.upscaler = ""

    @property
    def target_extent(self):
        return self._model.document.extent * self.factor


class ModelRegistry(QObject):
    """Singleton that keeps track of all models (one per open image document) and notifies
    widgets when new ones are created."""

    _instance = None
    _models: list[Model] = []
    _task: Optional[asyncio.Task] = None

    created = pyqtSignal(Model)

    def __init__(self):
        super().__init__()
        connection = Connection.instance()

        def handle_messages():
            if self._task is None and connection.state is ConnectionState.connected:
                self._task = eventloop._loop.create_task(self._handle_messages())
            elif self._task and connection.state is ConnectionState.disconnected:
                assert self._task.done()
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
        if doc := Document.active():
            model = next((m for m in self._models if m.is_active), None)
            if model is None:
                model = Model(doc)
                self._models.append(model)
                self.created.emit(model)
            return model

        return None

    async def stop_listening(self):
        if self._task is not None:
            self._task.cancel()
            await self._task

    def report_error(self, message: str):
        for m in self._models:
            m.report_error(message)

    def clear_error(self):
        for m in self._models:
            m.clear_error()

    def _find_model(self, job_id: str) -> Optional[Model]:
        return next((m for m in self._models if m.jobs.find(job_id)), None)

    async def _handle_messages(self):
        client = Connection.instance().client
        temporary_disconnect = False

        try:
            async for msg in client.listen():
                try:
                    if msg.event is ClientEvent.error and not msg.job_id:
                        self.report_error(f"Error communicating with server: {msg.error}")
                    elif msg.event is ClientEvent.disconnected:
                        temporary_disconnect = True
                        self.report_error("Disconnected from server, trying to reconnect...")
                    elif msg.event is ClientEvent.connected:
                        if temporary_disconnect:
                            temporary_disconnect = False
                            self.clear_error()
                    else:
                        model = self._find_model(msg.job_id)
                        if model is not None:
                            model.handle_message(msg)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    util.client_logger.exception(e)
                    self.report_error(f"Error handling server message: {str(e)}")
        except asyncio.CancelledError:
            pass  # shutdown
