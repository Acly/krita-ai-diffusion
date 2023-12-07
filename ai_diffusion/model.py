from __future__ import annotations
import asyncio
import random
from enum import Enum
from typing import NamedTuple, cast
from PyQt5.QtCore import QObject, pyqtSignal

from . import eventloop, workflow, util
from .settings import settings
from .network import NetworkError
from .image import Extent, Image, ImageCollection, Mask, Bounds
from .client import ClientMessage, ClientEvent, filter_supported_styles, resolve_sd_version
from .document import Document, LayerObserver
from .pose import Pose
from .style import Style, Styles, SDVersion
from .workflow import ControlMode, Conditioning, LiveParams
from .connection import Connection, ConnectionState
from .properties import Property, PropertyMeta
from .jobs import Job, JobKind, JobQueue, JobState
from .control import ControlLayer, ControlLayerList
import krita


class Workspace(Enum):
    generation = 0
    upscaling = 1
    live = 2


class Model(QObject, metaclass=PropertyMeta):
    """Represents diffusion workflows for a specific Krita document. Stores all inputs related to
    image generation. Launches generation jobs. Listens to server messages and keeps a
    list of finished, currently running and enqueued jobs.
    """

    _doc: Document
    _connection: Connection
    _layer: krita.Node | None = None
    _image_layers: LayerObserver

    workspace = Property(Workspace.generation, setter="set_workspace")
    style = Property(Styles.list().default)
    prompt = Property("")
    negative_prompt = Property("")
    control: ControlLayerList
    strength = Property(1.0)
    upscale: UpscaleWorkspace
    live: LiveWorkspace
    progress = Property(0.0)
    jobs: JobQueue
    error = Property("")
    can_apply_result = Property(False)

    workspace_changed = pyqtSignal(Workspace)
    style_changed = pyqtSignal(Style)
    prompt_changed = pyqtSignal(str)
    negative_prompt_changed = pyqtSignal(str)
    strength_changed = pyqtSignal(float)
    progress_changed = pyqtSignal(float)
    error_changed = pyqtSignal(str)
    can_apply_result_changed = pyqtSignal(bool)
    has_error_changed = pyqtSignal(bool)

    def __init__(self, document: Document, connection: Connection):
        super().__init__()
        self._doc = document
        self._image_layers = document.create_layer_observer()
        self._connection = connection
        self.jobs = JobQueue()
        self.control = ControlLayerList(self)
        self.upscale = UpscaleWorkspace(self)
        self.live = LiveWorkspace(self)

        self.jobs.job_finished.connect(self.update_preview)
        self.jobs.selection_changed.connect(self.update_preview)
        self.error_changed.connect(lambda: self.has_error_changed.emit(self.has_error))

        if client := connection.client_if_connected:
            self.style = next(iter(filter_supported_styles(Styles.list(), client)), self.style)
            self.upscale.upscaler = client.default_upscaler

    def generate(self):
        """Enqueue image generation for the current setup."""
        ok, msg = self._doc.check_color_mode()
        if not ok and msg:
            self.report_error(msg)
            return

        image = None
        extent = self._doc.extent

        if self._doc.active_layer.type() == "selectionmask":
            mask, image_bounds, selection_bounds = self._doc.create_mask_from_layer(
                settings.selection_padding / 100, is_inpaint=self.strength == 1.0
            )
        else:
            mask, selection_bounds = self._doc.create_mask_from_selection(
                grow=settings.selection_grow / 100,
                feather=settings.selection_feather / 100,
                padding=settings.selection_padding / 100,
            )
            image_bounds = workflow.compute_bounds(
                extent, mask.bounds if mask else None, self.strength
            )

        if mask is not None or self.strength < 1.0:
            image = self._get_current_image(image_bounds)
        if selection_bounds is not None:
            selection_bounds = Bounds.apply_crop(selection_bounds, image_bounds)
            selection_bounds = Bounds.minimum_size(selection_bounds, 64, image_bounds.extent)

        control = [c.get_image(image_bounds) for c in self.control]
        conditioning = Conditioning(self.prompt, self.negative_prompt, control)
        conditioning.area = selection_bounds if self.strength == 1.0 else None
        generator = self._generate(image_bounds, conditioning, self.strength, image, mask)

        self.clear_error()
        eventloop.run(_report_errors(self, generator))

    async def _generate(
        self,
        bounds: Bounds,
        conditioning: Conditioning,
        strength: float,
        image: Image | None,
        mask: Mask | None,
        live=LiveParams(),
    ):
        client = self._connection.client
        style = self.style
        if not self.jobs.any_executing():
            self.progress = 0.0

        if mask is not None:
            mask_bounds_rel = Bounds(  # mask bounds relative to cropped image
                mask.bounds.x - bounds.x, mask.bounds.y - bounds.y, *mask.bounds.extent
            )
            bounds = mask.bounds  # absolute mask bounds, required to insert result image
            mask.bounds = mask_bounds_rel

        if image is None and mask is None:
            assert strength == 1
            job = workflow.generate(client, style, bounds.extent, conditioning, live)
        elif mask is None and strength < 1:
            assert image is not None
            job = workflow.refine(client, style, image, conditioning, strength, live)
        elif strength == 1 and not live.is_active:
            assert image is not None and mask is not None
            job = workflow.inpaint(client, style, image, mask, conditioning)
        else:
            assert image is not None and mask is not None
            job = workflow.refine_region(client, style, image, mask, conditioning, strength, live)

        job_id = await client.enqueue(job)
        job_kind = JobKind.live_preview if live.is_active else JobKind.diffusion
        self.jobs.add(job_kind, job_id, conditioning.prompt, bounds)

    def upscale_image(self):
        image = self._doc.get_image(Bounds(0, 0, *self._doc.extent))
        job = self.jobs.add_upscale(Bounds(0, 0, *self.upscale.target_extent))
        self.clear_error()
        eventloop.run(_report_errors(self, self._upscale_image(job, image, self.upscale.params)))

    async def _upscale_image(self, job: Job, image: Image, params: UpscaleParams):
        client = self._connection.client
        upscaler = params.upscaler or client.default_upscaler
        if params.use_diffusion:
            work = workflow.upscale_tiled(
                client, image, upscaler, params.factor, self.style, params.strength
            )
        else:
            work = workflow.upscale_simple(client, image, params.upscaler, params.factor)
        job.id = await client.enqueue(work)
        self._doc.resize(params.target_extent)

    def generate_live(self):
        ver = resolve_sd_version(self.style, self._connection.client)
        image = None

        mask, _ = self._doc.create_mask_from_selection(
            grow=settings.selection_feather / 200,  # don't apply grow for live mode
            feather=settings.selection_feather / 100,
            padding=settings.selection_padding / 100,
            min_size=512 if ver is SDVersion.sd15 else 1024,
            square=True,
        )
        bounds = Bounds(0, 0, *self._doc.extent) if mask is None else mask.bounds
        if mask is not None or self.live.strength < 1.0:
            image = self._get_current_image(bounds)

        control = [c.get_image(bounds) for c in self.control]
        cond = Conditioning(self.prompt, self.negative_prompt, control)
        generator = self._generate(bounds, cond, self.live.strength, image, mask, self.live.params)

        self.clear_error()
        eventloop.run(_report_errors(self, generator))

    def _get_current_image(self, bounds: Bounds):
        exclude = [  # exclude control layers from projection
            c.layer for c in self.control if c.mode not in [ControlMode.image, ControlMode.blur]
        ]
        if self._layer:  # exclude preview layer
            exclude.append(self._layer)
        return self._doc.get_image(bounds, exclude_layers=exclude)

    def generate_control_layer(self, control: ControlLayer):
        ok, msg = self._doc.check_color_mode()
        if not ok and msg:
            self.report_error(msg)
            return

        image = self._doc.get_image(Bounds(0, 0, *self._doc.extent))
        job = self.jobs.add_control(control, Bounds(0, 0, *image.extent))
        self.clear_error()
        eventloop.run(_report_errors(self, self._generate_control_layer(job, image, control.mode)))
        return job

    async def _generate_control_layer(self, job: Job, image: Image, mode: ControlMode):
        client = self._connection.client
        work = workflow.create_control_image(image, mode)
        job.id = await client.enqueue(work)

    def cancel(self, active=False, queued=False):
        if queued:
            to_remove = [job for job in self.jobs if job.state is JobState.queued]
            if len(to_remove) > 0:
                self._connection.clear_queue()
                for job in to_remove:
                    self.jobs.remove(job)
        if active and self.jobs.any_executing():
            self._connection.interrupt()

    def report_progress(self, value):
        self.progress = value

    def report_error(self, message: str):
        self.error = message
        self.live.is_active = False

    def clear_error(self):
        if self.error != "":
            self.error = ""

    def handle_message(self, message: ClientMessage):
        job = self.jobs.find(message.job_id)
        if job is None:
            util.client_logger.error(f"Received message {message} for unknown job.")
            return

        if message.event is ClientEvent.progress:
            self.jobs.notify_started(job)
            self.report_progress(message.progress)
        elif message.event is ClientEvent.finished:
            if message.images:
                self.jobs.set_results(job, message.images)
            if job.kind is JobKind.control_layer:
                assert job.control is not None
                job.control.layer_id = self.add_control_layer(job, message.result).uniqueId()
            elif job.kind is JobKind.upscaling:
                self.add_upscale_layer(job)
            self.progress = 1
            self.jobs.notify_finished(job)
            if job.kind is not JobKind.diffusion:
                self.jobs.remove(job)
            elif job.kind is JobKind.diffusion and self._layer is None and job.id:
                self.jobs.select(job.id, 0)
        elif message.event is ClientEvent.interrupted:
            job.state = JobState.cancelled
            self.report_progress(0)
        elif message.event is ClientEvent.error:
            job.state = JobState.cancelled
            self.report_error(f"Server execution error: {message.error}")

    def update_preview(self):
        if selection := self.jobs.selection:
            self.show_preview(selection.job, selection.image)
            self.can_apply_result = True
        else:
            self.hide_preview()
            self.can_apply_result = False

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

    def hide_preview(self):
        if self._layer is not None:
            self._doc.hide_layer(self._layer)

    def apply_result(self):
        assert self._layer and self.can_apply_result
        self._layer.setLocked(False)
        self._layer.setName(self._layer.name().replace("[Preview]", "[Generated]"))
        self._layer = None
        self.jobs.selection = None

    def add_control_layer(self, job: Job, result: dict | None):
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

    def set_workspace(self, workspace: Workspace):
        if self.workspace is Workspace.live:
            self.live.is_active = False
        self._workspace = workspace
        self.workspace_changed.emit(workspace)

    @property
    def history(self):
        return (job for job in self.jobs if job.state is JobState.finished)

    @property
    def has_live_result(self):
        return self._live_result is not None

    @property
    def has_error(self):
        return self.error != ""

    @property
    def document(self):
        return self._doc

    @property
    def image_layers(self):
        return self._image_layers

    @property
    def is_active(self):
        return self._doc.is_active

    @property
    def is_valid(self):
        return self._doc.is_valid


class UpscaleParams(NamedTuple):
    upscaler: str
    factor: float
    use_diffusion: bool
    strength: float
    target_extent: Extent


class UpscaleWorkspace(QObject, metaclass=PropertyMeta):
    upscaler = Property("")
    factor = Property(2.0)
    use_diffusion = Property(True)
    strength = Property(0.3)
    target_extent = Property(Extent(1, 1))

    upscaler_changed = pyqtSignal(str)
    factor_changed = pyqtSignal(float)
    use_diffusion_changed = pyqtSignal(bool)
    strength_changed = pyqtSignal(float)
    target_extent_changed = pyqtSignal(Extent)

    _model: Model

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        if client := model._connection.client_if_connected:
            self.upscaler = client.default_upscaler
        self.factor_changed.connect(self._update_target_extent)
        self._update_target_extent()

    def _update_target_extent(self):
        self.target_extent = self._model.document.extent * self.factor

    @property
    def params(self):
        self._update_target_extent()
        return UpscaleParams(
            upscaler=self.upscaler,
            factor=self.factor,
            use_diffusion=self.use_diffusion,
            strength=self.strength,
            target_extent=self.target_extent,
        )


class LiveWorkspace(QObject, metaclass=PropertyMeta):
    is_active = Property(False, setter="toggle")
    strength = Property(0.3)
    seed = Property(0)
    has_result = Property(False)

    is_active_changed = pyqtSignal(bool)
    strength_changed = pyqtSignal(float)
    seed_changed = pyqtSignal(int)
    has_result_changed = pyqtSignal(bool)
    result_available = pyqtSignal(Image)

    _model: Model
    _result: Image | None = None
    _result_bounds: Bounds | None = None

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        self.generate_seed()
        model.jobs.job_finished.connect(self.handle_job_finished)

    def generate_seed(self):
        self.seed = random.randint(0, 2**31 - 1)

    def toggle(self, active: bool):
        if active != self.is_active:
            self._is_active = active
            self.is_active_changed.emit(active)
            if active:
                self._model.generate_live()

    def handle_job_finished(self, job: Job):
        if job.kind is JobKind.live_preview:
            if len(job.results) > 0:
                self.set_result(job.results[0], job.bounds)
            self.is_active = self._is_active and self._model.is_active
            if self.is_active:
                self._model.generate_live()

    def copy_result_to_layer(self):
        assert self.result is not None and self._result_bounds is not None
        doc = self._model.document
        doc.insert_layer(f"[Live] {self._model.prompt}", self.result, self._result_bounds)

    @property
    def result(self):
        return self._result

    def set_result(self, value: Image, bounds: Bounds):
        self._result = value
        self._result_bounds = bounds
        self.result_available.emit(value)
        self.has_result = True

    @property
    def params(self):
        return LiveParams(is_active=self.is_active, seed=self.seed)


async def _report_errors(parent, coro):
    try:
        return await coro
    except NetworkError as e:
        parent.report_error(f"{util.log_error(e)} [url={e.url}, code={e.code}]")
    except Exception as e:
        parent.report_error(util.log_error(e))
