from __future__ import annotations
import asyncio
from copy import copy
from pathlib import Path
from enum import Enum
from typing import Any, NamedTuple
from PyQt5.QtCore import QObject, QUuid, pyqtSignal
from PyQt5.QtGui import QImage
import uuid

from . import eventloop, workflow, util
from .api import (
    ConditioningInput,
    RegionInput,
    WorkflowKind,
    WorkflowInput,
    InpaintMode,
    InpaintParams,
    FillMode,
)
from .util import ensure, client_logger as log
from .settings import settings
from .network import NetworkError
from .image import Extent, Image, Mask, Bounds
from .client import ClientMessage, ClientEvent, filter_supported_styles, resolve_sd_version
from .document import Document, LayerObserver, KritaDocument
from .pose import Pose
from .style import Style, Styles, SDVersion
from .connection import Connection
from .properties import Property, ObservableProperties
from .jobs import Job, JobKind, JobParams, JobQueue, JobState
from .control import ControlLayer, ControlLayerList
from .resources import ControlMode
from .resolution import compute_bounds, compute_relative_bounds
import krita


class Workspace(Enum):
    generation = 0
    upscaling = 1
    live = 2
    animation = 3


class Region(QObject, ObservableProperties):
    _tree: "RegionTree"

    layer_id = Property("", persist=True)
    prompt = Property("", persist=True)
    negative_prompt = Property("", persist=True)
    control: ControlLayerList

    layer_id_changed = pyqtSignal(str)
    prompt_changed = pyqtSignal(str)
    negative_prompt_changed = pyqtSignal(str)
    modified = pyqtSignal(QObject, str)  # TODO: hook this up

    def __init__(self, tree: "RegionTree", model: "Model", layer_id: QUuid | str | None = None):
        super().__init__()
        self._tree = tree
        self.layer_id = _layer_id_str(layer_id)
        self.control = ControlLayerList(model)

    @property
    def layer(self):
        if not self.is_root:
            layer = self._tree._model.layers.updated().find(QUuid(self.layer_id))
            assert layer is not None, f"Region layer not found ({self.layer_id} {self.prompt})"
            return layer
        return None  # root region, no group layer

    @property
    def siblings(self):
        return self._tree.siblings(self)

    @property
    def is_root(self):
        return self.layer_id == ""

    @property
    def name(self):
        if layer := self.layer:
            return layer.name()
        return "Root"

    def to_api(self, bounds: Bounds | None = None):
        doc = self._tree._model.document
        layer = ensure(self.layer, "Non-root region required")
        return RegionInput(
            mask=doc.get_layer_mask(layer, bounds),
            positive=self.prompt,
            negative=self.negative_prompt,
            control=[c.to_api(bounds) for c in self.control],
        )


class RegionTree(QObject):
    _model: "Model"
    _root: Region
    _regions: list[Region]
    _active: Region | None = None
    _active_layer: QUuid | None = None

    active_changed = pyqtSignal(Region)
    added = pyqtSignal(Region)
    removed = pyqtSignal(Region)

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        self._root = Region(self, model)
        self._regions = []
        model.layers.active_changed.connect(self._update_active)
        model.layers.changed.connect(self._update_layers)

    @property
    def root(self):
        return self._root

    def _lookup_region(self, layer_id: QUuid | None = None):
        layer_id_str = _layer_id_str(layer_id)
        region = next((r for r in self._regions if r.layer_id == layer_id_str), None)
        if region is None:
            region = self._add(layer_id)
        return region

    def emplace(self):
        region = Region(self, self._model)
        self._regions.append(region)
        return region

    @property
    def active(self):
        self._update_active()
        return self._active or Region(self, self._model)

    @active.setter
    def active(self, region: Region):
        if self._active != region:
            self._active = region
            if layer := region.layer:
                non_group = (l for l in reversed(layer.childNodes()) if l.type() != "grouplayer")
                top_non_group = next(non_group, None)
                self._model.document.active_layer = top_non_group or layer
            self.active_changed.emit(region)

    def add_control(self):
        self.active.control.add()

    def to_api(self, parent_layer_id: QUuid | None, bounds: Bounds | None = None):
        # Assemble all regions by finding group layers which are direct children of the parent layer.
        # Ignore regions with no prompt or control layers.
        layers = self._model.layers
        parent_layer = layers.find(parent_layer_id) if parent_layer_id else layers.root
        api_regions: list[RegionInput] = []
        for layer in layers:
            if layer.type() == "grouplayer" and layer.parentNode() == parent_layer:
                region = self._lookup_region(layer.uniqueId())
                if region.prompt != "" or len(region.control) > 0:
                    api_regions.append(region.to_api(bounds))

        # Remove from each region mask any overlapping areas from regions above it.
        accumulated_mask = None
        for region in reversed(api_regions):
            if accumulated_mask is None:
                accumulated_mask = Image.mask_add(region.mask, region.mask)
            else:
                current = region.mask
                region.mask = Image.mask_subtract(region.mask, accumulated_mask)
                accumulated_mask = Image.mask_add(accumulated_mask, current)

        # If the regions don't cover the entire image, add a final region for the remaining area.
        if accumulated_mask is not None:
            average = Image.scale(accumulated_mask, Extent(1, 1)).pixel(0, 0)
            fully_covered = isinstance(average, tuple) and average[0] >= 254
            if not fully_covered:
                accumulated_mask.invert()
                api_regions.append(RegionInput(accumulated_mask, self.root.prompt))

        return ConditioningInput(
            positive=self.root.prompt,
            negative=self.root.negative_prompt,
            control=[c.to_api(bounds) for c in self.root.control],
            regions=api_regions,
        )

    def siblings(self, region: Region):
        def get_regions(layers: list[krita.Node]):
            return [self._lookup_region(l.uniqueId()) for l in layers]

        below, above = self._model.layers.siblings(region.layer, "grouplayer")
        return get_regions(below), get_regions(above)

    def _update_layers(self):
        self._prune()
        for layer in self._model.layers:
            if layer.type() == "grouplayer":
                self._lookup_region(layer.uniqueId())

    def _update_active(self):
        if not isinstance(self._model.document, KritaDocument):
            return
        layer = self._model.document.active_layer
        if layer.uniqueId() == self._active_layer:
            return
        self._active_layer = layer.uniqueId()

        region = self.root
        while layer is not None and layer.type() != "grouplayer":
            layer = layer.parentNode()
        if layer is not None and layer.parentNode() is not None:
            assert layer.type() == "grouplayer"
            region = self._lookup_region(layer.uniqueId())
        if region != self._active:
            self._active = region
            self.active_changed.emit(region)

    def _add(self, layer_id: str):
        region = Region(self, self._model, layer_id)
        self._regions.append(region)
        self.added.emit(region)
        # handle new group for active layer
        if layer := region.layer:
            if layer.type() == "grouplayer":
                self._active_layer = None  # force check for new region
                self._update_active()
        return region

    def _prune(self):
        layers = self._model.layers.updated()
        new_regions, removed = [], []
        for region in self._regions:
            if region.layer_id == "" or layers.find(QUuid(region.layer_id)):
                new_regions.append(region)
            else:
                removed.append(region)
        self._regions = new_regions
        for region in removed:
            self.removed.emit(region)

    def __len__(self):
        self._prune()
        return len(self._regions)

    def __iter__(self):
        self._prune()
        return iter(self._regions)


def _layer_id_str(a: QUuid | str | None):
    if isinstance(a, QUuid):
        return a.toString()
    if a is None:
        return ""
    return a


class Model(QObject, ObservableProperties):
    """Represents diffusion workflows for a specific Krita document. Stores all inputs related to
    image generation. Launches generation jobs. Listens to server messages and keeps a
    list of finished, currently running and enqueued jobs.
    """

    _doc: Document
    _connection: Connection
    _layer: krita.Node | None = None
    _layers: LayerObserver

    workspace = Property(Workspace.generation, setter="set_workspace", persist=True)
    regions: "RegionTree"
    style = Property(Styles.list().default, persist=True)
    strength = Property(1.0, persist=True)
    batch_count = Property(1, persist=True)
    seed = Property(0, persist=True)
    fixed_seed = Property(False, persist=True)
    queue_front = Property(False, persist=True)
    inpaint: CustomInpaint
    upscale: "UpscaleWorkspace"
    live: "LiveWorkspace"
    animation: "AnimationWorkspace"
    progress = Property(0.0)
    jobs: JobQueue
    error = Property("")

    workspace_changed = pyqtSignal(Workspace)
    style_changed = pyqtSignal(Style)
    strength_changed = pyqtSignal(float)
    batch_count_changed = pyqtSignal(int)
    seed_changed = pyqtSignal(int)
    fixed_seed_changed = pyqtSignal(bool)
    queue_front_changed = pyqtSignal(bool)
    progress_changed = pyqtSignal(float)
    error_changed = pyqtSignal(str)
    has_error_changed = pyqtSignal(bool)
    modified = pyqtSignal(QObject, str)

    def __init__(self, document: Document, connection: Connection):
        super().__init__()
        self._doc = document
        self._layers = document.create_layer_observer()
        self._connection = connection
        self.generate_seed()
        self.jobs = JobQueue()
        self.regions = RegionTree(self)
        self.inpaint = CustomInpaint()
        self.upscale = UpscaleWorkspace(self)
        self.live = LiveWorkspace(self)
        self.animation = AnimationWorkspace(self)

        self.jobs.selection_changed.connect(self.update_preview)
        self.error_changed.connect(lambda: self.has_error_changed.emit(self.has_error))
        connection.state_changed.connect(self._init_on_connect)
        Styles.list().changed.connect(self._init_on_connect)
        self._init_on_connect()

    def _init_on_connect(self):
        if client := self._connection.client_if_connected:
            styles = filter_supported_styles(Styles.list().filtered(), client)
            if self.style not in styles and len(styles) > 0:
                self.style = styles[0]
            if self.upscale.upscaler == "":
                self.upscale.upscaler = client.models.default_upscaler

    def generate(self):
        """Enqueue image generation for the current setup."""
        ok, msg = self._doc.check_color_mode()
        if not ok and msg:
            self.report_error(msg)
            return

        workflow_kind = WorkflowKind.generate if self.strength == 1.0 else WorkflowKind.refine
        client = self._connection.client
        image = None
        inpaint = None
        region = self.regions.root
        extent = self._doc.extent
        mask = self._doc.create_mask_from_selection(
            **get_selection_modifiers(self.inpaint.mode, self.strength), min_size=64
        )
        bounds = compute_bounds(extent, mask.bounds if mask else None, self.strength)
        bounds = self.inpaint.get_context(self, mask) or bounds

        conditioning = self.regions.to_api(None, bounds)

        if mask is not None or self.strength < 1.0:
            image = self._get_current_image(region, bounds)

        if mask is not None:
            if workflow_kind is WorkflowKind.generate:
                workflow_kind = WorkflowKind.inpaint
            elif workflow_kind is WorkflowKind.refine:
                workflow_kind = WorkflowKind.refine_region

            bounds, mask.bounds = compute_relative_bounds(bounds, mask.bounds)

            sd_version = client.models.version_of(self.style.sd_checkpoint)
            inpaint_mode = self.resolve_inpaint_mode()
            if inpaint_mode is InpaintMode.custom:
                inpaint = self.inpaint.get_params(mask)
            else:
                control = conditioning.control
                inpaint = workflow.detect_inpaint(
                    inpaint_mode, mask.bounds, sd_version, region.prompt, control, self.strength
                )
        try:
            input = workflow.prepare(
                workflow_kind,
                image or extent,
                conditioning,
                self.style,
                self.seed if self.fixed_seed else workflow.generate_seed(),
                client.models,
                client.performance_settings,
                mask=mask,
                strength=self.strength,
                inpaint=inpaint,
            )
        except Exception as e:
            self.report_error(util.log_error(e))
            return
        self.clear_error()
        enqueue_jobs = self.enqueue_jobs(
            input, JobKind.diffusion, JobParams(bounds, region.prompt), self.batch_count
        )
        eventloop.run(_report_errors(self, enqueue_jobs))

    async def enqueue_jobs(
        self, input: WorkflowInput, kind: JobKind, params: JobParams, count: int = 1
    ):
        sampling = ensure(input.sampling)
        params.negative_prompt = ensure(input.conditioning).negative
        params.strength = sampling.denoise_strength

        for i in range(count):
            sampling.seed = sampling.seed + i * settings.batch_size
            params.seed = sampling.seed
            job = self.jobs.add(kind, copy(params))
            await self._enqueue_job(job, input)

    async def _enqueue_job(self, job: Job, input: WorkflowInput):
        if not self.jobs.any_executing():
            self.progress = 0.0
        client = self._connection.client
        job.id = await client.enqueue(input, self.queue_front)

    def upscale_image(self):
        try:
            params = self.upscale.params
            image = self._doc.get_image(Bounds(0, 0, *self._doc.extent))
            client = self._connection.client
            upscaler = params.upscaler or client.models.default_upscaler

            if params.use_diffusion:
                inputs = workflow.prepare(
                    WorkflowKind.upscale_tiled,
                    image,
                    ConditioningInput("4k uhd"),
                    self.style,
                    params.seed,
                    client.models,
                    client.performance_settings,
                    strength=params.strength,
                    upscale_factor=params.factor,
                    upscale_model=upscaler,
                )
            else:
                inputs = workflow.prepare_upscale_simple(image, upscaler, params.factor)
            job = self.jobs.add_upscale(Bounds(0, 0, *self.upscale.target_extent), params.seed)
        except Exception as e:
            self.report_error(util.log_error(e))
            return

        self.clear_error()
        eventloop.run(_report_errors(self, self._enqueue_job(job, inputs)))

    def generate_live(self):
        eventloop.run(_report_errors(self, self._generate_live()))

    async def _generate_live(self, last_input: WorkflowInput | None = None):
        strength = self.live.strength
        workflow_kind = WorkflowKind.generate if strength == 1.0 else WorkflowKind.refine
        client = self._connection.client
        ver = client.models.version_of(self.style.sd_checkpoint)
        region = self.regions.root

        image = None
        mask = self._doc.create_mask_from_selection(
            grow=settings.selection_feather / 200,  # don't apply grow for live mode
            feather=settings.selection_feather / 100,
            padding=settings.selection_padding / 100,
            min_size=512 if ver is SDVersion.sd15 else 1024,
            square=True,
        )
        bounds = Bounds(0, 0, *self._doc.extent)
        if mask is not None:
            workflow_kind = WorkflowKind.refine_region
            bounds, mask.bounds = compute_relative_bounds(mask.bounds, mask.bounds)
        if mask is not None or self.live.strength < 1.0:
            image = self._get_current_image(region, bounds)

        input = workflow.prepare(
            workflow_kind,
            image or bounds.extent,
            self.regions.to_api(None, mask.bounds if mask else None),
            self.style,
            self.seed,
            client.models,
            client.performance_settings,
            mask=mask,
            strength=self.live.strength,
            inpaint=InpaintParams(InpaintMode.fill, mask.bounds) if mask else None,
            is_live=True,
        )
        if input != last_input:
            self.clear_error()
            await self.enqueue_jobs(input, JobKind.live_preview, JobParams(bounds, region.prompt))
            return input

        return None

    def _get_current_image(self, region: Region, bounds: Bounds):
        exclude = [  # exclude control layers from projection
            c.layer for c in region.control if not c.mode.is_part_of_image
        ]
        if self._layer:  # exclude preview layer
            exclude.append(self._layer)
        return self._doc.get_image(bounds, exclude_layers=exclude)

    def generate_control_layer(self, control: ControlLayer):
        ok, msg = self._doc.check_color_mode()
        if not ok and msg:
            self.report_error(msg)
            return

        try:
            image = self._doc.get_image(Bounds(0, 0, *self._doc.extent))
            mask = self.document.create_mask_from_selection(0, 0, padding=0.25, multiple=64)
            bounds = mask.bounds if mask else None
            perf = self._connection.client.performance_settings
            input = workflow.prepare_create_control_image(image, control.mode, perf, bounds)
            job = self.jobs.add_control(control, Bounds(0, 0, *image.extent))
        except Exception as e:
            self.report_error(util.log_error(e))
            return

        self.clear_error()
        eventloop.run(_report_errors(self, self._enqueue_job(job, input)))
        return job

    def cancel(self, active=False, queued=False):
        if queued:
            to_remove = [job for job in self.jobs if job.state is JobState.queued]
            if len(to_remove) > 0:
                self._connection.clear_queue()
                for job in to_remove:
                    self.jobs.remove(job)
        if active and self.jobs.any_executing():
            self._connection.interrupt()

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

        if message.event is ClientEvent.queued:
            self.jobs.notify_started(job)
            self.progress = -1
            self.progress_changed.emit(-1)
        elif message.event is ClientEvent.progress:
            self.jobs.notify_started(job)
            self.progress = message.progress
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
            elif settings.auto_preview and self._layer is None and job.id:
                self.jobs.select(job.id, 0)
        elif message.event is ClientEvent.interrupted:
            self.jobs.notify_cancelled(job)
            self.progress = 0
        elif message.event is ClientEvent.error:
            self.jobs.notify_cancelled(job)
            self.report_error(f"Server execution error: {message.error}")

    def update_preview(self):
        if selection := self.jobs.selection:
            self.show_preview(selection.job, selection.image)
        else:
            self.hide_preview()

    def show_preview(self, job_id: str, index: int, name_prefix="Preview"):
        job = self.jobs.find(job_id)
        assert job is not None, "Cannot show preview, invalid job id"
        name = f"[{name_prefix}] {job.params.prompt}"
        if self._layer and self._layer not in self.layers:
            self._layer = None  # layer was removed by user
        if self._layer is not None:
            self._layer.setName(name)
            self._doc.set_layer_content(self._layer, job.results[index], job.params.bounds)
            self._doc.move_to_top(self._layer)
        else:
            self._layer = self._doc.insert_layer(
                name, job.results[index], job.params.bounds, make_active=False
            )
            self._layer.setLocked(True)

    def hide_preview(self):
        if self._layer is not None:
            self._doc.hide_layer(self._layer)

    def apply_result(self, job_id: str, index: int):
        self.jobs.select(job_id, index)
        assert self._layer is not None
        self._layer.setLocked(False)
        self._layer.setName(self._layer.name().replace("[Preview]", "[Generated]"))
        self._doc.active_layer = self._layer
        self._layer = None
        self.jobs.selection = None
        self.jobs.notify_used(job_id, index)

    def add_control_layer(self, job: Job, result: dict | None):
        assert job.kind is JobKind.control_layer and job.control
        if job.control.mode is ControlMode.pose and result is not None:
            pose = Pose.from_open_pose_json(result)
            pose.scale(job.params.bounds.extent)
            return self._doc.insert_vector_layer(job.params.prompt, pose.to_svg())
        elif len(job.results) > 0:
            return self._doc.insert_layer(job.params.prompt, job.results[0], job.params.bounds)
        return self.document.active_layer  # Execution was cached and no image was produced

    def add_upscale_layer(self, job: Job):
        assert job.kind is JobKind.upscaling
        assert len(job.results) > 0, "Upscaling job did not produce an image"
        if self._layer:
            self._layer.remove()
            self._layer = None
        self._doc.resize(job.params.bounds.extent)
        self.upscale.target_extent_changed.emit(self.upscale.target_extent)
        self._doc.insert_layer(job.params.prompt, job.results[0], job.params.bounds)

    def set_workspace(self, workspace: Workspace):
        if self.workspace is Workspace.live:
            self.live.is_active = False
        self._workspace = workspace
        self.workspace_changed.emit(workspace)
        self.modified.emit(self, "workspace")

    def generate_seed(self):
        self.seed = workflow.generate_seed()

    def save_result(self, job_id: str, index: int):
        _save_job_result(self, self.jobs.find(job_id), index)

    def resolve_inpaint_mode(self):
        if self.inpaint.mode is InpaintMode.automatic:
            if bounds := self.document.selection_bounds:
                return workflow.detect_inpaint_mode(self.document.extent, bounds)
            return InpaintMode.fill
        return self.inpaint.mode

    @property
    def sd_version(self):
        return resolve_sd_version(self.style, self._connection.client_if_connected)

    @property
    def history(self):
        return (job for job in self.jobs if job.state is JobState.finished)

    @property
    def has_error(self):
        return self.error != ""

    @property
    def document(self):
        return self._doc

    @document.setter
    def document(self, doc):
        # Note: for some reason Krita sometimes creates a new object for an existing document.
        # The old object is deleted and unusable. This method is used to update the object,
        # but doesn't actually change the document identity.
        # TODO: 04/02/2024 is this still necessary? check log.
        assert doc == self._doc, "Cannot change document of model"
        if self._doc is not doc:
            log.warning(f"Document instance changed {self._doc} -> {doc}")
            self._doc = doc

    @property
    def layers(self):
        return self._layers


class InpaintContext(Enum):
    automatic = 0
    mask_bounds = 1
    entire_image = 2
    layer_bounds = 3


class CustomInpaint(QObject, ObservableProperties):
    mode = Property(InpaintMode.automatic, persist=True)
    fill = Property(FillMode.neutral, persist=True)
    use_inpaint = Property(True, persist=True)
    use_prompt_focus = Property(False, persist=True)
    context = Property(InpaintContext.automatic, persist=True)
    context_layer_id = Property(QUuid(), persist=True)

    mode_changed = pyqtSignal(InpaintMode)
    fill_changed = pyqtSignal(FillMode)
    use_inpaint_changed = pyqtSignal(bool)
    use_prompt_focus_changed = pyqtSignal(bool)
    context_changed = pyqtSignal(InpaintContext)
    context_layer_id_changed = pyqtSignal(QUuid)
    modified = pyqtSignal(QObject, str)

    def get_params(self, mask: Mask):
        params = InpaintParams(self.mode, mask.bounds, self.fill)
        params.use_inpaint_model = self.use_inpaint
        params.use_condition_mask = self.use_prompt_focus
        params.use_single_region = self.use_prompt_focus
        return params

    def get_context(self, model: Model, mask: Mask | None):
        if mask is None or self.mode is not InpaintMode.custom:
            return None
        if self.context is InpaintContext.mask_bounds:
            return mask.bounds
        if self.context is InpaintContext.entire_image:
            return Bounds(0, 0, *model.document.extent)
        if self.context is InpaintContext.layer_bounds:
            if layer := model.layers.find(self.context_layer_id):
                layer_bounds = model.document.get_mask_bounds(layer)
                return Bounds.expand(layer_bounds, include=mask.bounds)
        return None


class UpscaleParams(NamedTuple):
    upscaler: str
    factor: float
    use_diffusion: bool
    strength: float
    target_extent: Extent
    seed: int


class UpscaleWorkspace(QObject, ObservableProperties):
    upscaler = Property("", persist=True)
    factor = Property(2.0, persist=True)
    use_diffusion = Property(True, persist=True)
    strength = Property(0.3, persist=True)

    upscaler_changed = pyqtSignal(str)
    factor_changed = pyqtSignal(float)
    use_diffusion_changed = pyqtSignal(bool)
    strength_changed = pyqtSignal(float)
    target_extent_changed = pyqtSignal(Extent)
    modified = pyqtSignal(QObject, str)

    _model: Model

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        self.factor_changed.connect(lambda _: self.target_extent_changed.emit(self.target_extent))
        self._init_model()
        model._connection.models_changed.connect(self._init_model)

    def _init_model(self):
        if self.upscaler == "":
            if client := self._model._connection.client_if_connected:
                self.upscaler = client.models.default_upscaler

    @property
    def target_extent(self):
        return self._model.document.extent * self.factor

    @property
    def params(self):
        return UpscaleParams(
            upscaler=self.upscaler,
            factor=self.factor,
            use_diffusion=self.use_diffusion,
            strength=self.strength,
            target_extent=self.target_extent,
            seed=self._model.seed if self._model.fixed_seed else workflow.generate_seed(),
        )


class LiveWorkspace(QObject, ObservableProperties):
    is_active = Property(False, setter="toggle")
    is_recording = Property(False, setter="toggle_record")
    strength = Property(0.3, persist=True)
    has_result = Property(False)

    is_active_changed = pyqtSignal(bool)
    is_recording_changed = pyqtSignal(bool)
    strength_changed = pyqtSignal(float)
    seed_changed = pyqtSignal(int)
    has_result_changed = pyqtSignal(bool)
    result_available = pyqtSignal(Image)
    modified = pyqtSignal(QObject, str)

    _model: Model
    _last_input: WorkflowInput | None = None
    _result: Image | None = None
    _result_bounds: Bounds | None = None
    _result_seed: int | None = None
    _keyframes_folder: Path | None = None
    _keyframe_start = 0
    _keyframe_index = 0
    _keyframes: list[Path]

    _poll_rate = 0.1

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        self._keyframes = []
        model.jobs.job_finished.connect(self.handle_job_finished)

    def toggle(self, active: bool):
        if self.is_active != active:
            self._is_active = active
            self.is_active_changed.emit(active)
            if active:
                eventloop.run(_report_errors(self._model, self._continue_generating()))
            else:
                self.is_recording = False

    def toggle_record(self, active: bool):
        if self.is_recording != active:
            if active and not self._start_recording():
                self._model.report_error(
                    "Cannot save recorded frames, document must be saved first!"
                )
                return
            self._is_recording = active
            self.is_active = active
            self.is_recording_changed.emit(active)
            if not active:
                self._import_animation()

    def handle_job_finished(self, job: Job):
        if job.kind is JobKind.live_preview:
            if len(job.results) > 0:
                self.set_result(job.results[0], job.params.bounds, job.params.seed)
            self.is_active = self._is_active and self._model.document.is_active
            eventloop.run(_report_errors(self._model, self._continue_generating()))

    async def _continue_generating(self):
        while self.is_active and self._model.document.is_active:
            new_input = await self._model._generate_live(self._last_input)
            if new_input is not None:  # frame was scheduled
                self._last_input = new_input
                return
            # no changes in input data
            await asyncio.sleep(self._poll_rate)

    def copy_result_to_layer(self):
        assert self.result is not None and self._result_bounds is not None
        doc = self._model.document
        name = f"{self._model.regions.active.prompt} ({self._result_seed})"
        doc.insert_layer(name, self.result, self._result_bounds)
        if settings.new_seed_after_apply:
            self._model.generate_seed()

    @property
    def result(self):
        return self._result

    def set_result(self, value: Image, bounds: Bounds, seed: int):
        self._result = value
        self._result_bounds = bounds
        self._result_seed = seed
        self.result_available.emit(value)
        self.has_result = True

        if self.is_recording:
            self._save_frame(value, bounds)

    def _start_recording(self):
        doc_filename = self._model.document.filename
        if doc_filename:
            path = Path(doc_filename)
            folder = path.parent / f"{path.with_suffix('.live-frames')}"
            folder.mkdir(exist_ok=True)
            self._keyframes_folder = folder
            while (self._keyframes_folder / f"frame-{self._keyframe_index}.webp").exists():
                self._keyframe_index += 1
            self._keyframe_start = self._keyframe_index
        else:
            self._keyframes_folder = None
        return self._keyframes_folder

    def _save_frame(self, image: Image, bounds: Bounds):
        assert self._keyframes_folder is not None
        filename = self._keyframes_folder / f"frame-{self._keyframe_index}.webp"
        self._keyframe_index += 1

        extent = self._model.document.extent
        if bounds is not None and bounds.extent != extent:
            image = Image.crop(image, bounds)
        image.save(filename)
        self._keyframes.append(filename)

    def _import_animation(self):
        if len(self._keyframes) == 0:
            return  # button toggled without recording a frame in between
        self._model.document.import_animation(self._keyframes, self._keyframe_start)
        start, end = self._keyframe_start, self._keyframe_start + len(self._keyframes)
        prompt = self._model.regions.active.prompt
        self._model.document.active_layer.setName(f"[Rec] {start}-{end}: {prompt}")
        self._keyframes = []


class SamplingQuality(Enum):
    fast = 0
    quality = 1


class AnimationWorkspace(QObject, ObservableProperties):
    sampling_quality = Property(SamplingQuality.fast, persist=True)
    target_layer = Property(QUuid(), persist=True)
    batch_mode = Property(True, persist=True)

    sampling_quality_changed = pyqtSignal(SamplingQuality)
    target_layer_changed = pyqtSignal(QUuid)
    batch_mode_changed = pyqtSignal(bool)
    target_image_changed = pyqtSignal(Image)
    modified = pyqtSignal(QObject, str)

    _model: Model
    _keyframes_folder: Path | None = None
    _keyframes: dict[str, list[Path]]

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        self._keyframes = {}
        self.target_layer_changed.connect(self._update_target_image)
        model.document.current_time_changed.connect(self._update_target_image)
        model.jobs.job_finished.connect(self.handle_job_finished)

    def generate(self):
        if self.batch_mode:
            self.generate_batch()
        else:
            self.generate_frame()

    def generate_frame(self):
        self._model.clear_error()
        eventloop.run(_report_errors(self._model, self._generate_frame()))

    def _prepare_input(self, canvas: Image | Extent, seed: int):
        m = self._model
        bounds = Bounds(0, 0, *m.document.extent)
        return workflow.prepare(
            WorkflowKind.generate if m.strength == 1.0 else WorkflowKind.refine,
            canvas,
            m.regions.to_api(None, bounds),
            style=m.style,
            seed=seed,
            perf=m._connection.client.performance_settings,
            models=m._connection.client.models,
            strength=m.strength,
            is_live=self.sampling_quality is SamplingQuality.fast,
        )

    async def _generate_frame(self):
        m = self._model
        region = m.regions.root
        bounds = Bounds(0, 0, *m.document.extent)
        canvas = m._get_current_image(region, bounds) if m.strength < 1.0 else bounds.extent
        seed = m.seed if m.fixed_seed else workflow.generate_seed()
        inputs = self._prepare_input(canvas, seed)
        params = JobParams(bounds, region.prompt, frame=(m.document.current_time, 0, 0))
        await m.enqueue_jobs(inputs, JobKind.animation_frame, params)

    def generate_batch(self):
        doc = self._model.document
        if self._model.strength < 1.0 and not doc.active_layer.animated():
            self._model.report_error("The active layer does not contain an animation.")
            return

        if doc.filename:
            path = Path(doc.filename)
            folder = path.parent / f"{path.with_suffix('.animation')}"
            folder.mkdir(exist_ok=True)
            self._keyframes_folder = folder
        else:
            self._model.report_error("Document must be saved before generating an animation.")
            return

        self._model.clear_error()
        eventloop.run(_report_errors(self._model, self._generate_batch()))

    async def _generate_batch(self):
        doc = self._model.document
        layer = doc.active_layer
        start_frame, end_frame = doc.playback_time_range
        extent = doc.extent
        bounds = Bounds(0, 0, *extent)
        strength = self._model.strength
        seed = self._model.seed if self._model.fixed_seed else workflow.generate_seed()
        animation_id = str(uuid.uuid4())

        for frame in range(start_frame, end_frame + 1):
            if layer.hasKeyframeAtTime(frame) or strength == 1.0:
                canvas: Image | Extent = extent
                if strength < 1.0:
                    pixels = layer.pixelDataAtTime(0, 0, extent.width, extent.height, frame)
                    canvas = Image(
                        QImage(pixels, extent.width, extent.height, QImage.Format_ARGB32)
                    )

                inputs = self._prepare_input(canvas, seed)
                params = JobParams(bounds, self._model.regions.active.prompt)
                params.frame = (frame, start_frame, end_frame)
                params.animation_id = animation_id
                await self._model.enqueue_jobs(inputs, JobKind.animation_batch, params)

    def handle_job_finished(self, job: Job):
        if job.kind is JobKind.animation_batch:
            assert self._keyframes_folder is not None
            frame, _, end = job.params.frame
            keyframes = self._keyframes.setdefault(job.params.animation_id, [])
            if len(job.results) > 0:
                image = job.results[0]
                filename = self._keyframes_folder / f"frame-{frame}.png"
                image.save(filename)
                keyframes.append(filename)
                self.target_image_changed.emit(image)
            elif len(keyframes) > 0:
                # Execution was cached because image content is the same as previous frame
                keyframes.append(keyframes[-1])
            if frame == end:
                self._import_animation(job)

        elif job.kind is JobKind.animation_frame:
            if len(job.results) > 0:
                doc = self._model.document
                if job.params.frame[0] != doc.current_time:
                    self._model.report_error("Generated frame does not match current time")
                    return
                if layer := self._model.layers.find(self.target_layer):
                    image = job.results[0]
                    doc.set_layer_content(layer, image, job.params.bounds, make_visible=False)
                    self.target_image_changed.emit(image)
                else:
                    self._model.report_error("Target layer not found")

    def _import_animation(self, job: Job):
        doc = self._model.document
        keyframes = self._keyframes.pop(job.params.animation_id)
        _, start, end = job.params.frame
        doc.import_animation(keyframes, start)
        doc.active_layer.setName(f"[Generated] {start}-{end}: {job.params.prompt}")
        self.target_layer = doc.active_layer.uniqueId()

    def _update_target_image(self):
        if self.batch_mode:
            return
        if layer := self._model.layers.find(self.target_layer):
            bounds = Bounds(0, 0, *self._model.document.extent)
            image = self._model.document.get_layer_image(layer, bounds)
            self.target_image_changed.emit(image)


def get_selection_modifiers(inpaint_mode: InpaintMode, strength: float) -> dict[str, Any]:
    grow = settings.selection_grow / 100
    feather = settings.selection_feather / 100
    padding = settings.selection_padding / 100
    invert = False

    if inpaint_mode is InpaintMode.remove_object and strength == 1.0:
        # avoid leaving any border pixels of the object to be removed within the
        # area where the mask is 1.0, it will confuse inpainting models
        feather = min(feather, grow * 0.5)

    if inpaint_mode is InpaintMode.replace_background and strength == 1.0:
        # only minimal grow/feather as there is often no desired transition between
        # forground object and background (to be replaced by something else entirely)
        grow = min(grow, 0.01)
        feather = min(feather, 0.01)
        invert = True

    return dict(grow=grow, feather=feather, padding=padding, invert=invert)


async def _report_errors(parent: Model, coro):
    try:
        return await coro
    except NetworkError as e:
        parent.report_error(f"{util.log_error(e)} [url={e.url}, code={e.code}]")
    except Exception as e:
        parent.report_error(util.log_error(e))


def _save_job_result(model: Model, job: Job | None, index: int):
    assert job is not None, "Cannot save result, invalid job id"
    assert len(job.results) > index, "Cannot save result, invalid result index"
    assert model.document.filename, "Cannot save result, document is not saved"
    timestamp = job.timestamp.strftime("%Y%m%d-%H%M%S")
    prompt = util.sanitize_prompt(job.params.prompt)
    path = Path(model.document.filename)
    path = path.parent / f"{path.stem}-generated-{timestamp}-{index}-{prompt}.png"
    path = util.find_unused_path(path)
    base_image = model._get_current_image(model.regions.root, Bounds(0, 0, *model.document.extent))
    result_image = job.results[index]
    base_image.draw_image(result_image, job.params.bounds.offset)
    base_image.save(path)
