from __future__ import annotations
import asyncio
from copy import copy
from pathlib import Path
from enum import Enum
from typing import Any, NamedTuple
from PyQt5.QtCore import QObject, QUuid, pyqtSignal
from PyQt5.QtGui import QImage, QPainter
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
from .image import Extent, Image, Mask, Bounds, DummyImage
from .client import ClientMessage, ClientEvent, filter_supported_styles, resolve_sd_version
from .document import Document, Layer, LayerType
from .pose import Pose
from .style import Style, Styles, SDVersion
from .connection import Connection
from .properties import Property, ObservableProperties
from .jobs import Job, JobKind, JobParams, JobQueue, JobState, JobRegion
from .control import ControlLayer, ControlLayerList
from .resources import ControlMode
from .resolution import compute_bounds, compute_relative_bounds
import krita


class Workspace(Enum):
    generation = 0
    upscaling = 1
    live = 2
    animation = 3


class RegionLink(Enum):
    direct = 0  # layer is directly linked to a region
    indirect = 1  # layer is in a group which is linked to a region
    any = 3  # either direct or indirect link


class Region(QObject, ObservableProperties):
    _parent: "RootRegion"
    _layers: list[QUuid]

    layer_ids = Property("", persist=True, setter="_set_layer_ids")
    positive = Property("", persist=True)
    control: ControlLayerList

    layer_ids_changed = pyqtSignal(str)
    positive_changed = pyqtSignal(str)
    modified = pyqtSignal(QObject, str)  # TODO: hook this up

    def __init__(self, parent: "RootRegion", model: "Model"):
        super().__init__()
        self._parent = parent
        self._layers = []
        self.control = ControlLayerList(model)

    def _get_layers(self):
        col = self._parent._model.layers.updated()
        all = (col.find(id) for id in self._layers)
        pruned = [l for l in all if l is not None]
        self._set_layers([l.id for l in pruned])
        return pruned

    def _set_layers(self, ids: list[QUuid]):
        self._layers = ids
        new_ids_string = ",".join(id.toString() for id in ids)
        if self.layer_ids != new_ids_string:
            self._layer_ids = new_ids_string
            self.layer_ids_changed.emit(self._layer_ids)

    def _set_layer_ids(self, ids: str):
        if self._layer_ids == ids:
            return
        self._layer_ids = ids
        self._layers = [QUuid(id) for id in ids.split(",") if id]
        self.layer_ids_changed.emit(ids)

    @property
    def layers(self):
        return self._get_layers()

    @property
    def first_layer(self):
        layers = self.layers
        return layers[0] if len(layers) > 0 else None

    @property
    def name(self):
        return ", ".join(l.name for l in self.layers)

    def link(self, layer: Layer):
        if layer.id not in self._layers:
            self._set_layers(self._layers + [layer.id])

    def unlink(self, layer: Layer):
        if layer.id in self._layers:
            self._set_layers([l for l in self._layers if l != layer.id])

    def is_linked(self, layer: Layer, mode=RegionLink.any):
        target = layer
        if mode is not RegionLink.direct:
            target = Region.link_target(layer)
        if mode is RegionLink.indirect and target is layer:
            return False
        if mode is RegionLink.direct or target is layer:
            return layer.id in self._layers
        return self.root.find_linked(target) is self

    def link_active(self):
        self.link(self._parent.layers.active)

    def unlink_active(self):
        self.unlink(self._parent.layers.active)

    def toggle_active_link(self):
        if self.is_active_linked:
            self.unlink_active()
        else:
            self.link_active()

    @property
    def has_links(self):
        return len(self._layers) > 0

    @property
    def is_active_linked(self):
        return self.is_linked(self._parent.layers.active)

    def remove(self):
        self._parent.remove(self)

    @property
    def root(self):
        return self._parent

    @property
    def siblings(self):
        return self._parent.find_siblings(self)

    @staticmethod
    def link_target(layer: Layer):
        if layer.type is LayerType.group:
            return layer
        if parent := layer.parent_layer:
            if not parent.is_root and parent.type is LayerType.group:
                return parent
        return layer


class RootRegion(QObject, ObservableProperties):
    _model: Model
    _regions: list[Region]
    _active: Region | None = None
    _active_layer: QUuid | None = None

    positive = Property("", persist=True)
    negative = Property("", persist=True)
    control: ControlLayerList

    positive_changed = pyqtSignal(str)
    negative_changed = pyqtSignal(str)
    active_changed = pyqtSignal(Region)
    active_layer_changed = pyqtSignal()
    added = pyqtSignal(Region)
    removed = pyqtSignal(Region)

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        self._regions = []
        self.control = ControlLayerList(model)
        model.layers.active_changed.connect(self._update_active)

    def _find_region(self, layer: Layer):
        return next((r for r in self._regions if r.is_linked(layer, RegionLink.direct)), None)

    def emplace(self):
        region = Region(self, self._model)
        self._regions.append(region)
        return region

    @property
    def active(self):
        self._update_active()
        return self._active

    @active.setter
    def active(self, region: Region | None):
        if self._active != region:
            self._active = region
            self.active_changed.emit(region)

    @property
    def active_or_root(self):
        return self.active or self

    def add_control(self):
        self.active_or_root.control.add()

    def is_linked(self, layer: Layer, mode=RegionLink.any):
        return any(r.is_linked(layer, mode) for r in self._regions)

    def find_linked(self, layer: Layer, mode=RegionLink.any):
        return next((r for r in self._regions if r.is_linked(layer, mode)), None)

    def create_region_layer(self):
        self.create_region(group=False)

    def create_region_group(self):
        self.create_region(group=True)

    def create_region(self, group=True):
        """Create a new region. This action depends on context:
        If the active layer can be linked to a group and isn't the only layer in the document,
        it will be used as the initial link target for the new group. Otherwise, a new layer
        is inserted (or a group if group==True) and that will be linked instead.
        """
        layers = self._model.layers
        target = Region.link_target(layers.active)
        can_link = target.type in [LayerType.paint, LayerType.group] and not self.is_linked(target)
        if can_link and len(layers.images) > 1:
            layer = target
        elif group:
            layer = layers.create_group(f"Region {len(self)}")
            layers.create(f"Paint layer", parent=layer)
        else:
            layer = layers.create(f"Region {len(self)}")
        return self._add(layer)

    def remove(self, region: Region):
        if region in self._regions:
            if self.active == region:
                self.active = None
            self._regions.remove(region)
            self.removed.emit(region)

    def to_api(self, bounds: Bounds, parent_layer: Layer | None = None):
        parent_region = None
        if parent_layer is not None:
            parent_region = self.find_linked(parent_layer)

        parent_prompt = ""
        job_info = []
        if parent_layer and parent_region:
            parent_prompt = parent_region.positive
            job_info = [JobRegion(parent_layer.id_string, parent_prompt)]
        result = ConditioningInput(
            positive=workflow.merge_prompt(parent_prompt, self.positive),
            negative=self.negative,
            control=[c.to_api(bounds) for c in self.control],
        )

        # Check for regions linked to any child layers of the parent layer.
        parent_layer = parent_layer or self.layers.root
        child_layers = parent_layer.child_layers
        layer_regions = ((l, self.find_linked(l, RegionLink.direct)) for l in child_layers)
        layer_regions = [(l, r) for l, r in layer_regions if r is not None]
        if len(layer_regions) == 0:
            return result, job_info

        # Get region masks. Filter out regions with:
        # * no content (empty mask)
        # * less than 10% overlap (esimate based on bounding box)
        result_regions: list[tuple[RegionInput, JobRegion]] = []
        for layer, region in layer_regions:
            layer_bounds = layer.compute_bounds()
            if layer_bounds.area == 0:
                print(f"Skipping empty region {layer.name}")
                continue

            overlap_rough = Bounds.intersection(bounds, layer_bounds).area / bounds.area
            if overlap_rough < 0.1:
                print(f"Skipping region {region.positive[:10]}: overlap is {overlap_rough}")
                continue

            region_result = RegionInput(
                layer.get_mask(bounds),
                workflow.merge_prompt(region.positive, self.positive),
                control=[c.to_api(bounds) for c in region.control],
            )
            result_regions.append((region_result, JobRegion(layer.id_string, region.positive)))

        # Remove from each region mask any overlapping areas from regions above it.
        accumulated_mask = None
        for i in range(len(result_regions) - 1, -1, -1):
            region, job_region = result_regions[i]
            mask = region.mask
            if accumulated_mask is None:
                accumulated_mask = Image.copy(region.mask)
            else:
                mask = Image.mask_subtract(mask, accumulated_mask)

            coverage = mask.average()
            if coverage > 0.9:
                # Single region covers (almost) entire image, don't use regional conditioning.
                print(f"Using single region {region.positive[:10]}: coverage is {coverage}")
                result.control += region.control
                return result, [job_region]
            elif coverage < 0.1:
                # Region has less than 10% coverage, remove it.
                print(f"Skipping region {region.positive[:10]}: coverage is {coverage}")
                result_regions.pop(i)
            else:
                # Accumulate mask for next region, and store modified mask.
                accumulated_mask = Image.mask_add(accumulated_mask, region.mask)
                region.mask = mask

        # If there are no regions left, don't use regional conditioning.
        if len(result_regions) == 0:
            return result, job_info

        # If the region(s) don't cover the entire image, add a final region for the remaining area.
        assert accumulated_mask is not None, "Expecting at least one region mask"
        total_coverage = accumulated_mask.average()
        if total_coverage < 1:
            print(f"Adding background region: total coverage is {total_coverage}")
            accumulated_mask.invert()
            input = RegionInput(accumulated_mask, result.positive)
            job = JobRegion(parent_layer.id_string, "background", is_background=True)
            result_regions.append((input, job))

        result.regions = [r for r, _ in result_regions]
        return result, [j for _, j in result_regions]

    def _get_regions(self, layers: list[Layer], exclude: Region | None = None):
        regions = []
        for l in layers:
            r = self._find_region(l)
            if r is not None and r is not exclude and not r in regions:
                regions.append(r)
        return regions

    def find_siblings(self, region: Region):
        if layer := region.first_layer:
            below, above = layer.siblings
            return self._get_regions(below, region), self._get_regions(above, region)
        return [], []

    @property
    def siblings(self):
        if self.layers:
            layer = self.layers.root
            if active_layer := self._get_active_layer()[0]:
                active_layer = Region.link_target(active_layer)
                if self.is_linked(active_layer):
                    layer = active_layer.parent_layer or active_layer
            return [], self._get_regions(layer.child_layers)
        return [], []

    def _get_active_layer(self):
        if not self.layers:
            return None, False
        layer = self.layers.active
        if layer.id == self._active_layer:
            return layer, False
        self._active_layer = layer.id
        self.active_layer_changed.emit()
        return layer, True

    def _update_active(self):
        layer, changed = self._get_active_layer()
        if layer and changed:
            if region := self.find_linked(layer):
                self.active = region
            elif self._model.workspace is Workspace.live:
                self.active = None  # root region

    def _add(self, layer: Layer):
        region = Region(self, self._model)
        region.link(layer)
        self._regions.append(region)
        self.added.emit(region)
        self.active = region
        return region

    @property
    def layers(self):
        return self._model.layers

    def __len__(self):
        return len(self._regions)

    def __iter__(self):
        return iter(self._regions)


class Model(QObject, ObservableProperties):
    """Represents diffusion workflows for a specific Krita document. Stores all inputs related to
    image generation. Launches generation jobs. Listens to server messages and keeps a
    list of finished, currently running and enqueued jobs.
    """

    _doc: Document
    _connection: Connection
    _layer: Layer | None = None

    workspace = Property(Workspace.generation, setter="set_workspace", persist=True)
    regions: "RootRegion"
    style = Property(Styles.list().default, setter="set_style", persist=True)
    strength = Property(1.0, persist=True)
    region_only = Property(False, persist=True)
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
    region_only_changed = pyqtSignal(bool)
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
        self._connection = connection
        self.generate_seed()
        self.jobs = JobQueue()
        self.regions = RootRegion(self)
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

        try:
            input, job_params = self._prepare_workflow()
        except Exception as e:
            self.report_error(util.log_error(e))
            return
        self.clear_error()
        jobs = self.enqueue_jobs(input, JobKind.diffusion, job_params, self.batch_count)
        eventloop.run(_report_errors(self, jobs))

    def _prepare_workflow(self, with_images=True):
        workflow_kind = WorkflowKind.generate if self.strength == 1.0 else WorkflowKind.refine
        client = self._connection.client
        image = None
        inpaint_mode = InpaintMode.fill
        inpaint = None
        region_layer = None
        extent = self._doc.extent
        mask = self._doc.create_mask_from_selection(
            **get_selection_modifiers(self.inpaint.mode, self.strength), min_size=64
        )
        bounds = Bounds(0, 0, *extent)
        if mask is None:
            # Check for region inpaint
            target = Region.link_target(self.layers.active)
            if self.regions.is_linked(target):
                region_layer = target
            if region_layer:
                inpaint_mode = InpaintMode.add_object
                if not (self.region_only or region_layer.parent_layer is None):
                    region_layer = region_layer.parent_layer
                if not region_layer.is_root:
                    bounds = region_layer.compute_bounds()
                    bounds = Bounds.pad(bounds, settings.selection_padding, multiple=64)
                    bounds = Bounds.clamp(bounds, extent)
                    mask_img = region_layer.get_mask(bounds)
                    mask = Mask(bounds, mask_img._qimage)
        else:
            # Selection inpaint
            bounds = compute_bounds(extent, mask.bounds if mask else None, self.strength)
            bounds = self.inpaint.get_context(self, mask) or bounds
            inpaint_mode = self.resolve_inpaint_mode()

        conditioning, job_regions = self.regions.to_api(bounds, region_layer)

        if mask is not None or self.strength < 1.0:
            image = self._get_current_image(bounds) if with_images else DummyImage(extent)

        if mask is not None:
            if workflow_kind is WorkflowKind.generate:
                workflow_kind = WorkflowKind.inpaint
            elif workflow_kind is WorkflowKind.refine:
                workflow_kind = WorkflowKind.refine_region

            bounds, mask.bounds = compute_relative_bounds(bounds, mask.bounds)

            sd_version = client.models.version_of(self.style.sd_checkpoint)
            if inpaint_mode is InpaintMode.custom:
                inpaint = self.inpaint.get_params(mask)
            else:
                inpaint = workflow.detect_inpaint(
                    inpaint_mode,
                    mask.bounds,
                    sd_version,
                    conditioning.positive,
                    conditioning.control,
                    self.strength,
                )

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
        job_params = JobParams(bounds, conditioning.positive, regions=job_regions)
        return input, job_params

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

    def _prepare_upscale_image(self, with_images=True):
        extent = self._doc.extent
        image = self._doc.get_image(Bounds(0, 0, *extent)) if with_images else DummyImage(extent)
        params = self.upscale.params
        client = self._connection.client
        upscaler = params.upscaler or client.models.default_upscaler

        if params.use_diffusion:
            return workflow.prepare(
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
            return workflow.prepare_upscale_simple(image, upscaler, params.factor)

    def upscale_image(self):
        try:
            inputs = self._prepare_upscale_image()
            seed = inputs.sampling.seed if inputs.sampling else 0
            job = self.jobs.add_upscale(Bounds(0, 0, *self.upscale.target_extent), seed)
        except Exception as e:
            self.report_error(util.log_error(e))
            return

        self.clear_error()
        eventloop.run(_report_errors(self, self._enqueue_job(job, inputs)))

    def estimate_cost(self, kind=JobKind.diffusion):
        try:
            if kind is JobKind.diffusion:
                input, _ = self._prepare_workflow(with_images=False)
            elif kind is JobKind.upscaling:
                input = self._prepare_upscale_image(with_images=False)
            else:
                return 0
            return input.cost
        except Exception as e:
            util.client_logger.warning(f"Failed to estimate workflow cost: {type(e)} {str(e)}")
            return 0

    def generate_live(self):
        eventloop.run(_report_errors(self, self._generate_live()))

    async def _generate_live(self, last_input: WorkflowInput | None = None):
        strength = self.live.strength
        workflow_kind = WorkflowKind.generate if strength == 1.0 else WorkflowKind.refine
        client = self._connection.client
        ver = client.models.version_of(self.style.sd_checkpoint)
        extent = self._doc.extent
        region = None
        region_layer = self.layers.active
        job_regions: list[JobRegion] = []

        image = None
        mask = self._doc.create_mask_from_selection(
            grow=settings.selection_feather / 200,  # don't apply grow for live mode
            feather=settings.selection_feather / 100,
            padding=settings.selection_padding / 100,
            min_size=512 if ver is SDVersion.sd15 else 1024,
            square=True,
        )
        if mask is None:
            region = self.regions.find_linked(region_layer)
            if region is not None:
                bounds = region_layer.compute_bounds()
                bounds = Bounds.pad(
                    bounds, settings.selection_padding, multiple=64, min_size=512, square=True
                )
                bounds = Bounds.clamp(bounds, extent)
                mask_image = region_layer.get_mask(bounds)
                mask = Mask(bounds, mask_image._qimage)
                job_regions = [JobRegion(region_layer.id_string, region.positive)]

        bounds = Bounds(0, 0, *self._doc.extent)
        if mask is not None:
            workflow_kind = WorkflowKind.refine_region
            bounds, mask.bounds = compute_relative_bounds(mask.bounds, mask.bounds)
        if mask is not None or self.live.strength < 1.0:
            image = self._get_current_image(bounds)

        positive = workflow.merge_prompt("", self.regions.positive)
        control = [c.to_api(bounds) for c in self.regions.control]
        if region is not None:
            positive = workflow.merge_prompt(region.positive, self.regions.positive)
            control += [c.to_api(bounds) for c in region.control]

        input = workflow.prepare(
            workflow_kind,
            image or bounds.extent,
            ConditioningInput(positive, self.regions.negative, control=control),
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
            params = JobParams(bounds, positive, regions=job_regions)
            await self.enqueue_jobs(input, JobKind.live_preview, params)
            return input

        return None

    def _get_current_image(self, bounds: Bounds):
        exclude = [  # exclude control layers from projection
            c.layer for c in self.regions.control if not c.mode.is_part_of_image
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
                job.control.layer_id = self.add_control_layer(job, message.result).id
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
        if self._layer and self._layer.was_removed:
            self._layer = None  # layer was removed by user
        if self._layer is not None:
            self._layer.name = name
            self._layer.write_pixels(job.results[index], job.params.bounds)
            self._layer.move_to_top()
        else:
            self._layer = self.layers.create(
                name, job.results[index], job.params.bounds, make_active=False
            )
            self._layer.is_locked = True

    def hide_preview(self):
        if self._layer is not None:
            self._layer.hide()

    def write_result(self, image: Image, params: JobParams):
        """Write the generated image to the document, replace original content of the layer."""
        if len(params.regions) == 0:
            self.layers.active.write_pixels(image, params.bounds)
        else:
            for job_region in params.regions:
                if region_layer := self.layers.find(QUuid(job_region.layer_id)):
                    if region_layer.type is LayerType.group:
                        self.create_result_layer(image, params)
                    else:
                        region_layer.write_pixels(image, params.bounds)

    def create_result_layer(self, image: Image, params: JobParams, prefix=""):
        """Insert generated image as a new layer in the document (non-destructive apply)."""
        name = f"{prefix}{params.prompt} ({params.seed})"
        if len(params.regions) == 0:
            self.layers.create(name, image, params.bounds)
        else:
            for job_region in params.regions:
                region_layer = self.layers.find(QUuid(job_region.layer_id)) or self.layers.root

                # Promote layer to group if needed
                if region_layer.type is not LayerType.group:
                    paint_layer = region_layer
                    region_layer = self.layers.create_group_for(paint_layer)
                    if region := self.regions.find_linked(paint_layer, RegionLink.direct):
                        region.unlink(paint_layer)
                        region.link(region_layer)

                # Create transparency mask to correctly blend the generated image
                has_layers = len(region_layer.child_layers) > 0
                has_mask = any(l.type.is_mask for l in region_layer.child_layers)
                if has_layers and not has_mask:
                    mask = region_layer.get_mask(params.bounds)
                    self.layers.create_mask("Transparency Mask", mask, params.bounds, region_layer)

                # Handle auto-generated background region (not linked to any layers)
                layer_above = None
                if job_region.is_background:
                    for node in region_layer.child_layers:
                        if node.type is LayerType.group:
                            break
                        layer_above = node

                self.layers.create(name, image, params.bounds, above=layer_above)

    def apply_result(self, job_id: str, index: int):
        job = self.jobs.find(job_id)
        assert job is not None, "Cannot apply result, invalid job id"

        self.create_result_layer(job.results[index], job.params, "[Generated] ")

        if self._layer:
            self._layer.remove()
            self._layer = None
        self.jobs.selection = None
        self.jobs.notify_used(job_id, index)

    def add_control_layer(self, job: Job, result: dict | None):
        assert job.kind is JobKind.control_layer and job.control
        if job.control.mode is ControlMode.pose and result is not None:
            pose = Pose.from_open_pose_json(result)
            pose.scale(job.params.bounds.extent)
            return self.layers.create_vector(job.params.prompt, pose.to_svg())
        elif len(job.results) > 0:
            return self.layers.create(job.params.prompt, job.results[0], job.params.bounds)
        return self.layers.active  # Execution was cached and no image was produced

    def add_upscale_layer(self, job: Job):
        assert job.kind is JobKind.upscaling
        assert len(job.results) > 0, "Upscaling job did not produce an image"
        if self._layer:
            self._layer.remove()
            self._layer = None
        self._doc.resize(job.params.bounds.extent)
        self.upscale.target_extent_changed.emit(self.upscale.target_extent)
        self.layers.create(job.params.prompt, job.results[0], job.params.bounds)

    def set_workspace(self, workspace: Workspace):
        if self.workspace is Workspace.live:
            self.live.is_active = False
        self._workspace = workspace
        self.workspace_changed.emit(workspace)
        self.modified.emit(self, "workspace")

    def set_style(self, style: Style):
        if style is not self._style:
            if client := self._connection.client_if_connected:
                styles = filter_supported_styles(Styles.list().filtered(), client)
                if style not in styles:
                    return
            self._style = style
            self.style_changed.emit(style)
            self.modified.emit(self, "style")

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
        return self._doc.layers


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
                layer_bounds = layer.compute_bounds()
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
    _result_composition: Image | None = None
    _result_params: JobParams | None = None
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
                self.set_result(job.results[0], job.params)
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

    def write_result(self):
        assert self.result is not None and self._result_params is not None
        self._model.write_result(self.result, self._result_params)

    def create_result_layer(self):
        assert self.result is not None and self._result_params is not None
        self._model.create_result_layer(self.result, self._result_params)

        if settings.new_seed_after_apply:
            self._model.generate_seed()

    @property
    def result(self):
        return self._result

    @property
    def result_composition(self):
        return self._result_composition

    def set_result(self, value: Image, params: JobParams):
        canvas = self._model._get_current_image(params.bounds)
        painter = QPainter(canvas._qimage)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawImage(0, 0, value._qimage)
        painter.end()
        self._result = value
        self._result_composition = canvas
        self._result_params = params
        self.result_available.emit(canvas)
        self.has_result = True

        if self.is_recording:
            self._save_frame(value, params.bounds)

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
        prompt = self._model.regions.active_or_root.positive
        self._model.layers.active.name = f"[Rec] {start}-{end}: {prompt}"
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
        conditioning, job_regions = m.regions.to_api(bounds)
        return workflow.prepare(
            WorkflowKind.generate if m.strength == 1.0 else WorkflowKind.refine,
            canvas,
            conditioning,
            style=m.style,
            seed=seed,
            perf=m._connection.client.performance_settings,
            models=m._connection.client.models,
            strength=m.strength,
            is_live=self.sampling_quality is SamplingQuality.fast,
        )

    async def _generate_frame(self):
        m = self._model
        bounds = Bounds(0, 0, *m.document.extent)
        canvas = m._get_current_image(bounds) if m.strength < 1.0 else bounds.extent
        seed = m.seed if m.fixed_seed else workflow.generate_seed()
        inputs = self._prepare_input(canvas, seed)
        params = JobParams(bounds, m.regions.positive, frame=(m.document.current_time, 0, 0))
        await m.enqueue_jobs(inputs, JobKind.animation_frame, params)

    def generate_batch(self):
        doc = self._model.document
        if self._model.strength < 1.0 and not self._model.layers.active.is_animated:
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
        layer = self._model.layers.active
        start_frame, end_frame = doc.playback_time_range
        extent = doc.extent
        bounds = Bounds(0, 0, *extent)
        strength = self._model.strength
        seed = self._model.seed if self._model.fixed_seed else workflow.generate_seed()
        animation_id = str(uuid.uuid4())

        for frame in range(start_frame, end_frame + 1):
            if layer.node.hasKeyframeAtTime(frame) or strength == 1.0:
                canvas: Image | Extent = extent
                if strength < 1.0:
                    canvas = layer.get_pixels(time=frame)

                inputs = self._prepare_input(canvas, seed)
                params = JobParams(bounds, self._model.regions.active_or_root.positive)
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
                    layer.write_pixels(image, job.params.bounds, make_visible=False)
                    self.target_image_changed.emit(image)
                else:
                    self._model.report_error("Target layer not found")

    def _import_animation(self, job: Job):
        doc = self._model.document
        keyframes = self._keyframes.pop(job.params.animation_id)
        _, start, end = job.params.frame
        doc.import_animation(keyframes, start)
        doc.layers.active.name = f"[Generated] {start}-{end}: {job.params.prompt}"
        self.target_layer = doc.layers.active.id

    def _update_target_image(self):
        if self.batch_mode:
            return
        if layer := self._model.layers.find(self.target_layer):
            bounds = Bounds(0, 0, *self._model.document.extent)
            image = layer.get_pixels(bounds)
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
    base_image = model._get_current_image(Bounds(0, 0, *model.document.extent))
    result_image = job.results[index]
    base_image.draw_image(result_image, job.params.bounds.offset)
    base_image.save(path)
