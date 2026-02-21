from __future__ import annotations

import asyncio
import time
import uuid
import weakref
from collections import deque
from copy import copy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NamedTuple

from PyQt5.QtCore import QMetaObject, QObject, Qt, QUuid, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter

from . import eventloop, util, workflow
from .api import (
    ConditioningInput,
    ControlInput,
    CustomWorkflowInput,
    FillMode,
    ImageInput,
    InpaintContext,
    InpaintMode,
    InpaintParams,
    SamplingInput,
    UpscaleInput,
    WorkflowInput,
    WorkflowKind,
)
from .client import (
    Client,
    ClientEvent,
    ClientMessage,
    ClientOutput,
    filter_supported_styles,
    is_style_supported,
    resolve_arch,
)
from .connection import Connection, ConnectionState
from .control import ControlLayer
from .custom_workflow import CustomGenerationMode, CustomWorkspace, WorkflowCollection
from .document import Document, KritaDocument, SelectionModifiers
from .files import FileLibrary
from .image import Bounds, DummyImage, Extent, Image, Mask
from .jobs import Job, JobKind, JobParams, JobQueue, JobRegion, JobState
from .layer import Layer, LayerType, RestoreActiveLayer
from .localization import translate as _
from .network import NetworkError
from .pose import Pose
from .properties import ObservableProperties, Property
from .region import Region, RegionLink, RootRegion, get_region_inpaint_mask, process_regions
from .resolution import compute_bounds, compute_relative_bounds
from .resources import ControlMode
from .settings import (
    ApplyBehavior,
    ApplyRegionBehavior,
    GenerationFinishedAction,
    ImageFileFormat,
    settings,
)
from .style import Arch, Style, Styles
from .text import create_img_metadata, extract_layers
from .util import PluginError, clamp, ensure, trim_text, unique
from .util import client_logger as log


class QueueMode(Enum):
    back = 0
    front = 1
    replace = 2


class Workspace(Enum):
    generation = 0
    upscaling = 1
    live = 2
    animation = 3
    custom = 4


class ProgressKind(Enum):
    generation = 0
    upload = 1


class ErrorKind(Enum):
    none = 0
    plugin_error = 100
    server_error = 200
    insufficient_funds = 201
    warning = 300
    incompatible_lora = 301
    validation_warning = 302

    @property
    def is_warning(self):
        return self.value >= ErrorKind.warning.value


class Error(NamedTuple):
    kind: ErrorKind
    message: str
    data: dict[str, Any] | None = None

    def __bool__(self):
        return self.kind is not ErrorKind.none

    @staticmethod
    def from_string(s: str, fallback: ErrorKind | None = None):
        kind = ErrorKind[s] if s in ErrorKind.__members__ else fallback or ErrorKind.warning
        return Error(kind, s)


no_error = Error(ErrorKind.none, "")


class Model(QObject, ObservableProperties):
    """Represents diffusion workflows for a specific Krita document. Stores all inputs related to
    image generation. Launches generation jobs. Listens to server messages and keeps a
    list of finished, currently running and enqueued jobs.
    """

    workspace = Property(Workspace.generation, setter="set_workspace", persist=True)
    style = Property(Styles.list().default, setter="set_style", persist=True)
    strength = Property(1.0, persist=True)
    region_only = Property(False, persist=True)
    edit_mode = Property(False, persist=True)
    batch_count = Property(1, persist=True)
    seed = Property(0, persist=True)
    fixed_seed = Property(False, persist=True)
    resolution_multiplier = Property(1.0, persist=True)
    queue_mode = Property(QueueMode.back, persist=True)
    translation_enabled = Property(True, persist=True)
    layer_count = Property(4, persist=True)
    progress_kind = Property(ProgressKind.generation)
    progress = Property(0.0)
    error = Property(no_error)

    workspace_changed = pyqtSignal(Workspace)
    style_changed = pyqtSignal(Style)
    strength_changed = pyqtSignal(float)
    region_only_changed = pyqtSignal(bool)
    edit_mode_changed = pyqtSignal(bool)
    batch_count_changed = pyqtSignal(int)
    seed_changed = pyqtSignal(int)
    fixed_seed_changed = pyqtSignal(bool)
    resolution_multiplier_changed = pyqtSignal(float)
    queue_mode_changed = pyqtSignal(QueueMode)
    translation_enabled_changed = pyqtSignal(bool)
    layer_count_changed = pyqtSignal(int)
    progress_kind_changed = pyqtSignal(ProgressKind)
    progress_changed = pyqtSignal(float)
    error_changed = pyqtSignal(Error)
    modified = pyqtSignal(QObject, str)

    def __init__(self, document: Document, connection: Connection, workflows: WorkflowCollection):
        super().__init__()
        self._doc = document
        self._connection = connection
        self._layer: Layer | None = None
        self.generate_seed()
        self.jobs = JobQueue()
        self.regions = RootRegion(self)
        self.edit_regions = RootRegion(self)
        self.inpaint = CustomInpaint()
        self.upscale = UpscaleWorkspace(self)
        self.live = LiveWorkspace(self)
        self.animation = AnimationWorkspace(self)
        self.custom = CustomWorkspace(workflows, self._generate_custom, self.jobs)
        self._style_connection: QMetaObject.Connection | None = None

        self.jobs.selection_changed.connect(self.update_preview)
        connection.state_changed.connect(self._init_on_connect)
        connection.error_changed.connect(self._forward_error)
        self.custom.validation_error_changed.connect(self._forward_validation_error)
        Styles.list().changed.connect(self._init_on_connect)
        self._init_on_connect()

    def _init_on_connect(self):
        if self._connection.state is not ConnectionState.connected:
            return
        if client := self._connection.client_if_connected:
            styles = filter_supported_styles(Styles.list().filtered(), client)
            if self.style not in styles and len(styles) > 0:
                self.style = styles[0]
            if self.upscale.upscaler == "":
                self.upscale.upscaler = client.models.default_upscaler

    def _forward_error(self, error: str):
        self.report_error(error if error else no_error)

    def _forward_validation_error(self, error: str):
        if error:
            self.report_error(Error(ErrorKind.validation_warning, error))
        elif self.error.kind is ErrorKind.validation_warning:
            self.clear_error()

    def generate(self):
        """Enqueue image generation for the current setup."""
        self._generate(self.queue_mode)

    def generate_replace(self):
        """Enqueue image generation with queue mode set to replace."""
        self._generate(QueueMode.replace)

    def _generate(self, queue_mode: QueueMode):
        """Enqueue image generation for the current setup."""
        ok, msg = self._doc.check_color_mode()
        if not ok and msg:
            self.report_error(msg)
            return

        try:
            input, job_params, cond_orig = self._prepare_workflow()
        except Exception as e:
            self.report_error(util.log_error(e))
            return
        self.clear_error()
        jobs = self.enqueue_jobs(
            input, JobKind.diffusion, job_params, cond_orig, self.batch_count, queue_mode
        )
        eventloop.run(_report_errors(self, jobs))

    def _prepare_workflow(self, dryrun=False):
        arch = self.arch
        workflow_kind = WorkflowKind.generate
        strength = self.strength
        if arch is Arch.qwen_l:
            strength = 1.0
        if strength < 1.0 or self.is_editing:
            workflow_kind = WorkflowKind.refine
        client = self._connection.client
        image = None
        inpaint_mode: InpaintMode | None = None
        inpaint = None
        extent = self._doc.extent
        regions = self.active_regions
        region_layer = None

        smod = get_selection_modifiers(arch, self.inpaint.mode, strength)
        mask, selection_bounds = self._doc.create_mask_from_selection(smod)
        bounds = Bounds(0, 0, *extent)
        if mask is None:  # Check for region inpaint
            inpaint_mode = InpaintMode.fill
            region_layer = regions.get_active_region_layer(use_parent=not self.region_only)
            if not region_layer.is_root:
                mask = get_region_inpaint_mask(region_layer, extent)
                bounds = mask.bounds
                inpaint_mode = InpaintMode.add_object
        else:  # Selection inpaint
            bounds = compute_bounds(extent, mask.bounds if mask else None, workflow_kind)
            bounds = self.inpaint.get_context(self, mask) or bounds
            inpaint_mode = self.resolve_inpaint_mode()

        if not dryrun:
            conditioning, job_regions = process_regions(regions, bounds, region_layer)
            conditioning.language = self.prompt_translation_language
        else:
            conditioning, job_regions = ConditioningInput("", ""), []

        seed = self.seed if self.fixed_seed else workflow.generate_seed()
        if not dryrun:
            conditioning = self._add_reference_layers(conditioning)
        original_conditioning = conditioning
        conditioning, loras, prompt_meta = workflow.prepare_prompts(
            conditioning, self.style, seed, arch, inpaint_mode
        )

        if mask is not None or workflow_kind is WorkflowKind.refine:
            image = self._get_current_image(bounds) if not dryrun else DummyImage(bounds.extent)

        if mask is not None:
            if workflow_kind is WorkflowKind.generate:
                workflow_kind = WorkflowKind.inpaint
            elif workflow_kind is WorkflowKind.refine:
                workflow_kind = WorkflowKind.refine_region

            bounds, mask.bounds = compute_relative_bounds(bounds, mask.bounds)

            if inpaint_mode is InpaintMode.custom:
                inpaint = self.inpaint.get_params(mask, self.is_editing)
            else:
                inpaint = workflow.detect_inpaint(
                    inpaint_mode, mask.bounds, arch, conditioning, strength
                )
            inpaint = calc_selection_pre_process(inpaint, selection_bounds, smod)

        input = workflow.prepare(
            workflow_kind,
            image or extent,
            conditioning,
            self.active_style,
            seed,
            client.models,
            FileLibrary.instance(),
            self._performance_settings(client),
            mask=mask,
            strength=strength,
            loras=loras,
            inpaint=inpaint,
            layer_count=self.layer_count,
        )
        loras = input.models.loras if input.models else []
        job_name = prompt_meta.get("prompt_eval", prompt_meta["prompt"])
        job_params = JobParams(bounds, job_name, regions=job_regions)
        job_params.set_style(self.active_style, ensure(input.models).checkpoint)
        job_params.set_control(regions.control)
        job_params.inpaint_mode = inpaint_mode
        job_params.is_layered = arch is Arch.qwen_l
        job_params.metadata.update(prompt_meta)
        job_params.metadata["loras"] = [{"name": l.name, "weight": l.strength} for l in loras]
        job_params.metadata["strength"] = strength
        return input, job_params, original_conditioning

    async def enqueue_jobs(
        self,
        input: WorkflowInput,
        kind: JobKind,
        params: JobParams,
        original_cond: ConditioningInput | None = None,
        count: int = 1,
        queue_mode: QueueMode | None = None,
    ):
        sampling = ensure(input.sampling)
        params.has_mask = input.images is not None and input.images.hires_mask is not None
        queue_mode = queue_mode or self.queue_mode

        if queue_mode is QueueMode.replace:
            if to_cancel := self.clear_queued():
                await self._connection.client.cancel(to_cancel)
            queue_mode = QueueMode.back

        for i in range(count):
            seed = sampling.seed + i * settings.batch_size
            params.seed = seed
            if i > 0:
                input = replace(input, sampling=replace(sampling, seed=seed))
                if original_cond:  # re-evaluate wildcards in prompts after the seed change
                    next_prompt = workflow.prepare_prompts(
                        original_cond, self.style, seed, self.arch, params.inpaint_mode
                    )
                    input.conditioning = next_prompt.conditioning
                    params.metadata = params.metadata | next_prompt.metadata
                    params.name = params.metadata.get("prompt_eval", params.name)
            job = self.jobs.add(kind, copy(params))
            front = queue_mode is QueueMode.front
            await self._enqueue_job(job, input, front=front)

    async def _enqueue_job(self, job: Job, input: WorkflowInput, front: bool = False):
        if not self.jobs.any_executing():
            self.progress = 0.0
        client = self._connection.client
        job.id = await client.enqueue(input, front)

    def _prepare_upscale_image(self, dryrun=False):
        client = self._connection.client
        extent = self._doc.extent
        image = self._doc.get_image(Bounds(0, 0, *extent)) if not dryrun else DummyImage(extent)
        params = self.upscale.params
        params.upscale.model = params.upscale.model or client.models.default_upscaler
        if params.upscale.model not in client.models.upscalers:
            msg = _("The upscale model used by the document is not available on the server")
            self.report_error(Error(ErrorKind.warning, msg + f": {params.upscale.model}"))
            self.upscale.upscaler = params.upscale.model = client.models.default_upscaler
        bounds = Bounds(0, 0, *self._doc.extent)
        sys_prompt = "4k uhd"
        if self.arch.is_edit:
            sys_prompt = "Enhance image quality. Preserve original content."

        if params.use_prompt and not dryrun:
            conditioning, job_regions = process_regions(self.regions, bounds, min_coverage=0)
            conditioning.language = self.prompt_translation_language
            for region in job_regions:
                region.bounds = Bounds.scale(region.bounds, params.factor)
        else:
            conditioning, job_regions = ConditioningInput(sys_prompt), []
        models = client.models.for_arch(self.arch)
        has_unblur = models.find_control(ControlMode.blur) is not None
        if has_unblur and params.unblur_strength > 0.0:
            control = ControlInput(ControlMode.blur, None, params.unblur_strength)
            conditioning.control.append(control)

        if params.use_diffusion:
            input = workflow.prepare(
                WorkflowKind.upscale_tiled,
                image,
                conditioning,
                self.style,
                params.seed,
                client.models,
                FileLibrary.instance(),
                self._performance_settings(client),
                strength=params.strength,
                upscale_factor=params.factor,
                upscale=params.upscale,
            )
        else:
            input = workflow.prepare_upscale_simple(image, params.upscale.model, params.factor)

        target_bounds = Bounds(0, 0, *params.target_extent)
        name = f"{target_bounds.width}x{target_bounds.height}"
        job_params = JobParams(target_bounds, name, seed=params.seed, regions=job_regions)
        return input, job_params

    def upscale_image(self):
        try:
            self.clear_error()
            inputs, job_params = self._prepare_upscale_image()
            job = self.jobs.add(JobKind.upscaling, job_params)
        except Exception as e:
            self.report_error(util.log_error(e))
            return

        self.upscale.set_in_progress(True)
        eventloop.run(_report_errors(self, self._enqueue_job(job, inputs)))

        self._doc.resize(job.params.bounds.extent)
        self.upscale.target_extent_changed.emit(self.upscale.target_extent)

    def estimate_cost(self, kind=JobKind.diffusion):
        try:
            if kind is JobKind.diffusion:
                input, _job, _cond = self._prepare_workflow(dryrun=True)
            elif kind is JobKind.upscaling:
                input, _ = self._prepare_upscale_image(dryrun=True)
            else:
                return 0
            return input.cost * self.batch_count
        except Exception as e:
            util.client_logger.warning(f"Failed to estimate workflow cost: {type(e)} {e!s}")
            return 0

    def generate_live(self):
        input, job_params = self._prepare_live_workflow()
        eventloop.run(_report_errors(self, self._generate_live(input, job_params)))

    def _prepare_live_workflow(self):
        strength = self.live.strength
        workflow_kind = WorkflowKind.generate
        if strength < 1.0 or self.is_editing:
            workflow_kind = WorkflowKind.refine
        client = self._connection.client
        min_mask_size = 512 if self.arch is Arch.sd15 else 800
        extent = self._doc.extent
        region_layer = None
        job_regions: list[JobRegion] = []
        inpaint = InpaintParams(InpaintMode.fill, Bounds(0, 0, *extent))

        image = None
        smod = get_selection_modifiers(self.arch, inpaint.mode, strength, min_mask_size)
        mask, selection_bounds = self._doc.create_mask_from_selection(smod)
        inpaint = calc_selection_pre_process(inpaint, selection_bounds, smod)

        bounds = Bounds(0, 0, *self._doc.extent)
        region_layer = self.regions.get_active_region_layer(use_parent=False)
        if mask is None and region_layer.bounds != bounds:
            mask = get_region_inpaint_mask(region_layer, extent, min_size=min_mask_size)
            free_space = mask.bounds.extent - region_layer.compute_bounds().extent
            inpaint.grow = clamp(free_space.shortest_side // 2, 8, 128)
            inpaint.feather = inpaint.grow // 2
            inpaint.blend = max(inpaint.feather // 2, 15)

        if mask is not None:
            workflow_kind = WorkflowKind.refine_region
            bounds, mask.bounds = compute_relative_bounds(mask.bounds, mask.bounds)
        if mask is not None or workflow_kind is WorkflowKind.refine:
            image = self._get_current_image(bounds)

        conditioning, job_regions = process_regions(self.regions, bounds)
        conditioning.language = self.prompt_translation_language
        conditioning = self._add_reference_layers(conditioning)
        conditioning, loras, _ = workflow.prepare_prompts(
            conditioning, self.style, self.seed, self.arch, is_live=True
        )

        input = workflow.prepare(
            workflow_kind,
            image or bounds.extent,
            conditioning,
            self.style,
            self.seed,
            client.models,
            FileLibrary.instance(),
            self._performance_settings(client),
            mask=mask,
            strength=self.live.strength,
            loras=loras,
            inpaint=inpaint if mask else None,
            is_live=True,
        )
        params = JobParams(bounds, conditioning.positive, regions=job_regions)
        return input, params

    async def _generate_live(self, input: WorkflowInput, job_params: JobParams):
        self.clear_error()
        await self.enqueue_jobs(input, JobKind.live_preview, job_params)

    async def _generate_custom(self, previous_input: WorkflowInput | None):
        if self.workspace is not Workspace.custom or not self.document.is_active:
            return False

        try:
            wf = ensure(self.custom.graph)
            client = self._connection.client
            is_live = self.custom.mode is CustomGenerationMode.live
            is_anim = self.custom.mode is CustomGenerationMode.animation
            seed = self.seed if is_live or self.fixed_seed else workflow.generate_seed()
            canvas_bounds = Bounds(0, 0, *self._doc.extent)
            bounds = canvas_bounds
            mask = None

            if selection_node := next(wf.find(type="ETN_KritaSelection"), None):
                mods = get_selection_modifiers(Arch.sdxl, InpaintMode.fill, self.strength)
                mask, select_bounds = self._doc.create_mask_from_selection(mods)
                mask, bounds = self.custom.prepare_mask(selection_node, mask, select_bounds, bounds)

            img_input = ImageInput.from_extent(bounds.extent)
            img_input.initial_image = self._get_current_image(bounds)
            img_input.hires_mask = mask.to_image(bounds.extent) if mask else None

            params = self.custom.collect_parameters(
                self.layers, canvas_bounds, client.models, is_live, is_anim
            )

            custom_input = CustomWorkflowInput(wf.root, params)
            metadata: dict[str, Any] = dict(self.custom.params)
            job_params = JobParams(bounds, self.custom.job_name, metadata=metadata)

            style_node = next(wf.find(type="ETN_KritaStyleAndPrompt"), None)
            if style_node is not None:
                style = self.style
                is_live = style_node.input("sampler_preset", "auto") == "live"
                custom_input.models = style.get_models(client.models.checkpoints)
                custom_input.sampling = workflow.sampling_from_style(style, 1.0, is_live)

                cond = ConditioningInput(self.regions.positive, self.regions.negative)
                arch = resolve_arch(style, client)
                prepared = workflow.prepare_prompts(cond, style, seed, arch)
                custom_input.positive_evaluated = prepared.metadata["prompt_final"]
                custom_input.negative_evaluated = prepared.metadata["negative_prompt_final"]
                custom_input.models.loras = unique(
                    custom_input.models.loras + prepared.loras, key=lambda l: l.name
                )

                job_params.set_style(self.style, custom_input.models.checkpoint)
                metadata.update(prepared.metadata)
                metadata["loras"] = [
                    {"name": l.name, "weight": l.strength} for l in custom_input.models.loras
                ]

            input = WorkflowInput(
                WorkflowKind.custom,
                img_input,
                sampling=SamplingInput("custom", "custom", 1, 1000, seed=seed),
                inpaint=InpaintParams(InpaintMode.fill, bounds),
                custom_workflow=custom_input,
            )
            job_kind = {
                CustomGenerationMode.regular: JobKind.diffusion,
                CustomGenerationMode.live: JobKind.live_preview,
                CustomGenerationMode.animation: JobKind.animation,
            }[self.custom.mode]

            if input == previous_input:
                return None

            self.clear_error()
            await self.enqueue_jobs(input, job_kind, job_params, count=self.batch_count)

        except Exception as e:
            self.report_error(util.log_error(e))
            return False
        else:
            return input

    def _get_current_image(self, bounds: Bounds):
        exclude = []
        if self.workspace is not Workspace.live:
            exclude = [  # exclude control layers from projection
                c.layer for c in self.regions.control if not c.mode.is_part_of_image
            ]
            if self._layer:  # exclude preview layer
                exclude.append(self._layer)

        if not any(l.is_visible and l not in exclude for l in self.layers.images):
            warning = _(
                "Tried to capture the current image, but there are no visible layers! Preview and control layers are not considered to be part of the input image."
            )
            raise ValueError(warning)

        return self._doc.get_image(bounds, exclude_layers=exclude)

    def generate_control_layer(self, control: ControlLayer):
        doc = self.document
        ok, msg = doc.check_color_mode()
        if not ok and msg:
            self.report_error(msg)
            return

        try:
            image = doc.get_image(Bounds(0, 0, *self._doc.extent))
            mask, _ = doc.create_mask_from_selection(SelectionModifiers(pad_rel=0.25, multiple=64))
            bounds = mask.bounds if mask else None
            perf = self._performance_settings(self._connection.client)
            input = workflow.prepare_create_control_image(image, control.mode, perf, bounds)
            job = self.jobs.add_control(control, Bounds(0, 0, *image.extent))
        except Exception as e:
            self.report_error(util.log_error(e))
            return

        self.clear_error()
        eventloop.run(_report_errors(self, self._enqueue_job(job, input)))
        return job

    def cancel(self, active=False, queued=False):
        if queued and (to_cancel := self.clear_queued()):
            self._connection.cancel(to_cancel)
        if active and self.jobs.any_executing():
            self._connection.interrupt()

    def clear_queued(self):
        to_remove = [job for job in self.jobs if job.state is JobState.queued]
        for job in to_remove:
            self.jobs.remove(job)
        return [job.id for job in to_remove if job.id is not None]

    def report_error(self, error: Error | str):
        if isinstance(error, str):
            error = Error.from_string(error, ErrorKind.server_error)
        self.error = error
        self.live.is_active = False
        self.custom.is_live = False

    def clear_error(self):
        if self.error:
            self.error = no_error

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
            self.progress_kind = ProgressKind.generation
            self.progress = message.progress
        elif message.event is ClientEvent.upload:
            self.jobs.notify_started(job)
            self.progress_kind = ProgressKind.upload
            self.progress = message.progress
        elif message.event is ClientEvent.output:
            self.custom.handle_output(job, message.result)
        elif message.event is ClientEvent.finished:
            if message.error:  # successful jobs may have encountered some warnings
                self.report_error(Error.from_string(message.error, ErrorKind.warning))
            if message.images:
                self.jobs.set_results(job, message.images)
            if job.kind is JobKind.control_layer:
                assert job.control is not None
                job.control.layer_id = self.add_control_layer(job, message.result).id
            elif job.kind is JobKind.upscaling:
                self.add_upscale_layer(job)
            self._finish_job(job, message.event)
        elif message.event is ClientEvent.interrupted:
            self._finish_job(job, message.event)
        elif message.event is ClientEvent.error:
            self._finish_job(job, message.event)
            self.report_error(_("Server error") + f": {message.error}")
        elif message.event is ClientEvent.payment_required:
            self._finish_job(job, ClientEvent.error)
            assert isinstance(message.error, str) and isinstance(message.result, dict)
            self.report_error(Error(ErrorKind.insufficient_funds, message.error, message.result))

    def _finish_job(self, job: Job, event: ClientEvent):
        if job.kind is JobKind.upscaling:
            self.upscale.set_in_progress(False)

        if event is ClientEvent.finished:
            self.jobs.notify_finished(job)
            self.progress = 1

            if job.id and job.kind in [JobKind.diffusion, JobKind.animation]:
                action = settings.generation_finished_action
                if action is GenerationFinishedAction.preview and self._layer is None:
                    self.jobs.select(job.id, 0)
                elif action is GenerationFinishedAction.apply:
                    self.apply_generated_result(job.id, 0)
        else:
            self.jobs.notify_cancelled(job)
            self.progress = 0

    def update_preview(self):
        if selection := self.jobs.selection:
            self.show_preview(selection[0].job, selection[0].image)
        else:
            self.hide_preview()

    def show_preview(self, job_id: str, index: int, name_prefix="Preview"):
        job = self.jobs.find(job_id)
        assert job is not None, "Cannot show preview, invalid job id"
        if job.kind is JobKind.animation:
            return  # don't show animation preview on canvas (it's slow and clumsy)

        name = f"[{name_prefix}] {trim_text(job.params.name, 77)}"
        image = job.results[index]
        bounds = job.params.bounds
        if image.extent != bounds.extent:
            image = Image.crop(image, Bounds(0, 0, *Extent.min(bounds.extent, image.extent)))
        if self._layer and self._layer.was_removed:
            self._layer = None  # layer was removed by user
        if self._layer is not None:
            self._layer.name = name
            self._layer.write_pixels(image, bounds)
            self._layer.move_to_top()
        else:
            self._layer = self.layers.create(name, image, bounds, make_active=False)
            self._layer.is_locked = True

    def hide_preview(self, delete_layer=False):
        if self._layer is not None:
            if delete_layer:
                self._layer.remove()
                self._layer = None
            else:
                self._layer.hide()

    def apply_result(
        self,
        image: Image,
        params: JobParams,
        behavior=ApplyBehavior.layer,
        region_behavior=ApplyRegionBehavior.layer_group,
        prefix="",
    ):
        if params.resize_canvas and self.document.extent != image.extent:
            self.document.resize_canvas(*image.extent)

        bounds = Bounds(*params.bounds.offset, *image.extent)
        if len(params.regions) == 0 or region_behavior is ApplyRegionBehavior.none:
            if behavior is ApplyBehavior.replace:
                self.layers.update_layer_image(self.layers.active, image, bounds)
            else:
                name = f"{prefix}{trim_text(params.name, 200)} ({params.seed})"
                pos = self.layers.active if behavior is ApplyBehavior.layer_active else None
                self.layers.create(name, image, bounds, above=pos)
        else:  # apply to regions
            with RestoreActiveLayer(self.layers) as restore:
                active_id = Region.link_target(self.layers.active).id_string
                for job_region in params.regions:
                    result = self.create_result_layer(
                        image, params, job_region, region_behavior, prefix
                    )
                    if job_region.layer_id == active_id:
                        restore.target = result

    def create_result_layer(
        self,
        image: Image,
        params: JobParams,
        job_region: JobRegion,
        behavior: ApplyRegionBehavior,
        prefix="",
    ):
        name = f"{prefix}{job_region.prompt} ({params.seed})"
        region_layer = self.layers.find(QUuid(job_region.layer_id)) or self.layers.root
        # a previous apply from the same batch may have already created groups and re-linked
        region_layer = Region.link_target(region_layer)

        # Replace content if requested and not a group layer
        if behavior is ApplyRegionBehavior.replace and region_layer.type is not LayerType.group:
            region = self.regions.find_linked(region_layer)
            new_layer = self.layers.update_layer_image(
                region_layer, image, params.bounds, keep_alpha=True
            )
            if region is not None:
                region.link(new_layer)
            return new_layer

        # Promote layer to group if needed
        if region_layer.type is not LayerType.group:
            paint_layer = region_layer
            region_layer = self.layers.create_group_for(paint_layer)
            if region := self.regions.find_linked(paint_layer, RegionLink.direct):
                region.unlink(paint_layer)
                region.link(region_layer)

        # Crop the full image to the region bounds (+ padding for some flexibility)
        region_image = image
        region_bounds = params.bounds
        if job_region.bounds != params.bounds:
            padding = int(0.1 * job_region.bounds.extent.average_side)
            region_bounds = Bounds.pad(job_region.bounds, padding)
            region_bounds = Bounds.intersection(region_bounds, params.bounds)
            region_image = Image.crop(image, region_bounds.relative_to(params.bounds))

        # Restrict the image to the alpha mask of the region layer
        has_layers = len(region_layer.child_layers) > 0
        has_mask = any(l.type.is_mask for l in region_layer.child_layers)
        if not region_layer.is_root and has_layers and not has_mask:
            layer_bounds = region_layer.bounds
            if behavior is ApplyRegionBehavior.transparency_mask:
                mask = region_layer.get_mask(layer_bounds)
                self.layers.create_mask("Transparency Mask", mask, layer_bounds, region_layer)
            else:
                layer_image = region_layer.get_pixels(region_bounds)
                layer_image.draw_image(region_image, keep_alpha=True)
                region_image = layer_image
                if not (behavior is ApplyRegionBehavior.no_hide or params.has_mask):
                    for layer in region_layer.child_layers:
                        layer.is_visible = False

        # Handle auto-generated background region (not linked to any layers)
        insert_pos = None
        if job_region.is_background:
            insert_pos = self.regions.last_unlinked_layer(region_layer)

        return self.layers.create(
            name, region_image, region_bounds, parent=region_layer, above=insert_pos
        )

    def apply_generated_result(self, job_id: str, index: int):
        job = self.jobs.find(job_id)
        assert job is not None, "Cannot apply result, invalid job id"

        if job.kind is JobKind.animation and len(job.results) > 1:
            self.apply_animation(job)
        elif job.params.is_layered and len(job.results) > 1:
            for i, image in enumerate(job.results):
                self.apply_result(image, job.params, prefix=f"[Layer {i + 1}] ")
        else:
            self.apply_result(
                job.results[index],
                job.params,
                settings.apply_behavior,
                settings.apply_region_behavior,
                "[Generated] ",
            )
        if self._layer:
            self._layer.remove()
            self._layer = None
        self.jobs.selection = []
        self.jobs.notify_used(job_id, index)

    def apply_animation(self, job: Job):
        assert job.kind is JobKind.animation
        with TemporaryDirectory(prefix="animation") as temp_dir:
            frames = []
            for i, image in enumerate(job.results):
                filename = Path(temp_dir) / f"{i:03}.png"
                image.save(filename)
                frames.append(filename)
            self.document.import_animation(frames, self.document.playback_time_range[0])

        async def _set_layer_name():
            self.layers.active.name = f"[Animation] {trim_text(job.params.name, 200)}"

        eventloop.run(_set_layer_name())

    def add_control_layer(self, job: Job, result: ClientOutput | None):
        assert job.kind is JobKind.control_layer and job.control
        if job.control.mode is ControlMode.pose and isinstance(result, (dict, list)):
            pose = Pose.from_open_pose_json(result)
            pose.scale(job.params.bounds.extent)
            return self.layers.create_vector(job.params.name, pose.to_svg())
        elif len(job.results) > 0:
            return self.layers.create(job.params.name, job.results[0], job.params.bounds)
        return self.layers.active  # Execution was cached and no image was produced

    def add_upscale_layer(self, job: Job):
        assert job.kind is JobKind.upscaling
        assert len(job.results) > 0, "Upscaling job did not produce an image"
        if self._layer:
            self._layer.remove()
            self._layer = None
        self.apply_result(
            job.results[0],
            job.params,
            settings.apply_behavior,
            settings.apply_region_behavior,
            "[Upscale] ",
        )

    def set_workspace(self, workspace: Workspace):
        if self.workspace is Workspace.live:
            self.live.is_active = False
        self._workspace = workspace
        self.workspace_changed.emit(workspace)
        self.modified.emit(self, "workspace")

    def set_style(self, style: Style):
        if style is not self._style:
            if self._style_connection:
                QObject.disconnect(self._style_connection)
            self._style = style
            self._style_connection = style.changed.connect(self._handle_style_changed)
            self.style_changed.emit(style)
            self.modified.emit(self, "style")
            self.edit_mode = self.is_editing

    def _handle_style_changed(self):
        self.style_changed.emit(self.style)
        self.edit_mode = self.is_editing

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

    def _add_reference_layers(self, cond: ConditioningInput):
        def add_refs(control: list[ControlInput], layer_names: list[str]):
            for layer_name in layer_names:
                uid = next((l.id for l in self._doc.layers.images if l.name == layer_name), None)
                if uid is None:
                    raise PluginError(_("Layer not found") + f' "{layer_name}"')
                ctrl = ControlLayer(self, ControlMode.reference, uid, 0)
                control.append(ctrl.to_api())

        _prompt, layers = extract_layers(cond.positive)
        add_refs(cond.control, layers)

        for region in cond.regions:
            _prompt, region_layers = extract_layers(region.positive)
            add_refs(region.control, region_layers)

        cond.edit_reference = self.is_editing
        return cond

    def _performance_settings(self, client: Client):
        result = client.performance_settings
        if self.resolution_multiplier != 1.0:
            result.resolution_multiplier = self.resolution_multiplier
        return result

    def try_set_preview_layer(self, uid: str):
        if uid:
            try:
                self._layer = self.layers.find(QUuid(uid))
            except Exception:
                log.warning(f"Failed to set preview layer {uid}")
                self._layer = None

    @property
    def active_regions(self):
        is_edit = self.workspace is Workspace.generation and self.edit_mode
        return self.edit_regions if is_edit else self.regions

    @property
    def active_style(self):
        if self.workspace is Workspace.generation and self.edit_mode and self.edit_style:
            return self.edit_style
        return self.style

    @property
    def preview_layer_id(self):
        return self._layer.id_string if self._layer else ""

    @property
    def prompt_translation_language(self):
        return settings.prompt_translation if self.translation_enabled else ""

    @property
    def arch(self):
        return resolve_arch(self.active_style, self._connection.client_if_connected)

    @property
    def history(self):
        return (job for job in self.jobs if job.state is JobState.finished)

    @property
    def has_document(self):
        return isinstance(self._doc, KritaDocument)

    @property
    def document(self):
        return self._doc

    @document.setter
    def document(self, doc: Document):
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

    @property
    def name(self):
        return Path(self._doc.filename).stem

    @property
    def edit_style(self) -> Style | None:
        style_arch = resolve_arch(self.style, self._connection.client_if_connected)
        if style_arch.supports_edit:
            return self.style
        if style_id := self.style.linked_edit_style:
            if style := Styles.list().find(style_id):
                if is_style_supported(style, self._connection.client_if_connected):
                    return style
        return None

    @property
    def can_edit(self):
        return self.edit_style is not None

    @property
    def can_toggle_edit(self):
        style_arch = resolve_arch(self.style, self._connection.client_if_connected)
        return not style_arch.is_edit and self.can_edit

    @property
    def is_editing(self):
        return self.arch.is_edit or (self.can_edit and self.edit_mode)


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

    def get_params(self, mask: Mask, is_editing: bool):
        fill = FillMode.none if is_editing else self.fill
        params = InpaintParams(self.mode, mask.bounds, fill)
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


@dataclass(frozen=True)
class UpscaleParams:
    upscale: UpscaleInput
    factor: float
    use_diffusion: bool
    unblur_strength: float
    use_prompt: bool
    strength: float
    target_extent: Extent
    seed: int


class TileOverlapMode(Enum):
    auto = 0
    custom = 1


class UpscaleWorkspace(QObject, ObservableProperties):
    upscaler = Property("", persist=True)
    factor = Property(2.0, persist=True, setter="_set_factor")
    use_diffusion = Property(True, persist=True)
    strength = Property(0.3, persist=True)
    unblur_strength = Property(0.5, persist=True)
    tile_overlap_mode = Property(TileOverlapMode.auto, persist=True)
    tile_overlap = Property(48, persist=True)
    use_prompt = Property(False, persist=True)
    can_generate = Property(True)

    upscaler_changed = pyqtSignal(str)
    factor_changed = pyqtSignal(float)
    use_diffusion_changed = pyqtSignal(bool)
    strength_changed = pyqtSignal(float)
    unblur_strength_changed = pyqtSignal(float)
    tile_overlap_mode_changed = pyqtSignal(TileOverlapMode)
    tile_overlap_changed = pyqtSignal(int)
    use_prompt_changed = pyqtSignal(bool)
    target_extent_changed = pyqtSignal(Extent)
    can_generate_changed = pyqtSignal(bool)
    modified = pyqtSignal(QObject, str)

    def __init__(self, model: Model):
        super().__init__()
        self._model = weakref.ref(model)
        self._in_progress = False
        self.use_diffusion_changed.connect(self._update_can_generate)
        self._init_model()
        model._connection.models_changed.connect(self._init_model)

    def _init_model(self):
        model = ensure(self._model())
        if client := model._connection.client_if_connected:
            if self.upscaler not in client.models.upscalers:
                self.upscaler = client.models.default_upscaler

    def set_in_progress(self, in_progress: bool):
        self._in_progress = in_progress
        self._update_can_generate()

    def _set_factor(self, value: float):
        if self._factor != value:
            self._factor = value
            self.factor_changed.emit(value)
            self.target_extent_changed.emit(self.target_extent)
            self._update_can_generate()

    def _update_can_generate(self):
        self.can_generate = not self._in_progress

    @property
    def target_extent(self):
        return ensure(self._model()).document.extent * self.factor

    @property
    def params(self):
        model = ensure(self._model())
        overlap = self.tile_overlap if self.tile_overlap_mode is TileOverlapMode.custom else -1
        return UpscaleParams(
            upscale=UpscaleInput(self.upscaler, overlap),
            factor=self.factor,
            use_diffusion=self.use_diffusion,
            unblur_strength=self.unblur_strength,
            use_prompt=self.use_prompt,
            strength=1.0 if model.arch.is_edit else self.strength,
            target_extent=self.target_extent,
            seed=model.seed if model.fixed_seed else workflow.generate_seed(),
        )


class LiveScheduler:
    poll_rate = 0.1
    default_grace_period = 0.25  # seconds to delay after most recent document edit
    max_wait_time = 3.0  # maximum seconds to delay over total editing time
    delay_threshold = 1.5  # use delay only if average generation time exceeds this value

    def __init__(self):
        self._last_input: WorkflowInput | None = None
        self._last_change = 0.0
        self._oldest_change = 0.0
        self._has_changes = True
        self._generation_start_time = 0.0
        self._generation_times: deque[float] = deque(maxlen=10)

    def should_generate(self, input: WorkflowInput):
        now = time.monotonic()
        if self._last_input != input:
            self._last_input = input
            self._last_change = now
            if not self._has_changes:
                self._oldest_change = now
            self._has_changes = True

        time_since_last_change = now - self._last_change
        time_since_oldest_change = now - self._oldest_change
        return self._has_changes and (
            time_since_last_change >= self.grace_period
            or time_since_oldest_change >= self.max_wait_time
        )

    def notify_generation_started(self):
        self._generation_start_time = time.monotonic()
        self._has_changes = False

    def notify_generation_finished(self):
        self._generation_times.append(time.monotonic() - self._generation_start_time)

    @property
    def average_generation_time(self):
        return sum(self._generation_times) / max(1, len(self._generation_times))

    @property
    def grace_period(self):
        if self.average_generation_time > self.delay_threshold:
            return self.default_grace_period
        return 0.0


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

    def __init__(self, model: Model):
        super().__init__()
        self._model = weakref.ref(model)
        self._scheduler = LiveScheduler()
        self._result: Image | None = None
        self._result_composition: Image | None = None
        self._result_params: JobParams | None = None
        self._keyframes_folder: Path | None = None
        self._keyframe_start = 0
        self._keyframe_index = 0
        self._keyframes: list[Path] = []
        model.jobs.job_finished.connect(self.handle_job_finished)

    @property
    def model(self):
        return ensure(self._model())

    def toggle(self, active: bool):
        if self.is_active != active:
            self._is_active = active
            self.is_active_changed.emit(active)
            if active:
                eventloop.run(_report_errors(self.model, self._continue_generating()))
            else:
                self.is_recording = False

    def toggle_record(self, active: bool):
        if self.is_recording != active:
            if active and not self._start_recording():
                self.model.report_error(
                    _("Cannot save recorded frames, document must be saved first!")
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
            self.is_active = self._is_active and self.model.document.is_active
            self._scheduler.notify_generation_finished()
            eventloop.run(_report_errors(self.model, self._continue_generating()))

    async def _continue_generating(self):
        while self.is_active:
            if self.model.document.is_active:
                new_input, job_params = self.model._prepare_live_workflow()
                if self._scheduler.should_generate(new_input):
                    await self.model._generate_live(new_input, job_params)
                    self._scheduler.notify_generation_started()
                    return
            await asyncio.sleep(self._scheduler.poll_rate)

    def apply_result(self, layer_only=False):
        assert self.result is not None and self._result_params is not None
        params = copy(self._result_params)
        if layer_only and len(self._result_params.regions) > 0:
            active = Region.link_target(self.model.layers.active).id_string
            if region := next((r for r in params.regions if r.layer_id == active), None):
                params.regions = [region]

        behavior = settings.apply_behavior_live
        region_behavior = settings.apply_region_behavior_live
        if layer_only:
            behavior = ApplyBehavior.layer
            region_behavior = ApplyRegionBehavior.layer_group
        self.model.apply_result(self.result, params, behavior, region_behavior)

        if settings.new_seed_after_apply:
            self.model.generate_seed()

    @property
    def result(self):
        return self._result

    @property
    def result_composition(self):
        return self._result_composition

    def set_result(self, value: Image, params: JobParams):
        canvas = self.model._get_current_image(params.bounds)
        painter = QPainter(canvas._qimage)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Multiply)
        painter.setBrush(QBrush(QColor(0, 0, 96, 192), Qt.BrushStyle.DiagCrossPattern))
        painter.drawRect(0, 0, canvas.width, canvas.height)
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
        doc_filename = self.model.document.filename
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

        extent = self.model.document.extent
        if bounds is not None and bounds.extent != extent:
            image = Image.crop(image, bounds)
        image.save(filename)
        self._keyframes.append(filename)

    def _import_animation(self):
        if len(self._keyframes) == 0:
            return  # button toggled without recording a frame in between
        self.model.document.import_animation(self._keyframes, self._keyframe_start)
        start, end = self._keyframe_start, self._keyframe_start + len(self._keyframes)
        prompt = self.model.regions.active_or_root.positive
        self.model.layers.active.name = f"[Rec] {start}-{end}: {prompt}"
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

    def _prepare_input(self, canvas: Image | Extent, seed: int, time: int):
        m = self._model

        kind = WorkflowKind.generate
        if m.strength < 1.0 or m.is_editing:
            kind = WorkflowKind.refine
        bounds = Bounds(0, 0, *m.document.extent)
        is_live = self.sampling_quality is SamplingQuality.fast
        conditioning, _ = process_regions(m.regions, bounds, self._model.layers.root, time=time)
        conditioning.language = m.prompt_translation_language
        conditioning = m._add_reference_layers(conditioning)
        conditioning, loras, _prompt_meta = workflow.prepare_prompts(
            conditioning, m.style, seed, m.arch, is_live=is_live
        )

        return workflow.prepare(
            kind,
            canvas,
            conditioning,
            style=m.style,
            seed=seed,
            loras=loras,
            perf=m._performance_settings(m._connection.client),
            models=m._connection.client.models,
            files=FileLibrary.instance(),
            strength=m.strength,
            is_live=is_live,
        )

    async def _generate_frame(self):
        m = self._model
        requires_image = m.strength < 1.0 or m.arch.is_edit
        bounds = Bounds(0, 0, *m.document.extent)
        canvas = m._get_current_image(bounds) if requires_image else bounds.extent
        seed = m.seed if m.fixed_seed else workflow.generate_seed()
        inputs = self._prepare_input(canvas, seed, m.document.current_time)
        params = JobParams(bounds, m.regions.positive, frame=(m.document.current_time, 0, 0))
        await m.enqueue_jobs(inputs, JobKind.animation_frame, params)

    def generate_batch(self):
        m = self._model
        doc = m.document
        requires_image = m.strength < 1.0 or m.arch.is_edit
        if requires_image and not m.layers.active.is_animated:
            m.report_error(_("The active layer does not contain an animation."))
            return

        if doc.filename:
            path = Path(doc.filename)
            folder = path.parent / f"{path.with_suffix('.animation')}"
            folder.mkdir(exist_ok=True)
            self._keyframes_folder = folder
        else:
            m.report_error(_("Document must be saved before generating an animation."))
            return

        m.clear_error()
        eventloop.run(_report_errors(m, self._generate_batch()))

    async def _generate_batch(self):
        m = self._model
        doc = m.document
        layer = m.layers.active
        start_frame, end_frame = doc.playback_time_range
        extent = doc.extent
        bounds = Bounds(0, 0, *extent)
        seed = m.seed if m.fixed_seed else workflow.generate_seed()
        animation_id = str(uuid.uuid4())

        for frame in range(start_frame, end_frame + 1):
            if layer.node.hasKeyframeAtTime(frame) or m.strength == 1.0:
                canvas: Image | Extent = extent
                if m.strength < 1.0 or m.arch.is_edit:
                    canvas = layer.get_pixels(time=frame)

                inputs = self._prepare_input(canvas, seed, frame)
                params = JobParams(bounds, self._model.regions.active_or_root.positive)
                params.frame = (frame, start_frame, end_frame)
                params.animation_id = animation_id
                await self._model.enqueue_jobs(inputs, JobKind.animation_batch, params)

    def handle_job_finished(self, job: Job):
        if job.kind is JobKind.animation_batch:
            assert self._keyframes_folder is not None
            frame, __, end = job.params.frame
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
                    self._model.report_error(_("Generated frame does not match current time"))
                    return
                if layer := self._model.layers.find(self.target_layer):
                    image = job.results[0]
                    layer.write_pixels(image, job.params.bounds, make_visible=False)
                    self.target_image_changed.emit(image)
                else:
                    self._model.report_error(_("Target layer not found"))

    def _import_animation(self, job: Job):
        doc = self._model.document
        keyframes = self._keyframes.pop(job.params.animation_id)
        _, start, end = job.params.frame
        doc.import_animation(keyframes, start)
        eventloop.run(self._update_layer_name(f"[Generated] {start}-{end}: {job.params.name}"))

    async def _update_layer_name(self, name: str):
        doc = self._model.document
        doc.layers.active.name = name
        self.target_layer = doc.layers.active.id

    def _update_target_image(self):
        if self.batch_mode:
            return
        if layer := self._model.layers.find(self.target_layer):
            bounds = Bounds(0, 0, *self._model.document.extent)
            image = layer.get_pixels(bounds)
            self.target_image_changed.emit(image)


def get_selection_modifiers(arch: Arch, inpaint_mode: InpaintMode, strength: float, min_size=256):
    feather = settings.selection_feather / 100
    invert = False

    if inpaint_mode is InpaintMode.replace_background and strength == 1.0:
        # only minimal grow/feather as there is often no desired transition between
        # forground object and background (to be replaced by something else entirely)
        feather = min(feather, 0.01)
        invert = True

    return SelectionModifiers(
        feather_rel=feather * strength,
        feather_min_px=round(settings.selection_min_transition * strength),
        pad_rel=settings.selection_padding / 100,
        pad_offset_px=settings.selection_grow_offset,
        size_min_px=min_size,
        multiple=arch.latent_compression_factor,
        invert=invert,
    )


def calc_selection_pre_process(
    inpaint: InpaintParams, bounds: Bounds | None, mods: SelectionModifiers
):
    """
    Computes the parameters grow, feather and blend for mask processing in the workflow:
    * denoise_mask = selection -> dilate(size=grow) -> blur(size=feather)
    * composite_mask = denoise_mask -> erode(size=blend/2) -> blur(size=blend)
    Both masks should always be fully opaque inside the original selection mask.
    """
    inpaint = copy(inpaint)
    if bounds is None or settings.selection_feather == 0:
        inpaint.feather = 0
        inpaint.grow = 0
        inpaint.blend = 0
        return inpaint

    size_factor = bounds.extent.diagonal
    inpaint.feather = int(mods.feather_rel * size_factor)
    if not mods.invert:
        inpaint.feather = max(inpaint.feather, mods.feather_min_px)
    inpaint.grow = settings.selection_grow_offset + inpaint.feather // 2
    inpaint.blend = min(settings.selection_blend, inpaint.grow + inpaint.feather // 2)
    return inpaint


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
    cur_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prompt = util.sanitize_prompt(job.params.name)
    path = Path(model.document.filename)
    name_template = (
        settings.save_image_file_name_format
        or "{document_name}-generated-{job_timestamp}-{job_index}-{prompt}"
    )
    try:
        image_name = name_template.format(
            document_name=path.stem,
            job_timestamp=timestamp,
            current_timestamp=cur_timestamp,
            job_index=index,
            prompt=prompt,
        )
    except Exception:
        image_name = f"{path.stem}-generated-{timestamp}-{index}-{prompt}"

    ext = "." + settings.save_image_format.extension
    path = path.parent / f"{image_name}{ext}"
    path = util.find_unused_path(path)
    base_image = model._get_current_image(Bounds(0, 0, *model.document.extent))
    result_image = job.results[index]
    base_image.draw_image(result_image, job.params.bounds.offset)

    if settings.save_image_metadata and ext == ".png":
        metadata_text = create_img_metadata(job.params)
        base_image.save_png_with_metadata(
            filepath=path, metadata_text=metadata_text, format=settings.save_image_format
        )
    else:
        quality = None
        if settings.save_image_format is ImageFileFormat.webp:
            quality = settings.save_image_quality_webp
        elif settings.save_image_format is ImageFileFormat.jpeg:
            quality = settings.save_image_quality_jpeg

        base_image.save(path, settings.save_image_format, quality)
