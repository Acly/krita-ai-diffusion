#!/usr/bin/env python3
"""
Diffusers Server for General Image Diffusion

A FastAPI server that supports multiple diffusion pipelines including:
- Text-to-image generation
- Image-to-image refinement
- Inpainting
- Qwen Image Layered generation/segmentation

Usage:
    python diffusers_server.py --port 8189

Endpoints:
    POST /generate - Submit a generation job
    GET /status/{job_id} - Get job status and results
    POST /interrupt - Interrupt current job
    GET /system_stats - Get GPU/system information
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import io
import logging
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from PIL import Image

# Enable experimental Flash Attention on AMD GPUs (ROCm)
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("diffusers_server")

# Silence noisy loggers
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Try to import skrample for custom samplers
try:
    from skrample.diffusers import sampling as sk_sampling
    from skrample.diffusers import SkrampleWrapperScheduler
    # StructuredMultistep is the base class for higher-order samplers (DPM, Adams, UniPC, UniP)
    StructuredMultistep = sk_sampling.StructuredMultistep
    SKRAMPLE_AVAILABLE = True
    log.info("skrample available for custom samplers")
except ImportError:
    SKRAMPLE_AVAILABLE = False
    StructuredMultistep = None
    log.warning("skrample not installed - custom samplers disabled")


class JobStatus(Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    interrupted = "interrupted"


class ModelStatus(Enum):
    not_loaded = "not_loaded"
    loading = "loading"
    loaded = "loaded"
    error = "error"


class PipelineMode(Enum):
    """Mode for pipeline loading."""
    text_to_image = "text_to_image"
    image_to_image = "img2img"
    inpaint = "inpaint"
    qwen_layered = "qwen_layered"


class PipelineManager:
    """Manages loading and caching of different pipeline types.

    Adapted from quickdif patterns for pipeline loading, model switching,
    and quantization.
    """

    def __init__(
        self,
        device: str = "cuda",
        offload: str = "none",  # none, model, sequential
        dtype: str = "bf16",  # f16, bf16, f32
        quantization: str = "none",  # none, int8, int4
        vae_tiling: bool = True,
        ramtorch: bool = False,
        cache_on_cpu: bool = True,  # Keep one pipeline cached on CPU
    ):
        self.device = device
        self.offload = offload
        self.dtype = dtype
        self.quantization = quantization
        self.vae_tiling = vae_tiling
        self.ramtorch = ramtorch
        self.cache_on_cpu = cache_on_cpu

        self._pipe = None
        self._model_id: str | None = None
        self._mode: PipelineMode | None = None

        # CPU cache for one pipeline
        self._cached_pipe = None
        self._cached_model_id: str | None = None
        self._cached_mode: PipelineMode | None = None

        # Loading status
        self.status = ModelStatus.not_loaded
        self.loading_message = ""
        self.loading_progress = 0.0
        self.error_message = ""

    @property
    def torch_dtype(self):
        if self.dtype == "f16":
            return torch.float16
        elif self.dtype == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def unload(self):
        """Unload the current pipeline - move to CPU cache if enabled, otherwise delete."""
        if self._pipe is not None:
            if self.cache_on_cpu:
                # Move current pipeline to CPU cache
                log.info(f"Moving pipeline {self._model_id} to CPU cache")

                # First, clear the old cache if it exists
                if self._cached_pipe is not None:
                    log.info(f"Clearing cached pipeline {self._cached_model_id}")
                    del self._cached_pipe
                    self._cached_pipe = None
                    self._cached_model_id = None
                    self._cached_mode = None

                # Move current pipeline to CPU
                try:
                    self._pipe.to("cpu")
                    self._cached_pipe = self._pipe
                    self._cached_model_id = self._model_id
                    self._cached_mode = self._mode
                    log.info(f"Pipeline {self._model_id} cached on CPU")
                except Exception as e:
                    log.warning(f"Failed to cache pipeline on CPU: {e}, deleting instead")
                    del self._pipe
            else:
                log.info(f"Unloading pipeline for {self._model_id}")
                del self._pipe

            self._pipe = None
            self._model_id = None
            self._mode = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.status = ModelStatus.not_loaded

    def clear_cache(self):
        """Clear the CPU cache completely."""
        if self._cached_pipe is not None:
            log.info(f"Clearing CPU cache for {self._cached_model_id}")
            del self._cached_pipe
            self._cached_pipe = None
            self._cached_model_id = None
            self._cached_mode = None
            gc.collect()

    def update_settings(
        self,
        offload: str | None = None,
        quantization: str | None = None,
        vae_tiling: bool | None = None,
        ramtorch: bool | None = None,
    ) -> bool:
        """Update optimization settings. Returns True if pipeline needs reload."""
        needs_reload = False

        if offload is not None and offload != self.offload:
            log.info(f"Changing offload: {self.offload} -> {offload}")
            self.offload = offload
            needs_reload = True

        if quantization is not None and quantization != self.quantization:
            log.info(f"Changing quantization: {self.quantization} -> {quantization}")
            self.quantization = quantization
            needs_reload = True

        if ramtorch is not None and ramtorch != self.ramtorch:
            log.info(f"Changing ramtorch: {self.ramtorch} -> {ramtorch}")
            self.ramtorch = ramtorch
            needs_reload = True

        # VAE tiling can be changed without reload
        if vae_tiling is not None and vae_tiling != self.vae_tiling:
            log.info(f"Changing vae_tiling: {self.vae_tiling} -> {vae_tiling}")
            self.vae_tiling = vae_tiling
            if self._pipe is not None and hasattr(self._pipe, 'vae'):
                if vae_tiling:
                    self._pipe.vae.enable_tiling()
                else:
                    self._pipe.vae.disable_tiling()

        if needs_reload and self._pipe is not None:
            self.unload()

        return needs_reload

    def get_pipeline(self, model_id: str, mode: PipelineMode):
        """Get or load a pipeline for the given model and mode.

        If the same model is already loaded but in a different mode,
        convert it (e.g., txt2img -> img2img).
        Checks CPU cache before loading from scratch.
        """
        log.info(f"get_pipeline: requested model={model_id}, mode={mode.value}")
        log.info(f"get_pipeline: current model={self._model_id}, current mode={self._mode}")
        if self._cached_model_id:
            log.info(f"get_pipeline: cached model={self._cached_model_id}, cached mode={self._cached_mode}")

        # Already loaded?
        if self._model_id == model_id:
            if self._mode != mode and self._pipe is not None:
                # Same model, different mode - try to convert
                self._convert_pipeline_mode(mode)
            return self._pipe

        # Need different model - check cache first
        if self._cached_model_id == model_id:
            log.info(f"get_pipeline: restoring {model_id} from CPU cache")
            # Move current to cache, restore cached to active
            self.unload()  # This will cache the current pipe
            self._restore_from_cache(mode)
            return self._pipe

        # Not in cache - need to load fresh
        log.info(f"get_pipeline: model changed, loading fresh")
        self.unload()
        self._load_pipeline(model_id, mode)

        return self._pipe

    def _restore_from_cache(self, mode: PipelineMode):
        """Restore a pipeline from CPU cache to GPU."""
        if self._cached_pipe is None:
            return

        log.info(f"Restoring pipeline {self._cached_model_id} from CPU cache")
        self.status = ModelStatus.loading
        self.loading_message = "Restoring from cache..."

        try:
            # Move cached pipeline to GPU
            device = self.device if self.offload == "none" else "cpu"
            self._cached_pipe.to(device)

            self._pipe = self._cached_pipe
            self._model_id = self._cached_model_id
            self._mode = self._cached_mode

            self._cached_pipe = None
            self._cached_model_id = None
            self._cached_mode = None

            # Apply offloading if needed
            if self.offload == "model":
                self._pipe.enable_model_cpu_offload()
            elif self.offload == "sequential":
                self._pipe.enable_sequential_cpu_offload()

            # Handle mode conversion if needed
            if self._mode != mode:
                self._convert_pipeline_mode(mode)

            self.status = ModelStatus.loaded
            self.loading_message = "Restored from cache"
            log.info(f"Pipeline {self._model_id} restored from cache")

        except Exception as e:
            log.warning(f"Failed to restore from cache: {e}, loading fresh")
            self._cached_pipe = None
            self._cached_model_id = None
            self._cached_mode = None
            self._load_pipeline(self._cached_model_id, mode)

    def _load_pipeline(self, model_id: str, mode: PipelineMode):
        """Load a pipeline from HuggingFace or local path."""
        self.status = ModelStatus.loading
        self.loading_message = "Checking model..."
        self.loading_progress = 0.0

        try:
            if mode == PipelineMode.qwen_layered:
                self._load_qwen_layered(model_id)
            else:
                self._load_general_pipeline(model_id, mode)

            self._model_id = model_id
            self._mode = mode
            self.status = ModelStatus.loaded
            self.loading_message = "Model loaded successfully"
            self.loading_progress = 1.0
            log.info(f"Pipeline loaded: {model_id} ({mode.value})")

        except Exception as e:
            self.status = ModelStatus.error
            self.error_message = str(e)
            self.loading_message = f"Failed to load: {e}"
            log.error(f"Failed to load pipeline: {e}")
            raise

    def _load_qwen_layered(self, model_id: str):
        """Load Qwen Image Layered pipeline."""
        from diffusers import QwenImageLayeredPipeline
        from huggingface_hub import snapshot_download

        self.loading_message = "Checking Qwen model cache..."
        log.info(self.loading_message)

        # Check cache
        try:
            snapshot_download(model_id, local_files_only=True)
            self.loading_message = "Model found in cache, loading..."
        except Exception:
            self.loading_message = "Downloading Qwen model (~55GB)..."
            log.info(self.loading_message)
            snapshot_download(model_id)

        self.loading_message = "Loading Qwen pipeline..."
        self.loading_progress = 0.5

        self._pipe = QwenImageLayeredPipeline.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
        )

        self._apply_optimizations()

    def _load_general_pipeline(self, model_id: str, mode: PipelineMode):
        """Load a general diffusion pipeline (SD, SDXL, Flux, etc.)."""
        from diffusers import (
            AutoPipelineForText2Image,
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
        )

        self.loading_message = f"Loading {model_id}..."
        self.loading_progress = 0.3

        pipe_args = {
            "torch_dtype": self.torch_dtype,
            "safety_checker": None,
            "use_safetensors": True,
        }

        # Handle local safetensor files
        if model_id.lower().endswith((".safetensors", ".sft")):
            self._load_from_single_file(model_id, mode, pipe_args)
        else:
            # Load from HuggingFace
            if mode == PipelineMode.inpaint:
                self._pipe = AutoPipelineForInpainting.from_pretrained(model_id, **pipe_args)
            elif mode == PipelineMode.image_to_image:
                self._pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **pipe_args)
            else:
                self._pipe = AutoPipelineForText2Image.from_pretrained(model_id, **pipe_args)

        self._apply_optimizations()

    def _load_from_single_file(self, model_path: str, mode: PipelineMode, pipe_args: dict):
        """Load a pipeline from a single safetensor file."""
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            StableDiffusion3Pipeline,
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
        )

        self.loading_message = f"Loading from {model_path}..."

        # Try different pipeline types
        pipe_classes = [
            StableDiffusionXLPipeline,
            StableDiffusionPipeline,
            StableDiffusion3Pipeline,
        ]

        pipe = None
        for cls in pipe_classes:
            try:
                pipe = cls.from_single_file(model_path, **pipe_args)
                if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is None:
                    continue
                break
            except Exception:
                continue

        if pipe is None:
            raise ValueError(f"Could not load {model_path} as a diffusion pipeline")

        # Convert to appropriate mode
        if mode == PipelineMode.image_to_image:
            pipe = AutoPipelineForImage2Image.from_pipe(pipe)
        elif mode == PipelineMode.inpaint:
            pipe = AutoPipelineForInpainting.from_pipe(pipe)

        self._pipe = pipe

    def _convert_pipeline_mode(self, new_mode: PipelineMode):
        """Convert the current pipeline to a different mode."""
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting

        if self._mode == PipelineMode.qwen_layered:
            # Can't convert Qwen - need full reload
            log.warning("Cannot convert Qwen pipeline to other modes")
            return

        log.info(f"Converting pipeline from {self._mode.value} to {new_mode.value}")

        try:
            if new_mode == PipelineMode.image_to_image:
                self._pipe = AutoPipelineForImage2Image.from_pipe(self._pipe)
            elif new_mode == PipelineMode.inpaint:
                self._pipe = AutoPipelineForInpainting.from_pipe(self._pipe)
            # txt2img conversion not directly supported, would need reload

            self._mode = new_mode
        except Exception as e:
            log.error(f"Failed to convert pipeline: {e}")

    def _apply_optimizations(self):
        """Apply quantization, offloading, and other optimizations."""
        if self._pipe is None:
            return

        # Apply quantization
        if self.quantization != "none":
            self.loading_message = f"Applying {self.quantization} quantization..."
            self.loading_progress = 0.6
            self._apply_quantization()

        # Apply RamTorch
        if self.ramtorch:
            self.loading_message = "Applying RamTorch memory optimization..."
            self.loading_progress = 0.7
            self._apply_ramtorch()

        # VAE tiling
        if self.vae_tiling and hasattr(self._pipe, "vae"):
            self.loading_message = "Enabling VAE tiling..."
            log.info("Enabling VAE tiling")
            self._pipe.vae.enable_tiling()

        # Apply offloading or move to device
        self.loading_progress = 0.85
        if self.offload == "model":
            self.loading_message = "Enabling model CPU offload..."
            log.info("Enabling model CPU offload")
            self._pipe.enable_model_cpu_offload()
        elif self.offload == "sequential":
            self.loading_message = "Enabling sequential CPU offload..."
            log.info("Enabling sequential CPU offload")
            self._pipe.enable_sequential_cpu_offload()
        elif not self.ramtorch:
            self.loading_message = f"Moving model to {self.device}..."
            log.info(f"Moving pipeline to {self.device}")
            self._pipe = self._pipe.to(self.device)

        # Disable progress bar
        if hasattr(self._pipe, "set_progress_bar_config"):
            self._pipe.set_progress_bar_config(disable=True)

        self.loading_progress = 0.95

    def _apply_quantization(self):
        """Apply quantization to pipeline components."""
        try:
            from optimum.quanto import freeze, qint4, qint8, quantize

            qtype = qint8 if self.quantization == "int8" else qint4
            log.info(f"Applying {self.quantization} quantization...")

            if hasattr(self._pipe, "transformer"):
                self.loading_message = f"Quantizing transformer to {self.quantization}..."
                log.info("Quantizing transformer")
                quantize(self._pipe.transformer, qtype)
                freeze(self._pipe.transformer)

            if hasattr(self._pipe, "unet"):
                self.loading_message = f"Quantizing UNet to {self.quantization}..."
                log.info("Quantizing UNet")
                quantize(self._pipe.unet, qtype)
                freeze(self._pipe.unet)

            self.loading_message = "Quantization complete"
            log.info("Quantization applied")
        except ImportError:
            log.warning("optimum-quanto not installed, skipping quantization")
        except Exception as e:
            log.error(f"Failed to apply quantization: {e}")

    def _apply_ramtorch(self):
        """Apply RamTorch for memory-efficient inference."""
        try:
            from ramtorch.helpers import replace_linear_with_ramtorch

            log.info("Applying RamTorch...")

            if hasattr(self._pipe, "transformer"):
                self._pipe.transformer = replace_linear_with_ramtorch(
                    self._pipe.transformer, device=self.device
                )

            if hasattr(self._pipe, "unet"):
                self._pipe.unet = replace_linear_with_ramtorch(
                    self._pipe.unet, device=self.device
                )

            if hasattr(self._pipe, "text_encoder"):
                self._pipe.text_encoder = replace_linear_with_ramtorch(
                    self._pipe.text_encoder, device=self.device
                )

            if hasattr(self._pipe, "vae"):
                self._pipe.vae = self._pipe.vae.to(self.device)

            log.info("RamTorch applied")
        except ImportError:
            log.warning("ramtorch not installed, skipping")
        except Exception as e:
            log.error(f"Failed to apply RamTorch: {e}")


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.pending
    progress: float = 0.0
    message: str = ""  # Current status message
    images: list[str] = field(default_factory=list)  # base64 encoded
    error: str | None = None
    params: dict = field(default_factory=dict)


class DiffusersServer:
    """Server managing multiple diffusion pipelines."""

    def __init__(
        self,
        default_model: str = "Tongyi-MAI/Z-Image-Turbo",
        device: str = "cuda",
        offload: str = "none",
        dtype: str = "bf16",
        quantization: str = "none",
        vae_tiling: bool = True,
        ramtorch: bool = False,
        cache_on_cpu: bool = True,
    ):
        self.default_model = default_model
        self.device = device

        # Create pipeline manager
        self.pipeline_manager = PipelineManager(
            device=device,
            offload=offload,
            dtype=dtype,
            quantization=quantization,
            vae_tiling=vae_tiling,
            ramtorch=ramtorch,
            cache_on_cpu=cache_on_cpu,
        )

        self.jobs: dict[str, Job] = {}
        self.current_job: Job | None = None
        self._interrupt_requested = False
        self._lock = threading.Lock()

    # Proxy model status from pipeline manager
    @property
    def model_status(self) -> ModelStatus:
        return self.pipeline_manager.status

    @property
    def model_loading_message(self) -> str:
        return self.pipeline_manager.loading_message

    @property
    def model_loading_progress(self) -> float:
        return self.pipeline_manager.loading_progress

    @property
    def model_error(self) -> str:
        return self.pipeline_manager.error_message

    def load_pipeline(
        self,
        model_id: str | None = None,
        mode: str = "text_to_image",
        params: dict | None = None,
    ):
        """Load a pipeline for the given model and mode.

        Args:
            model_id: HuggingFace model ID or local path
            mode: Generation mode string
            params: Request params containing optimization settings
        """
        model_id = model_id or self.default_model
        params = params or {}

        # Apply per-request optimization settings
        self.pipeline_manager.update_settings(
            offload=params.get("offload"),
            quantization=params.get("quantization"),
            vae_tiling=params.get("vae_tiling"),
            ramtorch=params.get("ramtorch"),
        )

        # Map mode string to PipelineMode
        mode_map = {
            "text_to_image": PipelineMode.text_to_image,
            "img2img": PipelineMode.image_to_image,
            "image_to_image": PipelineMode.image_to_image,
            "inpaint": PipelineMode.inpaint,
            "layered_generate": PipelineMode.qwen_layered,
            "layered_segment": PipelineMode.qwen_layered,
            "qwen_layered": PipelineMode.qwen_layered,
        }
        pipeline_mode = mode_map.get(mode, PipelineMode.text_to_image)

        # Qwen layered mode requires specific model
        if pipeline_mode == PipelineMode.qwen_layered:
            model_id = "Qwen/Qwen-Image-Layered"

        return self.pipeline_manager.get_pipeline(model_id, pipeline_mode)

    def get_system_stats(self) -> dict:
        """Get system/GPU information with VRAM usage."""
        devices = []
        backend = "cpu"

        # Check for CUDA/ROCm (both use torch.cuda API)
        if torch.cuda.is_available():
            # Detect if running on ROCm (AMD) vs CUDA (NVIDIA)
            is_rocm = bool(getattr(torch.version, "hip", None))
            backend = "rocm" if is_rocm else "cuda"

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_total = props.total_memory

                # Get current VRAM usage
                vram_used = 0
                vram_percent = 0.0
                try:
                    # torch.cuda.mem_get_info returns (free, total)
                    free, total = torch.cuda.mem_get_info(i)
                    vram_used = total - free
                    vram_percent = (vram_used / total) * 100.0 if total > 0 else 0.0
                except Exception:
                    # Fallback to allocated memory (less accurate but works)
                    try:
                        vram_used = torch.cuda.memory_allocated(i)
                        vram_percent = (vram_used / vram_total) * 100.0 if vram_total > 0 else 0.0
                    except Exception:
                        pass

                devices.append({
                    "type": backend,
                    "name": props.name,
                    "vram_total": vram_total,
                    "vram_used": vram_used,
                    "vram_percent": round(vram_percent, 1),
                })

        # Check for Apple MPS
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            backend = "mps"
            vram_used = 0
            vram_total = 0
            vram_percent = 0.0

            try:
                # MPS memory functions (available in newer PyTorch)
                if hasattr(torch.mps, "driver_allocated_memory"):
                    vram_used = torch.mps.driver_allocated_memory()
                elif hasattr(torch.mps, "current_allocated_memory"):
                    vram_used = torch.mps.current_allocated_memory()

                if hasattr(torch.mps, "driver_total_memory"):
                    vram_total = torch.mps.driver_total_memory()
                    vram_percent = (vram_used / vram_total) * 100.0 if vram_total > 0 else 0.0
            except Exception:
                pass

            devices.append({
                "type": "mps",
                "name": "Apple Metal (MPS)",
                "vram_total": vram_total,
                "vram_used": vram_used,
                "vram_percent": round(vram_percent, 1),
            })

        else:
            # CPU fallback
            devices.append({
                "type": "cpu",
                "name": "CPU",
                "vram_total": 0,
                "vram_used": 0,
                "vram_percent": 0.0,
            })

        return {"devices": devices, "backend": backend}

    def create_job(self, params: dict) -> Job:
        """Create a new job."""
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, params=params)
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def request_interrupt(self):
        """Request interruption of current job."""
        self._interrupt_requested = True

    def _progress_callback(self, step: int, timestep: int, latents: torch.Tensor):
        """Callback for tracking progress."""
        if self._interrupt_requested:
            raise InterruptedError("Job interrupted by user")
        if self.current_job:
            # Estimate progress based on step
            total_steps = self.current_job.params.get("num_inference_steps", 50)
            self.current_job.progress = min(step / total_steps, 0.99)
            self.current_job.message = f"Step {step}/{total_steps}"

    def run_job(self, job: Job):
        """Run a generation job based on mode."""
        with self._lock:
            if self.current_job is not None:
                job.status = JobStatus.failed
                job.error = "Another job is already running"
                return

            self.current_job = job
            self._interrupt_requested = False

        try:
            job.status = JobStatus.processing
            job.message = "Starting..."
            params = job.params
            mode = params.get("mode", "text_to_image")
            model_id = params.get("model_id") or self.default_model

            log.info(f"Running job {job.id} mode={mode} model={model_id}")

            if mode == "layered_generate":
                # 2-stage generation: first txt2img, then Qwen segmentation
                job.images = self._run_layered_generate(params, model_id, job)
            elif mode == "layered_segment":
                # Direct Qwen segmentation of existing image
                job.message = "Loading Qwen model..."
                pipeline = self.load_pipeline(model_id, mode, params)
                job.message = "Segmenting..."
                job.images = self._run_qwen_layered(pipeline, params, job)
            else:
                # Load appropriate pipeline with per-request optimization settings
                job.message = "Loading model..."
                pipeline = self.load_pipeline(model_id, mode, params)

                # Dispatch to appropriate handler
                job.message = "Generating..."
                if mode == "inpaint":
                    job.images = self._run_inpaint(pipeline, params, job)
                elif mode in ("img2img", "image_to_image"):
                    job.images = self._run_img2img(pipeline, params, job)
                else:
                    job.images = self._run_txt2img(pipeline, params, job)

            job.status = JobStatus.completed
            job.progress = 1.0
            job.message = "Complete"
            log.info(f"Job {job.id} completed with {len(job.images)} images")

        except InterruptedError:
            job.status = JobStatus.interrupted
            job.error = "Job was interrupted"
            log.info(f"Job {job.id} was interrupted")

        except Exception as e:
            job.status = JobStatus.failed
            job.error = str(e)
            log.error(f"Job {job.id} failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            with self._lock:
                self.current_job = None
                self._interrupt_requested = False

    def _check_interrupt(self):
        """Check if interrupt was requested and raise if so."""
        if self._interrupt_requested:
            raise InterruptedError("Job interrupted by user")

    def _make_progress_callback(self, job: Job, total_steps: int, stage_prefix: str = "", stage_weight: float = 1.0, stage_offset: float = 0.0):
        """Create a callback function for pipeline progress updates.

        Args:
            job: The job to update progress on
            total_steps: Total number of inference steps
            stage_prefix: Prefix for the message (e.g., "Stage 1: ")
            stage_weight: How much of the total progress this stage represents (0-1)
            stage_offset: Starting progress offset for this stage (0-1)
        """
        def callback(pipeline, step: int, timestep: int, callback_kwargs: dict):
            # Check for interrupt request
            if self._interrupt_requested:
                pipeline._interrupt = True  # Signal pipeline to stop
                raise InterruptedError("Job interrupted by user")

            # Calculate progress within this stage
            step_progress = (step + 1) / total_steps
            # Apply stage weighting and offset
            job.progress = stage_offset + (step_progress * stage_weight)
            job.message = f"{stage_prefix}Step {step + 1}/{total_steps}"
            return callback_kwargs
        return callback

    def _apply_skrample_sampler(self, pipeline, sampler: str, order: int = 2, schedule: str = "default"):
        """Apply skrample sampler wrapper to the pipeline scheduler.

        Returns the original scheduler so it can be restored after generation.
        """
        if not SKRAMPLE_AVAILABLE:
            log.debug("skrample not available, using default scheduler")
            return None

        if (sampler == "default" or sampler is None) and (schedule == "default" or schedule is None):
            log.debug("Using default scheduler (no skrample)")
            return None

        # Map sampler names to skrample sampler classes
        sampler_map = {
            "euler": (sk_sampling.Euler, {}),
            "dpm": (sk_sampling.DPM, {"add_noise": False}),
            "sdpm": (sk_sampling.DPM, {"add_noise": True}),
            "adams": (sk_sampling.Adams, {}),
            "unipc": (sk_sampling.UniPC, {}),
            "unip": (sk_sampling.UniP, {}),
            "spc": (sk_sampling.SPC, {}),
        }

        # Determine sampler class and props
        sampler_class = None
        sampler_props = {}
        if sampler and sampler != "default" and sampler in sampler_map:
            sampler_class, sampler_props = sampler_map[sampler]

            # Apply order for higher-order samplers (StructuredMultistep subclasses)
            if StructuredMultistep is not None and issubclass(sampler_class, StructuredMultistep):
                max_order = sampler_class.max_order()
                if order > max_order:
                    log.warning(f"Order {order} exceeds {sampler_class.__name__} max order {max_order}, clamping")
                    order = max_order
                sampler_props["order"] = order

        # Build schedule modifiers list
        from skrample import scheduling
        schedule_modifiers = []
        if schedule and schedule != "default":
            schedule_map = {
                "beta": (scheduling.Beta, {"alpha": 0.6, "beta": 0.6}),
                "sigmoid": (scheduling.SigmoidCDF, {}),
                "karras": (scheduling.Karras, {}),
            }
            if schedule in schedule_map:
                mod_class, mod_props = schedule_map[schedule]
                schedule_modifiers.append((mod_class, mod_props))
                log.info(f"Using schedule modifier: {mod_class.__name__}")

        if sampler_class:
            log.info(f"Using skrample {sampler_class.__name__}" + (f" with order={order}" if "order" in sampler_props else ""))
        elif schedule_modifiers:
            log.info(f"Using default sampler with schedule modifier")

        # Save original scheduler
        original_scheduler = pipeline.scheduler

        # Wrap with skrample
        try:
            pipeline.scheduler = SkrampleWrapperScheduler.from_diffusers_config(
                original_scheduler,
                sampler=sampler_class,
                sampler_props=sampler_props,
                schedule_modifiers=schedule_modifiers,
            )
            return original_scheduler
        except Exception as e:
            log.warning(f"Failed to apply skrample sampler: {e}")
            return None

    def _run_txt2img(self, pipeline, params: dict, job: Job | None = None) -> list[str]:
        """Run text-to-image generation."""
        self._check_interrupt()

        seed = params.get("seed", 42)
        if seed < 0:
            import random
            seed = random.randint(0, 2**31 - 1)

        # Use CPU generator when offloading is enabled to avoid device mismatch
        gen_device = "cpu" if self.pipeline_manager.offload != "none" else self.device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        num_steps = params.get("num_inference_steps", 30)

        # Apply skrample sampler if requested
        sampler = params.get("sampler", "default")
        sampler_order = params.get("sampler_order", 2)
        schedule = params.get("schedule", "default")
        original_scheduler = self._apply_skrample_sampler(pipeline, sampler, sampler_order, schedule)

        try:
            # Build pipeline kwargs
            pipeline_kwargs = {
                "prompt": params.get("prompt", ""),
                "negative_prompt": params.get("negative_prompt", ""),
                "width": params.get("width", 1024),
                "height": params.get("height", 1024),
                "num_inference_steps": num_steps,
                "guidance_scale": params.get("guidance_scale", 7.5),
                "generator": generator,
                "output_type": "pil",
            }

            # Add progress callback if job provided
            if job is not None:
                pipeline_kwargs["callback_on_step_end"] = self._make_progress_callback(job, num_steps)

            output = pipeline(**pipeline_kwargs)

            return [self._encode_image(img) for img in output.images]
        finally:
            # Restore original scheduler if we changed it
            if original_scheduler is not None:
                pipeline.scheduler = original_scheduler

    def _run_layered_generate(self, params: dict, model_id: str, job: Job) -> list[str]:
        """Run 2-stage layered generation: txt2img then Qwen segmentation."""
        import random

        seed = params.get("seed", 42)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # Get step counts for progress calculation
        txt2img_steps = params.get("num_inference_steps", 30)
        qwen_steps = params.get("layered_steps", 50)
        log.info(f"Layered generate params: layered_steps={params.get('layered_steps')}, num_inference_steps={params.get('num_inference_steps')}, using txt2img={txt2img_steps}, qwen={qwen_steps}")
        total_steps = txt2img_steps + qwen_steps
        txt2img_weight = txt2img_steps / total_steps  # ~0.375 for 30/80
        qwen_weight = qwen_steps / total_steps  # ~0.625 for 50/80

        # Stage 1: Generate base image with selected model
        self._check_interrupt()  # Check before loading model
        job.message = "Loading generation model..."
        job.progress = 0.0
        log.info(f"Stage 1: Loading txt2img model {model_id}")

        # Use txt2img optimization settings from model preset
        txt2img_params = {
            "offload": params.get("offload"),
            "quantization": params.get("quantization"),
            "vae_tiling": params.get("vae_tiling"),
            "ramtorch": params.get("ramtorch"),
        }
        log.info(f"Stage 1 optimization: offload={txt2img_params['offload']}, quant={txt2img_params['quantization']}")
        txt2img_pipeline = self.load_pipeline(model_id, "text_to_image", txt2img_params)

        self._check_interrupt()  # Check after loading model
        job.message = "Generating base image..."
        log.info("Stage 1: Generating base image")

        gen_device = "cpu" if self.pipeline_manager.offload != "none" else self.device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Apply skrample sampler for Stage 1 if requested
        sampler = params.get("sampler", "default")
        sampler_order = params.get("sampler_order", 2)
        schedule = params.get("schedule", "default")
        original_scheduler = self._apply_skrample_sampler(txt2img_pipeline, sampler, sampler_order, schedule)

        # Stage 1 progress callback (0% to txt2img_weight%)
        stage1_callback = self._make_progress_callback(
            job, txt2img_steps, "Generating: ", txt2img_weight, 0.0
        )

        try:
            output = txt2img_pipeline(
                prompt=params.get("prompt", ""),
                negative_prompt=params.get("negative_prompt", ""),
                width=params.get("width", 1024),
                height=params.get("height", 1024),
                num_inference_steps=txt2img_steps,
                guidance_scale=params.get("guidance_scale", 7.5),
                generator=generator,
                output_type="pil",
                callback_on_step_end=stage1_callback,
            )
        finally:
            # Restore original scheduler
            if original_scheduler is not None:
                txt2img_pipeline.scheduler = original_scheduler

        base_image = output.images[0]
        log.info(f"Stage 1 complete: generated {base_image.size} image")

        # Check interrupt between stages
        self._check_interrupt()

        # Stage 2: Segment with Qwen
        job.message = "Loading Qwen model..."
        job.progress = txt2img_weight
        log.info("Stage 2: Loading Qwen layered model")

        # Use Qwen optimization settings from global diffusers settings
        qwen_params = {
            "offload": params.get("qwen_offload"),
            "quantization": params.get("qwen_quantization"),
            "vae_tiling": params.get("qwen_vae_tiling"),
            "ramtorch": params.get("qwen_ramtorch"),
        }
        log.info(f"Stage 2 optimization: offload={qwen_params['offload']}, quant={qwen_params['quantization']}")

        # Load Qwen pipeline (this will switch models)
        qwen_pipeline = self.load_pipeline("Qwen/Qwen-Image-Layered", "layered_segment", qwen_params)

        self._check_interrupt()  # Check after loading Qwen
        job.message = "Segmenting into layers..."
        log.info("Stage 2: Running Qwen segmentation")

        # Apply skrample sampler for Stage 2 (Qwen) if requested
        qwen_original_scheduler = self._apply_skrample_sampler(qwen_pipeline, sampler, sampler_order, schedule)

        try:
            # Convert base image to RGBA for Qwen
            base_rgba = base_image.convert("RGBA")
            original_size = base_rgba.size
            resolution = params.get("resolution", 640)

            # Scale image to fit within resolution bucket while preserving aspect ratio
            scaled_image = self._scale_to_resolution(base_rgba, resolution)

            # Always use CPU generator for Qwen
            qwen_generator = torch.Generator(device="cpu").manual_seed(seed)

            # Stage 2 progress callback (txt2img_weight% to 100%)
            stage2_callback = self._make_progress_callback(
                job, qwen_steps, "Segmenting: ", qwen_weight, txt2img_weight
            )

            qwen_inputs = {
                "generator": qwen_generator,
                "image": scaled_image,
                "prompt": params.get("prompt", ""),
                "negative_prompt": params.get("negative_prompt", " "),
                "true_cfg_scale": params.get("layered_cfg_scale", 4.0),
                "num_inference_steps": qwen_steps,
                "num_images_per_prompt": 1,
                "layers": params.get("layers", 4),
                "resolution": resolution,
                "cfg_normalize": True,
                "use_en_prompt": True,
                "callback_on_step_end": stage2_callback,
            }

            log.info(f"Qwen inputs: layers={qwen_inputs['layers']}, resolution={qwen_inputs['resolution']}, steps={qwen_inputs['num_inference_steps']}")
            log.info(f"Input image size: {scaled_image.size}, original: {original_size}")

            with torch.inference_mode():
                qwen_output = qwen_pipeline(**qwen_inputs)

            layer_images = qwen_output.images[0] if qwen_output.images else []
            log.info(f"Stage 2 complete: {len(layer_images)} layers")

            # Scale layers back to original size if we scaled down
            if original_size != scaled_image.size:
                log.info(f"Scaling layers back to original size: {original_size}")
                layer_images = [img.resize(original_size, Image.Resampling.LANCZOS) for img in layer_images]

            # Return base image first, then layers
            result = [self._encode_image(base_image)]
            result.extend([self._encode_image(img) for img in layer_images])
            return result
        finally:
            if qwen_original_scheduler is not None:
                qwen_pipeline.scheduler = qwen_original_scheduler

    def _run_img2img(self, pipeline, params: dict, job: Job | None = None) -> list[str]:
        """Run image-to-image generation."""
        self._check_interrupt()

        seed = params.get("seed", 42)
        if seed < 0:
            import random
            seed = random.randint(0, 2**31 - 1)

        # Use CPU generator when offloading is enabled to avoid device mismatch
        gen_device = "cpu" if self.pipeline_manager.offload != "none" else self.device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Decode input image
        input_image = self._decode_image(params["image"])
        width = params.get("width", input_image.width)
        height = params.get("height", input_image.height)
        input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)

        num_steps = params.get("num_inference_steps", 30)

        # Apply skrample sampler if requested
        sampler = params.get("sampler", "default")
        sampler_order = params.get("sampler_order", 2)
        schedule = params.get("schedule", "default")
        original_scheduler = self._apply_skrample_sampler(pipeline, sampler, sampler_order, schedule)

        try:
            pipeline_kwargs = {
                "prompt": params.get("prompt", ""),
                "negative_prompt": params.get("negative_prompt", ""),
                "image": input_image,
                "strength": params.get("strength", 0.75),
                "num_inference_steps": num_steps,
                "guidance_scale": params.get("guidance_scale", 7.5),
                "generator": generator,
                "output_type": "pil",
            }

            if job is not None:
                pipeline_kwargs["callback_on_step_end"] = self._make_progress_callback(job, num_steps)

            output = pipeline(**pipeline_kwargs)

            return [self._encode_image(img) for img in output.images]
        finally:
            if original_scheduler is not None:
                pipeline.scheduler = original_scheduler

    def _run_inpaint(self, pipeline, params: dict, job: Job | None = None) -> list[str]:
        """Run inpainting generation."""
        self._check_interrupt()

        seed = params.get("seed", 42)
        if seed < 0:
            import random
            seed = random.randint(0, 2**31 - 1)

        # Use CPU generator when offloading is enabled to avoid device mismatch
        gen_device = "cpu" if self.pipeline_manager.offload != "none" else self.device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Decode input image and mask
        input_image = self._decode_image(params["image"])
        mask_image = self._decode_image(params["mask"])

        width = params.get("width", input_image.width)
        height = params.get("height", input_image.height)
        input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
        mask_image = mask_image.resize((width, height), Image.Resampling.LANCZOS)

        num_steps = params.get("num_inference_steps", 30)

        # Apply skrample sampler if requested
        sampler = params.get("sampler", "default")
        sampler_order = params.get("sampler_order", 2)
        schedule = params.get("schedule", "default")
        original_scheduler = self._apply_skrample_sampler(pipeline, sampler, sampler_order, schedule)

        try:
            pipeline_kwargs = {
                "prompt": params.get("prompt", ""),
                "negative_prompt": params.get("negative_prompt", ""),
                "image": input_image,
                "mask_image": mask_image,
                "width": width,
                "height": height,
                "strength": params.get("strength", 1.0),
                "num_inference_steps": num_steps,
                "guidance_scale": params.get("guidance_scale", 7.5),
                "generator": generator,
                "output_type": "pil",
            }

            if job is not None:
                pipeline_kwargs["callback_on_step_end"] = self._make_progress_callback(job, num_steps)

            output = pipeline(**pipeline_kwargs)

            return [self._encode_image(img) for img in output.images]
        finally:
            if original_scheduler is not None:
                pipeline.scheduler = original_scheduler

    def _run_qwen_layered(self, pipeline, params: dict, job: Job | None = None) -> list[str]:
        """Run Qwen Image Layered generation/segmentation."""
        self._check_interrupt()

        seed = params.get("seed", 42)
        if seed < 0:
            import random
            seed = random.randint(0, 2**31 - 1)

        # Always use CPU generator for Qwen - it handles device placement internally
        # and quantization/offloading can cause device mismatches
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Qwen steps come from layered_steps, fallback to num_inference_steps for compatibility
        num_steps = params.get("layered_steps", params.get("num_inference_steps", 50))
        log.info(f"Qwen steps: layered_steps={params.get('layered_steps')}, num_inference_steps={params.get('num_inference_steps')}, using={num_steps}")

        # Apply skrample sampler if requested
        sampler = params.get("sampler", "default")
        sampler_order = params.get("sampler_order", 2)
        schedule = params.get("schedule", "default")
        original_scheduler = self._apply_skrample_sampler(pipeline, sampler, sampler_order, schedule)

        try:
            inputs: dict[str, Any] = {
                "generator": generator,
                "true_cfg_scale": params.get("cfg_scale", 4.0),
                "negative_prompt": params.get("negative_prompt", " "),
                "num_inference_steps": num_steps,
                "num_images_per_prompt": 1,
                "layers": params.get("layers", 4),
                "resolution": params.get("resolution", 640),
                "cfg_normalize": params.get("cfg_normalize", True),
                "use_en_prompt": params.get("use_en_prompt", True),
            }

            # Add progress callback if job provided
            if job is not None:
                inputs["callback_on_step_end"] = self._make_progress_callback(job, num_steps, "Segmenting: ")

            # Handle input image
            original_size = None
            resolution = params.get("resolution", 640)
            if params.get("image"):
                image = self._decode_image(params["image"]).convert("RGBA")
                original_size = image.size
                # Scale image to fit within resolution bucket while preserving aspect ratio
                image = self._scale_to_resolution(image, resolution)
                inputs["image"] = image
            else:
                image = Image.new("RGBA", (resolution, resolution), (128, 128, 128, 255))
                inputs["image"] = image

            if params.get("prompt"):
                inputs["prompt"] = params["prompt"]

            log.info(f"Qwen inputs: prompt={inputs.get('prompt', 'NONE')}, layers={inputs['layers']}, resolution={inputs['resolution']}, cfg={inputs['true_cfg_scale']}, steps={inputs['num_inference_steps']}")
            log.info(f"Pipeline type: {type(pipeline).__name__}, device: {self.device}")
            log.info(f"Input image size: {inputs['image'].size}, mode: {inputs['image'].mode}, original: {original_size}")

            with torch.inference_mode():
                output = pipeline(**inputs)

            # Output is list of layers
            layer_images = output.images[0] if output.images else []
            log.info(f"Output: {len(layer_images)} layers")

            # Scale layers back to original size if we scaled down
            if original_size and original_size != inputs['image'].size:
                log.info(f"Scaling layers back to original size: {original_size}")
                layer_images = [img.resize(original_size, Image.Resampling.LANCZOS) for img in layer_images]

            for i, img in enumerate(layer_images):
                log.info(f"  Layer {i}: size={img.size}, mode={img.mode}")

            # Prepend input image as first layer for debugging (use original size)
            if original_size:
                original_image = self._decode_image(params["image"]).convert("RGBA")
                result = [self._encode_image(original_image)]
            else:
                result = [self._encode_image(inputs['image'])]
            result.extend([self._encode_image(img) for img in layer_images])
            return result
        finally:
            if original_scheduler is not None:
                pipeline.scheduler = original_scheduler

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 PNG."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _decode_image(self, b64_data: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        image_data = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    def _scale_to_resolution(self, image: Image.Image, resolution: int) -> Image.Image:
        """Scale image to fit within resolution bucket while preserving aspect ratio.

        The longest side will be scaled to match the resolution.
        """
        width, height = image.size
        longest_side = max(width, height)

        if longest_side <= resolution:
            # Image already fits, no scaling needed
            return image

        # Calculate scale factor
        scale = resolution / longest_side
        new_width = int(width * scale)
        new_height = int(height * scale)

        log.info(f"Scaling image from {width}x{height} to {new_width}x{new_height} (resolution={resolution})")
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# Global server instance
server: DiffusersServer | None = None


def get_server() -> DiffusersServer:
    global server
    if server is None:
        server = DiffusersServer()
    return server


# FastAPI app
try:
    from fastapi import FastAPI, BackgroundTasks, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    log.error("FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)


app = FastAPI(
    title="Diffusers Server",
    description="Server for Qwen Image Layered Pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    # Mode selection
    mode: str = "text_to_image"  # text_to_image, img2img, inpaint, layered_generate, layered_segment
    model_id: str = ""  # HuggingFace model ID or local path (empty = use default)

    # Common parameters
    prompt: str | None = None
    negative_prompt: str = ""
    image: str | None = None  # base64 encoded input image (for img2img, inpaint, segment)
    mask: str | None = None   # base64 encoded mask (for inpaint)
    seed: int = -1  # -1 = random
    num_inference_steps: int = 30
    guidance_scale: float = 7.5

    # Resolution (for txt2img and as target for img2img/inpaint)
    width: int = 1024
    height: int = 1024

    # img2img / inpaint specific
    strength: float = 0.75

    # Qwen layered specific (legacy compatibility)
    cfg_scale: float = 4.0  # Qwen uses cfg_scale instead of guidance_scale
    layers: int = 4
    resolution: int = 640
    cfg_normalize: bool = True
    use_en_prompt: bool = True
    layered_steps: int | None = None  # Qwen inference steps (separate from txt2img num_inference_steps)

    # txt2img optimization settings (from model preset)
    offload: str | None = None  # none, model, sequential
    quantization: str | None = None  # none, int8, int4
    quantize_transformer: bool | None = None
    quantize_text_encoder: bool | None = None
    vae_tiling: bool | None = None
    ramtorch: bool | None = None

    # Qwen optimization settings (from global diffusers settings)
    qwen_offload: str | None = None
    qwen_quantization: str | None = None
    qwen_vae_tiling: bool | None = None
    qwen_ramtorch: bool | None = None

    # Sampler settings (via skrample)
    sampler: str | None = None  # default, euler, dpm, sdpm, adams, unipc, unip, spc
    sampler_order: int | None = None  # Solver order for higher-order samplers (1-9)
    schedule: str | None = None  # default, beta, sigmoid, karras


class GenerateResponse(BaseModel):
    job_id: str
    status: str


class StatusResponse(BaseModel):
    status: str
    progress: float
    message: str | None = None  # Status message (e.g., model loading progress)
    images: list[str] | None = None
    error: str | None = None


class ModelStatusResponse(BaseModel):
    status: str
    message: str
    progress: float
    error: str | None = None


# Token authentication
_auth_token: str | None = None


def set_auth_token(token: str | None):
    """Set the authentication token (call before starting server)."""
    global _auth_token
    _auth_token = token


@app.middleware("http")
async def auth_middleware(request, call_next):
    """Verify Bearer token if authentication is enabled."""
    if _auth_token is not None:
        # Skip auth for health check
        if request.url.path == "/health":
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )

        token = auth_header[7:]  # Remove "Bearer " prefix
        if token != _auth_token:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid token"},
            )

    return await call_next(request)


@app.on_event("startup")
async def startup_event():
    """Pre-load the pipeline on startup."""
    log.info("Server starting up...")
    # Optionally preload pipeline
    # get_server().load_pipeline()


@app.get("/system_stats")
async def system_stats():
    """Get system/GPU information."""
    return get_server().get_system_stats()


@app.get("/model_status", response_model=ModelStatusResponse)
async def model_status():
    """Get model loading status."""
    srv = get_server()
    return ModelStatusResponse(
        status=srv.model_status.value,
        message=srv.model_loading_message,
        progress=srv.model_loading_progress,
        error=srv.model_error if srv.model_status == ModelStatus.error else None,
    )


@app.post("/load_model")
async def load_model(background_tasks: BackgroundTasks):
    """Start loading the model in the background."""
    srv = get_server()
    if srv.model_status == ModelStatus.loaded:
        return {"status": "already_loaded"}
    if srv.model_status == ModelStatus.loading:
        return {"status": "loading"}

    background_tasks.add_task(srv.load_pipeline)
    return {"status": "loading_started"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Submit a generation/segmentation job."""
    srv = get_server()

    # Validate request based on mode
    mode = request.mode
    if mode == "text_to_image" and not request.prompt:
        raise HTTPException(
            status_code=400,
            detail="'prompt' is required for text_to_image mode",
        )
    if mode in ("img2img", "image_to_image") and not request.image:
        raise HTTPException(
            status_code=400,
            detail="'image' is required for img2img mode",
        )
    if mode == "inpaint" and (not request.image or not request.mask):
        raise HTTPException(
            status_code=400,
            detail="'image' and 'mask' are required for inpaint mode",
        )
    if mode == "layered_segment" and not request.image:
        raise HTTPException(
            status_code=400,
            detail="'image' is required for layered_segment mode",
        )
    # Note: layered_generate does NOT require a prompt - it can generate from scratch

    params = {
        "mode": mode,
        "model_id": request.model_id,
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "image": request.image,
        "mask": request.mask,
        "seed": request.seed,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "width": request.width,
        "height": request.height,
        "strength": request.strength,
        # Qwen layered specific
        "cfg_scale": request.cfg_scale,
        "layers": request.layers,
        "resolution": request.resolution,
        "cfg_normalize": request.cfg_normalize,
        "use_en_prompt": request.use_en_prompt,
        "layered_steps": request.layered_steps,
        # txt2img optimization settings (from model preset)
        "offload": request.offload,
        "quantization": request.quantization,
        "quantize_transformer": request.quantize_transformer,
        "quantize_text_encoder": request.quantize_text_encoder,
        "vae_tiling": request.vae_tiling,
        "ramtorch": request.ramtorch,
        # Qwen optimization settings (from global diffusers settings)
        "qwen_offload": request.qwen_offload,
        "qwen_quantization": request.qwen_quantization,
        "qwen_vae_tiling": request.qwen_vae_tiling,
        "qwen_ramtorch": request.qwen_ramtorch,
        # Sampler settings
        "sampler": request.sampler,
        "sampler_order": request.sampler_order,
        "schedule": request.schedule,
    }

    job = srv.create_job(params)

    # Run job in background
    background_tasks.add_task(srv.run_job, job)

    return GenerateResponse(job_id=job.id, status=job.status.value)


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Get job status and results."""
    srv = get_server()
    job = srv.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Include appropriate status message
    if job.status == JobStatus.processing and srv.model_status == ModelStatus.loading:
        # Model is still loading - show model loading message
        message = srv.model_loading_message
    else:
        # Show job progress message (step count, etc.)
        message = job.message

    return StatusResponse(
        status=job.status.value,
        progress=job.progress,
        message=message,
        images=job.images if job.status == JobStatus.completed else None,
        error=job.error,
    )


@app.post("/interrupt")
async def interrupt():
    """Interrupt the current job."""
    srv = get_server()
    srv.request_interrupt()
    return {"status": "interrupt requested"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="Diffusers Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8189, help="Port to bind to")
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo",
                        help="Default model ID (HuggingFace or local path)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
    parser.add_argument("--dtype", default="bf16", choices=["f16", "bf16", "f32"],
                        help="Data type for model weights")
    parser.add_argument("--offload", default="none", choices=["none", "model", "sequential"],
                        help="CPU offload mode (none, model, sequential)")
    parser.add_argument("--vae-tiling", action="store_true", default=True,
                        help="Enable VAE tiling for large images")
    parser.add_argument("--no-vae-tiling", action="store_false", dest="vae_tiling",
                        help="Disable VAE tiling")
    parser.add_argument("--quantization", default="none", choices=["none", "int8", "int4"],
                        help="Quantization level (requires optimum-quanto)")
    parser.add_argument("--ramtorch", action="store_true",
                        help="Enable RamTorch for memory-efficient inference")
    parser.add_argument("--cache-on-cpu", action="store_true", default=True,
                        help="Cache one pipeline on CPU when switching models (default: enabled)")
    parser.add_argument("--no-cache-on-cpu", action="store_false", dest="cache_on_cpu",
                        help="Disable CPU pipeline caching")
    parser.add_argument("--preload", action="store_true",
                        help="Preload default model on startup")

    # Security options
    parser.add_argument("--token", default=None,
                        help="Bearer token for authentication (enables auth if set)")
    parser.add_argument("--ssl-cert", default=None,
                        help="Path to SSL certificate file for HTTPS")
    parser.add_argument("--ssl-key", default=None,
                        help="Path to SSL private key file for HTTPS")

    args = parser.parse_args()

    global server
    server = DiffusersServer(
        default_model=args.model,
        device=args.device,
        offload=args.offload,
        dtype=args.dtype,
        quantization=args.quantization,
        vae_tiling=args.vae_tiling,
        ramtorch=args.ramtorch,
        cache_on_cpu=args.cache_on_cpu,
    )

    # Set up authentication if token provided
    if args.token:
        set_auth_token(args.token)
        log.info("Token authentication enabled")

    if args.preload:
        log.info(f"Preloading model: {args.model}")
        server.load_pipeline()

    # Determine protocol
    use_ssl = args.ssl_cert and args.ssl_key
    protocol = "https" if use_ssl else "http"

    log.info(f"Starting server on {protocol}://{args.host}:{args.port}")
    log.info(f"Default model: {args.model}")
    log.info(f"Device: {args.device}, dtype: {args.dtype}, offload: {args.offload}")
    if use_ssl:
        log.info(f"TLS enabled with cert: {args.ssl_cert}")

    try:
        import uvicorn

        uvicorn_kwargs = {
            "app": app,
            "host": args.host,
            "port": args.port,
            "access_log": False,
        }
        if use_ssl:
            uvicorn_kwargs["ssl_certfile"] = args.ssl_cert
            uvicorn_kwargs["ssl_keyfile"] = args.ssl_key
        uvicorn.run(**uvicorn_kwargs)
    except ImportError:
        log.error("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
