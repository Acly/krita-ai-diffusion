#!/usr/bin/env python3
"""
Diffusers Server for Qwen Image Layered Pipeline

A FastAPI server that runs QwenImageLayeredPipeline for generating
layered images or segmenting existing images into layers.

Usage:
    python diffusers_server.py --port 8189

Endpoints:
    POST /generate - Submit a generation/segmentation job
    GET /status/{job_id} - Get job status and results
    POST /interrupt - Interrupt current job
    GET /system_stats - Get GPU/system information
"""

from __future__ import annotations

import argparse
import asyncio
import base64
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("diffusers_server")


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


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.pending
    progress: float = 0.0
    images: list[str] = field(default_factory=list)  # base64 encoded
    error: str | None = None
    params: dict = field(default_factory=dict)


class DiffusersServer:
    """Server managing QwenImageLayeredPipeline inference."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image-Layered",
        device: str = "cuda",
        cpu_offload: bool = False,
        vae_tiling: bool = True,
        quantization: str = "none",
        quantize_transformer: bool = True,
        quantize_text_encoder: bool = False,
        ramtorch: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.cpu_offload = cpu_offload
        self.vae_tiling = vae_tiling
        self.quantization = quantization
        self.quantize_transformer = quantize_transformer
        self.quantize_text_encoder = quantize_text_encoder
        self.ramtorch = ramtorch
        self.pipeline = None
        self.jobs: dict[str, Job] = {}
        self.current_job: Job | None = None
        self._interrupt_requested = False
        self._lock = threading.Lock()

        # Model loading status
        self.model_status = ModelStatus.not_loaded
        self.model_loading_message = ""
        self.model_loading_progress = 0.0
        self.model_error = ""

    def load_pipeline(self):
        """Load the QwenImageLayeredPipeline."""
        if self.pipeline is not None:
            return
        if self.model_status == ModelStatus.loading:
            # Already loading
            return

        self.model_status = ModelStatus.loading
        self.model_loading_message = "Checking model cache..."
        self.model_loading_progress = 0.0
        self.model_error = ""

        log.info(f"Loading pipeline from {self.model_id}...")
        try:
            from diffusers import QwenImageLayeredPipeline
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import tqdm as hf_tqdm
            import functools

            # First, download the model with progress tracking
            self.model_loading_message = "Downloading model files (this may take a while on first run)..."
            log.info(self.model_loading_message)

            # Use snapshot_download to get progress on initial download
            # This will use cache if already downloaded
            try:
                # Get the cache directory used by diffusers
                from huggingface_hub import HfFileSystem, hf_hub_download
                from huggingface_hub.constants import HF_HUB_CACHE

                # Check if model is already cached
                cache_path = snapshot_download(
                    self.model_id,
                    local_files_only=True,
                )
                self.model_loading_message = "Model found in cache, loading..."
                log.info(self.model_loading_message)
            except Exception:
                # Model not in cache, need to download
                self.model_loading_message = "Downloading model (~55GB, first run only)..."
                log.info(self.model_loading_message)

                # Download with progress tracking
                def progress_callback(progress_info):
                    if hasattr(progress_info, 'n') and hasattr(progress_info, 'total'):
                        if progress_info.total and progress_info.total > 0:
                            self.model_loading_progress = progress_info.n / progress_info.total
                            # Update message with downloaded size
                            downloaded_gb = progress_info.n / (1024**3)
                            total_gb = progress_info.total / (1024**3)
                            self.model_loading_message = f"Downloading model: {downloaded_gb:.1f}GB / {total_gb:.1f}GB"

                cache_path = snapshot_download(
                    self.model_id,
                )
                log.info(f"Model downloaded to {cache_path}")

            # Now load the pipeline
            self.model_loading_message = "Loading model into memory..."
            self.model_loading_progress = 0.5
            log.info(self.model_loading_message)

            self.pipeline = QwenImageLayeredPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
            )

            # Apply quantization if requested
            if self.quantization != "none":
                self.model_loading_message = f"Applying {self.quantization} quantization..."
                log.info(self.model_loading_message)
                self._apply_quantization()

            # Apply RamTorch for memory-efficient inference
            if self.ramtorch:
                self.model_loading_message = "Applying RamTorch memory optimization..."
                log.info(self.model_loading_message)
                self._apply_ramtorch()

            # Enable VAE tiling for reduced VRAM
            if self.vae_tiling:
                log.info("Enabling VAE tiling")
                self.pipeline.vae.enable_tiling()

            # CPU offload or move to device
            if self.cpu_offload:
                self.model_loading_message = "Enabling CPU offload..."
                log.info("Enabling CPU offload for reduced VRAM usage")
                self.pipeline.enable_model_cpu_offload()
            elif not self.ramtorch:
                # RamTorch handles device placement automatically
                self.model_loading_message = f"Moving model to {self.device}..."
                log.info(self.model_loading_message)
                self.pipeline = self.pipeline.to(self.device)

            self.pipeline.set_progress_bar_config(disable=True)
            self.model_status = ModelStatus.loaded
            self.model_loading_message = "Model loaded successfully"
            self.model_loading_progress = 1.0
            log.info("Pipeline loaded successfully")
        except Exception as e:
            log.error(f"Failed to load pipeline: {e}")
            self.model_status = ModelStatus.error
            self.model_error = str(e)
            self.model_loading_message = f"Failed to load model: {e}"
            raise

    def _apply_quantization(self):
        """Apply quantization to model components."""
        try:
            from optimum.quanto import freeze, qint4, qint8, quantize

            qtype = qint8 if self.quantization == "int8" else qint4
            log.info(f"Applying {self.quantization} quantization...")

            if self.quantize_transformer and hasattr(self.pipeline, "transformer"):
                log.info(f"Quantizing transformer with {self.quantization}")
                quantize(self.pipeline.transformer, qtype)
                freeze(self.pipeline.transformer)

            if self.quantize_text_encoder and hasattr(self.pipeline, "text_encoder"):
                log.info(f"Quantizing text encoder with {self.quantization}")
                quantize(self.pipeline.text_encoder, qtype)
                freeze(self.pipeline.text_encoder)

            log.info("Quantization applied successfully")
        except ImportError:
            log.warning("optimum-quanto not installed, skipping quantization")
        except Exception as e:
            log.error(f"Failed to apply quantization: {e}")

    def _apply_ramtorch(self):
        """Apply RamTorch for memory-efficient inference."""
        try:
            from ramtorch.helpers import replace_linear_with_ramtorch

            log.info("Applying RamTorch to model components...")

            # Apply to transformer (main compute component)
            if hasattr(self.pipeline, "transformer"):
                log.info("Applying RamTorch to transformer")
                self.pipeline.transformer = replace_linear_with_ramtorch(
                    self.pipeline.transformer, rank=0
                )
                self.pipeline.transformer = self.pipeline.transformer.to(self.device)

            # Apply to text encoder
            if hasattr(self.pipeline, "text_encoder"):
                log.info("Applying RamTorch to text encoder")
                self.pipeline.text_encoder = replace_linear_with_ramtorch(
                    self.pipeline.text_encoder, rank=0
                )
                self.pipeline.text_encoder = self.pipeline.text_encoder.to(self.device)

            # Move VAE to device (usually small enough to fit)
            if hasattr(self.pipeline, "vae"):
                self.pipeline.vae = self.pipeline.vae.to(self.device)

            log.info("RamTorch applied successfully")
        except ImportError:
            log.warning("ramtorch not installed, skipping RamTorch optimization")
        except Exception as e:
            log.error(f"Failed to apply RamTorch: {e}")

    def get_system_stats(self) -> dict:
        """Get system/GPU information."""
        devices = []

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "type": "cuda",
                    "name": props.name,
                    "vram_total": props.total_memory,
                })
        else:
            devices.append({
                "type": "cpu",
                "name": "CPU",
                "vram_total": 0,
            })

        return {"devices": devices}

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

    def run_job(self, job: Job):
        """Run a generation/segmentation job."""
        with self._lock:
            if self.current_job is not None:
                job.status = JobStatus.failed
                job.error = "Another job is already running"
                return

            self.current_job = job
            self._interrupt_requested = False

        try:
            job.status = JobStatus.processing
            self.load_pipeline()

            params = job.params
            inputs: dict[str, Any] = {
                "generator": torch.Generator(device=self.device).manual_seed(
                    params.get("seed", 42)
                ),
                "true_cfg_scale": params.get("cfg_scale", 4.0),
                "negative_prompt": params.get("negative_prompt", " "),
                "num_inference_steps": params.get("num_inference_steps", 50),
                "num_images_per_prompt": 1,
                "layers": params.get("layers", 4),
                "resolution": params.get("resolution", 640),
                "cfg_normalize": params.get("cfg_normalize", True),
                "use_en_prompt": params.get("use_en_prompt", True),
            }

            # Handle input image
            if params.get("image"):
                # Segmentation mode: use provided image
                image_data = base64.b64decode(params["image"])
                image = Image.open(io.BytesIO(image_data)).convert("RGBA")
                inputs["image"] = image
            else:
                # Generation mode: create a blank/noise image as starting point
                # The pipeline requires an image for dimension calculation
                resolution = params.get("resolution", 640)
                # Create a blank RGBA image
                image = Image.new("RGBA", (resolution, resolution), (128, 128, 128, 255))
                inputs["image"] = image

            # Handle prompt for generation mode
            if params.get("prompt"):
                inputs["prompt"] = params["prompt"]

            # Add progress callback if supported
            # Note: Check if pipeline supports callback_on_step_end
            if hasattr(self.pipeline, "callback_on_step_end"):
                inputs["callback_on_step_end"] = self._progress_callback

            log.info(f"Running job {job.id} with params: {list(params.keys())}")

            with torch.inference_mode():
                output = self.pipeline(**inputs)

            # Convert output images to base64
            # output.images[0] is a list of PIL Images (one per layer)
            layer_images = output.images[0] if output.images else []

            job.images = []
            for i, img in enumerate(layer_images):
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                job.images.append(b64)
                log.info(f"Layer {i+1}: {img.size}")

            job.status = JobStatus.completed
            job.progress = 1.0
            log.info(f"Job {job.id} completed with {len(job.images)} layers")

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
    prompt: str | None = None
    negative_prompt: str = " "
    image: str | None = None  # base64 encoded
    seed: int = 42
    cfg_scale: float = 4.0
    num_inference_steps: int = 50
    layers: int = 4
    resolution: int = 640
    cfg_normalize: bool = True
    use_en_prompt: bool = True


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

    # Validate request
    if not request.prompt and not request.image:
        raise HTTPException(
            status_code=400,
            detail="Either 'prompt' or 'image' must be provided",
        )

    params = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "image": request.image,
        "seed": request.seed,
        "cfg_scale": request.cfg_scale,
        "num_inference_steps": request.num_inference_steps,
        "layers": request.layers,
        "resolution": request.resolution,
        "cfg_normalize": request.cfg_normalize,
        "use_en_prompt": request.use_en_prompt,
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

    # Include model loading message if model is still loading
    message = None
    if job.status == JobStatus.processing and srv.model_status == ModelStatus.loading:
        message = srv.model_loading_message

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
    parser.add_argument("--model", default="Qwen/Qwen-Image-Layered", help="Model ID")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload for reduced VRAM usage")
    parser.add_argument("--vae-tiling", action="store_true", default=True, help="Enable VAE tiling")
    parser.add_argument("--no-vae-tiling", action="store_false", dest="vae_tiling", help="Disable VAE tiling")
    parser.add_argument("--quantization", default="none", choices=["none", "int8", "int4"],
                        help="Quantization level (none, int8, int4)")
    parser.add_argument("--quantize-transformer", action="store_true", default=True,
                        help="Apply quantization to transformer")
    parser.add_argument("--no-quantize-transformer", action="store_false", dest="quantize_transformer",
                        help="Don't quantize transformer")
    parser.add_argument("--quantize-text-encoder", action="store_true",
                        help="Apply quantization to text encoder")
    parser.add_argument("--ramtorch", action="store_true",
                        help="Enable RamTorch for memory-efficient inference (slower but uses less VRAM)")
    parser.add_argument("--preload", action="store_true", help="Preload model on startup")

    args = parser.parse_args()

    global server
    server = DiffusersServer(
        model_id=args.model,
        device=args.device,
        cpu_offload=args.cpu_offload,
        vae_tiling=args.vae_tiling,
        quantization=args.quantization,
        quantize_transformer=args.quantize_transformer,
        quantize_text_encoder=args.quantize_text_encoder,
        ramtorch=args.ramtorch,
    )

    if args.preload:
        server.load_pipeline()

    log.info(f"Starting server on {args.host}:{args.port}")

    try:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except ImportError:
        log.error("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
