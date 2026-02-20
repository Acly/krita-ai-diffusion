#!/usr/bin/env python3

"""Utility script for downloading required and optional models
for running a ComfyUI server for the Krita Generative AI plugin.

Usage:
    python download_models.py [destination] [options]

    destination: Path to the directory where the models should be downloaded.
                 This can be your ComfyUI directory to place files directly,
                 or you can specify an empty directory and copy the files manually.

    Use --help for more options.
"""

import asyncio
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import aiohttp
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from ai_diffusion import platform_tools, resources
from ai_diffusion.resources import (
    Arch,
    ModelRequirements,
    ModelResource,
    ResourceKind,
    VerificationState,
    default_checkpoints,
    optional_models,
    required_models,
)

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


def _match_backend(model: ModelResource, backend: ModelRequirements):
    if backend is ModelRequirements.cuda:
        return model.requirements not in [ModelRequirements.cuda_fp4, ModelRequirements.no_cuda]
    if backend is ModelRequirements.cuda_fp4:
        return model.requirements not in [ModelRequirements.cuda, ModelRequirements.no_cuda]
    return model.requirements not in [ModelRequirements.cuda, ModelRequirements.cuda_fp4]


def list_models(
    sd15=False,
    sdxl=False,
    flux=False,
    flux2=False,
    illu=False,
    zimage=False,
    upscalers=False,
    checkpoints=None,
    controlnet=False,
    prefetch=False,
    deprecated=False,
    minimal=False,
    recommended=False,
    all=False,
    backend=ModelRequirements.no_cuda,
    exclude=None,
) -> set[ModelResource]:
    if exclude is None:
        exclude = []
    if checkpoints is None:
        checkpoints = []
    assert sum([minimal, recommended, all]) <= 1, (
        "Only one of --minimal, --recommended, --all can be specified"
    )

    versions = [Arch.all]
    if sd15 or minimal or all:
        versions.append(Arch.sd15)
    if sdxl or recommended or all:
        versions.append(Arch.sdxl)
    if flux or all:
        versions.append(Arch.flux)
    if flux2 or all:
        versions.append(Arch.flux2_4b)
    if illu or all:
        versions.append(Arch.illu)
        versions.append(Arch.illu_v)
    if zimage or all:
        versions.append(Arch.zimage)

    models: set[ModelResource] = set()
    models.update([m for m in default_checkpoints if all or (m.id.identifier in checkpoints)])
    if len(versions) > 1:
        models.update([m for m in required_models if m.arch in versions])
    if minimal:
        models.add(default_checkpoints[0])
    if recommended:
        models.update([m for m in default_checkpoints if m.arch is Arch.sdxl])
    if upscalers or recommended or all:
        models.update([m for m in required_models if m.kind is ResourceKind.upscaler])
        models.update(resources.upscale_models)
    if controlnet or recommended or all:
        kinds = [
            ResourceKind.controlnet,
            ResourceKind.ip_adapter,
            ResourceKind.clip_vision,
            ResourceKind.model_patch,
        ]
        models.update([m for m in optional_models if m.kind in kinds and m.arch in versions])
    if flux or all:
        lora = ResourceKind.lora
        models.update([m for m in optional_models if m.kind is lora and m.arch is Arch.flux])
    if prefetch or all:
        models.update(resources.prefetch_models)
    if deprecated:
        models.update([m for m in resources.deprecated_models if m.arch in versions])

    excluded_models = {
        m for m in models if (m.id.string in exclude) or (not _match_backend(m, backend))
    }
    models = models - excluded_models

    # Remove duplicate files listed under different IDs (apply to multiple architectures)
    file_ids = set()
    for m in list(models):
        if m.file_id in file_ids:
            models.remove(m)
        else:
            file_ids.add(m.file_id)

    if len(models) == 0:
        print("\nNo models selected for download.")

    return models


def detect_backend():
    devices = platform_tools.get_cuda_devices()
    if any(major >= 10 for (major, minor) in devices):  # Blackwell has compute capability 10.x
        return ModelRequirements.cuda_fp4
    elif len(devices) > 0:
        return ModelRequirements.cuda
    return ModelRequirements.no_cuda


def _progress(name: str, size: int | None, index=0):
    return tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, desc=name, position=index)


def _map_url(url: str):
    if replace_host := os.environ.get("AI_DIFFUSION_DOWNLOAD_URL"):
        url = url.replace("/".join(url.split("/")[:3]), replace_host)
    return url


async def download_with_retry(
    client: aiohttp.ClientSession,
    model: resources.ModelResource,
    destination: Path,
    verbose=False,
    dry_run=False,
    retry_attempts=5,
    continue_on_error=False,
    index=0,
):
    for attempt in range(retry_attempts):
        try:
            await download(client, model, destination, verbose, dry_run, index)
            break
        except Exception as e:
            print(f"Error downloading {model.name} (attempt {attempt}): {e}")
            if not continue_on_error:
                raise
    else:
        if not continue_on_error:
            raise RuntimeError(f"Failed to download {model.name} after {retry_attempts} attempts")


async def download(
    client: aiohttp.ClientSession,
    model: resources.ModelResource,
    destination: Path,
    verbose=False,
    dry_run=False,
    index=0,
):
    for file in model.files:
        target_file = destination / file.path
        url = _map_url(file.url)
        if verbose:
            print(f"Looking for {target_file}")
        if target_file.exists():
            print(f"{model.name}: found - skipping")
            continue
        if verbose:
            print(f"Downloading {url}")
        target_file.parent.mkdir(exist_ok=True, parents=True)
        if not dry_run:
            async with client.get(url) as resp:
                resp.raise_for_status()
                with open(target_file.with_suffix(".part"), "wb") as fd:  # noqa
                    with _progress(model.name, resp.content_length, index) as pbar:
                        async for chunk, is_end in resp.content.iter_chunks():
                            fd.write(chunk)
                            pbar.update(len(chunk))
                target_file.with_suffix(".part").rename(target_file)


async def download_models(
    destination: Path,
    models: set[ModelResource],
    verbose=False,
    dry_run=False,
    retry_attempts=5,
    continue_on_error=False,
    parallel_downloads=4,
):
    verbose = verbose or dry_run

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as client:
        tasks: list[asyncio.Task | None] = [None for _ in range(parallel_downloads)]
        for model in sorted(models, key=lambda m: m.name):
            if verbose:
                print(f"\n{model.name}")

            if not any(t is None or t.done() for t in tasks):
                await asyncio.wait([t for t in tasks if t], return_when=asyncio.FIRST_COMPLETED)

            tasks = [None if t is None or t.done() else t for t in tasks]
            index = tasks.index(None)
            download = download_with_retry(
                client,
                model,
                destination,
                verbose,
                dry_run,
                retry_attempts,
                continue_on_error,
                index,
            )
            tasks[index] = asyncio.create_task(download)
        await asyncio.gather(*[t for t in tasks if t is not None])


def verify_models(path: Path, models: set[ModelResource]):
    for model in models:
        for status in model.verify(path):
            if status.state is VerificationState.in_progress:
                print(f"Verifying {status.file.name}...", end="")
            elif status.state is VerificationState.verified:
                print(" OK")
            elif status.state is VerificationState.mismatch:
                print(" MISMATCH")
                print(f"  {status.file.path}")
                print(f"  Expected SHA256: {status.file.sha256}")
                print(f"  Actual SHA256:   {status.info}")
            else:
                print(f" ERROR ({status.state.name})")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="download_models.py",
        usage="%(prog)s [options] destination",
        description=(
            "Script which downloads required & optional models to run a ComfyUI"
            " server for the Krita Generative AI plugin."
        ),
    )
    parser.add_argument(
        "destination",
        type=Path,
        default=Path.cwd(),
        help=(
            "Path to the directory where the models should be downloaded. This can be your ComfyUI"
            " directory to place files directly, or you can specify an empty directory and copy the"
            " files manually."
        ),
    )
    checkpoint_names = [m.id.identifier for m in resources.default_checkpoints]
    # fmt: off
    parser.add_argument("-v", "--verbose", action="store_true", help="print URLs and filepaths")
    parser.add_argument("-l", "--list", action="store_true", help="list models to be downloaded and exit")
    parser.add_argument("-d", "--dry-run", action="store_true", help="don't actually download anything (but create directories)")
    parser.add_argument("-c", "--check", action="store_true", help="verify model integrity")
    parser.add_argument("-m", "--minimal", action="store_true", help="download the minimum viable set of models")
    parser.add_argument("-r", "--recommended", action="store_true", help="download a recommended set of models")
    parser.add_argument("-a", "--all", action="store_true", help="download ALL models")
    parser.add_argument("--sd15", action="store_true", help="[Workload] everything needed to run SD 1.5 (no checkpoints)")
    parser.add_argument("--sdxl", action="store_true", help="[Workload] everything needed to run SDXL (no checkpoints)")
    parser.add_argument("--illu", action="store_true", help="[Workload] everything needed to run Illustrious-SDXL (no checkpoints)")
    parser.add_argument("--flux", action="store_true", help="[Workload] everything needed to run Flux (no checkpoints)")
    parser.add_argument("--flux2", action="store_true", help="[Workload] everything needed to run Flux 2 (no checkpoints)")
    parser.add_argument("--zimage", action="store_true", help="[Workload] everything needed to run Z-Image (no checkpoints)")
    parser.add_argument("--checkpoints", action="store_true", dest="checkpoints", help="download all checkpoints for selected workloads")
    parser.add_argument("--controlnet", action="store_true", help="download ControlNet models for selected workloads")
    parser.add_argument("--checkpoint", action="append", choices=checkpoint_names, dest="checkpoint_list", help="download a specific checkpoint (can specify multiple times)")
    parser.add_argument("--upscalers", action="store_true", help="download additional upscale models")
    parser.add_argument("--prefetch", action="store_true", help="download models which would be automatically downloaded on first use")
    parser.add_argument("--deprecated", action="store_true", help="download old models which will be removed in the near future")
    parser.add_argument("--retry-attempts", type=int, default=5, metavar="N", help="number of retry attempts for downloading a model")
    parser.add_argument("--continue-on-error", action="store_true", help="continue downloading models even if an error occurs")
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "cuda_fp4", "xpu", "rocm", "mps"], default="auto", help="filter models for specific hardware")
    parser.add_argument("-j", "--jobs", type=int, default=4, metavar="N", help="number of parallel downloads")
    # fmt: on
    args = parser.parse_args()
    checkpoints = args.checkpoint_list or []
    if args.checkpoints and args.sd15:
        checkpoints += [m.id.identifier for m in default_checkpoints if m.arch is Arch.sd15]
    if args.checkpoints and args.sdxl:
        checkpoints += [m.id.identifier for m in default_checkpoints if m.arch is Arch.sdxl]
    if args.checkpoints and args.flux:
        checkpoints += [m.id.identifier for m in default_checkpoints if m.arch is Arch.flux]
    if args.checkpoints and args.flux2:
        checkpoints += [m.id.identifier for m in default_checkpoints if m.arch is Arch.flux2_4b]
    if args.checkpoints and args.illu:
        checkpoints += [m.id.identifier for m in default_checkpoints if m.arch is Arch.illu]
        checkpoints += [m.id.identifier for m in default_checkpoints if m.arch is Arch.illu_v]
    if args.checkpoints and args.zimage:
        checkpoints += [m.id.identifier for m in default_checkpoints if m.arch is Arch.zimage]

    print(f"Generative AI for Krita - Model download - v{resources.version}")

    backend = ModelRequirements.no_cuda
    if args.backend == "auto":
        backend = detect_backend()
    elif args.backend == "cuda":
        backend = ModelRequirements.cuda
    elif args.backend == "cuda_fp4":
        backend = ModelRequirements.cuda_fp4

    models = list_models(
        sd15=args.sd15,
        sdxl=args.sdxl,
        flux=args.flux,
        flux2=args.flux2,
        illu=args.illu,
        zimage=args.zimage,
        upscalers=args.upscalers,
        checkpoints=checkpoints,
        controlnet=args.controlnet,
        prefetch=args.prefetch,
        deprecated=args.deprecated,
        minimal=args.minimal,
        recommended=args.recommended,
        all=args.all,
        backend=backend,
    )
    if args.list:
        for model in sorted(models, key=lambda m: m.name):
            for file in model.files:
                print(f"{model.name:32} {file.name}")
        sys.exit(0)

    asyncio.run(
        download_models(
            args.destination,
            models,
            verbose=args.verbose,
            dry_run=args.dry_run,
            retry_attempts=args.retry_attempts,
            continue_on_error=args.continue_on_error,
            parallel_downloads=args.jobs,
        )
    )
    if args.check:
        verify_models(args.destination, models)
