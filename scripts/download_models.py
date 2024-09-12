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
import aiohttp
import os
import sys
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append(str(Path(__file__).parent.parent))
from ai_diffusion import resources
from ai_diffusion.resources import Arch, ResourceKind
from ai_diffusion.resources import required_models, default_checkpoints, optional_models


def _progress(name: str, size: int | None):
    return tqdm(
        total=size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=name,
    )


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
):
    for attempt in range(retry_attempts):
        try:
            await download(client, model, destination, verbose, dry_run)
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
):
    for filepath, url in model.files.items():
        target_file = destination / filepath
        url = _map_url(url)
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
                with open(target_file.with_suffix(".part"), "wb") as fd:
                    with _progress(model.name, resp.content_length) as pbar:
                        async for chunk, is_end in resp.content.iter_chunks():
                            fd.write(chunk)
                            pbar.update(len(chunk))
                target_file.with_suffix(".part").rename(target_file)


async def main(
    destination: Path,
    verbose=False,
    dry_run=False,
    sd15=False,
    sdxl=False,
    flux=False,
    upscalers=False,
    checkpoints=[],
    controlnet=False,
    prefetch=False,
    minimal=False,
    recommended=False,
    all=False,
    retry_attempts=5,
    continue_on_error=False,
):
    print(f"Generative AI for Krita - Model download - v{resources.version}")
    verbose = verbose or dry_run
    assert (
        sum([minimal, recommended, all]) <= 1
    ), "Only one of --minimal, --recommended, --all can be specified"

    versions = [Arch.all]
    if sd15 or minimal or all:
        versions.append(Arch.sd15)
    if sdxl or recommended or all:
        versions.append(Arch.sdxl)
    if flux:
        versions.append(Arch.flux)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as client:
        models = set()
        models.update([m for m in default_checkpoints if all or (m.id.identifier in checkpoints)])
        if minimal or recommended or all or sd15 or sdxl:
            models.update([m for m in required_models if m.arch in versions])
        if minimal:
            models.add(default_checkpoints[0])
        if recommended:
            models.update([m for m in default_checkpoints if m.arch is Arch.sdxl])
        if upscalers or recommended or all:
            models.update([m for m in required_models if m.kind is ResourceKind.upscaler])
            models.update(resources.upscale_models)
        if controlnet or recommended or all:
            kinds = [ResourceKind.controlnet, ResourceKind.ip_adapter]
            models.update([m for m in optional_models if m.kind in kinds and m.arch in versions])
        if prefetch or all:
            models.update(resources.prefetch_models)

        if len(models) == 0:
            print("\nNo models selected for download.")

        for model in sorted(models, key=lambda m: m.name):
            if verbose:
                print(f"\n{model.name}")
            await download_with_retry(
                client, model, destination, verbose, dry_run, retry_attempts, continue_on_error
            )


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
    parser.add_argument("-d", "--dry-run", action="store_true", help="don't actually download anything (but create directories)")
    parser.add_argument("-m", "--minimal", action="store_true", help="download the minimum viable set of models")
    parser.add_argument("-r", "--recommended", action="store_true", help="download a recommended set of models")
    parser.add_argument("-a", "--all", action="store_true", help="download ALL models")
    parser.add_argument("--sd15", action="store_true", help="[Workload] everything needed to run SD 1.5 (no checkpoints)")
    parser.add_argument("--sdxl", action="store_true", help="[Workload] everything needed to run SDXL (no checkpoints)")
    parser.add_argument("--flux", action="store_true", help="[Workload] everything needed to run Flux (no checkpoints)")
    parser.add_argument("--checkpoints", action="store_true", dest="checkpoints", help="download all checkpoints for selected workloads")
    parser.add_argument("--controlnet", action="store_true", help="download ControlNet models for selected workloads")
    parser.add_argument("--checkpoint", action="append", choices=checkpoint_names, dest="checkpoint_list", help="download a specific checkpoint (can specify multiple times)")
    parser.add_argument("--upscalers", action="store_true", help="download additional upscale models")
    parser.add_argument("--prefetch", action="store_true", help="download models which would be automatically downloaded on first use")
    parser.add_argument("--retry-attempts", type=int, default=5, metavar="N", help="number of retry attempts for downloading a model")
    parser.add_argument("--continue-on-error", action="store_true", help="continue downloading models even if an error occurs")
    # fmt: on
    args = parser.parse_args()
    checkpoints = args.checkpoint_list or []
    if args.checkpoints and args.sd15:
        checkpoints += [m.id.identifier for m in default_checkpoints if m.arch is Arch.sd15]
    if args.checkpoints and args.sdxl:
        checkpoints += [m.id.identifier for m in default_checkpoints if m.arch is Arch.sdxl]
    if args.checkpoints and args.flux:
        checkpoints += [m.id.identifier for m in default_checkpoints if m.arch is Arch.flux]
    asyncio.run(
        main(
            destination=args.destination,
            verbose=args.verbose,
            dry_run=args.dry_run,
            sd15=args.sd15,
            sdxl=args.sdxl,
            flux=args.flux,
            upscalers=args.upscalers,
            checkpoints=checkpoints,
            controlnet=args.controlnet,
            prefetch=args.prefetch,
            minimal=args.minimal,
            recommended=args.recommended,
            all=args.all,
            retry_attempts=args.retry_attempts,
            continue_on_error=args.continue_on_error,
        )
    )
