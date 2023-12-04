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
from itertools import chain
import aiohttp
import sys
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append(str(Path(__file__).parent.parent))
import ai_diffusion
from ai_diffusion import resources
from ai_diffusion.resources import SDVersion

version = f"v{ai_diffusion.__version__}"


def all_models():
    return chain(
        resources.required_models,
        resources.optional_models,
        resources.default_checkpoints,
        resources.upscale_models,
    )


def _progress(name: str, size: int | None):
    return tqdm(
        total=size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=name,
    )


async def download(
    client: aiohttp.ClientSession,
    model: resources.ModelResource,
    destination: Path,
    verbose=False,
    dry_run=False,
):
    target_dir = destination / model.folder
    target_file = target_dir / model.filename
    if verbose:
        print(f"Looking for {target_file}")
    if target_file.exists():
        print(f"{model.name}: found - skipping")
        return
    if verbose:
        print(f"Downloading {model.url}")
    if not dry_run:
        if not target_dir.exists():
            target_dir.mkdir(parents=True)
        async with client.get(model.url) as resp:
            resp.raise_for_status()
            with open(target_file, "wb") as fd:
                with _progress(model.name, resp.content_length) as pbar:
                    async for chunk, is_end in resp.content.iter_chunks():
                        fd.write(chunk)
                        pbar.update(len(chunk))


async def main(
    destination: Path,
    verbose=False,
    dry_run=False,
    no_sd15=False,
    no_sdxl=False,
    no_upscalers=False,
    no_checkpoints=False,
    no_controlnet=False,
):
    print(f"Generative AI for Krita - Model download - v{ai_diffusion.__version__}")
    verbose = verbose or dry_run

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)
    async with aiohttp.ClientSession(timeout=timeout) as client:
        for model in all_models():
            if (
                (no_sd15 and model.sd_version is SDVersion.sd15)
                or (no_sdxl and model.sd_version is SDVersion.sdxl)
                or (no_upscalers and model.kind is resources.ResourceKind.upscaler)
                or (no_checkpoints and model.kind is resources.ResourceKind.checkpoint)
                or (no_controlnet and model.kind is resources.ResourceKind.controlnet)
            ):
                continue
            if verbose:
                print(f"\n{model.name}")
            await download(client, model, destination, verbose, dry_run)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="download_models.py",
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
    parser.add_argument("-v", "--verbose", action="store_true", help="print URLs and filepaths")
    parser.add_argument(
        "-d", "--dry-run", action="store_true", help="don't actually download anything"
    )
    parser.add_argument("--no-sd15", action="store_true", help="skip SD1.5 models")
    parser.add_argument("--no-sdxl", action="store_true", help="skip SDXL models")
    parser.add_argument("--no-checkpoints", action="store_true", help="skip default checkpoints")
    parser.add_argument("--no-upscalers", action="store_true", help="skip upscale models")
    parser.add_argument("--no-controlnet", action="store_true", help="skip ControlNet models")
    args = parser.parse_args()
    asyncio.run(
        main(
            args.destination,
            args.verbose,
            args.dry_run,
            args.no_sd15,
            args.no_sdxl,
            args.no_upscalers,
            args.no_checkpoints,
            args.no_controlnet,
        )
    )
