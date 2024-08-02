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
from itertools import chain, islice
import aiohttp
import sys
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append(str(Path(__file__).parent.parent))
import ai_diffusion
from ai_diffusion import resources
from ai_diffusion.resources import SDVersion, all_models

version = f"v{ai_diffusion.__version__}"


def required_models():
    return chain(resources.required_models, islice(resources.default_checkpoints, 1))


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
    for filepath, url in model.files.items():
        target_file = destination / filepath
        if verbose:
            print(f"Looking for {target_file}")
        if target_file.exists():
            print(f"{model.name}: found - skipping")
            continue
        if verbose:
            print(f"Downloading {url}")
        if not dry_run:
            target_file.parent.mkdir(exist_ok=True, parents=True)
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
    no_sd15=False,
    no_sdxl=False,
    no_upscalers=False,
    no_checkpoints=False,
    default_checkpoints=[],
    no_controlnet=False,
    no_inpaint=False,
    prefetch=False,
    minimal=False,
):
    print(f"Generative AI for Krita - Model download - v{ai_diffusion.__version__}")
    verbose = verbose or dry_run
    models = required_models() if minimal else all_models()

    for checkpoint in default_checkpoints or []:
        if checkpoint not in [model.id.identifier for model in models]:
            raise ValueError(f"Invalid checkpoint identifier: {checkpoint}")

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)
    async with aiohttp.ClientSession(timeout=timeout) as client:
        for model in models:
            if (
                (no_sd15 and model.sd_version is SDVersion.sd15)
                or (no_sdxl and model.sd_version is SDVersion.sdxl)
                or (no_controlnet and model.kind is resources.ResourceKind.controlnet)
                or (no_upscalers and model.kind is resources.ResourceKind.upscaler)
                or (no_checkpoints and model.kind is resources.ResourceKind.checkpoint)
                or (
                    default_checkpoints
                    and model.kind is resources.ResourceKind.checkpoint
                    and model.id.identifier not in default_checkpoints
                )
                or (no_inpaint and model.kind is resources.ResourceKind.inpaint)
                or (not prefetch and model.kind is resources.ResourceKind.preprocessor)
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
    parser.add_argument("--checkpoint", action="append", dest="default_checkpoints", help="the identifier of the default checkpoint to download") # fmt: skip
    parser.add_argument("--no-upscalers", action="store_true", help="skip upscale models")
    parser.add_argument("--no-controlnet", action="store_true", help="skip ControlNet models")
    parser.add_argument("--no-inpaint", action="store_true", help="skip inpaint models")
    parser.add_argument("--prefetch", action="store_true", help="fetch models which would be automatically downloaded on first use") # fmt: skip
    parser.add_argument("-m", "--minimal", action="store_true", help="minimum viable set of models")
    args = parser.parse_args()
    args.no_sdxl = args.no_sdxl or args.minimal
    asyncio.run(
        main(
            args.destination,
            args.verbose,
            args.dry_run,
            args.no_sd15,
            args.no_sdxl,
            args.no_upscalers,
            args.no_checkpoints,
            args.default_checkpoints,
            args.no_controlnet,
            args.no_inpaint,
            args.prefetch,
            args.minimal,
        )
    )
