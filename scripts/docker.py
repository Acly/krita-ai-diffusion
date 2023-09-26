"""Utility script for building a docker image which runs
ComfyUI with all required custom nodes and models.
"""
import asyncio
from itertools import chain
import aiohttp
import sys
import subprocess
from pathlib import Path
import requests

sys.path.append(str(Path(__file__).parent.parent))
import ai_diffusion
from ai_diffusion import resources, network

version = f"v{ai_diffusion.__version__}"
docker_root = Path(__file__).parent / "docker"


def all_models():
    return chain(
        resources.required_models, resources.optional_models, resources.default_checkpoints
    )


def clean(models):
    expected = set(m.folder / m.filename for m in models)
    for path in (docker_root / "models").glob("**/*"):
        if path.is_file() and path.relative_to(docker_root) not in expected:
            print(f"- Deleting {path}")
            path.unlink()


async def download(client, model: resources.ModelResource):
    target_dir = docker_root / model.folder
    if model.kind is resources.ResourceKind.ip_adapter:
        target_dir = docker_root / "ip-adapter"
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
    target_file = target_dir / model.filename
    if target_file.exists():
        print(f"- Found {model.name}, skipping...")
        return
    print(f"+ Downloading {model.name}...")
    async with client.get(model.url) as resp:
        if resp.status == 200:
            with open(target_file, "wb") as fd:
                async for chunk, is_end in resp.content.iter_chunks():
                    fd.write(chunk)


async def main():
    print("Cleaning up cached models")
    clean(all_models())

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)
    async with aiohttp.ClientSession(timeout=timeout) as client:
        # Download all the models. Can be skipped by copying them manually.
        print("Downloading new models")
        for model in all_models():
            await download(client, model)

    print("Preparation complete.\n\nTo build run:")
    print(f"  docker build -t aclysia/sd-comfyui-krita:{version} scripts/docker")
    print("\nTo test the image:")
    print(f"  docker run --gpus all -p 3001:3000 -p 8888:8888 aclysia/sd-comfyui-krita:{version}")


if __name__ == "__main__":
    asyncio.run(main())
