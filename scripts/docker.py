"""Utility script for building a docker image which runs
ComfyUI with all required custom nodes and models.
"""

import asyncio
import sys
import shutil
import requests
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import ai_diffusion
from ai_diffusion import resources
from scripts.download_models import all_models, main as download_models

version = f"v{ai_diffusion.__version__}"
docker_root = Path(__file__).parent / "docker"
download_folder = docker_root / "downloads"
comfy_dir = docker_root / "ComfyUI"


def download_repository(url: str, target: Path, revision):
    if not target.exists():
        archive = target.parent / f"{target.name}.zip"
        if not url.endswith(".zip"):  # git repo URL
            url = f"{url}/archive/{revision}.zip"
        print("Downloading", url)
        with open(archive, "wb") as f:
            f.write(requests.get(url, allow_redirects=True).content)
        shutil.unpack_archive(archive, target.parent)
        archive.unlink()
        shutil.move(target.parent / f"{target.name}-{revision}", target)


def download_repositories():
    download_repository(resources.comfy_url, comfy_dir, resources.comfy_version)
    custom_nodes_dir = comfy_dir / "custom_nodes"
    for repo in resources.required_custom_nodes:
        download_repository(repo.url, custom_nodes_dir / repo.folder, repo.version)


def clean(models):
    expected = set(filepath for m in models for filepath in m.files.keys())
    for path in (download_folder).glob("**/*"):
        if path.is_file() and path.relative_to(download_folder) not in expected:
            print(f"- Deleting {path}")
            path.unlink()


async def main():
    print("Downloading repositories")
    download_repositories()

    print("Cleaning up cached models")
    clean(all_models())

    print("Downloading new models")
    await download_models(download_folder)

    print("Preparation complete.\n\nTo build run:")
    print(f"  docker build -t aclysia/sd-comfyui-krita:{version} scripts/docker")
    print("\nTo test the image:")
    print(f"  docker run --gpus all -p 3001:3000 -p 8888:8888 aclysia/sd-comfyui-krita:{version}")


if __name__ == "__main__":
    asyncio.run(main())
