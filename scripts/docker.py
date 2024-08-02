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

version = f"v{ai_diffusion.__version__}"
root_dir = Path(__file__).parent.parent
scripts_dir = root_dir / "scripts"
docker_dir = scripts_dir / "docker"
comfy_dir = docker_dir / "ComfyUI"


def copy_scripts():
    plugin_dir = root_dir / "ai_diffusion"
    target_dir = docker_dir / "scripts" / "ai_diffusion"
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in plugin_dir.glob("*.py"):
        shutil.copy(src, target_dir)
    websockets_dir = target_dir / "websockets" / "src"
    websockets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(scripts_dir / "download_models.py", docker_dir / "scripts" / "download_models.py")


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


async def main():
    print("Downloading repositories")
    download_repositories()

    print("Copying scripts")
    copy_scripts()

    print("Preparation complete.\n\nTo build run:")
    print(f"  docker build -t aclysia/sd-comfyui-krita:{version} scripts/docker/")
    print("\nTo test the image:")
    print(f"  docker run --gpus all -p 3001:3000 -p 8888:8888 aclysia/sd-comfyui-krita:{version}")


if __name__ == "__main__":
    asyncio.run(main())
