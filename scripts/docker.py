"""Utility script for building a docker image which runs
ComfyUI with all required custom nodes and models.
"""

import asyncio
import sys
import shutil
import subprocess
import requests
from pathlib import Path
from itertools import chain

sys.path.append(str(Path(__file__).parent.parent))
import ai_diffusion
from ai_diffusion import resources

version = f"v{ai_diffusion.__version__}"
root_dir = Path(__file__).parent.parent
docker_dir = root_dir / "scripts" / "docker"
comfy_dir = docker_dir / "ComfyUI"


def copy_scripts():
    repo_dir = docker_dir / "krita-ai-diffusion"
    for source_file, target_dir in [
        (root_dir / "ai_diffusion" / "resources.py", repo_dir / "ai_diffusion"),
        (
            root_dir / "ai_diffusion" / "presets" / "models.json",
            repo_dir / "ai_diffusion" / "presets",
        ),
        (root_dir / "scripts" / "download_models.py", repo_dir / "scripts"),
    ]:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_file, target_dir)


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
    custom_nodes_dir = comfy_dir / "custom_nodes"
    manager_url = "https://github.com/ltdrdata/ComfyUI-Manager"

    download_repository(resources.comfy_url, comfy_dir, resources.comfy_version)
    download_repository(manager_url, custom_nodes_dir / "ComfyUI-Manager", "main")
    for repo in chain(resources.required_custom_nodes, resources.optional_custom_nodes):
        download_repository(repo.url, custom_nodes_dir / repo.folder, repo.version)


def upgrade_python_dependencies():
    cmd = [
        "uv",
        "pip",
        "compile",
        "requirements.in",
        "ComfyUI/requirements.txt",
        "ComfyUI/custom_nodes/comfyui_controlnet_aux/requirements.txt",
        "ComfyUI/custom_nodes/comfyui-tooling-nodes/requirements.txt",
        "ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt",
        "ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt",
        "--no-deps",
        "--upgrade",
        "-o",
        "requirements.txt",
    ]
    subprocess.run(cmd, cwd=docker_dir, check=True)


def check_line_endings():
    for file in docker_dir.rglob("*.sh"):
        with open(file, "rb") as f:
            content = f.read()
        if b"\r\n" in content:
            print(f"Windows line endings detected in {file}, fixing...")
            with open(file, "wb") as f:
                f.write(content.replace(b"\r\n", b"\n"))


async def main(clean=False, upgrade=False):
    if clean:
        print("Deleting existing repositories")
        shutil.rmtree(comfy_dir, ignore_errors=True)
        shutil.rmtree(docker_dir / "krita-ai-diffusion", ignore_errors=True)

    print("Downloading repositories")
    download_repositories()

    if upgrade:
        print("Upgrading Python dependencies")
        upgrade_python_dependencies()

    print("Copying scripts")
    copy_scripts()

    check_line_endings()

    print("Preparation complete.\n\nTo build run:")
    print(f"  docker build -t aclysia/sd-comfyui-krita:{version} scripts/docker/")
    print("\nTo test the image:")
    print(f"  docker run --gpus all -p 3001:3000 -p 8888:8888 aclysia/sd-comfyui-krita:{version}")
    print("\nTo test the image with a local file server:")
    print("  python scripts/file_server.py")
    print(
        f"  docker run --gpus all -p 3001:3000 -p 8888:8888 -e AI_DIFFUSION_DOWNLOAD_URL=http://host.docker.internal:51222 aclysia/sd-comfyui-krita:{version}"
    )
    print("\nTo build the base image:")
    print("  docker build -t aclysia/sd-comfyui-krita:base --target base scripts/docker/")


if __name__ == "__main__":
    asyncio.run(main(clean="--clean" in sys.argv, upgrade="--upgrade" in sys.argv))
