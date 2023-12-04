"""Utility script for building a docker image which runs
ComfyUI with all required custom nodes and models.
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import ai_diffusion
from scripts.download_models import all_models, main as download_models

version = f"v{ai_diffusion.__version__}"
docker_root = Path(__file__).parent / "docker"


def clean(models):
    expected = set(m.folder / m.filename for m in models)
    for path in (docker_root / "models").glob("**/*"):
        if path.is_file() and path.relative_to(docker_root) not in expected:
            print(f"- Deleting {path}")
            path.unlink()


async def main():
    print("Cleaning up cached models")
    clean(all_models())

    print("Downloading new models")
    await download_models(docker_root)

    print("Preparation complete.\n\nTo build run:")
    print(f"  docker build -t aclysia/sd-comfyui-krita:{version} scripts/docker")
    print("\nTo test the image:")
    print(f"  docker run --gpus all -p 3001:3000 -p 8888:8888 aclysia/sd-comfyui-krita:{version}")


if __name__ == "__main__":
    asyncio.run(main())
