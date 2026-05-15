import asyncio
import os
import subprocess
import sys
from pathlib import Path
from shutil import copy, copytree, ignore_patterns, make_archive, rmtree
from typing import NamedTuple

import aiohttp
from markdown import markdown

sys.path.append(str(Path(__file__).parent.parent))
import ai_diffusion
from ai_diffusion.backend.resources import update_model_checksums

sys.path.append(str(Path(__file__).parent))
import translation

root = Path(__file__).parent.parent
package_dir = root / "scripts" / ".package"
version = ai_diffusion.__version__
package_name = f"krita_ai_diffusion-{version}"


def convert_markdown_to_html(markdown_file: Path, html_file: Path):
    with open(markdown_file, "r", encoding="utf-8") as f:
        text = f.read()
    html = markdown(text, extensions=["fenced_code", "codehilite"])
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)


def update_server_requirements():
    class Cfg(NamedTuple):
        platform: str
        extra_index: str | None = None
        override: str | None = None
        dependencies: str | None = None

    req_dir = root / "ai_diffusion" / "backend" / "requirements"
    configs = {
        "linux-cpu": Cfg("x86_64-unknown-linux-gnu", "cpu"),
        "linux-cuda": Cfg("x86_64-unknown-linux-gnu", "cu128"),
        "linux-cuda126": Cfg("x86_64-unknown-linux-gnu", "cu126"),
        "linux-xpu": Cfg("x86_64-unknown-linux-gnu", "xpu"),
        "linux-rocm": Cfg("x86_64-unknown-linux-gnu", "rocm7.2"),
        "macos": Cfg("aarch64-apple-darwin"),
        "windows-cpu": Cfg("x86_64-pc-windows-msvc", "cpu"),
        "windows-cuda": Cfg("x86_64-pc-windows-msvc", "cu128"),
        "windows-cuda126": Cfg("x86_64-pc-windows-msvc", "cu126"),
        "windows-xpu": Cfg("x86_64-pc-windows-msvc", "xpu"),
        "windows-rocm": Cfg(
            "x86_64-pc-windows-msvc", None, "rocm-windows.in", "rocm-windows-deps.in"
        ),
    }
    for name, cfg in configs.items():
        cmd = ["uv", "pip", "compile", str((req_dir / "base.in").relative_to(root))]
        if additional_reqs := cfg.dependencies:
            cmd += [str((req_dir / additional_reqs).relative_to(root))]
        cmd += ["--emit-index-annotation", "--emit-index-url"]
        cmd += ["--index-strategy", "unsafe-best-match"]
        cmd += ["--python-platform", cfg.platform, "--python-version", "3.12"]
        if override := cfg.override:
            cmd += ["--override", str((req_dir / override).relative_to(root))]
        cmd += ["--index-url", "https://pypi.org/simple"]
        if extra_index := cfg.extra_index:
            cmd += ["--extra-index-url", f"https://download.pytorch.org/whl/{extra_index}"]
        cmd += ["--quiet"]
        cmd += ["-o", str((req_dir / f"{name}.txt").relative_to(root))]
        print(f"{name}.txt")
        subprocess.run(cmd, cwd=root, check=True)


def precheck():
    translation.update_template()
    translation.update_all()

    update_model_checksums(root / "scripts" / "downloads")


def build_package():
    precheck()

    rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir()
    copy(root / "ai_diffusion.desktop", package_dir)

    plugin_src = root / "ai_diffusion"
    plugin_dst = package_dir / "ai_diffusion"

    def ignore(path, names):
        return ignore_patterns(".*", "*.pyc", "__pycache__", "debugpy")(path, names)

    copytree(plugin_src, plugin_dst, ignore=ignore)
    copy(root / "scripts" / "download_models.py", plugin_dst)
    copy(root / "LICENSE", plugin_dst)
    convert_markdown_to_html(root / "README.md", plugin_dst / "manual.html")

    make_archive(str(root / package_name), "zip", package_dir)

    # Do this afterwards to not include untested changes in the package
    # Option 1: test the dependency changes and do another package build
    # Option 2: revert the dependency changes, keep stable version for now
    update_server_requirements()


async def publish_package(package_path: Path, target: str):
    from service.pod.lib.environment import Config  # type: ignore

    config = Config.from_env()
    service_url = os.environ.get("TEST_SERVICE_URL", "http://localhost:8787")
    if target == "production":
        service_url = "https://api.interstice.cloud"
    headers = {"Authorization": f"Bearer {config.secrets.interstice_infra_token}"}

    archive_data = package_path.read_bytes()
    async with aiohttp.ClientSession(service_url, headers=headers) as session:
        print("Uploading package to", service_url)
        async with session.put(f"/plugin/upload/{version}", data=archive_data) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Failed to upload package: {response.status}", await response.text()
                )
            uploaded = await response.json()
            for key, value in uploaded.items():
                print(f"{key}: {value}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"

    if cmd == "build":
        print("Building package", root / package_name)
        build_package()

    elif cmd == "publish":
        target = sys.argv[2] if len(sys.argv) > 2 else "production"
        package = root / f"{package_name}.zip"
        print("Publishing package", str(package))
        asyncio.run(publish_package(package, target))

    elif cmd == "check":
        print("Performing precheck without building")
        precheck()

    elif cmd == "update":
        print("Updating server requirements without building")
        update_server_requirements()
