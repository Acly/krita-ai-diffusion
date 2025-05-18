import asyncio
import aiohttp
import sys
import dotenv
import os
import subprocess
from markdown import markdown
from shutil import rmtree, copy, copytree, ignore_patterns, make_archive
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import ai_diffusion
from ai_diffusion.resources import update_model_checksums

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
    subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "scripts/server_requirements.in",
            "--no-deps",
            "--no-annotate",
            "--universal",
            "--upgrade",
            "--quiet",
            "-o",
            "ai_diffusion/server_requirements.txt",
        ],
        cwd=root,
        check=True,
    )


def precheck():
    translation.update_template()
    translation.update_all()

    update_server_requirements()
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


async def publish_package(package_path: Path, target: str):
    dotenv.load_dotenv(root / "service" / "web" / ".env.local")
    service_url = os.environ["TEST_SERVICE_URL"]
    if target == "production":
        service_url = "https://api.interstice.cloud"
    service_token = os.environ["INTERSTICE_INFRA_TOKEN"]
    headers = {"Authorization": f"Bearer {service_token}"}

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
