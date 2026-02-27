"""Simple HTTP server for testing the installation process.
1) Run the download_models.py script to download all required models.
2) Run this script to serve the model files on localhost.
3) Set environment variable HOSTMAP=1 to replace all huggingface / civitai urls.
"""

import sys
from pathlib import Path
from urllib.parse import unquote as url_unquote

import anyio
from aiohttp import web

sys.path.append(str(Path(__file__).parent.parent))
from ai_diffusion import resources
from ai_diffusion.resources import Arch, ModelFile, ModelResource, ResourceId, ResourceKind

dir = Path(__file__).parent / "downloads"


def url_strip(url: str):
    without_host = "/" + url.split("/", 3)[-1]
    return without_host.split("?", 1)[0]


all_models = list(resources.all_models(include_deprecated=True))
all_models.append(
    ModelResource(
        name="Qwen3-VL-4B-Instruct",
        id=ResourceId(ResourceKind.checkpoint, Arch.qwen, "qwen3-vl-4b-instruct"),
        files=[
            ModelFile(
                Path("llm/Qwen3-VL-4B-Instruct-IQ4_XS.gguf"),
                "https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3-VL-4B-Instruct-IQ4_XS.gguf",
                ResourceId(ResourceKind.checkpoint, Arch.qwen, "qwen3-vl-4b-instruct"),
                "011018e3aba055a9250f9fb11efdeebfb6805df767922bfdb91607fce866416c",
            ),
            ModelFile(
                Path("llm/mmproj-Qwen3-VL-4B-Instruct-BF16.gguf"),
                "https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/resolve/main/mmproj-BF16.gguf",
                ResourceId(ResourceKind.clip_vision, Arch.qwen, "qwen3-vl-4b-instruct"),
                "364ef9a7502b7e9a834b9c013b953670f6ba2550b4a26fc043c317fa484eb53e",
            ),
        ],
    )
)
files = {url_unquote(url_strip(file.url)): dir / file.path for m in all_models for file in m.files}
urls = {url_strip(file.url) for m in all_models for file in m.files}


async def file_sender(file: Path):
    async with await anyio.open_file(file, "rb") as f:
        chunk = await f.read(2**14)
        while chunk:
            yield chunk
            chunk = await f.read(2**14)


def send_file(file: Path):
    return web.Response(
        headers={
            "Content-disposition": f"attachment; filename={file.name}",
            "Content-length": f"{file.stat().st_size}",
        },
        body=file_sender(file),
    )


async def handle(request: web.Request):
    print(f"Request: {request.path}?{request.query_string}")
    file = files.get(request.path, None)
    if file and file.exists():
        print(f"Sending {file}")
        try:
            return send_file(file)
        except Exception as e:
            print(f"Failed to send {file}: {e}")
            return web.Response(status=500)
    elif file:
        print(f"File not found: {file}")
        return web.Response(status=404)
    else:
        print(f"File not found: {request.path}")
        return web.Response(status=404)


def run(port=51222, verbose=False):
    if verbose:
        print("Serving files:")
        for url, path in files.items():
            print(f"- {url} -> {path}")

    app = web.Application()
    app.add_routes([web.get(url, handle) for url in urls])
    web.run_app(app, host="localhost", port=port)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 51222
    run(port)
