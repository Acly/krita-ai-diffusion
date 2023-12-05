"""Simple HTTP server for testing the installation process.
1) Run the download_models.py script to download all required models.
2) Run this script to serve the model files on localhost.
3) Set environment variable HOSTMAP=1 to replace all huggingface / civitai urls.
"""

from aiohttp import web
from itertools import chain
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from ai_diffusion import resources

dir = Path(__file__).parent / "docker" / "downloads"


def url_strip(url: str):
    without_host = "/" + url.split("/", 3)[-1]
    without_query = without_host.split("?", 1)[0]
    return without_query


def get_path(m: resources.ModelResource):
    return dir / m.folder / m.filename


models = chain(
    resources.required_models,
    resources.optional_models,
    resources.default_checkpoints,
    resources.upscale_models,
)
files = {url_strip(m.url): get_path(m) for m in models}


async def handle(request: web.Request):
    print(f"Request: {request.path}?{request.query_string}")
    file = files.get(request.path, None)
    if file and file.exists():
        print(f"Sending {file}")
        try:
            return web.FileResponse(file)
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
    app.add_routes([web.get(url, handle) for url in files.keys()])
    web.run_app(app, host="localhost", port=port)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 51222
    run(port)
