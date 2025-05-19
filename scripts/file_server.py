"""Simple HTTP server for testing the installation process.
1) Run the download_models.py script to download all required models.
2) Run this script to serve the model files on localhost.
3) Set environment variable HOSTMAP=1 to replace all huggingface / civitai urls.
"""

import sys
from aiohttp import web
from pathlib import Path
from urllib.parse import unquote as url_unquote

sys.path.append(str(Path(__file__).parent.parent))
from ai_diffusion import resources

dir = Path(__file__).parent / "downloads"


def url_strip(url: str):
    without_host = "/" + url.split("/", 3)[-1]
    without_query = without_host.split("?", 1)[0]
    return without_query


files = {
    url_unquote(url_strip(file.url)): dir / file.path
    for m in resources.all_models(include_deprecated=True)
    for file in m.files
}

urls = [
    url_strip(file.url) for m in resources.all_models(include_deprecated=True) for file in m.files
]


async def file_sender(file: Path):
    with open(file, "rb") as f:
        chunk = f.read(2**14)
        while chunk:
            yield chunk
            chunk = f.read(2**14)


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
