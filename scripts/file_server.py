"""Simple HTTP server for testing the installation process.
1) Run the docker.py script to download all required models.
2) Run this script to serve the model files on localhost.
3) Set environment variable HOSTMAP=1 to replace all huggingface / civitai urls.
"""

import http.server
from itertools import chain
import socketserver
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from ai_diffusion import resources

port = int(sys.argv[1]) if len(sys.argv) > 1 else 51222
dir = Path(__file__).parent / "docker"


def url_strip_host(url: str):
    return "/" + url.split("/", 3)[-1]


def get_path(m: resources.ModelResource):
    if m.kind is resources.ResourceKind.ip_adapter:
        return dir / "ip-adapter/" / m.filename
    else:
        return dir / m.folder / m.filename


models = chain(
    resources.required_models,
    resources.optional_models,
    resources.default_checkpoints,
    resources.upscale_models,
)
files = {url_strip_host(m.url): get_path(m) for m in models}

print("Serving files:")
for url, path in files.items():
    print(f"- {url} -> {path}")


def send_file(http: http.server.SimpleHTTPRequestHandler, filepath: Path):
    http.send_response(200)
    http.send_header("Content-type", "application/octet-stream")
    http.send_header("Content-Disposition", f"attachment; filename={filepath.name}")
    http.send_header("Content-Length", str(os.path.getsize(filepath)))
    http.end_headers()
    with open(filepath, "rb") as f:
        http.wfile.write(f.read())


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        file = files.get(self.path, None)
        if file and file.exists():
            print(f"Sending {file}")
            send_file(self, file)
        elif file:
            print(f"File not found: {file}")
            super().do_GET()
        else:
            print(f"File not found: {self.path}")
            super().do_GET()


with socketserver.TCPServer(("", port), Handler) as httpd:
    print(f"Serving files at http://localhost:{port}")
    httpd.serve_forever()
