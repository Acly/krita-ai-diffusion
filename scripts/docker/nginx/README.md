# Docker image for runpod.io

Adapted from
* https://github.com/ashleykleynhans/stable-diffusion-docker
* https://github.com/runpod/containers

## Template Requirements

| Port | Type (HTTP/TCP) | Function     |
|------|-----------------|--------------|
| 22   | TCP             | SSH          |
| 3001 | HTTP            | Comfy Web UI |
| 8888 | HTTP            | Jupyter Lab  |