# ComfyUI Server for Krita plugin - runpod.io Docker image

This image provides Stable Diffusion with all the prerequisites for use in Krita painting software via [this plugin](https://github.com/Acly/krita-ai-diffusion).

## Contents

* [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager.git)
* Required custom nodes (ControlNet aux, IP-adapter, etc.)
* Required models (Clip Vision, IP-Adapter, etc.)
* Stable Diffusion 1.5 checkpoints (RealisticVision, DreamShaper)
* ControlNet models for SD 1.5

Extensions and models are kept in sync with the local installer for the Krita plugin, listed [here](https://github.com/Acly/krita-ai-diffusion#optional-custom-comfyui-server)

## System

* Ubuntu 22.04 LTS
* CUDA 11.8
* Python 3.10.12

## Utilities

* [runpodctl](https://github.com/runpod/runpodctl)
* [croc](https://github.com/schollz/croc)
* [rclone](https://rclone.org/)

## Template Requirements

| Port | Type (HTTP/TCP) | Function     |
|------|-----------------|--------------|
| 22   | TCP             | SSH          |
| 3001 | HTTP            | Comfy Web UI |
| 8888 | HTTP            | Jupyter Lab  |

## Acknowledgements

The docker image is based on:
* https://github.com/ashleykleynhans/stable-diffusion-docker
* https://github.com/runpod/containers
