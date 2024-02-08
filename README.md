<h1><img width="64px" src="ai_diffusion/icons/logo-128.png"> Generative AI <i>for Krita</i></h1>

✨[Features](#features) | ⭳ [Download](https://github.com/Acly/krita-ai-diffusion/releases/latest) | 🛠️[Installation](#installation) | 🎞️ [Video](https://youtu.be/Ly6USRwTHe0) | 🖼️[Screenshots](#screenshots) | 📖[Wiki](https://github.com/Acly/krita-ai-diffusion/wiki) | 💬[Discussion](https://github.com/Acly/krita-ai-diffusion/discussions)

Generate images from within Krita with minimal fuss: Select an area, push a button,
and new content that matches your image will be generated. Or expand your canvas and
fill new areas with generated content that blends right in. Text prompts are optional.
No tweaking required!

This plugin seeks to provide what "Generative Fill/Expand" do in Photoshop - and go beyond.
Adjust strength to refine existing content _(img2img)_ or generate images from scratch.
Powerful customization is available for advanced users.

_Local. Open source. Free._

[![Watch video demo](media/screenshot-video-preview.webp)](https://youtu.be/Ly6USRwTHe0 "Watch video demo")

## <a name="features"></a> Features

Features are designed to fit an interactive workflow where AI generation is used as just another
tool while painting. They are meant to synergize with traditional tools and the layer stack.

* **Inpaint**: Use Krita's selection tools to mark an area and remove or replace existing content in the image. Simple text prompts can be used to steer generation.
* **Outpaint**: Extend your canvas, select a blank area and automatically fill it with content that seamlessly blends into the existing image.
* **Generate**: Create new images from scratch by decribing them with words or existing images. Supports SD1.5 and SDXL.
* **Refine**: Use the strength slider to refine existing image content instead of replacing it entirely. This also works great for adding new things to an image by painting a (crude) approximation and refining at high strength!
* **Live Painting**: Let AI interpret your canvas in real time for immediate feedback. [Watch Video](https://youtu.be/AF2VyqSApjA?si=Ve5uQJWcNOATtABU)
* **Control**: Guide image creation directly with sketches or line art. Use depth or normal maps from existing images or 3D scenes. Transfer character pose from snapshots. Control composition with segmentation maps.
* **Resolutions**: Work efficiently at any resolution. The plugin will automatically use resolutions appropriate for the AI model, and scale them to fit your image region.
* **Upscaling**: Upscale and enrich images to 4k, 8k and beyond without running out of memory.
* **Job Queue**: Depending on hardware, image generation can take some time. The plugin allows you to queue and cancel jobs while working on your image.
* **History**: Not every image will turn out a masterpiece. Preview results and browse previous generations and prompts at any time.
* **Strong Defaults**: Versatile default style presets allow for a simple UI which covers many scenarios.
* **Customization**: Create your own presets - select a Stable Diffusion checkpoint, add LoRA, tweak samplers and more.

## <a name="installation"></a> Getting Started

The plugin comes with an integrated installer for the Stable Diffusion backend.

### Requirements

* Windows, Linux, MacOS
* _On Linux/Mac:_ Python + venv must be installed
    * recommended version: 3.11 or 3.10
    * usually available via package manager, eg. `apt install python3-venv`

#### Hardware support

To run locally a powerful graphics card with at least 6 GB VRAM is recommended. Otherwise generating images will take very long!

<table>
<tr><td>NVIDIA GPU</td><td>supported via CUDA</td></tr>
<tr><td>AMD GPU</td><td>supported via DirectML on Windows, ROCm on Linux (only custom server)</td></tr>
<tr><td>Apple M1/M2</td><td>supported via MPS on macOS</td></tr>
<tr><td>CPU</td><td>supported, but very slow</td></tr>
<tr><td>Cloud GPU</td><td>supported, rent a GPU on an hourly basis, see <a href="#gpu-cloud">below</a></td></tr>
</table>

### Installation

1. If you haven't yet, go and install [Krita](https://krita.org/)! _Required version: 5.2.0 or newer_
1. [Download the plugin](https://github.com/Acly/krita-ai-diffusion/releases/latest). Unpack the archive into your `pykrita` folder.
    * _Windows:_ Usually `C:\Users\<user>\AppData\Roaming\krita\pykrita`
    * _Linux:_ Usually `~/.local/share/krita/pykrita`
    * _MacOS:_ Usually `~/Library/Application Support/krita/pykrita`
    * Check [Krita's official documentation](https://docs.krita.org/en/user_manual/python_scripting/install_custom_python_plugin.html) if you have trouble locating it.
1. Enable the plugin in Krita (Settings ‣ Configure Krita ‣ Python Plugins Manager) and restart.
1. Create a new document or open an existing image.
1. To show the plugin docker: Settings ‣ Dockers ‣ AI Image Generation.
1. In the plugin docker, click "Configure" to start server installation. _Requires 10+ GB free disk space._

> [!NOTE]
> If you encounter problems please check the [FAQ / list of common issues](https://github.com/Acly/krita-ai-diffusion/wiki/Common-Issues) for solutions.

### _Optional:_ Custom ComfyUI Server

The plugin uses [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as backend. As an alternative to the automatic installation,
you can install it manually or use an existing installation. If the server is already running locally before starting Krita, the plugin will
automatically try to connect. Using a remote server is also possible this way.

Please check the list of [required extensions and models](https://github.com/Acly/krita-ai-diffusion/wiki/ComfyUI-Setup) to make sure your installation is compatible.

### _Optional:_ Object selection tools (Segmentation)

If you're looking for a way to easily select objects in the image, there is a [separate plugin](https://github.com/Acly/krita-ai-tools) which adds AI segmentation tools.

### GPU Cloud

You can also rent a GPU instead of running locally. In that case, step 6 is not needed. Instead use the plugin to connect to a remote server.

There is a [step by step guide](https://github.com/Acly/krita-ai-diffusion/wiki/Cloud-GPU) on how to setup cloud GPU on [runpod.io](https://www.runpod.io) or [vast.ai](https://vast.ai).


## <a name="screenshots"></a> Screenshots

_Inpainting on a photo using a realistic model_
<img src="media/screenshot-2.png">

_Reworking and adding content to an AI generated image_
<img src="media/screenshot-1.png">

_Adding detail and iteratively refining small parts of the image_
<img src="media/screenshot-3.png">

_Using ControlNet to guide image generation with a crude scribble_
<img src="media/screenshot-4.png">

_Modifying the pose vector layer to control character stances (Click for video)_
[![Watch video demo](media/screenshot-5.png)](https://youtu.be/-QDPEcVmdLI "Watch video demo")

_Upscaling to improve image quality and add details_
<img src="media/screenshot-6.png">

_Server installation_
<img src="media/screenshot-installation.png">

_Style preset configuration_
<img src="media/screenshot-style.png">

## Contributing

Contributions are very welcome! Check the [contributing guide](CONTRIBUTING.md) to get started.

## Technology

* Image generation: [Stable Diffusion](https://github.com/Stability-AI/generative-models)
* Diffusion backend: [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* Inpainting: [ControlNet](https://github.com/lllyasviel/ControlNet), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)