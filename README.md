<div style="display:flex"><div><img width="80" src="ai_diffusion/icons/logo-128.png"></div><h1 style="margin-left:1em">Generative AI<br><i>for Krita</i></h1></div>

[Features](#features) | [Installation](#installation) | Demo

Generate images from within Krita with minimal fuss: Select an area, push a button,
and new content that matches your image will be generated. Or expand your canvas and
fill new areas with generated content that blends right in. Text prompts are optional.
No tweaking required!

This plugin seeks to provide what "Generative Fill/Expand" do in Photoshop - and go beyond.
Adjust strength to refine existing content _(img2img)_ or generate images from scratch.
Powerful customization is available for advanced users.

_Local. Open source. Free._

## <a name="features"></a> Features

Features are designed to fit an interactive workflow where AI generation is used as just another
tool while painting. They are meant to synergize with traditional tools and the layer stack.

* **Inpaint**: Use Krita's selection tools to mark an area and remove or replace existing content in the image. Simple text prompts can be used to steer generation.
* **Outpaint**: Extend your canvas, select a blank area and automatically fill it with content that seamlessly blends into the existing image.
* **Generate**: Create new images from scratch by decribing them with words.
* **Refine**: Use the strength slider to refine existing image content instead of replacing it entirely. This also works great for adding new things to an image by painting a (crude) sketch and refining at high strength!
* **Resolutions**: Work efficiently at any resolution. The plugin will automatically use resolutions appropriate for the AI model, and scale them to fit your image region.
* **Job Queue**: Depending on hardware, image generation can take some time. The plugin allows you to queue and cancel jobs while working on your image.
* **History**: Not every image will turn out a masterpiece. Preview results and browse previous generations and prompts at any time.
* **Strong Defaults**: Versatile default style presets allow for a simple UI which covers many scenarios.
* **Customization**: Create your own presets - select a Stable Diffusion checkpoint, add LoRA, tweak samplers and more.

## <a name="installation"></a> Getting Started

The plugin comes with an automated installer for the Stable Diffusion backend.

### Requirements

* Windows or Linux (MacOS is untested)
* NVIDIA graphics card with at least 4 GB RAM is recommended (CPU is supported, but slow)
* _Linux only:_ Python must be installed (available via package manager)

### Installation

* If you haven't yet, go and install [Krita](https://krita.org/)! _Recommended version: 5.2.0_
* _[TODO: no release yet]_ Download and install this plugin. Like other Krita Python plugins, unpack the archive into your `pykrita` folder.
  * _Windows:_ Usually `C:\Users\<user>\AppData\Roaming\krita\pykrita`
  * _Linux:_ Usually `~/.local/share/krita/pykrita`
  * See [Krita's official documentation](https://docs.krita.org/en/user_manual/python_scripting/install_custom_python_plugin.html) for more information.
* Activate the plugin in Krita and restart.
* In the plugin docker, click "Configure" to start server installation. _Requires ~10 GB free disk space._

### Custom ComfyUI Server

The plugin uses [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as backend. As an _alternative_ to the automatic installation,
you can install it manually, or use an existing installation. If the server is already running locally before starting Krita, the plugin will
automatically try to connect. Using a remote server is also possible this way.

To use an external installation, the following extensions and models are required:
* ComfyUI custom nodes:
  * [ControlNet preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux)
  * [IP-Adapter](https://github.com/laksjdjf/IPAdapter-ComfyUI)
  * [External tooling nodes](https://github.com/Acly/comfyui-tooling-nodes)
* Model files (paths are relative to ComfyUI install folder):
  * [ControlNet inpaint](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors) to `models/controlnet`
  * [Clip-Vision (SD1.5)](https://huggingface.co/h94/IP-Adapter/blob/main/models/image_encoder/pytorch_model.bin) to `models/clip_vision/SD1.5`
  * [IP-Adapter (SD1.5)](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin) to `custom_nodes/IPAdapter-ComfyUI/models`

## Technology

* Image generation: [Stable Diffusion](https://github.com/Stability-AI/generative-models)
* Diffusion backend: [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* Inpainting: [ControlNet](https://github.com/lllyasviel/ControlNet), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)