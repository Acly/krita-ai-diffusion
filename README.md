# Generative AI _for Krita_

Generate images from within Krita with minimal fuss: Select an area, push a button,
and new content that matches your image will be generated. Or expand your canvas and
fill it with generated content that fits the existing image. Text prompts are optional.
No tweaking required!

This plugin seeks to provide what "Generative Fill/Expand" do in Photoshop - and go beyond.
Adjust strength to refine existing content _(img2img)_ or generate images from scratch.
Powerful customization is available for advanced users.

And of course it's _local_, _open source_ and _free_.

## Features

Features are designed to fit an interactive workflow where AI generation as just another
tool when painting. They are meant to synergize with traditional tools and the layer stack.

* **Inpaint**: Use Krita's selection tools to mark an area to remove or replace existing content in the image. Simple text prompts can be used to steer generation.
* **Outpaint**: Extend your canvas, select a blank area and automatically fill it with content that seamlessly blends into the existing image.
* **Generate**: Create new image from scratch by decribing them with words.
* **Refine**: Use the strength slider to refine existing image content instead of replacing it entirely. This also works great for adding new things to an image by creating a (crude) sketch and refining at high strength!
* **Resolutions**: Work efficiently at any resolution! The plugin will automatically use resolutions appropriate for the AI model, and upscale them to fit your image region.
* **Job Queue**: Depending on hardware, image generation can take some time. The plugin allows you to queue and cancel jobs while working on your image.
* **History**: Not every image will turn out stellar. Preview results and browse previous generations and prompts at any time.
* **Defaults**: Strong default style presets allow a simple UI which covers many scenarios.
* **Customization**: Create your own presets - select a Stable Diffusion checkpoint, add LoRA and tweak samplers.

## Installation

* If you haven't yet, go and install [Krita](https://krita.org/)!<br>
  _Recommended version: 5.2.0_
* [Download]() and install this plugin. It works like any other Krita Python plugin by unpacking into the `pykrita` folder. See [official documentation](https://docs.krita.org/en/user_manual/python_scripting/install_custom_python_plugin.html) on how to locate.
* Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI). This is the backend server used by the plugin. There are multiple ways:
  * Install directly from [ComfyUI](https://github.com/comfyanonymous/ComfyUI#installing). 
  * Use [Stability Matrix](https://github.com/LykosAI/StabilityMatrix). Select ComfyUI and follow instructions.
  * Use an existing install of ComfyUI.
* Install the following ComfyUI custom nodes:
  * [ControlNet preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux)
  * [IP-Adapter](https://github.com/laksjdjf/IPAdapter-ComfyUI)
  * [External tooling nodes](https://github.com/Acly/comfyui-tooling-nodes)
* Download the following models (paths are relative to ComfyUI install folder):
  * [ControlNet inpaint](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors) to `models/controlnet`
  * [Clip-Vision](https://huggingface.co/h94/IP-Adapter/blob/main/models/image_encoder/pytorch_model.bin) to `models/clip_vision/SD1.5`
  * [IP-Adapter](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin) to `custom_nodes/IPAdapter-ComfyUI/models`
* _Optional:_ downloaded recommended checkpoints. These are used by the default styles.
  * [Realistic Vision](https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=pruned&fp=fp16) to `models/checkpoints`
  * [DreamShaper](https://civitai.com/api/download/models/109123?type=Model&format=SafeTensor&size=pruned&fp=fp16) to `models/checkpoints`
  