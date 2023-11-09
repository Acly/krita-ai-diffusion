# Requirements for ComfyUI

You can use the Krita plugin with your custom installation of ComfyUI. When connecting it will check
for a number of custom node extensions and models which it requires to function. These are listed below.

## Required custom nodes
Install custom nodes according to the instructions of the respective projects, or use [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager).

  * [ControlNet preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux)
  * [IP-Adapter](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
  * [Ultimate SD Upscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale)
  * [External tooling nodes](https://github.com/Acly/comfyui-tooling-nodes)

## Required models
Download models to the paths indicated below. If you are using `extra_model_paths.yml`, those will also work. Shared models are always required, and at least one of SD1.5 and SDXL is needed.

### Shared
  * [Clip-Vision](https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin) to `models/clip_vision/SD1.5`
  * [NMKD Superscale SP_178000_G](https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth) to `models/upscale_models`

### SD 1.5
  * [ControlNet inpaint](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors) to `models/controlnet`
  * [ControlNet tile](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors) to `models/controlnet`
  * [IP-Adapter (SD1.5)](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors) to `custom_nodes/ComfyUI_IPAdapter_plus/models`

### SD XL
  * [IP-Adapter (SDXL)](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors) to `custom_nodes/ComfyUI_IPAdapter_plus/models`

## Checkpoints
The following checkpoints are used by the default styles:
* [Realistic Vision](https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=pruned&fp=fp16)
* [DreamShaper](https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16)
* [JuggernautXL](https://civitai.com/api/download/models/198530)

At least one checkpoint is required, but it doesn't have to be one of the above.
