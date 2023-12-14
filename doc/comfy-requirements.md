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

This can be done manually or via script ([see below](#script)).


### Shared
  * [Clip-Vision](https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true) to `models/clip_vision/SD1.5`
  * [NMKD Superscale SP_178000_G](https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth) to `models/upscale_models`

### SD 1.5
  * [ControlNet inpaint](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors) to `models/controlnet`
  * [ControlNet tile](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors) to `models/controlnet`
  * [IP-Adapter (SD1.5)](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors) to `custom_nodes/ComfyUI_IPAdapter_plus/models`
  * [LCM-LoRA (SD1.5)](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors?download=true) to `models/loras/lcm-lora-sdv1-5.safetensors` _rename!_

### SD XL
  * [IP-Adapter (SDXL)](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors) to `custom_nodes/ComfyUI_IPAdapter_plus/models`
  * [LCM-LoRA (SDXL)](https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors?download=true) to `models/loras/lcm-lora-sdxl.safetensors` _rename!_

## Checkpoints
The following checkpoints are used by the default styles:
* [Realistic Vision](https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=pruned&fp=fp16)
* [DreamShaper](https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16)
* [JuggernautXL](https://civitai.com/api/download/models/198530)

At least one checkpoint is required, but it doesn't have to be one of the above.

## <a name="script"></a> Download Script
The models above mostly list strict requirements, there are a lot of additional models (like ControlNet)
which activate optional features in the plugin. If you have them already - great. Otherwise you can
use the `download_models.py` script to fetch all required and optional models.

Find the script in the plugin folder (called `ai_diffusion`). Open a command prompt and run:
```
python -m pip install aiohttp tqdm
python download_models.py /path/to/your/comfyui
```
This will download _all_ models supported by the plugin directly into the specified folder with the correct version, location, and filename.
The download location does not have to be your ComfyUI installation, you can use an empty folder if you want to avoid clashes and copy models afterwards.
There are also options to only download a subset, or list all relevant URLs without downloading.
```
python download_models.py --help
```

_Note: The script downloads models only. It does not install or modify ComfyUI or custom nodes!_

## Troubleshooting
If you're getting errors about missing resources, or workload not being installed, it's probably because one of the models wasn't found.
You can find the `client.log` file in the `.logs` folder where you installed the plugin. Check the log for warnings. Here you will also
find which models were found in your installation, and the patterns the plugin looks for.

Model paths must contain one of the search patterns entirely to match. The model path is allowed to be longer though: you may place models
in arbitrary subfolders and they will still  be found. If there are multiple matches, any files placed inside a `krita` subfolder are prioritized.