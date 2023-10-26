from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

from . import SDVersion


class CustomNode(NamedTuple):
    name: str
    folder: str
    url: str
    nodes: Sequence[str]


required_custom_nodes = [
    CustomNode(
        "ControlNet Preprocessors",
        "comfyui_controlnet_aux",
        "https://github.com/Fannovel16/comfyui_controlnet_aux",
        ["InpaintPreprocessor"],
    ),
    CustomNode(
        "IP-Adapter",
        "ComfyUI_IPAdapter_plus",
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        ["IPAdapterModelLoader", "IPAdapterApply"],
    ),
    CustomNode(
        "Ultimate SD Upscale",
        "ComfyUI_UltimateSDUpscale",
        "https://github.com/Acly/krita-ai-diffusion/releases/download/v0.1.0/ComfyUI_UltimateSDUpscale_6ea48202a76ccf5904ddfa85f826efa80dd50520.zip",
        ["UltimateSDUpscale"],
    ),
    CustomNode(
        "External Tooling Nodes",
        "comfyui-tooling-nodes",
        "https://github.com/Acly/comfyui-tooling-nodes",
        [
            "ETN_LoadImageBase64",
            "ETN_LoadMaskBase64",
            "ETN_SendImageWebSocket",
            "ETN_CropImage",
            "ETN_ApplyMaskToImage",
        ],
    ),
]


class ResourceKind(Enum):
    checkpoint = "Stable Diffusion Checkpoint"
    controlnet = "ControlNet model"
    clip_vision = "CLIP Vision model"
    ip_adapter = "IP-Adapter model"
    upscaler = "Upscale model"
    node = "custom node"


class ModelResource(NamedTuple):
    name: str
    kind: ResourceKind
    folder: Path
    filename: str
    url: str


required_models = [
    ModelResource(
        "ControlNet Inpaint",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_v11p_sd15_inpaint_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Tile",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors",
    ),
    ModelResource(
        "CLIP Vision model",
        ResourceKind.clip_vision,
        Path("models/clip_vision/SD1.5"),
        "pytorch_model.bin",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin",
    ),
    ModelResource(
        "IP-Adapter model",
        ResourceKind.ip_adapter,
        Path("custom_nodes/ComfyUI_IPAdapter_plus/models"),
        "ip-adapter_sd15.bin",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin",
    ),
    ModelResource(
        "NMKD Superscale model",
        ResourceKind.upscaler,
        Path("models/upscale_models"),
        "4x_NMKD-Superscale-SP_178000_G.pth",
        "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth",
    ),
]

default_checkpoints = [
    ModelResource(
        "Realistic Vision",
        ResourceKind.checkpoint,
        Path("models/checkpoints"),
        "realisticVisionV51_v51VAE.safetensors",
        "https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    ),
    ModelResource(
        "DreamShaper",
        ResourceKind.checkpoint,
        Path("models/checkpoints"),
        "dreamshaper_8.safetensors",
        "https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    ),
]

upscale_models = [
    ModelResource(
        "HAT Super-Resolution (quality)",
        ResourceKind.upscaler,
        Path("models/upscale_models"),
        "HAT_SRx4_ImageNet-pretrain.pth",
        "https://huggingface.co/Acly/hat/resolve/main/HAT_SRx4_ImageNet-pretrain.pth",
    ),
    ModelResource(
        "Real HAT GAN Super-Resolution (sharper)",
        ResourceKind.upscaler,
        Path("models/upscale_models"),
        "Real_HAT_GAN_sharper.pth",
        "https://huggingface.co/Acly/hat/resolve/main/Real_HAT_GAN_sharper.pth",
    ),
]

optional_models = [
    ModelResource(
        "ControlNet Scribble",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_lora_rank128_v11p_sd15_scribble_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_scribble_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Line Art",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_v11p_sd15_lineart_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Soft Edge",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_v11p_sd15_softedge_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Canny Edge",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_v11p_sd15_canny_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Depth",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Normal",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Pose",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_lora_rank128_v11p_sd15_openpose_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_openpose_fp16.safetensors",
    ),
    ModelResource(
        "ControlNet Segmentation",
        ResourceKind.controlnet,
        Path("models/controlnet"),
        "control_lora_rank128_v11p_sd15_seg_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_seg_fp16.safetensors",
    ),
]


class MissingResource(Exception):
    kind: ResourceKind
    names: Sequence[str] | Sequence[CustomNode] | None

    def __init__(
        self, kind: ResourceKind, names: Sequence[str] | Sequence[CustomNode] | None = None
    ):
        self.kind = kind
        self.names = names

    def __str__(self):
        return f"Missing {self.kind.value}: {', '.join(str(n) for n in self.names or [])}"


all = (
    [n.name for n in required_custom_nodes]
    + [m.name for m in required_models]
    + [c.name for c in default_checkpoints]
    + [m.name for m in upscale_models]
    + [m.name for m in optional_models]
)


class UpscalerName(Enum):
    default = "4x_NMKD-Superscale-SP_178000_G.pth"
    quality = "HAT_SRx4_ImageNet-pretrain.pth"
    sharp = "Real_HAT_GAN_sharper.pth"


class ControlMode(Enum):
    image = 0
    inpaint = 1
    scribble = 2
    line_art = 3
    soft_edge = 4
    canny_edge = 5
    depth = 6
    normal = 7
    pose = 8
    segmentation = 9
    blur = 10

    @property
    def is_lines(self):
        return self in [
            ControlMode.scribble,
            ControlMode.line_art,
            ControlMode.soft_edge,
            ControlMode.canny_edge,
        ]

    @property
    def text(self):
        return _control_text[self]

    def filenames(self, sd_version: SDVersion):
        return _control_filename[self][sd_version]


_control_text = {
    ControlMode.image: "Image",
    ControlMode.scribble: "Scribble",
    ControlMode.line_art: "Line Art",
    ControlMode.soft_edge: "Soft Edge",
    ControlMode.canny_edge: "Canny Edge",
    ControlMode.depth: "Depth",
    ControlMode.normal: "Normal",
    ControlMode.pose: "Pose",
    ControlMode.segmentation: "Segment",
    ControlMode.blur: "Blur",
}

_control_filename = {
    ControlMode.image: {  # uses clip vision / ip-adapter
        SDVersion.sd15: None,
        SDVersion.sdxl: None,
    },
    ControlMode.inpaint: {
        SDVersion.sd15: "control_v11p_sd15_inpaint",
        SDVersion.sdxl: None,
    },
    ControlMode.scribble: {
        SDVersion.sd15: ["control_v11p_sd15_scribble", "control_lora_rank128_v11p_sd15_scribble"],
        SDVersion.sdxl: None,
    },
    ControlMode.line_art: {
        SDVersion.sd15: ["control_v11p_sd15_lineart", "control_lora_rank128_v11p_sd15_lineart"],
        SDVersion.sdxl: ["control-lora-sketch-rank", "sai_xl_sketch_"],
    },
    ControlMode.soft_edge: {
        SDVersion.sd15: ["control_v11p_sd15_softedge", "control_lora_rank128_v11p_sd15_softedge"],
        SDVersion.sdxl: None,
    },
    ControlMode.canny_edge: {
        SDVersion.sd15: ["control_v11p_sd15_canny", "control_lora_rank128_v11p_sd15_canny"],
        SDVersion.sdxl: ["control-lora-canny-rank", "sai_xl_canny_"],
    },
    ControlMode.depth: {
        SDVersion.sd15: ["control_v11f1p_sd15_depth", "control_lora_rank128_v11f1p_sd15_depth"],
        SDVersion.sdxl: ["control-lora-depth-rank", "sai_xl_depth_"],
    },
    ControlMode.normal: {
        SDVersion.sd15: [
            "control_v11p_sd15_normalbae",
            "control_lora_rank128_v11p_sd15_normalbae",
        ],
        SDVersion.sdxl: None,
    },
    ControlMode.pose: {
        SDVersion.sd15: ["control_v11p_sd15_openpose", "control_lora_rank128_v11p_sd15_openpose"],
        SDVersion.sdxl: ["control-lora-openposeXL2-rank", "thibaud_xl_openpose"],
    },
    ControlMode.segmentation: {
        SDVersion.sd15: ["control_v11p_sd15_seg", "control_lora_rank128_v11p_sd15_seg"],
        SDVersion.sdxl: None,
    },
    ControlMode.blur: {
        SDVersion.sd15: ["control_v11f1e_sd15_tile", "control_lora_rank128_v11f1e_sd15_tile"],
        SDVersion.sdxl: None,
    },
}
