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
        "IPAdapter-ComfyUI",
        "https://github.com/laksjdjf/IPAdapter-ComfyUI",
        ["IPAdapter"],
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
        "CLIP Vision model",
        ResourceKind.clip_vision,
        Path("models/clip_vision/SD1.5"),
        "pytorch_model.bin",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin",
    ),
    ModelResource(
        "IP-Adapter model",
        ResourceKind.ip_adapter,
        Path("custom_nodes/IPAdapter-ComfyUI/models"),
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
    names: Optional[Sequence[str]]

    def __init__(self, kind: ResourceKind, names: Optional[Sequence[str]] = None):
        self.kind = kind
        self.names = names

    def __str__(self):
        return f"Missing {self.kind.value}: {', '.join(self.names)}"


all = (
    [n.name for n in required_custom_nodes]
    + [m.name for m in required_models]
    + [c.name for c in default_checkpoints]
    + [m.name for m in optional_models]
)


class ControlMode(Enum):
    inpaint = 0
    scribble = 1
    line_art = 2
    soft_edge = 3
    canny_edge = 4
    depth = 5
    normal = 6
    pose = 7
    segmentation = 8

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
    ControlMode.scribble: "Scribble",
    ControlMode.line_art: "Line Art",
    ControlMode.soft_edge: "Soft Edge",
    ControlMode.canny_edge: "Canny Edge",
    ControlMode.depth: "Depth",
    ControlMode.normal: "Normal",
    ControlMode.pose: "Pose",
    ControlMode.segmentation: "Segment",
}

_control_filename = {
    ControlMode.inpaint: {
        SDVersion.sd1_5: "control_v11p_sd15_inpaint",
        SDVersion.sdxl: None,
    },
    ControlMode.scribble: {
        SDVersion.sd1_5: ["control_v11p_sd15_scribble", "control_lora_rank128_v11p_sd15_scribble"],
        SDVersion.sdxl: None,
    },
    ControlMode.line_art: {
        SDVersion.sd1_5: ["control_v11p_sd15_lineart", "control_lora_rank128_v11p_sd15_lineart"],
        SDVersion.sdxl: "control-lora-sketch-rank256",
    },
    ControlMode.soft_edge: {
        SDVersion.sd1_5: ["control_v11p_sd15_softedge", "control_lora_rank128_v11p_sd15_softedge"],
        SDVersion.sdxl: None,
    },
    ControlMode.canny_edge: {
        SDVersion.sd1_5: ["control_v11p_sd15_canny", "control_lora_rank128_v11p_sd15_canny"],
        SDVersion.sdxl: "control-lora-canny-rank256",
    },
    ControlMode.depth: {
        SDVersion.sd1_5: ["control_v11f1p_sd15_depth", "control_lora_rank128_v11f1p_sd15_depth"],
        SDVersion.sdxl: "control-lora-depth-rank256",
    },
    ControlMode.normal: {
        SDVersion.sd1_5: [
            "control_v11p_sd15_normalbae",
            "control_lora_rank128_v11p_sd15_normalbae",
        ],
        SDVersion.sdxl: None,
    },
    ControlMode.pose: {
        SDVersion.sd1_5: ["control_v11p_sd15_openpose", "control_lora_rank128_v11p_sd15_openpose"],
        SDVersion.sdxl: "control-lora-openposeXL2-rank256",
    },
    ControlMode.segmentation: {
        SDVersion.sd1_5: ["control_v11p_sd15_seg", "control_lora_rank128_v11p_sd15_seg"],
        SDVersion.sdxl: None,
    },
}
