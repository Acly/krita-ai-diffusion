from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Sequence

# Version identifier for all the resources defined here. This is used as the server version.
# It usually follows the plugin version, but not all new plugin versions also require a server update.
version = "1.13.0"

comfy_url = "https://github.com/comfyanonymous/ComfyUI"
comfy_version = "af94eb14e3b44f6e6c48a7762a5e229cf3a006e4"


class CustomNode(NamedTuple):
    name: str
    folder: str
    url: str
    version: str
    nodes: Sequence[str]


required_custom_nodes = [
    CustomNode(
        "ControlNet Preprocessors",
        "comfyui_controlnet_aux",
        "https://github.com/Fannovel16/comfyui_controlnet_aux",
        "15d9861bdd947b299bf5dde1e2e3d8a962e4f26a",
        ["InpaintPreprocessor"],
    ),
    CustomNode(
        "IP-Adapter",
        "ComfyUI_IPAdapter_plus",
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        "4e898fe0c842259fb8eccbd9db1e2d533539cf9d",
        ["IPAdapterModelLoader", "IPAdapterApply"],
    ),
    CustomNode(
        "Ultimate SD Upscale",
        "ComfyUI_UltimateSDUpscale",
        "https://github.com/Acly/krita-ai-diffusion/releases/download/v0.1.0/ComfyUI_UltimateSDUpscale-6ea48202a76ccf5904ddfa85f826efa80dd50520-repack.zip",
        "6ea48202a76ccf5904ddfa85f826efa80dd50520",
        ["UltimateSDUpscale"],
    ),
    CustomNode(
        "External Tooling Nodes",
        "comfyui-tooling-nodes",
        "https://github.com/Acly/comfyui-tooling-nodes",
        "b2496a3f132f8c3f7d452a0960c422f55c33d128",
        [
            "ETN_LoadImageBase64",
            "ETN_LoadMaskBase64",
            "ETN_SendImageWebSocket",
            "ETN_ApplyMaskToImage",
        ],
    ),
]


class SDVersion(Enum):
    sd15 = "SD 1.5"
    sdxl = "SD XL"

    auto = "Automatic"
    all = "All"

    @staticmethod
    def from_string(string: str):
        if string == "sd15":
            return SDVersion.sd15
        if string == "sdxl":
            return SDVersion.sdxl
        return None

    @staticmethod
    def from_checkpoint_name(checkpoint: str):
        if SDVersion.sdxl.matches(checkpoint):
            return SDVersion.sdxl
        return SDVersion.sd15

    @staticmethod
    def match(a: SDVersion, b: SDVersion):
        if a is SDVersion.all or b is SDVersion.all:
            return True
        return a is b

    def matches(self, checkpoint: str):
        # Fallback check if it can't be queried from the server
        xl_in_name = "xl" in checkpoint.lower()
        return self is SDVersion.auto or ((self is SDVersion.sdxl) == xl_in_name)

    def resolve(self, checkpoint: str):
        if self is SDVersion.auto:
            return SDVersion.sdxl if SDVersion.sdxl.matches(checkpoint) else SDVersion.sd15
        return self

    @property
    def has_controlnet_inpaint(self):
        return self is SDVersion.sd15

    @property
    def has_controlnet_blur(self):
        return self is SDVersion.sd15


class ResourceKind(Enum):
    checkpoint = "Stable Diffusion Checkpoint"
    controlnet = "ControlNet model"
    clip_vision = "CLIP Vision model"
    ip_adapter = "IP-Adapter model"
    lora = "LoRA model"
    upscaler = "Upscale model"
    node = "custom node"


class ModelRequirements(Enum):
    none = 0
    insightface = 1


class ModelResource(NamedTuple):
    name: str
    kind: ResourceKind
    sd_version: SDVersion
    files: dict[Path, str]
    alternatives: list[Path] | None = None  # for backwards compatibility
    requirements: ModelRequirements = ModelRequirements.none

    @property
    def filename(self):
        assert len(self.files) == 1
        return next(iter(self.files)).name

    @property
    def folder(self):
        return next(iter(self.files)).parent

    @property
    def url(self):
        assert len(self.files) == 1
        return next(iter(self.files.values()))

    def exists_in(self, path: Path):
        exact = all((path / filepath).exists() for filepath in self.files.keys())
        alt = self.alternatives is not None and any((path / f).exists() for f in self.alternatives)
        return exact or alt


required_models = [
    ModelResource(
        "CLIP Vision model",
        ResourceKind.clip_vision,
        SDVersion.all,
        {
            Path(
                "models/clip_vision/SD1.5/model.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"
        },
        alternatives=[Path("models/clip_vision/SD1.5/pytorch_model.bin")],
    ),
    ModelResource(
        "NMKD Superscale model",
        ResourceKind.upscaler,
        SDVersion.all,
        {
            Path(
                "models/upscale_models/4x_NMKD-Superscale-SP_178000_G.pth"
            ): "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth"
        },
    ),
    ModelResource(
        "OmniSR Superscale model",
        ResourceKind.upscaler,
        SDVersion.all,
        {
            Path(
                "models/upscale_models/OmniSR_X2_DIV2K.safetensors"
            ): "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X2_DIV2K.safetensors",
            Path(
                "models/upscale_models/OmniSR_X3_DIV2K.safetensors"
            ): "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X3_DIV2K.safetensors",
            Path(
                "models/upscale_models/OmniSR_X4_DIV2K.safetensors"
            ): "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X4_DIV2K.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Inpaint",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_v11p_sd15_inpaint_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors"
        },
    ),
    ModelResource(
        "ControlNet Tile",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors",
        },
    ),
    ModelResource(
        "IP-Adapter (SD1.5)",
        ResourceKind.ip_adapter,
        SDVersion.sd15,
        {
            Path(
                "models/ipadapter/ip-adapter_sd15.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"
        },
        alternatives=[
            Path("custom_nodes/ComfyUI_IPAdapter_plus/models/ip-adapter_sd15.safetensors")
        ],
    ),
    ModelResource(
        "IP-Adapter (SDXL)",
        ResourceKind.ip_adapter,
        SDVersion.sdxl,
        {
            Path(
                "models/ipadapter/ip-adapter_sdxl_vit-h.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
        },
        alternatives=[
            Path("custom_nodes/ComfyUI_IPAdapter_plus/models/ip-adapter_sdxl_vit-h.safetensors")
        ],
    ),
    ModelResource(
        "LCM-LoRA (SD1.5)",
        ResourceKind.lora,
        SDVersion.sd15,
        {
            Path(
                "models/loras/lcm-lora-sdv1-5.safetensors"
            ): "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors",
        },
    ),
    ModelResource(
        "LCM-LoRA (SDXL)",
        ResourceKind.lora,
        SDVersion.sdxl,
        {
            Path(
                "models/loras/lcm-lora-sdxl.safetensors"
            ): "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors",
        },
    ),
]

default_checkpoints = [
    ModelResource(
        "Realistic Vision",
        ResourceKind.checkpoint,
        SDVersion.sd15,
        {
            Path(
                "models/checkpoints/realisticVisionV51_v51VAE.safetensors"
            ): "https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        },
    ),
    ModelResource(
        "DreamShaper",
        ResourceKind.checkpoint,
        SDVersion.sd15,
        {
            Path(
                "models/checkpoints/dreamshaper_8.safetensors"
            ): "https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        },
    ),
    ModelResource(
        "Juggernaut XL",
        ResourceKind.checkpoint,
        SDVersion.sdxl,
        {
            Path(
                "models/checkpoints/juggernautXL_version6Rundiffusion.safetensors"
            ): "https://civitai.com/api/download/models/198530"
        },
    ),
]

upscale_models = [
    ModelResource(
        "HAT Super-Resolution (quality)",
        ResourceKind.upscaler,
        SDVersion.all,
        {
            Path(
                "models/upscale_models/HAT_SRx4_ImageNet-pretrain.pth"
            ): "https://huggingface.co/Acly/hat/resolve/main/HAT_SRx4_ImageNet-pretrain.pth"
        },
    ),
    ModelResource(
        "Real HAT GAN Super-Resolution (sharper)",
        ResourceKind.upscaler,
        SDVersion.all,
        {
            Path(
                "models/upscale_models/Real_HAT_GAN_sharper.pth"
            ): "https://huggingface.co/Acly/hat/resolve/main/Real_HAT_GAN_sharper.pth"
        },
    ),
]

optional_models = [
    ModelResource(
        "ControlNet Scribble",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_scribble_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_scribble_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Line Art",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_v11p_sd15_lineart_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Soft Edge",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_v11p_sd15_softedge_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Canny Edge",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_v11p_sd15_canny_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Depth",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Normal",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Pose",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_openpose_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_openpose_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Segmentation",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_seg_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_seg_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Stencil",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_v1p_sd15_qrcode_monster.safetensors"
            ): "https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/control_v1p_sd15_qrcode_monster.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Hand Refiner",
        ResourceKind.controlnet,
        SDVersion.sd15,
        {
            Path(
                "models/controlnet/control_sd15_inpaint_depth_hand_fp16.safetensors"
            ): "https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/resolve/main/control_sd15_inpaint_depth_hand_fp16.safetensors",
        },
    ),
    ModelResource(
        "IP-Adapter Face (SD1.5)",
        ResourceKind.ip_adapter,
        SDVersion.sd15,
        {
            Path(
                "models/ipadapter/ip-adapter-faceid-plusv2_sd15.bin"
            ): "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin",
            Path(
                "models/loras/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors",
        },
        requirements=ModelRequirements.insightface,
    ),
    ModelResource(
        "ControlNet Line Art (XL)",
        ResourceKind.controlnet,
        SDVersion.sdxl,
        {
            Path(
                "models/controlnet/sai_xl_sketch_256lora.safetensors"
            ): "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_sketch_256lora.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Canny Edge (XL)",
        ResourceKind.controlnet,
        SDVersion.sdxl,
        {
            Path(
                "models/controlnet/sai_xl_canny_256lora.safetensors"
            ): "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_canny_256lora.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Depth (XL)",
        ResourceKind.controlnet,
        SDVersion.sdxl,
        {
            Path(
                "models/controlnet/sai_xl_depth_256lora.safetensors"
            ): "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_depth_256lora.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Pose (XL)",
        ResourceKind.controlnet,
        SDVersion.sdxl,
        {
            Path(
                "models/controlnet/thibaud_xl_openpose_256lora.safetensors"
            ): "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/thibaud_xl_openpose_256lora.safetensors",
        },
    ),
    ModelResource(
        "IP-Adapter Face (XL)",
        ResourceKind.ip_adapter,
        SDVersion.sdxl,
        {
            Path(
                "models/ipadapter/ip-adapter-faceid_sdxl.bin"
            ): "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin",
            Path(
                "models/loras/ip-adapter-faceid_sdxl_lora.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl_lora.safetensors",
        },
        requirements=ModelRequirements.insightface,
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


all_resources = (
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
    fast_2x = "OmniSR_X2_DIV2K.safetensors"
    fast_3x = "OmniSR_X3_DIV2K.safetensors"
    fast_4x = "OmniSR_X4_DIV2K.safetensors"

    @staticmethod
    def fast_x(x: int):
        return UpscalerName.__members__[f"fast_{x}x"]


class ControlMode(Enum):
    reference = 0
    face = 1
    inpaint = 2
    scribble = 3
    line_art = 4
    soft_edge = 5
    canny_edge = 6
    depth = 7
    normal = 8
    pose = 9
    segmentation = 10
    blur = 11
    stencil = 12
    hands = 13

    @property
    def is_lines(self):
        return self in [
            ControlMode.scribble,
            ControlMode.line_art,
            ControlMode.soft_edge,
            ControlMode.canny_edge,
        ]

    @property
    def has_preprocessor(self):
        return self.is_control_net and not self in [
            ControlMode.inpaint,
            ControlMode.blur,
            ControlMode.stencil,
        ]

    @property
    def is_control_net(self):
        return not self.is_ip_adapter

    @property
    def is_ip_adapter(self):
        return self in [ControlMode.reference, ControlMode.face]

    @property
    def text(self):
        return _control_text[self]


_control_text = {
    ControlMode.reference: "Reference",
    ControlMode.inpaint: "Inpaint",
    ControlMode.face: "Face",
    ControlMode.scribble: "Scribble",
    ControlMode.line_art: "Line Art",
    ControlMode.soft_edge: "Soft Edge",
    ControlMode.canny_edge: "Canny Edge",
    ControlMode.depth: "Depth",
    ControlMode.normal: "Normal",
    ControlMode.pose: "Pose",
    ControlMode.segmentation: "Segment",
    ControlMode.blur: "Blur",
    ControlMode.stencil: "Stencil",
    ControlMode.hands: "Hands",
}


def resource_id(
    kind: ResourceKind, version: SDVersion, identifier: ControlMode | UpscalerName | str
):
    if isinstance(identifier, Enum):
        identifier = identifier.name
    return f"{kind.name}-{identifier}-{version.name}"


def search_path(
    kind: ResourceKind, version: SDVersion, identifier: ControlMode | UpscalerName | str
):
    return search_paths.get(resource_id(kind, version, identifier), None)


def is_required(
    kind: ResourceKind, version: SDVersion, identifier: ControlMode | UpscalerName | str
):
    return resource_id(kind, version, identifier) in required_resource_ids


# fmt: off
search_paths: dict[str, list[str]] = {
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.inpaint):  ["control_v11p_sd15_inpaint"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.scribble): ["control_v11p_sd15_scribble", "control_lora_rank128_v11p_sd15_scribble"],
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.scribble): ["control-lora-sketch-rank", "sai_xl_sketch_"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.line_art): ["control_v11p_sd15_lineart", "control_lora_rank128_v11p_sd15_lineart"],
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.line_art): ["control-lora-sketch-rank", "sai_xl_sketch_"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.soft_edge): ["control_v11p_sd15_softedge", "control_lora_rank128_v11p_sd15_softedge"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.canny_edge): ["control_v11p_sd15_canny", "control_lora_rank128_v11p_sd15_canny"],
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.canny_edge): ["control-lora-canny-rank", "sai_xl_canny_"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.depth): ["control_v11f1p_sd15_depth", "control_lora_rank128_v11f1p_sd15_depth"],
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.depth): ["control-lora-depth-rank", "sai_xl_depth_"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.normal): ["control_v11p_sd15_normalbae", "control_lora_rank128_v11p_sd15_normalbae"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.pose): ["control_v11p_sd15_openpose", "control_lora_rank128_v11p_sd15_openpose"],
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.pose): ["control-lora-openposexl2-rank", "thibaud_xl_openpose"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.segmentation): ["control_v11p_sd15_seg", "control_lora_rank128_v11p_sd15_seg"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.blur): ["control_v11f1e_sd15_tile", "control_lora_rank128_v11f1e_sd15_tile"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.stencil): ["control_v1p_sd15_qrcode_monster"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.hands): ["control_sd15_inpaint_depth_hand"],
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.hands): ["control-lora-depth-rank", "sai_xl_depth_"],
    resource_id(ResourceKind.ip_adapter, SDVersion.sd15, ControlMode.reference): ["ip-adapter_sd15"],
    resource_id(ResourceKind.ip_adapter, SDVersion.sdxl, ControlMode.reference): ["ip-adapter_sdxl_vit-h"],
    resource_id(ResourceKind.ip_adapter, SDVersion.sd15, ControlMode.face): ["ip-adapter-faceid-plusv2_sd15"],
    resource_id(ResourceKind.ip_adapter, SDVersion.sdxl, ControlMode.face): ["ip-adapter-faceid_sdxl"],
    resource_id(ResourceKind.clip_vision, SDVersion.all, "ip_adapter"): ["sd1.5/pytorch_model.bin", "sd1.5/model.safetensors"],
    resource_id(ResourceKind.lora, SDVersion.sd15, "lcm"): ["lcm-lora-sdv1-5.safetensors", "lcm/sd1.5/pytorch_lora_weights.safetensors"],
    resource_id(ResourceKind.lora, SDVersion.sdxl, "lcm"): ["lcm-lora-sdxl.safetensors", "lcm/sdxl/pytorch_lora_weights.safetensors"],
    resource_id(ResourceKind.lora, SDVersion.sd15, ControlMode.face): ["ip-adapter-faceid-plusv2_sd15_lora"],
    resource_id(ResourceKind.lora, SDVersion.sdxl, ControlMode.face): ["ip-adapter-faceid_sdxl_lora"],
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.default): [UpscalerName.default.value],
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_2x): [UpscalerName.fast_2x.value],
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_3x): [UpscalerName.fast_3x.value],
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_4x): [UpscalerName.fast_4x.value],
}
# fmt: on

required_resource_ids = set([
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.inpaint),
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.blur),
    resource_id(ResourceKind.ip_adapter, SDVersion.sd15, ControlMode.reference),
    resource_id(ResourceKind.ip_adapter, SDVersion.sdxl, ControlMode.reference),
    resource_id(ResourceKind.clip_vision, SDVersion.all, "ip_adapter"),
    resource_id(ResourceKind.lora, SDVersion.sd15, "lcm"),
    resource_id(ResourceKind.lora, SDVersion.sdxl, "lcm"),
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.default),
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_2x),
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_3x),
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_4x),
])
