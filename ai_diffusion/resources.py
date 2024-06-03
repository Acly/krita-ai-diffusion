from __future__ import annotations
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import NamedTuple, Sequence

# Version identifier for all the resources defined here. This is used as the server version.
# It usually follows the plugin version, but not all new plugin versions also require a server update.
version = "1.17.0"

comfy_url = "https://github.com/comfyanonymous/ComfyUI"
comfy_version = "45ec1cbe963055798765645c4f727122a7d3e35e"


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
        "33d6e86b65c40e5f97052262c6900f413684ccde",
        ["InpaintPreprocessor"],
    ),
    CustomNode(
        "IP-Adapter",
        "ComfyUI_IPAdapter_plus",
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        "0d0a7b3693baf8903fe2028ff218b557d619a93d",
        ["IPAdapterModelLoader", "IPAdapter"],
    ),
    CustomNode(
        "External Tooling Nodes",
        "comfyui-tooling-nodes",
        "https://github.com/Acly/comfyui-tooling-nodes",
        "bcb591c7b998e13f12e2d47ee08cf8af8f791e50",
        [
            "ETN_LoadImageBase64",
            "ETN_LoadMaskBase64",
            "ETN_SendImageWebSocket",
            "ETN_ApplyMaskToImage",
        ],
    ),
    CustomNode(
        "Inpaint Nodes",
        "comfyui-inpaint-nodes",
        "https://github.com/Acly/comfyui-inpaint-nodes",
        "8469f5531116475abb6d7e9c04720d0a29485a66",
        ["INPAINT_LoadFooocusInpaint", "INPAINT_ApplyFooocusInpaint"],
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


class ResourceKind(Enum):
    checkpoint = "Stable Diffusion Checkpoint"
    controlnet = "ControlNet model"
    clip_vision = "CLIP Vision model"
    ip_adapter = "IP-Adapter model"
    lora = "LoRA model"
    upscaler = "Upscale model"
    inpaint = "Inpaint model"
    embedding = "Textual Embedding"
    preprocessor = "Preprocessor"
    node = "custom node"


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
    style = 14
    composition = 15
    face = 13
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
    stencil = 11
    hands = 12

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
        return self in [
            ControlMode.reference,
            ControlMode.face,
            ControlMode.style,
            ControlMode.composition,
        ]

    @property
    def is_part_of_image(self):  # not only used as guidance hint
        return self in [ControlMode.reference, ControlMode.line_art, ControlMode.blur]

    @property
    def is_structural(self):  # strong impact on image composition/structure
        return not (self.is_ip_adapter or self is ControlMode.inpaint)

    @property
    def text(self):
        return _control_text[self]


class ResourceId(NamedTuple):
    kind: ResourceKind
    version: SDVersion
    identifier: ControlMode | UpscalerName | str

    @property
    def string(self):
        return resource_id(self.kind, self.version, self.identifier)

    @property
    def name(self):
        ident = self.identifier.name if isinstance(self.identifier, Enum) else self.identifier
        return f"{self.kind.value} '{ident}' for {self.version.value}"


class ModelRequirements(Enum):
    none = 0
    insightface = 1


class ModelResource(NamedTuple):
    name: str
    id: ResourceId
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

    @property
    def kind(self):
        return self.id.kind

    @property
    def sd_version(self):
        return self.id.version


required_models = [
    ModelResource(
        "CLIP Vision model",
        ResourceId(ResourceKind.clip_vision, SDVersion.all, "ip_adapter"),
        {
            Path(
                "models/clip_vision/clip-vision_vit-h.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"
        },
        alternatives=[
            Path("models/clip_vision/SD1.5/model.safetensors"),
            Path("models/clip_vision/SD1.5/pytorch_model.bin"),
        ],
    ),
    ModelResource(
        "NMKD Superscale model",
        ResourceId(ResourceKind.upscaler, SDVersion.all, UpscalerName.default),
        {
            Path(
                "models/upscale_models/4x_NMKD-Superscale-SP_178000_G.pth"
            ): "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth"
        },
    ),
    ModelResource(
        "OmniSR Superscale model",
        ResourceId(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_4x),
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
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.inpaint),
        {
            Path(
                "models/controlnet/control_v11p_sd15_inpaint_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors"
        },
    ),
    ModelResource(
        "ControlNet Tile",
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.blur),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors",
        },
    ),
    ModelResource(
        "IP-Adapter (SD1.5)",
        ResourceId(ResourceKind.ip_adapter, SDVersion.sd15, ControlMode.reference),
        {
            Path(
                "models/ipadapter/ip-adapter_sd15.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"
        },
    ),
    ModelResource(
        "IP-Adapter (SDXL)",
        ResourceId(ResourceKind.ip_adapter, SDVersion.sdxl, ControlMode.reference),
        {
            Path(
                "models/ipadapter/ip-adapter_sdxl_vit-h.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
        },
    ),
    ModelResource(
        "LCM-LoRA (SD1.5)",
        ResourceId(ResourceKind.lora, SDVersion.sd15, "lcm"),
        {
            Path(
                "models/loras/lcm-lora-sdv1-5.safetensors"
            ): "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors",
        },
    ),
    ModelResource(
        "LCM-LoRA (SDXL)",
        ResourceId(ResourceKind.lora, SDVersion.sdxl, "lcm"),
        {
            Path(
                "models/loras/lcm-lora-sdxl.safetensors"
            ): "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors",
        },
    ),
    ModelResource(
        "Fooocus Inpaint",
        ResourceId(ResourceKind.inpaint, SDVersion.sdxl, "fooocus-inpaint"),
        {
            Path(
                "models/inpaint/fooocus_inpaint_head.pth"
            ): "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
            Path(
                "models/inpaint/inpaint_v26.fooocus.patch"
            ): "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch",
        },
    ),
    ModelResource(
        "MAT Inpaint",
        ResourceId(ResourceKind.inpaint, SDVersion.all, "default"),
        {
            Path(
                "models/inpaint/MAT_Places512_G_fp16.safetensors"
            ): "https://huggingface.co/Acly/MAT/resolve/main/MAT_Places512_G_fp16.safetensors",
        },
    ),
    ModelResource(
        "Easy Negative",
        ResourceId(ResourceKind.embedding, SDVersion.sd15, "easy-negative"),
        {
            Path(
                "models/embeddings/EasyNegative.safetensors"
            ): "https://huggingface.co/embed/EasyNegative/resolve/main/EasyNegative.safetensors"
        },
    ),
]

default_checkpoints = [
    ModelResource(
        "Realistic Vision (Photography)",
        ResourceId(ResourceKind.checkpoint, SDVersion.sd15, "realistic-vision"),
        {
            Path(
                "models/checkpoints/realisticVisionV51_v51VAE.safetensors"
            ): "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticVisionV51_v51VAE.safetensors",
        },
    ),
    ModelResource(
        "DreamShaper (Artwork)",
        ResourceId(ResourceKind.checkpoint, SDVersion.sd15, "dreamshaper"),
        {
            Path(
                "models/checkpoints/dreamshaper_8.safetensors"
            ): "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors",
        },
    ),
    ModelResource(
        "Flat2D AniMerge (Cartoon/Anime)",
        ResourceId(ResourceKind.checkpoint, SDVersion.sd15, "flat2d-animerge"),
        {
            Path(
                "models/checkpoints/flat2DAnimerge_v45Sharp.safetensors"
            ): "https://huggingface.co/Acly/SD-Checkpoints/resolve/main/flat2DAnimerge_v45Sharp.safetensors"
        },
    ),
    ModelResource(
        "Juggernaut XL",
        ResourceId(ResourceKind.checkpoint, SDVersion.sdxl, "juggernaut"),
        {
            Path(
                "models/checkpoints/juggernautXL_version6Rundiffusion.safetensors"
            ): "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_version6Rundiffusion.safetensors"
        },
    ),
]

upscale_models = [
    ModelResource(
        "HAT Super-Resolution (quality)",
        ResourceId(ResourceKind.upscaler, SDVersion.all, UpscalerName.quality),
        {
            Path(
                "models/upscale_models/HAT_SRx4_ImageNet-pretrain.pth"
            ): "https://huggingface.co/Acly/hat/resolve/main/HAT_SRx4_ImageNet-pretrain.pth"
        },
    ),
    ModelResource(
        "Real HAT GAN Super-Resolution (sharper)",
        ResourceId(ResourceKind.upscaler, SDVersion.all, UpscalerName.sharp),
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
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.scribble),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_scribble_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_scribble_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Line Art",
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.line_art),
        {
            Path(
                "models/controlnet/control_v11p_sd15_lineart_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Soft Edge",
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.soft_edge),
        {
            Path(
                "models/controlnet/control_v11p_sd15_softedge_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Canny Edge",
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.canny_edge),
        {
            Path(
                "models/controlnet/control_v11p_sd15_canny_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Depth",
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.depth),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Normal",
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.normal),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Pose",
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.pose),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_openpose_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_openpose_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Segmentation",
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.segmentation),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_seg_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_seg_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Stencil",
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.stencil),
        {
            Path(
                "models/controlnet/control_v1p_sd15_qrcode_monster.safetensors"
            ): "https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/control_v1p_sd15_qrcode_monster.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Hand Refiner",
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.hands),
        {
            Path(
                "models/controlnet/control_sd15_inpaint_depth_hand_fp16.safetensors"
            ): "https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/resolve/main/control_sd15_inpaint_depth_hand_fp16.safetensors",
        },
    ),
    ModelResource(
        "IP-Adapter Face (SD1.5)",
        ResourceId(ResourceKind.ip_adapter, SDVersion.sd15, ControlMode.face),
        {
            Path(
                "models/ipadapter/ip-adapter-faceid-plusv2_sd15.bin"
            ): "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin",
            Path(
                "models/loras/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors",
        },
        requirements=ModelRequirements.insightface,
    ),
    ModelResource(
        "ControlNet Line Art (XL)",
        ResourceId(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.line_art),
        {
            Path(
                "models/controlnet/sai_xl_sketch_256lora.safetensors"
            ): "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_sketch_256lora.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Canny Edge (XL)",
        ResourceId(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.canny_edge),
        {
            Path(
                "models/controlnet/sai_xl_canny_256lora.safetensors"
            ): "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_canny_256lora.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Depth (XL)",
        ResourceId(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.depth),
        {
            Path(
                "models/controlnet/sai_xl_depth_256lora.safetensors"
            ): "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_depth_256lora.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Pose (XL)",
        ResourceId(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.pose),
        {
            Path(
                "models/controlnet/thibaud_xl_openpose_256lora.safetensors"
            ): "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/thibaud_xl_openpose_256lora.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Unblur (XL)",
        ResourceId(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.blur),
        {
            Path(
                "models/controlnet/TTPLANET_Controlnet_Tile_realistic_v2_fp16.safetensors"
            ): "https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic/resolve/main/TTPLANET_Controlnet_Tile_realistic_v2_fp16.safetensors",
        },
    ),
    ModelResource(
        "IP-Adapter Face (XL)",
        ResourceId(ResourceKind.ip_adapter, SDVersion.sdxl, ControlMode.face),
        {
            Path(
                "models/ipadapter/ip-adapter-faceid-plusv2_sdxl.bin"
            ): "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin",
            Path(
                "models/loras/ip-adapter-faceid-plusv2_sdxl_lora.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
        },
        requirements=ModelRequirements.insightface,
    ),
]

prefetch_models = [
    ModelResource(
        "Scribble Preprocessor",
        ResourceId(ResourceKind.preprocessor, SDVersion.all, ControlMode.scribble),
        {
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/table5_pidinet.pth"
            ): "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth"
        },
    ),
    ModelResource(
        "Line Art Preprocessor",
        ResourceId(ResourceKind.preprocessor, SDVersion.all, ControlMode.line_art),
        {
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/sk_model.pth"
            ): "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth",
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/sk_model2.pth"
            ): "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth",
        },
    ),
    ModelResource(
        "Soft Edge Preprocessor",
        ResourceId(ResourceKind.preprocessor, SDVersion.all, ControlMode.soft_edge),
        {
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/ControlNetHED.pth"
            ): "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
        },
    ),
    ModelResource(
        "Depth Preprocessor",
        ResourceId(ResourceKind.preprocessor, SDVersion.all, ControlMode.depth),
        {
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/LiheYoung/Depth-Anything/checkpoints/depth_anything_vitb14.pth"
            ): "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth"
        },
    ),
    ModelResource(
        "Pose Preprocessor",
        ResourceId(ResourceKind.preprocessor, SDVersion.all, ControlMode.pose),
        {
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/hr16/yolo-nas-fp16/yolo_nas_l_fp16.onnx"
            ): "https://huggingface.co/hr16/yolo-nas-fp16/resolve/main/yolo_nas_l_fp16.onnx",
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose/dw-ll_ucoco_384.onnx"
            ): "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
        },
    ),
]


class MissingResource(Exception):
    kind: ResourceKind
    names: Sequence[str] | Sequence[ResourceId] | Sequence[CustomNode] | None

    def __init__(
        self,
        kind: ResourceKind,
        names: Sequence[str] | Sequence[ResourceId] | Sequence[CustomNode] | None = None,
    ):
        self.kind = kind
        self.names = names

    def __str__(self):
        names = self.names or []
        names = [getattr(n, "name", n) for n in names]
        return f"Missing {self.kind.value}: {', '.join(str(n) for n in names)}"

    @property
    def search_path_string(self):
        if names := self.names:
            paths = (
                search_path(n.kind, n.version, n.identifier)
                for n in names
                if isinstance(n, ResourceId)
            )
            items = (", ".join(sp) for sp in paths if sp)
            return "Checking for files with a (partial) match:\n" + "\n".join(items)
        return ""


all_resources = (
    [n.name for n in required_custom_nodes]
    + [m.name for m in required_models]
    + [c.name for c in default_checkpoints]
    + [m.name for m in upscale_models]
    + [m.name for m in optional_models]
)


def all_models():
    return chain(
        required_models,
        optional_models,
        default_checkpoints,
        upscale_models,
        prefetch_models,
    )


_control_text = {
    ControlMode.reference: "Reference",
    ControlMode.inpaint: "Inpaint",
    ControlMode.style: "Style",
    ControlMode.composition: "Composition",
    ControlMode.face: "Face",
    ControlMode.scribble: "Scribble",
    ControlMode.line_art: "Line Art",
    ControlMode.soft_edge: "Soft Edge",
    ControlMode.canny_edge: "Canny Edge",
    ControlMode.depth: "Depth",
    ControlMode.normal: "Normal",
    ControlMode.pose: "Pose",
    ControlMode.segmentation: "Segment",
    ControlMode.blur: "Unblur",
    ControlMode.stencil: "Stencil",
    ControlMode.hands: "Hands",
}


def resource_id(
    kind: ResourceKind, version: SDVersion, identifier: ControlMode | UpscalerName | str
):
    if isinstance(identifier, Enum):
        identifier = identifier.name
    return f"{kind.name}-{identifier}-{version.name}"


def find_resource(id: ResourceId):
    return next((m for m in all_models() if m.id == id), None)


def search_path(
    kind: ResourceKind, version: SDVersion, identifier: ControlMode | UpscalerName | str
):
    return search_paths.get(resource_id(kind, version, identifier), None)


def is_required(
    kind: ResourceKind, version: SDVersion, identifier: ControlMode | UpscalerName | str
):
    return ResourceId(kind, version, identifier) in required_resource_ids


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
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.depth): ["control_sd15_depth_anything", "control_v11f1p_sd15_depth", "control_lora_rank128_v11f1p_sd15_depth"],
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.depth): ["control-lora-depth-rank", "sai_xl_depth_"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.normal): ["control_v11p_sd15_normalbae", "control_lora_rank128_v11p_sd15_normalbae"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.pose): ["control_v11p_sd15_openpose", "control_lora_rank128_v11p_sd15_openpose"],
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.pose): ["control-lora-openposexl2-rank", "thibaud_xl_openpose"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.segmentation): ["control_v11p_sd15_seg", "control_lora_rank128_v11p_sd15_seg"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.blur): ["control_v11f1e_sd15_tile", "control_lora_rank128_v11f1e_sd15_tile"],
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.blur): ["ttplanetsdxlcontrolnet", "ttplanet_sdxl_controlnet_tile_realistic", "ttplanet_controlnet_tile_realistic"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.stencil): ["control_v1p_sd15_qrcode_monster"],
    resource_id(ResourceKind.controlnet, SDVersion.sd15, ControlMode.hands): ["control_sd15_inpaint_depth_hand"],
    resource_id(ResourceKind.controlnet, SDVersion.sdxl, ControlMode.hands): ["control-lora-depth-rank", "sai_xl_depth_"],
    resource_id(ResourceKind.ip_adapter, SDVersion.sd15, ControlMode.reference): ["ip-adapter_sd15"],
    resource_id(ResourceKind.ip_adapter, SDVersion.sdxl, ControlMode.reference): ["ip-adapter_sdxl_vit-h"],
    resource_id(ResourceKind.ip_adapter, SDVersion.sd15, ControlMode.face): ["ip-adapter-faceid-plusv2_sd15", "ip-adapter-faceid-plus_sd15"],
    resource_id(ResourceKind.ip_adapter, SDVersion.sdxl, ControlMode.face): ["ip-adapter-faceid-plusv2_sdxl", "ip-adapter-faceid_sdxl"],
    resource_id(ResourceKind.clip_vision, SDVersion.all, "ip_adapter"): ["sd1.5/pytorch_model.bin", "sd1.5/model.safetensors", "clip-vision_vit-h.safetensors", "clip-vit-h-14-laion2b-s32b-b79k"],
    resource_id(ResourceKind.lora, SDVersion.sd15, "lcm"): ["lcm-lora-sdv1-5.safetensors", "lcm/sd1.5/pytorch_lora_weights.safetensors"],
    resource_id(ResourceKind.lora, SDVersion.sdxl, "lcm"): ["lcm-lora-sdxl.safetensors", "lcm/sdxl/pytorch_lora_weights.safetensors"],
    resource_id(ResourceKind.lora, SDVersion.sdxl, "lightning"): ["sdxl_lightning_8step_lora"],
    resource_id(ResourceKind.lora, SDVersion.sd15, ControlMode.face): ["ip-adapter-faceid-plusv2_sd15_lora", "ip-adapter-faceid-plus_sd15_lora"],
    resource_id(ResourceKind.lora, SDVersion.sdxl, ControlMode.face): ["ip-adapter-faceid-plusv2_sdxl_lora", "ip-adapter-faceid_sdxl_lora"],
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.default): [UpscalerName.default.value],
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_2x): [UpscalerName.fast_2x.value],
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_3x): [UpscalerName.fast_3x.value],
    resource_id(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_4x): [UpscalerName.fast_4x.value],
    resource_id(ResourceKind.inpaint, SDVersion.sdxl, "fooocus_head"): ["fooocus_inpaint_head.pth"],
    resource_id(ResourceKind.inpaint, SDVersion.sdxl, "fooocus_patch"): ["inpaint_v26.fooocus.patch"],
    resource_id(ResourceKind.inpaint, SDVersion.all, "default"): ["MAT_Places512_G_fp16", "Places_512_FullData_G", "big-lama.pt"],
}
# fmt: on

required_resource_ids = set(
    [
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.inpaint),
        ResourceId(ResourceKind.controlnet, SDVersion.sd15, ControlMode.blur),
        ResourceId(ResourceKind.ip_adapter, SDVersion.sd15, ControlMode.reference),
        ResourceId(ResourceKind.ip_adapter, SDVersion.sdxl, ControlMode.reference),
        ResourceId(ResourceKind.clip_vision, SDVersion.all, "ip_adapter"),
        ResourceId(ResourceKind.lora, SDVersion.sd15, "lcm"),
        ResourceId(ResourceKind.lora, SDVersion.sdxl, "lcm"),
        ResourceId(ResourceKind.upscaler, SDVersion.all, UpscalerName.default),
        ResourceId(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_2x),
        ResourceId(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_3x),
        ResourceId(ResourceKind.upscaler, SDVersion.all, UpscalerName.fast_4x),
        ResourceId(ResourceKind.inpaint, SDVersion.sdxl, "fooocus_head"),
        ResourceId(ResourceKind.inpaint, SDVersion.sdxl, "fooocus_patch"),
        ResourceId(ResourceKind.inpaint, SDVersion.all, "default"),
    ]
)
