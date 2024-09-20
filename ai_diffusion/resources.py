from __future__ import annotations
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import NamedTuple, Sequence

# Version identifier for all the resources defined here. This is used as the server version.
# It usually follows the plugin version, but not all new plugin versions also require a server update.
version = "1.25.0"

comfy_url = "https://github.com/comfyanonymous/ComfyUI"
comfy_version = "dc96a1ae19b1d714a791f1fcb21578389955bbfd"


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
        "6c563c5032f77dd3336b603bde3d12b415d003ab",
        ["InpaintPreprocessor", "DepthAnythingV2Preprocessor"],
    ),
    CustomNode(
        "IP-Adapter",
        "ComfyUI_IPAdapter_plus",
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        "88a71407c545e4eb0f223294f5b56302ef8696f3",
        ["IPAdapterModelLoader", "IPAdapter"],
    ),
    CustomNode(
        "External Tooling Nodes",
        "comfyui-tooling-nodes",
        "https://github.com/Acly/comfyui-tooling-nodes",
        "1a24975f99b95fa9a17a917de05fa248d8dc9569",
        ["ETN_LoadImageBase64", "ETN_LoadMaskBase64", "ETN_SendImageWebSocket", "ETN_Translate"],
    ),
    CustomNode(
        "Inpaint Nodes",
        "comfyui-inpaint-nodes",
        "https://github.com/Acly/comfyui-inpaint-nodes",
        "6ce66ff1b5ed4e5819b23ccf1feb976ef479528a",
        ["INPAINT_LoadFooocusInpaint", "INPAINT_ApplyFooocusInpaint", "INPAINT_ExpandMask"],
    ),
]

optional_custom_nodes = [
    CustomNode(
        "GGUF",
        "ComfyUI-GGUF",
        "https://github.com/city96/ComfyUI-GGUF",
        "454955ead3336322215a206edbd7191eb130bba0",
        ["UnetLoaderGGUF", "DualCLIPLoaderGGUF"],
    )
]


class Arch(Enum):
    """Diffusion model architectures."""

    sd15 = "SD 1.5"
    sdxl = "SD XL"
    sd3 = "SD 3"
    flux = "Flux"

    auto = "Automatic"
    all = "All"

    @staticmethod
    def from_string(string: str):
        if string == "sd15":
            return Arch.sd15
        if string == "sdxl":
            return Arch.sdxl
        if string == "sd3":
            return Arch.sd3
        if string == "flux" or string == "flux-schnell":
            return Arch.flux
        return None

    @staticmethod
    def from_checkpoint_name(checkpoint: str):
        if Arch.sdxl.matches(checkpoint):
            return Arch.sdxl
        return Arch.sd15

    @staticmethod
    def match(a: Arch, b: Arch):
        if a is Arch.all or b is Arch.all:
            return True
        return a is b

    def matches(self, checkpoint: str):
        # Fallback check if it can't be queried from the server
        xl_in_name = "xl" in checkpoint.lower()
        return self is Arch.auto or ((self is Arch.sdxl) == xl_in_name)

    def resolve(self, checkpoint: str):
        if self is Arch.auto:
            return Arch.sdxl if Arch.sdxl.matches(checkpoint) else Arch.sd15
        return self

    @property
    def has_controlnet_inpaint(self):
        return self is Arch.sd15 or self is Arch.flux

    @property
    def supports_lcm(self):
        return self in [Arch.sd15, Arch.sdxl]

    @property
    def supports_clip_skip(self):
        return self in [Arch.sd15, Arch.sdxl]

    @property
    def supports_attention_guidance(self):
        return self in [Arch.sd15, Arch.sdxl]

    @property
    def text_encoders(self):
        match self:
            case Arch.sd15:
                return ["clip_l"]
            case Arch.sdxl:
                return ["clip_l", "clip_g"]
            case Arch.sd3:
                return ["clip_l", "clip_g"]
            case Arch.flux:
                return ["clip_l", "t5"]
        raise ValueError(f"Unsupported architecture: {self}")

    @staticmethod
    def list():
        return [Arch.sd15, Arch.sdxl, Arch.sd3, Arch.flux]

    @staticmethod
    def list_strings():
        return ["sd15", "sdxl", "sd3", "flux", "flux-schnell"]


class ResourceKind(Enum):
    checkpoint = "Diffusion checkpoint"
    text_encoder = "Text Encoder model"
    vae = "Image Encoder (VAE) model"
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
    universal = 16
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
            ControlMode.universal,
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
    def is_internal(self):  # don't show in control layer mode dropdown
        return self in [ControlMode.inpaint, ControlMode.universal]

    @property
    def is_part_of_image(self):  # not only used as guidance hint
        return self in [ControlMode.reference, ControlMode.line_art, ControlMode.blur]

    @property
    def is_structural(self):  # strong impact on image composition/structure
        return not (self.is_ip_adapter or self is ControlMode.inpaint)

    @property
    def text(self):
        from . import control

        return control.control_mode_text[self]


class ResourceId(NamedTuple):
    kind: ResourceKind
    arch: Arch
    identifier: ControlMode | UpscalerName | str

    @property
    def string(self):
        return resource_id(self.kind, self.arch, self.identifier)

    @property
    def name(self):
        ident = self.identifier.name if isinstance(self.identifier, Enum) else self.identifier
        return f"{self.kind.value} '{ident}' for {self.arch.value}"

    @staticmethod
    def parse(string: str):
        kind, identifier, arch = string.split("-")
        kind = ResourceKind[kind]
        arch = Arch[arch]
        if kind in [ResourceKind.controlnet, ResourceKind.ip_adapter]:
            identifier = ControlMode[identifier]
        elif kind == ResourceKind.upscaler:
            identifier = UpscalerName[identifier]
        return ResourceId(kind, arch, identifier)


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
    def arch(self):
        return self.id.arch

    def __hash__(self):
        return hash(self.id)


required_models = [
    ModelResource(
        "CLIP Vision model",
        ResourceId(ResourceKind.clip_vision, Arch.all, "ip_adapter"),
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
        ResourceId(ResourceKind.upscaler, Arch.all, UpscalerName.default),
        {
            Path(
                "models/upscale_models/4x_NMKD-Superscale-SP_178000_G.pth"
            ): "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth"
        },
    ),
    ModelResource(
        "OmniSR Superscale model",
        ResourceId(ResourceKind.upscaler, Arch.all, UpscalerName.fast_4x),
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
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.inpaint),
        {
            Path(
                "models/controlnet/control_v11p_sd15_inpaint_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors"
        },
    ),
    ModelResource(
        "ControlNet Unblur",
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.blur),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors",
        },
    ),
    ModelResource(
        "IP-Adapter (SD1.5)",
        ResourceId(ResourceKind.ip_adapter, Arch.sd15, ControlMode.reference),
        {
            Path(
                "models/ipadapter/ip-adapter_sd15.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"
        },
    ),
    ModelResource(
        "IP-Adapter (SDXL)",
        ResourceId(ResourceKind.ip_adapter, Arch.sdxl, ControlMode.reference),
        {
            Path(
                "models/ipadapter/ip-adapter_sdxl_vit-h.safetensors"
            ): "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
        },
    ),
    ModelResource(
        "Hyper-SD LoRA (SD1.5)",
        ResourceId(ResourceKind.lora, Arch.sd15, "hyper"),
        {
            Path(
                "models/loras/Hyper-SD15-8steps-CFG-lora.safetensors"
            ): "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-8steps-CFG-lora.safetensors",
        },
    ),
    ModelResource(
        "Hyper-SD LoRA (SDXL)",
        ResourceId(ResourceKind.lora, Arch.sdxl, "hyper"),
        {
            Path(
                "models/loras/Hyper-SDXL-8steps-CFG-lora.safetensors"
            ): "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-8steps-CFG-lora.safetensors",
        },
    ),
    ModelResource(
        "Fooocus Inpaint",
        ResourceId(ResourceKind.inpaint, Arch.sdxl, "fooocus-inpaint"),
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
        ResourceId(ResourceKind.inpaint, Arch.all, "default"),
        {
            Path(
                "models/inpaint/MAT_Places512_G_fp16.safetensors"
            ): "https://huggingface.co/Acly/MAT/resolve/main/MAT_Places512_G_fp16.safetensors",
        },
    ),
    ModelResource(
        "Easy Negative",
        ResourceId(ResourceKind.embedding, Arch.sd15, "easy-negative"),
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
        ResourceId(ResourceKind.checkpoint, Arch.sd15, "realistic-vision"),
        {
            Path(
                "models/checkpoints/realisticVisionV51_v51VAE.safetensors"
            ): "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticVisionV51_v51VAE.safetensors",
        },
    ),
    ModelResource(
        "DreamShaper (Artwork)",
        ResourceId(ResourceKind.checkpoint, Arch.sd15, "dreamshaper"),
        {
            Path(
                "models/checkpoints/dreamshaper_8.safetensors"
            ): "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors",
        },
    ),
    ModelResource(
        "Flat2D AniMerge (Cartoon/Anime)",
        ResourceId(ResourceKind.checkpoint, Arch.sd15, "flat2d-animerge"),
        {
            Path(
                "models/checkpoints/flat2DAnimerge_v45Sharp.safetensors"
            ): "https://huggingface.co/Acly/SD-Checkpoints/resolve/main/flat2DAnimerge_v45Sharp.safetensors"
        },
    ),
    ModelResource(
        "Juggernaut XL",
        ResourceId(ResourceKind.checkpoint, Arch.sdxl, "juggernaut"),
        {
            Path(
                "models/checkpoints/juggernautXL_version6Rundiffusion.safetensors"
            ): "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_version6Rundiffusion.safetensors"
        },
    ),
    ModelResource(
        "ZavyChroma XL",
        ResourceId(ResourceKind.checkpoint, Arch.sdxl, "zavychroma"),
        {
            Path(
                "models/checkpoints/zavychromaxl_v80.safetensors"
            ): "https://huggingface.co/misri/zavychromaxl_v80/resolve/main/zavychromaxl_v80.safetensors"
        },
    ),
    ModelResource(
        "Flux [dev]",
        ResourceId(ResourceKind.checkpoint, Arch.flux, "flux-dev"),
        {
            Path(
                "models/checkpoints/flux1-dev-fp8.safetensors"
            ): "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
        },
    ),
    ModelResource(
        "Flux [schnell]",
        ResourceId(ResourceKind.checkpoint, Arch.flux, "flux-schnell"),
        {
            Path(
                "models/checkpoints/flux1-schnell-fp8.safetensors"
            ): "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors"
        },
    ),
]

upscale_models = [
    ModelResource(
        "HAT Super-Resolution (quality)",
        ResourceId(ResourceKind.upscaler, Arch.all, UpscalerName.quality),
        {
            Path(
                "models/upscale_models/HAT_SRx4_ImageNet-pretrain.pth"
            ): "https://huggingface.co/Acly/hat/resolve/main/HAT_SRx4_ImageNet-pretrain.pth"
        },
    ),
    ModelResource(
        "Real HAT GAN Super-Resolution (sharper)",
        ResourceId(ResourceKind.upscaler, Arch.all, UpscalerName.sharp),
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
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.scribble),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_scribble_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_scribble_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Line Art",
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.line_art),
        {
            Path(
                "models/controlnet/control_v11p_sd15_lineart_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Soft Edge",
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.soft_edge),
        {
            Path(
                "models/controlnet/control_v11p_sd15_softedge_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Canny Edge",
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.canny_edge),
        {
            Path(
                "models/controlnet/control_v11p_sd15_canny_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Depth",
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.depth),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Normal",
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.normal),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Pose",
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.pose),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_openpose_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_openpose_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Segmentation",
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.segmentation),
        {
            Path(
                "models/controlnet/control_lora_rank128_v11p_sd15_seg_fp16.safetensors"
            ): "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11p_sd15_seg_fp16.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Stencil",
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.stencil),
        {
            Path(
                "models/controlnet/control_v1p_sd15_qrcode_monster.safetensors"
            ): "https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/control_v1p_sd15_qrcode_monster.safetensors",
        },
    ),
    ModelResource(
        "ControlNet Hand Refiner",
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.hands),
        {
            Path(
                "models/controlnet/control_sd15_inpaint_depth_hand_fp16.safetensors"
            ): "https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/resolve/main/control_sd15_inpaint_depth_hand_fp16.safetensors",
        },
    ),
    ModelResource(
        "IP-Adapter Face (SD1.5)",
        ResourceId(ResourceKind.ip_adapter, Arch.sd15, ControlMode.face),
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
        "ControlNet Universal (XL)",
        ResourceId(ResourceKind.controlnet, Arch.sdxl, ControlMode.universal),
        {
            Path(
                "models/controlnet/xinsir-controlnet-union-sdxl-1.0-promax.safetensors"
            ): "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors",
        },
    ),
    ModelResource(
        "IP-Adapter Face (XL)",
        ResourceId(ResourceKind.ip_adapter, Arch.sdxl, ControlMode.face),
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
    ModelResource(
        "ControlNet Inpaint (Flux)",
        ResourceId(ResourceKind.controlnet, Arch.flux, ControlMode.inpaint),
        {
            Path(
                "models/controlnet/FLUX.1-dev-Controlnet-Inpainting-Alpha.safetensors"
            ): "https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha/resolve/main/diffusion_pytorch_model.safetensors"
        },
    ),
    ModelResource(
        "ControlNet Lines (Flux)",
        ResourceId(ResourceKind.controlnet, Arch.flux, ControlMode.line_art),
        {
            Path(
                "models/controlnet/mistoline_flux.dev_v1.safetensors"
            ): "https://huggingface.co/TheMistoAI/MistoLine_Flux.dev/resolve/main/mistoline_flux.dev_v1.safetensors"
        },
    ),
    ModelResource(
        "ControlNet Canny (Flux)",
        ResourceId(ResourceKind.controlnet, Arch.flux, ControlMode.canny_edge),
        {
            Path(
                "models/controlnet/flux-canny-controlnet-v3.safetensors"
            ): "https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/flux-canny-controlnet-v3.safetensors"
        },
    ),
    ModelResource(
        "ControlNet Depth (Flux)",
        ResourceId(ResourceKind.controlnet, Arch.flux, ControlMode.depth),
        {
            Path(
                "models/controlnet/flux-depth-controlnet-v3.safetensors"
            ): "https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/flux-depth-controlnet-v3.safetensors"
        },
    ),
]

prefetch_models = [
    ModelResource(
        "Scribble Preprocessor",
        ResourceId(ResourceKind.preprocessor, Arch.all, ControlMode.scribble),
        {
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/table5_pidinet.pth"
            ): "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth"
        },
    ),
    ModelResource(
        "Line Art Preprocessor",
        ResourceId(ResourceKind.preprocessor, Arch.all, ControlMode.line_art),
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
        ResourceId(ResourceKind.preprocessor, Arch.all, ControlMode.soft_edge),
        {
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/TheMistoAI/MistoLine/Anyline/MTEED.pth"
            ): "https://huggingface.co/TheMistoAI/MistoLine/resolve/main/Anyline/MTEED.pth"
        },
    ),
    ModelResource(
        "Depth Preprocessor",
        ResourceId(ResourceKind.preprocessor, Arch.all, ControlMode.depth),
        {
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/depth-anything/Depth-Anything-V2-Base/depth_anything_v2_vitb.pth"
            ): "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
        },
    ),
    ModelResource(
        "Pose Preprocessor",
        ResourceId(ResourceKind.preprocessor, Arch.all, ControlMode.pose),
        {
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/hr16/yolo-nas-fp16/yolo_nas_l_fp16.onnx"
            ): "https://huggingface.co/hr16/yolo-nas-fp16/resolve/main/yolo_nas_l_fp16.onnx",
            Path(
                "custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose/dw-ll_ucoco_384.onnx"
            ): "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
        },
    ),
    ModelResource(
        "NSFW Filter",
        ResourceId(ResourceKind.preprocessor, Arch.all, "safetychecker"),
        {
            Path(
                "custom_nodes/comfyui-tooling-nodes/safetychecker/model.safetensors"
            ): "https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/refs%2Fpr%2F41/model.safetensors"
        },
    ),
]

deprecated_models = [
    ModelResource(
        "LCM-LoRA (SD1.5)",
        ResourceId(ResourceKind.lora, Arch.sd15, "lcm"),
        {
            Path(
                "models/loras/lcm-lora-sdv1-5.safetensors"
            ): "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors",
        },
    ),
    ModelResource(
        "LCM-LoRA (SDXL)",
        ResourceId(ResourceKind.lora, Arch.sdxl, "lcm"),
        {
            Path(
                "models/loras/lcm-lora-sdxl.safetensors"
            ): "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors",
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
                search_path(n.kind, n.arch, n.identifier)
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


def all_models(include_deprecated=False):
    result = chain(
        required_models,
        optional_models,
        default_checkpoints,
        upscale_models,
        prefetch_models,
    )
    if include_deprecated:
        result = chain(result, deprecated_models)
    return result


def resource_id(kind: ResourceKind, arch: Arch, identifier: ControlMode | UpscalerName | str):
    if isinstance(identifier, Enum):
        identifier = identifier.name
    return f"{kind.name}-{identifier}-{arch.name}"


def find_resource(id: ResourceId, include_deprecated=False):
    return next((m for m in all_models(include_deprecated) if m.id == id), None)


def search_path(kind: ResourceKind, arch: Arch, identifier: ControlMode | UpscalerName | str):
    return search_paths.get(resource_id(kind, arch, identifier), None)


def is_required(kind: ResourceKind, arch: Arch, identifier: ControlMode | UpscalerName | str):
    return ResourceId(kind, arch, identifier) in required_resource_ids


# fmt: off
search_paths: dict[str, list[str]] = {
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.inpaint): ["control_v11p_sd15_inpaint"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.inpaint): ["flux.1-dev-controlnet-inpaint"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.universal): ["union-sdxl", "xinsirunion"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.scribble): ["control_v11p_sd15_scribble", "control_lora_rank128_v11p_sd15_scribble"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.scribble): ["xinsirscribble", "scribble-sdxl", "mistoline_fp16", "mistoline_rank", "control-lora-sketch-rank", "sai_xl_sketch_"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.line_art): ["control_v11p_sd15_lineart", "control_lora_rank128_v11p_sd15_lineart"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.line_art): ["xinsirscribble", "mistoline_fp16", "mistoline_rank", "scribble-sdxl", "control-lora-sketch-rank", "sai_xl_sketch_"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.line_art): ["mistoline_flux"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.soft_edge): ["control_v11p_sd15_softedge", "control_lora_rank128_v11p_sd15_softedge"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.soft_edge): ["mistoline_fp16", "mistoline_rank", "xinsirscribble", "scribble-sdxl"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.soft_edge): ["mistoline_flux"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.canny_edge): ["control_v11p_sd15_canny", "control_lora_rank128_v11p_sd15_canny"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.canny_edge): ["xinsircanny", "canny-sdxl" "control-lora-canny-rank", "sai_xl_canny_"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.canny_edge): ["flux-canny", "mistoline_flux"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.depth): ["control_sd15_depth_anything", "control_v11f1p_sd15_depth", "control_lora_rank128_v11f1p_sd15_depth"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.depth): ["xinsirdepth", "depth-sdxl", "control-lora-depth-rank", "sai_xl_depth_"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.depth): ["flux-depth"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.normal): ["control_v11p_sd15_normalbae", "control_lora_rank128_v11p_sd15_normalbae"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.pose): ["control_v11p_sd15_openpose", "control_lora_rank128_v11p_sd15_openpose"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.pose): ["xinsiropenpose", "openpose-sdxl", "control-lora-openposexl2-rank", "thibaud_xl_openpose"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.segmentation): ["control_v11p_sd15_seg", "control_lora_rank128_v11p_sd15_seg"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.segmentation): ["sdxl_segmentation_ade20k_controlnet"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.blur): ["control_v11f1e_sd15_tile", "control_lora_rank128_v11f1e_sd15_tile"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.blur): ["xinsirtile", "tile-sdxl", "ttplanetsdxlcontrolnet", "ttplanet_sdxl_controlnet_tile_realistic", "ttplanet_controlnet_tile_realistic"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.stencil): ["control_v1p_sd15_qrcode_monster"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.stencil): ["sdxl_qrcode_monster"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.hands): ["control_sd15_inpaint_depth_hand"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.hands): ["control-lora-depth-rank", "sai_xl_depth_"],
    resource_id(ResourceKind.ip_adapter, Arch.sd15, ControlMode.reference): ["ip-adapter_sd15"],
    resource_id(ResourceKind.ip_adapter, Arch.sdxl, ControlMode.reference): ["ip-adapter_sdxl_vit-h"],
    resource_id(ResourceKind.ip_adapter, Arch.sd15, ControlMode.face): ["ip-adapter-faceid-plusv2_sd15", "ip-adapter-faceid-plus_sd15"],
    resource_id(ResourceKind.ip_adapter, Arch.sdxl, ControlMode.face): ["ip-adapter-faceid-plusv2_sdxl", "ip-adapter-faceid_sdxl"],
    resource_id(ResourceKind.clip_vision, Arch.all, "ip_adapter"): ["sd1.5/pytorch_model.bin", "sd1.5/model.safetensors", "clip-vision_vit-h.safetensors", "clip-vit-h-14-laion2b-s32b-b79k"],
    resource_id(ResourceKind.lora, Arch.sd15, "lcm"): ["lcm-lora-sdv1-5.safetensors", "lcm/sd1.5/pytorch_lora_weights.safetensors"],
    resource_id(ResourceKind.lora, Arch.sdxl, "lcm"): ["lcm-lora-sdxl.safetensors", "lcm/sdxl/pytorch_lora_weights.safetensors"],
    resource_id(ResourceKind.lora, Arch.sdxl, "lightning"): ["sdxl_lightning_8step_lora"],
    resource_id(ResourceKind.lora, Arch.sd15, "hyper"): ["Hyper-SD15-8steps-CFG-lora"],
    resource_id(ResourceKind.lora, Arch.sdxl, "hyper"): ["Hyper-SDXL-8steps-CFG-lora"],
    resource_id(ResourceKind.lora, Arch.sd15, ControlMode.face): ["ip-adapter-faceid-plusv2_sd15_lora", "ip-adapter-faceid-plus_sd15_lora"],
    resource_id(ResourceKind.lora, Arch.sdxl, ControlMode.face): ["ip-adapter-faceid-plusv2_sdxl_lora", "ip-adapter-faceid_sdxl_lora"],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.default): [UpscalerName.default.value],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_2x): [UpscalerName.fast_2x.value],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_3x): [UpscalerName.fast_3x.value],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_4x): [UpscalerName.fast_4x.value],
    resource_id(ResourceKind.inpaint, Arch.sdxl, "fooocus_head"): ["fooocus_inpaint_head.pth"],
    resource_id(ResourceKind.inpaint, Arch.sdxl, "fooocus_patch"): ["inpaint_v26.fooocus.patch"],
    resource_id(ResourceKind.inpaint, Arch.all, "default"): ["MAT_Places512_G_fp16", "Places_512_FullData_G", "big-lama.pt"],
    resource_id(ResourceKind.text_encoder, Arch.all, "clip_l"): ["clip_l"],
    resource_id(ResourceKind.text_encoder, Arch.all, "clip_g"): ["clip_g"],
    resource_id(ResourceKind.text_encoder, Arch.all, "t5"): ["t5"],
    resource_id(ResourceKind.vae, Arch.sd15, "default"): ["vae-ft-mse-840000-ema"],
    resource_id(ResourceKind.vae, Arch.sdxl, "default"): ["sdxl_vae"],
    resource_id(ResourceKind.vae, Arch.flux, "default"): ["ae.s"],
}
# fmt: on

required_resource_ids = set(
    [
        ResourceId(ResourceKind.text_encoder, Arch.sd3, "clip_l"),
        ResourceId(ResourceKind.text_encoder, Arch.sd3, "clip_g"),
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.inpaint),
        ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.blur),
        ResourceId(ResourceKind.ip_adapter, Arch.sd15, ControlMode.reference),
        ResourceId(ResourceKind.ip_adapter, Arch.sdxl, ControlMode.reference),
        ResourceId(ResourceKind.clip_vision, Arch.all, "ip_adapter"),
        ResourceId(ResourceKind.lora, Arch.sd15, "hyper"),
        ResourceId(ResourceKind.lora, Arch.sdxl, "hyper"),
        ResourceId(ResourceKind.upscaler, Arch.all, UpscalerName.default),
        ResourceId(ResourceKind.upscaler, Arch.all, UpscalerName.fast_2x),
        ResourceId(ResourceKind.upscaler, Arch.all, UpscalerName.fast_3x),
        ResourceId(ResourceKind.upscaler, Arch.all, UpscalerName.fast_4x),
        ResourceId(ResourceKind.inpaint, Arch.sdxl, "fooocus_head"),
        ResourceId(ResourceKind.inpaint, Arch.sdxl, "fooocus_patch"),
        ResourceId(ResourceKind.inpaint, Arch.all, "default"),
    ]
)
