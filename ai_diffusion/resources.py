from __future__ import annotations
from enum import Enum
from itertools import chain
import json
import hashlib
from pathlib import Path
from typing import Any, NamedTuple, Sequence

# Version identifier for all the resources defined here. This is used as the server version.
# It usually follows the plugin version, but not all new plugin versions also require a server update.
version = "1.35.0"

comfy_url = "https://github.com/comfyanonymous/ComfyUI"
comfy_version = "5a87757ef96f807cf1cf5b41c55a0a84c9551f20"


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
        "83463c2e4b04e729268e57f638b4212e0da4badc",
        ["InpaintPreprocessor", "DepthAnythingV2Preprocessor"],
    ),
    CustomNode(
        "IP-Adapter",
        "ComfyUI_IPAdapter_plus",
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        "b188a6cb39b512a9c6da7235b880af42c78ccd0d",
        ["IPAdapterModelLoader", "IPAdapter"],
    ),
    CustomNode(
        "External Tooling Nodes",
        "comfyui-tooling-nodes",
        "https://github.com/Acly/comfyui-tooling-nodes",
        "ca2b59248ece910579b55a1512b0a846928259ea",
        ["ETN_LoadImageBase64", "ETN_LoadMaskBase64", "ETN_SendImageWebSocket", "ETN_Translate"],
    ),
    CustomNode(
        "Inpaint Nodes",
        "comfyui-inpaint-nodes",
        "https://github.com/Acly/comfyui-inpaint-nodes",
        "b9039c22de926919f26b7242cfa4da00d8b6fbec",
        ["INPAINT_LoadFooocusInpaint", "INPAINT_ApplyFooocusInpaint", "INPAINT_ExpandMask"],
    ),
]

optional_custom_nodes = [
    CustomNode(
        "GGUF",
        "ComfyUI-GGUF",
        "https://github.com/city96/ComfyUI-GGUF",
        "a2b75978fd50c0227a58316619b79d525b88e570",
        ["UnetLoaderGGUF", "DualCLIPLoaderGGUF"],
    ),
    CustomNode(
        "WaveSpeed",
        "Comfy-WaveSpeed",
        "https://github.com/chengzeyi/Comfy-WaveSpeed",
        "16ec6f344f8cecbbf006d374043f85af22b7a51d",
        ["ApplyFBCacheOnModel"],
    ),
]


class Arch(Enum):
    """Diffusion model architectures."""

    sd15 = "SD 1.5"
    sdxl = "SD XL"
    sd3 = "SD 3"
    flux = "Flux"
    illu = "Illustrious"
    illu_v = "Illustrious v-prediction"

    auto = "Automatic"
    all = "All"

    @staticmethod
    def from_string(string: str, model_type: str = "eps"):
        if string == "sd15":
            return Arch.sd15
        if string == "sdxl" and model_type == "v-prediction":
            return Arch.illu_v
        elif string == "sdxl":
            return Arch.sdxl
        if string == "sd3":
            return Arch.sd3
        if string == "flux" or string == "flux-schnell":
            return Arch.flux
        if string == "illu":
            return Arch.illu
        if string == "illu_v":
            return Arch.illu_v
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
        return self in [Arch.sd15, Arch.sdxl, Arch.illu, Arch.illu_v]

    @property
    def supports_attention_guidance(self):
        return self in [Arch.sd15, Arch.sdxl, Arch.illu, Arch.illu_v]

    @property
    def is_sdxl_like(self):
        # illustrious technically uses sdxl architecture, but has a separate ecosystem
        return self in [Arch.sdxl, Arch.illu, Arch.illu_v]

    @property
    def text_encoders(self):
        match self:
            case Arch.sd15:
                return ["clip_l"]
            case Arch.sdxl | Arch.illu | Arch.illu_v:
                return ["clip_l", "clip_g"]
            case Arch.sd3:
                return ["clip_l", "clip_g"]
            case Arch.flux:
                return ["clip_l", "t5"]
        raise ValueError(f"Unsupported architecture: {self}")

    @staticmethod
    def list():
        return [Arch.sd15, Arch.sdxl, Arch.sd3, Arch.flux, Arch.illu, Arch.illu_v]

    @staticmethod
    def list_strings():
        return ["sd15", "sdxl", "sd3", "flux", "flux-schnell"]


class ResourceKind(Enum):
    checkpoint = "Diffusion checkpoint"
    text_encoder = "Text Encoder"
    vae = "Image Encoder (VAE)"
    controlnet = "ControlNet"
    clip_vision = "CLIP Vision"
    ip_adapter = "IP-Adapter"
    lora = "LoRA"
    upscaler = "Upscale"
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
        return self.is_control_net and self not in [
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

    def can_substitute_universal(self, arch: Arch):
        """True if this control mode is covered by univeral control-net."""
        if arch == Arch.sdxl:
            return self in [
                ControlMode.inpaint,
                ControlMode.scribble,
                ControlMode.line_art,
                ControlMode.soft_edge,
                ControlMode.canny_edge,
                ControlMode.depth,
                ControlMode.normal,
                ControlMode.pose,
                ControlMode.segmentation,
                ControlMode.blur,
                ControlMode.hands,  # same as depth
            ]
        if arch == Arch.flux:
            return self in [
                ControlMode.line_art,
                ControlMode.soft_edge,
                ControlMode.canny_edge,
                ControlMode.depth,
                ControlMode.pose,
                ControlMode.blur,
            ]
        return False


def resource_id(kind: ResourceKind, arch: Arch, identifier: ControlMode | UpscalerName | str):
    if isinstance(identifier, Enum):
        identifier = identifier.name
    return f"{kind.name}-{identifier}-{arch.name}"


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
        if kind in [ResourceKind.controlnet, ResourceKind.ip_adapter, ResourceKind.preprocessor]:
            if identifier in ControlMode.__members__:
                identifier = ControlMode[identifier]
        elif kind == ResourceKind.upscaler:
            identifier = UpscalerName[identifier]
        return ResourceId(kind, arch, identifier)


class VerificationState(Enum):
    not_verified = 0
    in_progress = 1
    verified = 2
    mismatch = 3
    error = 4


class VerificationStatus(NamedTuple):
    state: VerificationState
    file: ModelFile
    info: str | None = None


class ModelRequirements(Enum):
    none = 0
    insightface = 1


class ModelFile(NamedTuple):
    path: Path
    url: str
    id: ResourceId
    sha256: str | None = None

    @property
    def name(self):
        return self.path.name

    @staticmethod
    def parse(data: dict[str, Any], parent_id: ResourceId):
        id = ResourceId.parse(data.get("id", parent_id.string))
        sha256 = data.get("sha256", None)
        return ModelFile(Path(data["path"]), data["url"], id, sha256)

    def as_dict(self, with_id=True):
        result = {
            "id": self.id.string,
            "path": str(self.path.as_posix()),
            "url": self.url,
        }
        if not with_id:
            del result["id"]
        if self.sha256:
            result["sha256"] = self.sha256
        return result

    def verify(self, base_dir: Path):
        if self.sha256 is None:
            return VerificationStatus(VerificationState.not_verified, self)

        try:
            file_path = base_dir / self.path
            assert file_path.exists(), f"File {file_path} does not exist"

            actual_sha256 = compute_sha256(file_path)
            if actual_sha256 == self.sha256:
                return VerificationStatus(VerificationState.verified, self)
            else:
                return VerificationStatus(VerificationState.mismatch, self, actual_sha256)
        except Exception as e:
            return VerificationStatus(VerificationState.error, self, str(e))


class ModelResource(NamedTuple):
    name: str
    id: ResourceId
    files: list[ModelFile]
    alternatives: list[Path] | None = None  # for backwards compatibility
    requirements: ModelRequirements = ModelRequirements.none

    @property
    def filename(self):
        assert len(self.files) == 1
        return self.files[0].name

    @property
    def folder(self):
        return self.files[0].path.parent

    @property
    def url(self):
        assert len(self.files) == 1
        return self.files[0].url

    def exists_in(self, path: Path):
        exact = all((path / file.path).exists() for file in self.files)
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

    def as_dict(self):
        result = {
            "id": self.id.string,
            "name": self.name,
            "files": [f.as_dict(len(self.files) > 1) for f in self.files],
        }
        if self.alternatives:
            result["alternatives"] = [str(p.as_posix()) for p in self.alternatives]
        if self.requirements is not ModelRequirements.none:
            result["requirements"] = self.requirements.name
        return result

    @staticmethod
    def from_dict(data: dict[str, Any]):
        id = ResourceId.parse(data["id"])
        files = [ModelFile.parse(f, id) for f in data["files"]]
        alternatives = [Path(p) for p in data.get("alternatives", [])]
        requirements = (
            ModelRequirements[data["requirements"]]
            if "requirements" in data
            else ModelRequirements.none
        )
        return ModelResource(data["name"], id, files, alternatives, requirements)

    @staticmethod
    def from_list(data: list[dict[str, Any]]):
        return [ModelResource.from_dict(d) for d in data]

    def verify(self, base_dir: Path):
        for file in self.files:
            file_path = base_dir / file.path
            if file_path.exists():
                yield VerificationStatus(VerificationState.in_progress, file)
                yield file.verify(base_dir)


_models_file = Path(__file__).parent / "presets" / "models.json"
_models_dict = json.loads(_models_file.read_text())

required_models = ModelResource.from_list(_models_dict["required"])
default_checkpoints = ModelResource.from_list(_models_dict["checkpoints"])
upscale_models = ModelResource.from_list(_models_dict["upscale"])
optional_models = ModelResource.from_list(_models_dict["optional"])
prefetch_models = ModelResource.from_list(_models_dict["prefetch"])
deprecated_models = ModelResource.from_list(_models_dict["deprecated"])

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


def find_resource(id: ResourceId, include_deprecated=False):
    return next(
        (m for m in all_models(include_deprecated) if any(f.id == id for f in m.files)), None
    )


def compute_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def update_model_checksums(models_path: Path):
    modified = False

    categories = ["required", "checkpoints", "upscale", "optional", "prefetch", "deprecated"]
    for category in categories:
        for model_idx, model in enumerate(_models_dict[category]):
            for file_idx, file in enumerate(model["files"]):
                file_path = models_path / file["path"]
                if not file_path.exists():
                    print(f"Not found: {file_path}")
                    continue
                if "sha256" not in file:
                    try:
                        checksum = compute_sha256(file_path)
                        _models_dict[category][model_idx]["files"][file_idx]["sha256"] = checksum
                        print(f"Added checksum for {file['path']}: {checksum}")
                        modified = True
                    except Exception as e:
                        print(f"Error computing checksum for {file['path']}: {e}")

    if modified:
        with open(_models_file, "w") as f:
            json.dump(_models_dict, f, indent=2)
        print(f"Updated checksums written to {_models_file}")


def verify_model_integrity(base_dir: Path | None = None):
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent

    for model in all_models():
        yield from model.verify(base_dir)


def search_path(kind: ResourceKind, arch: Arch, identifier: ControlMode | UpscalerName | str):
    return search_paths.get(resource_id(kind, arch, identifier), None)


def is_required(kind: ResourceKind, arch: Arch, identifier: ControlMode | UpscalerName | str):
    return ResourceId(kind, arch, identifier) in required_resource_ids


# fmt: off
search_paths: dict[str, list[str]] = {
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.inpaint): ["control_v11p_sd15_inpaint"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.inpaint): ["flux.1-dev-controlnet-inpaint"],
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.inpaint): ["noobaiinpainting"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.universal): ["union-sdxl", "xinsirunion"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.universal): ["flux.1-dev-controlnet-union-pro-2.0", "flux.1-dev-controlnet-union-pro", "flux.1-dev-controlnet-union", "flux1devcontrolnetunion"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.scribble): ["control_v11p_sd15_scribble", "control_lora_rank128_v11p_sd15_scribble"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.scribble): ["xinsirscribble", "scribble-sdxl", "mistoline_fp16", "mistoline_rank", "control-lora-sketch-rank", "sai_xl_sketch_"],
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.scribble): ["noob-sdxl-controlnet-scribble_pidinet", "noobaixlcontrolnet_epsscribble", "noob-sdxl-controlnet-scribble"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.line_art): ["control_v11p_sd15_lineart", "control_lora_rank128_v11p_sd15_lineart"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.line_art): ["xinsirscribble", "mistoline_fp16", "mistoline_rank", "scribble-sdxl", "control-lora-sketch-rank", "sai_xl_sketch_"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.line_art): ["mistoline_flux"],
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.line_art): ["noob-sdxl-controlnet-lineart_anime", "noobaixlcontrolnet_epslineart", "noob-sdxl-controlnet-lineart"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.soft_edge): ["control_v11p_sd15_softedge", "control_lora_rank128_v11p_sd15_softedge"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.soft_edge): ["mistoline_fp16", "mistoline_rank", "xinsirscribble", "scribble-sdxl"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.soft_edge): ["mistoline_flux"],
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.soft_edge): ["noob-sdxl-controlnet-softedge", "noobaixlcontrolnet_epssoftedge"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.canny_edge): ["control_v11p_sd15_canny", "control_lora_rank128_v11p_sd15_canny"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.canny_edge): ["xinsircanny", "canny-sdxl" "control-lora-canny-rank", "sai_xl_canny_"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.canny_edge): ["flux-canny", "mistoline_flux"],
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.canny_edge): ["noob_sdxl_controlnet_canny", "noobaixlcontrolnet_epscanny"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.depth): ["control_sd15_depth_anything", "control_v11f1p_sd15_depth", "control_lora_rank128_v11f1p_sd15_depth"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.depth): ["xinsirdepth", "depth-sdxl", "control-lora-depth-rank", "sai_xl_depth_"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.depth): ["flux-depth"],
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.depth): ["noob-sdxl-controlnet-depth", "noobaixlcontrolnet_epsdepth"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.normal): ["control_v11p_sd15_normalbae", "control_lora_rank128_v11p_sd15_normalbae"],
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.normal): ["noob-sdxl-controlnet-normal", "noobaixlcontrolnet_epsnormal"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.pose): ["control_v11p_sd15_openpose", "control_lora_rank128_v11p_sd15_openpose"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.pose): ["xinsiropenpose", "openpose-sdxl", "control-lora-openposexl2-rank", "thibaud_xl_openpose"],
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.pose): ["noob-sdxl-controlnet-openpose", "noobaixlcontrolnet_openpose"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.segmentation): ["control_v11p_sd15_seg", "control_lora_rank128_v11p_sd15_seg"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.segmentation): ["sdxl_segmentation_ade20k_controlnet"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.blur): ["control_v11f1e_sd15_tile", "control_lora_rank128_v11f1e_sd15_tile"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.blur): ["xinsirtile", "tile-sdxl", "ttplanetsdxlcontrolnet", "ttplanet_sdxl_controlnet_tile_realistic", "ttplanet_controlnet_tile_realistic"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.blur): ["flux.1-dev-controlnet-upscale"],
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.blur): ["noob-sdxl-controlnet-tile", "noobaixlcontrolnet_epstile"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.stencil): ["control_v1p_sd15_qrcode_monster"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.stencil): ["sdxl_qrcode_monster"],
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.hands): ["control_sd15_inpaint_depth_hand"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.hands): ["control-lora-depth-rank", "sai_xl_depth_"],
    resource_id(ResourceKind.ip_adapter, Arch.sd15, ControlMode.reference): ["ip-adapter_sd15"],
    resource_id(ResourceKind.ip_adapter, Arch.sdxl, ControlMode.reference): ["ip-adapter_sdxl_vit-h"],
    resource_id(ResourceKind.ip_adapter, Arch.flux, ControlMode.reference): ["flux1-redux-dev"],
    resource_id(ResourceKind.ip_adapter, Arch.illu, ControlMode.reference): ["noobipa"],
    resource_id(ResourceKind.ip_adapter, Arch.sd15, ControlMode.face): ["ip-adapter-faceid-plusv2_sd15", "ip-adapter-faceid-plus_sd15"],
    resource_id(ResourceKind.ip_adapter, Arch.sdxl, ControlMode.face): ["ip-adapter-faceid-plusv2_sdxl", "ip-adapter-faceid_sdxl"],
    resource_id(ResourceKind.clip_vision, Arch.all, "ip_adapter"): ["sd1.5/pytorch_model.bin", "sd1.5/model.safetensors", "clip-vision_vit-h.safetensors", "clip-vit-h-14-laion2b-s32b-b79k"],
    resource_id(ResourceKind.clip_vision, Arch.flux, "redux"): ["sigclip_vision_patch14_384"],
    resource_id(ResourceKind.clip_vision, Arch.illu, "ip_adapter"): ["clip-vit-bigg", "clip_vision_g", "clip-vision_vit-g"],
    resource_id(ResourceKind.lora, Arch.sd15, "lcm"): ["lcm-lora-sdv1-5.safetensors", "lcm/sd1.5/pytorch_lora_weights.safetensors"],
    resource_id(ResourceKind.lora, Arch.sdxl, "lcm"): ["lcm-lora-sdxl.safetensors", "lcm/sdxl/pytorch_lora_weights.safetensors"],
    resource_id(ResourceKind.lora, Arch.sdxl, "lightning"): ["sdxl_lightning_8step_lora"],
    resource_id(ResourceKind.lora, Arch.sd15, "hyper"): ["Hyper-SD15-8steps-CFG-lora"],
    resource_id(ResourceKind.lora, Arch.sdxl, "hyper"): ["Hyper-SDXL-8steps-CFG-lora"],
    resource_id(ResourceKind.lora, Arch.sd15, ControlMode.face): ["ip-adapter-faceid-plusv2_sd15_lora", "ip-adapter-faceid-plus_sd15_lora"],
    resource_id(ResourceKind.lora, Arch.sdxl, ControlMode.face): ["ip-adapter-faceid-plusv2_sdxl_lora", "ip-adapter-faceid_sdxl_lora"],
    resource_id(ResourceKind.lora, Arch.flux, ControlMode.depth): ["flux1-depth"],
    resource_id(ResourceKind.lora, Arch.flux, ControlMode.canny_edge): ["flux1-canny"],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.default): [UpscalerName.default.value],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_2x): [UpscalerName.fast_2x.value],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_3x): [UpscalerName.fast_3x.value],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_4x): [UpscalerName.fast_4x.value],
    resource_id(ResourceKind.inpaint, Arch.sdxl, "fooocus_head"): ["fooocus_inpaint_head.pth"],
    resource_id(ResourceKind.inpaint, Arch.sdxl, "fooocus_patch"): ["inpaint_v26.fooocus"],
    resource_id(ResourceKind.inpaint, Arch.all, "default"): ["MAT_Places512_G_fp16", "Places_512_FullData_G", "big-lama.pt"],
    resource_id(ResourceKind.text_encoder, Arch.all, "clip_l"): ["clip_l"],
    resource_id(ResourceKind.text_encoder, Arch.all, "clip_g"): ["clip_g"],
    resource_id(ResourceKind.text_encoder, Arch.all, "t5"): ["t5"],
    resource_id(ResourceKind.vae, Arch.sd15, "default"): ["vae-ft-mse-840000-ema"],
    resource_id(ResourceKind.vae, Arch.sdxl, "default"): ["sdxl_vae"],
    resource_id(ResourceKind.vae, Arch.sd3, "default"): ["sd3"],
    resource_id(ResourceKind.vae, Arch.flux, "default"): ["flux", "ae.s"],
}
# fmt: on

required_resource_ids = set([
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
])
