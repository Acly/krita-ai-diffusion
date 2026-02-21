from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, NamedTuple

# Version identifier for all the resources defined here. This is used as the server version.
# It usually follows the plugin version, but not all new plugin versions also require a server update.
version = "1.48.0"

comfy_url = "https://github.com/comfyanonymous/ComfyUI"
comfy_version = "fe52843fe55b92dedaabff684294dd7a115d2204"


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
        "ed99942f861ff63af56c20f8ef4ff473e452c3c6",
        ["ETN_LoadImageCache", "ETN_SaveImageCache", "ETN_Translate"],
    ),
    CustomNode(
        "Inpaint Nodes",
        "comfyui-inpaint-nodes",
        "https://github.com/Acly/comfyui-inpaint-nodes",
        "d74ecec6c377073a6885697f07a019d050f0d545",
        [
            "INPAINT_LoadFooocusInpaint",
            "INPAINT_ShrinkMask",
            "INPAINT_StabilizeMask",
            "INPAINT_ColorMatch",
        ],
    ),
]

optional_custom_nodes = [
    CustomNode(
        "GGUF",
        "ComfyUI-GGUF",
        "https://github.com/city96/ComfyUI-GGUF",
        "01f8845bf30d89fff293c7bd50187bc59d9d53ea",
        ["UnetLoaderGGUF", "DualCLIPLoaderGGUF"],
    ),
    CustomNode(
        "Nunchaku",
        "ComfyUI-nunchaku",
        "https://github.com/nunchaku-tech/ComfyUI-nunchaku",
        "90999af9c26e4a40927fb26c028ece8875ac25b3",
        ["NunchakuFluxDiTLoader"],
    ),
]


class Arch(Enum):
    """Diffusion model architectures."""

    sd15 = "SD 1.5"
    sdxl = "SD XL"
    sd3 = "SD 3"
    flux = "Flux"
    flux_k = "Flux Kontext"
    flux2_4b = "Flux 2 Klein 4B"
    flux2_9b = "Flux 2 Klein 9B"
    illu = "Illustrious"
    illu_v = "Illustrious v-prediction"
    chroma = "Chroma"
    qwen = "Qwen"
    qwen_e = "Qwen Edit"
    qwen_e_p = "Qwen Edit Plus"
    qwen_l = "Qwen Layered"
    zimage = "Z-Image"

    auto = "Automatic"
    all = "All"

    @staticmethod
    def from_string(string: str, model_type: str = "eps", filename: str | None = None):
        filename = filename.lower() if filename else ""
        if string == "sd15":
            return Arch.sd15
        if string == "sdxl" and model_type == "v-prediction":
            return Arch.illu_v
        elif string == "sdxl":
            return Arch.sdxl
        if string == "sd3":
            return Arch.sd3
        if string == "flux" and "kontext" in filename:
            return Arch.flux_k
        if string in {"flux", "flux-schnell"}:
            return Arch.flux
        if string == "flux2_4b" or (string == "flux2" and model_type == "klein-4b"):
            return Arch.flux2_4b
        if string == "flux2_9b" or (string == "flux2" and model_type == "klein-9b"):
            return Arch.flux2_9b
        if string == "illu":
            return Arch.illu
        if string == "illu_v":
            return Arch.illu_v
        if string == "chroma":
            return Arch.chroma
        if string == "qwen-image" and "edit" in filename:
            if "2509" in filename or "2511" in filename:
                return Arch.qwen_e_p
            else:
                return Arch.qwen_e
        if string == "qwen-image" and "layered" in filename:
            return Arch.qwen_l
        if string == "qwen-image":
            return Arch.qwen
        if string in {"z-image", "zimage"}:
            return Arch.zimage
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

    @staticmethod
    def is_compatible(a: Arch, b: Arch):
        return (
            a is b
            or (a.is_sdxl_like and b.is_sdxl_like)
            or (a.is_flux_like and b.is_flux_like)
            or (a.is_qwen_like and b.is_qwen_like)
        )

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
        return self in (Arch.sd15, Arch.flux, Arch.zimage, Arch.qwen)

    @property
    def supports_regions(self):
        return self in [Arch.sd15, Arch.sdxl, Arch.illu, Arch.illu_v]

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
    def supports_cfg(self):
        return self not in [Arch.flux, Arch.flux_k]

    @property
    def is_edit(self):  # edit models make changes to input images
        return self in [Arch.flux_k, Arch.qwen_e, Arch.qwen_e_p, Arch.qwen_l]

    @property
    def supports_edit(self):  # includes text-to-image models that can also edit
        return self.is_edit or self.is_flux2

    @property
    def is_sdxl_like(self):
        # illustrious technically uses sdxl architecture, but has a separate ecosystem
        return self in [Arch.sdxl, Arch.illu, Arch.illu_v]

    @property
    def is_flux_like(self):
        return self in [Arch.flux, Arch.flux_k]

    @property
    def is_flux2(self):
        return self in [Arch.flux2_4b, Arch.flux2_9b]

    @property
    def is_qwen_like(self):
        return self in [Arch.qwen, Arch.qwen_e, Arch.qwen_e_p, Arch.qwen_l]

    @property
    def text_encoders(self):
        match self:
            case Arch.sd15:
                return ["clip_l"]
            case Arch.sdxl | Arch.illu | Arch.illu_v:
                return ["clip_l", "clip_g"]
            case Arch.sd3:
                return ["clip_l", "clip_g"]
            case Arch.flux | Arch.flux_k:
                return ["clip_l", "t5"]
            case Arch.flux2_4b:
                return ["qwen_3_4b"]
            case Arch.flux2_9b:
                return ["qwen_3_8b"]
            case Arch.chroma:
                return ["t5"]
            case Arch.qwen | Arch.qwen_e | Arch.qwen_e_p | Arch.qwen_l:
                return ["qwen"]
            case Arch.zimage:
                return ["qwen_3_4b"]
        raise ValueError(f"Unsupported architecture: {self}")

    @property
    def latent_compression_factor(self):
        return 16 if self.is_flux2 or self is Arch.sd3 else 8

    @staticmethod
    def list():
        return [
            Arch.sd15,
            Arch.sdxl,
            Arch.sd3,
            Arch.flux,
            Arch.flux_k,
            Arch.flux2_4b,
            Arch.flux2_9b,
            Arch.illu,
            Arch.illu_v,
            Arch.chroma,
            Arch.qwen,
            Arch.qwen_e,
            Arch.qwen_e_p,
            Arch.qwen_l,
            Arch.zimage,
        ]


class ResourceKind(Enum):
    checkpoint = "Diffusion checkpoint"
    text_encoder = "Text Encoder"
    vae = "Image Encoder (VAE)"
    controlnet = "ControlNet"
    clip_vision = "CLIP Vision"
    ip_adapter = "IP-Adapter"
    lora = "LoRA"
    model_patch = "Model Patch"
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
        if arch.is_sdxl_like or arch is Arch.qwen:
            return self in [
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
        if arch is Arch.zimage:
            return self in [
                ControlMode.inpaint,
                ControlMode.soft_edge,
                ControlMode.canny_edge,
                ControlMode.depth,
                ControlMode.pose,
            ]
        return False

    def can_substitute_instruction(self, arch: Arch):
        """True if this control mode is covered by instruction-following edit models."""
        if arch.is_flux2:
            return self in [
                ControlMode.style,
                ControlMode.composition,
                ControlMode.face,
                ControlMode.scribble,
                ControlMode.line_art,
                ControlMode.canny_edge,
                ControlMode.depth,
                ControlMode.pose,
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
    cuda = 2  # requires CUDA (NVIDIA only)
    cuda_fp4 = 3  # requires FP4 support (Blackwell)
    no_cuda = 4  # model alternative for hardware without CUDA support


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

    @property
    def file_id(self):
        return self.files[0].url  # unique identifier for the file(s)

    def __hash__(self):
        return hash(self.id)

    def as_dict(self, ids: list[ResourceId] | None = None):
        if ids is None:
            id = self.id.string
        elif len(ids) == 1:
            id = ids[0].string
        else:
            id = [id.string for id in ids]
        result = {
            "id": id,
            "name": self.name,
            "files": [f.as_dict(len(self.files) > 1) for f in self.files],
        }
        if self.alternatives:
            result["alternatives"] = [str(p.as_posix()) for p in self.alternatives]
        if self.requirements is not ModelRequirements.none:
            result["requirements"] = self.requirements.name
        return result

    @staticmethod
    def as_list(models: Sequence[ModelResource]):
        result = []
        i = 0
        while i < len(models):
            m = models[i]
            ids = [m.id]
            for j in range(i + 1, len(models)):
                if m.name == models[j].name:
                    ids.append(models[j].id)
                else:
                    break
            result.append(m.as_dict(ids))
            i += len(ids)
        return result

    @staticmethod
    def from_dict(data: dict[str, Any]):
        id_list = data["id"]
        if not isinstance(id_list, list):
            id_list = [id_list]
        for id_str in id_list:
            id = ResourceId.parse(id_str)
            files = [ModelFile.parse(f, id) for f in data["files"]]
            alternatives = [Path(p) for p in data.get("alternatives", [])]
            requirements = (
                ModelRequirements[data["requirements"]]
                if "requirements" in data
                else ModelRequirements.none
            )
            yield ModelResource(data["name"], id, files, alternatives, requirements)

    @staticmethod
    def from_list(data: list[dict[str, Any]]):
        return [m for d in data for m in ModelResource.from_dict(d)]

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
    + [m.id.string for m in required_models]
    + [c.id.string for c in default_checkpoints]
    + [m.id.string for m in upscale_models]
    + [m.id.string for m in optional_models]
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


def get_resource(id: ResourceId, include_deprecated=False):
    if resource := find_resource(id, include_deprecated):
        return resource
    raise ValueError(f"Resource {id.string} not found")


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

    verified = set()
    for model in all_models():
        if model.file_id not in verified:
            yield from model.verify(base_dir)
            verified.add(model.file_id)


def search_path(kind: ResourceKind, arch: Arch, identifier: ControlMode | UpscalerName | str):
    return search_paths.get(resource_id(kind, arch, identifier), None)


def is_required(kind: ResourceKind, arch: Arch, identifier: ControlMode | UpscalerName | str):
    return ResourceId(kind, arch, identifier) in required_resource_ids


# fmt: off
search_paths: dict[str, list[str]] = {
    resource_id(ResourceKind.controlnet, Arch.sd15, ControlMode.inpaint): ["control_v11p_sd15_inpaint"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.inpaint): ["flux.1-dev-controlnet-inpaint"],
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.inpaint): ["noobaiinpainting"],
    resource_id(ResourceKind.controlnet, Arch.qwen, ControlMode.inpaint): ["qwen-image-instantx-controlnet-inpainting"],
    resource_id(ResourceKind.controlnet, Arch.sdxl, ControlMode.universal): ["union-sdxl", "xinsirunion"],
    resource_id(ResourceKind.controlnet, Arch.illu, ControlMode.universal): ["union-sdxl", "xinsirunion"],
    resource_id(ResourceKind.controlnet, Arch.illu_v, ControlMode.universal): ["union-sdxl", "xinsirunion"],
    resource_id(ResourceKind.controlnet, Arch.flux, ControlMode.universal): ["flux.1-dev-controlnet-union-pro-2.0", "flux.1-dev-controlnet-union-pro", "flux.1-dev-controlnet-union", "flux1devcontrolnetunion"],
    resource_id(ResourceKind.controlnet, Arch.qwen, ControlMode.universal): ["qwen-image-instantx-controlnet-union"],
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
    resource_id(ResourceKind.clip_vision, Arch.sd15, "ip_adapter"): ["sd1.5/pytorch_model.bin", "sd1.5/model.safetensors", "clip-vision_vit-h.safetensors", "clip-vit-h-14-laion2b-s32b-b79k"],
    resource_id(ResourceKind.clip_vision, Arch.sdxl, "ip_adapter"): ["sd1.5/pytorch_model.bin", "sd1.5/model.safetensors", "clip-vision_vit-h.safetensors", "clip-vit-h-14-laion2b-s32b-b79k"],
    resource_id(ResourceKind.clip_vision, Arch.flux, "redux"): ["sigclip_vision_patch14_384"],
    resource_id(ResourceKind.clip_vision, Arch.illu, "ip_adapter"): ["clip-vit-bigg", "clip_vision_g", "clip-vision_vit-g"],
    resource_id(ResourceKind.lora, Arch.sd15, "lcm"): ["lcm-lora-sdv1-5.safetensors", "lcm/sd1.5/pytorch_lora_weights.safetensors"],
    resource_id(ResourceKind.lora, Arch.sdxl, "lcm"): ["lcm-lora-sdxl.safetensors", "lcm/sdxl/pytorch_lora_weights.safetensors"],
    resource_id(ResourceKind.lora, Arch.sdxl, "lightning"): ["sdxl_lightning_8step_lora"],
    resource_id(ResourceKind.lora, Arch.sd15, "hyper"): ["Hyper-SD15-8steps-CFG-lora"],
    resource_id(ResourceKind.lora, Arch.sdxl, "hyper"): ["Hyper-SDXL-8steps-CFG-lora"],
    resource_id(ResourceKind.lora, Arch.flux, "turbo"): ["flux.1-turbo"],
    resource_id(ResourceKind.lora, Arch.flux_k, "turbo"): ["flux.1-turbo"],
    resource_id(ResourceKind.lora, Arch.sd15, ControlMode.face): ["ip-adapter-faceid-plusv2_sd15_lora", "ip-adapter-faceid-plus_sd15_lora"],
    resource_id(ResourceKind.lora, Arch.sdxl, ControlMode.face): ["ip-adapter-faceid-plusv2_sdxl_lora", "ip-adapter-faceid_sdxl_lora"],
    resource_id(ResourceKind.lora, Arch.flux, ControlMode.depth): ["flux1-depth"],
    resource_id(ResourceKind.lora, Arch.flux, ControlMode.canny_edge): ["flux1-canny"],
    resource_id(ResourceKind.lora, Arch.flux2_4b, ControlMode.inpaint): ["flux-2-klein-4B-outpaint-lora", "4b-outpaint-lora", "LyNiaZ53Tudg0J6sT8Xbx"],
    resource_id(ResourceKind.model_patch, Arch.zimage, ControlMode.universal): ["z-image-turbo-fun-controlnet-union-2.1", "z-image-turbo-fun-controlnet-union"],
    resource_id(ResourceKind.model_patch, Arch.zimage, ControlMode.blur): ["z-image-turbo-fun-controlnet-tile-2.1", "z-image-turbo-fun-controlnet-tile"],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.default): [UpscalerName.default.value],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_2x): [UpscalerName.fast_2x.value],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_3x): [UpscalerName.fast_3x.value],
    resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_4x): [UpscalerName.fast_4x.value],
    resource_id(ResourceKind.inpaint, Arch.sdxl, "fooocus_head"): ["fooocus_inpaint_head.pth"],
    resource_id(ResourceKind.inpaint, Arch.sdxl, "fooocus_patch"): ["inpaint_v26.fooocus"],
    resource_id(ResourceKind.inpaint, Arch.all, "default"): ["MAT_Places512_G_fp16", "Places_512_FullData_G", "big-lama.pt"],
    resource_id(ResourceKind.text_encoder, Arch.all, "clip_l"): ["clip_l"],
    resource_id(ResourceKind.text_encoder, Arch.all, "clip_g"): ["clip_g"],
    resource_id(ResourceKind.text_encoder, Arch.all, "t5"): ["t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl_fp8_e4m3fn_scaled", "t5-v1_1-xxl", "t5"],
    resource_id(ResourceKind.text_encoder, Arch.all, "qwen"): ["qwen_2.5_vl_7b", "qwen2.5-vl-7b", "qwen_2", "qwen-2", "qwen"],
    resource_id(ResourceKind.text_encoder, Arch.all, "qwen_3_4b"): ["qwen_3_4b", "qwen3-4b", "qwen3_4b", "qwen_3", "qwen-3"],
    resource_id(ResourceKind.text_encoder, Arch.all, "qwen_3_8b"): ["qwen_3_8b", "qwen3-8b", "qwen3_8b"],
    resource_id(ResourceKind.vae, Arch.sd15, "default"): ["vae-ft-mse-840000-ema"],
    resource_id(ResourceKind.vae, Arch.sdxl, "default"): ["sdxl_vae"],
    resource_id(ResourceKind.vae, Arch.illu, "default"): ["sdxl_vae"],
    resource_id(ResourceKind.vae, Arch.illu_v, "default"): ["sdxl_vae"],
    resource_id(ResourceKind.vae, Arch.sd3, "default"): ["sd3"],
    resource_id(ResourceKind.vae, Arch.flux, "default"): ["flux-", "flux_", "flux/", "flux1", "ae.s"],
    resource_id(ResourceKind.vae, Arch.flux_k, "default"): ["flux-", "flux_", "flux/", "flux1", "ae.s"],
    resource_id(ResourceKind.vae, Arch.flux2_4b, "default"): ["flux2"],
    resource_id(ResourceKind.vae, Arch.flux2_9b, "default"): ["flux2"],
    resource_id(ResourceKind.vae, Arch.chroma, "default"): ["flux-", "flux_", "flux/", "flux1", "ae.s"],
    resource_id(ResourceKind.vae, Arch.qwen, "default"): ["qwen"],
    resource_id(ResourceKind.vae, Arch.qwen_e, "default"): ["qwen"],
    resource_id(ResourceKind.vae, Arch.qwen_e_p, "default"): ["qwen"],
    resource_id(ResourceKind.vae, Arch.qwen_l, "default"): ["qwen_image_layered_vae"],
    resource_id(ResourceKind.vae, Arch.zimage, "default"): ["z-image", "flux-", "flux_", "flux/", "flux1", "ae.s"],
}
# fmt: on

required_resource_ids = {
    ResourceId(ResourceKind.text_encoder, Arch.sd3, "clip_l"),
    ResourceId(ResourceKind.text_encoder, Arch.sd3, "clip_g"),
    ResourceId(ResourceKind.text_encoder, Arch.qwen, "qwen"),
    ResourceId(ResourceKind.text_encoder, Arch.qwen_e, "qwen"),
    ResourceId(ResourceKind.text_encoder, Arch.qwen_e_p, "qwen"),
    ResourceId(ResourceKind.text_encoder, Arch.zimage, "qwen_3_4b"),
    ResourceId(ResourceKind.text_encoder, Arch.flux2_4b, "qwen_3_4b"),
    ResourceId(ResourceKind.text_encoder, Arch.flux2_9b, "qwen_3_8b"),
    ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.inpaint),
    ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.blur),
    ResourceId(ResourceKind.ip_adapter, Arch.sd15, ControlMode.reference),
    ResourceId(ResourceKind.ip_adapter, Arch.sdxl, ControlMode.reference),
    ResourceId(ResourceKind.clip_vision, Arch.sd15, "ip_adapter"),
    ResourceId(ResourceKind.clip_vision, Arch.sdxl, "ip_adapter"),
    ResourceId(ResourceKind.lora, Arch.sd15, "hyper"),
    ResourceId(ResourceKind.lora, Arch.sdxl, "hyper"),
    ResourceId(ResourceKind.upscaler, Arch.all, UpscalerName.default),
    ResourceId(ResourceKind.upscaler, Arch.all, UpscalerName.fast_2x),
    ResourceId(ResourceKind.inpaint, Arch.sdxl, "fooocus_head"),
    ResourceId(ResourceKind.inpaint, Arch.sdxl, "fooocus_patch"),
    ResourceId(ResourceKind.inpaint, Arch.all, "default"),
    ResourceId(ResourceKind.vae, Arch.qwen, "default"),
    ResourceId(ResourceKind.vae, Arch.qwen_e, "default"),
    ResourceId(ResourceKind.vae, Arch.qwen_e_p, "default"),
    ResourceId(ResourceKind.vae, Arch.zimage, "default"),
    ResourceId(ResourceKind.vae, Arch.flux2_4b, "default"),
    ResourceId(ResourceKind.vae, Arch.flux2_9b, "default"),
}

recommended_resource_ids = [
    ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.scribble),
    ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.line_art),
    ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.soft_edge),
    ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.canny_edge),
    ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.depth),
    ResourceId(ResourceKind.controlnet, Arch.sd15, ControlMode.pose),
    ResourceId(ResourceKind.controlnet, Arch.sdxl, ControlMode.universal),
    ResourceId(ResourceKind.controlnet, Arch.illu, ControlMode.inpaint),
    ResourceId(ResourceKind.controlnet, Arch.illu, ControlMode.scribble),
    ResourceId(ResourceKind.controlnet, Arch.illu, ControlMode.line_art),
    ResourceId(ResourceKind.controlnet, Arch.illu, ControlMode.canny_edge),
    ResourceId(ResourceKind.controlnet, Arch.illu, ControlMode.depth),
    ResourceId(ResourceKind.controlnet, Arch.illu, ControlMode.pose),
    ResourceId(ResourceKind.ip_adapter, Arch.illu, ControlMode.reference),
    ResourceId(ResourceKind.clip_vision, Arch.illu, "ip_adapter"),
    ResourceId(ResourceKind.controlnet, Arch.flux, ControlMode.inpaint),
    ResourceId(ResourceKind.controlnet, Arch.flux, ControlMode.universal),
    ResourceId(ResourceKind.lora, Arch.flux, "turbo"),
    ResourceId(ResourceKind.model_patch, Arch.zimage, ControlMode.universal),
    ResourceId(ResourceKind.model_patch, Arch.zimage, ControlMode.blur),
]
recommended_models = [get_resource(rid) for rid in recommended_resource_ids]
