from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator, Iterable, NamedTuple
from PyQt5.QtCore import QObject, pyqtSignal

from .api import WorkflowInput
from .image import ImageCollection
from .properties import Property, ObservableProperties
from .files import FileLibrary, FileFormat
from .style import Style
from .settings import PerformanceSettings
from .resources import ControlMode, ResourceKind, Arch, UpscalerName
from .resources import CustomNode, ResourceId
from .localization import translate as _
from .util import client_logger as log


class ClientEvent(Enum):
    progress = 0
    finished = 1
    interrupted = 2
    error = 3
    connected = 4
    disconnected = 5
    queued = 6
    upload = 7
    published = 8
    output = 9
    payment_required = 10


class TextOutput(NamedTuple):
    key: str
    name: str
    text: str
    mime: str


class SharedWorkflow(NamedTuple):
    publisher: str
    workflow: dict


ClientOutput = dict | SharedWorkflow | TextOutput


class ClientMessage(NamedTuple):
    event: ClientEvent
    job_id: str = ""
    progress: float = 0
    images: ImageCollection | None = None
    result: ClientOutput | None = None
    error: str | None = None


class User(QObject, ObservableProperties):
    id: str
    name: str
    images_generated = Property(0)
    credits = Property(0)

    images_generated_changed = pyqtSignal(int)
    credits_changed = pyqtSignal(int)

    def __init__(self, id: str, name: str):
        super().__init__()
        self.id = id
        self.name = name


class DeviceInfo(NamedTuple):
    type: str
    name: str
    vram: int  # in GB

    @staticmethod
    def parse(data: dict):
        try:
            name = data["devices"][0]["name"]
            name = name.split(":")[1] if ":" in name else name
            vram = int(round(data["devices"][0]["vram_total"] / (1024**3)))
            return DeviceInfo(data["devices"][0]["type"], name, vram)
        except Exception as e:
            log.error(f"Could not parse device info {data}: {str(e)}")
            return DeviceInfo("cpu", "unknown", 0)


class MissingResources(Exception):
    def __init__(self, missing: dict[Arch, list[ResourceId]] | list[CustomNode]):
        self.missing = missing

    def __str__(self):
        return "Required custom nodes or model files are missing"

    def get(self, arch: Arch):
        if isinstance(self.missing, list):
            return self.missing
        return self.missing.get(arch, [])


class CheckpointInfo(NamedTuple):
    filename: str
    arch: Arch
    format: FileFormat = FileFormat.checkpoint

    @property
    def name(self):
        return self.filename.removesuffix(".safetensors")

    @staticmethod
    def deduce_from_filename(filename: str):
        return CheckpointInfo(filename, Arch.from_checkpoint_name(filename), FileFormat.checkpoint)


class ClientModels:
    """Collects names of AI models the client has access to."""

    checkpoints: dict[str, CheckpointInfo]
    vae: list[str]
    loras: list[str]
    upscalers: list[str]
    resources: dict[str, str | None]
    node_inputs: dict[str, dict[str, list[str | list | dict]]]

    def __init__(self) -> None:
        self.node_inputs = {}
        self.resources = {}

    def resource(
        self, kind: ResourceKind, identifier: ControlMode | UpscalerName | str, arch: Arch
    ):
        id = ResourceId(kind, arch, identifier)
        model = self.find(id)
        if model is None:
            raise Exception(f"{id.name} not found")
        return model

    def find(self, id: ResourceId):
        if result := self.resources.get(id.string):
            return result
        # Fallback to epsilon model if v-prediction model not found
        if id.arch is Arch.illu_v:
            if result := self.resources.get(id._replace(arch=Arch.illu).string):
                return result
        # Search for architecture-agnostic model
        return self.resources.get(id._replace(arch=Arch.all).string)

    def arch_of(self, checkpoint: str):
        if info := self.checkpoints.get(checkpoint):
            return info.arch
        return Arch.from_checkpoint_name(checkpoint)

    def for_arch(self, arch: Arch):
        return ModelDict(self, ResourceKind.upscaler, arch)

    @property
    def upscale(self):
        return ModelDict(self, ResourceKind.upscaler, Arch.all)

    @property
    def default_upscaler(self):
        return self.resource(ResourceKind.upscaler, UpscalerName.default, Arch.all)


class ModelDict:
    """Provides access to filtered list of models matching a certain Diffusion base model."""

    _models: ClientModels
    kind: ResourceKind
    arch: Arch

    def __init__(self, models: ClientModels, kind: ResourceKind, arch: Arch):
        self._models = models
        self.kind = kind
        self.arch = arch

    def __getitem__(self, key: ControlMode | UpscalerName | str):
        return self._models.resource(self.kind, key, self.arch)

    def find(self, key: ControlMode | UpscalerName | str, allow_universal=False) -> str | None:
        # Composition/Style modes use same IP-Adapter model as Reference with different weighting
        is_sd = self.arch is Arch.sd15 or self.arch.is_sdxl_like
        if key in [ControlMode.style, ControlMode.composition] and is_sd:
            key = ControlMode.reference

        result = self._models.find(ResourceId(self.kind, self.arch, key))
        # Fallback to universal model if not found
        if result is None and allow_universal:
            if isinstance(key, ControlMode) and key.can_substitute_universal(self.arch):
                result = self.find(ControlMode.universal)
        return result

    def for_version(self, arch: Arch):
        return ModelDict(self._models, self.kind, arch)

    @property
    def text_encoder(self):
        return ModelDict(self._models, ResourceKind.text_encoder, self.arch)

    @property
    def clip_vision(self):
        return self._models.resource(ResourceKind.clip_vision, "ip_adapter", self.arch)

    @property
    def upscale(self):
        return ModelDict(self._models, ResourceKind.upscaler, Arch.all)

    @property
    def control(self):
        return ModelDict(self._models, ResourceKind.controlnet, self.arch)

    @property
    def ip_adapter(self):
        return ModelDict(self._models, ResourceKind.ip_adapter, self.arch)

    @property
    def inpaint(self):
        return ModelDict(self._models, ResourceKind.inpaint, Arch.all)

    @property
    def lora(self):
        return ModelDict(self._models, ResourceKind.lora, self.arch)

    @property
    def vae(self):
        return self._models.resource(ResourceKind.vae, "default", self.arch)

    @property
    def fooocus_inpaint(self):
        assert self.arch is Arch.sdxl
        return dict(
            head=self._models.resource(ResourceKind.inpaint, "fooocus_head", Arch.sdxl),
            patch=self._models.resource(ResourceKind.inpaint, "fooocus_patch", Arch.sdxl),
        )

    @property
    def all(self):
        return self._models

    @property
    def node_inputs(self):
        return self._models.node_inputs

    @property
    def has_te_vae(self):
        if self._models.find(ResourceId(ResourceKind.vae, self.arch, "default")) is None:
            return False
        for te in self.arch.text_encoders:
            if self._models.find(ResourceId(ResourceKind.text_encoder, self.arch, te)) is None:
                return False
        return True


class TranslationPackage(NamedTuple):
    code: str
    name: str

    @staticmethod
    def from_dict(data: dict):
        return TranslationPackage(data["code"], data["name"])

    @staticmethod
    def from_list(data: list):
        return [TranslationPackage.from_dict(item) for item in data]


class ClientFeatures(NamedTuple):
    ip_adapter: bool = True
    translation: bool = True
    languages: list[TranslationPackage] = []
    max_upload_size: int = 0
    max_control_layers: int = 1000
    wave_speed: bool = False


class Client(ABC):
    url: str
    models: ClientModels
    device_info: DeviceInfo

    @staticmethod
    @abstractmethod
    async def connect(url: str, access_token: str = "") -> Client: ...

    @abstractmethod
    async def enqueue(self, work: WorkflowInput, front: bool = False) -> str: ...

    @abstractmethod
    def listen(self) -> AsyncGenerator[ClientMessage, Any]: ...

    @abstractmethod
    async def interrupt(self): ...

    @abstractmethod
    async def clear_queue(self): ...

    async def refresh(self):
        pass

    async def translate(self, text: str, lang: str) -> str:
        return text

    async def disconnect(self):
        pass

    @property
    def user(self) -> User | None:
        return None

    @property
    def missing_resources(self) -> MissingResources | None:
        return None

    def supports_arch(self, arch: Arch) -> bool:
        if self.missing_resources:
            return len(self.missing_resources.get(arch)) == 0
        return True

    @property
    def features(self) -> ClientFeatures: ...

    @property
    def performance_settings(self) -> PerformanceSettings: ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.disconnect()


def resolve_arch(style: Style, client: Client | ClientModels | None = None):
    arch = Arch.auto

    if client:
        models = client.models if isinstance(client, Client) else client
        checkpoint = style.preferred_checkpoint(models.checkpoints.keys())
        if checkpoint != "not-found":
            arch = models.arch_of(checkpoint)
    elif style.checkpoints:
        arch = style.architecture.resolve(style.checkpoints[0])

    if style.architecture.is_sdxl_like and arch.is_sdxl_like:
        arch = style.architecture  # user override with compatible arch

    return arch


def filter_supported_styles(styles: Iterable[Style], client: Client | None = None):
    if client:
        return [
            style
            for style in styles
            if client.supports_arch(resolve_arch(style, client))
            and style.preferred_checkpoint(client.models.checkpoints.keys()) != "not-found"
        ]
    return list(styles)


def loras_to_upload(workflow: WorkflowInput, client_models: ClientModels):
    workflow_loras = []
    if models := workflow.models:
        workflow_loras.extend(models.loras)
    if cond := workflow.conditioning:
        for region in cond.regions:
            workflow_loras.extend(region.loras)

    for lora in workflow_loras:
        if lora.name in client_models.loras:
            continue
        if not lora.storage_id and lora.name in _lcm_loras:
            raise ValueError(_lcm_warning)
        if not lora.storage_id:
            raise ValueError(f"Lora model is not available: {lora.name}")
        lora_file = FileLibrary.instance().loras.find_local(lora.name)
        if lora_file is None or lora_file.path is None:
            raise ValueError(f"Can't find Lora model: {lora.name}")
        if not lora_file.path.exists():
            raise ValueError(_("LoRA model file not found") + f" {lora_file.path}")
        assert lora.storage_id == lora_file.hash
        yield lora_file


_lcm_loras = ["lcm-lora-sdv1-5.safetensors", "lcm-lora-sdxl.safetensors"]
_lcm_warning = "LCM is no longer supported by the server. Please change the Style's sampling method to 'Realtime - Hyper'"
