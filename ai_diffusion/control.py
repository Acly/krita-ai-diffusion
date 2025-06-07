from __future__ import annotations
from PyQt5.QtCore import QObject, pyqtSignal, QUuid, Qt
from typing import Any, NamedTuple
from pathlib import Path
import json

from . import model, jobs, resources, util
from .api import ControlInput
from .layer import Layer, LayerType
from .resources import ControlMode, ResourceKind, Arch, resource_id
from .properties import Property, ObservableProperties
from .image import Bounds, Extent, Image
from .localization import translate as _
from .util import client_logger as log


class ControlLayer(QObject, ObservableProperties):
    max_preset_value = 4
    strength_multiplier = 50
    clip_vision_extent = Extent(224, 224)

    mode = Property(ControlMode.reference, persist=True, setter="set_mode")
    layer_id = Property(QUuid(), persist=True)
    preset_value = Property(2, persist=True, setter="set_preset_value")
    strength = Property(50, persist=True)
    start = Property(0.0, persist=True)
    end = Property(1.0, persist=True)
    use_custom_strength = Property(False, persist=True, setter="set_use_custom_strength")
    is_supported = Property(True)
    is_pose_vector = Property(False)
    can_generate = Property(True)
    has_active_job = Property(False)
    error_text = Property("")

    mode_changed = pyqtSignal(ControlMode)
    layer_id_changed = pyqtSignal(QUuid)
    preset_value_changed = pyqtSignal(int)
    strength_changed = pyqtSignal(int)
    start_changed = pyqtSignal(float)
    end_changed = pyqtSignal(float)
    use_custom_strength_changed = pyqtSignal(bool)
    is_supported_changed = pyqtSignal(bool)
    is_pose_vector_changed = pyqtSignal(bool)
    can_generate_changed = pyqtSignal(bool)
    has_active_job_changed = pyqtSignal(bool)
    error_text_changed = pyqtSignal(str)
    modified = pyqtSignal(QObject, str)

    def __init__(self, model: model.Model, mode: ControlMode, layer_id: QUuid, index: int):
        from .root import root

        super().__init__()
        self._model = model
        self._index = index
        self._generate_job: jobs.Job | None = None
        self.layer_id = layer_id
        self.mode = mode
        self._update_is_supported()

        self.mode_changed.connect(self._update_is_supported)
        model.style_changed.connect(self._update_is_supported)
        root.connection.state_changed.connect(self._update_is_supported)
        self.layer_id_changed.connect(self._update_is_pose_vector)
        model.jobs.job_finished.connect(self._update_active_job)

    @property
    def layer(self):
        layer = self._model.layers.updated().find(self.layer_id)
        assert layer is not None, "Control layer has been deleted"
        return layer

    def set_mode(self, mode: ControlMode):
        if mode != self.mode:
            self._mode = mode
            self.mode_changed.emit(mode)
            self._update_is_pose_vector()
            if not self.use_custom_strength:
                self._set_values_from_preset()

    def set_preset_value(self, value: int):
        if value != self.preset_value:
            self._preset_value = value
            self.preset_value_changed.emit(value)
            self._set_values_from_preset()

    def _set_values_from_preset(self):
        params = ControlPresets.instance().interpolate(
            self.mode, self._model.arch, self.preset_value / self.max_preset_value
        )
        self.strength = int(params.strength * self.strength_multiplier)
        self.start, self.end = params.range

    def set_use_custom_strength(self, value: bool):
        if value != self.use_custom_strength:
            self._use_custom_strength = value
            self.use_custom_strength_changed.emit(value)
            if not value:
                self._set_values_from_preset()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        self._update_is_supported()

    def to_api(self, bounds: Bounds | None = None, time: int | None = None):
        assert self.is_supported, "Control layer is not supported"
        layer = self.layer
        if self.mode.is_ip_adapter and not layer.bounds.is_zero:
            bounds = None  # ignore mask bounds, use layer bounds
        image = layer.get_pixels(bounds, time)
        if self.mode.is_lines or self.mode is ControlMode.stencil:
            image.make_opaque(background=Qt.GlobalColor.white)
        if self.mode.is_ip_adapter:
            image = Image.scale(image, self.clip_vision_extent)
        strength = self.strength / self.strength_multiplier
        return ControlInput(self.mode, image, strength, (self.start, self.end))

    def generate(self):
        self._generate_job = self._model.generate_control_layer(self)
        self.has_active_job = True

    def _update_is_supported(self):
        from .root import root

        is_supported = True
        if client := root.connection.client_if_connected:
            models = client.models.for_arch(self._model.arch)
            if self.mode.is_ip_adapter and models.arch in [Arch.illu, Arch.illu_v]:
                resid = resource_id(ResourceKind.clip_vision, Arch.illu, "ip_adapter")
                has_clip_vision = client.models.resources.get(resid, None) is not None
                if not has_clip_vision:
                    search = resources.search_path(
                        ResourceKind.clip_vision, Arch.illu, "ip_adapter"
                    )
                    self.error_text = _("The server is missing the ClipVision model") + f" {search}"
                    is_supported = False
            if self.mode.is_ip_adapter and models.ip_adapter.find(self.mode) is None:
                search_path = resources.search_path(ResourceKind.ip_adapter, models.arch, self.mode)
                if search_path:
                    self.error_text = (
                        _("The server is missing the IP-Adapter model") + f" {self.mode.text}"
                    )
                else:
                    self.error_text = _("Not supported for") + f" {models.arch.value}"
                if not client.features.ip_adapter:
                    self.error_text = _("IP-Adapter is not supported by this GPU")
                is_supported = False
            elif self.mode.is_control_net:
                cn_model = models.control.find(self.mode, allow_universal=True)
                lora_model = models.lora.find(self.mode)
                if cn_model is None and lora_model is None:
                    search_arch = Arch.illu if models.arch is Arch.illu_v else models.arch
                    search_path = resources.search_path(
                        ResourceKind.controlnet, search_arch, self.mode
                    ) or resources.search_path(ResourceKind.lora, models.arch, self.mode)
                    if search_path:
                        self.error_text = (
                            _("The ControlNet model is not installed") + f" {search_path}"
                        )
                    else:
                        self.error_text = _("Not supported for") + f" {models.arch.value}"
                    is_supported = False

            if self._index >= client.features.max_control_layers:
                self.error_text = _("Too many control layers")
                is_supported = False

        self.is_supported = is_supported
        self.can_generate = is_supported and self.mode.has_preprocessor

    def _update_is_pose_vector(self):
        self.is_pose_vector = self.mode is ControlMode.pose and self.layer.type is LayerType.vector

    def _update_active_job(self):
        from .jobs import JobState

        active = not (self._generate_job is None or self._generate_job.state is JobState.finished)
        if self.has_active_job and not active:
            self._job = None  # job done
        self.has_active_job = active


class ControlLayerList(QObject):
    """List of control layers for one document."""

    added = pyqtSignal(ControlLayer)
    removed = pyqtSignal(ControlLayer)

    _model: "model.Model"
    _layers: list[ControlLayer]
    _last_mode = ControlMode.scribble

    def __init__(self, model: "model.Model"):
        super().__init__()
        self._model = model
        self._layers = []
        self._model.layers.removed.connect(self._remove_layer)

    def add(self):
        layer = self._model.layers.active
        if layer.type.is_filter and layer.parent_layer and not layer.parent_layer.is_root:
            layer = layer.parent_layer
        if not layer.type.is_image:
            layer = next(iter(self._model.layers.images), None)
        if layer is None:  # shouldn't be possible, Krita doesn't allow removing all non-mask layers
            log.warning("Trying to add control layer, but document has no suitable layer")
            return
        control = ControlLayer(self._model, self._last_mode, layer.id, len(self._layers))
        control.mode_changed.connect(self._update_last_mode)
        self._layers.append(control)
        self.added.emit(control)

    def emplace(self):
        self.add()
        return self[-1]

    def remove(self, control: ControlLayer):
        self._layers.remove(control)
        self.removed.emit(control)

        for i, c in enumerate(self._layers):
            c.index = i

    def to_api(self, bounds: Bounds | None = None, time: int | None = None):
        for layer in (c for c in self._layers if not c.is_supported):
            log.warning(f"Trying to use control layer {layer.mode.name}: {layer.error_text}")
        return [c.to_api(bounds, time) for c in self._layers if c.is_supported]

    def _update_last_mode(self, mode: ControlMode):
        self._last_mode = mode

    def _remove_layer(self, layer: Layer):
        if control := next((c for c in self._layers if c.layer_id == layer.id), None):
            self.remove(control)

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    def __iter__(self):
        return iter(self._layers)


class ControlParams(NamedTuple):
    strength: float
    range: tuple[float, float]

    @staticmethod
    def from_dict(data: dict[str, Any]):
        return ControlParams(data["strength"], (data["start"], data["end"]))


class ControlPresets:
    _path: Path
    _user_path: Path
    _presets: dict[str, dict[str, list[dict[str, Any]]]]

    _instance: ControlPresets | None = None

    @classmethod
    def instance(cls) -> ControlPresets:
        if cls._instance is None:
            cls._instance = ControlPresets()
        return cls._instance

    def __init__(self):
        self._path = util.plugin_dir / "presets" / "control.json"
        self._user_path = util.user_data_dir / "presets" / "control.json"
        self._read()

    def get(self, mode: ControlMode, arch: Arch):
        default = self._presets["default"]
        versions = self._presets.get(mode.name, default)
        all = versions.get("all", None)
        presets = versions.get(arch.name, all)
        if presets is None:
            raise KeyError(f"No control strength presets found for {mode} and {arch}")
        return [ControlParams.from_dict(p) for p in presets]

    def interpolate(self, mode: ControlMode, arch: Arch, value: float):
        assert value >= 0 and value <= 1, f"Interpolate value out of range: {value}"
        presets = self.get(mode, arch)
        if len(presets) == 1 or value <= 0:
            return presets[0]
        if value == 1:
            return presets[-1]
        value = value * (len(presets) - 1)
        for i, p0 in enumerate(presets):
            if value < i + 1:
                p1 = presets[i + 1]
                t = value - i
                return ControlParams(
                    _lerp(p0.strength, p1.strength, t),
                    (_lerp(p0.range[0], p1.range[0], t), _lerp(p0.range[1], p1.range[1], t)),
                )
        assert False, f"Interpolation failed: {mode}, {arch}, value={value}, presets={presets}"

    def _read(self):
        self._presets = self._read_file(self._path)
        _validate_presets(self._path, self._presets)
        if self._user_path.exists():
            user = self._read_file(self._user_path)
            if _validate_presets(self._user_path, user):
                _recursive_update(self._presets, user)
        else:
            self._user_path.parent.mkdir(parents=True, exist_ok=True)
            self._user_path.write_text(json.dumps({}, indent=4))

    def _read_file(self, path: Path):
        try:
            return json.load(path.open("r"))
        except Exception as e:
            raise ValueError(f"Failed to read control layer presets file {path}: {e}") from e


def _validate_presets(filepath: Path, data: dict[str, Any]) -> bool:
    control_modes = ["default"] + list(ControlMode.__members__.keys())
    model_archs = list(Arch.__members__.keys())

    for mode, versions in data.items():
        if mode not in control_modes:
            log.error(
                f"Invalid control mode '{mode}' in presets file {filepath}."
                f" Valid modes are: {', '.join(control_modes)}"
            )
            return False
        if not isinstance(versions, dict):
            log.error(f"Invalid presets for mode '{mode}' in presets file {filepath}.")
            return False
        for arch, presets in versions.items():
            if arch not in model_archs:
                log.error(
                    f"Invalid Base model '{arch}' for mode '{mode}' in presets file {filepath}."
                    f" Valid versions are: {', '.join(model_archs)}"
                )
                return False
            if not isinstance(presets, list):
                log.error(
                    f"Invalid presets for '{mode}/{arch}' in presets file {filepath}."
                    f" Expected a list, got {presets}"
                )
                return False
            for p in presets:
                if not isinstance(p, dict) or not all(k in p for k in ("strength", "start", "end")):
                    log.error(
                        f"Invalid preset for '{mode}/{arch}' in presets file {filepath}."
                        f" Expected a {{strength, start, end}}, got {p}"
                    )
                    return False
    return True


control_mode_text = {
    ControlMode.reference: _("Reference"),
    ControlMode.inpaint: _("Inpaint"),
    ControlMode.style: _("Style"),
    ControlMode.composition: _("Composition"),
    ControlMode.face: _("Face"),
    ControlMode.universal: _("Universal"),
    ControlMode.scribble: _("Scribble"),
    ControlMode.line_art: _("Line Art"),
    ControlMode.soft_edge: _("Soft Edge"),
    ControlMode.canny_edge: _("Canny Edge"),
    ControlMode.depth: _("Depth"),
    ControlMode.normal: _("Normal"),
    ControlMode.pose: _("Pose"),
    ControlMode.segmentation: _("Segment"),
    ControlMode.blur: _("Unblur"),
    ControlMode.stencil: _("Stencil"),
    ControlMode.hands: _("Hands"),
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _recursive_update(a: dict[str, Any], b: dict[str, Any]):
    for k, v in b.items():
        if isinstance(v, dict):
            a[k] = _recursive_update(a.get(k, {}), v)
        else:
            a[k] = v
    return a
