from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from PyQt6.QtCore import QObject, Qt, QUuid, pyqtSignal
from PyQt6.QtGui import QColor

from .. import util
from ..backend import resources
from ..backend.api import ControlInput
from ..backend.resources import Arch, ControlMode, ResourceKind, resource_id
from ..image import BlendMode, Bounds, Extent, Image
from ..layer import Layer, LayerType
from ..localization import translate as _
from ..util import PluginError
from ..util import client_logger as log
from . import jobs
from .properties import ObservableProperties, Property

if TYPE_CHECKING:
    from .model import DocumentModel
    from .region import Region, RootRegion


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
    has_range = Property(True)
    has_active_job = Property(False)
    error_text = Property("")

    region_changed = pyqtSignal()
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
    has_range_changed = pyqtSignal(bool)
    error_text_changed = pyqtSignal(str)
    modified = pyqtSignal(QObject, str)

    def __init__(
        self,
        model: DocumentModel,
        mode: ControlMode,
        layer_id: QUuid,
        index: int,
        region: Region | RootRegion,
    ):
        super().__init__()
        self._model = model
        self._region = region
        self._index = index
        self._generate_job: jobs.Job | None = None
        self._error_text_short = ""
        self.layer_id = layer_id
        self.mode = mode
        self._update_is_supported()

        self.mode_changed.connect(self._update_is_supported)
        model.active_style_changed.connect(self._update_is_supported)
        model.edit_mode_changed.connect(self._update_is_supported)
        model._connection.state_changed.connect(self._update_is_supported)
        self.layer_id_changed.connect(self._update_is_pose_vector)
        model.jobs.job_finished.connect(self._update_active_job)

    @property
    def layer(self):
        layer = self._model.layers.updated().find(self.layer_id)
        assert layer is not None, "Control layer has been deleted"
        return layer

    @property
    def region(self):
        from .region import RootRegion

        if not self.mode.is_ip_adapter or isinstance(self._region, RootRegion):
            return None
        return self._region

    @region.setter
    def region(self, region: Region | RootRegion | None):
        region = region if region is not None else self._model.regions
        if region is not self._region:
            self._region = region
            self.region_changed.emit()
            self.modified.emit(self, "region")

    def set_mode(self, mode: ControlMode):
        if mode != self.mode:
            self._mode = mode
            self.mode_changed.emit(mode)
            self.region_changed.emit()
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

    @property
    def error_text_short(self):
        return self._error_text_short

    def to_api(self, bounds: Bounds | None = None, time: int | None = None):
        layer = self.layer
        if not self.is_supported:
            raise PluginError(f"Can't use '{layer.name}' as control layer: {self.error_text}")

        extent = bounds.extent if bounds else self._model.document.extent
        if self.mode.is_ip_adapter and not layer.bounds.is_zero:
            bounds = None  # ignore mask bounds, use layer bounds

        image = layer.get_pixels(bounds, time)

        if self.mode is ControlMode.segmentation:
            image.make_opaque(background=Qt.GlobalColor.white)

        if self.mode.is_lines or self.mode is ControlMode.stencil:
            image.make_opaque(background=Qt.GlobalColor.white)

        if self.mode.is_ip_adapter:
            if self._model.arch.supports_edit:
                if image.extent.height > extent.height:
                    w = (image.extent.width * extent.height) // image.extent.height
                    image = Image.scale(image, Extent(w, extent.height))
            else:
                image = Image.scale(image, self.clip_vision_extent)

        strength = self.strength / self.strength_multiplier
        return ControlInput(self.mode, image, strength, (self.start, self.end))

    def generate_segmentation(self):
        assert self.mode is ControlMode.segmentation

        ok, msg = self._model.document.check_color_mode()
        if not ok and msg:
            self._model.report_error(msg)
            return

        try:
            bounds = Bounds.from_extent(self._model.document.extent)
            image = self._segmentation_image_from_regions(bounds)
            if image is None:
                self._model.report_error(_("Text prompt regions have not been set up."))
                return

            layer = self._model.layers.create(f"[Control] {self.mode.text}", image, bounds)
            self.layer_id = layer.id
        except Exception as e:
            self._model.report_error(util.log_error(e))
        else:
            self._model.clear_error()

    def _segmentation_image_from_regions(self, bounds: Bounds):
        from .region import RegionLink

        image = Image.create(bounds.extent, fill=Qt.GlobalColor.white)
        has_region_layer = False
        root = self._model.regions

        layers = [
            layer
            for layer in root.layers.all
            if root.find_linked(layer, RegionLink.direct) is not None
            and Bounds.intersection(bounds, layer.compute_bounds()).area > 0
        ]

        for index, layer in enumerate(layers):
            color = _segmentation_colors[index % len(_segmentation_colors)]
            region_image = self._segmentation_region_image(layer, bounds, color)
            image.draw_image(region_image, blend=BlendMode.alpha)
            has_region_layer = True

        return image if has_region_layer else None

    def _segmentation_region_image(self, layer: Layer, bounds: Bounds, color: tuple[int, int, int]):
        mask = layer.get_mask(bounds)
        image = Image.create(bounds.extent, fill=QColor(*color, 255))
        image._qimage.setAlphaChannel(mask._qimage)
        return image

    def generate(self):
        self._generate_job = self._model.generate_control_layer(self)
        self.has_active_job = True

    def _update_is_supported(self):
        is_supported = True
        if client := self._model._connection.client_if_connected:
            models = client.models.for_arch(self._model.arch)

            if self.mode.is_ip_adapter and models.arch in [Arch.illu, Arch.illu_v]:
                resid = resource_id(ResourceKind.clip_vision, Arch.illu, "ip_adapter")
                has_clip_vision = client.models.resources.get(resid, None) is not None
                if not has_clip_vision:
                    search = resources.search_path(
                        ResourceKind.clip_vision, Arch.illu, "ip_adapter"
                    )
                    self.error_text = _("The server is missing the ClipVision model") + f" {search}"
                    self._error_text_short = _("Missing control model")
                    is_supported = False

            if self.mode.is_ip_adapter and models.arch.supports_edit:
                is_supported = True  # Reference images are merged into the conditioning context
            elif self.mode.is_ip_adapter and models.ip_adapter.find(self.mode) is None:
                search_path = resources.search_path(ResourceKind.ip_adapter, models.arch, self.mode)
                if search_path:
                    self.error_text = (
                        _("The server is missing the IP-Adapter model") + f" {self.mode.text}"
                    )
                    self._error_text_short = _("Missing control model")
                else:
                    self.error_text = _("Not supported for") + f" {models.arch.value}"
                    self._error_text_short = _("Not supported")
                if not client.features.ip_adapter:
                    self.error_text = _("IP-Adapter is not supported by this GPU")
                    self._error_text_short = _("Not supported")
                is_supported = False
            elif self.mode.is_control_net and models.arch.supports_edit:
                is_supported = self.mode.can_substitute_instruction(models.arch)
                if not is_supported:
                    self.error_text = _("Not supported for") + f" {models.arch.value}"
                    self._error_text_short = _("Not supported")
            elif self.mode.is_control_net:
                model = models.find_control(self.mode)
                self.has_range = model == models.control.find(self.mode, True)
                if model is None:
                    search_arch = Arch.illu if models.arch is Arch.illu_v else models.arch
                    search_path = (
                        resources.search_path(ResourceKind.controlnet, search_arch, self.mode)
                        or resources.search_path(ResourceKind.model_patch, search_arch, self.mode)
                        or resources.search_path(ResourceKind.lora, models.arch, self.mode)
                    )
                    if search_path:
                        self.error_text = (
                            _("The ControlNet model is not installed") + f" {search_path}"
                        )
                        self._error_text_short = _("Missing control model")
                    else:
                        self.error_text = _("Not supported for") + f" {models.arch.value}"
                        self._error_text_short = _("Not supported")
                    is_supported = False

            if self._index >= client.features.max_control_layers:
                self.error_text = _("Too many control layers")
                self._error_text_short = ""
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
    changed = pyqtSignal()

    def __init__(self, model: DocumentModel):
        super().__init__()
        self._model = model
        self._layers: list[ControlLayer] = []
        self._last_mode = ControlMode.scribble
        self._model.layers.removed.connect(self._remove_layer)
        model.regions.active_changed.connect(self.changed)
        model.regions.removed.connect(self._remove_region)
        model.edit_mode_changed.connect(self.changed)
        model.active_style_changed.connect(self.changed)
        model.workspace_changed.connect(self.changed)

    @property
    def all(self):
        return self._layers

    @property
    def active_region(self):
        return self._model.regions.active_or_root

    def for_region(self, region: Region | None):
        return [c for c in self._layers if self._is_mode_active(c) and c.region is region]

    def _is_mode_active(self, control: ControlLayer):
        return not self._model.is_editing or control.mode.is_ip_adapter

    def add(self, region: Region | RootRegion | None = None):
        layer = self._model.layers.active
        if layer.type.is_filter and layer.parent_layer and not layer.parent_layer.is_root:
            layer = layer.parent_layer
        if not layer.type.is_image:
            layer = next(iter(self._model.layers.images), None)
        if layer is None:  # shouldn't be possible, Krita doesn't allow removing all non-mask layers
            log.warning("Trying to add control layer, but document has no suitable layer")
            return
        mode = ControlMode.reference if self._model.is_editing else self._last_mode
        control = ControlLayer(
            self._model,
            mode,
            layer.id,
            len(self._layers),
            region if region is not None else self.active_region,
        )
        control.mode_changed.connect(self._update_last_mode)
        control.region_changed.connect(self.changed)
        self._layers.append(control)
        self.added.emit(control)

    def emplace(self, region: Region | RootRegion | None = None):
        self.add(region)
        return self._layers[-1]

    def remove(self, control: ControlLayer):
        self._layers.remove(control)
        self.removed.emit(control)

        for i, c in enumerate(self._layers):
            c.index = i

    def to_api(
        self,
        bounds: Bounds | None = None,
        time: int | None = None,
        region: Region | None = None,
    ):
        controls = self.for_region(region)
        for layer in (c for c in controls if not c.is_supported):
            log.warning(f"Trying to use control layer {layer.mode.name}: {layer.error_text}")
        return [c.to_api(bounds, time) for c in controls if c.is_supported and c.strength > 0]

    def _update_last_mode(self, mode: ControlMode):
        self._last_mode = mode

    def _remove_layer(self, layer: Layer):
        for control in list(self._layers):
            if control.layer_id == layer.id:
                self.remove(control)

    def _remove_region(self, region: Region):
        for control in list(self._layers):
            if control._region is region:
                self.remove(control)

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, i):
        return list(self)[i]

    def __iter__(self):
        active = self.active_region
        return (
            c
            for c in self._layers
            if self._is_mode_active(c) and (c.region is None or c.region is active)
        )


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
            raise PluginError(f"No control strength presets found for {mode} and {arch}")
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

_segmentation_colors = [
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (80, 80, 80),
    (160, 80, 0),
    (80, 160, 0),
    (0, 80, 160),
    (0, 160, 80),
]


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _recursive_update(a: dict[str, Any], b: dict[str, Any]):
    for k, v in b.items():
        if isinstance(v, dict):
            a[k] = _recursive_update(a.get(k, {}), v)
        else:
            a[k] = v
    return a
