from __future__ import annotations
from PyQt5.QtCore import QObject, pyqtSignal, QUuid, Qt

from . import model, jobs
from .settings import settings
from .resources import ControlMode
from .client import resolve_sd_version
from .properties import Property, PropertyMeta
from .image import Bounds
from .workflow import Control


class ControlLayer(QObject, metaclass=PropertyMeta):
    mode = Property(ControlMode.image)
    layer_id = Property(QUuid())
    strength = Property(100)
    end = Property(1.0)
    is_supported = Property(True)
    is_pose_vector = Property(False)
    can_generate = Property(True)
    has_active_job = Property(False)
    show_end = Property(False)
    error_text = Property("")

    mode_changed = pyqtSignal(ControlMode)
    layer_id_changed = pyqtSignal(QUuid)
    strength_changed = pyqtSignal(int)
    end_changed = pyqtSignal(float)
    is_supported_changed = pyqtSignal(bool)
    is_pose_vector_changed = pyqtSignal(bool)
    can_generate_changed = pyqtSignal(bool)
    has_active_job_changed = pyqtSignal(bool)
    show_end_changed = pyqtSignal(bool)
    error_text_changed = pyqtSignal(str)

    _model: model.Model
    _generate_job: jobs.Job | None = None

    def __init__(self, model: model.Model, mode: ControlMode, layer_id: QUuid):
        from .root import root

        super().__init__()
        self._model = model
        self.mode = mode
        self.layer_id = layer_id
        self._update_is_supported()
        self._update_is_pose_vector()

        self.mode_changed.connect(self._update_is_supported)
        model.style_changed.connect(self._update_is_supported)
        root.connection.state_changed.connect(self._update_is_supported)
        self.mode_changed.connect(self._update_is_pose_vector)
        self.layer_id_changed.connect(self._update_is_pose_vector)
        model.jobs.job_finished.connect(self._update_active_job)
        settings.changed.connect(self._handle_settings)

    @property
    def layer(self):
        layer = self._model.image_layers.find(self.layer_id)
        assert layer is not None, "Control layer has been deleted"
        return layer

    def get_image(self, bounds: Bounds | None = None):
        layer = self.layer
        if self.mode is ControlMode.image and not layer.bounds().isEmpty():
            bounds = None  # ignore mask bounds, use layer bounds
        image = self._model.document.get_layer_image(layer, bounds)
        if self.mode.is_lines or self.mode is ControlMode.stencil:
            image.make_opaque(background=Qt.GlobalColor.white)
        return Control(self.mode, image, self.strength / 100, self.end)

    def generate(self):
        self._generate_job = self._model.generate_control_layer(self)
        self.has_active_job = True

    def _update_is_supported(self):
        from .root import root

        is_supported = True
        if client := root.connection.client_if_connected:
            sdver = resolve_sd_version(self._model.style, client)
            if self.mode is ControlMode.image:
                if client.ip_adapter_model[sdver] is None:
                    self.error_text = f"The server is missing the IP-Adapter model"
                    is_supported = False
            elif client.control_model[self.mode][sdver] is None:
                filenames = self.mode.filenames(sdver)
                if filenames:
                    self.error_text = f"The ControlNet model is not installed {filenames}"
                else:
                    self.error_text = f"Not supported for {sdver.value}"
                is_supported = False

        self.is_supported = is_supported
        self.show_end = self.is_supported and settings.show_control_end
        self.can_generate = is_supported and self.mode not in [
            ControlMode.image,
            ControlMode.stencil,
        ]

    def _update_is_pose_vector(self):
        self.is_pose_vector = self.mode is ControlMode.pose and self.layer.type() == "vectorlayer"

    def _update_active_job(self):
        from .jobs import JobState

        active = not (self._generate_job is None or self._generate_job.state is JobState.finished)
        if self.has_active_job and not active:
            self._job = None  # job done
        self.has_active_job = active

    def _handle_settings(self, name: str, value: object):
        if name == "show_control_end":
            self.show_end = self.is_supported and settings.show_control_end


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
        model.image_layers.changed.connect(self._update_layer_list)

    def add(self):
        layer = self._model.document.active_layer.uniqueId()
        control = ControlLayer(self._model, self._last_mode, layer)
        control.mode_changed.connect(self._update_last_mode)
        self._layers.append(control)
        self.added.emit(control)

    def remove(self, control: ControlLayer):
        self._layers.remove(control)
        self.removed.emit(control)

    def _update_last_mode(self, mode: ControlMode):
        self._last_mode = mode

    def _update_layer_list(self):
        # Remove layers that have been deleted
        layer_ids = [l.uniqueId() for l in self._model.image_layers]
        to_remove = [l for l in self._layers if l.layer_id not in layer_ids]
        for l in to_remove:
            self.remove(l)

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    def __iter__(self):
        return iter(self._layers)
