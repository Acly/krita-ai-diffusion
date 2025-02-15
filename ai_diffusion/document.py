from __future__ import annotations
from pathlib import Path
from typing import Literal, cast
from weakref import WeakValueDictionary
import krita
from krita import Krita
from PyQt5.QtCore import QObject, QUuid, QByteArray, QTimer, pyqtSignal

from .image import Extent, Bounds, Mask, Image
from .layer import Layer, LayerManager, LayerType
from .pose import Pose
from .localization import translate as _
from .util import acquire_elements


class Document(QObject):
    """Document interface. Used as placeholder when there is no open Document in Krita."""

    selection_bounds_changed = pyqtSignal()
    current_time_changed = pyqtSignal()

    _layers: LayerManager

    def __init__(self):
        super().__init__()
        self._layers = LayerManager(None)

    @property
    def extent(self):
        return Extent(0, 0)

    @property
    def filename(self) -> str:
        return ""

    def check_color_mode(self) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
        return True, None

    def create_mask_from_selection(
        self, padding: float = 0.0, multiple=8, min_size=0, square=False, invert=False
    ) -> tuple[Mask, Bounds] | tuple[None, None]:
        raise NotImplementedError

    def get_image(
        self, bounds: Bounds | None = None, exclude_layers: list[Layer] | None = None
    ) -> Image:
        raise NotImplementedError

    def resize(self, extent: Extent):
        raise NotImplementedError

    def annotate(self, key: str, value: QByteArray):
        pass

    def find_annotation(self, key: str) -> QByteArray | None:
        return None

    def remove_annotation(self, key: str):
        pass

    def add_pose_character(self, layer: Layer):
        raise NotImplementedError

    def import_animation(self, files: list[Path], offset: int = 0):
        raise NotImplementedError

    @property
    def layers(self) -> LayerManager:
        return self._layers

    @property
    def selection_bounds(self) -> Bounds | None:
        return None

    @property
    def resolution(self) -> float:
        return 0.0

    @property
    def playback_time_range(self) -> tuple[int, int]:
        return 0, 0

    @property
    def current_time(self) -> int:
        return 0

    @property
    def is_valid(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return Krita.instance().activeDocument() is None


class KritaDocument(Document):
    """Wrapper around a Krita Document (opened image). Allows to retrieve and modify pixel data.
    Keeps track of selection and current time changes by polling at a fixed interval.
    """

    _doc: krita.Document
    _id: QUuid
    _layers: LayerManager
    _poller: QTimer
    _selection_bounds: Bounds | None = None
    _current_time: int = 0
    _instances: WeakValueDictionary[str, KritaDocument] = WeakValueDictionary()

    def __init__(self, krita_document: krita.Document):
        super().__init__()
        self._doc = krita_document
        self._id = krita_document.rootNode().uniqueId()
        self._poller = QTimer()
        self._poller.setInterval(20)
        self._poller.timeout.connect(self._poll)
        self._poller.start()
        self._instances[self._id.toString()] = self
        self._layers = LayerManager(krita_document)

    @classmethod
    def active(cls):
        if doc := Krita.instance().activeDocument():
            if (
                doc not in acquire_elements(Krita.instance().documents())
                or doc.activeNode() is None
            ):
                return None
            id = doc.rootNode().uniqueId().toString()
            return cls._instances.get(id) or KritaDocument(doc)
        return None

    @property
    def extent(self):
        return Extent(self._doc.width(), self._doc.height())

    @property
    def filename(self):
        return self._doc.fileName()

    @property
    def layers(self):
        return self._layers

    def check_color_mode(self):
        model = self._doc.colorModel()
        msg_fmt = _("Incompatible document: Color {0} must be {1} (current {0}: {2})")
        if model != "RGBA":
            return False, msg_fmt.format("model", "RGB/Alpha", model)
        depth = self._doc.colorDepth()
        if depth != "U8":
            return False, msg_fmt.format("depth", "8-bit integer", depth)
        return True, None

    def create_mask_from_selection(
        self, padding: float = 0.0, multiple=8, min_size=0, square=False, invert=False
    ):
        user_selection = self._doc.selection()
        if not user_selection:
            return None, None

        if _selection_is_entire_document(user_selection, self.extent):
            return None, None

        selection = user_selection.duplicate()
        original_bounds = Bounds(
            selection.x(), selection.y(), selection.width(), selection.height()
        )
        original_bounds = Bounds.clamp(original_bounds, self.extent)
        size_factor = original_bounds.extent.diagonal
        padding_pixels = int(padding * size_factor)

        if invert:
            selection.invert()

        bounds = _selection_bounds(selection)
        bounds = Bounds.pad(
            bounds, padding_pixels, multiple=multiple, min_size=min_size, square=square
        )
        bounds = Bounds.clamp(bounds, self.extent)
        data = selection.pixelData(*bounds)
        return Mask(bounds, data), original_bounds

    def get_image(self, bounds: Bounds | None = None, exclude_layers: list[Layer] | None = None):
        excluded: list[Layer] = []
        if exclude_layers:
            for layer in filter(lambda l: l.is_visible, exclude_layers):
                layer.hide()
                excluded.append(layer)
        if len(excluded) > 0:
            self._doc.refreshProjection()

        bounds = bounds or Bounds(0, 0, self._doc.width(), self._doc.height())
        img = Image.from_packed_bytes(self._doc.pixelData(*bounds), bounds.extent)

        for layer in excluded:
            layer.show()
        if len(excluded) > 0:
            self._doc.refreshProjection()
        return img

    def resize(self, extent: Extent):
        res = self._doc.resolution()
        self._doc.scaleImage(extent.width, extent.height, res, res, "Bilinear")

    def annotate(self, key: str, value: QByteArray):
        self._doc.setAnnotation(f"ai_diffusion/{key}", f"AI Diffusion Plugin: {key}", value)

    def find_annotation(self, key: str) -> QByteArray | None:
        result = self._doc.annotation(f"ai_diffusion/{key}")
        return result if result.size() > 0 else None

    def remove_annotation(self, key: str):
        self._doc.removeAnnotation(f"ai_diffusion/{key}")

    def add_pose_character(self, layer: Layer):
        assert layer.type is LayerType.vector
        _pose_layers.add_character(cast(krita.VectorLayer, layer.node))

    def import_animation(self, files: list[Path], offset: int = 0):
        success = self._doc.importAnimation([str(f) for f in files], offset, 1)
        if not success and len(files) > 0:
            folder = files[0].parent
            raise RuntimeError(f"Failed to import animation from {folder}")

    @property
    def selection_bounds(self):
        return self._selection_bounds

    @property
    def resolution(self):
        return self._doc.resolution() / 72.0  # KisImage::xRes which is applied to vectors

    @property
    def playback_time_range(self):
        return self._doc.playBackStartTime(), self._doc.playBackEndTime()

    @property
    def current_time(self):
        return self._doc.currentTime()

    @property
    def is_valid(self):
        return self._doc in acquire_elements(Krita.instance().documents())

    @property
    def is_active(self):
        return self._doc == Krita.instance().activeDocument()

    def _poll(self):
        if self.is_valid:
            selection = self._doc.selection()
            selection_bounds = _selection_bounds(selection) if selection else None
            if selection_bounds != self._selection_bounds:
                self._selection_bounds = selection_bounds
                self.selection_bounds_changed.emit()

            current_time = self.current_time
            if current_time != self._current_time:
                self._current_time = current_time
                self.current_time_changed.emit()
        else:
            self._poller.stop()

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, KritaDocument):
            return self._id == other._id
        return False


def _selection_bounds(selection: krita.Selection):
    return Bounds(selection.x(), selection.y(), selection.width(), selection.height())


def _selection_is_entire_document(selection: krita.Selection, extent: Extent):
    bounds = _selection_bounds(selection)
    if bounds.x > 0 or bounds.y > 0:
        return False
    if bounds.width + bounds.x < extent.width or bounds.height + bounds.y < extent.height:
        return False
    mask = selection.pixelData(*bounds)
    is_opaque = all(x == b"\xff" for x in mask)
    return is_opaque


class PoseLayers:
    _layers: dict[str, Pose] = {}
    _timer = QTimer()

    def __init__(self):
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    def update(self):
        doc = KritaDocument.active()
        if not doc:
            return
        try:
            layer = doc.layers.active
        except Exception:
            return
        if not layer or layer.type is not LayerType.vector:
            return

        layer = cast(krita.VectorLayer, layer.node)
        pose = self._layers.setdefault(layer.uniqueId(), Pose(doc.extent))
        self._update(layer, acquire_elements(layer.shapes()), pose, doc.resolution)

    def add_character(self, layer: krita.VectorLayer):
        doc = KritaDocument.active()
        assert doc is not None
        pose = self._layers.setdefault(layer.uniqueId(), Pose(doc.extent))
        svg = Pose.create_default(doc.extent, pose.people_count).to_svg()
        shapes = acquire_elements(layer.addShapesFromSvg(svg))
        self._update(layer, shapes, pose, doc.resolution)

    def _update(
        self, layer: krita.VectorLayer, shapes: list[krita.Shape], pose: Pose, resolution: float
    ):
        changes = pose.update(shapes, resolution)  # type: ignore
        if changes:
            shapes = layer.addShapesFromSvg(changes)
            for shape in shapes:
                shape.setZIndex(-1)


_pose_layers = PoseLayers()
