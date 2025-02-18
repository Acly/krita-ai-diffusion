from __future__ import annotations
from enum import Enum
from PyQt5.QtCore import QObject, QUuid, pyqtSignal

from . import eventloop, model, workflow
from .api import ConditioningInput, RegionInput
from .client import Client
from .image import Image, Bounds, Extent
from .document import Layer, LayerType
from .properties import Property, ObservableProperties
from .jobs import JobRegion
from .control import ControlLayerList
from .settings import settings


class RegionLink(Enum):
    direct = 0  # layer is directly linked to a region
    indirect = 1  # layer is in a group which is linked to a region
    any = 3  # either direct or indirect link


class Region(QObject, ObservableProperties):
    """A sub-area of the image where region-specific text prompts and control layers are applied.
    A region is linked to one or more layers. The layer's coverage mask defines the area of the region.
    """

    _parent: RootRegion
    _layers: list[QUuid]

    layer_ids = Property("", persist=True, setter="_set_layer_ids")
    positive = Property("", persist=True)
    control: ControlLayerList

    layer_ids_changed = pyqtSignal(str)
    positive_changed = pyqtSignal(str)
    modified = pyqtSignal(QObject, str)

    def __init__(self, parent: RootRegion, model: model.Model):
        super().__init__()
        self._parent = parent
        self._layers = []
        self.control = ControlLayerList(model)

    def _get_layers(self):
        col = self._parent._model.layers.updated()
        all = (col.find(id) for id in self._layers)
        pruned = [l for l in all if l is not None]
        self._set_layers([l.id for l in pruned])
        return pruned

    def _set_layers(self, ids: list[QUuid]):
        self._layers = ids
        new_ids_string = ",".join(id.toString() for id in ids)
        if self.layer_ids != new_ids_string:
            self._layer_ids = new_ids_string
            self.layer_ids_changed.emit(self._layer_ids)

    def _set_layer_ids(self, ids: str):
        if self._layer_ids == ids:
            return
        self._layer_ids = ids
        self._layers = [QUuid(id) for id in ids.split(",") if id]
        self.layer_ids_changed.emit(ids)
        self.modified.emit(self, "layer_ids")

    @property
    def layers(self):
        return self._get_layers()

    @property
    def first_layer(self):
        layers = self.layers
        return layers[0] if len(layers) > 0 else None

    @property
    def name(self):
        if len(self._layers) == 0:
            return "No layers linked"
        return ", ".join(l.name for l in self.layers)

    def link(self, layer: Layer):
        if layer.id not in self._layers:
            self._set_layers(self._layers + [layer.id])

    def unlink(self, layer: Layer):
        if layer.id in self._layers:
            self._set_layers([l for l in self._layers if l != layer.id])

    def is_linked(self, layer: Layer, mode=RegionLink.any):
        target = layer
        if mode is not RegionLink.direct:
            target = Region.link_target(layer)
        if mode is RegionLink.indirect and target is layer:
            return False
        if mode is RegionLink.direct or target is layer:
            return layer.id in self._layers
        return self.root.find_linked(target) is self

    def link_active(self):
        self.link(self._parent.layers.active)

    def unlink_active(self):
        self.unlink(self._parent.layers.active)

    def toggle_active_link(self):
        if self.is_active_linked:
            self.unlink_active()
        else:
            self.link_active()

    @property
    def has_links(self):
        return len(self._layers) > 0

    @property
    def is_active_linked(self):
        return self.is_linked(self._parent.layers.active)

    def remove(self):
        self._parent.remove(self)

    @property
    def root(self):
        return self._parent

    @property
    def siblings(self):
        return self._parent.find_siblings(self)

    @staticmethod
    def link_target(layer: Layer):
        if layer.type is LayerType.group:
            return layer
        if parent := layer.parent_layer:
            if not parent.is_root and parent.type is LayerType.group:
                return parent
        return layer

    async def translate_prompt(self, client: Client):
        if positive := self.positive:
            translated = await client.translate(positive, settings.prompt_translation)
            if positive == self.positive:
                self.positive = translated


class RootRegion(QObject, ObservableProperties):
    """Manages a collection of regions, each of which is linked to one or more layers in the document.
    Defines text prompt and control layers which are applied to all regions in the collection.
    If there are no regions, the root region is used as a default for the entire document.
    """

    _model: model.Model
    _regions: list[Region]
    _active: Region | None = None
    _active_layer: QUuid | None = None

    positive = Property("", persist=True)
    negative = Property("", persist=True)
    control: ControlLayerList

    positive_changed = pyqtSignal(str)
    negative_changed = pyqtSignal(str)
    active_changed = pyqtSignal(Region)
    active_layer_changed = pyqtSignal()
    added = pyqtSignal(Region)
    removed = pyqtSignal(Region)
    modified = pyqtSignal(QObject, str)

    def __init__(self, model: model.Model):
        super().__init__()
        self._model = model
        self._regions = []
        self.control = ControlLayerList(model)
        model.layers.active_changed.connect(self._update_active)
        model.layers.parent_changed.connect(self._update_group)

    def _find_region(self, layer: Layer):
        return next((r for r in self._regions if r.is_linked(layer, RegionLink.direct)), None)

    def emplace(self):
        region = Region(self, self._model)
        self._regions.append(region)
        return region

    @property
    def active(self):
        self._update_active()
        return self._active

    @active.setter
    def active(self, region: RootRegion | Region | None):
        if isinstance(region, RootRegion):
            region = None
        if self._active != region:
            self._active = region
            self.active_changed.emit(region)
            self._track_layer(region)

    @property
    def active_or_root(self):
        return self.active or self

    @property
    def region_for_active_layer(self):
        if layer := self._get_active_layer()[0]:
            return self.find_linked(layer)

    def get_active_region_layer(self, use_parent: bool):
        result = self.layers.root
        target = Region.link_target(self.layers.active)
        if self.is_linked(target):
            result = target
        if use_parent and result.parent_layer is not None:
            result = result.parent_layer
        return result

    def add_control(self):
        self.active_or_root.control.add()

    def is_linked(self, layer: Layer, mode=RegionLink.any):
        return any(r.is_linked(layer, mode) for r in self._regions)

    def find_linked(self, layer: Layer, mode=RegionLink.any):
        return next((r for r in self._regions if r.is_linked(layer, mode)), None)

    def create_region_layer(self):
        self.create_region(group=False)

    def create_region_group(self):
        self.create_region(group=True)

    def create_region(self, group=True):
        """Create a new region. This action depends on context:
        If the active layer can be linked to a group it will be used as the initial link
        target for the new group. Otherwise, a new layer is inserted (or a group if group==True)
        and that will be linked instead.
        """
        layers = self._model.layers
        target = Region.link_target(layers.active)
        can_link = target.type in [LayerType.paint, LayerType.group] and not self.is_linked(target)
        if can_link:
            layer = target
        elif group:
            layer = layers.create_group(f"Region {len(self)}")
            layers.create("Paint layer", parent=layer)
        else:
            layer = layers.create(f"Region {len(self)}")
        return self._add(layer)

    def remove(self, region: Region):
        if region in self._regions:
            if self.active == region:
                self.active = None
            self._regions.remove(region)
            self.removed.emit(region)

    def _get_regions(self, layers: list[Layer], exclude: Region | None = None):
        regions = []
        for l in layers:
            r = self._find_region(l)
            if r is not None and r is not exclude and r not in regions:
                regions.append(r)
        return regions

    def find_siblings(self, region: Region):
        if layer := region.first_layer:
            below, above = layer.siblings
            return self._get_regions(below, region), self._get_regions(above, region)
        return [], []

    @property
    def siblings(self):
        if self.layers:
            layer = self.layers.root
            if active_layer := self._get_active_layer()[0]:
                active_layer = Region.link_target(active_layer)
                if self.is_linked(active_layer):
                    layer = active_layer.parent_layer or active_layer
            return [], self._get_regions(layer.child_layers)
        return [], []

    def last_unlinked_layer(self, parent: Layer):
        result = None
        for node in parent.child_layers:
            if self.is_linked(node):
                break
            result = node
        return result

    def _get_active_layer(self):
        if not self.layers:
            return None, False
        layer = self.layers.active
        if layer.id == self._active_layer:
            return layer, False
        self._active_layer = layer.id
        self.active_layer_changed.emit()
        return layer, True

    def _update_active(self):
        layer, changed = self._get_active_layer()
        if layer and changed:
            if region := self.find_linked(layer):
                self.active = region

    def _track_layer(self, region: Region | None):
        if region and region.first_layer:
            layer, changed = self._get_active_layer()
            if layer and not changed and not region.is_linked(layer):
                target = region.first_layer
                if target.type is LayerType.group and len(target.child_layers) > 0:
                    target = target.child_layers[-1]
                self.layers.active = target

    def _update_group(self, layer: Layer):
        """If a layer is moved into a group, promote the region to non-destructive apply workflow."""
        if layer.type is not LayerType.group:
            if region := self.find_linked(layer, RegionLink.direct):
                if parent := layer.parent_layer:
                    if not parent.is_root:
                        region.unlink(layer)
                        region.link(parent)

    def _add(self, layer: Layer):
        region = Region(self, self._model)
        region.link(layer)
        self._regions.append(region)
        self.added.emit(region)
        self.active = region
        return region

    @property
    def layers(self):
        return self._model.layers

    async def translate_prompt(self, client: Client):
        if positive := self.positive:
            translated = await client.translate(positive, settings.prompt_translation)
            if positive == self.positive:
                self.positive = translated
        if self.negative:
            negative = self.negative
            translated = await client.translate(negative, settings.prompt_translation)
            if negative == self.negative:
                self.negative = translated

    def __len__(self):
        return len(self._regions)

    def __iter__(self):
        return iter(self._regions)


def translate_prompt(region: Region | RootRegion):
    from .root import root

    if client := root.connection.client_if_connected:
        if settings.prompt_translation and client.features.translation:
            eventloop.run(region.translate_prompt(client))


def get_region_inpaint_mask(region_layer: Layer, max_extent: Extent, min_size=0):
    region_bounds = region_layer.compute_bounds()
    padding = int((settings.selection_padding / 100) * region_bounds.extent.average_side)
    bounds = Bounds.pad(region_bounds, padding, min_size=min_size, square=min_size > 0)
    bounds = Bounds.clamp(bounds, max_extent)
    mask_image = region_layer.get_mask(bounds)
    return mask_image.to_mask(bounds)


def process_regions(
    root: RootRegion,
    bounds: Bounds,
    parent_layer: Layer | None = None,
    min_coverage=0.02,
    time: int | None = None,
):
    parent_region = None
    if parent_layer and not parent_layer.is_root:
        parent_region = root.find_linked(parent_layer)

    parent_prompt = ""
    job_info = []
    control = root.control.to_api(bounds, time)
    if parent_layer and parent_region:
        parent_prompt = parent_region.positive
        control += parent_region.control.to_api(bounds, time)
        job_info = [JobRegion(parent_layer.id_string, parent_prompt, bounds)]
    result = ConditioningInput(
        positive=workflow.merge_prompt(parent_prompt, root.positive),
        negative=root.negative,
        control=control,
    )

    # Collect layers with linked regions. Optionally restrict to to child layers of a region.
    if parent_layer is not None:
        child_layers = parent_layer.child_layers
    else:
        child_layers = root.layers.all
        parent_layer = root.layers.root
    layer_regions = ((l, root.find_linked(l, RegionLink.direct)) for l in child_layers)
    layer_regions = [(l, r) for l, r in layer_regions if r is not None]
    if len(layer_regions) == 0:
        return result, job_info

    # Get region masks. Filter out regions with:
    # * no content (empty mask)
    # * less than minimum overlap (estimate based on bounding box)
    result_regions: list[tuple[RegionInput, JobRegion]] = []
    for layer, region in layer_regions:
        layer_bounds = layer.compute_bounds()
        if layer_bounds.area == 0:
            continue

        coverage_rough = Bounds.intersection(bounds, layer_bounds).area / bounds.area
        if coverage_rough < 2 * min_coverage:
            continue

        region_result = RegionInput(
            layer.get_mask(bounds),
            layer_bounds,
            workflow.merge_prompt(region.positive, root.positive),
            control=region.control.to_api(bounds, time),
        )
        job_params = JobRegion(layer.id_string, region.positive, layer_bounds)
        result_regions.append((region_result, job_params))

    # Remove from each region mask any overlapping areas from regions above it.
    accumulated_mask = None
    for i in range(len(result_regions) - 1, -1, -1):
        region, job_region = result_regions[i]
        assert region.mask is not None
        mask = region.mask
        if accumulated_mask is not None:
            mask = Image.mask_subtract(mask, accumulated_mask)

        coverage = mask.average()
        if coverage > 0.9 and min_coverage > 0:
            # Single region covers (almost) entire image, don't use regional conditioning.
            result.positive = region.positive
            result.control += region.control
            return result, [job_region]
        elif coverage < min_coverage:
            # Region has less than minimum coverage, remove it.
            result_regions.pop(i)
        else:
            # Accumulate mask for next region, and store modified mask.
            if accumulated_mask is None:
                accumulated_mask = Image.copy(region.mask)
            accumulated_mask = Image.mask_add(accumulated_mask, region.mask)
            region.mask = mask

    # If there are no regions left, don't use regional conditioning.
    if len(result_regions) == 0:
        return result, job_info

    # If the region(s) don't cover the entire image, add a final region for the remaining area.
    assert accumulated_mask is not None, "Expecting at least one region mask"
    total_coverage = accumulated_mask.average()
    if total_coverage < 0.95:
        accumulated_mask.invert()
        input = RegionInput(accumulated_mask, bounds, result.positive)
        job = JobRegion(parent_layer.id_string, "background", bounds, is_background=True)
        result_regions.insert(0, (input, job))

    result.regions = [r for r, _ in result_regions]
    return result, [j for _, j in result_regions]
