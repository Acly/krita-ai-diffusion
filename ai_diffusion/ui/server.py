from __future__ import annotations

from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from krita import Krita
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ai_diffusion.network import DownloadProgress

from .. import eventloop, resources, server, util
from ..connection import ConnectionState
from ..localization import translate as _
from ..platform_tools import get_cuda_devices
from ..resources import CustomNode, ModelRequirements, ModelResource, ResourceId
from ..root import root
from ..server import Server, ServerBackend, ServerState
from ..settings import ServerMode, Settings, settings
from ..style import Arch
from ..util import ensure
from .theme import SignalBlocker, add_header, green, grey, highlight, red, set_text_clipped, yellow


class PackageState(Enum):
    installed = 0
    selected = 1
    available = 2
    disabled = 3


@dataclass
class PackageItem:
    label: QLabel
    status: QLabel | QCheckBox
    package: str | ModelResource | CustomNode
    state: PackageState


class PackageGroupWidget(QWidget):
    changed = pyqtSignal()

    def __init__(
        self,
        name: str,
        packages: list[str] | list[ModelResource] | list[CustomNode],
        description: str | None = None,
        is_expanded=True,
        is_checkable=False,
        parent=None,
    ):
        super().__init__(parent)
        self._workloads: list[Arch] = []
        self._backend = settings.server_backend
        self._is_checkable = False

        self._layout = QGridLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setColumnMinimumWidth(0, 300)
        self.setLayout(self._layout)

        self._header = QToolButton(self)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setContentsMargins(0, 0, 0, 0)
        self._header.setText(name)
        self._header.setCheckable(True)
        self._header.setChecked(is_expanded)
        self._header.setStyleSheet("font-weight:bold; border:none;")
        self._layout.addWidget(self._header, 0, 0)
        self._header.toggled.connect(self._update_visibility)

        self._status = QLabel(self)
        self._layout.addWidget(self._status, 0, 1)

        self._desc: QLabel | None = None
        if description:
            desc = self._desc = QLabel(self)
            desc.setText(description)
            desc.setContentsMargins(20, 0, 0, 0)
            desc.setWordWrap(True)
            desc.setTextFormat(Qt.TextFormat.RichText)
            desc.setOpenExternalLinks(True)
            self._layout.addWidget(desc, 1, 0, 1, 2)

        self._is_checkable = is_checkable
        self._items: list[PackageItem] = [self.add_item(p) for p in packages]
        self._update_visibility()

    def _update_item_visibility(self, item: PackageItem):
        supported = _backend_supports(self.backend, item)
        item.label.setVisible(supported and self._header.isChecked())
        item.status.setVisible(supported and self._header.isChecked())

    def _update_visibility(self):
        self._header.setArrowType(
            Qt.ArrowType.DownArrow if self._header.isChecked() else Qt.ArrowType.RightArrow
        )
        if self._desc:
            self._desc.setVisible(self._header.isChecked())
        for item in self._items:
            self._update_item_visibility(item)

    def expand(self):
        if not self._header.isChecked():
            self._header.setChecked(True)

    def add_item(self, package: str | ModelResource | CustomNode):
        item = PackageItem(
            package=package,
            state=PackageState.available,
            label=QLabel(self._package_name(package), self),
            status=QCheckBox(_("Install"), self) if self.is_checkable else QLabel(self),
        )
        item.label.setContentsMargins(20, 0, 0, 0)
        if self.is_checkable:
            item.status.setChecked(False)
            item.status.toggled.connect(self._handle_checkbox_toggle)
        self._layout.addWidget(item.label, self._layout.rowCount(), 0)
        self._layout.addWidget(item.status, self._layout.rowCount() - 1, 1)
        return item

    @property
    def is_checkable(self):
        return self._is_checkable

    @property
    def values(self):
        return [item.state for item in self._items]

    @values.setter
    def values(self, values: list[PackageState]):
        for item, value in zip(self._items, values):
            item.state = value
        self._update()

    def _update(self):
        for item in self._items:
            self._update_item_visibility(item)
            self._update_workload(item)
            if item.state is PackageState.installed:
                item.status.setText(_("Installed"))
                item.status.setStyleSheet(f"color:{green}")
            elif item.state is PackageState.available:
                item.status.setText(_("Not installed"))
                item.status.setStyleSheet("")
            if self.is_checkable:
                if item.state is PackageState.selected:
                    item.status.setText(_("Not installed"))
                    item.status.setStyleSheet("")
                elif item.state is PackageState.disabled:
                    if not _backend_supports(self.backend, item):
                        item.status.setText(_("Not supported"))
                    else:
                        item.status.setText(_("Workload not selected"))
                    item.status.setStyleSheet(f"color:{grey}")
                with SignalBlocker(item.status):
                    item.status.setChecked(
                        item.state in [PackageState.selected, PackageState.installed]
                    )
                    item.status.setEnabled(item.state is not PackageState.disabled)
        self._update_status()

    def _workload_matches(self, item: PackageItem):
        archs_with_workload = (
            Arch.sd15,
            Arch.sdxl,
            Arch.flux,
            Arch.flux_k,
            Arch.flux2_4b,
            Arch.zimage,
        )
        return (
            not isinstance(item.package, ModelResource)
            or item.package.arch in self.workloads
            or item.package.arch not in archs_with_workload
        )

    def _update_workload(self, item: PackageItem):
        enabled = _backend_supports(self.backend, item) and self._workload_matches(item)
        if not enabled and item.state in [PackageState.selected, PackageState.available]:
            item.state = PackageState.disabled
        elif enabled and item.state is PackageState.disabled:
            item.state = PackageState.available

    @property
    def package_names(self):
        return [self._package_id(item.package) for item in self._items]

    @property
    def selected_packages(self):
        return [
            self._package_id(item.package)
            for item in self._items
            if item.state is PackageState.selected
        ]

    @property
    def workloads(self):
        return self._workloads

    @workloads.setter
    def workloads(self, workloads: list[Arch]):
        self._workloads = workloads
        self._update()

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend: ServerBackend):
        self._backend = backend
        self._update()

    def set_installed(self, installed: list[bool]):
        for item, is_installed in zip(self._items, installed):
            if is_installed:
                item.state = PackageState.installed
            elif item.state is PackageState.installed:
                item.state = PackageState.available
        self._update()

    def _update_status(self):
        items = [i for i in self._items if _backend_supports(self.backend, i)]
        available = sum(item.state is PackageState.available for item in items)
        if all(item.state is PackageState.installed for item in items):
            self._status.setText(_("All installed"))
            self._status.setStyleSheet(f"color:{green}")
        elif self.is_checkable:
            selected = sum(item.state is PackageState.selected for item in items)
            if selected > 0:
                self._status.setText(
                    f"{selected} of {selected + available} " + _("packages selected")
                )
                self._status.setStyleSheet(f"color:{yellow}")
            else:
                self._status.setText(f"{available} " + _("packages available"))
                self._status.setStyleSheet(f"color:{grey}")
        else:
            self._status.setText(f"{available} " + _("packages require installation"))
            self._status.setStyleSheet(f"color:{yellow}")

    def _handle_checkbox_toggle(self):
        for item in self._items:
            if item.state in [PackageState.available, PackageState.selected]:
                item.state = (
                    PackageState.selected if item.status.isChecked() else PackageState.available
                )
        self._update()
        self.changed.emit()

    def _package_name(self, package: str | ModelResource | CustomNode):
        return package if isinstance(package, str) else package.name

    def _package_id(self, package: str | ModelResource | CustomNode):
        if isinstance(package, ModelResource):
            return package.id.string
        return self._package_name(package)


def _backend_supports(backend: ServerBackend, item: PackageItem | ModelResource):
    if isinstance(item, PackageItem) and isinstance(item.package, ModelResource):
        item = item.package
    if isinstance(item, ModelResource):
        req = item.requirements
        has_fp4 = any(major >= 10 for major, minor in get_cuda_devices())  # Blackwell and later
        if backend is ServerBackend.cuda and has_fp4:
            return req not in [ModelRequirements.no_cuda, ModelRequirements.cuda]
        elif backend is ServerBackend.cuda:
            return req not in [ModelRequirements.no_cuda, ModelRequirements.cuda_fp4]
        else:
            return req not in [ModelRequirements.cuda, ModelRequirements.cuda_fp4]
    return True


def _filter_by_arch(models: Iterable[ModelResource], archs: Arch | Iterable[Arch]):
    archs = (archs,) if isinstance(archs, Arch) else archs
    return [m.id.string for m in models if m.arch in archs]


def _enabled_workloads(selected: list[str], required: Iterable[ModelResource], server: Server):
    workloads = {arch: True for arch in Arch}
    for m in required:
        if not (m.id.string in selected or server.is_installed(m)):
            workloads[m.arch] = False
    if not workloads[Arch.all]:
        workloads = {k: False for k in workloads}
    return workloads


class CustomPackageTab(QWidget):
    title = _("Individual Packages")
    workloads = (Arch.sd15, Arch.sdxl, Arch.flux, Arch.flux2_4b, Arch.zimage)
    workload_models = resources.required_models

    selected_models_changed = pyqtSignal()

    def __init__(self, server: Server, parent=None):
        super().__init__(parent)
        self._server = server

        layout = QVBoxLayout()
        self.setLayout(layout)

        self._required_group = PackageGroupWidget(
            _("Core components"),
            ["Python", "ComfyUI", _("Custom nodes"), _("Required models")],
            is_expanded=False,
            parent=self,
        )
        layout.addWidget(self._required_group)

        self._workload_group = PackageGroupWidget(
            _("Workloads"),
            [_("Stable Diffusion 1.5"), _("Stable Diffusion XL"), "Flux", "Flux 2", "Z-Image"],
            description=(
                _("Choose a Diffusion base model to install its basic requirements.")
                + " <a href='https://docs.interstice.cloud/base-models'>"
                + _("Read more about workloads.")
                + "</a>"
            ),
            is_checkable=True,
            parent=self,
        )
        self._workload_group.changed.connect(self._change_workload)
        layout.addWidget(self._workload_group)

        optional_models = resources.default_checkpoints + resources.optional_models
        self._packages: dict[str, PackageGroupWidget] = {
            "upscalers": PackageGroupWidget(
                _("Upscalers (super-resolution)"),
                resources.upscale_models,
                is_checkable=True,
                parent=self,
            ),
            "sd15": PackageGroupWidget(
                _("Stable Diffusion 1.5 models"),
                [m for m in optional_models if m.arch is Arch.sd15],
                is_checkable=True,
                is_expanded=False,
                parent=self,
            ),
            "sdxl": PackageGroupWidget(
                _("Stable Diffusion XL models"),
                [m for m in optional_models if m.arch is Arch.sdxl],
                is_checkable=True,
                is_expanded=False,
                parent=self,
            ),
            "illu": PackageGroupWidget(
                _("Illustrious/NoobAI XL models"),
                [m for m in optional_models if m.arch in [Arch.illu, Arch.illu_v]],
                is_checkable=True,
                is_expanded=False,
                parent=self,
            ),
            "flux": PackageGroupWidget(
                _("Flux models"),
                [m for m in optional_models if m.arch in [Arch.flux, Arch.flux_k, Arch.chroma]],
                is_checkable=True,
                is_expanded=False,
                parent=self,
            ),
            "flux2": PackageGroupWidget(
                _("Flux 2 models"),
                [m for m in optional_models if m.arch.is_flux2],
                is_checkable=True,
                is_expanded=False,
                parent=self,
            ),
            "zimage": PackageGroupWidget(
                _("Z-Image models"),
                [m for m in optional_models if m.arch is Arch.zimage],
                is_checkable=True,
                is_expanded=False,
                parent=self,
            ),
        }

        for group in ["upscalers", "sd15", "sdxl", "illu", "flux", "flux2", "zimage"]:
            self._packages[group].changed.connect(self._change_models)
            layout.addWidget(self._packages[group])

        layout.addStretch()

        self.update_installed()

    def update_installed(self):
        has_missing_nodes = any(
            node.name in self._server.missing_resources for node in resources.required_custom_nodes
        )
        has_missing_models = any(
            model.id.string in self._server.missing_resources
            for model in resources.required_models
            if model.arch is Arch.all
        )
        installed_status = [
            self._server.has_python,
            self._server.has_comfy,
            not has_missing_nodes,
            not has_missing_models,
        ]
        self._required_group.set_installed(installed_status)

        self._workload_group.set_installed([
            self._server.all_installed(_filter_by_arch(self.workload_models, arch))
            for arch in self.workloads
        ])

        installed_workloads = self._selected_workloads(installed=True)
        for widget in self._packages.values():
            widget.workloads = installed_workloads
            widget.backend = self._server.backend
            widget.set_installed([self._server.is_installed(p) for p in widget.package_names])

    def update_backend(self):
        for widget in self._packages.values():
            widget.backend = self._server.backend

    def _update_workloads(self):
        workloads = self._selected_workloads(installed=True)
        for widget in self._packages.values():
            widget.workloads = workloads

    def _change_models(self):
        self.selected_models_changed.emit()

    def _change_workload(self):
        self._update_workloads()
        self.selected_models_changed.emit()

    def _selected_workloads(self, installed=False):
        check = (PackageState.selected,)
        if installed:
            check = (PackageState.selected, PackageState.installed)
        selected_or_installed = [state in check for state in self._workload_group.values]
        return [arch for arch, selected in zip(self.workloads, selected_or_installed) if selected]

    @property
    def selected_models(self):
        selected_workloads = [Arch.all] + self._selected_workloads(installed=False)
        workload_models = _filter_by_arch(self.workload_models, selected_workloads)
        optional_models = [
            model for widget in self._packages.values() for model in widget.selected_packages
        ]
        return workload_models + optional_models

    @selected_models.setter
    def selected_models(self, value: list[str]):
        workloads = _enabled_workloads(value, self.workload_models, self._server)
        new_states = copy(self._workload_group.values)
        for i, arch in enumerate(self.workloads):
            if new_states[i] is not PackageState.installed:
                if workloads[arch]:
                    new_states[i] = PackageState.selected
                else:
                    new_states[i] = PackageState.available
        if new_states != self._workload_group.values:
            self._workload_group.values = new_states
            self._update_workloads()

        for widget in self._packages.values():
            selected = [p for p in widget.package_names if p in value]
            states: list[PackageState] = []
            for state, pkg in zip(widget.values, widget.package_names):
                if state is PackageState.installed:
                    states.append(PackageState.installed)
                elif pkg in selected:
                    states.append(PackageState.selected)
                else:
                    states.append(PackageState.available)
            if states != widget.values:
                widget.values = states


class ModelPropsWidget(QWidget):
    def __init__(
        self, size: int, vram: int, speed: int, fidelity: int, understanding: int, parent=None
    ):
        super().__init__(parent)

        layout = QHBoxLayout()
        layout.setSpacing(4)
        self.setLayout(layout)

        size_label = QLabel(f"Install: <b>{size} GB</b>", self)
        size_label.setToolTip(_("Minimum download and installation size"))
        layout.addWidget(size_label)

        vram_label = QLabel(f"VRAM: <b>{vram} GB</b>", self)
        vram_label.setToolTip(_("Minimum recommended GPU VRAM to run"))
        layout.addWidget(vram_label)

        text = {
            -2: _("Very Slow"),
            -1: _("Slow"),
            0: _("Average"),
            1: _("Fast"),
            2: _("Very Fast"),
        }[speed]
        col = {-2: red, -1: red, 0: yellow, 1: green, 2: green}[speed]
        speed_label = QLabel(f"Speed: <span style='color:{col}'><b>{text}</b></span>", self)
        speed_label.setToolTip(_("How fast the model generates images"))
        layout.addWidget(speed_label)

        text = {
            -1: _("Low"),
            0: _("Average"),
            1: _("High"),
            2: _("Very High"),
        }[fidelity]
        col = {-1: red, 0: yellow, 1: green, 2: green}[fidelity]
        fidelity_label = QLabel(f"Fidelity: <span style='color:{col}'><b>{text}</b></span>", self)
        fidelity_label.setToolTip(_("Visual quality of the generated images"))
        layout.addWidget(fidelity_label)

        text = {
            -1: _("Poor"),
            0: _("Average"),
            1: _("Good"),
            2: _("Excellent"),
        }[understanding]
        col = {-1: red, 0: yellow, 1: green, 2: green}[understanding]
        understanding_label = QLabel(
            f"Understanding: <span style='color:{col}'><b>{text}</b></span>", self
        )
        understanding_label.setToolTip(_("How well the model understands and follows text prompts"))
        layout.addWidget(understanding_label)


class ModelCheckBox:
    def __init__(self, label: str, arch: Arch, model_ids: str | tuple, layout: QVBoxLayout):
        self.label = label
        self.arch = arch
        self.model_ids = model_ids
        self.widget = QCheckBox(label)
        self._state = PackageState.available

        layout.addWidget(self.widget)
        self.widget.toggled.connect(self._update_state)

    def _update_state(self):
        if self.state is PackageState.available and self.widget.isChecked():
            self.state = PackageState.selected
        elif self.state is PackageState.selected and not self.widget.isChecked():
            self.state = PackageState.available

    def model_id(self, backend: ServerBackend) -> str:
        if isinstance(self.model_ids, str):
            return self.model_ids
        for id in self.model_ids:
            res = resources.find_resource(ResourceId.parse(id))
            if res and _backend_supports(backend, res):
                return id
        return self.model_ids[0]

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value: PackageState):
        self._state = value
        with SignalBlocker(self.widget):
            if value is PackageState.installed:
                self.widget.setStyleSheet(f"color:{green}")
                self.widget.setEnabled(False)
                self.widget.setChecked(True)
            else:
                self.widget.setStyleSheet("")
                self.widget.setEnabled(True)
                self.widget.setChecked(value is PackageState.selected)


class WorkloadsTab(QWidget):
    title = _("Workloads")
    workloads = (Arch.sdxl, Arch.illu, Arch.flux2_4b, Arch.zimage, Arch.flux, Arch.sd15)
    workload_models = resources.required_models + resources.recommended_models

    selected_models_changed = pyqtSignal()

    def __init__(self, server: Server, parent=None):
        super().__init__(parent)
        self._server = server
        self._models: list[ModelCheckBox] = []

        layout = QVBoxLayout()
        self.setLayout(layout)

        self._pkg_sdxl = QWidget(self)
        layout.addWidget(self._pkg_sdxl)

        sdxl_layout = QVBoxLayout(self._pkg_sdxl)
        sdxl_header = QLabel("<b>SDXL - Stable Diffusion XL</b>", self._pkg_sdxl)
        sdxl_layout.addWidget(sdxl_header)
        sdxl_props = ModelPropsWidget(
            size=22, vram=6, speed=1, fidelity=1, understanding=0, parent=self
        )
        sdxl_layout.addWidget(sdxl_props)
        desc = (
            _("Flexible base model with a huge ecosystem. Great for iterating on images quickly.")
            + "<br>"
            + _("Choose models below depending on the content of your images:")
        )
        sdxl_desc = QLabel(desc, self._pkg_sdxl)
        sdxl_desc.setWordWrap(True)
        sdxl_layout.addWidget(sdxl_desc)
        self._models += [
            ModelCheckBox(
                "RealVis XL - " + _("for Photography and realistic images"),
                Arch.sdxl,
                "checkpoint-realvis-sdxl",
                sdxl_layout,
            ),
            ModelCheckBox(
                "ZavyChroma XL - " + _("for Illustrations and digital art"),
                Arch.sdxl,
                "checkpoint-zavychroma-sdxl",
                sdxl_layout,
            ),
            ModelCheckBox(
                "Nova Anime XL - " + _("for Anime and illustration"),
                Arch.illu,
                "checkpoint-nova-illu",
                sdxl_layout,
            ),
        ]

        self.add_separator(layout)

        self._pkg_flux2 = QWidget(self)
        layout.addWidget(self._pkg_flux2)

        flux2_layout = QVBoxLayout(self._pkg_flux2)
        flux2_header = QLabel("<b>FLUX 2</b>", self._pkg_flux2)
        flux2_layout.addWidget(flux2_header)
        flux2_props = ModelPropsWidget(
            size=7, vram=8, speed=1, fidelity=1, understanding=1, parent=self
        )
        flux2_layout.addWidget(flux2_props)
        desc = _(
            "Versatile model with sharp details. Can generate and edit images with instructions. Sometimes struggles with image continuity."
        )
        flux2_desc = QLabel(desc, self._pkg_flux2)
        flux2_desc.setWordWrap(True)
        flux2_layout.addWidget(flux2_desc)
        self._models += [
            ModelCheckBox(
                "Flux.2 [klein] 4B - " + _("Compact generation and edit model"),
                Arch.flux2_4b,
                ("checkpoint-fp8-flux2_4b", "checkpoint-q6_k-flux2_4b"),
                flux2_layout,
            ),
        ]

        self.add_separator(layout)

        self._pkg_zimage = QWidget(self)
        layout.addWidget(self._pkg_zimage)

        zimage_layout = QVBoxLayout(self._pkg_zimage)
        zimage_header = QLabel("<b>Z-Image</b>", self._pkg_zimage)
        zimage_layout.addWidget(zimage_header)
        zimage_props = ModelPropsWidget(
            size=12, vram=12, speed=0, fidelity=2, understanding=1, parent=self
        )
        zimage_layout.addWidget(zimage_props)
        desc = _(
            "Powerful and efficient model for stronger hardware. Good understanding of natural language (Chinese and English). The Turbo variant is fast and heavily tuned for realistic results."
        )
        zimage_desc = QLabel(desc, self._pkg_zimage)
        zimage_desc.setWordWrap(True)
        zimage_layout.addWidget(zimage_desc)
        self._models += [
            ModelCheckBox(
                "Z-Image Turbo - " + _("for Photography and realistic images"),
                Arch.zimage,
                "checkpoint-turbo_fp8-zimage",
                zimage_layout,
            ),
        ]

        self.add_separator(layout)

        self._pkg_flux = QWidget(self)
        layout.addWidget(self._pkg_flux)

        flux_layout = QVBoxLayout(self._pkg_flux)
        flux_header = QLabel("<b>FLUX 1</b>", self._pkg_flux)
        flux_layout.addWidget(flux_header)
        flux_props = ModelPropsWidget(
            size=26, vram=10, speed=-1, fidelity=2, understanding=1, parent=self
        )
        flux_layout.addWidget(flux_props)
        desc = _(
            "Strong base model with consistent high-quality compositions and details. Good understanding of natural language (English). Limited flexibility for art styles."
        )
        flux_desc = QLabel(desc, self._pkg_flux)
        flux_desc.setWordWrap(True)
        flux_layout.addWidget(flux_desc)
        self._models += [
            ModelCheckBox(
                "Flux Krea - " + _("General-purpose model for photography and illustration"),
                Arch.flux,
                (
                    "checkpoint-flux_dev-flux",
                    "checkpoint-flux_dev_nunchaku-flux",
                    "checkpoint-flux_dev_nunchaku_fp4-flux",
                ),
                flux_layout,
            ),
            ModelCheckBox(
                "Flux Kontext - " + _("Specialized model for instruction-based editing"),
                Arch.flux,
                (
                    "checkpoint-flux_kontext-flux",
                    "checkpoint-flux_kontext_nunchaku-flux",
                    "checkpoint-flux_kontext_nunchaku_fp4-flux",
                ),
                flux_layout,
            ),
        ]

        self.add_separator(layout)

        self._pkg_sd15 = QWidget(self)
        layout.addWidget(self._pkg_sd15)
        sd15_layout = QVBoxLayout(self._pkg_sd15)
        sd15_header = QLabel("<b>SD 1.5 - Stable Diffusion 1.5</b>", self._pkg_sd15)
        sd15_layout.addWidget(sd15_header)
        sd15_props = ModelPropsWidget(
            size=16, vram=4, speed=2, fidelity=-1, understanding=-1, parent=self
        )
        sd15_layout.addWidget(sd15_props)
        desc = (
            _(
                "Older base model with good flexibility and many extensions available. Great for live painting and systems without powerful hardware. Not recommended for generating full images from text."
            )
            + "<br>"
            + _("Choose models below depending on the content of your images:")
        )
        sd15_desc = QLabel(desc, self._pkg_sd15)
        sd15_desc.setWordWrap(True)
        sd15_layout.addWidget(sd15_desc)
        self._models += [
            ModelCheckBox(
                "Serenity - " + _("for Photography and realistic images"),
                Arch.sd15,
                "checkpoint-serenity-sd15",
                sd15_layout,
            ),
            ModelCheckBox(
                "DreamShaper - " + _("for Illustrations and digital art"),
                Arch.sd15,
                "checkpoint-dreamshaper-sd15",
                sd15_layout,
            ),
        ]

        layout.addStretch()

        self.update_installed()
        for m in self._models:
            m.widget.toggled.connect(self._change_models)

    @staticmethod
    def add_separator(layout: QVBoxLayout):
        line_sep = QFrame()
        line_sep.setFrameShape(QFrame.Shape.HLine)
        line_sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line_sep)

    def update_installed(self):
        workload_installed = _enabled_workloads([], self.workload_models, self._server)
        for m in self._models:
            id = m.model_id(self._server.backend)
            if workload_installed[m.arch] and self._server.is_installed(id):
                m.state = PackageState.installed
            elif m.state is PackageState.installed:
                m.state = PackageState.available

    def _change_models(self):
        self.selected_models_changed.emit()

    @property
    def selected_models(self):
        result: list[str] = []
        archs = {Arch.all}
        for m in self._models:
            if m.state is PackageState.selected:
                result.append(m.model_id(self._server.backend))
                archs.add(m.arch)
        result.extend(
            m.id.string
            for m in self.workload_models
            if m.arch in archs and not self._server.is_installed(m)
        )
        return result

    @selected_models.setter
    def selected_models(self, value: list[str]):
        workloads = _enabled_workloads(value, self.workload_models, self._server)
        for m in self._models:
            if m.state is not PackageState.installed:
                if m.model_id(self._server.backend) in value and workloads[m.arch]:
                    m.state = PackageState.selected
                else:
                    m.state = PackageState.available


class ServerWidget(QWidget):
    state_changed = pyqtSignal()

    def __init__(self, srv: Server, parent=None):
        super().__init__(parent)
        self._server = srv
        self._error = ""
        self._selected_models: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        add_header(layout, Settings._server_path)

        self._location_edit = QLineEdit(self)
        self._location_edit.textChanged.connect(self._change_location)

        self._location_select = QToolButton(self)
        self._location_select.setIcon(Krita.instance().icon("document-open"))
        self._location_select.clicked.connect(self._select_location)

        location_layout = QHBoxLayout()
        location_layout.addWidget(self._location_edit)
        location_layout.addWidget(self._location_select)
        layout.addLayout(location_layout)

        self._status_label = QLabel(self)
        self._status_label.setStyleSheet("font-weight:bold")
        self._status_label.setWordWrap(True)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumHeight(16)

        self._progress_info = QLabel(self)
        self._progress_info.setStyleSheet("font-style:italic")
        self._progress_info.setVisible(False)

        self._backend_select = QComboBox(self)
        self._backend_select.addItems([b.value[0] for b in ServerBackend.supported()])
        self._backend_select.currentIndexChanged.connect(self._change_backend)

        self._launch_button = QPushButton("Launch", self)
        self._launch_button.setMinimumWidth(150)
        self._launch_button.setMinimumHeight(35)
        self._launch_button.clicked.connect(self._launch)

        self._manage_button = QToolButton(self)
        self._manage_button.setText(_("Manage"))
        self._manage_button.setPopupMode(QToolButton.InstantPopup)
        self._manage_button.setMinimumWidth(150)

        menu = QMenu(self)
        verify_action = menu.addAction(_("Verify"))
        ensure(verify_action).triggered.connect(self.verify_models)
        reinstall_action = menu.addAction(_("Re-install"))
        ensure(reinstall_action).triggered.connect(self.reinstall)
        delete_action = menu.addAction(_("Delete"))
        ensure(delete_action).triggered.connect(self.uninstall)
        self._manage_button.setMenu(menu)

        anchor = _("View log files")
        open_log_button = QLabel(f"<a href='file://{util.log_dir}'>{anchor}</a>", self)
        open_log_button.setToolTip(str(util.log_dir))
        open_log_button.linkActivated.connect(self._open_logs)

        status_layout = QVBoxLayout()
        status_layout.addWidget(self._status_label)
        status_layout.addWidget(self._backend_select, 0, Qt.AlignmentFlag.AlignLeft)
        status_layout.addWidget(self._progress_bar)
        status_layout.addWidget(self._progress_info)
        status_layout.addStretch()

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self._launch_button)
        buttons_layout.addWidget(self._manage_button)
        buttons_layout.addWidget(open_log_button, 0, Qt.AlignmentFlag.AlignRight)

        launch_layout = QHBoxLayout()
        launch_layout.addLayout(status_layout, 1)
        launch_layout.addLayout(buttons_layout, 0)
        layout.addLayout(launch_layout)

        self._custom_tab = CustomPackageTab(srv, self)
        self._custom_tab.selected_models_changed.connect(self._update_selections)

        self._workloads_tab = WorkloadsTab(srv, self)
        self._workloads_tab.selected_models_changed.connect(self._update_workload_models)

        tabs = QTabWidget(self)
        for tab in [self._workloads_tab, self._custom_tab]:
            scroll = QScrollArea(tabs)
            scroll.setWidget(tab)
            scroll.setWidgetResizable(True)
            scroll.setFrameStyle(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            tabs.addTab(scroll, tab.title)

        layout.addWidget(tabs, 1)

        root.connection.state_changed.connect(self.update_ui)
        self.update_ui()

    def _change_location(self):
        if settings.server_path != self._location_edit.text():
            self._server.path = Path(self._location_edit.text())
            self._server.check_install()
            settings.server_path = self._location_edit.text()
            settings.save()
            self.update_ui()
            self._custom_tab.update_installed()
            self._workloads_tab.update_installed()
            self.state_changed.emit()

    def _select_location(self):
        path = self._server.path
        if not path.exists():
            path = path.parent
        if not path.exists():
            path = Path(Settings._server_path.default)
            path.mkdir(parents=True, exist_ok=True)
        path = QFileDialog.getExistingDirectory(
            self, _("Select Directory"), str(path), QFileDialog.ShowDirsOnly
        )
        if path:
            path = Path(path)
            if path != Path(Settings._server_path.default) and not (path / "ComfyUI").exists():
                path = path / "ComfyUI"
            self._location_edit.setText(str(path))

    def _change_backend(self):
        backends = ServerBackend.supported()
        try:
            backend = backends[self._backend_select.currentIndex()]
        except Exception:
            backend = backends[0]
        if settings.server_backend != backend:
            self._server.backend = backend
            settings.server_backend = backend
            settings.save()
            self._server.check_install()
            self._custom_tab.update_backend()
            self.update_ui()
            self.state_changed.emit()

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_dir)))

    def _check_cuda_support(self):
        if self._server.backend is ServerBackend.cuda and not get_cuda_devices():
            question = _(
                "The CUDA backend requires a NVIDIA GPU, but no compatible devices were found. Do you want to continue with installation anyway?"
            )
            answer = QMessageBox.warning(
                self,
                _("No CUDA Devices Found"),
                question,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.StandardButton.No,
            )
            return answer == QMessageBox.StandardButton.Yes
        return True

    def _launch(self):
        self._error = ""
        if self.requires_install:
            if not self._check_cuda_support():
                return
            eventloop.run(self._install())
        elif self._server.state is ServerState.update_required:
            eventloop.run(self._upgrade())
        elif self._server.state is ServerState.stopped:
            eventloop.run(self._start())
        elif self._server.state is ServerState.running:
            eventloop.run(self._stop())

    async def _start(self):
        self._launch_button.setEnabled(False)
        self._status_label.setText(_("Starting server..."))
        self._status_label.setStyleSheet("color:orange;font-weight:bold")
        try:
            url = await self._server.start()
            self.update_ui()
            self.state_changed.emit()
            self._status_label.setText(_("Server running - Connecting..."))
            self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            await root.connection._connect(url, ServerMode.managed)
        except Exception as e:
            self.show_error(e)
        self.update_ui()
        self.state_changed.emit()

    async def _stop(self):
        self._launch_button.setEnabled(False)
        self._status_label.setText(_("Stopping server..."))
        self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
        try:
            if root.connection.state is ConnectionState.connected:
                await root.connection.disconnect()
            await self._server.stop()
        except Exception as e:
            self.show_error(e)
        self.update_ui()
        self.state_changed.emit()

    async def _install(self):
        try:
            await self._prepare_for_install()

            if self._server.state is ServerState.update_required:
                await self._server.upgrade(self._handle_progress)

            if self._server.state in [ServerState.not_installed, ServerState.missing_resources]:
                await self._server.install(self._handle_progress)
            self._custom_tab.update_installed()

            if len(self.selected_models) > 0:
                await self._server.download(self.selected_models, self._handle_progress)

            self.selected_models = []
            self._custom_tab.update_installed()
            self._workloads_tab.update_installed()
            self.update_ui()

            await self._start()

        except Exception as e:
            self.show_error(e)
        self.update_ui()

    async def _upgrade(self):
        try:
            assert self._server.state in [ServerState.update_required, ServerState.running]

            await self._prepare_for_install()
            await self._server.upgrade(self._handle_progress)
            self.update_ui()
            await self._start()

        except Exception as e:
            self.show_error(e)
        self.update_ui()

    async def _prepare_for_install(self):
        if self._server.state is ServerState.running:
            await self._stop()

        self._launch_button.setEnabled(False)
        self._manage_button.setEnabled(False)
        self._status_label.setStyleSheet(f"color:{highlight};font-weight:bold")
        self._backend_select.setVisible(False)
        self._progress_bar.setVisible(True)
        self._progress_info.setVisible(True)
        self._progress_info.setText("")

    def _handle_progress(self, report: server.InstallationProgress):
        self._status_label.setText(f"{report.stage}...")
        set_text_clipped(self._progress_info, report.message)
        if isinstance(report.progress, DownloadProgress) and report.progress.total > 0:
            self._progress_bar.setMaximum(100)
            self._progress_bar.setValue(int(report.progress.value * 100))
            self._progress_bar.setFormat(
                f"{report.progress.received:.0f} MB of {report.progress.total:.0f} MB -"
                f" {report.progress.speed:.1f} MB/s"
            )
            self._progress_bar.setTextVisible(True)
        elif isinstance(report.progress, DownloadProgress):  # download, but unknown total size
            self._progress_bar.setMaximum(0)
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat(
                f"{report.progress.received:.0f} MB - {report.progress.speed:.1f} MB/s"
            )
            self._progress_bar.setTextVisible(True)
        elif isinstance(report.progress, tuple):
            self._progress_bar.setMinimum(0)
            self._progress_bar.setMaximum(report.progress[1])
            self._progress_bar.setValue(report.progress[0])
            self._progress_bar.setFormat(f"{report.progress[0]} / {report.progress[1]}")
            self._progress_bar.setTextVisible(True)
        else:
            self._progress_bar.setMaximum(0)
            self._progress_bar.setValue(0)
            self._progress_bar.setTextVisible(False)

    def verify_models(self):
        self._error = ""
        eventloop.run(self._verify_models())

    async def _verify_models(self):
        await self._prepare_for_install()
        try:
            bad_models = await self._server.verify(self._handle_progress)

            if not bad_models:
                QMessageBox.information(
                    self,
                    _("Verification Complete"),
                    _("All model files were verified successfully."),
                )
            else:
                failed_files = "\n".join([
                    f"• {status.file.path} - {status.info or status.state.name}"
                    for status in bad_models
                ])

                msg_box = QMessageBox(
                    QMessageBox.Warning,
                    _("Verification Failed"),
                    _("The following files failed verification:")
                    + f"\n\n{failed_files}\n\n"
                    + _("Would you like to delete and re-download these files?"),
                    QMessageBox.Yes | QMessageBox.No,
                    self,
                )

                if msg_box.exec_() == QMessageBox.Yes:
                    await self._server.fix_models(bad_models, self._handle_progress)
        except Exception as e:
            self.show_error(e)
        finally:
            self._progress_bar.setVisible(False)
            self._progress_info.setVisible(False)
            self.update_ui()

    def reinstall(self):
        self._error = ""
        eventloop.run(self._reinstall())

    async def _reinstall(self):
        msg_box = QMessageBox(
            QMessageBox.Question,
            _("Confirm Reinstallation"),
            _(
                "This will reinstall the server components while keeping your downloaded models. Continue?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            self,
        )

        if msg_box.exec_() != QMessageBox.Yes:
            return

        await self._prepare_for_install()
        try:
            await self._server.uninstall(self._handle_progress, delete_models=False)
            await self._server.install(self._handle_progress)
            if len(self.selected_models) > 0:
                await self._server.download(self.selected_models, self._handle_progress)
        except Exception as e:
            self.show_error(e)
        finally:
            self.update_ui()

    def uninstall(self):
        self._error = ""
        eventloop.run(self._uninstall())

    async def _uninstall(self):
        msg_box = QMessageBox(
            QMessageBox.Warning,
            _("Confirm Deletion"),
            _("WARNING: This will delete the entire server installation INCLUDING ALL MODELS!")
            + "\n\n"
            + _("This action cannot be undone.")
            + "\n\n"
            + _("Are you absolutely sure you want to continue?"),
            QMessageBox.Cancel,
            self,
        )
        msg_box.addButton(_("Delete"), QMessageBox.DestructiveRole)
        msg_box.setDefaultButton(QMessageBox.Cancel)
        if msg_box.exec_() != 0:  # Destructive role returns 0
            return

        await self._prepare_for_install()
        try:
            await self._server.uninstall(self._handle_progress, delete_models=True)
        except Exception as e:
            self.show_error(e)
        finally:
            self._progress_bar.setVisible(False)
            self._progress_info.setVisible(False)
            self.update_ui()

    def update_ui(self):
        if self._location_edit.text() != settings.server_path:
            self._location_edit.setText(settings.server_path)
        backends = ServerBackend.supported()
        try:
            index = backends.index(settings.server_backend)
        except Exception:
            index = 0
        self._backend_select.setCurrentIndex(index)
        self._progress_bar.setVisible(False)
        self._progress_info.setVisible(False)
        self._backend_select.setVisible(True)
        self._launch_button.setEnabled(True)
        self._manage_button.setEnabled(True)
        self._location_edit.setEnabled(True)

        state = self._server.state
        if state is ServerState.not_installed:
            self._status_label.setText(_("Server is not installed"))
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
            self._manage_button.setEnabled(False)
        elif state is ServerState.missing_resources:
            self._status_label.setText(_("Server is missing required components"))
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
            self._manage_button.setEnabled(False)
        elif state in [ServerState.installing, ServerState.verifying, ServerState.uninstalling]:
            self._location_edit.setEnabled(False)
            self._progress_bar.setVisible(True)
            self._progress_info.setVisible(True)
            self._backend_select.setVisible(False)
            self._launch_button.setEnabled(False)
            self._manage_button.setEnabled(False)
        elif state is ServerState.update_required:
            text = _("Upgrade required") + f": v{self._server.version} -> v{resources.version}"
            if self._server.version == "incomplete":
                text = _("Previous installation is incomplete")
            self._status_label.setText(text)
            self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            self._launch_button.setText(_("Upgrade"))
        elif state is ServerState.stopped:
            self._status_label.setText(_("Server stopped"))
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
            self._launch_button.setText(_("Launch"))
        elif state is ServerState.starting:
            self._status_label.setText(_("Starting server..."))
            self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            self._launch_button.setText(_("Launch"))
            self._launch_button.setEnabled(False)
            self._manage_button.setEnabled(False)
            self._location_edit.setEnabled(False)
        elif state is ServerState.running:
            connection_state = root.connection.state
            if connection_state is ConnectionState.disconnected:
                self._status_label.setText(_("Server running - Disconnected"))
                self._status_label.setStyleSheet(f"color:{grey};font-weight:bold")
            elif connection_state is ConnectionState.connecting:
                self._status_label.setText(_("Server running - Connecting..."))
                self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            elif connection_state is ConnectionState.connected:
                self._status_label.setText(_("Server running - Connected"))
                self._status_label.setStyleSheet(f"color:{green};font-weight:bold")
            elif connection_state is ConnectionState.error:
                text = _("Server running - Connection error")
                error = root.connection.error or "Unknown error"
                self._status_label.setText(f"<b>{text}:</b> {error}")
                self._status_label.setStyleSheet(f"color:{red}")
            self._launch_button.setText(_("Stop"))
            self._location_edit.setEnabled(False)

        if self.requires_install:
            self._launch_button.setText(_("Install"))
            if not self._server.version and not self._server.can_install:
                self._status_label.setText(
                    _(
                        "Invalid location: directory is not empty, but no previous installation was found"
                    )
                )
                self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
                self._launch_button.setEnabled(False)
            elif not self._server.has_comfy and len(self.selected_models) == 0:
                self._status_label.setText(_("Please select models for installation"))
                self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
                self._launch_button.setEnabled(False)
            else:
                self._launch_button.setEnabled(True)

        self.show_error(self._error)

    def show_error(self, error: str | Exception):
        if isinstance(error, Exception):
            self._error = str(error) or repr(error)
        else:
            self._error = error
        if self._error:
            error_text = "<b>Error:</b> " + self._error.replace("\n", "<br>")
            self._status_label.setText(error_text)
            self._status_label.setStyleSheet(f"color:{red}")

    def _update_selections(self):
        self.selected_models = self._custom_tab.selected_models
        self.update_ui()

    def _update_workload_models(self):
        self.selected_models = self._workloads_tab.selected_models
        self.update_ui()

    @property
    def requires_install(self):
        state = self._server.state
        install_required = state in [ServerState.not_installed, ServerState.missing_resources]
        install_optional = (
            state in [ServerState.stopped, ServerState.running] and len(self.selected_models) > 0
        )
        return install_required or install_optional

    @property
    def selected_models(self):
        return self._selected_models

    @selected_models.setter
    def selected_models(self, value: list[str]):
        self._selected_models = value
        if self._custom_tab.selected_models != value:
            self._custom_tab.selected_models = value
        if self._workloads_tab.selected_models != value:
            self._workloads_tab.selected_models = value
