from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QWidget,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QFileDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QToolButton,
    QScrollArea,
)
from krita import Krita

from ..settings import Settings, ServerMode, settings
from ..style import Arch
from ..resources import ModelResource, CustomNode
from ..server import Server, ServerBackend, ServerState
from ..connection import ConnectionState
from ..root import root
from ..localization import translate as _
from .. import eventloop, resources, server, util
from .theme import SignalBlocker, add_header, set_text_clipped, green, grey, red, yellow, highlight


class PackageState(Enum):
    installed = 0
    selected = 1
    available = 2
    disabled = 3


class PackageItem:
    label: QLabel
    status: QLabel | QCheckBox
    package: str | ModelResource | CustomNode
    state: PackageState


class PackageGroupWidget(QWidget):
    _layout: QGridLayout
    _items: list[PackageItem]
    _status: QLabel
    _desc: Optional[QLabel] = None
    _workload = Arch.all
    _is_checkable = False

    changed = pyqtSignal()

    def __init__(
        self,
        name: str,
        packages: list[str] | list[ModelResource] | list[CustomNode],
        description: Optional[str] = None,
        is_expanded=True,
        is_checkable=False,
        initial=PackageState.available,
        parent=None,
    ):
        super().__init__(parent)

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

        if description:
            desc = self._desc = QLabel(self)
            desc.setText(description)
            desc.setContentsMargins(20, 0, 0, 0)
            desc.setWordWrap(True)
            desc.setTextFormat(Qt.TextFormat.RichText)
            desc.setOpenExternalLinks(True)
            self._layout.addWidget(desc, 1, 0, 1, 2)

        self._is_checkable = is_checkable
        self._items = [self.add_item(p, initial) for p in packages]
        self._update_visibility()

    def _update_visibility(self):
        self._header.setArrowType(
            Qt.ArrowType.DownArrow if self._header.isChecked() else Qt.ArrowType.RightArrow
        )
        if self._desc:
            self._desc.setVisible(self._header.isChecked())
        for item in self._items:
            item.label.setVisible(self._header.isChecked())
            item.status.setVisible(self._header.isChecked())

    def add_item(self, package: str | ModelResource | CustomNode, initial: PackageState):
        item = PackageItem()
        item.package = package
        item.state = initial
        item.label = QLabel(self._package_name(package), self)
        item.label.setContentsMargins(20, 0, 0, 0)
        if self.is_checkable:
            item.status = QCheckBox(_("Install"), self)
            item.status.setChecked(initial in [PackageState.selected, PackageState.installed])
            item.status.toggled.connect(self._handle_checkbox_toggle)
        else:
            item.status = QLabel(self)
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
            if item.state is PackageState.installed:
                item.status.setText(_("Installed"))
                item.status.setStyleSheet(f"color:{green}")
            elif item.state is PackageState.available:
                item.status.setText(_("Not installed"))
                item.status.setStyleSheet("")
            if self.is_checkable:
                self._update_workload(item)
                if item.state is PackageState.selected:
                    item.status.setText(_("Not installed"))
                    item.status.setStyleSheet("")
                elif item.state is PackageState.disabled:
                    item.status.setText(_("Workload not selected"))
                    item.status.setStyleSheet(f"color:{grey}")
                with SignalBlocker(item.status):
                    item.status.setChecked(
                        item.state in [PackageState.selected, PackageState.installed]
                    )
                    item.status.setEnabled(item.state is not PackageState.disabled)
        self._update_status()

    def _update_workload(self, item: PackageItem):
        enabled = (
            not isinstance(item.package, ModelResource)
            or Arch.match(self._workload, item.package.arch)
            or item.package.arch not in [Arch.sd15, Arch.sdxl]
        )
        if not enabled and item.state in [PackageState.selected, PackageState.available]:
            item.state = PackageState.disabled
        elif enabled and item.state is PackageState.disabled:
            item.state = PackageState.available

    @property
    def package_names(self):
        return [self._package_name(item.package) for item in self._items]

    @property
    def selected_packages(self):
        return [
            self._package_name(item.package)
            for item in self._items
            if item.state is PackageState.selected
        ]

    @property
    def workload(self):
        return self._workload

    @workload.setter
    def workload(self, workload: Arch):
        self._workload = workload
        self._update()

    def set_installed(self, installed: list[bool]):
        for item, is_installed in zip(self._items, installed):
            if is_installed:
                item.state = PackageState.installed
            elif item.state is PackageState.installed:
                item.state = PackageState.available
        self._update()

    def _update_status(self):
        available = sum(item.state is PackageState.available for item in self._items)
        if all(item.state is PackageState.installed for item in self._items):
            self._status.setText(_("All installed"))
            self._status.setStyleSheet(f"color:{green}")
        elif self.is_checkable:
            selected = sum(item.state is PackageState.selected for item in self._items)
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
        self._update_status()
        self.changed.emit()

    def _package_name(self, package: str | ModelResource | CustomNode):
        return package if isinstance(package, str) else package.name


class ServerWidget(QWidget):
    _server: Server
    _error = ""
    _packages: dict[str, PackageGroupWidget]

    def __init__(self, srv: Server, parent=None):
        super().__init__(parent)
        self._server = srv

        layout = QVBoxLayout()
        self.setLayout(layout)

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
        buttons_layout.addWidget(open_log_button, 0, Qt.AlignmentFlag.AlignRight)

        launch_layout = QHBoxLayout()
        launch_layout.addLayout(status_layout, 1)
        launch_layout.addLayout(buttons_layout, 0)
        layout.addLayout(launch_layout)

        package_list = QWidget(self)
        package_layout = QVBoxLayout()
        package_layout.setContentsMargins(0, 0, 0, 0)
        package_list.setLayout(package_layout)

        scroll = QScrollArea(self)
        scroll.setWidget(package_list)
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)

        self._required_group = PackageGroupWidget(
            _("Core components"),
            ["Python", "ComfyUI", _("Custom nodes"), _("Required models")],
            is_expanded=False,
            parent=self,
        )
        package_layout.addWidget(self._required_group)

        self._workload_group = PackageGroupWidget(
            _("Workloads"),
            [_("Stable Diffusion 1.5"), _("Stable Diffusion XL")],
            description=(
                _("Choose a Diffusion base model to install its basic requirements.")
                + " <a href='https://docs.interstice.cloud/base-models'>"
                + _("Read more about workloads.")
                + "</a>"
            ),
            is_checkable=True,
            parent=self,
        )
        self._workload_group.values = [PackageState.available, PackageState.selected]
        self._workload_group.changed.connect(self.update_ui)
        package_layout.addWidget(self._workload_group)

        optional_models = resources.default_checkpoints + resources.optional_models
        self._packages = {
            "upscalers": PackageGroupWidget(
                _("Upscalers (super-resolution)"),
                resources.upscale_models,
                is_checkable=True,
                parent=self,
            ),
            "sd15": PackageGroupWidget(
                _("Stable Diffusion 1.5 models"),
                [m for m in optional_models if m.arch is Arch.sd15],
                description=_("Select at least one diffusion model. Control models are optional."),
                is_checkable=True,
                is_expanded=False,
                parent=self,
            ),
            "sdxl": PackageGroupWidget(
                _("Stable Diffusion XL models"),
                [m for m in optional_models if m.arch is Arch.sdxl],
                description=_("Select at least one diffusion model. Control models are optional."),
                is_checkable=True,
                parent=self,
            ),
            "illu": PackageGroupWidget(
                _("Illustrious/NoobAI XL models"),
                [m for m in optional_models if m.arch in [Arch.illu, Arch.illu_v]],
                description=_("Select at least one diffusion model. Control models are optional."),
                is_checkable=True,
                is_expanded=False,
                parent=self,
            ),
        }
        # Pre-select a recommended set of models if the server hasn't been installed yet
        if not self._server.has_comfy and self.selected_workload in [Arch.all, Arch.sdxl]:
            sdxl_packages = self._packages["sdxl"]
            sdxl_packages.workload = Arch.sdxl
            state = [PackageState.selected for _ in sdxl_packages.values]
            state[-2] = PackageState.available  # Stencil is optional
            state[-1] = PackageState.available  # Face model is optional
            sdxl_packages.values = state

        for group in ["upscalers", "sd15", "sdxl", "illu"]:
            self._packages[group].changed.connect(self.update_ui)
            package_layout.addWidget(self._packages[group])

        package_layout.addStretch()

        root.connection.state_changed.connect(self.update_ui)
        self.update_ui()
        self.update_required()

    def _change_location(self):
        if settings.server_path != self._location_edit.text():
            self._server.path = Path(self._location_edit.text())
            self._server.check_install()
            settings.server_path = self._location_edit.text()
            settings.save()
            self.update_ui()
            self.update_required()

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

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_dir)))

    def _launch(self):
        self._error = ""
        if self.requires_install:
            eventloop.run(self._install())
        elif self._server.upgrade_required:
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
            self._status_label.setText(_("Server running - Connecting..."))
            self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            await root.connection._connect(url, ServerMode.managed)
        except Exception as e:
            self.show_error(str(e))
        self.update_ui()

    async def _stop(self):
        self._launch_button.setEnabled(False)
        self._status_label.setText(_("Stopping server..."))
        self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
        try:
            if root.connection.state is ConnectionState.connected:
                await root.connection.disconnect()
            await self._server.stop()
        except Exception as e:
            self.show_error(str(e))
        self.update_ui()

    async def _install(self):
        try:
            await self._prepare_for_install()

            if self._server.upgrade_required:
                await self._server.upgrade(self._handle_progress)

            if self._server.state in [ServerState.not_installed, ServerState.missing_resources]:
                await self._server.install(self._handle_progress)
                await self._server.download_required(self._handle_progress)
            self.update_required()

            models_to_install = self.update_optional()
            if len(models_to_install) > 0:
                await self._server.download(models_to_install, self._handle_progress)
            self.update_ui()

            await self._start()

        except Exception as e:
            self.show_error(str(e))
        self.update_ui()

    async def _upgrade(self):
        try:
            assert self._server.state in [ServerState.stopped, ServerState.running]
            assert self._server.upgrade_required

            await self._prepare_for_install()
            await self._server.upgrade(self._handle_progress)
            self.update_ui()
            await self._start()

        except Exception as e:
            self.show_error(str(e))
        self.update_ui()

    async def _prepare_for_install(self):
        if self._server.state is ServerState.running:
            await self._stop()

        self._launch_button.setEnabled(False)
        self._status_label.setStyleSheet(f"color:{highlight};font-weight:bold")
        self._backend_select.setVisible(False)
        self._progress_bar.setVisible(True)
        self._progress_info.setVisible(True)

    def _handle_progress(self, report: server.InstallationProgress):
        self._status_label.setText(f"{report.stage}...")
        set_text_clipped(self._progress_info, report.message)
        if report.progress and report.progress.total > 0:
            self._progress_bar.setMaximum(100)
            self._progress_bar.setValue(int(report.progress.value * 100))
            self._progress_bar.setFormat(
                f"{report.progress.received:.0f} MB of {report.progress.total:.0f} MB -"
                f" {report.progress.speed:.1f} MB/s"
            )
            self._progress_bar.setTextVisible(True)
        elif report.progress:  # download, but unknown total size
            self._progress_bar.setMaximum(0)
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat(
                f"{report.progress.received:.0f} MB - {report.progress.speed:.1f} MB/s"
            )
            self._progress_bar.setTextVisible(True)
        else:
            self._progress_bar.setMaximum(0)
            self._progress_bar.setValue(0)
            self._progress_bar.setTextVisible(False)

    def update_ui(self):
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
        self._location_edit.setEnabled(True)

        state = self._server.state
        if state is ServerState.not_installed:
            self._status_label.setText(_("Server is not installed"))
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
        elif state is ServerState.missing_resources:
            self._status_label.setText(_("Server is missing required components"))
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
        elif state is ServerState.installing:
            self._location_edit.setEnabled(False)
            self._progress_bar.setVisible(True)
            self._progress_info.setVisible(True)
            self._backend_select.setVisible(False)
            self._launch_button.setEnabled(False)
        elif self._server.upgrade_required:
            self._status_label.setText(
                _("Upgrade required") + f": v{self._server.version} -> v{resources.version}"
            )
            self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            self._launch_button.setText(_("Upgrade"))
            self._launch_button.setEnabled(True)
        elif state is ServerState.stopped:
            self._status_label.setText(_("Server stopped"))
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
            self._launch_button.setText(_("Launch"))
        elif state is ServerState.starting:
            self._status_label.setText(_("Starting server..."))
            self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            self._launch_button.setText(_("Launch"))
            self._launch_button.setEnabled(False)
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
            else:
                self._launch_button.setEnabled(True)

        self.show_error(self._error)

    def show_error(self, error: str):
        self._error = error
        if self._error:
            self._status_label.setText(f"<b>Error:</b> {self._error}")
            self._status_label.setStyleSheet(f"color:{red}")

    def update_required(self):
        has_missing_nodes = any(
            node.name in self._server.missing_resources for node in resources.required_custom_nodes
        )
        has_missing_models = any(
            model.name in self._server.missing_resources
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

    def update_optional(self):
        workloads = [
            [m for m in resources.required_models if m.arch is Arch.sd15],
            [m for m in resources.required_models if m.arch is Arch.sdxl],
        ]
        self._workload_group.set_installed([self._server.all_installed(w) for w in workloads])
        to_install = [
            m.name
            for workload, state in zip(workloads, self._workload_group.values)
            if state is PackageState.selected
            for m in workload
        ]

        for widget in self._packages.values():
            widget.workload = self.selected_workload
            widget.set_installed([self._server.is_installed(p) for p in widget.package_names])

        to_install += [p for widget in self._packages.values() for p in widget.selected_packages]
        return to_install

    @property
    def requires_install(self):
        state = self._server.state
        checkpoints_to_install = self.update_optional()
        install_required = state in [ServerState.not_installed, ServerState.missing_resources]
        install_optional = (
            state in [ServerState.stopped, ServerState.running] and len(checkpoints_to_install) > 0
        )
        return install_required or install_optional

    @property
    def selected_workload(self):
        selected_or_installed = [
            state in [PackageState.selected, PackageState.installed]
            for state in self._workload_group.values
        ]
        if all(selected_or_installed):
            return Arch.all
        if selected_or_installed[0]:
            return Arch.sd15
        if selected_or_installed[1]:
            return Arch.sdxl
        return Arch.auto
