from pathlib import Path
from typing import List, Optional
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

from .. import (
    Server,
    ServerState,
    ServerBackend,
    Settings,
    eventloop,
    resources,
    server,
    settings,
    util,
)
from . import Connection, ConnectionState
from .theme import add_header, set_text_clipped, green, grey, red, yellow, highlight


class PackageGroupWidget(QWidget):
    _layout: QGridLayout
    _widgets: list
    _status: QLabel
    _desc: Optional[QLabel] = None

    changed = pyqtSignal()

    def __init__(
        self,
        name: str,
        packages: List[str],
        description: Optional[str] = None,
        is_expanded=True,
        is_optional=False,
        is_checked=False,
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
            self._desc = QLabel(self)
            self._desc.setText(description)
            self._desc.setContentsMargins(20, 0, 0, 0)
            self._desc.setWordWrap(True)
            self._layout.addWidget(self._desc, 1, 0, 1, 2)

        self._widgets = [self.add_widget(p, is_optional, is_checked) for p in packages]
        self._update_visibility()

    def _update_visibility(self):
        self._header.setArrowType(
            Qt.ArrowType.DownArrow if self._header.isChecked() else Qt.ArrowType.RightArrow
        )
        if self._desc:
            self._desc.setVisible(self._header.isChecked())
        for widget in self._widgets:
            widget[0].setVisible(self._header.isChecked())
            widget[1].setVisible(self._header.isChecked())

    def add_widget(self, name: str, is_optional=False, is_checked=False):
        key = QLabel(name, self)
        key.setContentsMargins(20, 0, 0, 0)
        if is_optional:
            value = QCheckBox("Install", self)
            value.setChecked(is_checked)
            value.toggled.connect(self._handle_checkbox_toggle)
        else:
            value = QLabel(self)
        self._layout.addWidget(key, self._layout.rowCount(), 0)
        self._layout.addWidget(value, self._layout.rowCount() - 1, 1)
        return key, value

    @property
    def is_checkable(self):
        return isinstance(self._widgets[0][1], QCheckBox)

    @property
    def values(self):
        if self.is_checkable:
            return [not widget[1].isEnabled() for widget in self._widgets]
        else:
            return [widget[1].text() == "Installed" for widget in self._widgets]

    @values.setter
    def values(self, values: List[bool]):
        for widget, value in zip(self._widgets, values):
            if self.is_checkable:
                widget[1].setText("Installed" if value else "Install")
                widget[1].setStyleSheet(f"color:{green}" if value else "")
                widget[1].setChecked(widget[1].isChecked() or value)
                widget[1].setEnabled(not value)
            else:
                widget[1].setText("Installed" if value else "Not installed")
                widget[1].setStyleSheet(f"color:{green}" if value else f"color:{grey}")
        self._update_status(values)

    @property
    def is_checked(self):
        return [widget[1].isEnabled() and widget[1].isChecked() for widget in self._widgets]

    def _update_status(self, installed: List[bool]):
        available = len(installed) - sum(installed)
        if available == 0:
            self._status.setText("All installed")
            self._status.setStyleSheet(f"color:{green}")
        elif self.is_checkable:
            selected = sum(self.is_checked)
            if selected > 0:
                self._status.setText(f"{selected} of {available} packages selected")
                self._status.setStyleSheet(f"color:{yellow}")
            else:
                self._status.setText(f"{available} packages available")
                self._status.setStyleSheet(f"color:{grey}")
        else:
            self._status.setText(f"{available} packages require installation")
            self._status.setStyleSheet(f"color:{yellow}")

    def _handle_checkbox_toggle(self):
        self._update_status(self.values)
        self.changed.emit()


class ServerWidget(QWidget):
    _server: Server
    _error = ""

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

        backend_supported = lambda b: b is not ServerBackend.directml or util.is_windows
        backends = [b.value for b in ServerBackend if backend_supported(b)]
        self._backend_select = QComboBox(self)
        self._backend_select.addItems(backends)
        self._backend_select.currentIndexChanged.connect(self._change_backend)

        self._launch_button = QPushButton("Launch", self)
        self._launch_button.setMinimumWidth(150)
        self._launch_button.setMinimumHeight(35)
        self._launch_button.clicked.connect(self._launch)

        open_log_button = QLabel(f"<a href='file://{util.log_path}'>View log files</a>", self)
        open_log_button.setToolTip(str(util.log_path))
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
            "Core components", ["Python", "ComfyUI"], parent=self
        )
        package_layout.addWidget(self._required_group)

        node_packages = [node.name for node in resources.required_custom_nodes]
        self._nodes_group = PackageGroupWidget(
            "Required custom nodes", node_packages, is_expanded=False, parent=self
        )
        package_layout.addWidget(self._nodes_group)

        model_packages = [model.name for model in resources.required_models]
        self._models_group = PackageGroupWidget(
            "Required models", model_packages, is_expanded=False, parent=self
        )
        package_layout.addWidget(self._models_group)

        self._checkpoint_group = PackageGroupWidget(
            "Recommended checkpoints",
            [checkpoint.name for checkpoint in resources.default_checkpoints],
            description=(
                "At least one Stable Diffusion checkpoint is required. Below are some popular"
                " choices, more can be found online."
            ),
            is_optional=True,
            is_checked=not self._server.has_comfy,
            parent=self,
        )
        self._checkpoint_group.changed.connect(self.update)
        package_layout.addWidget(self._checkpoint_group)

        self._upscaler_group = PackageGroupWidget(
            "Upscalers (super-resolution)",
            [model.name for model in resources.upscale_models],
            is_optional=True,
            parent=self,
        )
        self._upscaler_group.changed.connect(self.update)
        package_layout.addWidget(self._upscaler_group)

        self._control_group = PackageGroupWidget(
            "Control extensions",
            [control.name for control in resources.optional_models],
            is_optional=True,
            parent=self,
        )
        self._control_group.changed.connect(self.update)
        package_layout.addWidget(self._control_group)

        package_layout.addStretch()

        self.update()
        self.update_required()

    def _change_location(self):
        if settings.server_path != self._location_edit.text():
            self._server.path = Path(self._location_edit.text())
            self._server.check_install()
            settings.server_path = self._location_edit.text()
            settings.save()
            self.update()
            self.update_required()

    def _select_location(self):
        path = self._server.path
        if not self._server.path.exists():
            path = Path(Settings._server_path.default)
            path.mkdir(parents=True, exist_ok=True)
        path = QFileDialog.getExistingDirectory(
            self, "Select Directory", str(path), QFileDialog.ShowDirsOnly
        )
        if path:
            self._location_edit.setText(path)

    def _change_backend(self):
        backend = list(ServerBackend)[self._backend_select.currentIndex()]
        if settings.server_backend != backend:
            self._server.backend = backend
            settings.server_backend = backend
            settings.save()

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_path)))

    def _launch(self):
        self._error = ""
        if self.requires_install:
            eventloop.run(self._install())
        elif self._server.state is ServerState.stopped:
            eventloop.run(self._start())
        elif self._server.state is ServerState.running:
            eventloop.run(self._stop())

    async def _start(self):
        self._launch_button.setEnabled(False)
        self._status_label.setText("Starting server...")
        self._status_label.setStyleSheet("color:orange;font-weight:bold")
        try:
            url = await self._server.start()
            self.update()
            self._status_label.setText("Server running - Connecting...")
            self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            await Connection.instance()._connect(url)
        except Exception as e:
            self._error = str(e)
        self.update()

    async def _stop(self):
        self._launch_button.setEnabled(False)
        self._status_label.setText("Stopping server...")
        self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
        try:
            if Connection.instance().state is ConnectionState.connected:
                await Connection.instance().disconnect()
            await self._server.stop()
        except Exception as e:
            self._error = str(e)
        self.update()

    async def _install(self):
        try:
            if self._server.state is ServerState.running:
                await self._stop()

            self._launch_button.setEnabled(False)
            self._status_label.setStyleSheet(f"color:{highlight};font-weight:bold")
            self._backend_select.setVisible(False)
            self._progress_bar.setVisible(True)
            self._progress_info.setVisible(True)

            if self._server.state in [ServerState.not_installed, ServerState.missing_resources]:
                await self._server.install(self._handle_progress)
            self.update_required()

            checkpoints_to_install = self.update_optional()
            if len(checkpoints_to_install) > 0:
                await self._server.install_optional(checkpoints_to_install, self._handle_progress)
            self.update()

            await self._start()

        except Exception as e:
            self._error = str(e)
        self.update()

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

    def update(self):
        self._location_edit.setText(settings.server_path)
        self._backend_select.setCurrentIndex(list(ServerBackend).index(settings.server_backend))
        self._progress_bar.setVisible(False)
        self._progress_info.setVisible(False)
        self._backend_select.setVisible(True)
        self._launch_button.setEnabled(True)

        state = self._server.state
        if state is ServerState.not_installed:
            self._status_label.setText("Server is not installed")
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
        elif state is ServerState.missing_resources:
            self._status_label.setText("Server is missing required components")
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
        elif state is ServerState.installing:
            self._progress_bar.setVisible(True)
            self._progress_info.setVisible(True)
            self._backend_select.setVisible(False)
            self._launch_button.setEnabled(False)
        elif state is ServerState.stopped:
            self._status_label.setText("Server stopped")
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
            self._launch_button.setText("Launch")
        elif state is ServerState.starting:
            self._status_label.setText("Starting server...")
            self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            self._launch_button.setText("Launch")
            self._launch_button.setEnabled(False)
        elif state is ServerState.running:
            connection_state = Connection.instance().state
            if connection_state is ConnectionState.disconnected:
                self._status_label.setText("Server running - Disconnected")
                self._status_label.setStyleSheet(f"color:{grey};font-weight:bold")
            elif connection_state is ConnectionState.connecting:
                self._status_label.setText("Server running - Connecting...")
                self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            elif connection_state is ConnectionState.connected:
                self._status_label.setText("Server running - Connected")
                self._status_label.setStyleSheet(f"color:{green};font-weight:bold")
            elif connection_state is ConnectionState.error:
                error = Connection.instance().error or "Unknown error"
                self._status_label.setText(f"<b>Server running - Connection error:</b> {error}")
                self._status_label.setStyleSheet(f"color:{red}")
            self._launch_button.setText("Stop")

        if self.requires_install:
            self._launch_button.setText("Install")
            self._launch_button.setEnabled(True)

        if self._error:
            self._status_label.setText(f"<b>Error:</b> {self._error}")
            self._status_label.setStyleSheet(f"color:{red}")

    def update_required(self):
        self._required_group.values = [self._server.has_python, self._server.has_comfy]
        self._nodes_group.values = [
            node.name not in self._server.missing_resources
            for node in resources.required_custom_nodes
        ]
        self._models_group.values = [
            model.name not in self._server.missing_resources for model in resources.required_models
        ]

    def update_optional(self):
        self._checkpoint_group.values = [
            r.name not in self._server.missing_resources for r in resources.default_checkpoints
        ]
        self._upscaler_group.values = [
            r.name not in self._server.missing_resources for r in resources.upscale_models
        ]
        self._control_group.values = [
            r.name not in self._server.missing_resources for r in resources.optional_models
        ]

        def checked_packages(group, pkgs):
            return [pkg.name for pkg, checked in zip(pkgs, group.is_checked) if checked]

        to_install = checked_packages(self._checkpoint_group, resources.default_checkpoints)
        to_install += checked_packages(self._upscaler_group, resources.upscale_models)
        to_install += checked_packages(self._control_group, resources.optional_models)
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
