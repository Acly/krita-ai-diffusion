from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QFileDialog,
    QPlainTextEdit,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QToolButton,
)
from krita import Krita

from .. import Server, ServerState, ServerBackend, Setting, Settings, eventloop, server, settings
from . import Connection, ConnectionState

green = "#3b3"
yellow = "#cc3"
red = "#c33"
grey = "#888"


class ServerWidget(QWidget):
    _server: Server
    _error = ""

    def __init__(self, srv: Server, parent=None):
        super().__init__(parent)
        self._server = srv

        layout = QVBoxLayout()
        self.setLayout(layout)

        _add_header(layout, Settings._server_path)

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

        self._use_cuda = QCheckBox("Use CUDA (requires NVIDIA GPU)", self)
        self._use_cuda.stateChanged.connect(self._change_backend)

        self._launch_button = QPushButton("Launch", self)
        self._launch_button.setMinimumWidth(150)
        self._launch_button.setMinimumHeight(35)
        self._launch_button.clicked.connect(self._launch)

        status_layout = QVBoxLayout()
        status_layout.addWidget(self._status_label)
        status_layout.addWidget(self._use_cuda)
        status_layout.addWidget(self._progress_bar)

        launch_layout = QHBoxLayout()
        launch_layout.addLayout(status_layout)
        launch_layout.addWidget(self._launch_button, 0, Qt.AlignRight)
        layout.addLayout(launch_layout)

        required_text = (
            "The following extensions and models are required by the plugin and can be installed"
            " automatically."
        )
        _add_header(layout, Setting("Required components", "", required_text))
        package_layout = QGridLayout()
        self._python_package = self._add_package_widget("Python", package_layout)
        self._comfy_package = self._add_package_widget("ComfyUI", package_layout)
        self._node_packages = [
            self._add_package_widget(node.name, package_layout)
            for node in server.required_custom_nodes
        ]
        self._model_packages = [
            self._add_package_widget(model.name, package_layout) for model in server.required_models
        ]
        layout.addLayout(package_layout)

        layout.addStretch()

        self.update()
        self.update_packages()

    def _add_package_widget(self, name: str, layout: QGridLayout):
        name = QLabel(name, self)
        name.setContentsMargins(5, 0, 0, 0)
        status = QLabel(self)
        layout.addWidget(name, layout.rowCount(), 0)
        layout.addWidget(status, layout.rowCount() - 1, 1)
        return status

    def _change_location(self):
        if settings.server_path != self._location_edit.text():
            self._server.path = Path(self._location_edit.text())
            self._server.check_install()
            settings.server_path = self._location_edit.text()
            settings.save()
            self.update()
            self.update_packages()

    def _select_location(self):
        path = self._server.path
        if not self._server.path.exists():
            path = Path(Settings._server_path.default)
            path.mkdir(parents=True, exist_ok=True)
        path = QFileDialog.getExistingDirectory(
            self, "Select Directory", str(path), QFileDialog.ShowDirsOnly
        )
        self._location_edit.setText(path)

    def _change_backend(self):
        backend = ServerBackend.cuda if self._use_cuda.isChecked() else ServerBackend.cpu
        if settings.server_backend != backend:
            self._server.backend = backend
            settings.server_backend = backend
            settings.save()

    def _launch(self):
        self._error = ""
        if self._server.state is ServerState.stopped:
            eventloop.run(self._start())
        elif self._server.state is ServerState.running:
            eventloop.run(self._stop())
        elif self._server.state in [
            ServerState.missing_python,
            ServerState.missing_comfy,
            ServerState.missing_resources,
        ]:
            eventloop.run(self._install())

    async def _start(self):
        self._launch_button.setEnabled(False)
        self._status_label.setText("Starting server...")
        self._status_label.setStyleSheet("color:orange;font-weight:bold")
        try:
            await self._server.start()
            self.update()
            self._status_label.setText("Server running - Connecting...")
            self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            await Connection.instance()._connect(self._server.url)
        except Exception as e:
            self._error = str(e)
        self.update()

    async def _stop(self):
        self._launch_button.setEnabled(False)
        self._status_label.setText("Stopping server...")
        self._status_label.setStyleSheet("color:orange;font-weight:bold")
        try:
            await self._server.stop()
        except Exception as e:
            self._error = str(e)
        self.update()

    async def _install(self):
        self._launch_button.setEnabled(False)
        self._status_label.setStyleSheet("font-style:italic;font-weight:bold")
        self._use_cuda.setVisible(False)
        self._progress_bar.setVisible(True)
        try:
            await self._server.install(self._handle_progress)
            self.update()
            self.update_packages()
            await self._start()
        except Exception as e:
            self._error = str(e)
        self.update()

    def _handle_progress(self, report: server.InstallationProgress):
        self._status_label.setText(f"{report.stage}...")
        if report.progress >= 0:
            self._progress_bar.setMaximum(100)
            self._progress_bar.setValue(int(report.progress * 100))
        else:
            self._progress_bar.setMaximum(0)
            self._progress_bar.setValue(0)

    def update(self):
        self._location_edit.setText(settings.server_path)
        self._use_cuda.setChecked(settings.server_backend is ServerBackend.cuda)

        self._progress_bar.setVisible(False)
        self._use_cuda.setVisible(True)
        self._launch_button.setEnabled(True)

        if self._server.state in [ServerState.missing_comfy, ServerState.missing_python]:
            self._status_label.setText("Server is not installed")
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
        elif self._server.state is ServerState.missing_resources:
            self._status_label.setText("Server is missing required components")
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")

        if self._server.state in [
            ServerState.missing_comfy,
            ServerState.missing_python,
            ServerState.missing_resources,
        ]:
            self._launch_button.setText("Install")
            self._launch_button.setEnabled(True)
        elif self._server.state is ServerState.installing:
            self._progress_bar.setVisible(True)
            self._use_cuda.setVisible(False)
            self._launch_button.setText("Cancel")
        elif self._server.state is ServerState.stopped:
            self._status_label.setText("Server stopped")
            self._status_label.setStyleSheet(f"color:{red};font-weight:bold")
            self._launch_button.setText("Launch")
        elif self._server.state is ServerState.starting:
            self._status_label.setText("Starting server...")
            self._status_label.setStyleSheet(f"color:{yellow};font-weight:bold")
            self._launch_button.setText("Launch")
            self._launch_button.setEnabled(False)
        elif self._server.state is ServerState.running:
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
                self._status_label.setText(
                    "<b>Server running - Connection error:</b> " + Connection.instance().error
                )
                self._status_label.setStyleSheet(f"color:{red}")
            self._launch_button.setText("Stop")

        if self._error:
            self._status_label.setText(f"<b>Error:</b> {self._error}")
            self._status_label.setStyleSheet(f"color:{red}")

    def update_packages(self):
        _update_package(self._python_package, self._server.state is not ServerState.missing_python)
        _update_package(
            self._comfy_package,
            self._server.state not in [ServerState.missing_python, ServerState.missing_comfy],
        )
        for package, node in zip(self._node_packages, server.required_custom_nodes):
            _update_package(package, node.name not in self._server.missing_resources)
        for package, model in zip(self._model_packages, server.required_models):
            _update_package(package, model.name not in self._server.missing_resources)


def _update_package(package_widget: QLabel, installed: bool):
    package_widget.setText("Installed" if installed else "Not installed")
    package_widget.setStyleSheet(f"color:{green}" if installed else f"color:{grey}")


def _add_header(layout: QVBoxLayout, setting: Setting):
    title_label = QLabel(setting.name)
    title_label.setStyleSheet("font-weight:bold")
    desc_label = QLabel(setting.desc)
    desc_label.setWordWrap(True)
    layout.addSpacing(6)
    layout.addWidget(title_label)
    layout.addWidget(desc_label)
