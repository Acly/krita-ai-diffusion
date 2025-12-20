from __future__ import annotations
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget
from PyQt5.QtWidgets import QCheckBox
from krita import Krita, DockWidget
import krita

from ..model import Model, Workspace
from ..server import Server, ServerState
from ..connection import ConnectionState
from ..diffusers_connection import DiffusersConnectionState
from ..settings import ServerMode, settings
from ..updates import UpdateState
from ..root import root
from ..localization import translate as _
from . import theme
from .generation import GenerationWidget
from .custom_workflow import CustomWorkflowWidget, CustomWorkflowPlaceholder
from .upscale import UpscaleWidget
from .live import LiveWidget
from .animation import AnimationWidget
from .diffusers import DiffusersWidget


class AutoUpdateWidget(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)

        update_message = QLabel(_("A new plugin version is available!"), self)
        self._update_status = QLabel(self)
        self._update_status.setStyleSheet(f"color: {theme.yellow}; font-weight: bold;")
        self._update_error = QLabel(self)

        self._update_checkbox = QCheckBox(_("Check for updates on startup"), self)
        self._update_checkbox.setChecked(settings.auto_update)
        self._update_checkbox.stateChanged.connect(self._toggle_auto_update)

        self._update_button = QPushButton(_("Download and Install"), self)
        self._update_button.setMinimumHeight(32)
        self._update_button.clicked.connect(self._run_update)

        update_layout = QVBoxLayout()
        update_layout.addWidget(update_message)
        update_layout.addWidget(self._update_status)
        update_layout.addWidget(self._update_error)
        update_layout.addWidget(self._update_checkbox)
        update_layout.addWidget(self._update_button)
        self.setLayout(update_layout)

        root.auto_update.latest_version_changed.connect(self.update_content)
        root.auto_update.error_changed.connect(self.update_content)
        settings.changed.connect(self.update_content)
        self.update_content()

    def update_content(self):
        self._update_checkbox.setChecked(settings.auto_update)

        au = root.auto_update
        match au.state:
            case UpdateState.checking:
                self._update_status.setText(_("Checking for updates..."))
                self._update_button.setEnabled(False)
            case UpdateState.available:
                self._update_status.setText(_("Latest version") + f": {au.latest_version}")
                self._update_button.setEnabled(True)
            case UpdateState.downloading:
                self._update_status.setText(_("Downloading package..."))
                self._update_button.setEnabled(False)
            case UpdateState.installing:
                self._update_status.setText(_("Installing new version..."))
                self._update_button.setEnabled(False)
            case UpdateState.failed_update:
                self._update_status.setText(_("Update failed"))
                self._update_error.setText(au.error)
                self._update_button.setEnabled(True)
            case UpdateState.restart_required:
                self._update_status.setText(_("Please restart Krita to complete the update!"))
                self._update_button.setEnabled(False)

        self.setVisible(self.is_visible)

    @property
    def is_visible(self):
        return settings.auto_update and root.auto_update.state not in [
            UpdateState.latest,
            UpdateState.failed_check,
            UpdateState.checking,
        ]

    def _toggle_auto_update(self):
        settings.auto_update = self._update_checkbox.isChecked()
        settings.save()
        root.auto_update.state_changed.emit(root.auto_update.state)

    def _run_update(self):
        root.auto_update.run()


class ConnectionWidget(QWidget):
    def __init__(self, server: Server, parent: QWidget):
        super().__init__(parent)
        self._server = server

        self._connect_status = QLabel(_("Not connected to server."), self)

        self._connect_error = QLabel(self)
        self._connect_error.setVisible(False)
        self._connect_error.setWordWrap(True)
        self._connect_error.setStyleSheet(f"color: {theme.yellow};")

        self._settings_button = QPushButton(theme.icon("settings"), _("Configure"), self)
        self._settings_button.setMinimumHeight(32)
        self._settings_button.clicked.connect(self.show_settings)

        layout = QVBoxLayout()
        layout.addWidget(self._connect_status)
        layout.addSpacing(6)
        layout.addWidget(self._connect_error)
        layout.addWidget(self._settings_button)
        self.setLayout(layout)

        root.connection.state_changed.connect(self.update_content)
        self.update_content()

    def update_content(self):
        connection = root.connection
        if connection.state in [ConnectionState.disconnected, ConnectionState.error]:
            self._connect_status.setText(_("Not connected to server."))
        if connection.state is ConnectionState.error:
            self._connect_error.setText(
                _("Connection attempt failed! Click below to configure and reconnect.")
            )
            self._connect_error.setVisible(True)
        if (
            connection.state is ConnectionState.disconnected
            and settings.server_mode is ServerMode.managed
        ):
            if self._server.state is ServerState.update_required:
                self._connect_error.setText(
                    _("Server version is outdated. Click below to upgrade.")
                )
            else:
                self._connect_error.setText(
                    _("Server is not installed or not running. Click below to start.")
                )
            self._connect_error.setVisible(True)
        if connection.state is ConnectionState.auth_missing:
            self._connect_status.setText(_("Not signed in. Click below to connect."))
        if connection.state is ConnectionState.connecting:
            self._connect_status.setText(_("Connecting to server..."))
        if connection.state is ConnectionState.discover_models:
            progress = f" ({connection.progress[0]}/{connection.progress[1]})"
            self._connect_status.setText(_("Discovering models") + progress)
        if connection.state is ConnectionState.connected:
            self._connect_status.setText(
                _(
                    "Connected to server at {url}.\n\nCreate a new document or open an existing image to start!",
                    url=connection.client.url,
                )
            )
            self._connect_error.setVisible(False)

    def show_settings(self):
        Krita.instance().action("ai_diffusion_settings").trigger()


class WelcomeWidget(QWidget):
    def __init__(self, server: Server):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        header_layout = QHBoxLayout()
        header_logo = QLabel(self)
        header_logo.setPixmap(theme.logo().scaled(64, 64))
        header_logo.setMaximumSize(64, 64)
        header_text = QLabel("AI Image\nGeneration", self)
        header_text.setStyleSheet("font-size: 12pt")
        header_layout.addWidget(header_logo)
        header_layout.addWidget(header_text)

        self._update_widget = AutoUpdateWidget(self)
        self._connection_widget = ConnectionWidget(server, self)

        # Diffusers mode section (for Diffusers backend)
        self._diffusers_mode_widget = QWidget(self)
        diffusers_layout = QVBoxLayout(self._diffusers_mode_widget)
        diffusers_layout.setContentsMargins(0, 0, 0, 0)

        diffusers_label = QLabel(
            _("Or use <b>Diffusers Mode</b> for txt2img, img2img, inpaint and Qwen layered generation."),
            self
        )
        diffusers_label.setWordWrap(True)

        self._diffusers_button = QPushButton(theme.icon("workspace-generation"), _("Switch to Diffusers Mode"), self)
        self._diffusers_button.setMinimumHeight(32)
        self._diffusers_button.clicked.connect(self._switch_to_diffusers)

        diffusers_layout.addWidget(diffusers_label)
        diffusers_layout.addWidget(self._diffusers_button)

        info = QLabel(
            "<a href='https://www.interstice.cloud'>Interstice.cloud</a> | "
            + "<a href='https://github.com/Acly/krita-ai-diffusion'>GitHub Project</a> | "
            + "<a href='https://discord.gg/pWyzHfHHhU'>Discord</a>",
            self,
        )
        info.setOpenExternalLinks(True)

        layout.addLayout(header_layout)
        layout.addSpacing(12)
        layout.addWidget(self._update_widget)
        layout.addWidget(self._connection_widget)
        layout.addSpacing(12)
        layout.addWidget(self._diffusers_mode_widget)
        layout.addSpacing(24)
        layout.addWidget(info, 0, Qt.AlignmentFlag.AlignRight)
        layout.addStretch()

        self.update_content()
        root.auto_update.state_changed.connect(self.update_content)
        root.diffusers_connection.state_changed.connect(self.update_content)

    def _switch_to_diffusers(self):
        """Switch the active model to diffusers workspace."""
        model = root.model_for_active_document()
        if model is None:
            # Create a model for the active document if needed
            doc = Krita.instance().activeDocument()
            if doc is not None:
                model = root.model_for_document(doc)
        if model is not None:
            model.workspace = Workspace.layered

    def update_content(self):
        self._update_widget.update_content()
        self._connection_widget.update_content()
        self._connection_widget.setVisible(not self._update_widget.is_visible)
        # Show diffusers mode option when diffusers is enabled
        self._diffusers_mode_widget.setVisible(
            settings.diffusers_enabled and not self._update_widget.is_visible
        )

    @property
    def requires_update(self):
        return self._update_widget.is_visible


class ImageDiffusionWidget(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("AI Image Generation"))
        self._welcome = WelcomeWidget(root.server)
        self._generation = GenerationWidget()
        self._upscaling = UpscaleWidget()
        self._animation = AnimationWidget()
        self._live = LiveWidget()
        self._custom = CustomWorkflowWidget()
        self._custom_placeholder = CustomWorkflowPlaceholder()
        self._diffusers = DiffusersWidget()
        self._frame = QStackedWidget(self)
        self._frame.addWidget(self._welcome)
        self._frame.addWidget(self._generation)
        self._frame.addWidget(self._upscaling)
        self._frame.addWidget(self._live)
        self._frame.addWidget(self._animation)
        self._frame.addWidget(self._custom)
        self._frame.addWidget(self._custom_placeholder)
        self._frame.addWidget(self._diffusers)
        self.setWidget(self._frame)

        root.connection.state_changed.connect(self.update_content)
        root.diffusers_connection.state_changed.connect(self.update_content)
        root.auto_update.state_changed.connect(self.update_content)
        root.model_created.connect(self.register_model)

    def canvasChanged(self, canvas: krita.Canvas):
        if canvas is not None and canvas.view() is not None:
            self.update_content()

    def register_model(self, model: Model):
        model.workspace_changed.connect(self.update_content)
        self.update_content()

    def update_content(self):
        model = root.model_for_active_document()
        connection = root.connection
        diffusers_connection = root.diffusers_connection
        requires_update = self._welcome.requires_update
        is_cloud = settings.server_mode is ServerMode.cloud
        comfy_connected = connection.state is ConnectionState.connected
        diffusers_connected = diffusers_connection.state is DiffusersConnectionState.connected

        # Diffusers workspace can work with just the diffusers connection
        if model is not None and model.workspace is Workspace.layered:
            self._diffusers.model = model
            self._frame.setCurrentWidget(self._diffusers)
        elif model is None or not comfy_connected or requires_update:
            self._frame.setCurrentWidget(self._welcome)
        elif model.workspace is Workspace.generation:
            self._generation.model = model
            self._frame.setCurrentWidget(self._generation)
        elif model.workspace is Workspace.upscaling:
            self._upscaling.model = model
            self._frame.setCurrentWidget(self._upscaling)
        elif model.workspace is Workspace.live:
            self._live.model = model
            self._frame.setCurrentWidget(self._live)
        elif model.workspace is Workspace.animation:
            self._animation.model = model
            self._frame.setCurrentWidget(self._animation)
        elif model.workspace is Workspace.custom and is_cloud:
            self._custom_placeholder.model = model
            self._frame.setCurrentWidget(self._custom_placeholder)
        elif model.workspace is Workspace.custom:
            self._custom.model = model
            self._frame.setCurrentWidget(self._custom)
