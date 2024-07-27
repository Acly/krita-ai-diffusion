from __future__ import annotations
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget
from krita import Krita, DockWidget
import krita

from ..model import Model, Workspace
from ..server import Server
from ..connection import ConnectionState
from ..settings import ServerMode, settings
from ..root import root
from ..localization import translate as _
from . import theme
from .generation import GenerationWidget
from .upscale import UpscaleWidget
from .live import LiveWidget
from .animation import AnimationWidget


class WelcomeWidget(QWidget):
    _server: Server

    def __init__(self, server: Server):
        super().__init__()
        self._server = server

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
        layout.addLayout(header_layout)
        layout.addSpacing(12)

        self._connect_status = QLabel(_("Not connected to server."), self)
        layout.addWidget(self._connect_status)
        layout.addSpacing(6)

        self._connect_error = QLabel(self)
        self._connect_error.setVisible(False)
        self._connect_error.setWordWrap(True)
        self._connect_error.setStyleSheet(f"color: {theme.yellow};")
        layout.addWidget(self._connect_error)

        self._settings_button = QPushButton(theme.icon("settings"), _("Configure"), self)
        self._settings_button.setMinimumHeight(32)
        self._settings_button.clicked.connect(self.show_settings)
        layout.addWidget(self._settings_button)

        layout.addStretch()

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
            if self._server.upgrade_required:
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


class ImageDiffusionWidget(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("AI Image Generation"))
        self._welcome = WelcomeWidget(root.server)
        self._generation = GenerationWidget()
        self._upscaling = UpscaleWidget()
        self._animation = AnimationWidget()
        self._live = LiveWidget()
        self._frame = QStackedWidget(self)
        self._frame.addWidget(self._welcome)
        self._frame.addWidget(self._generation)
        self._frame.addWidget(self._upscaling)
        self._frame.addWidget(self._live)
        self._frame.addWidget(self._animation)
        self.setWidget(self._frame)

        root.connection.state_changed.connect(self.update_content)
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
        if model is None or connection.state is not ConnectionState.connected:
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
