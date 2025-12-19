"""
UI Widget for Qwen Image Layered generation and segmentation.
"""

from PyQt5.QtCore import Qt, QMetaObject
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
)

from ..diffusers_connection import (
    DiffusersConnection,
    DiffusersConnectionState,
    get_diffusers_connection,
)
from ..jobs import JobKind
from ..model import Model, Error, ErrorKind
from ..localization import translate as _
from ..root import root
from ..settings import settings
from .widget import WorkspaceSelectWidget, GenerateButton, ErrorBox
from . import theme


class LayeredWidget(QWidget):
    """Widget for Qwen Image Layered generation and segmentation."""

    _model: Model
    _model_bindings: list[QMetaObject.Connection]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        self._diffusers = get_diffusers_connection()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 0)
        self.setLayout(layout)

        # Workspace selector
        self.workspace_select = WorkspaceSelectWidget(self)
        layout.addWidget(self.workspace_select)

        # Connection status
        self.connection_status = QLabel(self)
        self.connection_status.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(self.connection_status)

        # Connect/Disconnect/Install buttons
        button_layout = QHBoxLayout()

        self.connect_button = QPushButton(_("Connect"), self)
        self.connect_button.clicked.connect(self._connect_server)
        button_layout.addWidget(self.connect_button)

        self.disconnect_button = QPushButton(_("Disconnect"), self)
        self.disconnect_button.clicked.connect(self._disconnect_server)
        self.disconnect_button.setVisible(False)
        button_layout.addWidget(self.disconnect_button)

        self.install_button = QPushButton(_("Install Server"), self)
        self.install_button.clicked.connect(self._install_server)
        self.install_button.setVisible(False)
        button_layout.addWidget(self.install_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Mode selection
        mode_group = QGroupBox(_("Mode"), self)
        mode_layout = QVBoxLayout(mode_group)

        self.mode_group = QButtonGroup(self)
        self.generate_radio = QRadioButton(_("Generate Layered Image"), self)
        self.generate_radio.setToolTip(_("Generate a new layered image from a text prompt"))
        self.segment_radio = QRadioButton(_("Segment Existing Image"), self)
        self.segment_radio.setToolTip(_("Segment the current canvas image into separate layers"))

        self.mode_group.addButton(self.generate_radio, 0)
        self.mode_group.addButton(self.segment_radio, 1)
        self.generate_radio.setChecked(True)

        mode_layout.addWidget(self.generate_radio)
        mode_layout.addWidget(self.segment_radio)
        layout.addWidget(mode_group)

        # Prompt input (only for generate mode)
        self.prompt_group = QGroupBox(_("Prompt"), self)
        prompt_layout = QVBoxLayout(self.prompt_group)

        self.prompt_input = QTextEdit(self)
        self.prompt_input.setPlaceholderText(_("Enter a description for the layered image..."))
        self.prompt_input.setMaximumHeight(80)
        prompt_layout.addWidget(self.prompt_input)

        self.negative_prompt_input = QTextEdit(self)
        self.negative_prompt_input.setPlaceholderText(_("Negative prompt (optional)..."))
        self.negative_prompt_input.setMaximumHeight(50)
        prompt_layout.addWidget(self.negative_prompt_input)

        layout.addWidget(self.prompt_group)

        # Settings
        settings_group = QGroupBox(_("Settings"), self)
        settings_layout = QVBoxLayout(settings_group)

        # Number of layers
        layers_layout = QHBoxLayout()
        layers_layout.addWidget(QLabel(_("Layers:"), self))
        self.layers_spin = QSpinBox(self)
        self.layers_spin.setMinimum(2)
        self.layers_spin.setMaximum(6)
        self.layers_spin.setValue(4)
        self.layers_spin.setToolTip(_("Number of layers to generate (2-6)"))
        layers_layout.addWidget(self.layers_spin)
        layers_layout.addStretch()
        settings_layout.addLayout(layers_layout)

        # Resolution
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel(_("Resolution:"), self))
        self.resolution_combo = QComboBox(self)
        self.resolution_combo.addItem("640", 640)
        self.resolution_combo.addItem("1024", 1024)
        self.resolution_combo.setCurrentIndex(0)
        self.resolution_combo.setToolTip(_("Resolution bucket for generation"))
        resolution_layout.addWidget(self.resolution_combo)
        resolution_layout.addStretch()
        settings_layout.addLayout(resolution_layout)

        # Seed
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel(_("Seed:"), self))
        self.seed_spin = QSpinBox(self)
        self.seed_spin.setMinimum(-1)
        self.seed_spin.setMaximum(2147483647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText(_("Random"))
        self.seed_spin.setToolTip(_("Random seed (-1 for random)"))
        seed_layout.addWidget(self.seed_spin)
        seed_layout.addStretch()
        settings_layout.addLayout(seed_layout)

        layout.addWidget(settings_group)

        # Generate button
        self.generate_button = QPushButton(_("Generate Layers"), self)
        self.generate_button.setMinimumHeight(32)
        self.generate_button.clicked.connect(self._generate)
        layout.addWidget(self.generate_button)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        # Error box
        self.error_box = ErrorBox(self)
        layout.addWidget(self.error_box)

        layout.addStretch()

        # Model loading status label
        self.model_loading_label = QLabel(self)
        self.model_loading_label.setWordWrap(True)
        self.model_loading_label.setStyleSheet(f"color: {theme.grey};")
        self.model_loading_label.setVisible(False)
        layout.insertWidget(layout.indexOf(self.progress_bar), self.model_loading_label)

        # Connect signals
        self.mode_group.buttonClicked.connect(self._update_mode)
        self._diffusers.state_changed.connect(self._update_connection_state)
        self._diffusers.model_loading_changed.connect(self._update_model_loading)

        # Initial state
        self._update_mode()
        self._update_connection_state(self._diffusers.state)

    def _update_mode(self):
        """Update UI based on selected mode."""
        is_generate = self.generate_radio.isChecked()
        self.prompt_group.setVisible(is_generate)
        self.generate_button.setText(
            _("Generate Layers") if is_generate else _("Segment to Layers")
        )

    def _update_connection_state(self, state: DiffusersConnectionState):
        """Update UI based on connection state."""
        # Hide model loading UI by default
        self.model_loading_label.setVisible(False)
        # Reset error box on state change (except for error state)
        if state != DiffusersConnectionState.error:
            self.error_box.reset()

        if state == DiffusersConnectionState.connected:
            self.connection_status.setText(_("Connected to diffusers server"))
            self.connection_status.setStyleSheet(f"color: {theme.green};")
            self.connect_button.setVisible(False)
            self.disconnect_button.setVisible(True)
            self.install_button.setVisible(False)
            self.generate_button.setEnabled(True)
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(0)
        elif state == DiffusersConnectionState.server_not_installed:
            self.connection_status.setText(_("Diffusers server not installed"))
            self.connection_status.setStyleSheet(f"color: {theme.yellow};")
            self.connect_button.setVisible(False)
            self.disconnect_button.setVisible(False)
            self.install_button.setVisible(True)
            self.generate_button.setEnabled(False)
        elif state == DiffusersConnectionState.connecting:
            self.connection_status.setText(_("Connecting..."))
            self.connection_status.setStyleSheet(f"color: {theme.grey};")
            self.connect_button.setEnabled(False)
            self.disconnect_button.setVisible(True)
            self.generate_button.setEnabled(False)
        elif state == DiffusersConnectionState.server_starting:
            self.connection_status.setText(_("Starting server..."))
            self.connection_status.setStyleSheet(f"color: {theme.grey};")
            self.connect_button.setEnabled(False)
            self.disconnect_button.setVisible(True)
            self.generate_button.setEnabled(False)
        elif state == DiffusersConnectionState.loading_model:
            self.connection_status.setText(_("Loading model..."))
            self.connection_status.setStyleSheet(f"color: {theme.yellow};")
            self.connect_button.setVisible(False)
            self.disconnect_button.setVisible(True)
            self.install_button.setVisible(False)
            self.generate_button.setEnabled(False)
            # Show model loading progress
            self.model_loading_label.setVisible(True)
            self.model_loading_label.setText(
                self._diffusers.model_loading_message or _("Loading model...")
            )
            # Set progress bar to show loading
            progress = self._diffusers.model_loading_progress
            if progress > 0:
                self.progress_bar.setMaximum(100)
                self.progress_bar.setValue(int(progress * 100))
            else:
                self.progress_bar.setMaximum(0)  # Indeterminate
        elif state == DiffusersConnectionState.error:
            self.connection_status.setText(_("Connection error"))
            self.connection_status.setStyleSheet(f"color: {theme.red};")
            self.connect_button.setVisible(True)
            self.connect_button.setEnabled(True)
            self.disconnect_button.setVisible(False)
            self.generate_button.setEnabled(False)
            if self._diffusers.error:
                self.error_box.error = Error(ErrorKind.network_error, self._diffusers.error)
        else:  # disconnected
            self.connection_status.setText(_("Not connected"))
            self.connection_status.setStyleSheet(f"color: {theme.grey};")
            self.connect_button.setVisible(True)
            self.connect_button.setEnabled(True)
            self.disconnect_button.setVisible(False)
            self.generate_button.setEnabled(False)

    def _update_model_loading(self, message: str, progress: float):
        """Update model loading progress display."""
        if self._diffusers.state == DiffusersConnectionState.loading_model:
            self.model_loading_label.setText(message)
            if progress > 0:
                self.progress_bar.setMaximum(100)
                self.progress_bar.setValue(int(progress * 100))
            else:
                self.progress_bar.setMaximum(0)  # Indeterminate

    def _connect_server(self):
        """Connect to the diffusers server."""
        server = self._diffusers.check_server()
        if server.is_installed:
            # Start managed server and connect
            self._diffusers.start_server_and_connect()
        else:
            self._update_connection_state(DiffusersConnectionState.server_not_installed)

    def _disconnect_server(self):
        """Disconnect from the diffusers server."""
        from .. import eventloop
        eventloop.run(self._diffusers.disconnect())

    def _install_server(self):
        """Open server installation dialog."""
        # TODO: Implement installation dialog
        # For now, just show a message
        from PyQt5.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            _("Install Diffusers Server"),
            _(
                "To install the diffusers server, use the following commands:\n\n"
                "1. Navigate to the server directory\n"
                "2. Run: python -m ai_diffusion.diffusers_server install\n\n"
                "Or use the plugin's server management UI once available."
            ),
        )

    def _generate(self):
        """Start generation or segmentation."""
        model = root.active_model
        if model is None:
            return

        num_layers = self.layers_spin.value()
        resolution = self.resolution_combo.currentData()
        seed = self.seed_spin.value()

        if self.generate_radio.isChecked():
            # Generate mode
            prompt = self.prompt_input.toPlainText().strip()
            negative_prompt = self.negative_prompt_input.toPlainText().strip()

            if not prompt:
                self.error_box.error = Error(ErrorKind.plugin_error, _("Please enter a prompt"))
                return

            model.generate_layered(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_layers=num_layers,
                resolution=resolution,
                seed=seed,
            )
        else:
            # Segment mode
            model.segment_to_layers(
                num_layers=num_layers,
                resolution=resolution,
                seed=seed,
            )

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model != model:
            for binding in self._model_bindings:
                if isinstance(binding, QMetaObject.Connection):
                    self._model.disconnect(binding)
            self._model = model
            self._model_bindings = []

            # Bind to model signals
            self._model_bindings.append(
                model.progress_changed.connect(self._update_progress)
            )
            self._model_bindings.append(model.error_changed.connect(self._update_error))
            self._model_bindings.append(
                model.status_message_changed.connect(self._update_status_message)
            )

    def _update_progress(self, progress: float):
        """Update progress bar."""
        if progress < 0:
            self.progress_bar.setMaximum(0)  # Indeterminate
        else:
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(int(progress * 100))

    def _update_error(self, error):
        """Update error display."""
        if error and error.message:
            self.error_box.error = error
        else:
            self.error_box.reset()

    def _update_status_message(self, message: str):
        """Update status message display during job processing."""
        if message:
            self.model_loading_label.setText(message)
            self.model_loading_label.setVisible(True)
        else:
            self.model_loading_label.setVisible(False)
