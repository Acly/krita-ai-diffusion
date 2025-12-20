"""
UI Widget for general Diffusers generation (txt2img, img2img, inpaint) and Qwen layered.
"""

from PyQt5.QtCore import Qt, QMetaObject, QTimer
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QSlider,
    QLineEdit,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
)

from ..diffusers_connection import (
    DiffusersConnection,
    DiffusersConnectionState,
    get_diffusers_connection,
)
from ..api import DiffusersMode
from ..jobs import JobKind
from ..model import Model, Error, ErrorKind
from ..localization import translate as _
from ..root import root
from ..settings import settings, DiffusersModelPreset
from .widget import WorkspaceSelectWidget, ErrorBox
from . import theme


class DiffusersWidget(QWidget):
    """Unified widget for all diffusers generation modes."""

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

        # VRAM usage bar
        vram_layout = QHBoxLayout()
        vram_layout.setContentsMargins(0, 0, 0, 0)
        self.vram_label = QLabel(_("VRAM:"), self)
        self.vram_label.setStyleSheet(f"color: {theme.grey};")
        vram_layout.addWidget(self.vram_label)

        self.vram_bar = QProgressBar(self)
        self.vram_bar.setMinimum(0)
        self.vram_bar.setMaximum(100)
        self.vram_bar.setValue(0)
        self.vram_bar.setFixedHeight(14)
        self.vram_bar.setTextVisible(True)
        self.vram_bar.setFormat("%p%")
        vram_layout.addWidget(self.vram_bar)

        self.vram_info = QLabel(self)
        self.vram_info.setStyleSheet(f"color: {theme.grey}; font-size: 10px;")
        vram_layout.addWidget(self.vram_info)

        self.vram_widget = QWidget(self)
        self.vram_widget.setLayout(vram_layout)
        self.vram_widget.setVisible(False)  # Hidden until connected
        layout.addWidget(self.vram_widget)

        # Timer to poll VRAM usage
        self._vram_timer: QTimer | None = None

        # Connect/Disconnect/Install/Configure buttons
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

        self.configure_button = QPushButton(_("Configure"), self)
        self.configure_button.clicked.connect(self._open_settings)
        button_layout.addWidget(self.configure_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Mode selection
        mode_group = QGroupBox(_("Mode"), self)
        mode_layout = QVBoxLayout(mode_group)

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem(_("Text to Image"), DiffusersMode.text_to_image)
        self.mode_combo.addItem(_("Image to Image"), DiffusersMode.image_to_image)
        self.mode_combo.addItem(_("Inpaint"), DiffusersMode.inpaint)
        self.mode_combo.addItem(_("Layered Generate"), DiffusersMode.layered_generate)
        self.mode_combo.addItem(_("Layered Segment"), DiffusersMode.layered_segment)
        self.mode_combo.currentIndexChanged.connect(self._update_mode)
        self.mode_combo.currentIndexChanged.connect(self._save_mode)
        mode_layout.addWidget(self.mode_combo)

        layout.addWidget(mode_group)

        # Model preset selection (for non-layered modes)
        self.model_group = QGroupBox(_("Model Preset"), self)
        model_layout = QVBoxLayout(self.model_group)

        self.preset_combo = QComboBox(self)
        self.preset_combo.setToolTip(_("Select a model preset with optimized settings"))
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        model_layout.addWidget(self.preset_combo)

        # Preset info label
        self.preset_info = QLabel(self)
        self.preset_info.setWordWrap(True)
        self.preset_info.setStyleSheet(f"color: {theme.grey}; font-size: 10px;")
        model_layout.addWidget(self.preset_info)

        # Preset management buttons
        preset_buttons = QHBoxLayout()
        self.manage_presets_button = QPushButton(_("Manage..."), self)
        self.manage_presets_button.clicked.connect(self._manage_presets)
        preset_buttons.addWidget(self.manage_presets_button)
        preset_buttons.addStretch()
        model_layout.addLayout(preset_buttons)

        self._populate_presets()
        layout.addWidget(self.model_group)

        # Prompt input
        self.prompt_group = QGroupBox(_("Prompt"), self)
        prompt_layout = QVBoxLayout(self.prompt_group)

        self.prompt_input = QTextEdit(self)
        self.prompt_input.setPlaceholderText(_("Enter a description..."))
        self.prompt_input.setMaximumHeight(80)
        prompt_layout.addWidget(self.prompt_input)

        self.negative_prompt_input = QTextEdit(self)
        self.negative_prompt_input.setPlaceholderText(_("Negative prompt (optional)..."))
        self.negative_prompt_input.setMaximumHeight(50)
        prompt_layout.addWidget(self.negative_prompt_input)

        layout.addWidget(self.prompt_group)

        # Generation settings
        self.settings_group = QGroupBox(_("Settings"), self)
        settings_layout = QVBoxLayout(self.settings_group)

        # Resolution presets (for non-layered modes)
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel(_("Resolution:"), self))
        self.resolution_combo = QComboBox(self)
        self.resolution_combo.addItem("1024 × 1024", (1024, 1024))
        self.resolution_combo.addItem("1152 × 896", (1152, 896))
        self.resolution_combo.addItem("896 × 1152", (896, 1152))
        self.resolution_combo.addItem("1216 × 832", (1216, 832))
        self.resolution_combo.addItem("832 × 1216", (832, 1216))
        self.resolution_combo.addItem("1344 × 768", (1344, 768))
        self.resolution_combo.addItem("768 × 1344", (768, 1344))
        self.resolution_combo.addItem("1536 × 640", (1536, 640))
        self.resolution_combo.addItem("640 × 1536", (640, 1536))
        self.resolution_combo.addItem(_("Custom..."), None)
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self.resolution_combo.setToolTip(_("Output resolution"))
        resolution_layout.addWidget(self.resolution_combo)
        resolution_layout.addStretch()
        settings_layout.addLayout(resolution_layout)

        # Custom resolution inputs (hidden by default)
        self.custom_resolution_widget = QWidget(self)
        custom_res_layout = QHBoxLayout(self.custom_resolution_widget)
        custom_res_layout.setContentsMargins(0, 0, 0, 0)

        custom_res_layout.addWidget(QLabel(_("W:"), self))
        self.width_spin = QSpinBox(self)
        self.width_spin.setMinimum(256)
        self.width_spin.setMaximum(4096)
        self.width_spin.setSingleStep(64)
        self.width_spin.setValue(1024)
        custom_res_layout.addWidget(self.width_spin)

        custom_res_layout.addWidget(QLabel(_("H:"), self))
        self.height_spin = QSpinBox(self)
        self.height_spin.setMinimum(256)
        self.height_spin.setMaximum(4096)
        self.height_spin.setSingleStep(64)
        self.height_spin.setValue(1024)
        custom_res_layout.addWidget(self.height_spin)

        custom_res_layout.addStretch()
        self.custom_resolution_widget.setVisible(False)
        settings_layout.addWidget(self.custom_resolution_widget)

        # Guidance scale
        self.guidance_layout = QHBoxLayout()
        self.guidance_label = QLabel(_("Guidance:"), self)
        self.guidance_layout.addWidget(self.guidance_label)
        self.guidance_spin = QDoubleSpinBox(self)
        self.guidance_spin.setMinimum(1.0)
        self.guidance_spin.setMaximum(30.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(1.0)
        self.guidance_spin.setToolTip(_("CFG scale / guidance scale"))
        self.guidance_layout.addWidget(self.guidance_spin)
        self.guidance_layout.addStretch()
        settings_layout.addLayout(self.guidance_layout)

        # Steps
        self.steps_layout = QHBoxLayout()
        self.steps_label = QLabel(_("Steps:"), self)
        self.steps_layout.addWidget(self.steps_label)
        self.steps_spin = QSpinBox(self)
        self.steps_spin.setMinimum(1)
        self.steps_spin.setMaximum(150)
        self.steps_spin.setValue(8)
        self.steps_spin.setToolTip(_("Number of inference steps"))
        self.steps_layout.addWidget(self.steps_spin)
        self.steps_layout.addStretch()
        settings_layout.addLayout(self.steps_layout)

        # Sampler
        sampler_layout = QHBoxLayout()
        sampler_layout.addWidget(QLabel(_("Sampler:"), self))
        self.sampler_combo = QComboBox(self)
        self.sampler_combo.addItem(_("Default"), "default")
        self.sampler_combo.addItem("Euler", "euler")
        self.sampler_combo.addItem("DPM", "dpm")
        self.sampler_combo.addItem("DPM SDE", "sdpm")
        self.sampler_combo.addItem("Adams", "adams")
        self.sampler_combo.addItem("UniPC", "unipc")
        self.sampler_combo.addItem("UniP", "unip")
        self.sampler_combo.addItem("SPC", "spc")
        self.sampler_combo.setCurrentIndex(0)
        self.sampler_combo.setToolTip(_("Sampling method (via skrample)"))
        self.sampler_combo.currentIndexChanged.connect(self._update_sampler_order)
        sampler_layout.addWidget(self.sampler_combo)
        sampler_layout.addStretch()
        settings_layout.addLayout(sampler_layout)

        # Sampler Order (for higher-order samplers)
        self.order_layout = QHBoxLayout()
        self.order_label = QLabel(_("Order:"), self)
        self.order_layout.addWidget(self.order_label)
        self.order_spin = QSpinBox(self)
        self.order_spin.setMinimum(1)
        self.order_spin.setMaximum(9)
        self.order_spin.setValue(2)
        self.order_spin.setToolTip(_("Solver order (higher = potentially better quality, but may be unstable)"))
        self.order_layout.addWidget(self.order_spin)
        self.order_layout.addStretch()
        settings_layout.addLayout(self.order_layout)
        # Hide order by default, show when sampler supports it
        self._set_order_visible(False)

        # Schedule modifier
        self.schedule_layout = QHBoxLayout()
        self.schedule_label = QLabel(_("Schedule:"), self)
        self.schedule_layout.addWidget(self.schedule_label)
        self.schedule_combo = QComboBox(self)
        self.schedule_combo.addItem(_("Default"), "default")
        self.schedule_combo.addItem("Beta", "beta")
        self.schedule_combo.addItem("Sigmoid", "sigmoid")
        self.schedule_combo.addItem("Karras", "karras")
        self.schedule_combo.setCurrentIndex(0)
        self.schedule_combo.setToolTip(_("Schedule modifier (via skrample)"))
        self.schedule_layout.addWidget(self.schedule_combo)
        self.schedule_layout.addStretch()
        settings_layout.addLayout(self.schedule_layout)

        # Strength (for img2img/inpaint)
        self.strength_layout = QHBoxLayout()
        self.strength_layout.addWidget(QLabel(_("Strength:"), self))
        self.strength_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setValue(75)
        self.strength_slider.setToolTip(_("Denoising strength (0-100%)"))
        self.strength_layout.addWidget(self.strength_slider)
        self.strength_label = QLabel("75%", self)
        self.strength_layout.addWidget(self.strength_label)
        self.strength_slider.valueChanged.connect(
            lambda v: self.strength_label.setText(f"{v}%")
        )
        settings_layout.addLayout(self.strength_layout)

        # Seed
        self.seed_layout = QHBoxLayout()
        self.seed_label = QLabel(_("Seed:"), self)
        self.seed_layout.addWidget(self.seed_label)
        self.seed_spin = QSpinBox(self)
        self.seed_spin.setMinimum(-1)
        self.seed_spin.setMaximum(2147483647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText(_("Random"))
        self.seed_spin.setToolTip(_("Random seed (-1 for random)"))
        self.seed_layout.addWidget(self.seed_spin)
        self.seed_layout.addStretch()
        settings_layout.addLayout(self.seed_layout)

        layout.addWidget(self.settings_group)

        # Layered-specific settings (only shown for layered modes)
        self.layered_group = QGroupBox(_("Layered Settings"), self)
        layered_layout = QVBoxLayout(self.layered_group)

        layers_layout = QHBoxLayout()
        layers_layout.addWidget(QLabel(_("Layers:"), self))
        self.layers_spin = QSpinBox(self)
        self.layers_spin.setMinimum(2)
        self.layers_spin.setMaximum(6)
        self.layers_spin.setValue(4)
        self.layers_spin.setToolTip(_("Number of layers to generate (2-6)"))
        layers_layout.addWidget(self.layers_spin)
        layers_layout.addStretch()
        layered_layout.addLayout(layers_layout)

        qwen_resolution_layout = QHBoxLayout()
        qwen_resolution_layout.addWidget(QLabel(_("Resolution:"), self))
        self.qwen_resolution_combo = QComboBox(self)
        self.qwen_resolution_combo.addItem("640", 640)
        self.qwen_resolution_combo.addItem("1024", 1024)
        self.qwen_resolution_combo.setCurrentIndex(0)
        self.qwen_resolution_combo.setToolTip(_("Resolution bucket for Qwen layered"))
        qwen_resolution_layout.addWidget(self.qwen_resolution_combo)
        qwen_resolution_layout.addStretch()
        layered_layout.addLayout(qwen_resolution_layout)

        layered_steps_layout = QHBoxLayout()
        layered_steps_layout.addWidget(QLabel(_("Steps:"), self))
        self.layered_steps_spin = QSpinBox(self)
        self.layered_steps_spin.setMinimum(1)
        self.layered_steps_spin.setMaximum(100)
        self.layered_steps_spin.setValue(50)
        self.layered_steps_spin.setToolTip(_("Number of inference steps for layered generation"))
        layered_steps_layout.addWidget(self.layered_steps_spin)
        layered_steps_layout.addStretch()
        layered_layout.addLayout(layered_steps_layout)

        layered_seed_layout = QHBoxLayout()
        layered_seed_layout.addWidget(QLabel(_("Seed:"), self))
        self.layered_seed_spin = QSpinBox(self)
        self.layered_seed_spin.setMinimum(-1)
        self.layered_seed_spin.setMaximum(2147483647)
        self.layered_seed_spin.setValue(-1)
        self.layered_seed_spin.setSpecialValueText(_("Random"))
        self.layered_seed_spin.setToolTip(_("Random seed (-1 for random)"))
        layered_seed_layout.addWidget(self.layered_seed_spin)
        layered_seed_layout.addStretch()
        layered_layout.addLayout(layered_seed_layout)

        layout.addWidget(self.layered_group)

        # Generate button
        self.generate_button = QPushButton(_("Generate"), self)
        self.generate_button.setMinimumHeight(32)
        self.generate_button.clicked.connect(self._generate)
        layout.addWidget(self.generate_button)

        # Stop button (hidden by default)
        self.stop_button = QPushButton(_("Stop Generation"), self)
        self.stop_button.setMinimumHeight(32)
        self.stop_button.clicked.connect(self._stop_generation)
        self.stop_button.setVisible(False)
        layout.addWidget(self.stop_button)

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
        self._diffusers.state_changed.connect(self._update_connection_state)
        self._diffusers.model_loading_changed.connect(self._update_model_loading)

        # Restore last used mode and update UI
        self._restore_mode()
        self._update_mode()
        self._update_connection_state(self._diffusers.state)

    def _restore_mode(self):
        """Restore last used mode from settings."""
        mode_map = {
            "text_to_image": DiffusersMode.text_to_image,
            "image_to_image": DiffusersMode.image_to_image,
            "inpaint": DiffusersMode.inpaint,
            "layered_generate": DiffusersMode.layered_generate,
            "layered_segment": DiffusersMode.layered_segment,
        }
        last_mode = settings.diffusers_last_mode
        if last_mode in mode_map:
            target_mode = mode_map[last_mode]
            for i in range(self.mode_combo.count()):
                if self.mode_combo.itemData(i) == target_mode:
                    self.mode_combo.setCurrentIndex(i)
                    break

    def _save_mode(self):
        """Save current mode to settings."""
        mode = self.mode_combo.currentData()
        mode_str_map = {
            DiffusersMode.text_to_image: "text_to_image",
            DiffusersMode.image_to_image: "image_to_image",
            DiffusersMode.inpaint: "inpaint",
            DiffusersMode.layered_generate: "layered_generate",
            DiffusersMode.layered_segment: "layered_segment",
        }
        if mode in mode_str_map:
            settings.diffusers_last_mode = mode_str_map[mode]
            settings.save()

    def _update_mode(self):
        """Update UI based on selected mode."""
        mode = self.mode_combo.currentData()
        is_layered = mode in (DiffusersMode.layered_generate, DiffusersMode.layered_segment)
        is_segment = mode == DiffusersMode.layered_segment
        is_img2img = mode in (DiffusersMode.image_to_image, DiffusersMode.inpaint)

        # Show/hide groups based on mode
        # Layered Generate needs model preset and generation settings (it generates the base image)
        # Layered Segment only needs layered settings (it uses existing image)
        self.model_group.setVisible(not is_segment)  # Show for Generate, hide for Segment
        self.layered_group.setVisible(is_layered)
        # Settings group: always show (contains sampler controls needed for all modes)
        self.settings_group.setVisible(True)
        # Layered Generate needs prompt (generates from text), Layered Segment doesn't (uses existing image)
        self.prompt_group.setVisible(not is_segment)
        self.prompt_group.setEnabled(not is_segment)

        # Show/hide resolution controls (hide for segment and img2img modes)
        # Layered Generate needs resolution for the base image generation
        show_resolution = not is_segment and not is_img2img
        self.resolution_combo.setVisible(show_resolution)
        # Find and hide the resolution label too
        res_layout = self.resolution_combo.parent().layout()
        if res_layout:
            for i in range(res_layout.count()):
                item = res_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setVisible(show_resolution)
        # Only show custom resolution widget if custom is selected and not hidden
        is_custom = self.resolution_combo.currentData() is None
        self.custom_resolution_widget.setVisible(show_resolution and is_custom)

        # Layered Segment uses fixed values; Layered Generate can use custom guidance/steps
        # Hide guidance and steps entirely for segment mode (they're in layered_group)
        self.guidance_spin.setVisible(not is_segment)
        self.guidance_label.setVisible(not is_segment)
        self.steps_spin.setVisible(not is_segment)
        self.steps_label.setVisible(not is_segment)
        # Hide the main seed control for segment mode (layered_group has its own seed)
        self.seed_spin.setVisible(not is_segment)
        self.seed_label.setVisible(not is_segment)

        # Show strength slider only for img2img/inpaint
        for i in range(self.strength_layout.count()):
            item = self.strength_layout.itemAt(i)
            if item.widget():
                item.widget().setVisible(is_img2img)

        # Update button text
        if mode == DiffusersMode.text_to_image:
            self.generate_button.setText(_("Generate"))
        elif mode == DiffusersMode.image_to_image:
            self.generate_button.setText(_("Transform Image"))
        elif mode == DiffusersMode.inpaint:
            self.generate_button.setText(_("Inpaint Selection"))
        elif mode == DiffusersMode.layered_generate:
            self.generate_button.setText(_("Generate Layers"))
        elif mode == DiffusersMode.layered_segment:
            self.generate_button.setText(_("Segment to Layers"))

    def _set_order_visible(self, visible: bool):
        """Show/hide the order control."""
        self.order_label.setVisible(visible)
        self.order_spin.setVisible(visible)

    def _update_sampler_order(self):
        """Update order spinner visibility based on selected sampler."""
        sampler = self.sampler_combo.currentData()
        # Higher-order samplers: DPM, Adams, UniPC, UniP, SPC
        high_order_samplers = {"dpm", "sdpm", "adams", "unipc", "unip", "spc"}
        self._set_order_visible(sampler in high_order_samplers)

    def _on_resolution_changed(self):
        """Handle resolution preset change."""
        resolution = self.resolution_combo.currentData()
        if resolution is None:
            # Custom resolution selected
            self.custom_resolution_widget.setVisible(True)
        else:
            # Preset resolution selected
            self.custom_resolution_widget.setVisible(False)
            self.width_spin.setValue(resolution[0])
            self.height_spin.setValue(resolution[1])

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
            # Start VRAM polling
            self._start_vram_polling()
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
            self.connect_button.setVisible(False)
            self.disconnect_button.setVisible(True)
            self.disconnect_button.setEnabled(True)
            self.generate_button.setEnabled(False)
        elif state == DiffusersConnectionState.server_starting:
            self.connection_status.setText(_("Starting server..."))
            self.connection_status.setStyleSheet(f"color: {theme.grey};")
            self.connect_button.setVisible(False)
            self.disconnect_button.setVisible(True)
            self.disconnect_button.setEnabled(True)
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
            self._stop_vram_polling()
            if self._diffusers.error:
                self.error_box.error = Error(ErrorKind.server_error, self._diffusers.error)
        else:  # disconnected
            self.connection_status.setText(_("Not connected"))
            self.connection_status.setStyleSheet(f"color: {theme.grey};")
            self.connect_button.setVisible(True)
            self.connect_button.setEnabled(True)
            self.disconnect_button.setVisible(False)
            self.generate_button.setEnabled(False)
            self._reset_buttons()  # Hide stop button, show generate button
            self._stop_vram_polling()

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

    def _open_settings(self):
        """Open settings dialog."""
        from .settings import SettingsDialog
        dialog = SettingsDialog(root.server)
        dialog.exec_()

    def _generate(self):
        """Start generation based on selected mode."""
        model = root.active_model
        if model is None:
            return

        # Show stop button, hide generate button
        self.generate_button.setVisible(False)
        self.stop_button.setVisible(True)

        mode = self.mode_combo.currentData()

        if mode in (DiffusersMode.layered_generate, DiffusersMode.layered_segment):
            # Use existing layered generation
            num_layers = self.layers_spin.value()
            resolution = self.qwen_resolution_combo.currentData()
            steps = self.layered_steps_spin.value()
            seed = self.layered_seed_spin.value()

            if mode == DiffusersMode.layered_generate:
                # Layered Generate needs both txt2img params and Qwen params
                prompt = self.prompt_input.toPlainText().strip()
                negative_prompt = self.negative_prompt_input.toPlainText().strip()

                # Get txt2img settings for base image generation
                preset = self._get_current_preset()
                model_id = preset.model_id if preset else ""

                model.generate_layered(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_layers=num_layers,
                    resolution=resolution,
                    seed=seed,
                    steps=steps,
                    # Base image generation params
                    model_id=model_id,
                    width=self.width_spin.value(),
                    height=self.height_spin.value(),
                    guidance_scale=self.guidance_spin.value(),
                    num_steps=self.steps_spin.value(),
                    preset=preset,
                    sampler=self.sampler_combo.currentData(),
                    sampler_order=self.order_spin.value(),
                    schedule=self.schedule_combo.currentData(),
                )
            else:
                model.segment_to_layers(
                    num_layers=num_layers,
                    resolution=resolution,
                    seed=seed,
                    steps=steps,
                    sampler=self.sampler_combo.currentData(),
                    sampler_order=self.order_spin.value(),
                    schedule=self.schedule_combo.currentData(),
                )
        else:
            # General diffusers generation
            seed = self.seed_spin.value()
            prompt = self.prompt_input.toPlainText().strip()
            negative_prompt = self.negative_prompt_input.toPlainText().strip()

            if mode == DiffusersMode.text_to_image and not prompt:
                self.error_box.error = Error(
                    ErrorKind.plugin_error, _("Please enter a prompt")
                )
                return

            mode_str = {
                DiffusersMode.text_to_image: "text_to_image",
                DiffusersMode.image_to_image: "img2img",
                DiffusersMode.inpaint: "inpaint",
            }[mode]

            # Get preset settings
            preset = self._get_current_preset()
            model_id = preset.model_id if preset else ""

            model.generate_diffusers(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_id=model_id,
                mode=mode_str,
                width=self.width_spin.value(),
                height=self.height_spin.value(),
                guidance_scale=self.guidance_spin.value(),
                num_steps=self.steps_spin.value(),
                strength=self.strength_slider.value() / 100.0,
                seed=seed,
                preset=preset,  # Pass full preset for optimization settings
                sampler=self.sampler_combo.currentData(),
                sampler_order=self.order_spin.value(),
                schedule=self.schedule_combo.currentData(),
            )

    def _stop_generation(self):
        """Stop the current generation."""
        from .. import eventloop

        async def do_stop():
            client = self._diffusers.client_if_connected
            if client:
                try:
                    await client.interrupt()
                except Exception as e:
                    from ..util import client_logger as log
                    log.warning(f"Failed to interrupt generation: {e}")

        eventloop.run(do_stop())
        self._reset_buttons()

    def _reset_buttons(self):
        """Reset generate/stop button visibility."""
        self.generate_button.setVisible(True)
        self.stop_button.setVisible(False)

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
            # Reset buttons when complete
            if progress >= 1.0:
                self._reset_buttons()

    def _update_error(self, error):
        """Update error display."""
        if error and error.message:
            self.error_box.error = error
            self._reset_buttons()  # Also reset on error
        else:
            self.error_box.reset()

    def _update_status_message(self, message: str):
        """Update status message display during job processing."""
        if message:
            self.model_loading_label.setText(message)
            self.model_loading_label.setVisible(True)
        else:
            self.model_loading_label.setVisible(False)

    def _start_vram_polling(self):
        """Start polling VRAM usage from server."""
        if self._vram_timer is None:
            self._vram_timer = QTimer(self)
            self._vram_timer.timeout.connect(self._poll_vram)
        self._vram_timer.start(2000)  # Poll every 2 seconds
        self.vram_widget.setVisible(True)
        # Do initial poll
        self._poll_vram()

    def _stop_vram_polling(self):
        """Stop VRAM polling."""
        if self._vram_timer is not None:
            self._vram_timer.stop()
        self.vram_widget.setVisible(False)

    def _poll_vram(self):
        """Poll VRAM usage from server."""
        from .. import eventloop

        async def fetch_vram():
            client = self._diffusers.client_if_connected
            if client is None:
                return None
            try:
                return await client.get_vram_usage()
            except Exception:
                return None

        def on_result(result):
            if result is None:
                return
            devices = result.get("devices", [])
            backend = result.get("backend", "unknown")

            if devices:
                # Use first device for the bar
                device = devices[0]
                percent = device.get("vram_percent", 0)
                used_gb = device.get("vram_used", 0) / (1024**3)
                total_gb = device.get("vram_total", 0) / (1024**3)

                self.vram_bar.setValue(int(percent))
                self.vram_info.setText(f"{used_gb:.1f}/{total_gb:.1f} GB")
                self.vram_label.setText(f"VRAM ({backend}):")
            elif backend == "cpu":
                self.vram_bar.setValue(0)
                self.vram_info.setText("N/A")
                self.vram_label.setText("VRAM (CPU):")

        eventloop.run(fetch_vram()).add_done_callback(
            lambda f: on_result(f.result()) if not f.cancelled() else None
        )

    def _populate_presets(self):
        """Populate the preset combo box."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()

        presets = settings.get_diffusers_presets()
        for preset in presets:
            self.preset_combo.addItem(preset.name, preset)

        # Restore selected preset
        selected = settings.diffusers_selected_preset
        for i in range(self.preset_combo.count()):
            if self.preset_combo.itemText(i) == selected:
                self.preset_combo.setCurrentIndex(i)
                break

        self.preset_combo.blockSignals(False)
        self._update_preset_info()

    def _on_preset_changed(self):
        """Handle preset selection change."""
        preset = self.preset_combo.currentData()
        if preset:
            settings.diffusers_selected_preset = preset.name
            settings.save()

            # Update UI with preset defaults
            self.steps_spin.setValue(preset.default_steps)
            self.guidance_spin.setValue(preset.default_guidance)

            self._update_preset_info()

    def _update_preset_info(self):
        """Update the preset info label."""
        preset = self.preset_combo.currentData()
        if preset:
            info_parts = [preset.model_id]
            if preset.offload != "none":
                info_parts.append(f"offload: {preset.offload}")
            if preset.quantization != "none":
                info_parts.append(f"quant: {preset.quantization}")
            if preset.ramtorch:
                info_parts.append("ramtorch")
            self.preset_info.setText(" | ".join(info_parts))
        else:
            self.preset_info.setText("")

    def _get_current_preset(self) -> DiffusersModelPreset | None:
        """Get the currently selected preset."""
        return self.preset_combo.currentData()

    def _manage_presets(self):
        """Open the preset management dialog."""
        dialog = PresetManagerDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self._populate_presets()


class PresetManagerDialog(QDialog):
    """Dialog for managing diffusers model presets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_("Manage Model Presets"))
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        layout = QVBoxLayout(self)

        # Preset list
        self.preset_list = QListWidget(self)
        self.preset_list.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(self.preset_list)

        # Buttons for list management
        list_buttons = QHBoxLayout()

        self.add_button = QPushButton(_("Add New"), self)
        self.add_button.clicked.connect(self._add_preset)
        list_buttons.addWidget(self.add_button)

        self.edit_button = QPushButton(_("Edit"), self)
        self.edit_button.clicked.connect(self._edit_preset)
        self.edit_button.setEnabled(False)
        list_buttons.addWidget(self.edit_button)

        self.delete_button = QPushButton(_("Delete"), self)
        self.delete_button.clicked.connect(self._delete_preset)
        self.delete_button.setEnabled(False)
        list_buttons.addWidget(self.delete_button)

        list_buttons.addStretch()
        layout.addLayout(list_buttons)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._populate_list()

    def _populate_list(self):
        """Populate the preset list."""
        self.preset_list.clear()
        presets = settings.get_diffusers_presets()
        for preset in presets:
            item = QListWidgetItem(preset.name)
            item.setData(Qt.UserRole, preset)
            # Mark default presets
            if settings.is_default_diffusers_preset(preset.name):
                item.setText(f"{preset.name} (built-in)")
            self.preset_list.addItem(item)

    def _on_selection_changed(self, row):
        """Handle selection change."""
        if row >= 0:
            item = self.preset_list.item(row)
            preset = item.data(Qt.UserRole)
            is_default = settings.is_default_diffusers_preset(preset.name)
            self.edit_button.setEnabled(not is_default)
            self.delete_button.setEnabled(not is_default)
        else:
            self.edit_button.setEnabled(False)
            self.delete_button.setEnabled(False)

    def _add_preset(self):
        """Add a new preset."""
        dialog = PresetEditDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            preset = dialog.get_preset()
            if preset:
                settings.add_diffusers_preset(preset)
                self._populate_list()

    def _edit_preset(self):
        """Edit the selected preset."""
        item = self.preset_list.currentItem()
        if item:
            preset = item.data(Qt.UserRole)
            dialog = PresetEditDialog(self, preset)
            if dialog.exec_() == QDialog.Accepted:
                new_preset = dialog.get_preset()
                if new_preset:
                    # Remove old and add new
                    settings.remove_diffusers_preset(preset.name)
                    settings.add_diffusers_preset(new_preset)
                    self._populate_list()

    def _delete_preset(self):
        """Delete the selected preset."""
        item = self.preset_list.currentItem()
        if item:
            preset = item.data(Qt.UserRole)
            reply = QMessageBox.question(
                self,
                _("Delete Preset"),
                _("Are you sure you want to delete '{}'?").format(preset.name),
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                settings.remove_diffusers_preset(preset.name)
                self._populate_list()


class PresetEditDialog(QDialog):
    """Dialog for editing a single preset."""

    def __init__(self, parent=None, preset: DiffusersModelPreset | None = None):
        super().__init__(parent)
        self.setWindowTitle(_("Edit Preset") if preset else _("New Preset"))
        self.setMinimumWidth(400)

        layout = QFormLayout(self)

        # Name
        self.name_edit = QLineEdit(self)
        self.name_edit.setText(preset.name if preset else "")
        layout.addRow(_("Name:"), self.name_edit)

        # Model ID
        self.model_id_edit = QLineEdit(self)
        self.model_id_edit.setText(preset.model_id if preset else "")
        self.model_id_edit.setPlaceholderText(_("e.g., black-forest-labs/FLUX.1-dev"))
        layout.addRow(_("Model ID:"), self.model_id_edit)

        # Default steps
        self.steps_spin = QSpinBox(self)
        self.steps_spin.setRange(1, 150)
        self.steps_spin.setValue(preset.default_steps if preset else 30)
        layout.addRow(_("Default Steps:"), self.steps_spin)

        # Default guidance
        self.guidance_spin = QDoubleSpinBox(self)
        self.guidance_spin.setRange(1.0, 30.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(preset.default_guidance if preset else 7.5)
        layout.addRow(_("Default Guidance:"), self.guidance_spin)

        # Offload mode
        self.offload_combo = QComboBox(self)
        self.offload_combo.addItem(_("None"), "none")
        self.offload_combo.addItem(_("Model CPU Offload"), "model")
        self.offload_combo.addItem(_("Sequential CPU Offload"), "sequential")
        if preset:
            for i in range(self.offload_combo.count()):
                if self.offload_combo.itemData(i) == preset.offload:
                    self.offload_combo.setCurrentIndex(i)
                    break
        layout.addRow(_("CPU Offload:"), self.offload_combo)

        # Quantization
        self.quant_combo = QComboBox(self)
        self.quant_combo.addItem(_("None"), "none")
        self.quant_combo.addItem(_("INT8"), "int8")
        self.quant_combo.addItem(_("INT4"), "int4")
        if preset:
            for i in range(self.quant_combo.count()):
                if self.quant_combo.itemData(i) == preset.quantization:
                    self.quant_combo.setCurrentIndex(i)
                    break
        layout.addRow(_("Quantization:"), self.quant_combo)

        # Quantize transformer
        self.quant_transformer_check = QCheckBox(self)
        self.quant_transformer_check.setChecked(preset.quantize_transformer if preset else True)
        layout.addRow(_("Quantize Transformer:"), self.quant_transformer_check)

        # Quantize text encoder
        self.quant_text_encoder_check = QCheckBox(self)
        self.quant_text_encoder_check.setChecked(preset.quantize_text_encoder if preset else False)
        layout.addRow(_("Quantize Text Encoder:"), self.quant_text_encoder_check)

        # VAE tiling
        self.vae_tiling_check = QCheckBox(self)
        self.vae_tiling_check.setChecked(preset.vae_tiling if preset else True)
        layout.addRow(_("VAE Tiling:"), self.vae_tiling_check)

        # RamTorch
        self.ramtorch_check = QCheckBox(self)
        self.ramtorch_check.setChecked(preset.ramtorch if preset else False)
        self.ramtorch_check.setToolTip(
            _("Tensor-wise streaming between CPU and GPU. More aggressive than CPU offload.")
        )
        layout.addRow(_("RamTorch:"), self.ramtorch_check)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def _validate_and_accept(self):
        """Validate input and accept dialog."""
        name = self.name_edit.text().strip()
        model_id = self.model_id_edit.text().strip()

        if not name:
            QMessageBox.warning(self, _("Error"), _("Please enter a preset name."))
            return
        if not model_id:
            QMessageBox.warning(self, _("Error"), _("Please enter a model ID."))
            return

        self.accept()

    def get_preset(self) -> DiffusersModelPreset | None:
        """Get the preset from dialog values."""
        name = self.name_edit.text().strip()
        model_id = self.model_id_edit.text().strip()

        if not name or not model_id:
            return None

        return DiffusersModelPreset(
            name=name,
            model_id=model_id,
            default_steps=self.steps_spin.value(),
            default_guidance=self.guidance_spin.value(),
            offload=self.offload_combo.currentData(),
            quantization=self.quant_combo.currentData(),
            quantize_transformer=self.quant_transformer_check.isChecked(),
            quantize_text_encoder=self.quant_text_encoder_check.isChecked(),
            vae_tiling=self.vae_tiling_check.isChecked(),
            ramtorch=self.ramtorch_check.isChecked(),
        )
