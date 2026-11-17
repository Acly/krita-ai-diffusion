from __future__ import annotations

from pathlib import Path
from typing import cast

from krita import Krita
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..backend.client import resolve_arch
from ..backend.resources import Arch, ResourceId, ResourceKind, search_paths
from ..backend.server import Server
from ..files import FileFilter, FileFormat
from ..localization import translate as _
from ..model.root import root
from ..settings import ServerMode, Setting, settings
from ..style import SamplerPresets, Style, Styles, StyleSettings
from . import theme
from .lora import LoraList
from .settings_widgets import (
    ComboBoxSetting,
    ExpanderButton,
    LineEditSetting,
    SettingsTab,
    SettingWidget,
    SettingWidgetBase,
    SliderSetting,
    SpinBoxSetting,
    SwitchSetting,
    TextSetting,
)
from .theme import SignalBlocker, add_header


class SamplerWidget(QWidget):
    prefix: str

    value_changed = pyqtSignal()

    def __init__(self, prefix: str, title: str, parent):
        super().__init__(parent)
        self.prefix = prefix

        expander = ExpanderButton(title, self)

        self._preset = QComboBox(self)
        self._preset.addItems(SamplerPresets.instance().names())
        self._preset.setMinimumWidth(230)
        self._preset.currentIndexChanged.connect(self._select_preset)

        header_layout = QHBoxLayout()
        header_layout.addWidget(expander)
        header_layout.addStretch()
        header_layout.addWidget(self._preset)

        anchor = _("Edit custom presets")
        self._user_presets_link = QLabel(f"<a href='samplers.json'>{anchor}</a>", self)
        self._user_presets_link.linkActivated.connect(self._open_user_presets)

        self._sampler_info = QLabel("", self)

        info_layout = QHBoxLayout()
        info_layout.addWidget(self._sampler_info)
        info_layout.addStretch()
        info_layout.addWidget(self._user_presets_link)

        self._steps = SliderSetting(StyleSettings.sampler_steps, self, 1, 100, suffix=" steps")
        self._steps.soft_max = 50
        self._steps.value_changed.connect(self.notify_changed)

        self._cfg = SliderSetting(StyleSettings.cfg_scale, self, 1.0, 24.0, decimals=1)
        self._cfg.soft_max = 12.0
        self._cfg.value_changed.connect(self.notify_changed)

        extended_layout = QVBoxLayout()
        extended_layout.setContentsMargins(16, 2, 0, 2)
        extended_layout.addLayout(info_layout)
        extended_layout.addWidget(self._steps)
        extended_layout.addWidget(self._cfg)

        self._extended = QWidget(self)
        self._extended.setLayout(extended_layout)
        self._extended.setVisible(False)
        expander.toggled.connect(self._extended.setVisible)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addLayout(header_layout)
        layout.addWidget(self._extended)
        self.setLayout(layout)

    @property
    def preset(self):
        name = self._preset.currentText()
        return SamplerPresets.instance()[name]

    def _select_preset(self, index: int):
        preset = self.preset
        self._steps.value = preset.steps
        self._cfg.value = preset.cfg
        self._update_info()
        self.notify_changed()

    def _update_info(self):
        preset = self.preset
        text = "<b>" + _("Sampler") + f":</b> {preset.sampler} / {preset.scheduler}"
        if preset.lora:
            text += f" +LoRA '{preset.lora}'"
        self._sampler_info.setText(text)

    def _open_user_presets(self):
        path = SamplerPresets.instance().write_stub()
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            # No associated application, open the folder instead
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    def notify_changed(self):
        self.value_changed.emit()

    def read(self, style: Style):
        self._preset.setCurrentText(getattr(style, f"{self.prefix}sampler"))
        self._steps.value = getattr(style, f"{self.prefix}sampler_steps")
        self._cfg.value = getattr(style, f"{self.prefix}cfg_scale")
        self._update_info()

    def write(self, style: Style):
        setattr(style, f"{self.prefix}sampler", self._preset.currentText())
        setattr(style, f"{self.prefix}sampler_steps", self._steps.value)
        setattr(style, f"{self.prefix}cfg_scale", self._cfg.value)


class StylePresets(SettingsTab):
    def __init__(self, server: Server):
        super().__init__(_("Style Presets"))
        self.server = server

        self._style_list = QComboBox(self)
        self._style_list.currentIndexChanged.connect(self._change_style)

        self._create_style_button = QToolButton(self)
        self._create_style_button.setIcon(Krita.instance().icon("list-add"))
        self._create_style_button.setToolTip(_("Create a new style"))
        self._create_style_button.clicked.connect(self._create_style)

        self._duplicate_style_button = QToolButton(self)
        self._duplicate_style_button.setIcon(Krita.instance().icon("duplicate"))
        self._duplicate_style_button.setToolTip(_("Duplicate the current style"))
        self._duplicate_style_button.clicked.connect(self._duplicate_style)

        self._delete_style_button = QToolButton(self)
        self._delete_style_button.setIcon(Krita.instance().icon("deletelayer"))
        self._delete_style_button.setToolTip(_("Delete the current style"))
        self._delete_style_button.clicked.connect(self._delete_style)

        self._refresh_button = QToolButton(self)
        self._refresh_button.setIcon(Krita.instance().icon("reload-preset"))
        self._refresh_button.setToolTip(_("Look for new style files"))
        self._refresh_button.clicked.connect(Styles.list().reload)

        self._open_folder_button = QToolButton(self)
        self._open_folder_button.setIcon(Krita.instance().icon("document-open"))
        self._open_folder_button.setToolTip(_("Open folder containing style files"))
        self._open_folder_button.clicked.connect(self._open_style_folder)

        self._builtin_message = QLabel(_("Built-in styles cannot be modified."), self)
        self._builtin_message.setStyleSheet(f"font-style: italic; color: {theme.highlight};")
        self._builtin_message.setVisible(False)
        self._builtin_copy = QLabel("<a href='copy'>Click to edit a copy</a>", self)
        self._builtin_copy.linkActivated.connect(self._duplicate_style)
        self._builtin_copy.setVisible(False)

        self._show_builtin_checkbox = QCheckBox(_("Show pre-installed styles"), self)
        self._show_builtin_checkbox.setChecked(settings.show_builtin_styles)
        self._show_builtin_checkbox.toggled.connect(self.write)

        style_control_layout = QHBoxLayout()
        style_control_layout.setContentsMargins(0, 0, 0, 0)
        style_control_layout.addWidget(self._style_list)
        style_control_layout.addWidget(self._create_style_button)
        style_control_layout.addWidget(self._duplicate_style_button)
        style_control_layout.addWidget(self._delete_style_button)
        style_control_layout.addWidget(self._refresh_button)
        style_control_layout.addWidget(self._open_folder_button)
        builtin_layout = QHBoxLayout()
        builtin_layout.setContentsMargins(6, 1, 1, 1)
        builtin_layout.addWidget(self._builtin_message)
        builtin_layout.addWidget(self._builtin_copy)
        builtin_layout.addStretch()
        builtin_layout.addWidget(self._show_builtin_checkbox)
        frame_layout = QVBoxLayout()
        frame_layout.addLayout(style_control_layout)
        frame_layout.addLayout(builtin_layout)

        frame = QFrame(self)
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setLineWidth(1)
        frame.setLayout(frame_layout)
        self._layout.addWidget(frame)

        self._style_widgets: dict[str, SettingWidgetBase] = {}

        def add(name: str, widget: SettingWidgetBase | SettingWidget):
            self._style_widgets[name] = widget
            self._layout.addWidget(widget)
            widget.value_changed.connect(self.write)
            return widget

        add("name", TextSetting(StyleSettings.name, self))
        self._style_widgets["name"].value_changed.connect(self._update_name)

        checkpoints = FileFilter(root.files.checkpoints)
        checkpoints.available_only = True
        self._checkpoint_select = ComboBoxSetting(
            StyleSettings.checkpoints, model=checkpoints, parent=self
        )
        self._checkpoint_select.value_changed.connect(self.write)
        self._checkpoint_select.add_button(
            Krita.instance().icon("reload-preset"),
            _("Look for new checkpoint files"),
            root.connection.refresh,
        )
        self._layout.addWidget(self._checkpoint_select)

        self._checkpoint_warning = QLabel(self)
        self._checkpoint_warning.setStyleSheet(f"font-style: italic; color: {theme.yellow};")
        self._checkpoint_warning.setVisible(False)
        self._layout.addWidget(self._checkpoint_warning, alignment=Qt.AlignmentFlag.AlignRight)

        checkpoint_advanced = ExpanderButton(_("Checkpoint configuration (advanced)"), self)
        checkpoint_advanced.toggled.connect(self._toggle_checkpoint_advanced)
        self._layout.addWidget(checkpoint_advanced)

        self._arch_select = add(
            "architecture", ComboBoxSetting(StyleSettings.architecture, parent=self)
        )
        self._vae = add("vae", ComboBoxSetting(StyleSettings.vae, parent=self))

        self._clip_skip = self._style_widgets["clip_skip"] = SpinBoxSetting(
            StyleSettings.clip_skip, self, 0, 12
        )
        self._clip_skip.value_changed.connect(self.write)
        self._layout.addWidget(self._clip_skip)
        self._clip_skip_check = self._clip_skip.add_checkbox(_("Override"))
        self._clip_skip_check.toggled.connect(self._toggle_clip_skip)

        self._resolution_spin = self._style_widgets["preferred_resolution"] = SpinBoxSetting(
            StyleSettings.preferred_resolution, self, 0, 2048, step=8
        )
        self._resolution_spin.value_changed.connect(self.write)
        self._layout.addWidget(self._resolution_spin)
        resolution_check = self._resolution_spin.add_checkbox(_("Override"))
        resolution_check.toggled.connect(self._toggle_preferred_resolution)

        self._zsnr = add(
            "v_prediction_zsnr", SwitchSetting(StyleSettings.v_prediction_zsnr, parent=self)
        )

        self._sag = add(
            "self_attention_guidance",
            SwitchSetting(StyleSettings.self_attention_guidance, parent=self),
        )

        self._checkpoint_advanced_widgets = [
            self._arch_select,
            self._vae,
            self._clip_skip,
            self._resolution_spin,
            self._zsnr,
            self._sag,
        ]
        for widget in self._checkpoint_advanced_widgets:
            widget.indent = 1
        self._toggle_checkpoint_advanced(False)

        self._loras = LoraList(StyleSettings.loras, self)
        self._layout.addWidget(self._loras)
        self._loras.value_changed.connect(self.write)

        add("style_prompt", LineEditSetting(StyleSettings.style_prompt, self))
        add("negative_prompt", LineEditSetting(StyleSettings.negative_prompt, self))

        sdesc = _("Configure sampler type, steps and CFG to tweak the quality of generated images.")
        add_header(self._layout, Setting(_("Sampler Settings"), "", sdesc))

        self._default_sampler = SamplerWidget("", _("Quality Preset (generate and upscale)"), self)
        self._default_sampler.value_changed.connect(self.write)
        self._layout.addWidget(self._default_sampler)

        self._live_sampler = SamplerWidget("live_", _("Performance Preset (live mode)"), self)
        self._live_sampler.value_changed.connect(self.write)
        self._layout.addWidget(self._live_sampler)

        self._layout.addStretch()

        if settings.server_mode is ServerMode.managed:
            self._checkpoint_select.add_button(
                Krita.instance().icon("document-open"),
                _("Open the folder where checkpoints are stored"),
                self._open_checkpoints_folder,
            )
        self._populate_style_list()
        Styles.list().changed.connect(self._update_style_list)

    @property
    def current_style(self) -> Style:
        styles = Styles.list()
        return styles.find(self._style_list.currentData()) or styles.default

    @current_style.setter
    def current_style(self, style: Style):
        index = self._style_list.findData(style.filename)
        if index >= 0:
            self._style_list.setCurrentIndex(index)
            self._read_style(style)

    def update_model_lists(self):
        with self._write_guard:
            self._read()

    def _create_style(self):
        cp = self._checkpoint_select.value
        new_style = Styles.list().create(checkpoint=str(cp))
        self.current_style = new_style

    def _duplicate_style(self):
        self.current_style = Styles.list().create(
            self.current_style.filename, copy_from=self.current_style
        )

    def _delete_style(self):
        Styles.list().delete(self.current_style)

    def _open_style_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Styles.list().user_folder)))

    def _populate_style_list(self):
        for style in Styles.list().filtered():
            self._style_list.addItem(f"{style.name} ({style.filename})", style.filename)

    def _update_style_list(self):
        previous = None
        with SignalBlocker(self._style_list):
            if self._style_list.count() > 0:
                previous = self._style_list.currentData()
                self._style_list.clear()
            self._populate_style_list()
            if previous is not None:
                i = self._style_list.findData(previous)
                self._style_list.setCurrentIndex(max(0, i))
        self._change_style()

    def _update_name(self):
        index = self._style_list.currentIndex()
        style = self.current_style
        self._style_list.setItemText(index, f"{style.name} ({style.filename})")
        Styles.list().name_changed.emit()

    def _change_style(self):
        self._read_style(self.current_style)

    def _open_checkpoints_folder(self):
        arch = resolve_arch(self.current_style, root.connection.client_if_connected)
        if arch.is_sdxl_like or arch is Arch.sd15:
            self._open_folder(Path("models/checkpoints"))
        else:
            self._open_folder(Path("models/diffusion_models"))

    def _open_folder(self, subfolder: Path):
        if self.server.comfy_dir is not None:
            folder = self.server.path / subfolder
            folder.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _set_checkpoint_warning(self):
        self._checkpoint_warning.setVisible(False)
        if client := root.connection.client_if_connected:
            warn = []
            preferred_cp = self.current_style.preferred_checkpoint(client.models.checkpoints.keys())
            file = root.files.checkpoints.find(preferred_cp)
            if file is None:
                warn.append(_("The checkpoint used by this style is not installed."))

            arch = resolve_arch(self.current_style, client)
            if file and not client.supports_arch(arch):
                warn.append(
                    _(
                        "This is a {version} checkpoint, but the {version} workload has not been installed.",
                        version=arch.value,
                    )
                )

            if file and file.format is FileFormat.diffusion:
                vae_id = ResourceId(ResourceKind.vae, arch, "default")
                if client.models.resources.get(vae_id.string) is None:
                    paths = search_paths.get(vae_id.string, [])
                    text = _("The VAE for this diffusion model is not installed")
                    text += ": " + ", ".join(str(p) for p in paths)
                    warn.append(text)
                for te in arch.text_encoders:
                    te_id = ResourceId(ResourceKind.text_encoder, Arch.all, te)
                    if client.models.resources.get(te_id.string) is None:
                        paths = search_paths.get(te_id.string, [])
                        text = _("The text encoder for this diffusion model is not installed")
                        text += ": " + ", ".join(str(p) for p in paths)
                        warn.append(text)
            if warn:
                self._checkpoint_warning.setText("\n".join(warn))
                self._checkpoint_warning.setVisible(True)

    def _read_checkpoint(self, style: Style):
        if client := root.connection.client_if_connected:
            checkpoint = style.preferred_checkpoint(client.models.checkpoints.keys())
            self._checkpoint_select.value = checkpoint
        elif style.checkpoints:
            self._checkpoint_select.value = style.checkpoints[0]
        self._set_checkpoint_warning()

    def _write_checkpoint(self, style: Style):
        value = self._checkpoint_select.value
        if isinstance(value, str) and value != "":
            style.checkpoints = [value]
        self._set_checkpoint_warning()

    def _toggle_preferred_resolution(self, checked: bool):
        if checked and self._resolution_spin.value == 0:
            sd_ver = resolve_arch(self.current_style, root.connection.client_if_connected)
            self._resolution_spin.value = 640 if sd_ver is Arch.sd15 else 1024
        elif not checked and self._resolution_spin.value > 0:
            self._resolution_spin.value = 0

    def _toggle_clip_skip(self, checked: bool):
        if checked and self._clip_skip.value == 0:
            arch = resolve_arch(self.current_style, root.connection.client_if_connected)
            self._clip_skip.value = 1 if arch is Arch.sd15 else 2
        elif not checked and self._clip_skip.value > 0:
            self._clip_skip.value = 0

    def _toggle_checkpoint_advanced(self, checked: bool):
        for widget in self._checkpoint_advanced_widgets:
            widget.visible = checked

    def _show_builtin_info(self, style: Style):
        is_builtin = Styles.list().is_builtin(style)
        if self._builtin_message.isVisible() != is_builtin:
            self._builtin_message.setVisible(is_builtin)
            self._builtin_copy.setVisible(is_builtin)
            self._checkpoint_select.setEnabled(not is_builtin)
            self._loras.setEnabled(not is_builtin)
            for widget in self._style_widgets.values():
                widget.setEnabled(not is_builtin)
            for widget in self._checkpoint_advanced_widgets:
                widget.setEnabled(not is_builtin)
            self._default_sampler.setEnabled(not is_builtin)
            self._live_sampler.setEnabled(not is_builtin)

    def _enable_checkpoint_advanced(self):
        arch = resolve_arch(self.current_style, root.connection.client_if_connected)
        self._loras.set_style_arch(arch)
        if arch.is_sdxl_like:
            valid_archs = (Arch.auto, Arch.sdxl, Arch.illu, Arch.illu_v)
        elif arch.is_flux_like:
            valid_archs = (Arch.auto, Arch.flux, Arch.flux_k)
        elif arch.is_qwen_like:
            valid_archs = (Arch.auto, Arch.qwen, Arch.qwen_e, Arch.qwen_e_p)
        elif arch.is_flux2:
            valid_archs = (Arch.auto, Arch.flux2_4b, Arch.flux2_9b)
        else:
            valid_archs = (Arch.auto, arch)
        with SignalBlocker(self._arch_select):
            self._arch_select.set_items([(e.value, e.name) for e in valid_archs])
            if self.current_style.architecture in valid_archs:
                self._arch_select.value = self.current_style.architecture
        self._clip_skip_check.setEnabled(arch.supports_clip_skip)
        self._clip_skip.enabled = arch.supports_clip_skip and self.current_style.clip_skip > 0
        self._zsnr.enabled = arch.supports_attention_guidance
        self._sag.enabled = arch.supports_attention_guidance

    def _read_style(self, style: Style):
        with self._write_guard:
            for name, widget in self._style_widgets.items():
                widget.value = getattr(style, name)
            self._loras.value = style.loras  # type: ignore
            self._default_sampler.read(style)
            self._live_sampler.read(style)
        self._show_builtin_info(style)
        self._read_checkpoint(style)
        self._enable_checkpoint_advanced()
        self._resolution_spin.enabled = style.preferred_resolution > 0

    def _read(self):
        self._show_builtin_checkbox.setChecked(settings.show_builtin_styles)
        if client := root.connection.client_if_connected:
            default_vae = cast(str, StyleSettings.vae.default)
            self._style_widgets["vae"].set_items([default_vae] + client.models.vae)
        self._read_style(self.current_style)

    def _write(self):
        if settings.show_builtin_styles != self._show_builtin_checkbox.isChecked():
            settings.show_builtin_styles = self._show_builtin_checkbox.isChecked()
        style = self.current_style
        for name, widget in self._style_widgets.items():
            if widget.value is not None:
                setattr(style, name, widget.value)
        style.loras = self._loras.value
        self._write_checkpoint(style)
        self._default_sampler.write(style)
        self._live_sampler.write(style)
        self._enable_checkpoint_advanced()
        style.save()
