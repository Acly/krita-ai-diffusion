from __future__ import annotations

from typing import Optional, cast
from pathlib import Path
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
    QFrame,
    QLabel,
    QSpinBox,
    QToolButton,
    QComboBox,
    QWidget,
    QCompleter,
    QFileDialog,
    QMessageBox,
    QLineEdit,
)
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QPalette, QColor
from krita import Krita

from ..client import resolve_arch
from ..resources import Arch, ResourceId, ResourceKind, search_paths
from ..settings import Setting, ServerMode, settings
from ..server import Server
from ..files import File, FileFilter, FileSource, FileFormat
from ..style import Style, Styles, StyleSettings, SamplerPresets
from ..localization import translate as _
from ..root import root
from .settings_widgets import ExpanderButton, SpinBoxSetting, SliderSetting, SwitchSetting
from .settings_widgets import ComboBoxSetting, TextSetting, LineEditSetting, SettingWidget
from .settings_widgets import SettingsTab, WarningIcon
from .widget import create_framed_label
from .theme import SignalBlocker, add_header, icon
from .switch import SwitchWidget
from . import theme


class LoraItem(QWidget):
    changed = pyqtSignal()
    removed = pyqtSignal(QWidget)

    def __init__(self, name_filter: str, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        self._loras = FileFilter(root.files.loras)
        self._loras.available_only = True
        self._loras.name_prefix = name_filter
        self._current: File | None = None

        completer = QCompleter(self._loras)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)

        small_font = self.font()
        small_font.setPointSize(small_font.pointSize() - 1)

        grey_text = self.palette()
        grey_text.setColor(QPalette.ColorRole.Foreground, QColor(theme.grey))

        self._advanced_button = ExpanderButton(parent=self)
        self._advanced_button.toggled.connect(self._expand)

        self._select = QComboBox(self)
        self._select.setEditable(True)
        self._select.setModel(self._loras)
        self._select.setCompleter(completer)
        self._select.setMaxVisibleItems(20)
        self._select.setMinimumWidth(200)
        self._select.currentIndexChanged.connect(self._select_lora)

        expander_layout = QHBoxLayout()
        expander_layout.setContentsMargins(0, 0, 0, 0)
        expander_layout.setSpacing(0)
        expander_layout.addWidget(self._advanced_button)
        expander_layout.addWidget(self._select)

        self._warning_icon = WarningIcon(self)

        self._enabled = SwitchWidget(self)
        self._enabled.setChecked(True)
        self._enabled.toggled.connect(self._notify_changed)

        self._strength = QSpinBox(self)
        self._strength.setMinimum(-400)
        self._strength.setMaximum(400)
        self._strength.setSingleStep(5)
        self._strength.setValue(100)
        self._strength.setPrefix(_("Strength") + ": ")
        self._strength.setSuffix("%")
        self._strength.valueChanged.connect(self._notify_changed)

        self._remove = QToolButton(self)
        self._remove.setIcon(icon("discard"))
        self._remove.clicked.connect(self.remove)

        item_layout = QHBoxLayout()
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.addLayout(expander_layout, 3)
        item_layout.addWidget(self._enabled)
        item_layout.addWidget(self._strength, 1)
        item_layout.addWidget(self._warning_icon)
        item_layout.addWidget(self._remove)

        self._advanced = QWidget(self)
        self._advanced.setVisible(False)

        self._warning_text = QLabel(self._advanced)
        self._warning_text.setStyleSheet(f"color: {theme.yellow}; font-weight: bold;")
        self._warning_text.setVisible(False)

        trigger_label = QLabel(_("Trigger words"), parent=self._advanced)
        trigger_help = _("Optional text which is added to the prompt when the LoRA is used")
        self._trigger_edit = QLineEdit(parent=self._advanced)
        self._trigger_edit.setPlaceholderText(trigger_help)
        self._trigger_edit.textChanged.connect(self._set_triggers)

        trigger_layout = QVBoxLayout()
        trigger_layout.addWidget(trigger_label)
        trigger_layout.addWidget(self._trigger_edit)

        default_label = QLabel(_("Default Strength"), parent=self._advanced)
        default_frame, self._default_strength_value = create_framed_label("100%", self._advanced)
        self._default_strength_button = QPushButton(_("Set Default"), parent=self._advanced)
        self._default_strength_button.clicked.connect(self._set_default_strength)

        default_strength_frame_layout = QHBoxLayout()
        default_strength_frame_layout.setContentsMargins(0, 0, 0, 0)
        default_strength_frame_layout.addWidget(default_frame)
        default_strength_frame_layout.addWidget(self._default_strength_button)
        default_strength_layout = QVBoxLayout()
        default_strength_layout.addWidget(default_label)
        default_strength_layout.addLayout(default_strength_frame_layout)

        meta_layout = QHBoxLayout()
        meta_layout.addLayout(trigger_layout, 3)
        meta_layout.addLayout(default_strength_layout, 1)

        self._file_id_label = QLabel(parent=self._advanced)
        self._file_id_label.setFont(small_font)
        self._file_id_label.setPalette(grey_text)

        self._file_path_label = QLabel(parent=self._advanced)
        self._file_path_label.setFont(small_font)
        self._file_path_label.setPalette(grey_text)

        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(3, 2, 0, 2)
        advanced_layout.addWidget(self._warning_text)
        advanced_layout.addLayout(meta_layout)
        advanced_layout.addWidget(self._file_id_label)
        advanced_layout.addWidget(self._file_path_label)

        line = QFrame(self)
        line.setObjectName("LeftIndent")
        line.setStyleSheet(f"#LeftIndent {{ color: {theme.line};  }}")
        line.setFrameShape(QFrame.Shape.VLine)
        line.setLineWidth(1)

        pad_layout = QHBoxLayout()
        pad_layout.setContentsMargins(7, 0, 34, 10)
        pad_layout.addWidget(line)
        pad_layout.addLayout(advanced_layout)
        self._advanced.setLayout(pad_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(item_layout)
        layout.addWidget(self._advanced)
        self.setLayout(layout)

        if self._loras.rowCount() > 0:
            self._select_lora()

    def _expand(self):
        self._advanced.setVisible(self._advanced_button.isChecked())

    def _notify_changed(self):
        self._update()
        self.changed.emit()

    def _update(self):
        if self._current:
            self._file_id_label.setText(f"ID: {self._current.id}")
            if FileSource.local in self._current.source and self._current.path:
                path = str(self._current.path)
                if len(path) > 80:
                    path = "..." + path[-80:]
                self._file_path_label.setText(_("Local file") + f": {path}")
                self._file_path_label.setVisible(True)
            else:
                self._file_path_label.setVisible(False)
            if strength := self._current.meta("lora_strength"):
                istrength = int(strength * 100)
                self._default_strength_value.setText(f"{istrength}%")
                self._default_strength_button.setEnabled(istrength != self._strength.value())
            else:
                self._default_strength_value.setText("100%")
                self._default_strength_button.setEnabled(self._strength.value() != 100)
            self._trigger_edit.setText(self._current.meta("lora_triggers", ""))
            self._show_lora_warnings(self._current)

    def _select_lora(self):
        id = self._select.currentData()
        file = root.files.loras.find(id)
        if file and file != self._current:
            self._current = file
            default_strength = int(file.meta("lora_strength", 1.0) * 100)
            if default_strength != self._strength.value():
                self._strength.setValue(default_strength)
            if trigger_words := file.meta("lora_triggers", ""):
                self._trigger_edit.setText(trigger_words)
            self._notify_changed()

    def _set_triggers(self):
        value = self._trigger_edit.text()
        if self._current and self._current.meta("lora_triggers") != value:
            root.files.loras.set_meta(self._current, "lora_triggers", value)

    def _set_default_strength(self):
        if self._current and self._current.meta("lora_strength") != self.strength:
            root.files.loras.set_meta(self._current, "lora_strength", self.strength)
            self._update()

    def remove(self):
        self.removed.emit(self)

    @property
    def strength(self):
        return self._strength.value() / 100

    @strength.setter
    def strength(self, value: float):
        value_int = int(value * 100)
        if value_int != self._strength.value():
            self._strength.setValue(value_int)

    @property
    def value(self):
        if self._current is None:
            return dict(name="", strength=1.0, enabled=True)
        return dict(
            name=self._current.id, strength=self.strength, enabled=self._enabled.isChecked()
        )

    @value.setter
    def value(self, v: dict):
        new_value = root.files.loras.find(v["name"]) or File.remote(v["name"])
        if self._current is None or new_value.id != self._current.id:
            self._current = new_value
            index = self._select.findData(new_value.id)
            if index >= 0:
                self._select.setCurrentIndex(index)
            else:
                self._select.setEditText(self._current.name)
        self.strength = v["strength"]
        self._enabled.setChecked(v.get("enabled", True))
        self._update()

    def apply_filter(self, name_filter: str):
        with SignalBlocker(self._select):
            self._loras.name_prefix = name_filter
        if self._current and self._current.id != self._select.currentData():
            self._select.setEditText(self._current.name)

    def _show_lora_warnings(self, lora: File):
        if client := root.connection.client_if_connected:
            special_loras = [
                file
                for res, file in client.models.resources.items()
                if file is not None and res.startswith("lora-")
            ]
            file_ref = root.files.loras.find(lora.id)
            if file_ref is None or file_ref.source is FileSource.unavailable:
                self._warning_icon.show_message(_lora_not_installed_warning)
                self._warning_text.setText(_lora_not_installed_warning)
                self._warning_text.setVisible(True)
            elif lora.id in special_loras:
                self._warning_icon.show_message(_special_lora_warning)
                self._warning_text.setText(_special_lora_warning)
                self._warning_text.setVisible(True)
            else:
                self._warning_icon.hide()
                self._warning_text.setVisible(False)


_lora_not_installed_warning = _("The LoRA file is not installed on the server.")
_special_lora_warning = _(
    "This LoRA is usually added automatically by a Sampler or Control Layer when needed.\nIt is not required to add it manually here."
)


class LoraList(QWidget):
    value_changed = pyqtSignal()

    open_folder_button: Optional[QToolButton] = None
    last_filter = "All"

    _items: list[LoraItem]

    def __init__(self, setting: Setting, parent=None):
        super().__init__(parent)
        self._items = []

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_text_layout = QVBoxLayout()
        add_header(header_text_layout, setting)
        header_layout.addLayout(header_text_layout, 5)

        self._add_button = QPushButton(_("Add"), self)
        self._add_button.setMinimumWidth(100)
        self._add_button.clicked.connect(self._add_item)
        header_layout.addWidget(self._add_button, 1)

        self._upload_button = QPushButton(icon("upload"), "  " + _("Upload"), self)
        self._upload_button.setToolTip(_("Import a LoRA file from your local system"))
        self._upload_button.clicked.connect(self._upload_lora)
        header_layout.addWidget(self._upload_button, 1)

        self._filter_combo = QComboBox(self)
        self._filter_combo.currentIndexChanged.connect(self._set_filtered_names)
        header_layout.addWidget(self._filter_combo, 2)

        self._refresh_button = QToolButton(self)
        self._refresh_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._refresh_button.setIcon(Krita.instance().icon("reload-preset"))
        self._refresh_button.setToolTip(_("Look for new LoRA files"))
        self._refresh_button.clicked.connect(root.connection.refresh)
        header_layout.addWidget(self._refresh_button, 0)

        if settings.server_mode is ServerMode.managed:
            open_folder = self.open_folder_button = QToolButton(self)
            open_folder.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            open_folder.setIcon(Krita.instance().icon("document-open"))
            open_folder.setToolTip(_("Open folder containing LoRA files"))
            header_layout.addWidget(open_folder, 0)

        self._layout.addLayout(header_layout)

        self._item_list = QVBoxLayout()
        self._item_list.setContentsMargins(0, 0, 0, 0)
        self._layout.addLayout(self._item_list)

        root.files.loras.rowsInserted.connect(self._collect_filters)
        root.files.loras.rowsRemoved.connect(self._collect_filters)
        self._collect_filters()

    def _add_item(self, lora: dict | File | None = None):
        assert self._item_list is not None
        item = LoraItem(self.filter_prefix, parent=self)
        if isinstance(lora, dict):
            item.value = lora
        elif isinstance(lora, File):
            item.value = dict(name=lora.id, strength=1.0)
        item.changed.connect(self._update_item)
        item.removed.connect(self._remove_item)
        self._items.append(item)
        self._item_list.addWidget(item)
        self.value_changed.emit()

    def _remove_item(self, item: QWidget):
        self._items.remove(item)
        self._item_list.removeWidget(item)
        item.deleteLater()
        self.value_changed.emit()

    def _update_item(self):
        self.value_changed.emit()

    def _collect_filters(self):
        with SignalBlocker(self._filter_combo):
            self._filter_combo.clear()
            self._filter_combo.addItem(icon("filter"), "All")
            folders = set()
            for lora in root.files.loras:
                if lora.source is not FileSource.unavailable:
                    parts = Path(lora.id).parts
                    for i in range(1, len(parts)):
                        folders.add("/".join(parts[:i]))
            folder_icon = Krita.instance().icon("document-open")
            for folder in sorted(folders, key=lambda x: x.lower()):
                self._filter_combo.addItem(folder_icon, folder)
        self._filter_combo.setCurrentText(LoraList.last_filter)
        self._add_button.setEnabled(root.files.loras.rowCount() > 0)

    def _set_filtered_names(self):
        LoraList.last_filter = self.filter
        for item in self._items:
            item.apply_filter(self.filter_prefix)

    def _upload_lora(self):
        filepath = QFileDialog.getOpenFileName(
            self, _("Select LoRA file"), None, "LoRA files (*.safetensors)"
        )
        if filepath[0]:
            path = Path(filepath[0])
            if client := root.connection.client_if_connected:
                max_size = client.features.max_upload_size
                if max_size and path.stat().st_size > max_size:
                    _show_file_too_large_warning(max_size, self)
                    return
            file = File.local(path, FileFormat.lora, compute_hash=True)
            root.files.loras.add(file)
            self._add_item(file)

    @property
    def filter(self):
        return self._filter_combo.currentText()

    @property
    def filter_prefix(self):
        return self.filter if self.filter != "All" else ""

    @property
    def value(self):
        return [item.value for item in self._items]

    @value.setter
    def value(self, v):
        while not len(self._items) == 0:
            self._remove_item(self._items[-1])
        for lora in v:
            self._add_item(lora)


def _show_file_too_large_warning(max_size: int, parent=None):
    msg = _("The file is too large to be uploaded. Files up to a size of {size} MB are supported.")
    QMessageBox.warning(parent, _("File too large"), msg.format(size=max_size / (1024**2)))


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

        self._steps = SliderSetting(StyleSettings.sampler_steps, self, 1, 100)
        self._steps.value_changed.connect(self.notify_changed)

        self._cfg = SliderSetting(StyleSettings.cfg_scale, self, 1.0, 20.0)
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
    _checkpoint_advanced_widgets: list[SettingWidget]
    _default_sampler_widgets: list[SettingWidget]
    _live_sampler_widgets: list[SettingWidget]

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
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setLineWidth(1)
        frame.setLayout(frame_layout)
        self._layout.addWidget(frame)

        self._style_widgets: dict[str, SettingWidget] = {}

        def add(name: str, widget: SettingWidget):
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

        self._arch_select: ComboBoxSetting = add(
            "architecture", ComboBoxSetting(StyleSettings.architecture, parent=self)
        )
        self._vae = add("vae", ComboBoxSetting(StyleSettings.vae, parent=self))

        self._clip_skip = add("clip_skip", SpinBoxSetting(StyleSettings.clip_skip, self, 0, 12))
        self._clip_skip_check = self._clip_skip.add_checkbox(_("Override"))
        self._clip_skip_check.toggled.connect(self._toggle_clip_skip)

        self._resolution_spin = add(
            "preferred_resolution",
            SpinBoxSetting(StyleSettings.preferred_resolution, self, 0, 2048, step=8),
        )
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

        add("loras", LoraList(StyleSettings.loras, self))
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
        if self._style_widgets["loras"].open_folder_button:
            self._style_widgets["loras"].open_folder_button.clicked.connect(self._open_lora_folder)

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
        self._open_folder(Path("models/checkpoints"))

    def _open_lora_folder(self):
        self._open_folder(Path("models/loras"))

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
            for widget in self._style_widgets.values():
                widget.setEnabled(not is_builtin)
            for widget in self._checkpoint_advanced_widgets:
                widget.setEnabled(not is_builtin)
            self._default_sampler.setEnabled(not is_builtin)
            self._live_sampler.setEnabled(not is_builtin)

    def _enable_checkpoint_advanced(self):
        arch = resolve_arch(self.current_style, root.connection.client_if_connected)
        if arch.is_sdxl_like:
            valid_archs = (Arch.auto, Arch.sdxl, Arch.illu, Arch.illu_v)
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
        self._write_checkpoint(style)
        self._default_sampler.write(style)
        self._live_sampler.write(style)
        self._enable_checkpoint_advanced()
        style.save()
