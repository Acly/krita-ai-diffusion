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
)
from PyQt5.QtCore import Qt, QUrl, QStringListModel, QSortFilterProxyModel, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from krita import Krita

from ..client import resolve_sd_version
from ..resources import SDVersion
from ..settings import Setting, ServerMode, settings
from ..server import Server
from ..text import LoraId
from ..style import Style, Styles, StyleSettings, SamplerPresets
from ..localization import translate as _
from ..root import root
from .settings_widgets import ExpanderButton, SpinBoxSetting, SliderSetting, SwitchSetting
from .settings_widgets import ComboBoxSetting, TextSetting, LineEditSetting, SettingWidget
from .settings_widgets import SettingsTab, WarningIcon
from .theme import SignalBlocker, add_header, icon, sd_version_icon, yellow


class LoraList(QWidget):

    class Item(QWidget):
        changed = pyqtSignal()
        removed = pyqtSignal(QWidget)

        _loras: list[LoraId]
        _current = LoraId("", "")

        def __init__(self, loras: list[LoraId], filter: str, parent=None):
            super().__init__(parent)
            self.setContentsMargins(0, 0, 0, 0)

            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            self.setLayout(layout)

            self._select = QComboBox(self)
            self._select.setEditable(True)
            self._select.setMaxVisibleItems(20)
            self._select.currentIndexChanged.connect(self._select_lora)

            self._warning_icon = WarningIcon(self)

            self._strength = QSpinBox(self)
            self._strength.setMinimum(-400)
            self._strength.setMaximum(400)
            self._strength.setSingleStep(5)
            self._strength.setValue(100)
            self._strength.setPrefix(_("Strength") + ": ")
            self._strength.setSuffix("%")
            self._strength.valueChanged.connect(self._update)

            self._remove = QToolButton(self)
            self._remove.setIcon(icon("discard"))
            self._remove.clicked.connect(self.remove)

            layout.addWidget(self._select, 3)
            layout.addWidget(self._strength, 1)
            layout.addWidget(self._warning_icon)
            layout.addWidget(self._remove)

            self.set_names(loras, filter)

        def _update(self):
            self.changed.emit()

        def _select_lora(self):
            name = self._select.currentText()
            id = next((l for l in self._loras if l.name == name), LoraId("", ""))
            if id.file and id != self._current:
                self._current = id
                self._update()
                self._show_lora_warnings(id)

        def set_names(self, loras: list[LoraId], filter: str):
            self._loras = loras

            with SignalBlocker(self._select):
                model = QStringListModel([l.name for l in loras])
                sorted = QSortFilterProxyModel()
                sorted.setSourceModel(model)
                sorted.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                if filter != "All":
                    sorted.setFilterFixedString(f"{filter}/")

                completer = QCompleter(sorted)
                completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                completer.setFilterMode(Qt.MatchFlag.MatchContains)

                self._select.setModel(sorted)
                self._select.setCompleter(completer)

                if not self._current.name:
                    self._select_lora()
                else:
                    self._select.setEditText(self._current.name)

        def remove(self):
            self.removed.emit(self)

        @property
        def value(self):
            return dict(name=self._current.file, strength=self._strength.value() / 100)

        @value.setter
        def value(self, v):
            new_value = LoraId.normalize(v["name"])
            if new_value.file != self._current.file:
                self._current = new_value
                if self._select.findText(new_value.name) >= 0:
                    self._select.setCurrentText(self._current.name)
                else:
                    self._select.setEditText(self._current.name)
                self._strength.setValue(int(v["strength"] * 100))
                self._show_lora_warnings(new_value)

        def _show_lora_warnings(self, id: LoraId):
            if client := root.connection.client_if_connected:
                special_loras = [
                    file
                    for res, file in client.models.resources.items()
                    if file is not None and res.startswith("lora-")
                ]
                if id.file not in client.models.loras:
                    self._warning_icon.show_message(
                        _("The LoRA file is not installed on the server.")
                    )
                elif id.file in special_loras:
                    self._warning_icon.show_message(
                        _(
                            "This LoRA is usually added automatically by a Sampler or Control Layer when needed.\nIt is not required to add it manually here."
                        )
                    )
                else:
                    self._warning_icon.hide()

    value_changed = pyqtSignal()

    open_folder_button: Optional[QToolButton] = None
    last_filter = "All"

    _loras: list[LoraId]
    _items: list[Item]

    def __init__(self, setting: Setting, parent=None):
        super().__init__(parent)
        self._loras = []
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

        self.setEnabled(settings.server_mode is not ServerMode.cloud)
        settings.changed.connect(self._handle_settings_change)

    def _add_item(self, lora=None):
        assert self._item_list is not None
        item = self.Item(self._loras, self.filter, self)
        if isinstance(lora, dict):
            item.value = lora
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

    def _collect_filters(self, names: list[str]):
        with SignalBlocker(self._filter_combo):
            self._filter_combo.clear()
            self._filter_combo.addItem(icon("filter"), "All")
            folders = set()
            for name in names:
                parts = Path(name.replace("\\", "/")).parts
                for i in range(1, len(parts)):
                    folders.add("/".join(parts[:i]))
            folder_icon = Krita.instance().icon("document-open")
            for folder in sorted(folders, key=lambda x: x.lower()):
                self._filter_combo.addItem(folder_icon, folder)
        self._filter_combo.setCurrentText(LoraList.last_filter)

    def _set_filtered_names(self):
        LoraList.last_filter = self.filter
        for item in self._items:
            item.set_names(self._loras, self.filter)

    def _handle_settings_change(self, key: str, value):
        if key == "server_mode":
            self.setEnabled(value is not ServerMode.cloud)

    @property
    def filter(self):
        return self._filter_combo.currentText()

    @property
    def names(self):
        return self._loras

    @names.setter
    def names(self, v: list[str]):
        self._loras = [LoraId.normalize(name) for name in v]
        self._collect_filters(v)
        self._set_filtered_names()

    @property
    def value(self):
        return [item.value for item in self._items]

    @value.setter
    def value(self, v):
        while not len(self._items) == 0:
            self._remove_item(self._items[-1])
        for lora in v:
            self._add_item(lora)


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

        self._show_builtin_checkbox = QCheckBox(_("Show pre-installed styles"), self)
        self._show_builtin_checkbox.toggled.connect(self.write)

        style_control_layout = QHBoxLayout()
        style_control_layout.setContentsMargins(0, 0, 0, 0)
        style_control_layout.addWidget(self._style_list)
        style_control_layout.addWidget(self._create_style_button)
        style_control_layout.addWidget(self._delete_style_button)
        style_control_layout.addWidget(self._refresh_button)
        style_control_layout.addWidget(self._open_folder_button)
        frame_layout = QVBoxLayout()
        frame_layout.addLayout(style_control_layout)
        frame_layout.addWidget(self._show_builtin_checkbox, alignment=Qt.AlignmentFlag.AlignRight)

        frame = QFrame(self)
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setLineWidth(1)
        frame.setLayout(frame_layout)
        self._layout.addWidget(frame)

        self._style_widgets = {}

        def add(name: str, widget: SettingWidget):
            self._style_widgets[name] = widget
            self._layout.addWidget(widget)
            widget.value_changed.connect(self.write)
            return widget

        add("name", TextSetting(StyleSettings.name, self))
        self._style_widgets["name"].value_changed.connect(self._update_name)

        add("sd_checkpoint", ComboBoxSetting(StyleSettings.sd_checkpoint, self))
        self._style_widgets["sd_checkpoint"].add_button(
            Krita.instance().icon("reload-preset"),
            _("Look for new checkpoint files"),
            root.connection.refresh,
        )
        self._checkpoint_warning = QLabel(self)
        self._checkpoint_warning.setStyleSheet(f"font-style: italic; color: {yellow};")
        self._checkpoint_warning.setVisible(False)
        self._layout.addWidget(self._checkpoint_warning, alignment=Qt.AlignmentFlag.AlignRight)

        checkpoint_advanced = ExpanderButton(_("Checkpoint configuration (advanced)"), self)
        checkpoint_advanced.toggled.connect(self._toggle_checkpoint_advanced)
        self._layout.addWidget(checkpoint_advanced)

        self._checkpoint_advanced_widgets = [add("vae", ComboBoxSetting(StyleSettings.vae, self))]

        self._clip_skip = add("clip_skip", SpinBoxSetting(StyleSettings.clip_skip, self, 0, 12))
        clip_skip_check = self._clip_skip.add_checkbox(_("Override"))
        clip_skip_check.toggled.connect(self._toggle_clip_skip)
        self._checkpoint_advanced_widgets.append(self._clip_skip)

        self._resolution_spin = add(
            "preferred_resolution",
            SpinBoxSetting(StyleSettings.preferred_resolution, self, 0, 2048, step=8),
        )
        resolution_check = self._resolution_spin.add_checkbox(_("Override"))
        resolution_check.toggled.connect(self._toggle_preferred_resolution)
        self._checkpoint_advanced_widgets.append(self._resolution_spin)

        self._checkpoint_advanced_widgets.append(
            add("v_prediction_zsnr", SwitchSetting(StyleSettings.v_prediction_zsnr, parent=self))
        )
        self._checkpoint_advanced_widgets.append(
            add(
                "self_attention_guidance",
                SwitchSetting(StyleSettings.self_attention_guidance, parent=self),
            )
        )
        self._checkpoint_advanced_widgets.append(
            add(
                "perturbed_attention_guidance",
                SwitchSetting(StyleSettings.perturbed_attention_guidance, parent=self),
            )
        )
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
            self._style_widgets["sd_checkpoint"].add_button(
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
        cp = self._style_widgets["sd_checkpoint"].value or StyleSettings.sd_checkpoint.default
        new_style = Styles.list().create(checkpoint=cp)
        self.current_style = new_style

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
        if self.server.comfy_dir is not None:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.server.comfy_dir / "models" / "checkpoints"))
            )

    def _open_lora_folder(self):
        if self.server.comfy_dir is not None:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.server.comfy_dir / "models" / "loras"))
            )

    def _set_checkpoint_warning(self):
        self._checkpoint_warning.setVisible(False)
        if client := root.connection.client_if_connected:
            if self.current_style.sd_checkpoint not in client.models.checkpoints:
                self._checkpoint_warning.setText(
                    _("The checkpoint used by this style is not installed.")
                )
                self._checkpoint_warning.setVisible(True)
            else:
                version = resolve_sd_version(self.current_style, client)
                if not client.supports_version(version):
                    self._checkpoint_warning.setText(
                        _(
                            "This is a {version} checkpoint, but the {version} workload has not been installed.",
                            version=version.value,
                        )
                    )
                    self._checkpoint_warning.setVisible(True)

    def _toggle_preferred_resolution(self, checked: bool):
        if checked and self._resolution_spin.value == 0:
            sd_ver = resolve_sd_version(self.current_style, root.connection.client_if_connected)
            self._resolution_spin.value = 640 if sd_ver is SDVersion.sd15 else 1024
        elif not checked and self._resolution_spin.value > 0:
            self._resolution_spin.value = 0

    def _toggle_clip_skip(self, checked: bool):
        if checked and self._clip_skip.value == 0:
            sd_ver = resolve_sd_version(self.current_style, root.connection.client_if_connected)
            self._clip_skip.value = 1 if sd_ver is SDVersion.sd15 else 2
        elif not checked and self._clip_skip.value > 0:
            self._clip_skip.value = 0

    def _toggle_checkpoint_advanced(self, checked: bool):
        for widget in self._checkpoint_advanced_widgets:
            widget.visible = checked

    def _read_style(self, style: Style):
        with self._write_guard:
            for name, widget in self._style_widgets.items():
                widget.value = getattr(style, name)
            self._default_sampler.read(style)
            self._live_sampler.read(style)
        self._set_checkpoint_warning()
        self._clip_skip.enabled = style.clip_skip > 0
        self._resolution_spin.enabled = style.preferred_resolution > 0

    def _read(self):
        self._show_builtin_checkbox.setChecked(settings.show_builtin_styles)
        if client := root.connection.client_if_connected:
            default_vae = cast(str, StyleSettings.vae.default)
            checkpoints = [
                (cp.name, cp.filename, sd_version_icon(cp.sd_version, client))
                for cp in client.models.checkpoints.values()
                if not (cp.is_refiner or cp.is_inpaint)
            ]
            self._style_widgets["sd_checkpoint"].set_items(checkpoints)
            self._style_widgets["loras"].names = client.models.loras
            self._style_widgets["vae"].set_items([default_vae] + client.models.vae)
        self._read_style(self.current_style)

    def _write(self):
        if settings.show_builtin_styles != self._show_builtin_checkbox.isChecked():
            settings.show_builtin_styles = self._show_builtin_checkbox.isChecked()
        style = self.current_style
        for name, widget in self._style_widgets.items():
            if widget.value is not None:
                setattr(style, name, widget.value)
        self._default_sampler.write(style)
        self._live_sampler.write(style)
        self._set_checkpoint_warning()
        style.save()
