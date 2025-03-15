from __future__ import annotations
from krita import Krita

from typing import Optional
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QPushButton,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QSpinBox,
    QStackedWidget,
    QRadioButton,
    QComboBox,
    QWidget,
    QMessageBox,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QMetaObject, QSize, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QGuiApplication, QCursor, QFontMetrics

from ..client import Client, User, MissingResources
from ..cloud_client import CloudClient
from ..resources import Arch, ResourceId
from ..settings import Settings, ServerMode, PerformancePreset, settings
from ..server import Server
from ..style import Style
from ..root import root
from ..connection import ConnectionState, apply_performance_preset
from ..updates import UpdateState
from ..properties import Binding
from ..localization import Localization, translate as _
from .. import resources, eventloop, util, __version__
from .server import ServerWidget
from .settings_widgets import SpinBoxSetting, SliderSetting, SwitchSetting
from .settings_widgets import SettingsTab, ComboBoxSetting, FileListSetting
from .style import StylePresets
from .theme import add_header, logo, red, yellow, green, grey


class UserWidget(QFrame):
    _user: User | None = None
    _connections: list[QMetaObject.Connection | Binding]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connections = []

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setVisible(False)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self._user_name = QLabel("", self)
        self._user_name.setStyleSheet("font-weight:bold")
        user_name_layout = QHBoxLayout()
        user_name_layout.addWidget(QLabel(_("Account:"), self), 0)
        user_name_layout.addWidget(self._user_name, 1)
        layout.addLayout(user_name_layout)

        self._images_generated = QLabel("", self)
        image_count_layout = QHBoxLayout()
        image_count_layout.addWidget(QLabel(_("Total generated:"), self), 0)
        image_count_layout.addWidget(self._images_generated, 1)
        layout.addLayout(image_count_layout)

        self._tokens_remaining = QLabel("", self)
        self._tokens_remaining.setStyleSheet("font-weight:bold")
        image_remaining_layout = QHBoxLayout()
        image_remaining_layout.addWidget(QLabel(_("Image tokens remaining:"), self), 0)
        image_remaining_layout.addWidget(self._tokens_remaining, 1)
        layout.addLayout(image_remaining_layout)

        self._logout_button = QPushButton(_("Sign out"), self)
        self._logout_button.setMinimumWidth(200)
        self._logout_button.clicked.connect(self._logout)
        layout.addWidget(self._logout_button)

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, user: User | None):
        if self._user is not user:
            Binding.disconnect_all(self._connections)
            self.setVisible(user is not None)

            self._user = user
            if user is not None:
                self._user_name.setText(user.name)
                self._connections = [
                    user.images_generated_changed.connect(self._update_counts),
                    user.credits_changed.connect(self._update_counts),
                ]
                self._update_counts()

    def _update_counts(self):
        user = util.ensure(self.user)
        self._images_generated.setText(str(user.images_generated))
        self._tokens_remaining.setText(str(user.credits))

    def _logout(self):
        eventloop.run(self._disconnect_and_logout())

    async def _disconnect_and_logout(self):
        await root.connection.disconnect()
        settings.access_token = ""
        settings.save()


class CloudWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 12, 4, 4)
        self.setLayout(layout)

        service_url = CloudClient.default_web_url
        service_url_text = service_url.removeprefix("https://").removesuffix("/")
        service_label = QLabel(f"<a href='{service_url}'>{service_url_text}</a>", self)
        service_label.setStyleSheet("font-size: 12pt")
        service_label.setTextFormat(Qt.TextFormat.RichText)
        service_label.setOpenExternalLinks(True)
        layout.addWidget(service_label)

        self._connection_status = QLabel(self)
        self._connection_status.setWordWrap(True)
        self._connection_status.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._connection_status)

        self.connect_button = QPushButton(_("Login"), self)
        self.connect_button.setMinimumWidth(200)
        self.connect_button.setMinimumHeight(int(1.3 * self.connect_button.sizeHint().height()))
        self.connect_button.clicked.connect(self._connect)

        self._sign_out_button = QPushButton(_("Sign out"), self)
        self._sign_out_button.setVisible(False)
        self._sign_out_button.setMinimumWidth(200)
        self._sign_out_button.clicked.connect(self._sign_out)

        self._user_widget = UserWidget(self)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.connect_button)
        buttons_layout.addWidget(self._sign_out_button)

        connect_layout = QHBoxLayout()
        connect_layout.addLayout(buttons_layout)
        connect_layout.addWidget(self._user_widget)
        connect_layout.addStretch()
        layout.addLayout(connect_layout)

        layout.addStretch()

    def update_connection_state(self, state: ConnectionState):
        is_connected = state == ConnectionState.connected
        self.connect_button.setVisible(not is_connected)
        self._sign_out_button.setVisible(False)
        self._user_widget.user = root.connection.user

        if state in [ConnectionState.auth_missing, ConnectionState.auth_error]:
            self.connect_button.setText(_("Sign in"))
            self.connect_button.setEnabled(True)
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
        elif state is ConnectionState.auth_pending:
            self.connect_button.setText(_("Sign in"))
            self.connect_button.setEnabled(False)
            self._connection_status.setText(_("Waiting for sign-in to complete..."))
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
            self._connection_status.setVisible(True)
        elif state is ConnectionState.connected:
            self._connection_status.setText(_("Connected"))
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
            self._user_widget.user = root.connection.user
        else:
            can_connect = state in [ConnectionState.disconnected, ConnectionState.error]
            self.connect_button.setEnabled(can_connect)
            self.connect_button.setText(_("Connect") if can_connect else _("Connected"))
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")

        if state in [ConnectionState.error, ConnectionState.auth_error]:
            error = root.connection.error or "Unknown error"
            self._connection_status.setText(f"<b>Error</b>: {error.removeprefix('Error: ')}")
            self._connection_status.setStyleSheet(f"color: {red}; font-weight:bold")
            self._connection_status.setVisible(True)
            if settings.access_token:
                self._sign_out_button.setVisible(True)

    def _connect(self):
        connection = root.connection
        if connection.state in [ConnectionState.auth_missing, ConnectionState.auth_error]:
            connection.sign_in()
        else:
            connection.connect()

    def _sign_out(self):
        settings.access_token = ""
        settings.save()


class ConnectionSettings(SettingsTab):
    def __init__(self, server: Server):
        super().__init__(_("Server Configuration"))

        self._server_cloud = QRadioButton(_("Online Service"), self)
        self._server_managed = QRadioButton(_("Local Managed Server"), self)
        self._server_external = QRadioButton(_("Custom Server (local or remote)"), self)
        info_cloud = QLabel(_("Generate images via GPU Cloud Service"), self)
        info_managed = QLabel(
            _("Let the Krita plugin install and run a local server on your machine"), self
        )
        info_external = QLabel(
            _("Connect to a running ComfyUI instance which you set up and maintain yourself"), self
        )
        for button in (self._server_cloud, self._server_managed, self._server_external):
            button.setStyleSheet("font-weight:bold")
            button.toggled.connect(self._change_server_mode)
        for label in (info_cloud, info_managed, info_external):
            label.setContentsMargins(20, 0, 0, 0)

        self._cloud_widget = CloudWidget(self)
        self._server_widget = ServerWidget(server, self)
        self._connection_widget = QWidget(self)
        self._server_stack = QStackedWidget(self)
        self._server_stack.addWidget(self._cloud_widget)
        self._server_stack.addWidget(self._server_widget)
        self._server_stack.addWidget(self._connection_widget)

        connection_layout = QVBoxLayout()
        self._connection_widget.setLayout(connection_layout)

        add_header(connection_layout, Settings._server_url)
        server_layout = QHBoxLayout()
        self._server_url = QLineEdit(self._connection_widget)
        self._server_url.textChanged.connect(self.write)
        server_layout.addWidget(self._server_url)
        self._connect_button = QPushButton(_("Connect"), self._connection_widget)
        self._connect_button.clicked.connect(self._connect)
        server_layout.addWidget(self._connect_button)
        connection_layout.addLayout(server_layout)

        self._connection_status = QLabel(self._connection_widget)
        self._supported_workloads = QLabel(self._connection_widget)
        self._supported_workloads.setWordWrap(True)
        self._supported_workloads.setTextFormat(Qt.TextFormat.RichText)
        self._supported_workloads.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self._supported_workloads.setOpenExternalLinks(True)

        anchor = _("View log files")
        open_log_button = QLabel(f"<a href='file://{util.log_dir}'>{anchor}</a>", self)
        open_log_button.setToolTip(str(util.log_dir))
        open_log_button.linkActivated.connect(self._open_logs)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self._connection_status)
        status_layout.addWidget(open_log_button, alignment=Qt.AlignmentFlag.AlignRight)

        connection_layout.addLayout(status_layout)
        connection_layout.addWidget(self._supported_workloads)
        connection_layout.addStretch()

        self._layout.addWidget(self._server_managed)
        self._layout.addWidget(info_managed)
        self._layout.addWidget(self._server_external)
        self._layout.addWidget(info_external)
        self._layout.addWidget(self._server_cloud)
        self._layout.addWidget(info_cloud)
        self._layout.addWidget(self._server_stack)

        root.connection.state_changed.connect(self.update_server_status)
        root.connection.error_changed.connect(self.update_server_status)
        self.update_server_status()

    @property
    def server_mode(self):
        if self._server_cloud.isChecked():
            return ServerMode.cloud
        elif self._server_managed.isChecked():
            return ServerMode.managed
        elif self._server_external.isChecked():
            return ServerMode.external
        else:
            return ServerMode.undefined

    @server_mode.setter
    def server_mode(self, mode: ServerMode):
        if self.server_mode != mode:
            self._server_cloud.setChecked(mode is ServerMode.cloud)
            self._server_managed.setChecked(mode is ServerMode.managed)
            self._server_external.setChecked(mode is ServerMode.external)
        widget = {
            ServerMode.cloud: self._cloud_widget,
            ServerMode.managed: self._server_widget,
            ServerMode.external: self._connection_widget,
            ServerMode.undefined: self._connection_widget,
        }[mode]
        self._server_stack.setCurrentWidget(widget)

    def update_ui(self):
        self._server_widget.update_ui()

    def _read(self):
        self.server_mode = settings.server_mode
        self._server_url.setText(settings.server_url)

    def _write(self):
        settings.server_mode = self.server_mode
        settings.server_url = self._server_url.text()

    def _change_server_mode(self, checked: bool):
        if self._server_cloud.isChecked():
            self.server_mode = ServerMode.cloud
        elif self._server_managed.isChecked():
            self.server_mode = ServerMode.managed
        elif self._server_external.isChecked():
            self.server_mode = ServerMode.external
        self.write()

    def _connect(self):
        root.connection.connect()

    def update_server_status(self):
        connection = root.connection
        self._cloud_widget.update_connection_state(connection.state)
        self._connect_button.setEnabled(connection.state != ConnectionState.connecting)
        if connection.state == ConnectionState.connected:
            self._connection_status.setText(_("Connected"))
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
        elif connection.state == ConnectionState.connecting:
            self._connection_status.setText(_("Connecting"))
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
        elif connection.state == ConnectionState.disconnected:
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
        elif connection.state == ConnectionState.error:
            msg = connection.error.removeprefix("Error: ") if connection.error else "Unknown error"
            self._connection_status.setText("<b>" + _("Error") + f"</b>: {msg}")
            self._connection_status.setStyleSheet(f"color: {red};")

        self._supported_workloads.clear()
        if connection.state in [ConnectionState.connected, ConnectionState.error]:
            if connection.missing_resources is not None:
                self._show_missing_resources(connection.missing_resources, connection.state)

    def _show_missing_resources(self, res: MissingResources, state: ConnectionState):
        def model_name(id: ResourceId, with_file=False):
            if res := resources.find_resource(id):
                if with_file:
                    return f"{res.name} ({', '.join(f.name for f in res.files)})"
                return res.name
            if isinstance(id.identifier, str):
                return id.identifier
            return f"{id.kind.value} {id.identifier.value}"

        text = ""
        if isinstance(res.missing, list):
            text = (
                _("The following ComfyUI custom nodes are missing")
                + ":<ul>"
                + "\n".join(
                    (f"<li>{p.name} <a href='{p.url}'>{p.url}</a></li>" for p in res.missing)
                )
                + "</ul>"
                + _(
                    "Please install or update the custom node package, then restart the server and try again."
                )
                + "<br>"
            )
        else:
            basic = [m for lst in res.missing.values() for m in lst if m.arch is Arch.all]
            basic = util.unique(basic, key=lambda m: m.string)
            if len(basic) > 0:
                text = _("Missing common models (required)") + ":\n<ul>"
                text += "\n".join((f"<li>{model_name(m, True)}</li>" for m in basic))
                text += "</ul>"
            text += _("Detected base models:") + "\n<ul>"
            for arch, missing in res.missing.items():
                if arch in [Arch.all, Arch.illu_v]:
                    continue
                text += f"<li><b>{arch.value}</b>: "
                if len(missing) == 0:
                    text += _("supported")
                else:
                    names = [model_name(m) for m in missing if m.arch is arch]
                    if len(names) > 0:
                        text += _("missing") + " " + ", ".join(names)
                    else:
                        text += _("models found")
                text += "</li>"
            text += "</ul>"

        link = "<a href='https://docs.interstice.cloud/comfyui-setup'>Custom ComfyUI Setup</a>"
        text += _(
            "See {link} for required models.<br>Check the client.log file for more details.",
            link=link,
        )
        style = "" if state is ConnectionState.error else f"color: {grey};"
        self._supported_workloads.setStyleSheet(style)
        self._supported_workloads.setText(text)

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_dir)))


class DiffusionSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Diffusion Settings"))

        S = Settings
        self.add("selection_grow", SliderSetting(S._selection_grow, self, 0, 25, "{} %"))
        self.add("selection_feather", SliderSetting(S._selection_feather, self, 0, 25, "{} %"))
        self.add("selection_padding", SliderSetting(S._selection_padding, self, 0, 25, "{} %"))
        self.add("nsfw_filter", ComboBoxSetting(S._nsfw_filter, parent=self))

        nsfw_settings = [(_("Disabled"), 0.0), (_("Basic"), 0.65), (_("Strict"), 0.8)]
        self._widgets["nsfw_filter"].set_items(nsfw_settings)
        DiffusionSettings._warning_shown = self._warning_shown or settings.nsfw_filter > 0

        self._layout.addStretch()

    _warning_shown = False

    def _write(self):
        if self._widgets["nsfw_filter"].value > 0 and not self._warning_shown:
            DiffusionSettings._warning_shown = True
            QMessageBox.warning(
                self,
                _("NSFW Filter Warning"),
                _(
                    "The NSFW filter is a basic tool to exclude explicit content from generated images. It is NOT a guarantee and may not catch all inappropriate content. Please use responsibly and always review the generated images."
                ),
            )


class InterfaceSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Interface Settings"))

        S = Settings
        self.add("language", ComboBoxSetting(S._language, parent=self))
        self.add("prompt_translation", ComboBoxSetting(S._prompt_translation, parent=self))
        self.add("prompt_line_count", SpinBoxSetting(S._prompt_line_count, self, 1, 10))
        self.add(
            "show_negative_prompt",
            SwitchSetting(S._show_negative_prompt, (_("Show"), _("Hide")), self),
        )
        self.add("show_steps", SwitchSetting(S._show_steps, parent=self))

        self.add("tag_files", FileListSetting(S._tag_files, files=self._tag_files(), parent=self))
        self._layout.addWidget(self._widgets["tag_files"].list_widget)
        self._widgets["tag_files"].add_button(
            Krita.instance().icon("reload-preset"),
            _("Look for new tag files"),
            self._update_tag_files,
        )
        self._widgets["tag_files"].add_button(
            Krita.instance().icon("document-open"),
            _("Open folder where custom tag files can be placed"),
            self._open_tag_folder,
        )

        self.add(
            "generation_finished_action",
            ComboBoxSetting(S._generation_finished_action, parent=self),
        )
        self.add("apply_behavior", ComboBoxSetting(S._apply_behavior, parent=self))
        self.add("apply_region_behavior", ComboBoxSetting(S._apply_region_behavior, parent=self))
        self.add("apply_behavior_live", ComboBoxSetting(S._apply_behavior_live, parent=self))
        self.add(
            "apply_region_behavior_live",
            ComboBoxSetting(S._apply_region_behavior_live, parent=self),
        )
        self.add("new_seed_after_apply", SwitchSetting(S._new_seed_after_apply, parent=self))
        self.add("debug_dump_workflow", SwitchSetting(S._debug_dump_workflow, parent=self))

        languages = [(lang.name, lang.id) for lang in Localization.available]
        self._widgets["language"].set_items(languages)
        self.update_translation(root.connection.client_if_connected)

        for w in ["apply_region_behavior", "apply_region_behavior_live"]:
            self._widgets[w].show_label = False

        self._layout.addStretch()

    def _tag_files(self) -> list[str]:
        plugin_tags_path = util.plugin_dir / "tags"
        user_tags_path = util.user_data_dir / "tags"
        files = set()
        for path in plugin_tags_path.glob("*.csv"):
            files.add(path.stem)
        for path in user_tags_path.glob("*.csv"):
            files.add(path.stem)

        return list(files)

    def _update_tag_files(self):
        self._widgets["tag_files"].reset_files(self._tag_files())

    def _open_tag_folder(self):
        user_tag_folder = util.user_data_dir / "tags"
        user_tag_folder.mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_tag_folder)))

    def update_translation(self, client: Client | None):
        translation: ComboBoxSetting = self._widgets["prompt_translation"]
        languages = [("Disabled", "")]
        if client:
            languages += [(lang.name, lang.code) for lang in client.features.languages]
        translation.enabled = client is not None
        translation.set_items(languages)
        self.read()


class HistorySizeWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, maximum: int, step: int, parent=None):
        super().__init__(parent)

        self._history_size = QSpinBox(self)
        self._history_size.setMinimum(5)
        self._history_size.setMaximum(maximum)
        self._history_size.setSingleStep(step)
        self._history_size.setSuffix(" MB")
        self._history_size.valueChanged.connect(self._change_value)

        self._history_usage = QLabel(self)
        self._history_usage.setStyleSheet(f"font-style:italic; color: {green};")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._history_size)
        layout.addWidget(self._history_usage)
        self.setLayout(layout)

    def _change_value(self):
        self.value_changed.emit()

    @property
    def value(self):
        return self._history_size.value()

    @value.setter
    def value(self, v):
        self._history_size.setValue(v)

    def update_usage(self, usage: float):
        self._history_usage.setText(_("Currently using") + f" {usage:.1f} MB")


class PerformanceSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Performance Settings"))

        add_header(self._layout, Settings._history_size)
        self._history_size = HistorySizeWidget(maximum=10000, step=100, parent=self)
        self._history_size.value_changed.connect(self.write)
        self._layout.addWidget(self._history_size)

        add_header(self._layout, Settings._history_storage)
        self._history_storage = HistorySizeWidget(maximum=100, step=5, parent=self)
        self._history_storage.value_changed.connect(self.write)
        self._layout.addWidget(self._history_storage)

        add_header(self._layout, Settings._performance_preset)
        self._device_info = QLabel(self)
        self._device_info.setStyleSheet("font-style:italic")
        self._layout.addWidget(self._device_info)

        self._performance_preset = QComboBox(self)
        for preset in PerformancePreset:
            self._performance_preset.addItem(preset.value)
        self._performance_preset.currentIndexChanged.connect(self._change_performance_preset)
        self._layout.addWidget(self._performance_preset, alignment=Qt.AlignmentFlag.AlignLeft)

        self._advanced = QWidget(self)
        self._advanced.setEnabled(settings.performance_preset is PerformancePreset.custom)
        self._advanced.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._advanced)
        advanced_layout = QVBoxLayout()
        self._advanced.setLayout(advanced_layout)

        self._batch_size = SliderSetting(Settings._batch_size, self._advanced, 1, 16)
        self._batch_size.value_changed.connect(self.write)
        advanced_layout.addWidget(self._batch_size)

        self._resolution_multiplier = SliderSetting(
            Settings._resolution_multiplier, self._advanced, 0.3, 1.5, "{:.1f}x"
        )
        self._resolution_multiplier.value_changed.connect(self.write)
        advanced_layout.addWidget(self._resolution_multiplier)

        self._max_pixel_count = SpinBoxSetting(
            Settings._max_pixel_count, self._advanced, 1, 99, 1, " MP"
        )
        self._max_pixel_count.value_changed.connect(self.write)
        advanced_layout.addWidget(self._max_pixel_count)

        self._tiled_vae = SwitchSetting(
            Settings._tiled_vae, text=(_("Always"), _("Automatic")), parent=self._advanced
        )
        self._tiled_vae.value_changed.connect(self.write)
        advanced_layout.addWidget(self._tiled_vae)

        self._dynamic_caching = SwitchSetting(Settings._dynamic_caching, parent=self)
        self._dynamic_caching.value_changed.connect(self.write)
        self._layout.addWidget(self._dynamic_caching)

        self._layout.addStretch()

    def _change_performance_preset(self, index):
        self.write()
        is_custom = settings.performance_preset is PerformancePreset.custom
        self._advanced.setEnabled(is_custom)
        if (
            settings.performance_preset is PerformancePreset.auto
            and root.connection.state is ConnectionState.connected
        ):
            apply_performance_preset(settings, root.connection.client.device_info)
        if not is_custom:
            self.read()

    def update_client_info(self):
        if root.connection.state is ConnectionState.connected:
            client = root.connection.client
            self._device_info.setText(
                _("Device")
                + f": [{client.device_info.type.upper()}] {client.device_info.name} ({client.device_info.vram} GB)"
            )
            self._dynamic_caching.enabled = client.features.wave_speed
            self._dynamic_caching.setToolTip(
                _("The {node_name} node is not installed.").format(node_name="Comfy-WaveSpeed")
                if not client.features.wave_speed
                else ""
            )

    def _read(self):
        self._history_size.value = settings.history_size
        self._history_size.update_usage(root.active_model.jobs.memory_usage)
        self._history_storage.value = settings.history_storage
        self._history_storage.update_usage(root.get_active_model_used_storage() / (1024**2))
        self._batch_size.value = settings.batch_size
        self._performance_preset.setCurrentIndex(
            list(PerformancePreset).index(settings.performance_preset)
        )
        self._resolution_multiplier.value = settings.resolution_multiplier
        self._max_pixel_count.value = settings.max_pixel_count
        self._tiled_vae.value = settings.tiled_vae
        self._dynamic_caching.value = settings.dynamic_caching
        self.update_client_info()

    def _write(self):
        settings.history_size = self._history_size.value
        settings.history_storage = self._history_storage.value
        settings.batch_size = int(self._batch_size.value)
        settings.resolution_multiplier = self._resolution_multiplier.value
        settings.max_pixel_count = self._max_pixel_count.value
        settings.tiled_vae = self._tiled_vae.value
        settings.performance_preset = list(PerformancePreset)[
            self._performance_preset.currentIndex()
        ]
        settings.dynamic_caching = self._dynamic_caching.value


class AboutSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Plugin Information and Updates"))

        large = self.font()
        large.setPointSize(large.pointSize() + 2)

        extra_large = self.font()
        extra_large.setPointSize(extra_large.pointSize() + 4)

        bold = self.font()
        bold.setBold(True)

        italic = self.font()
        italic.setItalic(True)

        header_layout = QHBoxLayout()
        header_logo = QLabel(self)
        font_height = QFontMetrics(extra_large).height() + 4
        header_logo.setPixmap(logo().scaled(font_height * 2, font_height * 2))
        header_logo.setMaximumSize(font_height * 2, font_height * 2)
        header_text = QLabel("Generative AI\nfor Krita", self)
        header_text.setFont(extra_large)
        header_layout.addWidget(header_logo)
        header_layout.addWidget(header_text)

        current_version_name = QLabel(_("Current version") + ":", self)
        current_version_value = QLabel(__version__, self)

        latest_version_name = QLabel(_("Latest version") + ":", self)
        self._latest_version_value = QLabel(self)
        self._latest_version_value.setFont(bold)

        self._update_error = QLabel(self)
        self._update_error.setFont(italic)

        self._update_checkbox = QCheckBox(_("Check for updates on startup"), self)
        self._update_checkbox.setChecked(settings.auto_update)
        self._update_checkbox.stateChanged.connect(self.write)

        self._check_button = QPushButton(_("Check for Updates"), self)
        self._check_button.setMinimumWidth(font_height * 6)
        self._check_button.clicked.connect(self._check_updates)

        self._update_button = QPushButton(_("Download and Install"), self)
        self._update_button.setMinimumWidth(font_height * 6)
        self._update_button.clicked.connect(self._run_update)

        doc_header = QLabel(_("Documentation and Support"), self)
        doc_header.setFont(large)

        doc_links = QLabel(_links_text, self)
        doc_links.setOpenExternalLinks(True)
        doc_contact = QLabel(_contact_text, self)
        doc_contact.setOpenExternalLinks(True)

        self._layout.addLayout(header_layout)
        self._layout.addSpacing(10)
        current_version_layout = QHBoxLayout()
        current_version_layout.addWidget(current_version_name)
        current_version_layout.addWidget(current_version_value)
        current_version_layout.addStretch()
        self._layout.addLayout(current_version_layout)
        latest_version_layout = QHBoxLayout()
        latest_version_layout.addWidget(latest_version_name)
        latest_version_layout.addWidget(self._latest_version_value)
        latest_version_layout.addStretch()
        self._layout.addLayout(latest_version_layout)
        self._layout.addWidget(self._update_error)
        self._layout.addWidget(self._update_checkbox)
        update_layout = QHBoxLayout()
        update_layout.addWidget(self._check_button)
        update_layout.addWidget(self._update_button)
        update_layout.addStretch()
        self._layout.addLayout(update_layout)
        self._layout.addSpacing(20)
        self._layout.addWidget(doc_header)
        self._layout.addSpacing(5)
        doc_layout = QHBoxLayout()
        doc_layout.addWidget(doc_links)
        doc_layout.addSpacing(40)
        doc_layout.addWidget(doc_contact)
        doc_layout.addStretch()
        self._layout.addLayout(doc_layout)
        self._layout.addStretch()

        root.auto_update.state_changed.connect(self._update_content)
        self._update_content()

    def _update_content(self):
        self._check_button.setEnabled(False)
        self._update_button.setEnabled(False)
        self._update_error.clear()

        au = root.auto_update
        match au.state:
            case UpdateState.unknown:
                self._latest_version_value.setText(_("Not checked"))
                self._check_button.setEnabled(True)
            case UpdateState.checking:
                self._latest_version_value.setText(_("Checking for updates..."))
            case UpdateState.latest:
                self._latest_version_value.setText(au.latest_version)
                self._check_button.setEnabled(True)
            case UpdateState.available:
                self._latest_version_value.setText(au.latest_version)
                self._check_button.setEnabled(True)
                self._update_button.setEnabled(True)
            case UpdateState.downloading:
                self._latest_version_value.setText(_("Downloading package..."))
            case UpdateState.installing:
                self._latest_version_value.setText(_("Installing new version..."))
            case UpdateState.failed_check:
                self._latest_version_value.setText(_("Unknown"))
                self._update_error.setText(au.error)
                self._check_button.setEnabled(True)
            case UpdateState.failed_update:
                self._latest_version_value.setText(_("Update failed"))
                self._update_error.setText(au.error)
                self._check_button.setEnabled(True)
                self._update_button.setEnabled(True)
            case UpdateState.restart_required:
                self._latest_version_value.setText(
                    _("Please restart Krita to complete the update!")
                )

    def _check_updates(self):
        root.auto_update.check()

    def _run_update(self):
        root.auto_update.run()

    def _read(self):
        self._update_checkbox.setChecked(settings.auto_update)

    def _write(self):
        settings.auto_update = self._update_checkbox.isChecked()


_links_text = """
<a href='https://www.interstice.cloud'>Website</a><br><br>
<a href='https://docs.interstice.cloud'>Handbook: Guides and Tips</a><br>
<a href='https://github.com/Acly/krita-ai-diffusion'>GitHub</a><br><br>
"""

_contact_text = """
<a href='https://github.com/Acly/krita-ai-diffusion/issues'>Issues</a><br><br>
<a href='https://github.com/Acly/krita-ai-diffusion/discussions'>Discussions</a><br><br>
<a href='https://discord.gg/pWyzHfHHhU'>Discord</a>
"""


class SettingsDialog(QDialog):
    _instance = None

    @classmethod
    def instance(cls) -> "SettingsDialog":
        assert cls._instance is not None
        return cls._instance

    def __init__(self, server: Server):
        super().__init__()
        type(self)._instance = self

        self.setWindowTitle(_("Configure Image Diffusion"))
        self.setMinimumSize(QSize(840, 480))
        if screen := QGuiApplication.screenAt(QCursor.pos()):
            size = screen.availableSize()
            self.resize(QSize(max(900, int(size.width() * 0.6)), int(size.height() * 0.8)))

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.connection = ConnectionSettings(server)
        self.styles = StylePresets(server)
        self.diffusion = DiffusionSettings()
        self.interface = InterfaceSettings()
        self.performance = PerformanceSettings()
        self.about = AboutSettings()

        self._stack = QStackedWidget(self)
        self._list = QListWidget(self)
        self._list.setFixedWidth(120)

        def create_list_item(text: str, widget: QWidget):
            item = QListWidgetItem(text, self._list)
            item.setSizeHint(QSize(112, 24))
            self._stack.addWidget(widget)

        create_list_item(_("Connection"), self.connection)
        create_list_item(_("Styles"), self.styles)
        create_list_item(_("Diffusion"), self.diffusion)
        create_list_item(_("Interface"), self.interface)
        create_list_item(_("Performance"), self.performance)
        create_list_item(_("Plugin"), self.about)

        self._list.setCurrentRow(0)
        self._list.currentRowChanged.connect(self._change_page)
        layout.addWidget(self._list)

        inner = QVBoxLayout()
        layout.addLayout(inner)
        inner.addWidget(self._stack)
        inner.addSpacing(6)

        self._restore_button = QPushButton(_("Restore Defaults"), self)
        self._restore_button.clicked.connect(self.restore_defaults)

        version_label = QLabel(_("Plugin version") + f": {__version__}", self)
        version_label.setStyleSheet(f"font-style:italic; color: {grey};")

        anchor = _("Open Settings folder")
        self._open_folder_link = QLabel(f"<a href='file://{util.user_data_dir}'>{anchor}</a>", self)
        self._open_folder_link.linkActivated.connect(self._open_settings_folder)
        self._open_folder_link.setToolTip(str(util.user_data_dir))

        self._close_button = QPushButton(_("Ok"), self)
        self._close_button.clicked.connect(self._close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._restore_button)
        button_layout.addStretch()
        button_layout.addWidget(version_label)
        button_layout.addStretch()
        button_layout.addWidget(self._open_folder_link)
        button_layout.addSpacing(8)
        button_layout.addWidget(self._close_button)
        inner.addLayout(button_layout)

        root.connection.state_changed.connect(self._update_connection)
        root.connection.models_changed.connect(self.styles.update_model_lists)

    def read(self):
        self.connection.read()
        self.styles.read()
        self.diffusion.read()
        self.interface.read()
        self.performance.read()
        self.about.read()

    def restore_defaults(self):
        settings.restore()
        settings.save()
        self.read()

    def show(self, style: Optional[Style] = None):
        self.read()
        self.connection.update_ui()
        super().show()

        if style:
            self._list.setCurrentRow(1)
            self.styles.current_style = style
        self._close_button.setFocus()

    def _change_page(self, index):
        self._stack.setCurrentIndex(index)

    def _update_connection(self):
        self.connection.update_server_status()
        if root.connection.state == ConnectionState.connected:
            self.interface.update_translation(root.connection.client)
            self.performance.update_client_info()

    def _open_settings_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.user_data_dir)))

    def _close(self):
        _ = self.close()
