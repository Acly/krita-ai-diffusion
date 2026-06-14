import hashlib
import os
import shutil
from enum import Enum
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

from PyQt5.QtCore import QObject, pyqtSignal

from .. import __version__, eventloop
from ..backend.network import RequestManager
from ..platform_tools import ZipFile
from ..util import client_logger as log
from .properties import ObservableProperties, Property


class UpdateState(Enum):
    unknown = 1
    checking = 2
    available = 3
    latest = 4
    downloading = 5
    installing = 6
    restart_required = 7
    failed_check = 8
    failed_update = 9


class UpdatePackage(NamedTuple):
    version: str
    url: str
    sha256: str


def _version_parts(version: str) -> list[int]:
    result = []
    for part in version.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        result.append(int(digits or 0))
    return result


def _is_newer_version(version: str, current_version: str) -> bool:
    for a, b in zip_longest(_version_parts(version), _version_parts(current_version), fillvalue=0):
        if a != b:
            return a > b
    return False


class AutoUpdate(QObject, ObservableProperties):
    default_api_url = os.getenv("INTERSTICE_URL", "https://api.interstice.cloud")

    state = Property(UpdateState.unknown)
    latest_version = Property("")
    error = Property("")

    state_changed = pyqtSignal(UpdateState)
    latest_version_changed = pyqtSignal(str)
    error_changed = pyqtSignal(str)

    def __init__(
        self,
        plugin_dir: Path | None = None,
        current_version: str | None = None,
        api_url: str | None = None,
    ):
        super().__init__()
        self.plugin_dir = plugin_dir or Path(__file__).parent.parent
        self.current_version = current_version or __version__
        self.api_url = api_url or self.default_api_url
        self._package: UpdatePackage | None = None
        self._temp_dir: TemporaryDirectory | None = None
        self._request_manager: RequestManager | None = None

    def check(self):
        return eventloop.run(
            self._handle_errors(
                self._check, UpdateState.failed_check, "Failed to check for new plugin version"
            )
        )

    async def _check(self):
        if self.state is UpdateState.restart_required:
            return

        self.state = UpdateState.checking
        self._package = None
        log.info(f"Checking for latest plugin version at {self.api_url}")
        result = await self._net.get(
            f"{self.api_url}/plugin/latest?version={self.current_version}", timeout=10
        )
        self.latest_version = result.get("version")
        if not self.latest_version:
            log.error(f"Invalid plugin update information: {result}")
            self.state = UpdateState.failed_check
            self.error = "Failed to retrieve plugin update information"
        elif not _is_newer_version(self.latest_version, self.current_version):
            log.info("Plugin is up to date!")
            self.state = UpdateState.latest
        elif "url" not in result or "sha256" not in result:
            log.error(f"Invalid plugin update information: {result}")
            self.state = UpdateState.failed_check
            self.error = "Plugin update package is incomplete"
        else:
            log.info(f"New plugin version available: {self.latest_version}")
            self._package = UpdatePackage(
                version=self.latest_version,
                url=result["url"],
                sha256=result["sha256"],
            )
            self.state = UpdateState.available

    def run(self):
        return eventloop.run(
            self._handle_errors(self._run, UpdateState.failed_update, "Failed to update plugin")
        )

    async def _run(self):
        if self.state is not UpdateState.available or self._package is None:
            raise RuntimeError("No plugin update package is available")
        if self._package.version != self.latest_version:
            raise RuntimeError("Plugin update package does not match latest version")
        package = self._package

        self._temp_dir = TemporaryDirectory()
        archive_path = Path(self._temp_dir.name) / f"krita_ai_diffusion-{package.version}.zip"
        log.info(f"Downloading plugin update {package.url}")
        self.state = UpdateState.downloading
        archive_data = await self._net.download(package.url)

        sha256 = hashlib.sha256(archive_data).hexdigest()
        if sha256 != package.sha256:
            log.error(f"Update package hash mismatch: {sha256} != {package.sha256}")
            raise RuntimeError("Downloaded plugin package is corrupted or incomplete")

        archive_path.write_bytes(archive_data)
        source_dir = Path(self._temp_dir.name) / f"krita_ai_diffusion-{package.version}"
        log.info(f"Extracting plugin archive into {source_dir}")
        self.state = UpdateState.installing
        with ZipFile(archive_path) as zip_file:
            zip_file.extractall(source_dir)

        log.info(f"Installing new plugin version to {self.plugin_dir}")
        package_plugin_dir = source_dir / "ai_diffusion"
        if package_plugin_dir.is_dir():
            shutil.rmtree(self.plugin_dir / "ai_diffusion", ignore_errors=True)
            (self.plugin_dir / "ai_diffusion.desktop").unlink(missing_ok=True)
            shutil.copytree(package_plugin_dir, self.plugin_dir, dirs_exist_ok=True)

            package_desktop = source_dir / "ai_diffusion.desktop"
            if package_desktop.exists():
                shutil.copy2(package_desktop, self.plugin_dir.parent / "ai_diffusion.desktop")

            package_action = package_plugin_dir / "ai_diffusion.action"
            if package_action.exists():
                actions_dir = self.plugin_dir.parent.parent / "actions"
                actions_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(package_action, actions_dir / "ai_diffusion.action")
        else:
            shutil.copytree(source_dir, self.plugin_dir, dirs_exist_ok=True)
        self.current_version = self.latest_version
        self.state = UpdateState.restart_required

    @property
    def is_available(self):
        return bool(self.latest_version) and _is_newer_version(
            self.latest_version, self.current_version
        )

    @property
    def _net(self):
        if self._request_manager is None:
            self._request_manager = RequestManager()
        return self._request_manager

    async def _handle_errors(self, func, error_state: UpdateState, message: str):
        try:
            return await func()
        except Exception as e:
            log.exception(e)
            self.error = f"{message}: {e}"
            self.state = error_state
            return None
