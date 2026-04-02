import hashlib
import os
import shutil
from base64 import b64decode
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

from PyQt5.QtCore import QObject, pyqtSignal

from . import __version__, eventloop
from .network import RequestManager
from .platform_tools import ZipFile
from .properties import ObservableProperties, Property
from .util import client_logger as log


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
    signature: str


class AutoUpdate(QObject, ObservableProperties):
    default_api_url = os.getenv("INTERSTICE_URL", "https://api.interstice.cloud")
    default_update_public_key = os.getenv("INTERSTICE_UPDATE_PUBLIC_KEY", "6f1f5f6fcb3f4f77f8f3419febe08d4f0d76d1234e1b4e6f6d9f2c1d7a8b9c0d")

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
        update_public_key: str | None = None,
    ):
        super().__init__()
        self.plugin_dir = plugin_dir or Path(__file__).parent.parent
        self.current_version = current_version or __version__
        self.api_url = api_url or self.default_api_url
        self.update_public_key = update_public_key or self.default_update_public_key
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
        log.info(f"Checking for latest plugin version at {self.api_url}")
        result = await self._net.get(
            f"{self.api_url}/plugin/latest?version={self.current_version}", timeout=10
        )
        self.latest_version = result.get("version")
        if not self.latest_version:
            log.error(f"Invalid plugin update information: {result}")
            self.state = UpdateState.failed_check
            self.error = "Failed to retrieve plugin update information"
        elif self.latest_version == self.current_version:
            log.info("Plugin is up to date!")
            self.state = UpdateState.latest
        elif "url" not in result or "sha256" not in result or "signature" not in result:
            log.error(f"Invalid plugin update information: {result}")
            self.state = UpdateState.failed_check
            self.error = "Plugin update package is incomplete"
        else:
            package = UpdatePackage(
                version=self.latest_version,
                url=result["url"],
                sha256=result["sha256"],
                signature=result["signature"],
            )
            self._verify_package_signature(package)
            log.info(f"New plugin version available: {self.latest_version}")
            self._package = package
            self.state = UpdateState.available

    def run(self):
        return eventloop.run(
            self._handle_errors(self._run, UpdateState.failed_update, "Failed to update plugin")
        )

    async def _run(self):
        assert self.latest_version and self._package

        self._temp_dir = TemporaryDirectory()
        archive_path = Path(self._temp_dir.name) / f"krita_ai_diffusion-{self.latest_version}.zip"
        log.info(f"Downloading plugin update {self._package.url}")
        self.state = UpdateState.downloading
        archive_data = await self._net.download(self._package.url)

        sha256 = hashlib.sha256(archive_data).hexdigest()
        if sha256 != self._package.sha256:
            log.error(f"Update package hash mismatch: {sha256} != {self._package.sha256}")
            raise RuntimeError("Downloaded plugin package is corrupted or incomplete")

        archive_path.write_bytes(archive_data)
        source_dir = Path(self._temp_dir.name) / f"krita_ai_diffusion-{self.latest_version}"
        log.info(f"Extracting plugin archive into {source_dir}")
        self.state = UpdateState.installing
        with ZipFile(archive_path) as zip_file:
            zip_file.extractall(source_dir)

        log.info(f"Installing new plugin version to {self.plugin_dir}")
        shutil.copytree(source_dir, self.plugin_dir, dirs_exist_ok=True)
        self.current_version = self.latest_version
        self.state = UpdateState.restart_required

    def _verify_package_signature(self, package: UpdatePackage):
        if not self.update_public_key:
            raise RuntimeError("Plugin update verification key is missing")

        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except Exception as e:
            raise RuntimeError("Plugin update signature verification is unavailable") from e

        try:
            public_key = bytes.fromhex(self.update_public_key)
            signature = b64decode(package.signature, validate=True)
        except Exception as e:
            raise RuntimeError("Plugin update signature data is invalid") from e

        if len(public_key) != 32 or len(signature) != 64:
            raise RuntimeError("Plugin update signature data is invalid")

        payload = f"{package.version}\\n{package.url}\\n{package.sha256}".encode("utf-8")
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
        except InvalidSignature as e:
            raise RuntimeError("Plugin update signature is invalid") from e

    @property
    def is_available(self):
        return self.latest_version is not None and self.latest_version != self.current_version

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
