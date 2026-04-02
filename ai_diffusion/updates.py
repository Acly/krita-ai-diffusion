import asyncio
import hashlib
import os
import shutil
from enum import Enum
from http.client import HTTPSConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple
from urllib.parse import urljoin, urlparse

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


class AutoUpdate(QObject, ObservableProperties):
    default_api_url = os.getenv("INTERSTICE_URL", "https://api.interstice.cloud")
    max_redirects = 5

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
        self._trusted_update_hosts = self._collect_trusted_update_hosts(self.api_url)
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
        elif "url" not in result or "sha256" not in result:
            log.error(f"Invalid plugin update information: {result}")
            self.state = UpdateState.failed_check
            self.error = "Plugin update package is incomplete"
        else:
            package_url = result["url"]
            self._validate_download_url(package_url)
            log.info(f"New plugin version available: {self.latest_version}")
            self._package = UpdatePackage(
                version=self.latest_version,
                url=package_url,
                sha256=result["sha256"],
            )
            self.state = UpdateState.available

    def run(self):
        return eventloop.run(
            self._handle_errors(self._run, UpdateState.failed_update, "Failed to update plugin")
        )

    async def _run(self):
        assert self.latest_version and self._package

        self._temp_dir = TemporaryDirectory()
        archive_path = Path(self._temp_dir.name) / f"krita_ai_diffusion-{self.latest_version}.zip"
        download_url = await asyncio.to_thread(self._resolve_download_url, self._package.url)
        log.info(f"Downloading plugin update {download_url}")
        self.state = UpdateState.downloading
        archive_data = await self._net.download(download_url)

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

    @staticmethod
    def _collect_trusted_update_hosts(api_url: str):
        hosts = {
            host.strip().lower()
            for host in os.getenv("INTERSTICE_UPDATE_HOSTS", "").split(",")
            if host.strip()
        }
        api_host = urlparse(api_url).hostname
        if api_host:
            hosts.add(api_host.lower())
        return hosts

    def _validate_download_url(self, url: str):
        parsed_url = urlparse(url)
        host = parsed_url.hostname.lower() if parsed_url.hostname else None
        if parsed_url.scheme != "https":
            raise RuntimeError("Plugin update URL must use HTTPS")
        if host is None or host not in self._trusted_update_hosts:
            raise RuntimeError("Plugin update URL host is not trusted")
        return parsed_url

    def _resolve_download_url(self, url: str):
        current_url = url
        current_parsed = self._validate_download_url(current_url)
        for _ in range(self.max_redirects):
            path = current_parsed.path or "/"
            if current_parsed.query:
                path = f"{path}?{current_parsed.query}"
            connection = HTTPSConnection(current_parsed.hostname, current_parsed.port, timeout=10)
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                if response.status in {301, 302, 303, 307, 308}:
                    redirect_url = response.getheader("Location")
                    if not redirect_url:
                        raise RuntimeError("Plugin update URL redirect is missing location")
                    redirected_url = urljoin(current_url, redirect_url)
                    redirected_parsed = self._validate_download_url(redirected_url)
                    if (
                        redirected_parsed.scheme != current_parsed.scheme
                        or redirected_parsed.hostname != current_parsed.hostname
                    ):
                        raise RuntimeError("Plugin update URL redirect changed host or scheme")
                    current_url = redirected_url
                    current_parsed = redirected_parsed
                    continue
                return current_url
            finally:
                connection.close()
        raise RuntimeError("Plugin update URL has too many redirects")

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
