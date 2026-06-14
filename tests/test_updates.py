import hashlib
import os
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp import ClientSession
from PyQt5.QtCore import pyqtBoundSignal

from ai_diffusion.model.updates import AutoUpdate, UpdateState, _is_newer_version
from ai_diffusion.platform_tools import ZipFile

from .conftest import CloudService, qtapp


class SignalObserver:
    def __init__(self, signal: pyqtBoundSignal):
        self.events = []
        signal.connect(self.on_changed)

    def on_changed(self, value):
        self.events.append(value)

    def reset(self):
        self.events = []


class FakeUpdateService:
    def __init__(self, response: dict, archive_data: bytes = b""):
        self.response = response
        self.archive_data = archive_data

    async def get(self, url: str, timeout: int = 10):
        return self.response

    async def download(self, url: str):
        return self.archive_data


def http_session(service_url: str):
    service_token = os.environ["INTERSTICE_INFRA_TOKEN"]
    headers = {"Authorization": f"Bearer {service_token}"}
    return ClientSession(service_url, headers=headers)


@pytest.mark.parametrize(
    ("version", "current_version", "expected"),
    [
        ("1.51.1", "1.51.1", False),
        ("1.51.0", "1.51.1", False),
        ("1.52.0", "1.51.9", True),
        ("1.51.10", "1.51.9", True),
        ("1.51", "1.51.0", False),
    ],
)
def test_is_newer_version(version: str, current_version: str, expected: bool):
    assert _is_newer_version(version, current_version) is expected


@qtapp
async def test_update_check_ignores_older_versions(tmp_path: Path):
    updater = AutoUpdate(
        current_version="1.51.1",
        plugin_dir=tmp_path / "pykrita" / "ai_diffusion",
        api_url="https://example.invalid",
    )
    updater._request_manager = cast(
        Any,
        FakeUpdateService({
            "version": "1.51.2",
            "url": "https://example.invalid/krita_ai_diffusion-1.51.2.zip",
            "sha256": "unused",
        }),
    )

    state_changes = SignalObserver(updater.state_changed)
    await updater.check()
    assert updater.state is UpdateState.available
    assert updater._package is not None

    updater._request_manager = cast(
        Any,
        FakeUpdateService({
            "version": "1.51.0",
            "url": "https://example.invalid/krita_ai_diffusion-1.51.0.zip",
            "sha256": "unused",
        }),
    )

    state_changes.reset()
    await updater.check()

    assert state_changes.events == [UpdateState.checking, UpdateState.latest]
    assert updater.state is UpdateState.latest
    assert updater._package is None
    assert not updater.is_available

    await updater.run()
    assert updater.state is UpdateState.failed_update
    assert "No plugin update package is available" in updater.error


@qtapp
async def test_update_installs_release_package_layout(tmp_path: Path):
    install_dir = tmp_path / "krita" / "pykrita" / "ai_diffusion"
    install_dir.mkdir(parents=True)
    (install_dir / "test_file.txt").write_text("old plugin")
    (install_dir / "ai_diffusion").mkdir()
    (install_dir / "ai_diffusion" / "old_nested_file.txt").write_text("old nested plugin")
    (install_dir / "ai_diffusion.desktop").write_text("misplaced desktop metadata")
    actions_dir = tmp_path / "krita" / "actions"

    build_dir = tmp_path / "build"
    build_plugin_dir = build_dir / "ai_diffusion"
    build_plugin_dir.mkdir(parents=True)
    (build_dir / "ai_diffusion.desktop").write_text("desktop metadata")
    (build_plugin_dir / "ai_diffusion.action").write_text("action metadata")
    (build_plugin_dir / "test_file.txt").write_text("new plugin")

    archive_path = build_dir / "krita_ai_diffusion-1.51.2.zip"
    with ZipFile(archive_path, "w") as zip_file:
        zip_file.write(build_dir / "ai_diffusion.desktop", "ai_diffusion.desktop")
        for file in build_plugin_dir.iterdir():
            zip_file.write(file, f"ai_diffusion/{file.name}")
    archive_data = archive_path.read_bytes()

    updater = AutoUpdate(
        current_version="1.51.1",
        plugin_dir=install_dir,
        api_url="https://example.invalid",
    )
    updater._request_manager = cast(
        Any,
        FakeUpdateService(
            {
                "version": "1.51.2",
                "url": "https://example.invalid/krita_ai_diffusion-1.51.2.zip",
                "sha256": hashlib.sha256(archive_data).hexdigest(),
            },
            archive_data,
        ),
    )

    await updater.check()
    assert updater.state is UpdateState.available
    await updater.run()

    assert updater.state is UpdateState.restart_required
    assert (install_dir / "test_file.txt").read_text() == "new plugin"
    assert not (install_dir / "ai_diffusion").exists()
    assert not (install_dir / "ai_diffusion.desktop").exists()
    assert (install_dir.parent / "ai_diffusion.desktop").read_text() == "desktop metadata"
    assert (actions_dir / "ai_diffusion.action").read_text() == "action metadata"


@qtapp
async def test_auto_update(cloud_service: CloudService, tmp_path: Path):
    if not cloud_service.enabled:
        pytest.skip("Cloud service not running")
    await run_auto_update_test(cloud_service, tmp_path)


async def run_auto_update_test(service: CloudService, tmp_path: Path):
    async with http_session(service.url) as session:
        last_version = new_version = "666.6.6"

        # Get the latest plugin version (set from previous test)
        async with session.get(f"/plugin/latest?version={new_version}") as response:
            assert response.status == 200
            result = await response.json()
            last_version = result["version"]
            a, b, c = last_version.split(".")
            new_version = f"{a}.{b}.{int(c) + 1}"

        # Create an existing installation
        install_dir = tmp_path / "install"
        install_plugin_dir = install_dir / "test_plugin"
        install_plugin_dir.mkdir(parents=True)
        install_test_file = install_plugin_dir / "test_file.txt"
        install_test_file.write_text("local produce is the best")

        updater = AutoUpdate(
            current_version=last_version,
            plugin_dir=install_dir,
            api_url=service.url,
        )
        assert updater.state is UpdateState.unknown

        state_changes = SignalObserver(updater.state_changed)
        await updater.check()
        assert state_changes.events == [UpdateState.checking, UpdateState.latest]
        assert updater.state is UpdateState.latest

        # Create a new plugin version
        build_dir = tmp_path / "build"
        build_plugin_dir = build_dir / "test_plugin"
        build_plugin_dir.mkdir(parents=True)
        build_test_file = build_plugin_dir / "test_file.txt"
        build_test_file.write_text("if you're feeling orange, try flying a kite")
        build_archive = build_dir / f"test_plugin-{new_version}.zip"

        # Build the plugin archive
        with ZipFile(build_archive, "w") as zip_file:
            for file in build_plugin_dir.iterdir():
                zip_file.write(file, f"test_plugin/{file.name}")

        # Upload the plugin as new version
        archive_data = build_archive.read_bytes()
        async with session.put(f"/plugin/upload/{new_version}", data=archive_data) as response:
            assert response.status == 200
            uploaded = await response.json()
            assert uploaded["status"] == "uploaded" and uploaded["version"] == new_version

        # Check for new version
        state_changes.reset()
        await updater.check()
        assert state_changes.events == [UpdateState.checking, UpdateState.available]
        assert updater.state is UpdateState.available
        assert updater.latest_version == new_version

        # Run the update
        state_changes.reset()
        await updater.run()
        assert state_changes.events == [
            UpdateState.downloading,
            UpdateState.installing,
            UpdateState.restart_required,
        ]
        assert updater.state is UpdateState.restart_required
        assert updater.latest_version == new_version
        assert install_test_file.read_text() == "if you're feeling orange, try flying a kite"


async def test_authorization(cloud_service: CloudService):
    if not cloud_service.enabled:
        pytest.skip("Cloud service not running")
    async with ClientSession(cloud_service.url) as session:
        # Version check is public
        async with session.get("/plugin/latest?version=1.2.3") as response:
            assert response.status == 200

        # Upload requires authorization
        async with session.put("/plugin/upload/1.2.3") as response:
            assert response.status == 401
