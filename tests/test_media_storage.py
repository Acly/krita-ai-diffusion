import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PyQt5.QtNetwork import QNetworkRequest

from ai_diffusion import media_storage, util
from ai_diffusion.backend import civitai
from ai_diffusion.backend.lora_manager import LoraInfo, fetch_model_preview_bytes
from ai_diffusion.backend.network import RequestManager
from ai_diffusion.image import Extent, Image
from ai_diffusion.prompt_library import PromptLibrary

from .conftest import qtapp


@pytest.fixture
def media(tmp_path: Path, monkeypatch):
    media_storage.shutdown()
    monkeypatch.setattr(util, "user_data_dir", tmp_path / "plugin")
    monkeypatch.setattr(util, "log_dir", tmp_path / "logs")
    util.log_dir.mkdir()
    util.user_data_dir.mkdir()
    yield media_storage
    media_storage.shutdown()


def test_sessions_cleanup_without_touching_live_or_unowned_directories(tmp_path: Path):
    first = media_storage.MediaSession(tmp_path)
    (first.folder("preview") / "image.png").write_bytes(b"image")
    unowned = tmp_path / "user-images"
    unowned.mkdir()
    (unowned / "saved.png").touch()
    second = media_storage.MediaSession(tmp_path)
    assert first.path.exists()
    second.close()
    assert first.path.exists()
    first.close()
    assert (unowned / "saved.png").exists()


def test_crash_debris_is_cleaned_next_start(tmp_path: Path):
    script = (
        "from ai_diffusion.media_storage import MediaSession; import os,sys; "
        "from pathlib import Path; s=MediaSession(Path(sys.argv[1])); "
        "(s.folder('preview')/'image.png').write_bytes(b'image'); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)
    old = next(tmp_path.glob("session-*"))
    current = media_storage.MediaSession(tmp_path)
    assert not old.exists()
    current.close()
    assert list(tmp_path.iterdir()) == []


def test_sweep_does_not_follow_links(tmp_path: Path, monkeypatch):
    path = tmp_path / ("session-" + "a" * 32)
    path.mkdir()
    (path / "keep.png").touch()
    original = media_storage._is_link
    monkeypatch.setattr(media_storage, "_is_link", lambda p: p == path or original(p))
    current = media_storage.MediaSession(tmp_path)
    assert (path / "keep.png").exists()
    current.close()


def test_managed_paths_are_isolated_and_cleanup_preserves_exports(media, tmp_path: Path):
    exported = tmp_path / "manually-saved.png"
    exported.touch()
    env = media.managed_server_environment(tmp_path / "server")
    config = json.loads((Path(env["LORA_MANAGER_SETTINGS_DIR"]) / "settings.json").read_text())
    assert config["sidecar_storage_mode"] == "centralized"
    for name in ["sidecar_storage_path", "example_images_path"]:
        assert Path(config[name]).is_relative_to(media.session().path)
    args = media.managed_server_arguments()
    assert all(Path(args[i]).is_relative_to(media.session().path) for i in [1, 3, 5, 7])
    assert Path(env["TEMP"]).is_relative_to(media.session().path)
    session_path = media.session().path
    (Path(config["sidecar_storage_path"]) / "preview.png").touch()
    media.shutdown()
    assert not session_path.exists()
    assert exported.exists()


def test_legacy_cleanup_is_limited_to_plugin_owned_staging(media):
    for name in ["eagle_upload", "animadex/thumbs", "prompt_previews"]:
        folder = util.user_data_dir / name
        folder.mkdir(parents=True)
        (folder / "image.png").touch()
    keep = util.user_data_dir / "my-export.png"
    keep.touch()
    media.initialize()
    assert keep.exists()
    assert not (util.user_data_dir / "eagle_upload").exists()
    assert not (util.user_data_dir / "animadex/thumbs").exists()


@qtapp
async def test_local_model_preview_fetches_from_internet_only(monkeypatch):
    lora = LoraInfo(
        "model", sha256="abc", preview_url="http://localhost:8188/api/lm/previews?path=old.png"
    )
    requests = AsyncMock()
    requests.get.return_value = {
        "images": [{"url": "https://image.civitai.com/id/original=true/test.jpeg"}]
    }
    fetch = AsyncMock(return_value=b"image")
    monkeypatch.setattr(civitai, "requests", lambda: requests)
    monkeypatch.setattr(civitai, "fetch_image", fetch)
    assert await fetch_model_preview_bytes(lora) == b"image"
    assert requests.get.await_args is not None
    assert fetch.await_args is not None
    assert "model-versions/by-hash/abc" in requests.get.await_args.args[0]
    assert fetch.await_args.args[0].startswith("https://image.civitai.com/")
    assert "anim=false" in fetch.await_args.args[0]


@qtapp
async def test_missing_online_preview_does_not_read_local_file(monkeypatch):
    requests = AsyncMock()
    requests.get.return_value = {"images": []}
    fetch = AsyncMock()
    monkeypatch.setattr(civitai, "requests", lambda: requests)
    monkeypatch.setattr(civitai, "fetch_image", fetch)
    lora = LoraInfo("private-model", sha256="abc", preview_url="file:///private.png")
    assert await fetch_model_preview_bytes(lora) is None
    fetch.assert_not_awaited()


def test_prompt_thumbnail_is_ram_only(tmp_path: Path):
    library = PromptLibrary(tmp_path / "prompts.json", tmp_path / "previews")
    library.save_preview("test", Image.create(Extent(2, 2)))
    assert library.has_preview("test")
    assert not (tmp_path / "previews").exists()
    second = PromptLibrary(tmp_path / "prompts.json", tmp_path / "previews")
    assert not second.has_preview("test")


def test_network_requests_never_use_or_save_disk_cache():
    request = RequestManager()._prepare_request("https://example.com/preview.png")
    assert request.attribute(QNetworkRequest.CacheSaveControlAttribute) is False
    assert (
        request.attribute(QNetworkRequest.CacheLoadControlAttribute)
        == QNetworkRequest.AlwaysNetwork
    )


def test_external_server_is_denied_without_explicit_acceptance(media):
    with pytest.raises(util.PluginError, match="not approved"):
        media.require_external_acceptance("http://external-server")


def test_custom_workflow_is_denied_without_explicit_acceptance(media):
    with pytest.raises(util.PluginError, match="not approved"):
        media.require_workflow_acceptance({"1": {"class_type": "SaveImage"}})


def test_temporary_transfer_is_removed_on_failure(media):
    folder = None
    with pytest.raises(RuntimeError), media.temporary_directory("animation-") as temp:
        folder = Path(temp)
        (folder / "frame.png").touch()
        raise RuntimeError("import failed")
    assert folder is not None and not folder.exists()


def test_cleanup_refuses_unowned_directory(tmp_path: Path):
    owned = media_storage.MediaSession(tmp_path / "sessions")
    keep = tmp_path / "exports"
    keep.mkdir()
    with pytest.raises(ValueError, match="outside"):
        owned._remove(keep)
    assert keep.exists()
    owned.close()


def test_metadata_preferences_survive_but_media_does_not(tmp_path: Path):
    cache = tmp_path / "metadata"
    first = media_storage.MediaSession(tmp_path / "sessions", cache)
    sidecars = first.folder("model-sidecars")
    (sidecars / "model.metadata.json").write_text(
        json.dumps({"favorite": True, "tags": ["test"], "preview_url": "local.png"})
    )
    (sidecars / "preview.png").write_bytes(b"media")
    first.close()
    assert not (cache / "preview.png").exists()
    assert json.loads((cache / "model.metadata.json").read_text())["preview_url"] == ""
    second = media_storage.MediaSession(tmp_path / "sessions", cache)
    metadata = json.loads((second.folder("model-sidecars") / "model.metadata.json").read_text())
    assert metadata["preview_url"] == ""
    assert metadata["favorite"] is True
    assert not (second.path / "model-sidecars/preview.png").exists()
    second.close()


@qtapp
async def test_remote_non_civitai_preview_keeps_original_url(monkeypatch):
    url = "https://huggingface.co/example/resolve/main/preview.png"
    fetch = AsyncMock(return_value=b"image")
    monkeypatch.setattr(civitai, "fetch_image", fetch)
    assert await fetch_model_preview_bytes(LoraInfo("model", preview_url=url)) == b"image"
    fetch.assert_awaited_once_with(url)
