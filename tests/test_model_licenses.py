from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from ai_diffusion.backend import model_licenses, resources, server
from ai_diffusion.backend.server import Server
from ai_diffusion.settings import settings
from scripts import download_models

from .conftest import qtapp


@pytest.fixture
def unaccepted_terms(monkeypatch):
    monkeypatch.setattr(settings, "krea2_v2_license_revision", "")
    monkeypatch.setattr(settings, "krea2_v2_license_accepted_at", "")


def test_acceptance_is_bound_to_model_document_revision(unaccepted_terms, monkeypatch):
    assert not model_licenses.accepted()
    monkeypatch.setattr(settings, "krea2_v2_license_revision", "older-revision")
    with pytest.raises(RuntimeError, match="Review and accept"):
        model_licenses.require_accepted(["lora-identity_edit-krea2"])
    monkeypatch.setattr(settings, "krea2_v2_license_revision", model_licenses.revision)
    assert model_licenses.accepted()
    model_licenses.require_accepted(["lora-general_edit-krea2"])


def test_acknowledgement_records_date_and_saves(unaccepted_terms, monkeypatch):
    saved = Mock()
    monkeypatch.setattr(settings, "save", saved)
    model_licenses.acknowledge()
    assert settings.krea2_v2_license_revision == model_licenses.revision
    assert settings.krea2_v2_license_accepted_at.endswith("+00:00")
    saved.assert_called_once()


@qtapp
async def test_server_rejects_v2_download_before_network(
    tmp_path: Path, unaccepted_terms, monkeypatch
):
    srv = Server(str(tmp_path))
    network = Mock()
    monkeypatch.setattr(server, "QNetworkAccessManager", network)
    with pytest.raises(RuntimeError, match="Review and accept"):
        await srv.download(["lora-identity_edit-krea2"], lambda *_: None)
    network.assert_not_called()


@qtapp
async def test_documents_installed_before_v2_weights(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "krea2_v2_license_revision", model_licenses.revision)
    srv = Server(str(tmp_path))
    srv.comfy_dir = tmp_path / "ComfyUI"
    monkeypatch.setattr(srv, "check_install", lambda: None)
    calls = []

    async def download(name, network, url, target, cb):
        docs = tmp_path / "models/loras/krea2-edit-v2-licenses"
        assert all((docs / doc["name"]).exists() for doc in model_licenses.manifest["files"])
        calls.append(target)
        target.write_bytes(b"weights")

    monkeypatch.setattr(server, "_download_cached", download)
    await srv.download(["lora-identity_edit-krea2"], lambda *_: None)
    assert len(calls) == 1
    assert calls[0].name == "krea2_identity_edit_v2.safetensors"


def test_invalid_bundled_license_fails_before_installing_documents(tmp_path: Path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for doc in model_licenses.manifest["files"]:
        (bundle / doc["name"]).write_bytes((model_licenses.bundle / doc["name"]).read_bytes())
    (bundle / "LICENSE.pdf").write_bytes(b"invalid")
    monkeypatch.setattr(model_licenses, "bundle", bundle)
    destination = tmp_path / "server"
    with pytest.raises(RuntimeError, match="failed verification"):
        model_licenses.install_documents(destination)
    assert not destination.exists()


@qtapp
async def test_cli_v2_download_requires_explicit_flag(tmp_path: Path):
    model = resources.find_resource(resources.ResourceId.parse("lora-identity_edit-krea2"))
    assert model is not None
    with pytest.raises(ValueError, match="acknowledgement"):
        await download_models.download_models(tmp_path, {model})
    assert not list(tmp_path.iterdir())


@qtapp
async def test_cli_dry_run_does_not_acknowledge_terms(tmp_path: Path, unaccepted_terms):
    model = resources.find_resource(resources.ResourceId.parse("lora-identity_edit-krea2"))
    assert model is not None
    await download_models.download(AsyncMock(), model, tmp_path, dry_run=True)
    assert not model_licenses.accepted()
    assert not list(tmp_path.rglob("*.safetensors"))
    assert not (tmp_path / "models/loras/krea2-edit-v2-licenses").exists()
