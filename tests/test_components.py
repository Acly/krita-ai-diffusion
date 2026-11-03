from pathlib import Path
from unittest.mock import ANY, AsyncMock

import pytest
from PyQt5.QtNetwork import QNetworkAccessManager

from ai_diffusion.backend import resources, server
from ai_diffusion.backend.comfy_workflow import ComfyWorkflow, Output
from ai_diffusion.backend.server import Server
from ai_diffusion.settings import settings

from .conftest import qtapp


def test_v2_download_does_not_accept_old_weights(tmp_path: Path):
    resource = resources.find_resource(resources.ResourceId.parse("lora-identity_edit-krea2"))
    assert resource is not None
    old = tmp_path / "models/loras/krea2_identity_edit_v1_2.safetensors"
    old.parent.mkdir(parents=True)
    old.touch()
    assert not resource.exists_in(tmp_path)
    assert resource.files[0].sha256
    (tmp_path / resource.files[0].path).touch()
    assert resource.exists_in(tmp_path)


def test_krea_v2_graph_keeps_pixel_path_and_grounding():
    w = ComfyWorkflow()
    positive = w.krea2_edit_grounded_encode(Output(1, 0), "restage", Output(2, 0), Output(3, 0))
    negative = w.krea2_edit_grounded_encode(Output(1, 0), "", Output(2, 0), Output(3, 0))
    patch = w.krea2_edit_model_patch(
        Output(4, 0),
        Output(5, 0),
        vae=Output(6, 0),
        source_image=Output(2, 0),
        source_image_b=Output(3, 0),
        target_latent=Output(7, 0),
        ref_position="beside (right of target)",
    )
    assert positive != negative
    inputs = w.root[str(patch.node)]["inputs"]
    assert inputs["target_latent"] == ["7", 0]
    assert inputs["fit_mode"] == "fit"
    assert inputs["ref_position"] == "beside (right of target)"
    assert inputs["ref_position_a"] == "aligned (trained)"


@pytest.mark.parametrize("fail", [False, True])
@qtapp
async def test_component_dependencies_and_rollback(tmp_path: Path, monkeypatch, fail: bool):
    srv = Server(str(tmp_path))
    srv.comfy_dir = tmp_path / "ComfyUI"
    pkg = next(p for p in resources.selectable_custom_nodes if p.name == "LoRA Browser metadata")
    folder = srv.comfy_dir / "custom_nodes" / pkg.folder
    folder.mkdir(parents=True)
    (folder / "existing.txt").write_text("keep")
    srv._cache_dir.mkdir(parents=True, exist_ok=True)

    async def extract(*args):
        extracted = folder.with_name(folder.name + "-" + pkg.version)
        extracted.mkdir()
        (extracted / "requirements.txt").write_text("aiosqlite")

    monkeypatch.setattr(server, "_download_cached", AsyncMock())
    monkeypatch.setattr(server, "_extract_archive", extract)
    pip = AsyncMock(side_effect=RuntimeError("pip failed") if fail else None)
    monkeypatch.setattr(srv, "_pip_install", pip)
    if fail:
        with pytest.raises(RuntimeError, match="pip failed"):
            await srv._install_custom_node(pkg, QNetworkAccessManager(), lambda *_: None)
        assert (folder / "existing.txt").read_text() == "keep"
        assert not srv.component_installed(pkg)
    else:
        await srv._install_custom_node(pkg, QNetworkAccessManager(), lambda *_: None)
        assert srv.component_installed(pkg)
        pip.assert_awaited_once_with(pkg.name, ["-r", str(folder / "requirements.txt")], ANY)
    assert not folder.with_name(folder.name + ".krita-backup").exists()


@qtapp
async def test_only_selected_components_are_installed(tmp_path: Path, monkeypatch):
    srv = Server(str(tmp_path))
    srv.comfy_dir = tmp_path / "ComfyUI"
    installed = AsyncMock()
    monkeypatch.setattr(srv, "_install_custom_node", installed)
    monkeypatch.setattr(srv, "check_install", lambda: None)
    previous = settings.server_components
    try:
        settings.server_components = []
        await srv.install_components(lambda *_: None)
        installed.assert_not_awaited()
        settings.server_components = ["LoRA Browser metadata"]
        await srv.install_components(lambda *_: None)
        assert installed.await_count == 1
        assert installed.await_args is not None
        assert installed.await_args.args[0].name == "LoRA Browser metadata"
    finally:
        settings.server_components = previous


def test_krea_selection_installs_base_and_encoders(tmp_path: Path):
    srv = Server(str(tmp_path))
    selected = srv.resolve_model_dependencies(["lora-identity_edit-krea2"])
    assert "checkpoint-turbo_fp8-krea2" in selected
    assert "text_encoder-qwen_3vl_4b-krea2" in selected
    assert "vae-default-krea2" in selected
    assert len(selected) == len(set(selected))
    base = resources.find_resource(resources.ResourceId.parse("checkpoint-turbo_fp8-krea2"))
    assert base is not None
    target = tmp_path / base.files[0].path
    target.parent.mkdir(parents=True)
    target.touch()
    assert "checkpoint-turbo_fp8-krea2" not in srv.resolve_model_dependencies([
        "lora-identity_edit-krea2"
    ])
    assert srv.resolve_model_dependencies(["checkpoint-realvis-sdxl"]) == [
        "checkpoint-realvis-sdxl"
    ]
