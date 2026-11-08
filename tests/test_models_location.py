import hashlib
from pathlib import Path

from ai_diffusion.backend.resources import (
    ModelFile,
    ModelResource,
    ResourceId,
    ResourceKind,
    VerificationState,
    resolve_model_path,
    resolve_model_paths,
)
from ai_diffusion.backend.server import (
    _configure_extra_model_paths,
    _extra_model_paths_section,
    _remove_managed_section,
    find_missing,
)
from ai_diffusion.style import Arch


def _make_resource(files: list[str], sha256: str | None = None) -> ModelResource:
    id = ResourceId(ResourceKind.checkpoint, Arch.sdxl, "test-model")
    urls = [ModelFile(Path(p), "https://example.com/x", id, sha256) for p in files]
    return ModelResource("test-model", id, urls, None)


class TestResolveModelPath:
    def test_default_location(self):
        path = resolve_model_path(Path("server"), Path("models/checkpoints/a.safetensors"), None)
        assert path == Path("server/models/checkpoints/a.safetensors")

    def test_shared_location(self):
        path = resolve_model_path(
            Path("server"), Path("models/checkpoints/a.safetensors"), Path("T:/models")
        )
        assert path == Path("T:/models/checkpoints/a.safetensors")

    def test_non_model_path_is_not_remapped(self):
        path = resolve_model_path(
            Path("server"), Path("custom_nodes/x/data.bin"), Path("T:/models")
        )
        assert path == Path("server/custom_nodes/x/data.bin")


class TestResolveModelPaths:
    def test_shared_location_comes_first(self):
        paths = resolve_model_paths(
            [Path("server"), Path("server/ComfyUI")],
            Path("models/checkpoints/a.safetensors"),
            Path("T:/models"),
        )
        assert paths == [
            Path("T:/models/checkpoints/a.safetensors"),
            Path("server/models/checkpoints/a.safetensors"),
            Path("server/ComfyUI/models/checkpoints/a.safetensors"),
        ]

    def test_default_lists_all_bases(self):
        paths = resolve_model_paths(
            [Path("server"), Path("server/ComfyUI")],
            Path("models/checkpoints/a.safetensors"),
            None,
        )
        assert paths == [
            Path("server/models/checkpoints/a.safetensors"),
            Path("server/ComfyUI/models/checkpoints/a.safetensors"),
        ]


class TestFindMissing:
    def test_default_finds_models_in_comfyui_dir(self, tmp_path: Path):
        # regression: with no shared directory configured, models in ComfyUI/models
        # must still be found (the pre-existing behavior)
        comfy_dir = tmp_path / "ComfyUI"
        (comfy_dir / "models" / "checkpoints").mkdir(parents=True)
        (comfy_dir / "models" / "checkpoints" / "a.safetensors").touch()

        res = _make_resource(["models/checkpoints/a.safetensors"])
        assert find_missing([tmp_path / "server", comfy_dir], [res]) == []

    def test_found_in_shared_location(self, tmp_path: Path):
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "a.safetensors").touch()

        res = _make_resource(["models/checkpoints/a.safetensors"])
        assert find_missing([Path("server")], [res], None, tmp_path) == []

    def test_found_in_previous_location_after_sharing_was_enabled(self, tmp_path: Path):
        # models left in the default location keep working after a shared
        # directory has been configured (they are not moved automatically)
        server_dir = tmp_path / "server"
        (server_dir / "models" / "checkpoints").mkdir(parents=True)
        (server_dir / "models" / "checkpoints" / "a.safetensors").touch()

        res = _make_resource(["models/checkpoints/a.safetensors"])
        assert find_missing([server_dir, Path("nowhere")], [res], None, tmp_path / "shared") == []

    def test_missing_when_not_in_any_location(self, tmp_path: Path):
        res = _make_resource(["models/checkpoints/does_not_exist.safetensors"])
        assert find_missing([tmp_path], [res], None, Path("T:/nowhere")) == [res.id.string]


class TestVerify:
    def test_verifies_file_in_previous_location(self, tmp_path: Path):
        # the file is verified where it was found, not at the primary location
        server_dir = tmp_path / "server"
        file_path = server_dir / "models" / "checkpoints" / "a.safetensors"
        file_path.parent.mkdir(parents=True)
        file_path.write_bytes(b"model data")

        sha256 = hashlib.sha256(b"model data").hexdigest()
        res = _make_resource(["models/checkpoints/a.safetensors"], sha256=sha256)
        statuses = list(res.verify(server_dir, tmp_path / "shared"))
        assert [s.state for s in statuses] == [
            VerificationState.in_progress,
            VerificationState.verified,
        ]
        assert statuses[-1].path == file_path


class TestExtraModelPathsSection:
    def test_shared_location_section(self):
        text = _extra_model_paths_section(Path("T:/models"))
        assert 'base_path: "T:/models"' in text
        assert "is_default: true" in text
        assert "checkpoints: checkpoints" in text
        assert "model_patches: model_patches" in text
        assert "\\" not in text
        # the previous (server-local) location stays visible to the server
        assert "krita-managed-local:" in text
        assert 'base_path: "../models"' in text

    def test_default_section(self):
        text = _extra_model_paths_section(None)
        assert 'base_path: "../models"' in text
        assert "is_default" not in text
        assert "krita-managed-local" not in text

    def test_paths_with_special_characters_are_quoted(self):
        text = _extra_model_paths_section(Path("T:/my models #2"))
        assert 'base_path: "T:/my models #2"' in text


class TestRemoveManagedSection:
    def test_remove_both_managed_sections(self):
        contents = (
            "other:\n"
            "    base_path: y\n"
            "\n"
            "krita-managed:\n"
            '    base_path: "T:/models"\n'
            "\n"
            "krita-managed-local:\n"
            '    base_path: "../models"\n'
            "    vae: vae\n"
            "\n"
            "final:\n"
            "    base_path: z\n"
        )
        result = _remove_managed_section(contents)
        assert "krita-managed" not in result
        assert "base_path: y" in result
        assert "base_path: z" in result

    def test_rebuild_roundtrip(self):
        contents = 'krita-managed:\n    base_path: "../models"\n    vae: vae\n'
        rebuilt = _remove_managed_section(contents) + _extra_model_paths_section(Path("T:/models"))
        assert rebuilt.count("krita-managed:") == 1
        assert 'base_path: "T:/models"' in rebuilt
        assert "is_default: true" in rebuilt


class TestConfigureExtraModelPaths:
    def test_fresh_install_with_shared_location(self, tmp_path: Path):
        comfy_dir = tmp_path / "ComfyUI"
        comfy_dir.mkdir()
        (comfy_dir / "extra_model_paths.yaml.example").write_text("# example\n")

        _configure_extra_model_paths(comfy_dir, tmp_path / "shared")

        contents = (comfy_dir / "extra_model_paths.yaml").read_text()
        assert contents.startswith("# example\n")
        assert f'base_path: "{(tmp_path / "shared").as_posix()}"' in contents
        assert "is_default: true" in contents
        assert "krita-managed-local:" in contents
        assert "model_patches: model_patches" in contents

    def test_clearing_the_setting_restores_default_location(self, tmp_path: Path):
        comfy_dir = tmp_path / "ComfyUI"
        comfy_dir.mkdir()
        yaml_path = comfy_dir / "extra_model_paths.yaml"
        yaml_path.write_text(_extra_model_paths_section(tmp_path / "shared"))

        _configure_extra_model_paths(comfy_dir, None)

        contents = yaml_path.read_text()
        assert 'base_path: "../models"' in contents
        assert "is_default" not in contents
        assert "krita-managed-local" not in contents
