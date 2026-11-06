from pathlib import Path

from ai_diffusion.backend.resources import (
    ModelFile,
    ModelResource,
    ResourceId,
    ResourceKind,
    resolve_model_path,
)
from ai_diffusion.backend.server import (
    _extra_model_paths_section,
    _remove_managed_section,
    find_missing,
)
from ai_diffusion.style import Arch


def _make_resource(files: list[str]) -> ModelResource:
    id = ResourceId(ResourceKind.checkpoint, Arch.sdxl, "test-model")
    urls = [ModelFile(Path(p), "https://example.com/x", id) for p in files]
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


class TestFindMissing:
    def test_found_in_shared_location(self, tmp_path: Path):
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "a.safetensors").touch()

        res = _make_resource(["models/checkpoints/a.safetensors"])
        assert find_missing([Path("server")], [res], None, tmp_path) == []

    def test_missing_when_not_in_shared_location(self, tmp_path: Path):
        res = _make_resource(["models/checkpoints/does_not_exist.safetensors"])
        assert find_missing([tmp_path], [res], None, Path("T:/nowhere")) == [res.id.string]


class TestExtraModelPathsSection:
    def test_shared_location_section(self):
        text = _extra_model_paths_section(Path("T:/models"))
        assert "krita-managed:" in text
        assert "base_path: T:/models" in text
        assert "is_default: true" in text
        assert "checkpoints: checkpoints" in text
        assert "model_patches: model_patches" in text
        assert "\\" not in text

    def test_default_section(self):
        text = _extra_model_paths_section(None)
        assert "base_path: ../models" in text
        assert "is_default" not in text


class TestRemoveManagedSection:
    def test_remove_section_between_others(self):
        contents = (
            "comfyui:\n"
            "    base_path: x\n"
            "\n"
            "krita-managed:\n"
            "    base_path: ../models\n"
            "    vae: vae\n"
            "\n"
            "other:\n"
            "    base_path: y\n"
        )
        result = _remove_managed_section(contents)
        assert "krita-managed" not in result
        assert "base_path: x" in result
        assert "base_path: y" in result

    def test_rebuild_roundtrip(self):
        contents = "krita-managed:\n    base_path: ../models\n    vae: vae\n"
        rebuilt = _remove_managed_section(contents) + _extra_model_paths_section(Path("T:/models"))
        assert rebuilt.count("krita-managed:") == 1
        assert "base_path: T:/models" in rebuilt
        assert "is_default: true" in rebuilt
