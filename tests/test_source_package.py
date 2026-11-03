import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.source_package import build_source, collect_files


def test_source_archive_records_exact_bytes_without_private_files(tmp_path: Path):
    (tmp_path / "ai_diffusion").mkdir()
    (tmp_path / "ai_diffusion/__init__.py").write_text("version = 'test'\n")
    (tmp_path / "LICENSE").write_text("license")
    (tmp_path / ".appdata").mkdir()
    (tmp_path / ".appdata/secret.txt").write_text("private")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/.env").write_text("secret")
    archive = build_source(tmp_path, "test")
    manifest = json.loads((tmp_path / "SOURCE-MANIFEST.json").read_text())
    assert (
        manifest["files"]["ai_diffusion/__init__.py"]
        == hashlib.sha256((tmp_path / "ai_diffusion/__init__.py").read_bytes()).hexdigest()
    )
    with ZipFile(archive) as z:
        assert not any("secret" in p or ".env" in p for p in z.namelist())
        assert (
            z.read("krita_ai_diffusion-test-source/ai_diffusion/__init__.py")
            == (tmp_path / "ai_diffusion/__init__.py").read_bytes()
        )
    assert collect_files(tmp_path) == sorted(manifest["files"])
