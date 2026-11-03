from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root_files = {
    "LICENSE",
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "BUILD-REPORT.md",
    "THIRD-PARTY-NOTICES.md",
    "RELEASE-NOTES.md",
    "requirements.txt",
    "pyproject.toml",
    "ai_diffusion.desktop",
    "install.py",
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
}
source_folders = {"ai_diffusion", "scripts", "tests", "custom_workflows", "branding", "media"}


def allowed(path: Path):
    if len(path.parts) == 1:
        return path.name in root_files
    if path.parts[0] not in source_folders:
        return False
    if any(part.startswith(".") or part in ("__pycache__", "debugpy") for part in path.parts):
        return False
    return path.suffix not in (".pyc", ".zip", ".sha256")


def collect_files(root: Path):
    candidates: set[str] = set()
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        candidates.update(p.decode("utf-8") for p in result.stdout.split(b"\0") if p)
    elif (root / "SOURCE-MANIFEST.json").exists():
        manifest = json.loads((root / "SOURCE-MANIFEST.json").read_text(encoding="utf-8"))
        candidates.update(manifest["files"])
    else:
        candidates.update(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    # Submodule contents are part of the released plugin and must accompany its source.
    runtime = root / "ai_diffusion"
    if runtime.exists():
        candidates.update(p.relative_to(root).as_posix() for p in runtime.rglob("*") if p.is_file())
    candidates.update(name for name in root_files if (root / name).is_file())
    return sorted(name for name in candidates if allowed(Path(name)) and (root / name).is_file())


def build_source(root: Path, version: str):
    files = collect_files(root)
    manifest = {
        "format": "krita-ai-diffusion-source-v1",
        "version": version,
        "modified_date": "2026-11-03",
        "snapshot": "Exact working-copy sources; file hashes are authoritative, including uncommitted changes.",
        "provenance": {
            "Acly/krita-ai-diffusion": "0217cd2197fcadbd70d7e63af25e29cc21cb7c8b",
            "FatGuy84/krita-ai-reloaded": "08082efef6f2d4a90a80c32a90895a8c0e3f913d",
            "python-websockets/websockets": "624a36cc9c1ea1369971387d7bb23533b2797350",
        },
        "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (root / "SOURCE-MANIFEST.json").write_bytes(manifest_text.encode("utf-8"))
    prefix = f"krita_ai_diffusion-{version}-source"
    destination = root / f"{prefix}.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(root / name, f"{prefix}/{name}")
        archive.writestr(f"{prefix}/SOURCE-MANIFEST.json", manifest_text)
    return destination


def write_checksums(root: Path, version: str, artifacts: list[Path]):
    content = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in artifacts
    )
    destination = root / f"SHA256SUMS-{version}.txt"
    destination.write_text(content, encoding="utf-8")
    return destination
