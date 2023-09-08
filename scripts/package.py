import sys
from shutil import rmtree, copy, copytree, ignore_patterns, make_archive
from pathlib import Path
from zipfile import ZipFile

root = Path(__file__).parent.parent
package_dir = root / "scripts" / ".package"

sys.path.append(str(root))
import ai_diffusion

version = ai_diffusion.__version__
package_name = f"krita_ai_diffusion-{version}"


def build_package():
    rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir()
    copy(root / "ai_diffusion.desktop", package_dir)

    plugin_src = root / "ai_diffusion"
    plugin_dst = package_dir / "ai_diffusion"

    def ignore(path, names):
        filtered = ignore_patterns(".*", "*.json", "*.pyc", "__pycache__")(path, names)
        if path.endswith("styles"):
            filtered.remove("cinematic-photo.json")
            filtered.remove("digital-artwork.json")
        return filtered

    copytree(plugin_src, plugin_dst, ignore=ignore)
    copy(root / "README.md", plugin_dst)
    copy(root / "LICENSE", plugin_dst)

    make_archive(root / package_name, "zip", package_dir)


if __name__ == "__main__":
    print("Building package", root / package_name)
    build_package()
