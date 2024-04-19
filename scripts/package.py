import sys
from markdown import markdown
from shutil import rmtree, copy, copytree, ignore_patterns, make_archive
from pathlib import Path

root = Path(__file__).parent.parent
package_dir = root / "scripts" / ".package"

sys.path.append(str(root))
import ai_diffusion

version = ai_diffusion.__version__
package_name = f"krita_ai_diffusion-{version}"


def convert_markdown_to_html(markdown_file: Path, html_file: Path):
    with open(markdown_file, "r", encoding="utf-8") as f:
        text = f.read()
    html = markdown(text, extensions=["fenced_code", "codehilite"])
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)


def build_package():
    rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir()
    copy(root / "ai_diffusion.desktop", package_dir)

    plugin_src = root / "ai_diffusion"
    plugin_dst = package_dir / "ai_diffusion"

    def ignore(path, names):
        return ignore_patterns(".*", "*.pyc", "__pycache__")(path, names)

    copytree(plugin_src, plugin_dst, ignore=ignore)
    copy(root / "scripts" / "download_models.py", plugin_dst)
    copy(root / "LICENSE", plugin_dst)
    convert_markdown_to_html(root / "README.md", plugin_dst / "manual.html")

    make_archive(str(root / package_name), "zip", package_dir)


if __name__ == "__main__":
    print("Building package", root / package_name)
    build_package()
