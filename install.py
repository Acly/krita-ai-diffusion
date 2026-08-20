#!/usr/bin/env python3

import os
import sys
from pathlib import Path


def _pykrita_path() -> Path | None:
    if sys.platform.startswith("linux"):
        return Path.home() / ".local" / "share" / "krita" / "pykrita"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Krita" / "pykrita"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "krita" / "pykrita"
        return Path.home() / "AppData" / "Roaming" / "krita" / "pykrita"
    return None


def main() -> None:
    pykrita = _pykrita_path()
    if pykrita is None:
        print("Error: Unsupported operating system:", sys.platform, file=sys.stderr)
        return

    if not pykrita.is_dir():
        print("Error: Krita pykrita folder does not exist:", pykrita, file=sys.stderr)
        return

    repository = Path(__file__).resolve().parent
    for name in ("ai_diffusion", "ai_diffusion.desktop"):
        source = repository / name
        link = pykrita / name
        if not link.exists() or link.is_symlink():
            link.unlink(missing_ok=True)
            link.symlink_to(source, target_is_directory=source.is_dir())
            print("Created symlink", link)
        else:
            print("Error: Existing file or folder at", link, file=sys.stderr)
            return


if __name__ == "__main__":
    main()
