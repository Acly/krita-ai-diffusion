"""File-based wildcards: __name__ in a prompt picks a random line from
<user_data_dir>/wildcards/name.txt (or wildcards/folder/name.txt for
__folder/name__), matching the convention used by collections like
https://github.com/mattjaybe/sd-wildcards - drop such a repo's `wildcards/`
folder in directly and it works with no extra setup."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .util import client_logger as log
from .util import user_data_dir


class WildcardLibrary:
    default_folder = user_data_dir / "wildcards"

    def __init__(self, folder: Path | None = None):
        self.folder = folder or self.default_folder
        self._entries: dict[str, list[str]] = {}
        self._paths: dict[str, Path] = {}
        self.reload()

    def reload(self):
        """Rescan the wildcards folder. Call after adding/removing files."""
        entries: dict[str, list[str]] = {}
        paths: dict[str, Path] = {}
        if self.folder.exists():
            for path in self.folder.rglob("*.txt"):
                name = path.relative_to(self.folder).with_suffix("").as_posix().lower()
                lines = _read_lines(path)
                if lines:
                    entries[name] = lines
                    paths[name] = path
                else:
                    log.warning(f"Wildcard file has no usable lines: {path}")
        self._entries = entries
        self._paths = paths

    def get(self, name: str) -> list[str] | None:
        return self._entries.get(name.lower())

    def path_for(self, name: str) -> Path | None:
        return self._paths.get(name.lower())

    def names(self) -> list[str]:
        return sorted(self._entries.keys())

    def save(self, name: str, lines: list[str]) -> bool:
        """Overwrite the file's content, one line per option."""
        path = self.path_for(name)
        if path is None:
            return False
        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            log.warning(f"Could not write wildcard file {path}: {e}")
            return False
        self.reload()
        return True

    def create(self, name: str, lines: list[str]) -> bool:
        """Create a new wildcard file. Fails if a file with that name already exists.
        `name` may include a `folder/` prefix like the __folder/name__ prompt syntax."""
        name = name.strip().strip("/\\")
        if not name or not lines:
            return False
        self.folder.mkdir(parents=True, exist_ok=True)
        path = (self.folder / name).with_suffix(".txt")
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(self.folder.resolve()):
                return False
        except Exception:
            return False
        if path.exists():
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            log.warning(f"Could not create wildcard file {path}: {e}")
            return False
        self.reload()
        return True

    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename/move a wildcard file, keeping its content. `new_name` may include
        a `folder/` prefix like the __folder/name__ prompt syntax to move it."""
        old_path = self.path_for(old_name)
        if old_path is None:
            return False
        new_name = new_name.strip().strip("/\\")
        if not new_name:
            return False
        new_path = (self.folder / new_name).with_suffix(".txt")
        try:
            resolved = new_path.resolve()
            if not resolved.is_relative_to(self.folder.resolve()):
                return False
        except Exception:
            return False
        if new_path.exists():
            return False
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)
        except Exception as e:
            log.warning(f"Could not rename wildcard file {old_path} -> {new_path}: {e}")
            return False
        self.reload()
        return True

    def delete(self, name: str, move_to_trash: Callable[[str], bool] | None = None) -> bool:
        """Remove a wildcard file. `move_to_trash` (e.g. QFile.moveToTrash) keeps a slip
        recoverable; the file is deleted outright only if it's missing or fails.
        Folders left empty are removed as well."""
        path = self.path_for(name)
        if path is None:
            return False
        try:
            if not (move_to_trash and move_to_trash(str(path))):
                path.unlink()
            parent = path.parent
            while parent != self.folder and parent.is_relative_to(self.folder):
                if any(parent.iterdir()):
                    break
                parent.rmdir()
                parent = parent.parent
        except Exception as e:
            log.warning(f"Could not delete wildcard file {path}: {e}")
            return False
        finally:
            self.reload()
        return not path.exists()

    def __len__(self):
        return len(self._entries)

    _instance: WildcardLibrary | None = None

    @classmethod
    def instance(cls) -> WildcardLibrary:
        if cls._instance is None:
            cls._instance = WildcardLibrary()
        return cls._instance


def _read_lines(path: Path) -> list[str]:
    try:
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
        return lines
    except Exception as e:
        log.warning(f"Could not read wildcard file {path}: {e}")
        return []
