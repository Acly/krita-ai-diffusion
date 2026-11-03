"""A small library of hand-picked prompt snippets the user wants to reuse, independent
of style/checkpoint (unlike a Recipe, which bundles a checkpoint + LoRA stack + prompt
fetched from the ComfyUI-Lora-Manager server). Stored as a single local JSON file, same
convention as prompt_enhance.json."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field

from PyQt5.QtCore import QObject, pyqtSignal

from .image import Extent, Image
from .util import client_logger as log
from .util import user_data_dir


@dataclass
class PromptEntry:
    id: str
    title: str
    text: str
    negative: str = ""
    category: str = ""
    favorite: bool = False
    created: float = field(default_factory=time.time)
    last_used: float = 0.0


class PromptLibrary(QObject):
    """Singleton, same pattern as Styles: load once, mutate in place, emit `changed`
    so every open picker/dialog stays in sync."""

    changed = pyqtSignal()

    _instance: PromptLibrary | None = None
    default_path = user_data_dir / "prompts.json"
    default_preview_folder = user_data_dir / "prompt_previews"

    def __init__(self, path=None, preview_folder=None):
        super().__init__()
        self.path = path or self.default_path
        self.preview_folder = preview_folder or self.default_preview_folder
        self._entries: dict[str, PromptEntry] = {}
        self.reload()

    @classmethod
    def instance(cls) -> PromptLibrary:
        if cls._instance is None:
            cls._instance = PromptLibrary()
        return cls._instance

    def reload(self):
        entries: dict[str, PromptEntry] = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for item in data.get("entries") or []:
                    entry = PromptEntry(
                        id=item.get("id", "") or str(uuid.uuid4()),
                        title=item.get("title", ""),
                        text=item.get("text", ""),
                        negative=item.get("negative", ""),
                        category=item.get("category", ""),
                        favorite=bool(item.get("favorite", False)),
                        created=float(item.get("created", 0.0)),
                        last_used=float(item.get("last_used", 0.0)),
                    )
                    if entry.id and entry.title:
                        entries[entry.id] = entry
            except Exception as e:
                log.error(f"Failed to read prompt library from {self.path}: {e}")
        self._entries = entries

    def _save(self):
        data = {"entries": [asdict(e) for e in self._entries.values()]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            log.error(f"Failed to write prompt library to {self.path}: {e}")

    def entries(self) -> list[PromptEntry]:
        return list(self._entries.values())

    def get(self, id: str) -> PromptEntry | None:
        return self._entries.get(id)

    def categories(self) -> list[str]:
        return sorted({e.category for e in self._entries.values() if e.category})

    def add(self, title: str, text: str, negative: str = "", category: str = "") -> PromptEntry:
        entry = PromptEntry(
            id=str(uuid.uuid4()), title=title, text=text, negative=negative, category=category
        )
        self._entries[entry.id] = entry
        self._save()
        self.changed.emit()
        return entry

    def update(self, id: str, **fields) -> bool:
        entry = self._entries.get(id)
        if entry is None:
            return False
        for key, value in fields.items():
            setattr(entry, key, value)
        self._save()
        self.changed.emit()
        return True

    def remove(self, id: str) -> bool:
        if id not in self._entries:
            return False
        del self._entries[id]
        self._save()
        self.delete_preview(id)
        self.changed.emit()
        return True

    # -- preview thumbnail (one PNG file per entry, named by id - kept out of the
    # JSON so saving/parsing the library stays cheap even with many entries) --

    def preview_path(self, id: str):
        return self.preview_folder / f"{id}.png"

    def has_preview(self, id: str) -> bool:
        return self.preview_path(id).exists()

    def save_preview(self, id: str, image: Image, max_size: int = 160):
        scaled = Image.scale_to_fit(image, Extent(max_size, max_size))
        try:
            self.preview_folder.mkdir(parents=True, exist_ok=True)
            scaled.save(self.preview_path(id))
        except Exception as e:
            log.error(f"Failed to write prompt preview for {id}: {e}")

    def delete_preview(self, id: str):
        path = self.preview_path(id)
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            log.error(f"Failed to delete prompt preview for {id}: {e}")

    def mark_used(self, id: str):
        entry = self._entries.get(id)
        if entry is None:
            return
        entry.last_used = time.time()
        self._save()
        self.changed.emit()
