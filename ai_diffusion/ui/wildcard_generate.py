from __future__ import annotations

import asyncio
import random
import re
import time

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..backend import ollama
from ..backend.network import NetworkError
from ..localization import translate as _
from ..model.root import root
from ..settings import settings
from ..util import client_logger as log
from ..util import user_data_dir
from ..wildcards import WildcardLibrary
from . import theme

_name_pattern = re.compile(r"[^a-z0-9_-]+")


def wildcard_file_name(category: str) -> str:
    slug = _name_pattern.sub("-", category.strip().splitlines()[0].lower()).strip("-")
    return f"ai/{slug}" if slug else "ai/terms"


class WildcardGenerator(QWidget):
    """Writes a wildcard file of interchangeable options for one category - hairstyles,
    poses, outfits - using the local language model.

    Sits next to the library rather than in the prompt batch dialog: the result is a
    file for the library, not a prompt for the current generation, and it lands right
    where it can be reviewed and edited.
    """

    wildcard_created = pyqtSignal(str)  # name, as used in __name__

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._results: list[str] = []
        self._job: ollama.Generation | None = None
        self._running = False
        self._cancelled = False

        self._category = QLineEdit(self)
        self._category.setPlaceholderText(_("hairstyles / hair colors / poses / outfits"))
        self._category.textChanged.connect(self._update_file_name)
        self._category.returnPressed.connect(self._start_or_stop)

        category_row = QHBoxLayout()
        category_row.addWidget(QLabel(_("Category") + ":", self))
        category_row.addWidget(self._category, 1)

        self._count = QSpinBox(self)
        self._count.setRange(5, 200)
        self._count.setValue(30)
        self._count.setToolTip(_("How many options to write"))

        self._file_name = QLineEdit("ai/terms", self)
        self._file_name.setToolTip(_("Name of the wildcard file, folders allowed"))

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel(_("Count") + ":", self))
        name_row.addWidget(self._count)
        name_row.addSpacing(12)
        name_row.addWidget(QLabel(_("File") + ":", self))
        name_row.addWidget(QLabel("__", self))
        name_row.addWidget(self._file_name, 1)
        name_row.addWidget(QLabel("__", self))

        self._progress = QProgressBar(self)
        self._progress.setVisible(False)
        self._status = QLabel(self)
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"font-style: italic; color: {theme.grey};")

        self._list = QListWidget(self)
        self._list.setAlternatingRowColors(True)

        self._start_button = QPushButton(_("Generate"), self)
        self._start_button.clicked.connect(self._start_or_stop)
        self._save_button = QPushButton(_("Save as Wildcard File"), self)
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._save)
        self._remove_button = QPushButton(_("Remove Selected"), self)
        self._remove_button.setToolTip(_("Drop entries you do not want before saving"))
        self._remove_button.clicked.connect(self._remove_selected)

        button_row = QHBoxLayout()
        button_row.addWidget(self._start_button)
        button_row.addWidget(self._remove_button)
        button_row.addStretch(1)
        button_row.addWidget(self._save_button)

        layout = QVBoxLayout()
        layout.addLayout(category_row)
        layout.addLayout(name_row)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        layout.addWidget(self._list, 1)
        layout.addLayout(button_row)
        self.setLayout(layout)

    def _update_file_name(self):
        if not self._running:
            self._file_name.setText(wildcard_file_name(self._category.text() or ""))

    def _start_or_stop(self):
        if self._running:
            self._cancelled = True
            if self._job is not None:
                self._job.cancel()
        elif not settings.ollama_model:
            self._status.setText(_("No language model selected (Settings -> Integrations)"))
        elif not self._category.text().strip():
            self._status.setText(_("Enter a category first"))
        else:
            eventloop.run(self._run())

    async def _run(self):
        model = root.active_model
        if model is None:
            return
        family = model.active_style.effective_family(model.arch)
        profile = ollama.Profiles.instance().for_family(family)
        if profile is None:
            self._status.setText(_("No prompt profile found for") + f" {family}")
            return

        category = self._category.text().strip()
        count = self._count.value()
        self._running = True
        self._cancelled = False
        self._results = []
        self._list.clear()
        self._save_button.setEnabled(False)
        self._start_button.setText(_("Stop"))
        self._progress.setVisible(True)
        self._progress.setRange(0, count)
        self._progress.setValue(0)
        started = time.monotonic()
        try:
            await self._write_terms(profile, category, count, started)
        except asyncio.CancelledError:
            pass  # stopped by the user
        except NetworkError as e:
            self._status.setText(_("Could not reach Ollama") + f" ({ollama.url()}): {e}")
        except Exception as e:
            log.exception("Wildcard generation failed")
            self._status.setText(_("Wildcard generation failed") + f": {e}")
        finally:
            self._running = False
            self._job = None
            self._start_button.setText(_("Generate"))
            self._progress.setVisible(False)
            self._save_button.setEnabled(len(self._results) > 0)
            if self._results:
                elapsed = time.monotonic() - started
                self._status.setText(
                    f"{len(self._results)} {_('entries')} - {elapsed:.0f}s - "
                    + _("review the list, then save")
                )

    async def _write_terms(
        self, profile: ollama.Profile, category: str, count: int, started: float
    ):
        """Written in rounds rather than one call per entry: entries are a few words
        each, so one call at a time would spend all the time on model overhead. Rounds
        repeat until the target count is reached; duplicates are dropped, and a model
        that only repeats itself ends the run instead of looping forever."""
        seen: set[str] = set()
        empty_rounds = 0
        while len(self._results) < count and not self._cancelled and empty_rounds < 3:
            missing = count - len(self._results)
            self._job = ollama.Generation()
            self._status.setText(
                _("Writing options")
                + f" {len(self._results)}/{count} - {time.monotonic() - started:.0f}s"
            )
            text = await self._job.run(
                ollama.build_terms_prompt(category, min(max(missing, 5), 25), self._results),
                system=ollama.terms_system_prompt(),
                model=profile.model,
                seed=random.randint(0, 2**31 - 1),
                keep_alive=300,
            )
            added = 0
            for term in ollama.parse_terms(text):
                key = ollama.normalize_term(term)
                if not key or key in seen:
                    continue
                seen.add(key)
                self._results.append(term)
                QListWidgetItem(term, self._list)
                added += 1
                if len(self._results) >= count:
                    break
            self._progress.setValue(len(self._results))
            empty_rounds = 0 if added else empty_rounds + 1
        await ollama.unload()

    def _remove_selected(self):
        for item in self._list.selectedItems():
            row = self._list.row(item)
            self._list.takeItem(row)
            if 0 <= row < len(self._results):
                self._results.pop(row)
        self._save_button.setEnabled(len(self._results) > 0)

    def _save(self):
        if not self._results:
            return
        name = _name_pattern.sub("-", self._file_name.text().strip().lower().replace("\\", "/"))
        name = "/".join(part for part in name.split("/") if part)
        if not name:
            self._status.setText(_("Enter a name for the wildcard file"))
            return
        path = user_data_dir / "wildcards" / f"{name}.txt"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(self._results) + "\n", encoding="utf-8")
        except Exception as e:
            self._status.setText(_("Could not write") + f" {path}: {e}")
            return
        WildcardLibrary.instance().reload()
        self._status.setText(_("Saved") + f" {len(self._results)} {_('entries')} - {path}")
        self.wildcard_created.emit(name)

    def shutdown(self):
        self._cancelled = True
        if self._job is not None:
            self._job.cancel()
