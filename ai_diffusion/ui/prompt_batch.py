from __future__ import annotations

import asyncio
import random
import re
import time

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..backend import ollama
from ..backend.network import NetworkError
from ..backend.ollama import PoolMode
from ..localization import translate as _
from ..model.region import Region, RootRegion
from ..model.root import root
from ..settings import settings
from ..text import format_option_group
from ..util import client_logger as log
from ..util import user_data_dir
from ..wildcards import WildcardLibrary
from . import theme

_name_pattern = re.compile(r"[^a-z0-9_-]+")


class PromptBatchDialog(QDialog):
    """Produces a pool of prompts up front - one Ollama call per entry - and puts it
    into the prompt as a sequential wildcard group or a wildcard file. Image generation
    only starts once the pool is complete, so the language model and the diffusion model
    never compete for VRAM."""

    def __init__(self, region: RootRegion | Region, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle(_("Prompt Batch"))
        self.setModal(False)

        self._region = region
        self._results: list[str] = []
        self._job: ollama.Generation | None = None
        self._running = False
        self._cancelled = False
        self._protected = ollama.protect(region.positive)

        self._variation_mode = QRadioButton(_("Vary current prompt"), self)
        self._variation_mode.setChecked(bool(self._protected.text))
        self._random_mode = QRadioButton(_("Random for a theme"), self)
        self._random_mode.setChecked(not self._protected.text)
        self._variation_mode.toggled.connect(self._update_mode)
        # radio buttons sharing a parent are auto-exclusive as one group - without
        # explicit groups, picking a mode would clear the output choice and vice versa
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._variation_mode)
        self._mode_group.addButton(self._random_mode)

        self._base_label = QLabel(self)
        self._base = QPlainTextEdit(self._protected.text, self)
        self._base.setFixedHeight(3 * self.fontMetrics().lineSpacing() + 10)

        self._count = QSpinBox(self)
        self._count.setRange(2, 200)
        self._count.setValue(max(2, settings.ollama_variation_count))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row.addWidget(self._variation_mode)
        mode_row.addWidget(self._random_mode)
        mode_row.addStretch()

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel(_("Count") + ":", self))
        count_row.addWidget(self._count)
        count_row.addStretch()

        self._as_group = QRadioButton(_("Sequential group [[a|b|c]]"), self)
        self._as_group.setChecked(True)
        self._as_file = QRadioButton(_("Wildcard file"), self)
        self._file_name = QLineEdit("ai/batch", self)
        self._file_name.setEnabled(False)
        self._as_file.toggled.connect(self._file_name.setEnabled)
        self._output_group = QButtonGroup(self)
        self._output_group.addButton(self._as_group)
        self._output_group.addButton(self._as_file)
        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        output_row.addWidget(self._as_group)
        output_row.addWidget(self._as_file)
        output_row.addStretch()

        file_row = QHBoxLayout()
        file_row.addSpacing(20)
        file_row.addWidget(QLabel("__", self))
        file_row.addWidget(self._file_name, 1)
        file_row.addWidget(QLabel("__", self))

        self._set_batch_count = QCheckBox(_("Set batch count"), self)
        self._set_batch_count.setChecked(True)
        self._generate_after = QCheckBox(_("Generate images when finished"), self)

        check_row = QHBoxLayout()
        check_row.setSpacing(8)
        check_row.addWidget(self._set_batch_count)
        check_row.addWidget(self._generate_after)
        check_row.addStretch()

        self._progress = QProgressBar(self)
        self._progress.setVisible(False)
        self._status = QLabel(self)
        self._status.setStyleSheet(f"font-style: italic; color: {theme.grey};")

        self._list = QListWidget(self)
        self._list.setAlternatingRowColors(True)
        self._list.setMinimumHeight(110)
        self._list.setVisible(False)  # only takes up space once there are results

        self._start_button = QPushButton(_("Generate"), self)
        self._start_button.clicked.connect(self._start_or_stop)
        self._apply_button = QPushButton(_("Apply"), self)
        self._apply_button.setEnabled(False)
        self._apply_button.clicked.connect(self._apply)
        close_button = QPushButton(_("Close"), self)
        close_button.clicked.connect(self.reject)
        button_row = QHBoxLayout()
        button_row.setSpacing(4)
        button_row.addWidget(self._start_button)
        button_row.addWidget(self._apply_button)
        button_row.addStretch()
        button_row.addWidget(close_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addLayout(mode_row)
        layout.addWidget(self._base_label)
        layout.addWidget(self._base)
        layout.addLayout(count_row)
        layout.addSpacing(4)
        layout.addLayout(output_row)
        layout.addLayout(file_row)
        layout.addSpacing(4)
        layout.addLayout(check_row)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        layout.addWidget(self._list, 1)
        layout.addLayout(button_row)
        self.setLayout(layout)

        self._update_mode()
        layout.activate()
        self.adjustSize()
        self.resize(QSize(min(self.width(), 360), min(self.height(), 260)))
        if window := parent.window():
            center = window.frameGeometry().center()
            self.move(center.x() - self.width() // 2, center.y() - self.height() // 2)

    def _update_mode(self):
        if self._variation_mode.isChecked():
            self._base_label.setText(_("Prompt to vary") + ":")
        else:
            self._base_label.setText(_("Theme or idea") + ":")

    @property
    def _mode(self):
        return PoolMode.variation if self._variation_mode.isChecked() else PoolMode.random

    def _start_or_stop(self):
        if self._running:
            self._cancelled = True
            if self._job is not None:
                self._job.cancel()
        elif not settings.ollama_model:
            self._status.setText(_("No language model selected (Settings -> Prompt AI)"))
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

        count = self._count.value()
        base = self._base.toPlainText().strip()
        if not base:
            self._status.setText(_("Enter a prompt, a theme or a category first"))
            return
        self._running = True
        self._cancelled = False
        self._results = []
        self._list.clear()
        if not self._list.isVisible():  # make room for it instead of squeezing the form
            self._list.setVisible(True)
            self.resize(self.width(), self.height() + self._list.minimumHeight() + 8)
        self._apply_button.setEnabled(False)
        self._start_button.setText(_("Stop"))
        self._progress.setVisible(True)
        self._progress.setRange(0, count)
        self._progress.setValue(0)
        started = time.monotonic()

        try:
            await self._run_prompts(profile, base, count, started)
        except asyncio.CancelledError:
            pass  # stopped by the user
        except NetworkError as e:
            self._status.setText(_("Could not reach Ollama") + f" ({ollama.url()}): {e}")
        except Exception as e:
            log.exception("Prompt batch failed")
            self._status.setText(_("Prompt batch failed") + f": {e}")
        finally:
            self._running = False
            self._job = None
            self._start_button.setText(_("Generate"))
            self._progress.setVisible(False)
            self._apply_button.setEnabled(len(self._results) > 0)
            if self._results:
                elapsed = time.monotonic() - started
                self._status.setText(f"{len(self._results)} {_('prompts')} - {elapsed:.0f}s")
            if not self._cancelled and self._results:
                self._apply()

    async def _run_prompts(self, profile: ollama.Profile, base: str, count: int, started: float):
        for i in range(count):
            if self._cancelled:
                break
            # The model is kept loaded across the whole batch: loading it costs about as
            # much as writing one prompt, and it is unloaded again at the end.
            last = i == count - 1
            request = ollama.build_pool_prompt(self._mode, base, self._results[-3:])
            self._job = ollama.Generation()
            self._status.setText(
                _("Writing prompt") + f" {i + 1}/{count} - {time.monotonic() - started:.0f}s"
            )
            text = await self._job.run(
                request,
                system=profile.system,
                model=profile.model,
                seed=random.randint(0, 2**31 - 1),
                keep_alive=0 if last else 300,
            )
            text = _first_prompt(text)
            if text and text not in self._results:
                self._results.append(text)
                QListWidgetItem(text, self._list)
            self._progress.setValue(i + 1)

    def _apply(self):
        if not self._results:
            return
        if self._as_file.isChecked():
            prompt = self._write_wildcard_file()
            if prompt is None:
                return
        else:
            prompt = format_option_group(self._results, sequential=True)
        # protected tokens (LoRA tags, wildcards) sit outside the group, so they apply to
        # every entry instead of being repeated inside each one
        self._region.positive = self._protected.restore(prompt)

        model = root.active_model
        if model is not None and self._set_batch_count.isChecked():
            model.batch_count = min(len(self._results), 1000)
        if model is not None and self._generate_after.isChecked():
            model.generate()
            self.accept()

    def _write_wildcard_file(self):
        name = _name_pattern.sub("-", self._file_name.text().strip().lower().replace("\\", "/"))
        name = "/".join(part for part in name.split("/") if part)
        if not name:
            self._status.setText(_("Enter a name for the wildcard file"))
            return None
        path = user_data_dir / "wildcards" / f"{name}.txt"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(self._results) + "\n", encoding="utf-8")
        except Exception as e:
            self._status.setText(_("Could not write") + f" {path}: {e}")
            return None
        WildcardLibrary.instance().reload()
        self._status.setText(_("Saved") + f" {len(self._results)} {_('prompts')} - {path}")
        return f"__{name}__"

    def closeEvent(self, a0):
        self._cancelled = True
        if self._job is not None:
            self._job.cancel()
        return super().closeEvent(a0)


def _first_prompt(text: str) -> str:
    """Models sometimes answer with several lines even when asked for one prompt -
    keep the first non-empty one and drop list markers."""
    for line in text.splitlines():
        line = line.strip(" -*\t")
        if line:
            return line
    return ""
