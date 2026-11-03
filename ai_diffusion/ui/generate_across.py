from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, NamedTuple

from PyQt5.QtCore import QObject, QRect, Qt
from PyQt5.QtGui import QColor, QFont, QPainter
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QMessageBox, QWidget

from ..backend.client import ClientModels, resolve_arch
from ..backend.lora_manager import arch_for_base_model, load_cached_loras
from ..backend.resources import Arch
from ..image import Bounds, Extent, Image
from ..localization import translate as _
from ..model.jobs import Job, JobKind, JobState
from ..model.model import DocumentModel, _short_model_name
from ..model.root import root
from ..style import Style
from ..text import pattern_lora
from ..util import trim_text


def short_name(checkpoint: str) -> str:
    return Path(checkpoint.replace("\\", "/")).stem


def _lora_key(name: str) -> str:
    return short_name(name).lower()


class LoraArchCheck:
    """Decides which of the user's LoRAs can run on a model of a given arch.

    Lora Manager's cached base model is preferred; LoRAs it doesn't know are assumed
    to fit the arch of the current style, since that's what they were picked for.
    LoRAs not in `user_loras` (a compared style's own, or sampling LoRAs like LCM)
    are always kept.
    """

    def __init__(self, user_loras: list[str], original_arch: Arch):
        self._user = {_lora_key(n) for n in user_loras}
        self._original = original_arch
        self._known: dict[str, Arch] = {}
        client = root.connection.client_if_connected
        for info in load_cached_loras(client.url if client else "") or []:
            if arch_name := arch_for_base_model(info.base_model):
                self._known[info.name.lower()] = Arch[arch_name]

    def fits(self, lora: str, arch: Arch) -> bool:
        key = _lora_key(lora)
        if key not in self._user:
            return True
        return Arch.is_compatible(self._known.get(key, self._original), arch)

    def dropped(self, arch: Arch) -> list[str]:
        return sorted(n for n in self._user if not self.fits(n, arch))


def _prompt_loras(model: DocumentModel):
    return [m[1] for m in pattern_lora.finditer(model.regions.positive)]


class AcrossOptions(QWidget):
    """Grid / layer toggles shown next to a "Generate across" button. Remembered for
    the session and shared by the style and checkpoint browsers."""

    _grid = True
    _layers = False

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.grid = QCheckBox(_("Grid"), self)
        self.grid.setChecked(AcrossOptions._grid)
        self.grid.setToolTip(_("When all are done, add one layer with the results side by side"))
        self.layers = QCheckBox(_("Layers"), self)
        self.layers.setChecked(AcrossOptions._layers)
        self.layers.setToolTip(_("Also add each result as a layer (they always go to the history)"))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.grid)
        layout.addWidget(self.layers)

    def remember(self):
        AcrossOptions._grid = self.grid.isChecked()
        AcrossOptions._layers = self.layers.isChecked()


class _Entry(NamedTuple):
    key: str  # what the finished job's metadata holds for this entry
    label: str  # name in layer names and above the grid cell
    arch: Arch
    apply: Callable[[], None]


class _CompareRun(QObject):
    """Collects the results of one "generate across" as the jobs finish, then adds
    them as layers and/or a labelled grid. Outlives the browser that started it."""

    _active: ClassVar[set[_CompareRun]] = set()

    def __init__(
        self,
        model: DocumentModel,
        entries: list[_Entry],
        key_of: Callable[[Job], str],
        seed: int,
        options: AcrossOptions,
    ):
        super().__init__()
        self._model = model
        self._entries = entries
        self._pending = {e.key for e in entries}
        self._key_of = key_of
        self._seed = seed
        self._as_layers = options.layers.isChecked()
        self._make_grid = options.grid.isChecked()
        self._started = datetime.now(timezone.utc)
        self._results: dict[str, Job] = {}
        if self._as_layers or self._make_grid:
            model.jobs.job_finished.connect(self._collect)
            model.jobs.count_changed.connect(self._check_abandoned)
            _CompareRun._active.add(self)

    def _collect(self, job: Job):
        key = self._key_of(job)
        if (
            job.kind is not JobKind.diffusion
            or key not in self._pending
            or job.timestamp < self._started
            or job.params.seed != self._seed
            or len(job.results) == 0
        ):
            return
        self._pending.discard(key)
        self._results[key] = job
        if not self._pending:
            self._finish()

    def _check_abandoned(self):
        # Jobs got cancelled: build from what arrived, once nothing is left to wait for
        jobs = self._model.jobs
        if self._results and jobs.count(JobState.queued) + jobs.count(JobState.executing) == 0:
            self._finish()

    def _finish(self):
        jobs = self._model.jobs
        jobs.job_finished.disconnect(self._collect)
        jobs.count_changed.disconnect(self._check_abandoned)
        _CompareRun._active.discard(self)
        done = [(e.label, self._results[e.key]) for e in self._entries if e.key in self._results]
        layers = self._model.layers
        if self._as_layers:
            for label, job in done:
                image = job.results[0]
                bounds = Bounds(*job.params.bounds.offset, *image.extent)
                layers.create(
                    f"[{label}] {trim_text(job.params.name, 200)} ({self._seed})", image, bounds
                )
        if self._make_grid and len(done) > 1:
            grid = build_grid([(label, job.results[0]) for label, job in done], self._model)
            prompt = trim_text(done[0][1].params.name, 200)
            layers.create(f"[Grid] {prompt} ({self._seed})", grid, Bounds(0, 0, *grid.extent))


def build_grid(entries: list[tuple[str, Image]], model: DocumentModel) -> Image:
    """Images in a near-square grid with a name strip above each, scaled down so the
    whole grid fits the canvas."""
    cols = math.ceil(math.sqrt(len(entries)))
    rows = math.ceil(len(entries) / cols)
    first = entries[0][1].extent
    canvas = model.document.extent
    label_ratio = 0.08
    scale = min(
        1.0,
        canvas.width / (cols * first.width),
        canvas.height / (rows * first.height * (1 + label_ratio)),
    )
    cell_w = max(1, int(first.width * scale))
    cell_h = max(1, int(first.height * scale))
    label_h = max(18, int(cell_h * label_ratio))
    grid = Image.create(Extent(cols * cell_w, rows * (cell_h + label_h)), fill=QColor(24, 24, 24))

    painter = QPainter(grid._qimage)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    font = QFont()
    font.setPixelSize(max(10, int(label_h * 0.6)))
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor(235, 235, 235))
    metrics = painter.fontMetrics()
    for i, (name, image) in enumerate(entries):
        x, y = (i % cols) * cell_w, (i // cols) * (cell_h + label_h)
        text = metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, cell_w - 8)
        painter.drawText(QRect(x, y, cell_w, label_h), Qt.AlignmentFlag.AlignCenter, text)
        painter.drawImage(QRect(x, y + label_h, cell_w, cell_h), image._qimage)
    painter.end()
    return grid


def _confirm_dropped(entries: list[_Entry], check: LoraArchCheck, parent: QWidget | None):
    lines = [f"{e.label}: {', '.join(d)}" for e in entries if (d := check.dropped(e.arch))]
    if not lines:
        return True
    answer = QMessageBox.question(
        parent,
        _("Generate across"),
        _("These LoRAs belong to another architecture and are left out for:")
        + "\n\n"
        + "\n".join(lines)
        + "\n\n"
        + _("Continue?"),
        QMessageBox.Yes | QMessageBox.Cancel,
    )
    return answer == QMessageBox.StandardButton.Yes


def _run(
    model: DocumentModel,
    entries: list[_Entry],
    key_of: Callable[[Job], str],
    seed: int,
    check: LoraArchCheck,
    restore: Callable[[], None],
    options: AcrossOptions,
    parent: QWidget | None,
):
    if not _confirm_dropped(entries, check, parent):
        return False
    options.remember()
    original_batch = model.batch_count

    def prepare(entry: _Entry):
        def apply():
            entry.apply()
            model.batch_count = 1
            model.lora_filter = lambda lora: check.fits(lora, entry.arch)

        return apply

    def restore_all():
        restore()
        model.batch_count = original_batch
        model.lora_filter = None

    _CompareRun(model, entries, key_of, seed, options)
    model.generate_across([prepare(e) for e in entries], seed, restore_all)
    return True


def generate_across_checkpoints(
    model: DocumentModel,
    models: ClientModels,
    checkpoints: list[str],
    seed: int,
    options: AcrossOptions,
    parent: QWidget | None = None,
):
    """Current prompt and style once per checkpoint: only the checkpoint changes. The
    style's own LoRAs are checked too, they may not fit another checkpoint's arch."""
    style = model.active_style
    style_loras = [str(l["name"]) for l in style.loras if l.get("enabled", True)]
    check = LoraArchCheck(style_loras + _prompt_loras(model), resolve_arch(style, models))
    original = (list(style.checkpoints), style.architecture)

    def use(checkpoint: str, arch: Arch):
        def apply():
            style.checkpoints = [checkpoint]
            # a pinned arch that doesn't fit would force the wrong pipeline
            if not Arch.is_compatible(original[1], arch):
                style.architecture = Arch.auto

        return apply

    def restore():
        style.checkpoints, style.architecture = original

    entries = []
    for c in checkpoints:
        arch = models.arch_of(c)
        entries.append(_Entry(c, _short_model_name(c), arch, use(c, arch)))
    return _run(
        model,
        entries,
        lambda job: job.params.metadata.get("checkpoint", ""),
        seed,
        check,
        restore,
        options,
        parent,
    )


def generate_across_styles(
    model: DocumentModel,
    models: ClientModels,
    styles: list[Style],
    seed: int,
    options: AcrossOptions,
    parent: QWidget | None = None,
):
    """Current prompt once per style. A style brings its own LoRAs, only the ones in
    the prompt are checked against each style's arch."""
    check = LoraArchCheck(_prompt_loras(model), resolve_arch(model.active_style, models))
    original = model.style

    def use(style: Style):
        return lambda: setattr(model, "style", style)

    entries = [_Entry(s.filename, s.name, resolve_arch(s, models), use(s)) for s in styles]
    return _run(
        model,
        entries,
        lambda job: job.params.style,
        seed,
        check,
        lambda: setattr(model, "style", original),
        options,
        parent,
    )
