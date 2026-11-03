from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..localization import translate as _
from ..model.model import DocumentModel
from . import theme


def sweep_values(start: int, end: int, step: int) -> list[int]:
    """Strength percentages from start towards end (rising or falling), end included if hit."""
    step = max(1, step)
    if end < start:
        step = -step
    values = list(range(start, end + (1 if step > 0 else -1), step))
    return values


class StrengthSweepDialog(QDialog):
    """Generate one image per strength value, all with the same seed."""

    _last = (40, 75, 5)

    def __init__(self, model: DocumentModel, parent: QWidget | None = None):
        super().__init__(parent)
        self._model = model
        self.setWindowTitle(_("Strength Sweep"))

        self._start = self._spin(1, 100, self._last[0], "%")
        self._end = self._spin(1, 100, self._last[1], "%")
        self._step = self._spin(1, 100, self._last[2], "%")
        self._start.setToolTip(_("First strength. Higher than 'To' sweeps downwards."))
        self._end.setToolTip(_("Last strength (generated if the steps land on it)"))

        self._fixed = QCheckBox(_("Fixed seed"), self)
        self._fixed.setChecked(True)
        self._fixed.setToolTip(
            _(
                "Same seed for every image, so strength is the only difference. "
                "Off: every image gets a random seed."
            )
        )
        self._seed = QSpinBox(self)
        self._seed.setRange(0, 2**31 - 1)
        self._seed.setValue(int(model.seed) % (2**31))
        self._count = self._spin(1, 50, 1, "")
        self._count.setToolTip(_("Images generated for each strength value (random seeds)"))

        form = QFormLayout()
        form.addRow(_("From:"), self._start)
        form.addRow(_("To:"), self._end)
        form.addRow(_("Step:"), self._step)
        form.addRow(self._fixed, self._seed)
        form.addRow(_("Images per step:"), self._count)

        self._summary = QLabel(self)
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(f"color: {theme.grey};")

        self._button = QPushButton(self)
        self._button.clicked.connect(self._generate)
        cancel = QPushButton(_("Close"), self)
        cancel.clicked.connect(self.close)  # type: ignore (slot returns bool)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(self._button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._summary)
        layout.addLayout(buttons)
        self.setMinimumWidth(320)

        for spin in (self._start, self._end, self._step, self._count):
            spin.valueChanged.connect(self._update)
        self._fixed.toggled.connect(self._update)
        self._update()

    def _spin(self, low: int, high: int, value: int, suffix: str):
        spin = QSpinBox(self)
        spin.setRange(low, high)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def values(self):
        return sweep_values(self._start.value(), self._end.value(), self._step.value())

    def per_step(self):
        return 1 if self._fixed.isChecked() else self._count.value()

    def _update(self):
        values = self.values()
        fixed = self._fixed.isChecked()
        self._seed.setEnabled(fixed)
        self._count.setEnabled(not fixed)
        per_step = self.per_step()
        total = len(values) * per_step
        shown = ", ".join(f"{v}%" for v in values)
        text = _("{n} images: {values}").format(n=total, values=shown)
        if per_step > 1:
            text = (
                _("{steps} steps x {per} images = ").format(steps=len(values), per=per_step) + text
            )
        self._summary.setText(text)
        self._button.setText(_("Generate {n} images").format(n=total))

    def _generate(self):
        values = self.values()
        model = self._model
        StrengthSweepDialog._last = (self._start.value(), self._end.value(), self._step.value())
        original_strength, original_batch = model.strength, model.batch_count

        if self._fixed.isChecked():

            def use(value: int):
                def apply():
                    model.strength = value / 100
                    model.batch_count = 1

                return apply

            def restore():
                model.strength = original_strength
                model.batch_count = original_batch

            model.generate_across([use(v) for v in values], self._seed.value(), restore)
        else:
            original_fixed = model.fixed_seed
            model.fixed_seed = False
            try:
                for value in values:
                    model.strength = value / 100
                    model.batch_count = self._count.value()
                    model.generate()
            finally:
                model.strength = original_strength
                model.batch_count = original_batch
                model.fixed_seed = original_fixed
        self.close()
