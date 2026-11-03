from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from ..localization import translate as _
from ..settings import settings

_PLACEHOLDER = " "  # combo entry that is a prompt, not a set


class GenerateSetBar(QWidget):
    """Save the current multi-selection under a name and restore it later.

    "Generate across" is only useful with a curated selection of styles or
    checkpoints, and picking the same dozen entries out of a long list again for
    the next comparison is the tedious part. A set is just the list of keys, kept
    in the settings file, so the same comparison can be re-run any time.
    """

    def __init__(
        self,
        setting_name: str,
        get_keys: Callable[[], list[str]],
        apply_keys: Callable[[list[str]], None],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._setting_name = setting_name
        self._get_keys = get_keys
        self._apply_keys = apply_keys

        self._combo = QComboBox(self)
        self._combo.setMinimumWidth(140)
        self._combo.setToolTip(_("Restore a saved selection"))
        self._combo.activated.connect(self._on_activated)

        self._save_btn = QPushButton(_("Save Set"), self)
        self._save_btn.setToolTip(_("Save the current selection under a name"))
        self._save_btn.clicked.connect(self._save)

        self._delete_btn = QPushButton(_("Delete Set"), self)
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._delete)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(_("Set:"), self))
        layout.addWidget(self._combo)
        layout.addWidget(self._save_btn)
        layout.addWidget(self._delete_btn)
        self.setLayout(layout)

        self._reload()

    # -- storage --

    def _sets(self) -> dict[str, list[str]]:
        value = getattr(settings, self._setting_name)
        return dict(value) if isinstance(value, dict) else {}

    def _store(self, sets: dict[str, list[str]], select: str | None = None):
        setattr(settings, self._setting_name, sets)
        settings.save()
        self._reload(select)

    def _reload(self, select: str | None = None):
        sets = self._sets()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem(_("(saved sets)"), _PLACEHOLDER)
        for name in sorted(sets, key=str.lower):
            self._combo.addItem(f"{name}  ({len(sets[name])})", name)
        idx = self._combo.findData(select) if select else -1
        self._combo.setCurrentIndex(max(idx, 0))
        self._combo.blockSignals(False)
        self._update_actions()

    def _current_name(self) -> str | None:
        data = self._combo.currentData()
        return None if data == _PLACEHOLDER else data

    def _update_actions(self):
        self._delete_btn.setEnabled(self._current_name() is not None)

    # -- actions --

    def _on_activated(self, _index: int):
        self._update_actions()
        name = self._current_name()
        if name is None:
            return
        self._apply_keys(self._sets().get(name, []))

    def _save(self):
        keys = self._get_keys()
        if not keys:
            QMessageBox.information(self, _("Save Set"), _("Select one or more entries first."))
            return
        name, ok = QInputDialog.getText(
            self, _("Save Set"), _("Name:"), text=self._current_name() or ""
        )
        name = name.strip()
        if not ok or not name:
            return
        sets = self._sets()
        if name in sets:
            confirm = QMessageBox.question(
                self,
                _("Save Set"),
                _("A set named '{name}' already exists. Replace it?").format(name=name),
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        sets[name] = keys
        self._store(sets, select=name)

    def _delete(self):
        name = self._current_name()
        if name is None:
            return
        confirm = QMessageBox.question(
            self,
            _("Delete Set"),
            _("Delete the set '{name}'?").format(name=name),
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        sets = self._sets()
        sets.pop(name, None)
        self._store(sets)
