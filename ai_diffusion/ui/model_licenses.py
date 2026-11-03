from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..backend import model_licenses
from .theme import grey


class Krea2TermsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Krea 2 Edit V2 — Model Terms")
        self.setMinimumWidth(640)
        self.resize(720, 490)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        title = QLabel("<h2>Krea 2 Edit V2</h2><p>Identity Edit V2 &amp; General Edit V2</p>", self)
        layout.addWidget(title)
        summary = QLabel(
            "<b>Personal &amp; non-commercial use</b> — free under the model terms.<br><br>"
            "<b>Commercial use</b> — a separate written license from Lars Bouaraba is required, "
            "including for client work, paid outputs, advertising and business use. "
            "Additional Krea licensing may also apply.",
            self,
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        links = QLabel(
            f'<a href="{model_licenses.terms_url}">V2 model license</a> &nbsp;·&nbsp; '
            f'<a href="{model_licenses.community_url}">Krea 2 Community License</a> &nbsp;·&nbsp; '
            f'<a href="{model_licenses.notice_url}">NOTICE</a><br>'
            '<a href="mailto:l.bouaraba@gmail.com">Contact the model author about commercial use</a>',
            self,
        )
        links.setOpenExternalLinks(True)
        links.setWordWrap(True)
        layout.addWidget(links)
        detail = QLabel(
            "These terms apply to the model weights and their use, separately from this plugin's GPLv3 license. "
            "Accepting here does not grant a commercial license. Copies of all three documents will be installed "
            "in models/loras/krea2-edit-v2-licenses before the selected weights are downloaded.",
            self,
        )
        detail.setWordWrap(True)
        detail.setStyleSheet(f"color:{grey}")
        layout.addWidget(detail)
        self.reviewed = QCheckBox(
            "I have reviewed and accept the linked model terms.",
            self,
        )
        self.understood = QCheckBox(
            "I understand that accepting here does not grant commercial-use rights.",
            self,
        )
        layout.addWidget(self.reviewed)
        layout.addWidget(self.understood)
        layout.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel, Qt.Orientation.Horizontal, self)
        self.accept_button = QPushButton("Accept model terms", self)
        buttons.addButton(self.accept_button, QDialogButtonBox.AcceptRole)
        self.accept_button.setEnabled(False)
        self.reviewed.toggled.connect(self._update_accept)
        self.understood.toggled.connect(self._update_accept)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_accept(self):
        self.accept_button.setEnabled(self.reviewed.isChecked() and self.understood.isChecked())


def confirm_krea2_terms(parent: QWidget | None = None):
    if model_licenses.accepted():
        return True
    if not isinstance(QApplication.instance(), QApplication):
        return False
    if Krea2TermsDialog(parent).exec_() != QDialog.Accepted:
        return False
    model_licenses.acknowledge()
    return True
