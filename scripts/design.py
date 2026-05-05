import argparse
import sys
from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "tests" / "mock"))

import krita


class ControlPane(QWidget):
    """Side panel that lets you manipulate fake Krita state while inspecting the UI."""

    def __init__(self, dock: QWidget, parent=None):
        super().__init__(parent)
        self._dock = dock
        self._doc_counter = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        open_btn = QPushButton("Open document")
        open_btn.setToolTip("Create a new mock Document and trigger canvasChanged")
        open_btn.clicked.connect(self._open_document)
        layout.addWidget(open_btn)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_document(self):
        self._doc_counter += 1
        k = krita.Krita.instance()
        k.openDocument(f"mock-document-{self._doc_counter}.kra")
        canvas = krita.Canvas()
        canvas._view = krita.View()  # type: ignore[attr-defined]
        self._dock.canvasChanged(canvas)


def main():
    parser = argparse.ArgumentParser(description="Krita AI Diffusion – design preview")
    parser.add_argument("--exit", action="store_true")
    parser.add_argument("--no-connect", action="store_true")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    from ai_diffusion import eventloop
    from ai_diffusion.root import root
    from ai_diffusion.settings import settings
    from ai_diffusion.ui.diffusion import ImageDiffusionWidget
    from ai_diffusion.ui.settings import SettingsDialog

    eventloop.setup()
    settings.load()
    root.init()

    settings_dialog = SettingsDialog(root.server)
    dock = ImageDiffusionWidget()
    controls = ControlPane(dock)

    k = krita.Krita.instance()
    k.action("ai_diffusion_settings").triggered.connect(settings_dialog.show)
    if not args.no_connect:
        eventloop.run(root.autostart(settings_dialog.connection.update_ui))

    container = QWidget()
    container.setWindowTitle("AI Image Generation – design preview")
    layout = QHBoxLayout(container)
    layout.addWidget(controls)
    layout.addWidget(dock, stretch=1)
    container.resize(960, 720)
    container.show()

    if args.exit:
        QTimer.singleShot(0, app.quit)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
