"""design.py: A standalone script to preview the UI without running the full plugin.

Left side: a control pane to modify mock Krita state and trigger events.
Right side: the actual plugin UI, running as it would inside Krita, reacting to the mock state and events.
"""

import argparse
import random
import sys
from collections.abc import Generator
from pathlib import Path

from PyQt5.QtCore import QByteArray, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "tests" / "mock"))

import krita

_PREVIEW_SIZE = 512


# ---------------------------------------------------------------------------
# Sub-widgets used inside ControlPane
# ---------------------------------------------------------------------------


class DocInfoWidget(QGroupBox):
    """Displays filename, pixel dimensions, and DPI of the active document.

    The resize controls call ``Document.resizeImage`` to change the pixel
    dimensions (and regenerate the preview via ``pixelDataChanged``).
    """

    def __init__(self, parent=None):
        super().__init__("Document", parent)
        layout = QVBoxLayout(self)

        self._info_label = QLabel("No document open")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        resize_row = QHBoxLayout()
        resize_row.addWidget(QLabel("W:"))
        self._w_spin = QSpinBox()
        self._w_spin.setRange(64, 4096)
        self._w_spin.setValue(512)
        self._w_spin.setEnabled(False)
        resize_row.addWidget(self._w_spin)
        resize_row.addWidget(QLabel("H:"))
        self._h_spin = QSpinBox()
        self._h_spin.setRange(64, 4096)
        self._h_spin.setValue(512)
        self._h_spin.setEnabled(False)
        resize_row.addWidget(self._h_spin)
        self._resize_btn = QPushButton("Resize")
        self._resize_btn.setEnabled(False)
        self._resize_btn.clicked.connect(self._apply_resize)
        resize_row.addWidget(self._resize_btn)
        layout.addLayout(resize_row)

        self._doc: krita.Document | None = None

    def refresh(self, doc: "krita.Document | None") -> None:
        self._doc = doc
        enabled = doc is not None
        self._w_spin.setEnabled(enabled)
        self._h_spin.setEnabled(enabled)
        self._resize_btn.setEnabled(enabled)
        if doc is None:
            self._info_label.setText("No document open")
        else:
            name = Path(doc.fileName()).name or "(untitled)"
            self._info_label.setText(
                f"<b>{name}</b><br/>"
                f"{doc.width()} × {doc.height()} px  |  {doc.resolution():.0f} DPI"
            )
            self._w_spin.setValue(doc.width())
            self._h_spin.setValue(doc.height())

    def _apply_resize(self) -> None:
        if self._doc is not None:
            self._doc.resizeImage(0, 0, self._w_spin.value(), self._h_spin.value())
            # resizeImage triggers pixelDataChanged → ImagePreviewWidget auto-updates;
            # refresh the info label to show the new dimensions.
            self.refresh(self._doc)


class LayerListWidget(QGroupBox):
    """Lists every node in the active document; clicking a row toggles it as the active node."""

    def __init__(self, parent=None):
        super().__init__("Layers", parent)
        layout = QVBoxLayout(self)
        self._list = QListWidget()
        self._list.setMaximumHeight(180)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)
        self._doc: krita.Document | None = None
        self._updating = False

    def refresh(self, doc: "krita.Document | None") -> None:
        self._doc = doc
        self._updating = True
        self._list.clear()
        if doc is not None:
            active = doc.activeNode()
            active_id = active.uniqueId() if active else None
            for depth, node in _iter_nodes(doc.rootNode().childNodes()):
                indent = "  " * depth
                item = QListWidgetItem(f"{indent}{node.name()}  [{node.type()}]")
                item.setData(Qt.ItemDataRole.UserRole, node)
                self._list.addItem(item)
                if active_id is not None and node.uniqueId() == active_id:
                    item.setSelected(True)
                    self._list.setCurrentItem(item)
        self._updating = False

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        if self._doc is None or self._updating:
            return
        node: krita.Node = item.data(Qt.ItemDataRole.UserRole)
        active = self._doc.activeNode()
        if active is not None and active.uniqueId() == node.uniqueId():
            # Second click on the active layer → deactivate
            self._doc.setActiveNode(None)  # type: ignore[arg-type]
            self._list.clearSelection()
        else:
            self._doc.setActiveNode(node)


class ImagePreviewWidget(QGroupBox):
    """Renders the document's pixel data in a capped 512×512 canvas.

    Subscribes to ``Document.pixelDataChanged`` so the view stays current after
    the plugin (or the control pane) modifies pixel data.
    """

    def __init__(self, parent=None):
        super().__init__("Document image", parent)
        layout = QVBoxLayout(self)
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setFixedSize(_PREVIEW_SIZE, _PREVIEW_SIZE)
        self._canvas.setStyleSheet("background: #222; border: 1px solid #555;")
        layout.addWidget(self._canvas)
        self._doc: krita.Document | None = None

    def refresh(self, doc: "krita.Document | None") -> None:
        if self._doc is not None:
            try:
                self._doc.pixelDataChanged.disconnect(self._update_image)  # type: ignore[attr-defined]
            except RuntimeError:
                pass  # signal was already disconnected
        self._doc = doc
        if doc is not None:
            doc.pixelDataChanged.connect(self._update_image)  # type: ignore[attr-defined]
        self._update_image()

    def _update_image(self) -> None:
        if self._doc is None:
            self._canvas.clear()
            return
        w, h = self._doc.width(), self._doc.height()
        raw = bytes(self._doc.pixelData(0, 0, w, h))
        # Krita returns BGRA bytes; Format_ARGB32 uses the same memory layout on
        # little-endian systems (x86), so no channel swapping is needed.
        img = QImage(raw, w, h, w * 4, QImage.Format.Format_ARGB32)
        if w > _PREVIEW_SIZE or h > _PREVIEW_SIZE:
            img = img.scaled(
                _PREVIEW_SIZE,
                _PREVIEW_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._canvas.setPixmap(QPixmap.fromImage(img))


# ---------------------------------------------------------------------------
# Control pane
# ---------------------------------------------------------------------------


class ControlPane(QWidget):
    """Side panel that lets you manipulate fake Krita state while inspecting the UI."""

    def __init__(self, dock: QWidget, parent=None):
        super().__init__(parent)
        self._dock = dock
        self._doc_counter = 0

        # Scrollable so everything stays accessible regardless of window height.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- document controls ---
        doc_btns = QHBoxLayout()
        open_btn = QPushButton("Open document")
        open_btn.setToolTip("Create a new mock Document and trigger canvasChanged")
        open_btn.clicked.connect(self._open_document)
        doc_btns.addWidget(open_btn)
        close_btn = QPushButton("Close document")
        close_btn.setToolTip("Close the active document")
        close_btn.clicked.connect(self._close_document)
        doc_btns.addWidget(close_btn)
        layout.addLayout(doc_btns)

        # --- sub-widgets ---
        self._doc_info = DocInfoWidget()
        layout.addWidget(self._doc_info)

        self._layer_list = LayerListWidget()
        layout.addWidget(self._layer_list)

        self._image_preview = ImagePreviewWidget()
        layout.addWidget(self._image_preview)

        modify_btn = QPushButton("Modify pixels")
        modify_btn.setToolTip("Paint random shapes into the document to test image refresh")
        modify_btn.clicked.connect(self._modify_pixels)
        layout.addWidget(modify_btn)

        layout.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

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
        self._refresh()

    def _close_document(self):
        k = krita.Krita.instance()
        doc = k.activeDocument()
        if doc is not None:
            doc.close()
        canvas = krita.Canvas()  # canvas with no view → no active document
        self._dock.canvasChanged(canvas)
        self._refresh()

    def _modify_pixels(self):
        doc = krita.Krita.instance().activeDocument()
        if doc is None:
            return
        w, h = doc.width(), doc.height()
        # Start from a fresh light-gray background then paint coloured rectangles.
        buf = bytearray([200, 200, 200, 255] * (w * h))
        for _ in range(24):
            rx = random.randint(0, max(w - 60, 0))
            ry = random.randint(0, max(h - 60, 0))
            rw = random.randint(20, min(100, w))
            rh = random.randint(20, min(100, h))
            # BGRA byte order for Format_ARGB32
            b, g, r = random.randint(40, 255), random.randint(40, 255), random.randint(40, 255)
            pixel = bytes([b, g, r, 255])
            for row in range(ry, min(ry + rh, h)):
                col_end = min(rx + rw, w)
                start = (row * w + rx) * 4
                buf[start : start + (col_end - rx) * 4] = pixel * (col_end - rx)
        node = doc.rootNode().childNodes()[0]
        node.setPixelData(QByteArray(bytes(buf)), 0, 0, w, h)

    def _refresh(self):
        doc = krita.Krita.instance().activeDocument()
        self._doc_info.refresh(doc)
        self._layer_list.refresh(doc)
        self._image_preview.refresh(doc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_nodes(nodes: list, depth: int = 0) -> Generator:
    for node in nodes:
        yield depth, node
        yield from _iter_nodes(node.childNodes(), depth + 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    container.resize(1080, 800)
    container.show()

    if args.exit:
        QTimer.singleShot(0, app.quit)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
