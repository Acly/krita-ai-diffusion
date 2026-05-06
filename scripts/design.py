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
    QCheckBox,
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
    """Lists every node in the active document; clicking a row toggles it as the active node.

    Subscribes to ``Document.nodeCreated`` to append newly added layers without
    a full rebuild, and to ``Document.activeNodeChanged`` to keep the selection
    highlight in sync.
    """

    def __init__(self, parent=None):
        super().__init__("Layers", parent)
        layout = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add layer")
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._add_layer)
        btn_row.addWidget(self._add_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._list = QListWidget()
        self._list.setMaximumHeight(180)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)
        self._doc: krita.Document | None = None
        self._layer_counter = 0
        self._updating = False

    def refresh(self, doc: "krita.Document | None") -> None:
        if self._doc is not None:
            try:
                self._doc.nodesChanged.disconnect(self._on_nodes_changed)  # type: ignore[attr-defined]
            except RuntimeError:
                pass
            try:
                self._doc.activeNodeChanged.disconnect(self._on_active_node_changed)  # type: ignore[attr-defined]
            except RuntimeError:
                pass
        self._doc = doc
        self._add_btn.setEnabled(doc is not None)
        if doc is not None:
            doc.nodesChanged.connect(self._on_nodes_changed)  # type: ignore[attr-defined]
            doc.activeNodeChanged.connect(self._on_active_node_changed)  # type: ignore[attr-defined]
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        """Repopulate the list from the current document tree."""
        self._updating = True
        self._list.clear()
        if self._doc is not None:
            active = self._doc.activeNode()
            active_id = active.uniqueId() if active else None
            for depth, node in _iter_nodes(self._doc.rootNode().childNodes()):
                indent = "  " * depth
                item = QListWidgetItem(f"{indent}{node.name()}  [{node.type()}]")
                item.setData(Qt.ItemDataRole.UserRole, node)
                self._list.addItem(item)
                if active_id is not None and node.uniqueId() == active_id:
                    item.setSelected(True)
                    self._list.setCurrentItem(item)
        self._updating = False

    def _on_nodes_changed(self) -> None:
        self._rebuild_list()

    def _on_active_node_changed(self, node: "krita.Node | None") -> None:
        """Sync the list selection with the document's active node."""
        self._updating = True
        self._list.clearSelection()
        if node is not None:
            for i in range(self._list.count()):
                item = self._list.item(i)
                if item and item.data(Qt.ItemDataRole.UserRole).uniqueId() == node.uniqueId():
                    item.setSelected(True)
                    self._list.setCurrentItem(item)
                    break
        self._updating = False

    def _add_layer(self) -> None:
        if self._doc is None:
            return
        self._layer_counter += 1
        node = self._doc.createNode(f"Layer {self._layer_counter}", "paintlayer")
        self._doc.rootNode().addChildNode(node, None)
        self._doc.setActiveNode(node)

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
    """Renders pixel data in a capped 512×512 canvas.

    Shows the active layer's pixel data.  Falls back to the document's
    composite pixel data when no layer is active.  Subscribes to
    ``Document.pixelDataChanged`` (document-level refresh), the active
    node's ``pixelDataChanged``, and ``Document.activeNodeChanged`` so the
    view stays current whenever pixels or the active layer change.
    """

    def __init__(self, parent=None):
        super().__init__("Image preview", parent)
        layout = QVBoxLayout(self)
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setFixedSize(_PREVIEW_SIZE, _PREVIEW_SIZE)
        self._canvas.setStyleSheet("background: #222; border: 1px solid #555;")
        layout.addWidget(self._canvas)
        self._doc: krita.Document | None = None
        self._active_node: krita.Node | None = None

    def refresh(self, doc: "krita.Document | None") -> None:
        self._disconnect_all()
        self._doc = doc
        self._active_node = doc.activeNode() if doc is not None else None
        if doc is not None:
            doc.pixelDataChanged.connect(self._update_image)  # type: ignore[attr-defined]
            doc.activeNodeChanged.connect(self._on_active_node_changed)  # type: ignore[attr-defined]
        if self._active_node is not None:
            self._active_node.pixelDataChanged.connect(self._update_image)  # type: ignore[attr-defined]
        self._update_image()

    # ------------------------------------------------------------------

    def _disconnect_all(self) -> None:
        if self._doc is not None:
            try:
                self._doc.pixelDataChanged.disconnect(self._update_image)  # type: ignore[attr-defined]
            except RuntimeError:
                pass
            try:
                self._doc.activeNodeChanged.disconnect(self._on_active_node_changed)  # type: ignore[attr-defined]
            except RuntimeError:
                pass
        if self._active_node is not None:
            try:
                self._active_node.pixelDataChanged.disconnect(self._update_image)  # type: ignore[attr-defined]
            except RuntimeError:
                pass

    def _on_active_node_changed(self, node: "krita.Node | None") -> None:
        if self._active_node is not None:
            try:
                self._active_node.pixelDataChanged.disconnect(self._update_image)  # type: ignore[attr-defined]
            except RuntimeError:
                pass
        self._active_node = node
        if node is not None:
            node.pixelDataChanged.connect(self._update_image)  # type: ignore[attr-defined]
        self._update_image()

    def _update_image(self) -> None:
        if self._doc is None:
            self._canvas.clear()
            self.setTitle("Image preview")
            return
        w, h = self._doc.width(), self._doc.height()
        if self._active_node is not None:
            raw = bytes(self._active_node.pixelData(0, 0, w, h))
            self.setTitle(f"Layer: {self._active_node.name()}")
        else:
            raw = bytes(self._doc.pixelData(0, 0, w, h))
            self.setTitle("Document")
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


class SelectionWidget(QGroupBox):
    """Checkbox that creates or clears a centered half-size selection on the active document."""

    def __init__(self, parent=None):
        super().__init__("Selection", parent)
        layout = QVBoxLayout(self)
        self._checkbox = QCheckBox("Active (half-size, centered)")
        self._checkbox.setEnabled(False)
        self._checkbox.stateChanged.connect(self._on_toggled)
        layout.addWidget(self._checkbox)
        self._doc: krita.Document | None = None
        self._updating = False

    def refresh(self, doc: "krita.Document | None") -> None:
        self._doc = doc
        self._updating = True
        self._checkbox.setEnabled(doc is not None)
        self._checkbox.setChecked(doc is not None and doc.selection() is not None)
        self._updating = False

    def _on_toggled(self, state: int) -> None:
        if self._updating or self._doc is None:
            return
        if state == Qt.CheckState.Checked:
            w, h = self._doc.width() // 2, self._doc.height() // 2
            x, y = self._doc.width() // 4, self._doc.height() // 4
            sel = krita.Selection()
            sel.setPixelData(QByteArray(bytes([255] * (w * h))), x, y, w, h)
            self._doc.setSelection(sel)
        else:
            self._doc.setSelection(None)


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

        self._selection = SelectionWidget()
        layout.addWidget(self._selection)

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
        self._selection.refresh(doc)


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
    container.resize(1600, 1400)
    container.show()

    if args.exit:
        QTimer.singleShot(0, app.quit)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
