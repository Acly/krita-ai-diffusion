from krita import Canvas, Krita, View
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTransform
from PyQt6.QtWidgets import QApplication, QMainWindow

from ai_diffusion.model.root import root
from ai_diffusion.settings import settings
from ai_diffusion.ui.live import LivePreviewDocker


class ClosingView(View):
    def __init__(self):
        super().__init__()
        self.closed = False

    def visible(self):
        return not self.closed

    def flakeToImageTransform(self):
        assert not self.closed, "Krita's transform getters cannot handle a deleted view"
        return QTransform()

    def flakeToCanvasTransform(self):
        assert not self.closed, "Krita's transform getters cannot handle a deleted view"
        return QTransform.fromScale(2, 2)


class PreviewCanvas(Canvas):
    def __init__(self, view: View):
        super().__init__()
        self._preview_view = view

    def view(self):
        return self._preview_view


def test_close_tracked_preview(qtapp):
    settings.load()
    root.init()
    doc = Krita.instance().openDocument("test")
    Krita.instance().setActiveDocument(doc)
    model = root.active_model
    model.live.track = True

    window = QMainWindow()
    docker = LivePreviewDocker()
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, docker)
    view = ClosingView()
    docker.canvasChanged(PreviewCanvas(view))
    window.show()
    QApplication.processEvents()
    area = docker.preview_area
    area._track_view()
    assert area._tracked == QTransform.fromScale(2, 2)
    assert area._poller.isActive()

    try:
        # Krita hides dockers while switching to the welcome page with updates disabled.
        window.setUpdatesEnabled(False)
        doc.close()
        docker.hide()
        assert not area._poller.isActive()
        assert area.model is model  # no reentrant document lookup during layout changes
        docker.canvasChanged(None)
        view.closed = True
        assert area._view is None
        assert area._tracked is None
        window.setUpdatesEnabled(True)
        QApplication.processEvents()
        assert area.model is root.active_model
        assert area.model is not model
        assert window.updatesEnabled()
    finally:
        window.close()
        window.deleteLater()
        QApplication.processEvents()


def test_track_closed_view(qtapp):
    settings.load()
    root.init()
    docker = LivePreviewDocker()
    area = docker.preview_area
    view = ClosingView()
    docker.canvasChanged(PreviewCanvas(view))
    try:
        assert area._read_view_transform() == QTransform.fromScale(2, 2)
        view.closed = True
        assert area._read_view_transform() is None
    finally:
        docker.deleteLater()
        QApplication.processEvents()
