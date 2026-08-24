"""Tests for the HistoryWidget top bar (search, star filter, sort order)."""

from pathlib import Path

import pytest
from krita import Krita
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QLineEdit, QToolButton

from ai_diffusion.backend.api import WorkflowKind
from ai_diffusion.document import KritaDocument
from ai_diffusion.image import Bounds, Extent, Image, ImageCollection
from ai_diffusion.model.connection import Connection
from ai_diffusion.model.custom_workflow import WorkflowCollection
from ai_diffusion.model.jobs import JobKind, JobParams
from ai_diffusion.model.model import DocumentModel
from ai_diffusion.model.root import root
from ai_diffusion.settings import settings
from ai_diffusion.ui.history import HistoryWidget

from ..conftest import qtapp


@pytest.fixture()
def workflows_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "workflows"
    folder.mkdir()
    return folder


def _add_job(model: DocumentModel, id: str, prompt: str, count=2, used=()):
    params = JobParams(Bounds(0, 0, 64, 64), prompt, metadata={"prompt": prompt})
    params.workflow_kind = WorkflowKind.generate
    job = model.jobs.add(JobKind.diffusion, params)
    job.id = id
    images = ImageCollection([
        Image.create(Extent(64, 64), QColor("#ff0000")) for _ in range(count)
    ])
    model.jobs.set_results(job, images)
    for i in used:
        job.in_use[i] = True
    model.jobs.notify_finished(job)
    return job


def _items(widget: HistoryWidget):
    return [
        (item.data(Qt.ItemDataRole.UserRole), item.data(Qt.ItemDataRole.UserRole + 1))
        for i in range(widget.count())
        if (item := widget.item(i)) is not None
    ]


@qtapp
async def test_history_top_bar(workflows_dir: Path):
    settings.load()
    root.init()

    krita_doc = Krita.instance().openDocument("test_history_top_bar")
    Krita.instance().setActiveDocument(krita_doc)
    doc = KritaDocument.active()
    assert doc is not None

    conn = Connection()
    model = DocumentModel(doc, conn, WorkflowCollection(conn, folder=workflows_dir))
    widget = HistoryWidget(None)
    widget.model_ = model

    try:
        assert isinstance(widget._search, QLineEdit)
        assert isinstance(widget._star_button, QToolButton)
        assert isinstance(widget._sort_button, QToolButton)
        assert not widget._star_button.isChecked()
        assert widget._sort_descending

        _add_job(model, "a", "a cat sitting", used=(0,))
        _add_job(model, "b", "a dog running", used=(1,))
        _add_job(model, "c", "a cat sleeping")

        # default order: oldest first, header per job, all images visible
        assert _items(widget) == [
            ("a", None),
            ("a", 0),
            ("a", 1),
            ("b", None),
            ("b", 0),
            ("b", 1),
            ("c", None),
            ("c", 0),
            ("c", 1),
        ]

        # search filters by prompt substring
        widget._search.setText("cat")
        assert _items(widget) == [
            ("a", None),
            ("a", 0),
            ("a", 1),
            ("c", None),
            ("c", 0),
            ("c", 1),
        ]

        # star filter hides images that have not been applied
        widget._search.setText("")
        widget._star_button.setChecked(True)
        assert _items(widget) == [
            ("a", None),
            ("a", 0),
            ("b", None),
            ("b", 1),
        ]

        # sort toggle reverses the order (newest first)
        widget._star_button.setChecked(False)
        widget._toggle_sort()
        assert not widget._sort_descending
        assert _items(widget) == [
            ("c", None),
            ("c", 0),
            ("c", 1),
            ("b", None),
            ("b", 0),
            ("b", 1),
            ("a", None),
            ("a", 0),
            ("a", 1),
        ]
    finally:
        widget.deleteLater()
        await conn.disconnect()
