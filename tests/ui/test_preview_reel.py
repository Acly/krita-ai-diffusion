"""Tests for the PreviewReel widget (tests/mock krita + offscreen Qt)."""

import asyncio
from pathlib import Path

import pytest
from krita import Krita
from PyQt6.QtCore import QAbstractAnimation, QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QKeyEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication

from ai_diffusion.backend.api import InpaintMode, WorkflowKind
from ai_diffusion.document import KritaDocument
from ai_diffusion.image import Bounds, Extent, Image, ImageCollection
from ai_diffusion.model.connection import Connection, ConnectionState
from ai_diffusion.model.custom_workflow import WorkflowCollection
from ai_diffusion.model.jobs import Job, JobKind, JobParams, JobQueue, JobState
from ai_diffusion.model.model import DocumentModel
from ai_diffusion.model.root import root
from ai_diffusion.settings import settings
from ai_diffusion.style import Style
from ai_diffusion.ui.history import PreviewReel, PreviewReelItem

from ..conftest import qtapp
from ..mock.client import MockClient


def _make_style(checkpoint: str = "test_sd15.safetensors") -> Style:
    style = Style(Path("test.json"))
    style.checkpoints = [checkpoint]
    return style


async def _wait_for_state(conn: Connection, *exclude: ConnectionState, timeout: int = 100):
    for _ in range(timeout):
        await asyncio.sleep(0)
        if conn.state not in exclude:
            return
    raise TimeoutError(f"Connection stuck at {conn.state!r} after {timeout} iterations")


def _image(r=200, g=40, b=40):
    return Image.create(Extent(256, 256), QColor(r, g, b))


def _make_job(
    model: DocumentModel, id: str, workflow_kind=WorkflowKind.generate, kind=JobKind.diffusion
):
    params = JobParams(Bounds(0, 0, 512, 512), f"name {id}")
    params.workflow_kind = workflow_kind
    if workflow_kind is WorkflowKind.inpaint:
        params.inpaint_mode = InpaintMode.fill
    job = Job(id, kind, params)
    model.jobs.add_job(job)
    return job


def _finish(model: DocumentModel, job: Job, count=1):
    images = ImageCollection([_image() for _ in range(count)])
    model.jobs.set_results(job, images)
    model.jobs.notify_finished(job)


def _wheel(reel: PreviewReel, delta: int, pos: QPoint | None = None):
    position = QPointF(pos if pos is not None else reel.rect().center())
    event = QWheelEvent(
        position,
        position,
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(reel, event)


def _key(reel: PreviewReel, key: Qt.Key, modifiers=Qt.KeyboardModifier.NoModifier):
    event = QKeyEvent(QEvent.Type.KeyPress, key, modifiers)
    QApplication.sendEvent(reel, event)


async def _settle(reel: PreviewReel):
    for _ in range(50):
        await asyncio.sleep(0.02)
        if reel._scroll_anim.state() is not QAbstractAnimation.State.Running:
            return


@pytest.fixture()
def workflows_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "workflows"
    folder.mkdir()
    return folder


@qtapp
async def test_preview_reel(workflows_dir: Path):
    settings.load()
    root.init()

    krita_doc = Krita.instance().openDocument("test_preview_reel")
    Krita.instance().setActiveDocument(krita_doc)
    doc = KritaDocument.active()
    assert doc is not None

    client = MockClient()
    conn = Connection()
    conn.connect(client)
    await _wait_for_state(conn, ConnectionState.connecting, ConnectionState.disconnected)
    assert conn.state is ConnectionState.connected

    wf_coll = WorkflowCollection(conn, folder=workflows_dir)
    model = DocumentModel(doc, conn, wf_coll)
    model.style = _make_style()
    conn.message_received.connect(model.handle_message)
    root._connection = conn

    reel = PreviewReel(None)
    reel.resize(600, reel.height())
    reel.model_ = model

    try:
        assert reel._items == []
        assert reel.sizeHint().height() == reel.height()  # fixed height even when empty

        # finished jobs are shown newest first, one item per result image
        job1 = _make_job(model, "job-1")
        _finish(model, job1, count=2)
        assert [it.job.id for it in reel._items] == ["job-1", "job-1"]

        job2 = _make_job(model, "job-2")
        _finish(model, job2)
        assert [it.job.id for it in reel._items] == ["job-2", "job-1", "job-1"]
        assert all(it.kind is PreviewReelItem.Kind.result for it in reel._items)

        # queued and executing jobs show as placeholders on the left
        job3 = _make_job(model, "job-3", WorkflowKind.inpaint)
        model.jobs.notify_started(job3)
        _make_job(model, "job-4")
        _make_job(model, "job-5", WorkflowKind.refine)
        kinds = [(it.job.id, it.kind) for it in reel._items]
        assert kinds == [
            ("job-5", PreviewReelItem.Kind.queued),
            ("job-4", PreviewReelItem.Kind.queued),
            ("job-3", PreviewReelItem.Kind.executing),
            ("job-2", PreviewReelItem.Kind.result),
            ("job-1", PreviewReelItem.Kind.result),
            ("job-1", PreviewReelItem.Kind.result),
        ]
        # refine job captured the document as input, generate job has none (white)
        assert reel._items[0].input is not None
        assert reel._items[1].input is None

        # finishing a job converts its placeholder into results in place
        _finish(model, job3)
        assert [(it.job.id, it.kind) for it in reel._items] == [
            ("job-5", PreviewReelItem.Kind.queued),
            ("job-4", PreviewReelItem.Kind.queued),
            ("job-3", PreviewReelItem.Kind.result),
            ("job-2", PreviewReelItem.Kind.result),
            ("job-1", PreviewReelItem.Kind.result),
            ("job-1", PreviewReelItem.Kind.result),
        ]

        # marking a result as used keeps the item (star is painted into the thumbnail)
        model.jobs.notify_used("job-2", 0)
        assert [it.job.id for it in reel._items].count("job-2") == 1

        # hover a result item: becomes active and sets the canvas preview selection
        index = next(i for i, it in enumerate(reel._items) if it.job.id == "job-2")
        reel._update_hover(reel._item_rect(index).center())
        assert reel._active is reel._items[index]
        assert model.jobs.selection == [JobQueue.Item("job-2", 0)]

        # the info label is a child overlay that never intercepts mouse input,
        # so hovering over it does not clear the active item
        assert reel._info is not None and not reel._info.isWindow()
        assert reel._info.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        assert reel.childAt(reel._info.geometry().center()) is None

        # hovering the gap between items keeps the previous item active (no flicker)
        gap = QPoint(reel._item_rect(index).right() + 2, reel.height() // 2)
        reel._update_hover(gap)
        assert reel._item_at(gap) is None  # really in a gap
        assert reel._active is reel._items[index]
        assert model.jobs.selection == [JobQueue.Item("job-2", 0)]

        # placeholders cannot be active; leaving the widget hides the preview
        reel._update_hover(reel._item_rect(0).center())
        assert reel._active is None
        assert model.jobs.selection == []
        reel._update_hover(reel._item_rect(index).center())
        reel.leaveEvent(None)
        assert reel._active is None
        assert model.jobs.selection == []

        # clicking a result applies it (preview layer is replaced by the result layer)
        layer_names = [layer.name for layer in model.layers.images]
        index = next(i for i, it in enumerate(reel._items) if it.job.id == "job-2")
        reel._click(reel._items[index], Qt.KeyboardModifier.NoModifier)
        new_layers = [l for l in model.layers.images if l.name not in layer_names]
        assert len(new_layers) == 1 and new_layers[0].name.startswith("[Generated]")
        assert ensure_job(model, "job-2").result_was_used(0)

        # clicking a queued placeholder cancels only that job
        reel._click(reel._items[0], Qt.KeyboardModifier.NoModifier)
        assert model.jobs.find("job-5") is None
        assert model.jobs.find("job-4") is not None

        # shift+click a queued placeholder cancels all queued jobs
        _make_job(model, "job-6")
        assert reel._items[0].job.id == "job-6"
        reel._click(reel._items[0], Qt.KeyboardModifier.ShiftModifier)
        assert model.jobs.count(JobState.queued) == 0

        # discarding a result image removes the item, the job once it is empty
        model.jobs.discard("job-1", 0)
        assert [it.job.id for it in reel._items].count("job-1") == 1
        model.jobs.discard("job-1", 0)
        assert [it.job.id for it in reel._items].count("job-1") == 0

        # wheel scroll moves exactly one item per step (add enough jobs to overflow)
        i = 0
        while reel._max_offset() < 3 * reel._stride:
            _finish(model, _make_job(model, f"extra-{i}"))
            i += 1
        assert reel._max_offset() > 0
        assert reel._offset == 0
        _wheel(reel, -120)
        for _ in range(50):
            await asyncio.sleep(0.02)
            if reel._scroll_anim.state() is not QAbstractAnimation.State.Running:
                break
        assert abs(reel._offset - reel._stride) < 1

        # new items appear on the left; when scrolled, offset compensates so items stay put
        before = reel._offset
        _make_job(model, "job-7")
        assert reel._items[0].job.id == "job-7"
        assert abs(reel._offset - (before + reel._stride)) < 1

        # ... unless the view is already scrolled all the way left
        async def scroll_to(offset: float):
            for _ in range(40):
                if abs(reel._offset - offset) < 1:
                    return
                _wheel(reel, 120 if reel._offset > offset else -120)
                await asyncio.sleep(0.15)
            raise AssertionError(f"scroll_to({offset}) stuck at {reel._offset}")

        await scroll_to(0)
        assert reel._offset == 0
        _make_job(model, "job-8")
        assert reel._items[0].job.id == "job-8"
        assert reel._offset == 0

        # wheel scrolling immediately activates the item the cursor will hover at the
        # target position, without waiting for the scroll animation to finish
        pos = reel._item_rect(3).center()
        reel._update_hover(pos)
        assert reel._active is reel._items[3]
        _wheel(reel, -120, pos)
        assert reel._scroll_anim.state() is QAbstractAnimation.State.Running
        assert reel._active is reel._items[4]
        for _ in range(50):
            await asyncio.sleep(0.02)
            if reel._scroll_anim.state() is not QAbstractAnimation.State.Running:
                break
        assert abs(reel._offset - reel._stride) < 1
        assert reel._active is reel._items[4]  # active item did not change during animation

        # scrolling so that a placeholder ends up under the cursor deactivates the preview
        pos = reel._item_rect(2).center()
        reel._update_hover(pos)
        assert reel._active is reel._items[2]
        _wheel(reel, 120, pos)
        assert reel._active is None
        assert model.jobs.selection == []
        for _ in range(50):
            await asyncio.sleep(0.02)
            if reel._scroll_anim.state() is not QAbstractAnimation.State.Running:
                break
        assert abs(reel._offset) < 1

        # cursor resting in the gap between items: scrolling keeps the current active item
        pos = reel._item_rect(2).center()
        reel._update_hover(pos)
        assert reel._active is reel._items[2]
        gap = QPoint(reel._item_rect(2).right() + 2, reel.height() // 2)
        assert reel._item_at(gap) is None and reel._item_at(gap, reel._stride) is None
        _wheel(reel, -120, gap)
        assert reel._active is reel._items[2]
    finally:
        reel.deleteLater()
        await asyncio.sleep(0.05)  # let the message handler task start before cancelling it
        await conn.disconnect()


@qtapp
async def test_preview_reel_navigation(workflows_dir: Path):
    settings.load()
    root.init()

    krita_doc = Krita.instance().openDocument("test_preview_reel_navigation")
    Krita.instance().setActiveDocument(krita_doc)
    doc = KritaDocument.active()
    assert doc is not None

    client = MockClient()
    conn = Connection()
    conn.connect(client)
    await _wait_for_state(conn, ConnectionState.connecting, ConnectionState.disconnected)
    assert conn.state is ConnectionState.connected

    wf_coll = WorkflowCollection(conn, folder=workflows_dir)
    model = DocumentModel(doc, conn, wf_coll)
    model.style = _make_style()
    conn.message_received.connect(model.handle_message)
    root._connection = conn

    reel = PreviewReel(None)
    reel.resize(600, reel.height())
    reel.model_ = model

    try:
        # no items: both buttons hidden
        assert reel._nav_left.isHidden()
        assert reel._nav_right.isHidden()

        # results first, then executing + queued placeholders (newest left)
        for i in range(12):
            _finish(model, _make_job(model, f"result-{i}"))
        exec_job = _make_job(model, "executing")
        model.jobs.notify_started(exec_job)
        for i in range(3):
            _make_job(model, f"queued-{i}")

        assert [it.kind for it in reel._items][:4] == [
            PreviewReelItem.Kind.queued,
            PreviewReelItem.Kind.queued,
            PreviewReelItem.Kind.queued,
            PreviewReelItem.Kind.executing,
        ]
        assert reel._offset == 0
        assert reel._nav_left.isHidden()
        assert not reel._nav_right.isHidden()
        assert reel._queue_count._count == 3
        assert not reel._queue_count.isHidden()
        assert reel._queue_count.pos() == QPoint(reel._nav_button_width + 3, 1)
        assert reel._queue_count.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # buttons overlay items: fixed width, item height, item geometry is independent
        assert reel._nav_left.width() == 24
        assert reel._nav_left.height() == reel._thumb + 1
        assert reel._nav_right.width() == 24
        assert reel._nav_right.height() == reel._thumb + 1

        # right click with queued items in view -> jump to first non-queued item
        reel._nav_right_clicked()
        await _settle(reel)
        assert abs(reel._offset - 3 * reel._stride) < 1
        assert not reel._nav_left.isHidden()
        assert not reel._nav_right.isHidden()
        assert not any(
            it.kind is PreviewReelItem.Kind.queued and reel._is_item_visible(i)
            for i, it in enumerate(reel._items)
        )

        # right click with no queued items in view -> page scroll
        before = reel._offset
        reel._nav_right_clicked()
        await _settle(reel)
        step = max(1, reel._items_per_view - 1) * reel._stride
        assert abs(reel._offset - min(before + step, reel._max_offset())) < 1

        # left click with non-queued items hidden to the left -> jump to first non-queued
        reel._nav_left_clicked()
        await _settle(reel)
        assert abs(reel._offset - 3 * reel._stride) < 1

        # left click with only queued items hidden to the left -> jump to first item
        reel._set_offset(2 * reel._stride)
        reel._nav_left_clicked()
        await _settle(reel)
        assert abs(reel._offset) < 1

        # queue count overlay is independent of the nav button and follows queued items
        assert reel._queue_count._count == 3
        for item in list(reel._items):
            if item.kind is PreviewReelItem.Kind.queued:
                reel._items.remove(item)
        reel._update_nav_buttons()
        assert reel._queue_count._count == 0
        assert reel._queue_count.isHidden()
    finally:
        reel.deleteLater()
        await asyncio.sleep(0.05)
        await conn.disconnect()


@qtapp
async def test_preview_reel_keyboard_scroll(workflows_dir: Path):
    settings.load()
    root.init()

    krita_doc = Krita.instance().openDocument("test_preview_reel_keyboard_scroll")
    Krita.instance().setActiveDocument(krita_doc)
    doc = KritaDocument.active()
    assert doc is not None

    client = MockClient()
    conn = Connection()
    conn.connect(client)
    await _wait_for_state(conn, ConnectionState.connecting, ConnectionState.disconnected)
    assert conn.state is ConnectionState.connected

    wf_coll = WorkflowCollection(conn, folder=workflows_dir)
    model = DocumentModel(doc, conn, wf_coll)
    model.style = _make_style()
    conn.message_received.connect(model.handle_message)
    root._connection = conn

    reel = PreviewReel(None)
    reel.resize(600, reel.height())
    reel.model_ = model

    try:
        for i in range(8):
            _finish(model, _make_job(model, f"result-{i}"))
        exec_job = _make_job(model, "executing")
        model.jobs.notify_started(exec_job)
        _make_job(model, "queued-0")
        _make_job(model, "queued-1")

        # placeholders on the left, results on the right
        assert reel._items[0].kind is PreviewReelItem.Kind.queued
        assert reel._items[3].kind is PreviewReelItem.Kind.result

        # arrow keys scroll by exactly one item per press
        assert reel._offset == 0
        _key(reel, Qt.Key.Key_Right)
        await _settle(reel)
        assert abs(reel._offset - reel._stride) < 1
        _key(reel, Qt.Key.Key_Left)
        await _settle(reel)
        assert abs(reel._offset) < 1

        # right arrow at the right edge activates the next result
        reel._set_offset(reel._max_offset())
        second_last = reel._items[-2]
        last = reel._items[-1]
        reel._set_active(second_last)
        _key(reel, Qt.Key.Key_Right)
        assert reel._active is last

        # mouse wheel does the same at the right edge
        reel._set_active(second_last)
        _wheel(reel, -120)
        assert reel._active is last

        # no active item: edge scrolling does nothing
        reel._set_active(None)
        _key(reel, Qt.Key.Key_Right)
        assert reel._active is None

        # active item already rightmost: edge scrolling does nothing
        reel._set_active(last)
        _key(reel, Qt.Key.Key_Right)
        assert reel._active is last

        # left arrow at the left edge activates the previous result
        reel._set_offset(0)
        reel._set_active(reel._items[4])
        _key(reel, Qt.Key.Key_Left)
        assert reel._active is reel._items[3]

        # mouse wheel does the same at the left edge
        reel._set_active(reel._items[4])
        _wheel(reel, 120)
        assert reel._active is reel._items[3]
    finally:
        reel.deleteLater()
        await asyncio.sleep(0.05)
        await conn.disconnect()


@qtapp
async def test_preview_reel_keyboard_actions(workflows_dir: Path):
    settings.load()
    root.init()

    krita_doc = Krita.instance().openDocument("test_preview_reel_keyboard_actions")
    Krita.instance().setActiveDocument(krita_doc)
    doc = KritaDocument.active()
    assert doc is not None

    client = MockClient()
    conn = Connection()
    conn.connect(client)
    await _wait_for_state(conn, ConnectionState.connecting, ConnectionState.disconnected)
    assert conn.state is ConnectionState.connected

    wf_coll = WorkflowCollection(conn, folder=workflows_dir)
    model = DocumentModel(doc, conn, wf_coll)
    model.style = _make_style()
    conn.message_received.connect(model.handle_message)
    root._connection = conn

    reel = PreviewReel(None)
    reel.resize(600, reel.height())
    reel.model_ = model

    try:
        for i in range(3):
            _finish(model, _make_job(model, f"result-{i}"))
        _make_job(model, "queued-0")
        _make_job(model, "queued-1")
        exec_job = _make_job(model, "executing")
        model.jobs.notify_started(exec_job)

        # items: [executing, queued-1, queued-0, result-2, result-1, result-0]
        assert [it.kind for it in reel._items] == [
            PreviewReelItem.Kind.executing,
            PreviewReelItem.Kind.queued,
            PreviewReelItem.Kind.queued,
            PreviewReelItem.Kind.result,
            PreviewReelItem.Kind.result,
            PreviewReelItem.Kind.result,
        ]

        # Space toggles the active item and restores it on the next press
        item_a = reel._items[3]
        item_b = reel._items[4]
        reel._set_active(item_a)
        _key(reel, Qt.Key.Key_Space)
        assert reel._active is None
        assert model.jobs.selection == []
        _key(reel, Qt.Key.Key_Space)
        assert reel._active is item_a

        # changing the active item resets the Space toggle state
        reel._set_active(item_b)
        _key(reel, Qt.Key.Key_Space)
        assert reel._active is None
        _key(reel, Qt.Key.Key_Space)
        assert reel._active is item_b

        # Enter applies the active result
        layer_names = [layer.name for layer in model.layers.images]
        reel._update_hover(reel._item_rect(3).center())
        assert reel._active is reel._items[3]
        _key(reel, Qt.Key.Key_Return)
        new_layers = [l for l in model.layers.images if l.name not in layer_names]
        assert len(new_layers) == 1 and new_layers[0].name.startswith("[Generated]")

        # Enter cancels the hovered queued placeholder
        reel._update_hover(reel._item_rect(1).center())
        assert reel._hover_item is reel._items[1]
        assert reel._active is None
        _key(reel, Qt.Key.Key_Return)
        assert model.jobs.find("queued-1") is None
        assert model.jobs.find("queued-0") is not None

        # Shift+Enter cancels all queued jobs
        reel._update_hover(reel._item_rect(1).center())
        _key(reel, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        assert model.jobs.count(JobState.queued) == 0
    finally:
        reel.deleteLater()
        await asyncio.sleep(0.05)
        await conn.disconnect()


@qtapp
async def test_preview_reel_hover_cancel_icon(workflows_dir: Path):
    settings.load()
    root.init()

    krita_doc = Krita.instance().openDocument("test_preview_reel_hover_cancel_icon")
    Krita.instance().setActiveDocument(krita_doc)
    doc = KritaDocument.active()
    assert doc is not None

    client = MockClient()
    conn = Connection()
    conn.connect(client)
    await _wait_for_state(conn, ConnectionState.connecting, ConnectionState.disconnected)
    assert conn.state is ConnectionState.connected

    wf_coll = WorkflowCollection(conn, folder=workflows_dir)
    model = DocumentModel(doc, conn, wf_coll)
    model.style = _make_style()
    conn.message_received.connect(model.handle_message)
    root._connection = conn

    reel = PreviewReel(None)
    reel.resize(600, reel.height())
    reel.model_ = model

    try:
        _make_job(model, "queued-0")
        assert reel._items[0].kind is PreviewReelItem.Kind.queued
        rect = reel._item_rect(0)

        cancel = reel._cancel_color

        def _center_color():
            pixmap = reel.grab(rect)
            return pixmap.toImage().pixelColor(rect.width() // 2, rect.height() // 2)

        def _is_cancel(color):
            return (
                color.red() == cancel.red()
                and color.green() == cancel.green()
                and color.blue() == cancel.blue()
            )

        # not hovered: the regular queue icon is painted (grey, not the cancel color)
        color = _center_color()
        assert not _is_cancel(color)

        # hovered: the icon is exchanged with a red X
        reel._update_hover(rect.center())
        assert reel._hover_item is reel._items[0]
        color = _center_color()
        assert _is_cancel(color)

        # executing placeholders show the red X as well
        exec_job = _make_job(model, "executing")
        model.jobs.notify_started(exec_job)
        assert reel._items[0].kind is PreviewReelItem.Kind.executing
        rect = reel._item_rect(0)
        reel._update_hover(rect.center())
        assert reel._hover_item is reel._items[0]
        color = _center_color()
        assert _is_cancel(color)
    finally:
        reel.deleteLater()
        await asyncio.sleep(0.05)
        await conn.disconnect()


def ensure_job(model: DocumentModel, id: str) -> Job:
    job = model.jobs.find(id)
    assert job is not None
    return job
