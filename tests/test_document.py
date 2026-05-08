import asyncio

from krita import Document as MockDocument
from krita import Krita, Selection
from PyQt5.QtCore import QByteArray

from ai_diffusion.document import KritaDocument
from ai_diffusion.image import Bounds

from .conftest import qtapp


def _copy_id_annotation(src, dst) -> None:
    data = src.annotation("ai_diffusion/document_id")
    if data and data.size() > 0:
        dst.setAnnotation("ai_diffusion/document_id", "document unique identifier", data)


@qtapp
async def test_document_active():
    doc = KritaDocument.active()
    assert doc is None


def test_active_returns_same_instance():
    """Retrieving the active KritaDocument twice yields the same Python object."""
    Krita.instance().openDocument("")

    kd1 = KritaDocument.active()
    kd2 = KritaDocument.active()

    assert kd1 is not None
    assert kd1 is kd2


def test_active_not_in_document_list_yet():
    """A KritaDocument can be retrieved even before the underlying krita.Document
    appears in Krita's document list (which can happen in Krita during startup).
    Once the document is added to the list, the same instance is returned."""
    doc = MockDocument()
    # Set as active but do NOT add to documents list yet.
    Krita.instance().setActiveDocument(doc)

    kd1 = KritaDocument.active()
    assert kd1 is not None

    # Simulate Krita completing the registration of the pending document.
    # openDocument() detects the unregistered active doc and adds it to the list.
    Krita.instance().openDocument("")

    kd2 = KritaDocument.active()
    assert kd2 is kd1


def test_active_open_copy_gets_new_instance_and_id():
    """When 'File > Open a copy' is used in Krita, the copy shares the same
    document-id annotation as the original.  KritaDocument.active() must detect
    this collision, create a brand-new KritaDocument with a fresh unique id for
    the copy, and leave the original untouched."""
    original = Krita.instance().openDocument("")
    kd_original = KritaDocument.active()
    assert kd_original is not None
    original_id = kd_original.id

    # Open the copy and give it the same annotation id as the original.
    copy = Krita.instance().openDocument("")
    _copy_id_annotation(original, copy)

    kd_copy = KritaDocument.active()
    assert kd_copy is not None

    # Must be a distinct wrapper with a distinct id.
    assert kd_copy is not kd_original
    assert kd_copy.id != original_id

    # Retrieving the copy a second time returns the same new instance.
    kd_copy2 = KritaDocument.active()
    assert kd_copy2 is kd_copy


@qtapp
async def test_selection_bounds_changed():
    """Mutating the selection on the underlying krita.Document triggers the signal."""
    Krita.instance().openDocument("")
    kd = KritaDocument.active()
    assert kd is not None

    received: list[bool] = []
    kd.selection_bounds_changed.connect(lambda: received.append(True))

    sel = Selection()
    sel.setPixelData(QByteArray(bytes(100 * 80)), 10, 20, 100, 80)
    kd._doc._selection = sel  # type: ignore[attr-defined]

    # The poller fires every 20 ms; wait up to ~300 ms for the signal.
    for _ in range(15):
        await asyncio.sleep(0.02)
        if received:
            break

    assert received, "selection_bounds_changed was not emitted within timeout"
    assert kd.selection_bounds == Bounds(10, 20, 100, 80)


@qtapp
async def test_current_time_changed():
    """Mutating the current time on the underlying krita.Document triggers the signal."""
    Krita.instance().openDocument("")
    kd = KritaDocument.active()
    assert kd is not None

    received: list[bool] = []
    kd.current_time_changed.connect(lambda: received.append(True))

    kd._doc._current_time = 7  # type: ignore[attr-defined]

    # The poller fires every 20 ms; wait up to ~300 ms for the signal.
    for _ in range(15):
        await asyncio.sleep(0.02)
        if received:
            break

    assert received, "current_time_changed was not emitted within timeout"
    assert kd.current_time == 7


def test_active_after_close_and_reopen():
    """Closing a document and reopening it (new krita.Document object, but the
    same persisted annotation id) must produce a new KritaDocument wrapper rather
    than reusing the stale cached one."""
    doc1 = Krita.instance().openDocument("")
    kd1 = KritaDocument.active()
    assert kd1 is not None
    saved_id = kd1.id

    doc1.close()

    # Reopen: a fresh Document carries the saved id annotation (loaded from file).
    doc2 = Krita.instance().openDocument("")
    _copy_id_annotation(doc1, doc2)

    kd2 = KritaDocument.active()
    assert kd2 is not None

    # Must be a new wrapper even though the id annotation is the same.
    assert kd2 is not kd1
    # The id is preserved from the annotation (not regenerated).
    assert kd2.id == saved_id

    # Subsequent retrieval returns the same new wrapper.
    kd3 = KritaDocument.active()
    assert kd3 is kd2
