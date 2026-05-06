"""Tests for polling behavior of LayerManager in ai_diffusion/layer.py."""

from __future__ import annotations

from krita import Document

from ai_diffusion.layer import LayerManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_manager(doc: Document | None = None) -> tuple[LayerManager, Document]:
    """Create a LayerManager backed by a mock document, with the timer disabled."""
    if doc is None:
        doc = Document()
    mgr = LayerManager(doc)
    mgr._timer.stop()
    return mgr, doc


def track(signal) -> list:
    """Return a list that accumulates all argument tuples emitted by *signal*."""
    calls: list = []
    signal.connect(lambda *a: calls.append(a))
    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_layer_added():
    """Adding a node to the document is picked up on the next update."""
    mgr, doc = make_manager()
    changed = track(mgr.changed)

    new_node = doc.createNode("NewLayer", "paintlayer")
    doc.rootNode().addChildNode(new_node, None)
    mgr.update()

    assert len(changed) == 1
    assert mgr.find(new_node.uniqueId()) is not None


def test_layer_removed():
    """Removing a node from the document emits removed and changed."""
    mgr, doc = make_manager()
    bg_node = doc.rootNode().childNodes()[0]
    bg_layer = mgr.find(bg_node.uniqueId())

    removed = track(mgr.removed)
    changed = track(mgr.changed)

    bg_node.remove()
    mgr.update()

    assert len(removed) == 1
    assert removed[0][0] is bg_layer
    assert len(changed) == 1
    assert mgr.find(bg_node.uniqueId()) is None


def test_active_layer_changes():
    """Changing the active node emits active_changed but not changed."""
    mgr, doc = make_manager()

    second_node = doc.createNode("Layer 2", "paintlayer")
    doc.rootNode().addChildNode(second_node, None)
    mgr.update()  # absorb the structural addition first

    active_changed = track(mgr.active_changed)
    changed = track(mgr.changed)

    doc.setActiveNode(second_node)
    mgr.update()

    assert len(active_changed) == 1
    assert len(changed) == 0  # only active changed, no structural change


def test_layer_removal_and_active_change_together():
    """Removing the active layer and changing active node in a single update
    emits removed, active_changed and changed all at once."""
    mgr, doc = make_manager()
    root = doc.rootNode()

    second_node = doc.createNode("Layer 2", "paintlayer")
    root.addChildNode(second_node, None)
    mgr.update()
    doc.setActiveNode(second_node)
    mgr.update()  # settle: second_node is now the known active layer

    removed = track(mgr.removed)
    active_changed = track(mgr.active_changed)
    changed = track(mgr.changed)

    bg_node = root.childNodes()[0]
    second_node.remove()
    doc.setActiveNode(bg_node)
    mgr.update()

    assert len(removed) == 1
    assert removed[0][0]._node is second_node
    assert len(active_changed) == 1
    assert len(changed) == 1


def test_layer_moved_to_different_parent():
    """Moving a layer into a group emits parent_changed and changed."""
    mgr, doc = make_manager()
    root = doc.rootNode()
    bg_node = root.childNodes()[0]
    bg_layer = mgr.find(bg_node.uniqueId())

    group_node = doc.createGroupLayer("Group")
    root.addChildNode(group_node, None)
    mgr.update()  # absorb group addition

    parent_changed = track(mgr.parent_changed)
    changed = track(mgr.changed)

    root.removeChildNode(bg_node)
    group_node.addChildNode(bg_node, None)
    mgr.update()

    assert len(parent_changed) == 1
    assert parent_changed[0][0] is bg_layer
    assert len(changed) == 1
    # The layer wrapper must have updated its cached parent id.
    assert bg_layer is not None
    assert bg_layer._parent == group_node.uniqueId()


def test_layer_name_changed():
    """Renaming a layer emits changed but not removed or active_changed."""
    mgr, doc = make_manager()
    bg_node = doc.rootNode().childNodes()[0]

    removed = track(mgr.removed)
    active_changed = track(mgr.active_changed)
    changed = track(mgr.changed)

    bg_node.setName("Renamed Background")
    mgr.update()

    assert len(changed) == 1
    assert len(removed) == 0
    assert len(active_changed) == 0
    bg_layer = mgr.find(bg_node.uniqueId())
    assert bg_layer is not None
    assert bg_layer.name == "Renamed Background"


def test_document_closed():
    """Once the document reference is cleared, update() becomes a no-op and
    the manager reports as falsy."""
    mgr, _doc = make_manager()

    removed = track(mgr.removed)
    active_changed = track(mgr.active_changed)
    changed = track(mgr.changed)

    mgr._doc = None
    mgr.update()

    assert len(removed) == 0
    assert len(active_changed) == 0
    assert len(changed) == 0
    assert not bool(mgr)
