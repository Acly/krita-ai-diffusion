"""Tests for polling behavior of LayerManager in ai_diffusion/layer.py."""

from __future__ import annotations

from krita import Document
from PyQt5.QtCore import Qt

from ai_diffusion.eventloop import process_python_events
from ai_diffusion.image import BlendMode, Bounds, Extent, Image
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


# ---------------------------------------------------------------------------
# Tests for LayerManager.active
# ---------------------------------------------------------------------------


def test_active_already_observed():
    """Fast path: active node is already in _layers, no update is triggered."""
    mgr, doc = make_manager()
    root = doc.rootNode()

    second_node = doc.createNode("Layer 2", "paintlayer")
    root.addChildNode(second_node, None)
    mgr.update()  # second_node is now in _layers
    doc.setActiveNode(second_node)
    mgr.update()  # _active_id is updated to second_node

    # Record the update count to confirm no extra update is triggered.
    update_calls = track(mgr.changed)

    layer = mgr.active

    assert layer._node is second_node
    assert len(update_calls) == 0  # _layers already had the answer


def test_active_in_tree_but_manager_stale():
    """Update path: active node exists in the document hierarchy but the manager
    hasn't polled since it was added.  Accessing .active triggers a forced update
    that discovers the node and returns it."""
    mgr, doc = make_manager()
    root = doc.rootNode()

    # Add a new node and make it active WITHOUT calling mgr.update() first.
    second_node = doc.createNode("Layer 2", "paintlayer")
    root.addChildNode(second_node, None)
    doc.setActiveNode(second_node)

    # Manager is stale: second_node is not yet in _layers.
    assert mgr.find(second_node.uniqueId()) is None

    layer = mgr.active

    # .active must have run an internal update to discover the node.
    assert layer._node is second_node
    assert mgr.find(second_node.uniqueId()) is not None


def test_active_not_in_tree_falls_back_to_last_active():
    """Fallback path: the node Krita reports as active is not part of the document
    hierarchy at all (e.g. immediately after a merge/create operation).  .active
    must return the most recently observed active layer instead."""
    mgr, doc = make_manager()
    root = doc.rootNode()

    # Establish a known active layer so _last_active is set.
    second_node = doc.createNode("Layer 2", "paintlayer")
    root.addChildNode(second_node, None)
    mgr.update()
    doc.setActiveNode(second_node)
    mgr.update()  # _last_active = second layer, _active_id = second_node.id

    # Confirm second_node is the current active.
    assert mgr.active._node is second_node

    # Simulate the transient state: second_node is removed from the tree
    # (so _layers no longer holds it) and a brand-new node that hasn't been
    # inserted anywhere is reported as active by Krita.
    second_node.remove()
    mgr.update()  # purges second_node from _layers; _active_id still points to it

    orphan_node = doc.createNode("Orphan", "paintlayer")  # not added to any parent
    doc.setActiveNode(orphan_node)

    # find() misses orphan, updated()._layers.get(_active_id) also misses the
    # removed second_node → property falls back to _last_active.
    layer = mgr.active

    assert layer._node is second_node


# ---------------------------------------------------------------------------
# Tests for LayerManager.update_layer_image
# ---------------------------------------------------------------------------


def test_write_pixels():
    mgr, _doc = make_manager()
    layer_bounds = Bounds(0, 0, 4, 4)
    new_bounds = Bounds(0, 0, 2, 4)

    initial = Image.create(Extent(4, 4), fill=Qt.GlobalColor.red)
    new_img = Image.create(Extent(2, 4), fill=Qt.GlobalColor.blue)

    test_layer = mgr.create("Layer", Image.copy(initial), layer_bounds)
    test_layer.write_pixels(new_img, new_bounds)

    expected = Image.create(Extent(4, 4), fill=0)
    expected.draw_image(new_img, new_bounds.offset)
    assert Image.compare(test_layer.get_pixels(), expected) <= 0.001


def test_update_layer_image():
    mgr, _doc = make_manager()
    layer_bounds = Bounds(0, 0, 4, 4)
    write_bounds = Bounds(0, 0, 2, 4)

    initial = Image.create(Extent(4, 4), fill=Qt.GlobalColor.red)
    new_img = Image.create(Extent(2, 4), fill=Qt.GlobalColor.blue)

    test_layer = mgr.create("Layer", Image.copy(initial), layer_bounds)
    old_id = test_layer.id
    replacement = mgr.update_layer_image(test_layer, new_img, write_bounds)

    process_python_events()
    mgr.update()

    assert mgr.find(old_id) is None
    assert replacement.name == "Layer"

    expected = Image.copy(initial)
    expected.draw_image(new_img, write_bounds.offset, BlendMode.replace)
    assert Image.compare(replacement.get_pixels(), expected) <= 0.001
