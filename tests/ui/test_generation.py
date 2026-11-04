import asyncio
from pathlib import Path

import pytest
from krita import Krita, Selection
from PyQt6.QtCore import QByteArray

from ai_diffusion.backend.resources import ControlMode
from ai_diffusion.backend.workflow import InpaintMode
from ai_diffusion.document import KritaDocument
from ai_diffusion.model.connection import Connection, ConnectionState
from ai_diffusion.model.custom_workflow import WorkflowCollection
from ai_diffusion.model.model import DocumentModel
from ai_diffusion.model.root import root
from ai_diffusion.settings import settings
from ai_diffusion.style import Style
from ai_diffusion.ui.generation import GenerationWidget

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


@pytest.fixture()
def workflows_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "workflows"
    folder.mkdir()
    return folder


@qtapp
async def test_generate_button_label(workflows_dir: Path):
    # -- setup (mirrors design.py main + ControlPane._open_document) --
    settings.load()
    root.init()

    krita_doc = Krita.instance().openDocument("test")
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

    widget = GenerationWidget()
    widget.model = model

    try:
        # -- 1. no selection, strength = 1.0 → "Generate" --
        model.strength = 1.0
        assert widget.generate_button.operation == "Generate"
        assert widget.mode_button.text() == "Generate"
        assert widget.region_prompt.parentWidget() is widget.prompt_frame
        assert widget.prompt_frame.parentWidget() is widget.mode_area
        assert widget.control_list.parentWidget() is widget
        mode_button_size = widget.mode_button.size()
        mode_actions = widget.inpaint_menu.actions()
        assert [action.text() for action in mode_actions] == [
            "Generate",
            "Fill",
            "Expand",
            "Add Content",
            "Remove Content",
            "Replace Background",
            "Edit",
        ]
        assert all(not action.isEnabled() for action in mode_actions[1:-1])

        # -- 2. create a selection → "Fill" --
        w, h = krita_doc.width() // 2, krita_doc.height() // 2
        x, y = krita_doc.width() // 4, krita_doc.height() // 4
        sel = Selection()
        sel.setPixelData(QByteArray(bytes([255] * (w * h))), x, y, w, h)
        krita_doc.setSelection(sel)

        # KritaDocument polls for selection changes on a 20 ms timer
        for _ in range(200):
            await asyncio.sleep(0.01)
            if doc.selection_bounds is not None:
                break
        assert doc.selection_bounds is not None

        assert widget.generate_button.operation == "Fill"
        assert widget.mode_button.text() == "Fill"
        assert widget.mode_button.size() == mode_button_size
        assert all(action.isEnabled() for action in mode_actions[1:-1])

        widget.change_inpaint_mode(InpaintMode.add_object, False)
        assert widget.generate_button.operation == "Add Content"
        assert widget.mode_button.text() == "Add Content"
        assert widget.mode_button.size() == mode_button_size
        widget.change_inpaint_mode(InpaintMode.automatic, False)

        # -- 3. remove selection, lower strength → "Refine" --
        krita_doc.setSelection(None)
        for _ in range(200):
            await asyncio.sleep(0.01)
            if doc.selection_bounds is None:
                break
        assert doc.selection_bounds is None

        model.strength = 0.5
        assert widget.generate_button.operation == "Refine"
        assert widget.mode_button.text() == "Generate"
        assert widget.mode_button.size() == mode_button_size
        assert all(not action.isEnabled() for action in mode_actions[1:-1])
        control = model.control.emplace()
        control.mode = ControlMode.reference
        controls = widget.control_list
        controls.toggle_advanced(controls._widgets[0])
        panel = controls._advanced_panel
        assert panel is not None
        assert not panel.link_button.isHidden()
        control.mode = ControlMode.depth
        assert panel.link_button.isHidden()
        control.mode = ControlMode.reference
        assert not panel.link_button.isHidden()
        assert not panel.link_button.isEnabled()
        assert panel.link_button.toolTip() == "Not linked to any region."
        region = model.regions.create_region(group=False)
        assert panel.link_button.isEnabled()
        assert f"Click to link to {region.name}" in panel.link_button.toolTip()
        panel.link_button.click()
        assert control.region is region
        assert "Click to unlink." in panel.link_button.toolTip()
        panel.link_button.click()
        assert control.region is None
        panel.link_button.click()
        model.regions.active = model.regions
        assert controls._widgets == []
        assert controls._advanced_panel is None

        def minimized_header_size():
            header = next(w for w in widget.region_prompt._inactive_regions if w.region is region)
            layout = header.layout()
            assert layout is not None
            return layout.count()

        assert minimized_header_size() == 3
        control.region = None
        assert minimized_header_size() == 2
        control.region = region
        assert minimized_header_size() == 3
        control.mode = ControlMode.depth
        assert minimized_header_size() == 2
        control.mode = ControlMode.reference
        assert minimized_header_size() == 3
        other_control = model.control.emplace(region)
        assert minimized_header_size() == 4
        model.control.remove(other_control)
        assert minimized_header_size() == 3
        model.regions.active = region
        assert [w.control for w in controls._widgets] == [control]
        region.remove()
        assert controls._widgets == []
    finally:
        await conn.disconnect()
