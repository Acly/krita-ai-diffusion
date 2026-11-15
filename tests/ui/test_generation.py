import asyncio
from pathlib import Path

import pytest
from krita import Krita, Selection
from PyQt6.QtCore import QByteArray, QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QAbstractSlider

from ai_diffusion.backend.resources import Arch, ControlMode
from ai_diffusion.backend.workflow import InpaintMode
from ai_diffusion.document import KritaDocument
from ai_diffusion.files import File, FileFormat
from ai_diffusion.model.connection import Connection, ConnectionState
from ai_diffusion.model.custom_workflow import WorkflowCollection
from ai_diffusion.model.model import DocumentModel
from ai_diffusion.model.root import root
from ai_diffusion.settings import settings
from ai_diffusion.style import Style
from ai_diffusion.ui.control import ControlAdvancedPanel, LoraAdvancedPanel
from ai_diffusion.ui.generation import GenerationWidget
from ai_diffusion.ui.lora import LoraSelectionDialog
from ai_diffusion.ui.style import LoraPreviewMenu

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
        minimum_height = widget.minimumSizeHint().height()
        control = model.control.emplace()
        assert widget.minimumSizeHint().height() == minimum_height
        extra_controls = [model.control.emplace() for _ in range(3)]
        assert widget.minimumSizeHint().height() == minimum_height
        for extra in extra_controls:
            model.control.remove(extra)
        assert widget.minimumSizeHint().height() == minimum_height
        control.mode = ControlMode.reference
        controls = widget.control_list
        controls.toggle_advanced(controls._widgets[0])
        panel = controls._advanced_panel
        assert isinstance(panel, ControlAdvancedPanel)
        assert not panel.link_button.isHidden()
        control.mode = ControlMode.depth
        assert panel.link_button.isHidden()
        control.mode = ControlMode.reference
        assert not panel.link_button.isHidden()
        assert not panel.link_button.isEnabled()
        assert panel.link_button.toolTip() == "Not linked to any region."

        lora_file = File.remote("test-loras/detail.safetensors", FileFormat.lora)
        lora_file.arch = Arch.sd15
        root.files.loras.add(lora_file)
        lora_file = root.files.loras.find(lora_file.id)
        assert lora_file is not None
        root.files.loras.set_meta(lora_file, "lora_strength", 0.65)
        incompatible = File.remote("test-loras/xl.safetensors", FileFormat.lora)
        incompatible.arch = Arch.sdxl
        root.files.loras.add(incompatible)
        root.files.loras.add(File.remote("test-loras/unknown.safetensors", FileFormat.lora))
        root.files.loras.add(File.remote("test-loras/sampler.safetensors", FileFormat.lora))
        client.models.resources["lora-test-sd15"] = "test-loras/sampler.safetensors"
        dialog = LoraSelectionDialog(model, controls)
        dialog.selected.connect(controls._add_lora)
        dialog.folders.setCurrentText("test-loras")
        assert dialog.items.count() == 2
        dialog.open_at(QPoint(10, 10))
        assert dialog.search.hasFocus()
        for key, row in [
            (Qt.Key.Key_Down, 1),
            (Qt.Key.Key_Down, 1),
            (Qt.Key.Key_Up, 0),
            (Qt.Key.Key_Up, 0),
        ]:
            QTest.keyClick(dialog.search, key)  # type: ignore
            assert dialog.items.currentRow() == row
            assert dialog.search.hasFocus()
        QTest.keyClicks(dialog.search, "DETAIL")  # type: ignore
        assert dialog.items.count() == 1
        assert dialog.search.text() == "DETAIL"
        assert dialog.search.hasFocus()
        dialog.grab()  # Paint the custom list delegate.
        assert not model.loras
        QTest.keyClick(dialog.search, Qt.Key.Key_Return)  # type: ignore
        assert len(model.loras) == 1
        lora_widget = controls._lora_widgets[0]
        assert lora_widget.strength == 0.65
        first = controls._flow_layout.itemAt(0)
        assert first is not None and first.widget() is lora_widget
        controls._refresh()
        first = controls._flow_layout.itemAt(0)
        assert first is not None and first.widget() is lora_widget
        assert controls._add_control_button.size() == controls._add_lora_button.size()
        QTest.mouseClick(lora_widget.thumbnail, Qt.MouseButton.LeftButton)  # type: ignore
        lora_panel = controls._advanced_panel
        assert isinstance(lora_panel, LoraAdvancedPanel)
        assert controls.advanced_owner is lora_widget
        lora_widget.slider.triggerAction(QAbstractSlider.SliderAction.SliderSingleStepAdd)
        assert lora_widget.strength == 0.7
        lora_widget.slider.triggerAction(QAbstractSlider.SliderAction.SliderSingleStepSub)
        assert lora_widget.strength == 0.65
        preview_menu = LoraPreviewMenu(lora_file, lora_panel, model.document)
        assert [action.text() for action in preview_menu.actions()] == [
            "Use current canvas",
            "Open file...",
        ]
        assert all(action.isEnabled() for action in preview_menu.actions())
        assert not lora_panel.preview_button.isHidden()
        lora_widget.slider.setValue(0)
        assert lora_widget.strength == 0
        assert lora_panel.strength.value() == 0
        lora_panel.strength.setValue(-125)
        assert lora_widget.strength == -1.25
        lora_panel.strength.setValue(150)
        assert lora_widget.strength == 1.5
        lora_panel.triggers.setText("fine details")
        assert lora_file.meta("lora_triggers") == "fine details"
        lora_widget.grab()
        controls.toggle_advanced(controls._widgets[0])
        assert controls.advanced_owner is controls._widgets[0]
        controls.toggle_advanced(lora_widget)
        lora_widget.remove()
        assert not model.loras and not controls._lora_widgets
        assert controls.advanced_owner is None
        controls.toggle_advanced(controls._widgets[0])
        panel = controls._advanced_panel
        assert isinstance(panel, ControlAdvancedPanel)

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
            prompts = widget.region_prompt
            assert prompts._regions_above.count() + prompts._regions_below.count() == len(
                prompts._inactive_regions
            )
            header = next(w for w in prompts._inactive_regions if w.region is region)
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
        for file in list(root.files.loras):
            if file.id.startswith("test-loras/"):
                root.files.loras.remove(root.files.loras.find_index(file.id))
        await conn.disconnect()
