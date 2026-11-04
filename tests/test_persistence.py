"""Tests for ai_diffusion/persistence.py - the layer that persists DocumentModel state to Krita
document annotations and the plugin's settings.json file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from krita import Document as MockKritaDocument
from krita import Krita
from PyQt6.QtCore import QByteArray

from ai_diffusion.backend.api import InpaintMode
from ai_diffusion.backend.resources import ControlMode
from ai_diffusion.document import KritaDocument
from ai_diffusion.image import Bounds, Extent, Image, ImageCollection
from ai_diffusion.model.connection import Connection
from ai_diffusion.model.custom_workflow import WorkflowCollection
from ai_diffusion.model.jobs import Job, JobKind, JobParams, JobState
from ai_diffusion.model.model import DocumentModel, InpaintContext, QueueMode
from ai_diffusion.persistence import ModelSync, RecentlyUsedSync
from ai_diffusion.settings import Settings
from ai_diffusion.style import Style

from .conftest import qtapp

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def workflows_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "workflows"
    folder.mkdir()
    return folder


def _make_model(krita_doc: MockKritaDocument, workflows_dir: Path) -> DocumentModel:
    Krita.instance().setActiveDocument(krita_doc)
    doc = KritaDocument.active()
    assert doc is not None
    conn = Connection()
    wf_coll = WorkflowCollection(conn, folder=workflows_dir)
    return DocumentModel(doc, conn, wf_coll)


def _make_style(filename: str = "test.json", checkpoint: str = "test_sd15.safetensors") -> Style:
    style = Style(Path(filename))
    style.checkpoints = [checkpoint]
    return style


# ---------------------------------------------------------------------------
# test_recently_used
# ---------------------------------------------------------------------------


def test_recently_used(workflows_dir: Path, tmp_path: Path):
    """RecentlyUsedSync writes recently used parameters to settings and re-applies them to new
    models that have no stored annotation yet."""
    settings_path = tmp_path / "settings.json"

    # Use a private Settings instance so the test doesn't pollute the global one.
    local_settings = Settings()
    local_settings.default_path = settings_path

    # Patch the module-level settings reference used inside persistence.py
    import ai_diffusion.persistence as persistence_mod
    import ai_diffusion.settings as settings_mod

    original_settings = settings_mod.settings
    persistence_mod.settings = local_settings  # type: ignore[assignment]
    settings_mod.settings = local_settings  # type: ignore[assignment]
    try:
        recently_used = RecentlyUsedSync()

        # ── model 1: set several tracked properties ──────────────────────
        krita_doc1 = Krita.instance().openDocument("test1")
        model1 = _make_model(krita_doc1, workflows_dir)
        recently_used.track(model1)

        style1 = _make_style("custom.json", "custom_sd15.safetensors")
        model1.style = style1
        model1.batch_count = 4
        model1.translation_enabled = False
        model1.inpaint.mode = InpaintMode.fill
        model1.inpaint.use_inpaint = False
        model1.inpaint.context = InpaintContext.entire_image
        model1.upscale.upscaler = "RealESRGAN_x4plus.pth"

        # ── model 2: fresh doc, load recently used ────────────────────────
        recently_used2 = RecentlyUsedSync.from_settings()
        krita_doc2 = Krita.instance().openDocument("test2")
        model2 = _make_model(krita_doc2, workflows_dir)
        recently_used2.track(model2)

        # model2 has no annotation → values are taken from recently_used
        assert model2.batch_count == 4
        assert model2.translation_enabled is False
        assert model2.inpaint.mode is InpaintMode.fill
        assert model2.inpaint.use_inpaint is False
        assert model2.inpaint.context is InpaintContext.entire_image
        assert model2.upscale.upscaler == "RealESRGAN_x4plus.pth"
    finally:
        persistence_mod.settings = original_settings  # type: ignore[assignment]
        settings_mod.settings = original_settings  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# test_sync
# ---------------------------------------------------------------------------


@qtapp
async def test_load_legacy_document(workflows_dir: Path):
    """Documents saved before custom inpaint mode, fill mode and prompt focus were removed
    still load correctly."""
    krita_doc = Krita.instance().openDocument("test")
    legacy_state = {
        "version": 1,
        "batch_count": 3,
        "inpaint": {
            "mode": 6,  # InpaintMode.custom, removed in v2.0
            "fill": 2,  # FillMode.blur, removed as custom setting in v2.0
            "use_inpaint": False,
            "use_prompt_focus": True,  # removed in v2.0
            "context": InpaintContext.entire_image.value,
        },
    }
    annotation = QByteArray(json.dumps(legacy_state).encode())
    krita_doc.setAnnotation("ai_diffusion/ui.json", "AI Diffusion Plugin: ui.json", annotation)

    model = _make_model(krita_doc, workflows_dir)
    _sync = ModelSync(model)

    assert model.batch_count == 3
    assert model.inpaint.mode is InpaintMode.automatic
    assert model.inpaint.use_inpaint is False
    assert model.inpaint.context is InpaintContext.entire_image


@qtapp
async def test_sync(workflows_dir: Path):
    """ModelSync persists DocumentModel state as a document annotation and restores it when a new
    ModelSync is created for the same document."""

    krita_doc = Krita.instance().openDocument("test")
    model1 = _make_model(krita_doc, workflows_dir)

    style = _make_style("synced.json", "synced_sd15.safetensors")
    model1.style = style
    model1.batch_count = 3
    model1.strength = 0.75
    model1.seed = 12345
    model1.fixed_seed = True
    model1.translation_enabled = False
    model1.queue_mode = QueueMode.front
    model1.regions.positive = "a majestic mountain"
    model1.regions.negative = "blurry"
    model1.inpaint.mode = InpaintMode.fill
    model1.inpaint.use_inpaint = False
    model1.live.strength = 0.55
    model1.upscale.upscaler = "4x-UltraSharp.pt"
    model1.upscale.strength = 0.42
    model1.upscale.use_diffusion = False
    region = model1.regions.emplace()
    region.positive = "regional prompt"
    edit_region = model1.edit_regions.emplace()
    edit_region.positive = "edit prompt"
    model1.control.emplace(model1.regions).mode = ControlMode.reference
    model1.control.emplace(region).mode = ControlMode.depth
    model1.control.emplace(edit_region).mode = ControlMode.reference

    # Create sync and let it flush (it saves immediately on construction and on demand)
    sync1 = ModelSync(model1)
    sync1._save()  # force synchronous write so annotation is populated before we read it

    # ── restore from the same krita_doc ──────────────────────────────────
    # Build a second DocumentModel backed by the same MockKritaDocument (same annotation store)
    Krita.instance().setActiveDocument(krita_doc)
    doc2 = KritaDocument(krita_doc, None)
    conn2 = Connection()
    wf_coll2 = WorkflowCollection(conn2, folder=workflows_dir)
    model2 = DocumentModel(doc2, conn2, wf_coll2)
    _sync2 = ModelSync(model2)  # triggers _load() from annotation

    assert model2.batch_count == model1.batch_count
    assert model2.strength == pytest.approx(model1.strength)
    assert model2.seed == model1.seed
    assert model2.fixed_seed == model1.fixed_seed
    assert model2.translation_enabled == model1.translation_enabled
    assert model2.queue_mode is model1.queue_mode
    assert model2.regions.positive == model1.regions.positive
    assert model2.regions.negative == model1.regions.negative
    assert model2.inpaint.mode is model1.inpaint.mode
    assert model2.inpaint.use_inpaint == model1.inpaint.use_inpaint
    assert model2.live.strength == pytest.approx(model1.live.strength)
    assert model2.upscale.upscaler == model1.upscale.upscaler
    assert model2.upscale.strength == pytest.approx(model1.upscale.strength)
    assert model2.upscale.use_diffusion == model1.upscale.use_diffusion
    assert len(model2.control.all) == 3
    common, regional, edit = model2.control.all
    assert common.region is None
    assert regional.region is None
    regional.mode = ControlMode.reference
    restored_region = next(iter(model2.regions))
    restored_edit_region = next(iter(model2.edit_regions))
    assert regional.region is restored_region
    assert restored_region.positive == "regional prompt"
    assert edit.region is restored_edit_region
    assert restored_edit_region.positive == "edit prompt"


# ---------------------------------------------------------------------------
# test_history
# ---------------------------------------------------------------------------


@qtapp
async def test_history(workflows_dir: Path):
    """ModelSync encodes finished job result images into the document annotations and
    restores them (with correct images and metadata) when loaded by a new ModelSync."""

    krita_doc = Krita.instance().openDocument("test")
    model1 = _make_model(krita_doc, workflows_dir)
    sync1 = ModelSync(model1)

    # Build a finished diffusion job with two result images
    bounds = Bounds(0, 0, 512, 512)
    params = JobParams(bounds=bounds, name="history test", seed=99)
    job1 = model1.jobs.add_job(Job("history-job-1", JobKind.diffusion, params))

    img_a = Image.create(Extent(512, 512), fill=0xFFFF0000)  # red
    img_b = Image.create(Extent(512, 512), fill=0xFF0000FF)  # blue
    results = ImageCollection([img_a, img_b])
    model1.jobs.set_results(job1, results)
    model1.jobs.notify_finished(job1)

    # Wait for the async image-saving task to complete
    if sync1._image_task is not None:
        await sync1._image_task

    # Force the state JSON to be written as well
    sync1._save()

    # ── restore ───────────────────────────────────────────────────────────
    Krita.instance().setActiveDocument(krita_doc)
    doc2 = KritaDocument(krita_doc, None)
    conn2 = Connection()
    wf_coll2 = WorkflowCollection(conn2, folder=workflows_dir)
    model2 = DocumentModel(doc2, conn2, wf_coll2)
    _sync2 = ModelSync(model2)

    # The finished job should have been restored in model2's queue
    restored_job = model2.jobs.find("history-job-1")
    assert restored_job is not None, "finished job not found in restored model"
    assert restored_job.state is JobState.finished
    assert restored_job.params.name == "history test"
    assert restored_job.params.seed == 99
    assert len(restored_job.results) == 2

    # Check image content is intact
    restored_a = restored_job.results[0]
    restored_b = restored_job.results[1]
    assert restored_a.extent == Extent(512, 512)
    assert restored_b.extent == Extent(512, 512)
    # Pixel values should be close to the originals (allow for WebP compression rounding)
    assert Image.compare(restored_a, img_a) < 0.02
    assert Image.compare(restored_b, img_b) < 0.02
