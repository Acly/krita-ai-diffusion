"""Tests for ai_diffusion.model.DocumentModel - the document view model that collects UI parameters
and document data and forwards them as WorkflowInput to image generation clients."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from krita import Document as MockKritaDocument
from krita import Krita, Selection
from PyQt5.QtCore import QByteArray, Qt

from ai_diffusion.backend.api import WorkflowInput, WorkflowKind
from ai_diffusion.backend.client import CheckpointInfo, ClientEvent, ClientMessage
from ai_diffusion.backend.resources import Arch, ControlMode
from ai_diffusion.document import KritaDocument
from ai_diffusion.image import BlendMode, Bounds, Extent, Image, ImageCollection
from ai_diffusion.layer import Layer, LayerType
from ai_diffusion.model.connection import Connection, ConnectionState
from ai_diffusion.model.custom_workflow import WorkflowCollection
from ai_diffusion.model.jobs import Job, JobKind, JobParams, JobRegion, JobState
from ai_diffusion.model.model import DocumentModel, ErrorKind, ProgressKind, no_error
from ai_diffusion.settings import ApplyBehavior, ApplyRegionBehavior
from ai_diffusion.style import Style

from .conftest import qtapp
from .mock.client import MockClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_for_state(conn: Connection, *exclude: ConnectionState, timeout: int = 100):
    for _ in range(timeout):
        await asyncio.sleep(0)
        if conn.state not in exclude:
            return
    raise TimeoutError(f"Connection stuck at {conn.state!r} after {timeout} iterations")


def _make_style(checkpoint: str = "test_sd15.safetensors") -> Style:
    style = Style(Path("test.json"))
    style.checkpoints = [checkpoint]
    return style


@asynccontextmanager
async def _model_env(
    krita_doc: MockKritaDocument, workflows_folder: Path
) -> AsyncIterator[tuple[DocumentModel, MockClient]]:
    """Async context manager that sets up a fully wired DocumentModel/MockClient pair and tears down
    the Connection cleanly on exit to avoid pending-task warnings."""
    from ai_diffusion.model.root import root as plugin_root

    client = MockClient()

    Krita.instance().setActiveDocument(krita_doc)
    doc = KritaDocument.active()
    assert doc is not None, "KritaDocument.active() returned None"

    conn = Connection()
    conn.connect(client)
    await _wait_for_state(conn, ConnectionState.connecting, ConnectionState.disconnected)
    assert conn.state is ConnectionState.connected

    wf_coll = WorkflowCollection(conn, folder=workflows_folder)
    model = DocumentModel(doc, conn, wf_coll)
    model.style = _make_style()
    conn.message_received.connect(model.handle_message)
    previous_connection = getattr(plugin_root, "_connection", None)
    plugin_root._connection = conn
    try:
        yield model, client
    finally:
        await conn.disconnect()
        if previous_connection is None:
            del plugin_root._connection
        else:
            plugin_root._connection = previous_connection


async def _wait_for_enqueue(
    client: MockClient, count: int = 1, timeout: int = 200
) -> list[WorkflowInput]:
    """Yield to the event loop until *count* WorkflowInputs have been enqueued."""
    for _ in range(timeout):
        await asyncio.sleep(0)
        if len(client.enqueued) >= count:
            return client.enqueued[:count]
    raise TimeoutError(f"Only {len(client.enqueued)}/{count} jobs enqueued within the timeout")


async def _run_generate(model: DocumentModel, client: MockClient) -> Job:
    """Call model.generate() and wait for the new job to appear in the queue with its ID set."""
    n = len(client.enqueued)
    model.generate()
    for _ in range(200):
        await asyncio.sleep(0)
        if len(client.enqueued) > n:
            if job := model.jobs.find(f"mock-job-{n}"):
                return job
    raise TimeoutError("Job was not enqueued in model.jobs")


async def _wait_for_job_state(job: Job, state: JobState, timeout: int = 100):
    """Yield to the event loop until the job reaches the expected state."""
    for _ in range(timeout):
        await asyncio.sleep(0)
        if job.state is state:
            return
    raise TimeoutError(f"Job stuck at {job.state!r}, expected {state!r}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workflows_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "workflows"
    folder.mkdir()
    return folder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@qtapp
async def test_generate_simple(workflows_dir: Path):
    """With no selection and strength=1.0 the forwarded workflow should be WorkflowKind.generate."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, client):
        model.strength = 1.0
        model.regions.positive = "a red cat"
        model.generate()

        result = await _wait_for_enqueue(client)
        assert result[0].kind is WorkflowKind.generate
        # Pure generation: no input image is passed to the workflow
        assert result[0].images is not None
        assert result[0].images.initial_image is None


@qtapp
async def test_generate_references(workflows_dir: Path):
    krita_doc = Krita.instance().openDocument("test")

    # Add three layers with distinct solid colours so we can identify them by image content
    layer_colors = [
        ("a", Qt.GlobalColor.red),
        ("b", Qt.GlobalColor.green),
        ("c", Qt.GlobalColor.blue),
    ]
    layer_images: dict[str, Image] = {}
    layer_nodes = {}
    for name, color in layer_colors:
        node = krita_doc.createNode(name, "paintlayer")
        img = Image.create(Extent(512, 512), fill=color)
        node.setPixelData(img.to_packed_bytes(), 0, 0, 512, 512)
        krita_doc.rootNode().addChildNode(node, None)
        layer_images[name] = img
        layer_nodes[name] = node

    async with _model_env(krita_doc, workflows_dir) as (model, client):
        # Register a flux2_4b checkpoint so the model resolves to that architecture;
        # flux2_4b uses the "image X" replacement format for <layer:name> tokens.
        client.models.checkpoints["test_flux2.safetensors"] = CheckpointInfo(
            "test_flux2.safetensors", Arch.flux2_4b
        )
        model.style = _make_style("test_flux2.safetensors")
        model.strength = 1.0

        # Add an explicit reference control layer for layer "a"
        krita_doc.setActiveNode(layer_nodes["a"])
        ctrl = model.regions.control.emplace()
        ctrl.set_mode(ControlMode.reference)

        # Reference layers "b" and "c" (with a duplicate "b") via the prompt
        model.regions.positive = "Use <layer:b> and <layer:c>, repeat <layer:b>"
        model.generate()

        result = await _wait_for_enqueue(client)
        assert result[0].kind is WorkflowKind.generate

        cond = result[0].conditioning
        assert cond is not None

        # All three layers were added as reference controls in order a, b, c
        assert len(cond.control) == 3
        assert all(c.mode is ControlMode.reference for c in cond.control)
        ctrl_imgs = [c.image for c in cond.control]
        assert all(img is not None for img in ctrl_imgs)
        img_a, img_b, img_c = ctrl_imgs[0], ctrl_imgs[1], ctrl_imgs[2]
        assert img_a is not None and img_b is not None and img_c is not None
        assert Image.compare(img_a, layer_images["a"]) < 0.01
        assert Image.compare(img_b, layer_images["b"]) < 0.01
        assert Image.compare(img_c, layer_images["c"]) < 0.01

        # Indexing starts at 2 because in edit mode the first image is always
        # implicitly the canvas.
        assert cond.positive == "Use image 2 and image 3, repeat image 2"


@qtapp
async def test_generate_refine(workflows_dir: Path):
    """With an input image and strength<1.0 the forwarded workflow should be WorkflowKind.refine
    and the initial_image should reflect the content of the Krita document."""
    krita_doc = Krita.instance().openDocument("test")
    red_img = Image.create(Extent(512, 512), fill=Qt.GlobalColor.red)
    bg_node = krita_doc.rootNode().childNodes()[0]
    bg_node.setPixelData(red_img.to_packed_bytes(), 0, 0, 512, 512)

    async with _model_env(krita_doc, workflows_dir) as (model, client):
        model.strength = 0.5
        model.regions.positive = "a blue sky"
        model.generate()

        result = await _wait_for_enqueue(client)
        assert result[0].kind is WorkflowKind.refine

        # The workflow must carry the document image as initial input
        assert result[0].images is not None
        assert result[0].images.initial_image is not None
        assert Image.compare(result[0].images.initial_image, red_img) < 0.01


@qtapp
async def test_generate_inpaint(workflows_dir: Path):
    """With an active selection and strength=1.0 the forwarded workflow should be
    WorkflowKind.inpaint; the initial_image and hires_mask must reflect the document
    image and the active selection respectively."""
    krita_doc = Krita.instance().openDocument("test")
    # Paint a solid colour so pixels are easy to reason about
    blue_bgra = bytes([200, 0, 0, 255] * 512 * 512)  # BGRA: B=200 G=0 R=0 A=255
    bg_node = krita_doc.rootNode().childNodes()[0]
    bg_node.setPixelData(QByteArray(blue_bgra), 0, 0, 512, 512)

    # Place a 256×256 selection in the centre of the 512×512 canvas
    sel = Selection()
    sel_x, sel_y, sel_w, sel_h = 128, 128, 256, 256
    sel.setPixelData(QByteArray(bytes([0xFF] * sel_w * sel_h)), sel_x, sel_y, sel_w, sel_h)
    krita_doc.setSelection(sel)

    async with _model_env(krita_doc, workflows_dir) as (model, client):
        model.strength = 1.0
        model.regions.positive = "a white cloud"
        model.generate()

        result = await _wait_for_enqueue(client)
        assert result[0].kind is WorkflowKind.inpaint

        # An inpaint workflow must supply both an input image and a mask
        assert result[0].images is not None
        assert result[0].images.initial_image is not None
        assert result[0].images.hires_mask is not None

        # The image must not be empty (some extent covering the selection context area)
        img = result[0].images.initial_image
        assert img.extent.width > 0 and img.extent.height > 0

        # The mask is a grayscale image the same size as the input image; it must
        # have fully-white pixels where the selection was active.
        mask_img = result[0].images.hires_mask
        assert mask_img.extent == img.extent
        # The selection centre maps to (sel_cx - bounds.x, sel_cy - bounds.y) within the
        # cropped canvas.  We only verify that at least some mask pixels are fully selected
        # (value 255 in grayscale) to confirm the selection was transferred.
        mask_arr = mask_img.to_array()  # shape (H, W, 1), values in [0, 1]
        assert float(mask_arr.max()) > 0.99, "hires_mask should have fully-selected pixels"


@qtapp
async def test_generate_batch(workflows_dir: Path):
    """With batch_count=8 and a wildcard prompt, DocumentModel.generate should enqueue 8 separate jobs.
    Each job re-evaluates the wildcard with a different seed, so both options must appear."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, client):
        # Fix the seed so wildcard resolution is deterministic across runs
        model.fixed_seed = True
        model.seed = 0
        model.batch_count = 8
        model.regions.positive = "{apple|banana}"
        model.generate()

        results = await _wait_for_enqueue(client, count=8)
        assert len(results) == 8

        prompts = [r.conditioning.positive for r in results]  # type: ignore[union-attr]
        assert all(p in ("apple", "banana") for p in prompts)
        # With seed=0 and the default batch_size both options are guaranteed to appear
        assert "apple" in prompts
        assert "banana" in prompts


# ---------------------------------------------------------------------------
# Job processing tests
# ---------------------------------------------------------------------------


@qtapp
async def test_job_queued_progress_finished(workflows_dir: Path):
    """Happy path: queued → progress → finished delivers result images and marks the job done."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, client):
        job = await _run_generate(model, client)
        assert job.id is not None

        client.push(ClientMessage(ClientEvent.queued, job.id))
        await _wait_for_job_state(job, JobState.executing)
        assert model.progress == -1

        client.push(ClientMessage(ClientEvent.progress, job.id, progress=0.5))
        await asyncio.sleep(0)
        assert job.state is JobState.executing
        assert model.progress == pytest.approx(0.5)
        assert model.progress_kind is ProgressKind.generation

        result_images = ImageCollection([Image.create(Extent(512, 512))])
        client.push(ClientMessage(ClientEvent.finished, job.id, images=result_images))
        await _wait_for_job_state(job, JobState.finished)
        assert model.progress == pytest.approx(1.0)
        assert len(job.results) == 1


@qtapp
async def test_job_queued_upload_progress_interrupted(workflows_dir: Path):
    """Upload phase followed by generation progress, then server interruption → job cancelled."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, client):
        job = await _run_generate(model, client)
        assert job.id is not None

        client.push(ClientMessage(ClientEvent.queued, job.id))
        await _wait_for_job_state(job, JobState.executing)

        client.push(ClientMessage(ClientEvent.upload, job.id, progress=0.3))
        await asyncio.sleep(0)
        assert job.state is JobState.executing
        assert model.progress_kind is ProgressKind.upload
        assert model.progress == pytest.approx(0.3)

        client.push(ClientMessage(ClientEvent.progress, job.id, progress=0.6))
        await asyncio.sleep(0)
        assert model.progress_kind is ProgressKind.generation

        client.push(ClientMessage(ClientEvent.interrupted, job.id))
        await _wait_for_job_state(job, JobState.cancelled)
        assert model.progress == pytest.approx(0.0)


@qtapp
async def test_job_payment_required(workflows_dir: Path):
    """Payment required response marks the job as cancelled and sets an insufficient-funds error."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, client):
        job = await _run_generate(model, client)
        assert job.id is not None

        client.push(ClientMessage(ClientEvent.queued, job.id))
        await _wait_for_job_state(job, JobState.executing)

        client.push(
            ClientMessage(
                ClientEvent.payment_required,
                job.id,
                error="insufficient funds",
                result={"balance": 0, "cost": 10},
            )
        )
        await _wait_for_job_state(job, JobState.cancelled)
        assert model.error.kind is ErrorKind.insufficient_funds


@qtapp
async def test_job_queued_progress_error(workflows_dir: Path):
    """Server error mid-generation marks the job as cancelled and surfaces the error message."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, client):
        job = await _run_generate(model, client)
        assert job.id is not None

        client.push(ClientMessage(ClientEvent.queued, job.id))
        await _wait_for_job_state(job, JobState.executing)

        client.push(ClientMessage(ClientEvent.progress, job.id, progress=0.5))
        await asyncio.sleep(0)
        assert job.state is JobState.executing

        client.push(ClientMessage(ClientEvent.error, job.id, error="CUDA out of memory"))
        await _wait_for_job_state(job, JobState.cancelled)
        assert model.error.kind is ErrorKind.server_error
        assert "CUDA out of memory" in model.error.message


@qtapp
async def test_job_disconnect_reconnect(workflows_dir: Path):
    """Sporadic disconnect during generation: the first job's result is lost but its state is
    cleaned up (cancelled) when a second job completes successfully after reconnecting."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, client):
        # Enqueue first job and drive it into executing state
        job1 = await _run_generate(model, client)
        assert job1.id is not None
        client.push(ClientMessage(ClientEvent.queued, job1.id))
        await _wait_for_job_state(job1, JobState.executing)
        client.push(ClientMessage(ClientEvent.progress, job1.id, progress=0.4))
        await asyncio.sleep(0)

        # Simulate a sporadic server disconnect followed immediately by reconnect.
        # Connection handles these internally; job states are not touched yet.
        client.push(ClientMessage(ClientEvent.disconnected, ""))
        client.push(ClientMessage(ClientEvent.connected, ""))
        await asyncio.sleep(0)
        assert job1.state is JobState.executing  # still executing, just a hiccup

        # After reconnect the user triggers a new generation
        job2 = await _run_generate(model, client)
        assert job2.id is not None
        client.push(ClientMessage(ClientEvent.queued, job2.id))
        await _wait_for_job_state(job2, JobState.executing)
        client.push(ClientMessage(ClientEvent.progress, job2.id, progress=0.7))
        await asyncio.sleep(0)

        result_images = ImageCollection([Image.create(Extent(512, 512))])
        client.push(ClientMessage(ClientEvent.finished, job2.id, images=result_images))
        await _wait_for_job_state(job2, JobState.finished)

        # Finishing job2 cleans up job1 via _cancel_earlier_jobs
        assert job1.state is JobState.cancelled
        assert model.error == no_error


# ---------------------------------------------------------------------------
# Helpers for result / preview tests
# ---------------------------------------------------------------------------

_DOC_EXTENT = Extent(512, 512)
_DOC_BOUNDS = Bounds(0, 0, 512, 512)

# Solid fill colours chosen so images are visually distinct and easy to compare.
_RED = 0xFFFF0000
_GREEN = 0xFF00FF00


def _make_finished_job(
    model: DocumentModel,
    result_images: list[Image],
    regions: list[JobRegion] | None = None,
    bounds: Bounds = _DOC_BOUNDS,
    seed: int = 42,
) -> Job:
    """Add a finished diffusion job with the given result images to *model*'s queue."""
    params = JobParams(bounds=bounds, name="test job", seed=seed)
    if regions:
        params.regions = regions
    job = model.jobs.add(JobKind.diffusion, params)
    job.id = "test-job"
    assert job.id is not None
    model.jobs.set_results(job, ImageCollection(result_images))
    job.state = JobState.finished
    return job


def _paint_layers(model: DocumentModel) -> list[Layer]:
    """Return all paint layers currently visible in the document tree."""
    return [l for l in model.layers.updated().images if l.type is LayerType.paint]


def _layer_matches(layer: Layer, expected: Image, bounds: Bounds) -> bool:
    """True when *layer*'s pixel content at *bounds* is nearly identical to *expected*.

    Image.compare() returns a normalised RMSE in [0, 1] (values are divided by 255
    before comparison), so 0.01 is a tight but noise-tolerant threshold.
    """
    if layer.bounds.is_zero:
        return False
    return Image.compare(layer.get_pixels(bounds), expected) < 0.01


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@qtapp
async def test_show_preview(workflows_dir: Path):
    """show_preview creates a locked layer the first time; the second call reuses it."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, _client):
        await asyncio.sleep(0)  # let _handle_messages task start before disconnect
        result = Image.create(_DOC_EXTENT, fill=_RED)
        job = _make_finished_job(model, [result])
        assert job.id is not None

        # ── first call: a brand-new layer should be created ───────────────
        model.show_preview(job.id, 0)

        # Identify the preview layer from the document tree: it is the one locked layer.
        all_layers = _paint_layers(model)
        assert len(all_layers) == 2, "expected Background + 1 preview layer"
        preview_layers = [l for l in all_layers if l.is_locked]
        assert len(preview_layers) == 1, "expected exactly one locked (preview) layer"
        preview_layer = preview_layers[0]

        # Layer content must match the job result image.
        assert _layer_matches(preview_layer, result, _DOC_BOUNDS)

        # ── second call: must reuse the same layer, not create a new one ───
        model.show_preview(job.id, 0)

        all_layers_after = _paint_layers(model)
        assert len(all_layers_after) == 2, "second call must not add another layer"
        # The locked layer's identity (id) must be unchanged.
        preview_layers_after = [l for l in all_layers_after if l.is_locked]
        assert len(preview_layers_after) == 1
        assert preview_layers_after[0].id == preview_layer.id, (
            "second show_preview must reuse the existing layer"
        )


@qtapp
@pytest.mark.parametrize(
    "behavior",
    [ApplyBehavior.layer, ApplyBehavior.layer_active, ApplyBehavior.replace],
    ids=["layer", "layer_active", "replace"],
)
async def test_apply_result(workflows_dir: Path, behavior: ApplyBehavior):
    """apply_result writes the image to the correct document layer for every ApplyBehavior."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, _client):
        await asyncio.sleep(0)  # let _handle_messages task start before disconnect
        result = Image.create(_DOC_EXTENT, fill=_RED)
        params = JobParams(bounds=_DOC_BOUNDS, name="diffusion", seed=7)

        model.apply_result(result, params, behavior=behavior)

        # Regardless of behavior, a paint layer with the result content must exist.
        matches = [l for l in _paint_layers(model) if _layer_matches(l, result, _DOC_BOUNDS)]
        assert len(matches) >= 1, f"no layer with the result image found for {behavior}"

        if behavior in (ApplyBehavior.layer, ApplyBehavior.layer_active):
            # A completely new layer was added; the document grows by one.
            assert len(_paint_layers(model)) == 2
        # For ApplyBehavior.replace, update_layer_image schedules the old layer for async
        # removal, so only the presence of the result content is asserted above.


@qtapp
async def test_apply_preview(workflows_dir: Path):
    """Applying a result removes the preview layer that was shown beforehand."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, _client):
        await asyncio.sleep(0)  # let _handle_messages task start before disconnect
        result = Image.create(_DOC_EXTENT, fill=_GREEN)
        job = _make_finished_job(model, [result])
        assert job.id is not None

        # Show preview → a locked layer appears in the document tree.
        model.show_preview(job.id, 0)
        layers_with_preview = _paint_layers(model)
        preview_layers = [l for l in layers_with_preview if l.is_locked]
        assert len(preview_layers) == 1, "expected exactly one locked (preview) layer"
        preview_layer_id = preview_layers[0].id

        # Apply the result → preview layer must be removed synchronously.
        model.apply_generated_result(job.id, 0)

        # The preview node is no longer in the layer tree.
        assert model.layers.updated().find(preview_layer_id) is None, (
            "preview layer must be removed from the document after apply"
        )

        # The result image was written to a new layer in the document.
        result_layers = [l for l in _paint_layers(model) if _layer_matches(l, result, _DOC_BOUNDS)]
        assert len(result_layers) >= 1


@qtapp
async def test_apply_region_replace(workflows_dir: Path):
    """ApplyRegionBehavior.replace updates the linked region layers in place and re-links regions."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, _client):
        await asyncio.sleep(0)  # let _handle_messages task start before disconnect
        result = Image.create(_DOC_EXTENT, fill=_RED)

        # ── document setup: two paint layers ──────────────────────────────
        # 'Background' is already active.  Add a second paint layer on top.
        layer1 = model.layers.active  # Background
        layer2 = model.layers.create("Layer 2", Image.create(_DOC_EXTENT, fill=_GREEN), _DOC_BOUNDS)

        # ── region setup: each region linked to one layer ─────────────────
        region1 = model.regions.emplace()
        region1.link(layer1)

        region2 = model.regions.emplace()
        region2.link(layer2)

        # ── build a job with two JobRegions, one per document layer ───────
        job_regions = [
            JobRegion(layer_id=layer1.id_string, prompt="prompt A", bounds=_DOC_BOUNDS),
            JobRegion(layer_id=layer2.id_string, prompt="prompt B", bounds=_DOC_BOUNDS),
        ]
        job = _make_finished_job(model, [result], regions=job_regions)

        # ── apply with replace behavior ───────────────────────────────────
        model.apply_result(
            result,
            job.params,
            behavior=ApplyBehavior.layer,
            region_behavior=ApplyRegionBehavior.replace,
        )

        # Two replacement layers must exist, each carrying the result image content.
        # (The original layers are scheduled for async removal via remove_later().)
        all_paint = _paint_layers(model)
        replacements = [l for l in all_paint if _layer_matches(l, result, _DOC_BOUNDS)]
        assert len(replacements) == 2, (
            f"expected 2 replacement layers with result content, got {len(replacements)}"
        )

        # Each replacement layer must carry the result image content.
        for layer in replacements:
            assert _layer_matches(layer, result, _DOC_BOUNDS), (
                f"layer '{layer.name}' does not contain the expected result image"
            )

        # Each region must now be linked to one of the replacement layers.
        for region in (region1, region2):
            linked = [l for l in replacements if region.is_linked(l)]
            assert len(linked) >= 1, f"region '{region.positive}' lost its layer link after apply"


@qtapp
async def test_apply_region_group(workflows_dir: Path):
    """ApplyRegionBehavior.layer_group places results inside groups, hides old layers,
    and applies the group's alpha mask to the new result content."""
    krita_doc = Krita.instance().openDocument("test")
    async with _model_env(krita_doc, workflows_dir) as (model, _client):
        await asyncio.sleep(0)  # let _handle_messages task start before disconnect

        HALF = _DOC_EXTENT.width // 2

        # ── document setup ────────────────────────────────────────────────
        # layer1: root-level paint layer, left half transparent, right half opaque.
        layer1_content = Image.create(_DOC_EXTENT, fill=0)
        layer1_content.draw_image(
            Image.create(Extent(HALF, _DOC_EXTENT.height), fill=_GREEN),
            (HALF, 0),
            blend=BlendMode.replace,
        )
        layer1 = model.layers.create("Layer 1", layer1_content, _DOC_BOUNDS)

        # group: an existing group layer containing one paint layer.
        # The paint layer has the complementary alpha: left opaque, right transparent.
        group_child_content = Image.create(_DOC_EXTENT, fill=0)
        group_child_content.draw_image(
            Image.create(Extent(HALF, _DOC_EXTENT.height), fill=_GREEN),
            (0, 0),
            blend=BlendMode.replace,
        )
        group = model.layers.create_group("Layer Group")
        layer_in_group = model.layers.create(
            "In Group", group_child_content, _DOC_BOUNDS, parent=group
        )

        # ── regions ───────────────────────────────────────────────────────
        region1 = model.regions.emplace()
        region1.link(layer1)

        region2 = model.regions.emplace()
        region2.link(group)

        # ── job ──────────────────────────────────────────────────────────
        result = Image.create(_DOC_EXTENT, fill=_RED)
        job_regions = [
            JobRegion(layer_id=layer1.id_string, prompt="region1", bounds=_DOC_BOUNDS),
            JobRegion(layer_id=group.id_string, prompt="region2", bounds=_DOC_BOUNDS),
        ]
        job = _make_finished_job(model, [result], regions=job_regions)

        # ── apply ────────────────────────────────────────────────────────
        model.apply_result(
            result,
            job.params,
            behavior=ApplyBehavior.layer,
            region_behavior=ApplyRegionBehavior.layer_group,
        )

        # ── check: layer1 placed into a new group ─────────────────────────
        new_group = layer1.parent_layer
        assert new_group is not None
        assert new_group.type is LayerType.group
        assert not new_group.is_root

        # The result layer is the last child (topmost) of new_group; layer1 is below it.
        children1 = new_group.child_layers
        assert len(children1) == 2
        assert children1[-1] is not layer1  # result layer, not the original
        result1 = children1[-1]

        # region1 must now be linked to the new group (not the ungrouped layer1).
        assert region1.is_linked(new_group)

        # ── check: result for the already-grouped region is in the existing group ──
        children2 = group.child_layers
        assert len(children2) == 2
        assert children2[-1] is not layer_in_group
        result2 = children2[-1]

        # ── check: old layers hidden, new result layers visible ───────────
        assert not layer1.is_visible, "layer1 must be hidden after apply"
        assert not layer_in_group.is_visible, "layer_in_group must be hidden after apply"
        assert result1.is_visible, "result1 must be visible"
        assert result2.is_visible, "result2 must be visible"

        # ── check: alpha from existing layers applied to result content ───
        # layer1 had right half opaque → result1 right is red, left transparent.
        p1 = result1.get_pixels(_DOC_BOUNDS)
        r1_right = p1.pixel(HALF + 50, 50)
        r1_left = p1.pixel(50, 50)
        assert isinstance(r1_right, tuple) and r1_right[3] == 255, (
            "result1: right side must be opaque"
        )
        assert r1_right[:3] == (255, 0, 0), "result1: right side must be red"
        assert isinstance(r1_left, tuple) and r1_left[3] == 0, (
            "result1: left side must be transparent"
        )

        # layer_in_group had left half opaque → result2 left is red, right transparent.
        p2 = result2.get_pixels(_DOC_BOUNDS)
        r2_left = p2.pixel(50, 50)
        r2_right = p2.pixel(HALF + 50, 50)
        assert isinstance(r2_left, tuple) and r2_left[3] == 255, "result2: left side must be opaque"
        assert r2_left[:3] == (255, 0, 0), "result2: left side must be red"
        assert isinstance(r2_right, tuple) and r2_right[3] == 0, (
            "result2: right side must be transparent"
        )
