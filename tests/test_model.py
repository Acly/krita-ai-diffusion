"""Tests for ai_diffusion.model.Model - the document view model that collects UI parameters
and document data and forwards them as WorkflowInput to image generation clients."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from krita import Document as MockKritaDocument
from krita import Krita, Selection
from PyQt5.QtCore import QByteArray

from ai_diffusion.api import WorkflowInput, WorkflowKind
from ai_diffusion.client import ClientEvent, ClientMessage
from ai_diffusion.connection import Connection, ConnectionState
from ai_diffusion.custom_workflow import WorkflowCollection
from ai_diffusion.document import KritaDocument
from ai_diffusion.image import Bounds, Extent, Image, ImageCollection
from ai_diffusion.jobs import Job, JobState
from ai_diffusion.model import ErrorKind, Model, ProgressKind, no_error
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
) -> AsyncIterator[tuple[Model, MockClient]]:
    """Async context manager that sets up a fully wired Model/MockClient pair and tears down
    the Connection cleanly on exit to avoid pending-task warnings."""
    client = MockClient()

    Krita.instance().setActiveDocument(krita_doc)
    doc = KritaDocument.active()
    assert doc is not None, "KritaDocument.active() returned None"

    conn = Connection()
    conn.connect(client)
    await _wait_for_state(conn, ConnectionState.connecting, ConnectionState.disconnected)
    assert conn.state is ConnectionState.connected

    wf_coll = WorkflowCollection(conn, folder=workflows_folder)
    model = Model(doc, conn, wf_coll)
    model.style = _make_style()
    conn.message_received.connect(model.handle_message)
    try:
        yield model, client
    finally:
        await conn.disconnect()


async def _wait_for_enqueue(
    client: MockClient, count: int = 1, timeout: int = 200
) -> list[WorkflowInput]:
    """Yield to the event loop until *count* WorkflowInputs have been enqueued."""
    for _ in range(timeout):
        await asyncio.sleep(0)
        if len(client.enqueued) >= count:
            return client.enqueued[:count]
    raise TimeoutError(f"Only {len(client.enqueued)}/{count} jobs enqueued within the timeout")


async def _run_generate(model: Model, client: MockClient) -> Job:
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


@pytest.fixture(autouse=True)
def reset_krita_state():
    """Ensure a clean Krita singleton and KritaDocument cache for each test."""
    Krita._instance = None  # type: ignore[assignment]
    KritaDocument._instances.clear()
    yield
    Krita._instance = None  # type: ignore[assignment]
    KritaDocument._instances.clear()


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
    krita_doc = MockKritaDocument()
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
async def test_generate_refine(workflows_dir: Path):
    """With an input image and strength<1.0 the forwarded workflow should be WorkflowKind.refine
    and the initial_image should reflect the content of the Krita document."""
    krita_doc = MockKritaDocument()
    # Paint a solid, distinctive red colour so we can verify the image transfer
    red_bgra = bytes([0, 0, 200, 255] * 512 * 512)  # BGRA: B=0 G=0 R=200 A=255
    bg_node = krita_doc.rootNode().childNodes()[0]
    bg_node.setPixelData(QByteArray(red_bgra), 0, 0, 512, 512)

    async with _model_env(krita_doc, workflows_dir) as (model, client):
        model.strength = 0.5
        model.regions.positive = "a blue sky"
        model.generate()

        result = await _wait_for_enqueue(client)
        assert result[0].kind is WorkflowKind.refine

        # The workflow must carry the document image as initial input
        assert result[0].images is not None
        assert result[0].images.initial_image is not None

        # The initial_image content should match the document's pixel data:
        # fetch the same region that _prepare_workflow would have read (the full canvas)
        doc_wrapper = KritaDocument.active()
        assert doc_wrapper is not None
        expected = doc_wrapper.get_image(Bounds(0, 0, 512, 512))
        actual = result[0].images.initial_image

        assert actual.extent == expected.extent
        assert actual.pixel(0, 0) == expected.pixel(0, 0)
        assert actual.pixel(256, 256) == expected.pixel(256, 256)


@qtapp
async def test_generate_inpaint(workflows_dir: Path):
    """With an active selection and strength=1.0 the forwarded workflow should be
    WorkflowKind.inpaint; the initial_image and hires_mask must reflect the document
    image and the active selection respectively."""
    krita_doc = MockKritaDocument()
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
    """With batch_count=8 and a wildcard prompt, Model.generate should enqueue 8 separate jobs.
    Each job re-evaluates the wildcard with a different seed, so both options must appear."""
    krita_doc = MockKritaDocument()
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
    krita_doc = MockKritaDocument()
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
    krita_doc = MockKritaDocument()
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
    krita_doc = MockKritaDocument()
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
    krita_doc = MockKritaDocument()
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
    krita_doc = MockKritaDocument()
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
