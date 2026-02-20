import asyncio
from pathlib import Path
from timeit import default_timer as timer

import aiohttp
import pytest

from ai_diffusion.api import (
    CheckpointInput,
    ConditioningInput,
    ControlInput,
    ExtentInput,
    ImageInput,
    RegionInput,
    SamplingInput,
    WorkflowInput,
    WorkflowKind,
)
from ai_diffusion.client import Client, ClientEvent
from ai_diffusion.cloud_client import CloudClient, apply_limits, enumerate_features
from ai_diffusion.image import Bounds, Extent, Image, ImageCollection
from ai_diffusion.resources import Arch, ControlMode
from ai_diffusion.util import ensure

from .config import result_dir, test_dir
from .conftest import CloudService


async def receive_images(client: Client, work: WorkflowInput | list[WorkflowInput]):
    job_id = None
    images = ImageCollection()
    if not isinstance(work, list):
        work = [work]
    async for msg in client.listen():
        if job_id is None:
            job_id = [await client.enqueue(w) for w in work]
        if msg.event is ClientEvent.finished and msg.job_id in job_id:
            assert msg.images is not None
            images.append(msg.images)
            job_id.remove(msg.job_id)
            if len(job_id) == 0:
                await client.disconnect()
        if msg.event is ClientEvent.error:
            raise RuntimeError(msg.error)
    return images


async def connect_cloud(service: CloudService):
    user = await service.create_user("workflow-tester")
    return await CloudClient.connect(service.url, user["token"])


@pytest.fixture(scope="module")
def cloud_client(pytestconfig, qtapp, cloud_service: CloudService):
    if pytestconfig.getoption("--ci"):
        pytest.skip("Diffusion is disabled on CI")
    if not cloud_service.enabled:
        pytest.skip("Cloud service not running")

    client = qtapp.run(connect_cloud(cloud_service))
    yield client
    qtapp.run(client.disconnect())


def run_and_save(
    qtapp,
    client: Client,
    work: WorkflowInput,
    filename: str,
    output_dir: Path = result_dir,
):
    async def runner():
        return await receive_images(client, work)

    results = qtapp.run(runner())
    for i, img in enumerate(results):
        img.save(output_dir / f"{filename}_{i}.png")
    return results[0]


def create_simple_workflow(prompt="fluffy ball", input: Image | Extent | None = None):
    start = 0
    if isinstance(input, Image):
        images = ImageInput.from_extent(input.extent)
        images.initial_image = input
        images.hires_image = input
        start = 4
    elif isinstance(input, Extent):
        images = ImageInput.from_extent(input)
    else:
        images = ImageInput.from_extent(Extent(512, 512))

    return WorkflowInput(
        WorkflowKind.generate if images.initial_image is None else WorkflowKind.refine,
        images=images,
        models=CheckpointInput("dreamshaper_8.safetensors"),
        sampling=SamplingInput(
            "dpmpp_2m", "normal", cfg_scale=5.0, total_steps=20, start_step=start
        ),
        conditioning=ConditioningInput(prompt),
        batch_count=2,
    )


def test_simple(qtapp, cloud_client):
    workflow = create_simple_workflow()
    run_and_save(qtapp, cloud_client, workflow, "pod_simple")


def test_large_image(qtapp, cloud_client):
    extent = Extent(2304, 1536)
    input_image = Image.load(test_dir / "images" / "beach_1536x1024.webp")
    input_image = Image.scale(input_image, extent)
    workflow = WorkflowInput(
        WorkflowKind.refine,
        images=ImageInput(ExtentInput(extent, extent, extent, extent), input_image),
        models=CheckpointInput("dreamshaper_8.safetensors"),
        sampling=SamplingInput("dpmpp_2m", "normal", cfg_scale=3.0, total_steps=10, start_step=4),
        conditioning=ConditioningInput(
            "beach, jungle", control=[ControlInput(ControlMode.blur, input_image)]
        ),
    )
    run_and_save(qtapp, cloud_client, workflow, "pod_large_image")


@pytest.mark.parametrize("scenario", ["resolution", "steps", "max_pixels"])
def test_validation(qtapp, cloud_client: CloudClient, scenario: str):
    workflow = create_simple_workflow()
    if scenario == "resolution":
        workflow.images = ImageInput.from_extent(Extent(19000, 512))
    elif scenario == "steps":
        ensure(workflow.sampling).total_steps = 200
    elif scenario == "max_pixels":
        workflow.images = ImageInput.from_extent(Extent(3840, 2168))  # > 4k

    expected = "Image size" if scenario == "resolution" else "Validation error"
    with pytest.raises(Exception, match=expected):
        run_and_save(qtapp, cloud_client, workflow, "pod_validation")


cost_params = {
    "sd15-live-512x512": (Arch.sd15, 1, 512, 512, 1),
    "sd15-8x512x512": (Arch.sd15, 8, 512, 512, 20),
    "sd15-2x1024x1024": (Arch.sd15, 2, 1024, 1024, 20),
    "sd15-1x1024x2048": (Arch.sd15, 1, 1024, 2048, 20),
    "sdxl-2x1024x1024": (Arch.sdxl, 2, 1024, 1024, 20),
    "sdxl-highstep": (Arch.sdxl, 1, 1536, 1024, 50),
    "sdxl-refine": (Arch.sdxl, 1, 1536, 1024, 20),
    "inpaint-initial": (Arch.sdxl, 2, 1024, 1024, 24),
    "inpaint-crop": (Arch.sd15, 2, 512, 512, 24),
    "upscale-tiled": (Arch.sd15, 1, 512, 512, 10),
    "upscale-tiled-2": (Arch.sd15, 1, 320, 640, 10),
    "upscaled-invalid": (Arch.sd15, 1, 512, 512, 10),
    "illustrious": (Arch.illu, 2, 1024, 1024, 20),
    "illustrious-v": (Arch.illu_v, 2, 1024, 1024, 20),
    "flux2": (Arch.flux2_4b, 2, 1024, 1024, 4),
    "zimage": (Arch.zimage, 2, 1024, 1024, 9),
}


@pytest.mark.parametrize("params", cost_params.keys())
def test_compute_cost(qtapp, cloud_client: CloudClient, params):
    sdversion, batch_count, width, height, steps = cost_params[params]
    extent = Extent(width, height)
    input = WorkflowInput(
        WorkflowKind.generate,
        images=ImageInput.from_extent(extent),
        models=CheckpointInput("ckpt", sdversion),
        sampling=SamplingInput("dpmpp_2m", "normal", cfg_scale=5.0, total_steps=steps),
        batch_count=batch_count,
    )
    if params == "sdxl-refine":
        input.kind = WorkflowKind.refine
        ensure(input.sampling).start_step = 16
    elif params == "inpaint-initial":
        input.kind = WorkflowKind.inpaint
        input.crop_upscale_extent = Extent(640, 768)
    elif params == "inpaint-crop":
        input.kind = WorkflowKind.inpaint
        input.crop_upscale_extent = Extent(1024, 768)
    elif params == "upscale-tiled":
        input.kind = WorkflowKind.upscale_tiled
        input.extent.target = Extent(1024, 1024)
    elif params == "upscale-tiled-2":
        input.kind = WorkflowKind.upscale_tiled
        input.extent.target = Extent(2000, 1600)
    elif params == "upscaled-invalid":
        input.kind = WorkflowKind.upscale_tiled
        input.extent.target = Extent(200, 200)

    async def check():
        service_cost = await cloud_client.compute_cost(input)
        assert service_cost == input.cost

    qtapp.run(check())


def test_features_limits():
    features = enumerate_features({"max_control_layers": 2})
    image = Image.create(Extent(4, 4))
    control_layers = [
        ControlInput(ControlMode.blur, image),
        ControlInput(ControlMode.line_art, image),
        ControlInput(ControlMode.depth, image),
    ]
    work = WorkflowInput(
        WorkflowKind.generate,
        models=CheckpointInput("ckpt", self_attention_guidance=True),
        conditioning=ConditioningInput(
            "prompt",
            control=control_layers,
            regions=[RegionInput(image, Bounds(0, 0, 4, 4), "positive", control_layers)],
        ),
    )
    apply_limits(work, features)
    assert work.conditioning and len(work.conditioning.control) == 2
    assert work.conditioning and len(work.conditioning.regions[0].control) == 2
    assert work.models and work.models.self_attention_guidance is False


def test_multiple_jobs(pytestconfig, qtapp, cloud_service: CloudService):
    if not pytestconfig.getoption("--benchmark"):
        pytest.skip("Only runs with --benchmark")
    if not cloud_service.enabled:
        pytest.skip("Cloud service not running")

    async def create_client(i: int):
        user = await cloud_service.create_user(f"multi-job-tester-{i}")
        return await CloudClient.connect(cloud_service.url, user["token"])

    input_image = Image.load(test_dir / "images" / "flowers.webp")
    input_image = Image.scale(input_image, Extent(512, 512))
    prompts = [
        "potted flowers, red petals, sunlight",
        "potted flowers, blue petals, dawn",
        "potted flowers, yellow petals, sunlight",
        "potted flowers, purple petals, night",
        "potted flowers, orange petals, sunset",
    ]

    async def run_job(client: CloudClient, index: int):
        workflow = create_simple_workflow(prompt=prompts[index], input=input_image)
        images = await receive_images(client, [workflow, workflow])
        for i, result in enumerate(images):
            filename = result_dir / f"cloud_multi_user{index}_image{i}.png"
            result.save(filename)

    async def main():
        clients = await asyncio.gather(*(create_client(i) for i in range(5)))
        await asyncio.gather(*(run_job(client, i) for i, client in enumerate(clients)))

    start_time = timer()

    qtapp.run(main())

    end_time = timer()
    duration = end_time - start_time
    print(f"Completed 5 x 2 jobs in {duration:.2f} seconds", end=" ")


def test_error_workflow(qtapp, cloud_client: CloudClient):
    workflow = create_simple_workflow()
    workflow.kind = WorkflowKind.refine  # Error: refine requires an input image
    with pytest.raises(Exception, match="failed"):
        run_and_save(qtapp, cloud_client, workflow, "error_workflow")


async def _reset_worker_config(cloud_service: CloudService):
    for _attempt in range(5):
        try:
            await cloud_service.update_worker_config()
            break
        except aiohttp.ClientConnectionError:
            await asyncio.sleep(2)  # Wait for worker to be back up


def test_timeout(pytestconfig, qtapp, cloud_service: CloudService):
    if not pytestconfig.getoption("--benchmark"):
        pytest.skip("Only runs with --benchmark")
    if not cloud_service.enabled:
        pytest.skip("Cloud service not running")

    async def main():
        user = await cloud_service.create_user("timeout-tester")
        client = await CloudClient.connect(cloud_service.url, user["token"])
        big_workflow = create_simple_workflow(input=Extent(2048, 1536))

        try:
            await cloud_service.update_worker_config({"job_timeout": 5})

            with pytest.raises(Exception, match="timeout"):
                await receive_images(client, big_workflow)
        finally:
            await _reset_worker_config(cloud_service)

        # Worker should be restarted and accept new jobs
        small_workflow = create_simple_workflow()
        images = await receive_images(client, small_workflow)
        assert len(images) == 2

    qtapp.run(main())


@pytest.mark.parametrize("scenario", ["max_uptime", "max_memory"])
def test_restart(pytestconfig, qtapp, cloud_service: CloudService, scenario: str):
    if not pytestconfig.getoption("--benchmark"):
        pytest.skip("Only runs with --benchmark")
    if not cloud_service.enabled:
        pytest.skip("Cloud service not running")

    async def main():
        user = await cloud_service.create_user("restart-tester")
        client = await CloudClient.connect(cloud_service.url, user["token"])
        workflow = create_simple_workflow()

        try:
            if scenario == "max_uptime":
                await cloud_service.update_worker_config({"max_uptime": 2})
            elif scenario == "max_memory":
                await cloud_service.update_worker_config({"max_memory_usage": 0.1})

            images = await receive_images(client, workflow)
            assert len(images) == 2
        finally:
            await _reset_worker_config(cloud_service)

    qtapp.run(main())
