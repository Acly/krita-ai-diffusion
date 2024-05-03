from pathlib import Path
import pytest
import subprocess
import os
import sys
import asyncio
import dotenv

from ai_diffusion.api import WorkflowInput, WorkflowKind, ControlInput, ImageInput, CheckpointInput
from ai_diffusion.api import SamplingInput, ConditioningInput, ExtentInput
from ai_diffusion.client import Client, ClientEvent
from ai_diffusion.cloud_client import CloudClient
from ai_diffusion.image import Extent, Image
from ai_diffusion.resources import ControlMode, SDVersion
from ai_diffusion.util import ensure
from .config import root_dir, test_dir, result_dir

dotenv.load_dotenv(root_dir / "service" / "web" / ".env.local")
pod_main = root_dir / "service" / "pod" / "pod.py"
run_dir = test_dir / "pod"


@pytest.fixture(scope="module")
def pod_server(qtapp, pytestconfig):
    async def serve(process: asyncio.subprocess.Process):
        try:
            async for line in ensure(process.stdout):
                print(line.decode("utf-8"), end="")
        except asyncio.CancelledError:
            process.terminate()
            await process.wait()

    async def start():
        env = os.environ.copy()
        args = ["-u", "-Xutf8", str(pod_main), "--rp_serve_api"]
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            *args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        async for line in ensure(process.stdout):
            text = line.decode("utf-8")
            print(text[:80], end="")
            if "Uvicorn running" in text:
                break

        return process, asyncio.create_task(serve(process))

    async def stop(process, task):
        process.terminate()
        task.cancel()
        await process.communicate()

    if not pytestconfig.getoption("--pod-process") or pytestconfig.getoption("--ci"):
        yield None  # For using local docker image or deployed serverless endpoint
    else:
        process, task = qtapp.run(start())
        yield process
        qtapp.run(stop(process, task))


async def receive_images(client: Client, work: WorkflowInput):
    job_id = None
    async for msg in client.listen():
        if job_id is None:
            job_id = await client.enqueue(work)
        if msg.event is ClientEvent.finished and msg.job_id == job_id:
            assert msg.images is not None
            return msg.images
        if msg.event is ClientEvent.error:
            raise Exception(msg.error)
    assert False, "Connection closed without receiving images"


@pytest.fixture()
def cloud_client(pytestconfig, qtapp, pod_server):
    if pytestconfig.getoption("--ci"):
        pytest.skip("Diffusion is disabled on CI")
    url = os.environ["TEST_SERVICE_URL"]
    token = os.environ["TEST_SERVICE_TOKEN"]
    return qtapp.run(CloudClient.connect(url, token))


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


def create_simple_workflow():
    return WorkflowInput(
        WorkflowKind.generate,
        images=ImageInput.from_extent(Extent(512, 512)),
        models=CheckpointInput("dreamshaper_8.safetensors"),
        sampling=SamplingInput("dpmpp_2m", "normal", cfg_scale=5.0, total_steps=20),
        conditioning=ConditioningInput("fluffy ball"),
        batch_count=2,
    )


def test_simple(qtapp, cloud_client):
    workflow = create_simple_workflow()
    run_and_save(qtapp, cloud_client, workflow, "pod_simple")


def test_large_image(qtapp, cloud_client):
    extent = Extent(3072, 2048)
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


@pytest.mark.parametrize("scenario", ["resolution", "steps", "control", "max_pixels"])
def test_validation(qtapp, cloud_client: CloudClient, scenario: str):
    workflow = create_simple_workflow()
    if scenario == "resolution":
        workflow.images = ImageInput.from_extent(Extent(9000, 512))
    elif scenario == "steps":
        ensure(workflow.sampling).total_steps = 200
    elif scenario == "control":
        img = Image.create(Extent(4, 4))
        control = ensure(workflow.conditioning).control
        for i in range(7):
            control.append(ControlInput(ControlMode.depth, img))
    elif scenario == "max_pixels":
        workflow.images = ImageInput.from_extent(Extent(3840, 2168))  # > 4k

    with pytest.raises(Exception, match="Validation error"):
        run_and_save(qtapp, cloud_client, workflow, "pod_validation")


cost_params = {
    "sd15-live-512x512": (SDVersion.sd15, 1, 512, 512, 1),
    "sd15-8x512x512": (SDVersion.sd15, 8, 512, 512, 20),
    "sd15-2x1024x1024": (SDVersion.sd15, 2, 1024, 1024, 20),
    "sd15-1x1024x2048": (SDVersion.sd15, 1, 1024, 2048, 20),
    "sdxl-2x1024x1024": (SDVersion.sdxl, 2, 1024, 1024, 20),
    "sdxl-highstep": (SDVersion.sdxl, 1, 1536, 1024, 50),
    "sdxl-refine": (SDVersion.sdxl, 1, 1536, 1024, 20),
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

    async def check():
        service_cost = await cloud_client.compute_cost(
            input.kind, sdversion, batch_count, extent, ensure(input.sampling).actual_steps
        )
        assert service_cost == input.cost

    qtapp.run(check())
