from pathlib import Path
import pytest
import subprocess
import os
import sys
import asyncio
import dotenv

from ai_diffusion.api import WorkflowInput, WorkflowKind, ControlInput, ImageInput, CheckpointInput
from ai_diffusion.api import SamplingInput, ConditioningInput, ExtentInput, RegionInput
from ai_diffusion.client import Client, ClientEvent
from ai_diffusion.cloud_client import CloudClient, enumerate_features, apply_limits
from ai_diffusion.image import Extent, Image, Bounds
from ai_diffusion.resources import ControlMode, Arch
from ai_diffusion.util import ensure
from .conftest import has_local_cloud
from .config import root_dir, test_dir, result_dir

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
    if not has_local_cloud:
        pytest.skip("Local cloud service not found")
    dotenv.load_dotenv(root_dir / "service" / "web" / ".env.local")
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
