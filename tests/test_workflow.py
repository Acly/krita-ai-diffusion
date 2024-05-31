import itertools
import pytest
import dotenv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_diffusion import workflow
from ai_diffusion.api import LoraInput, WorkflowKind, WorkflowInput, ControlInput
from ai_diffusion.api import InpaintMode, FillMode, ConditioningInput
from ai_diffusion.client import ClientModels, CheckpointInfo
from ai_diffusion.comfy_client import ComfyClient
from ai_diffusion.cloud_client import CloudClient
from ai_diffusion.resources import ControlMode
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.image import Mask, Bounds, Extent, Image, ImageCollection
from ai_diffusion.client import Client, ClientEvent
from ai_diffusion.style import SDVersion, Style
from ai_diffusion.pose import Pose
from ai_diffusion.workflow import detect_inpaint
from . import config
from .config import root_dir, image_dir, result_dir, reference_dir, default_checkpoint

service_available = (root_dir / "service" / "web" / ".env.local").exists()
client_params = ["local", "cloud"] if service_available else ["local"]


async def connect_cloud():
    dotenv.load_dotenv(root_dir / "service" / "web" / ".env.local")
    url = os.environ["TEST_SERVICE_URL"]
    token = os.environ.get("TEST_SERVICE_TOKEN", "")
    if not token:
        client = CloudClient(url)
        sign_in = client.sign_in()
        auth_url = await anext(sign_in)
        print("\nSign-in required:", auth_url)
        token = await anext(sign_in)
        print("\nToken received:", token, "\n")
    return await CloudClient.connect(url, token)


@pytest.fixture(params=client_params, scope="module")
def client(pytestconfig, request, qtapp):
    if pytestconfig.getoption("--ci"):
        pytest.skip("Diffusion is disabled on CI")
    if request.param == "local":
        return qtapp.run(ComfyClient.connect())
    else:
        return qtapp.run(connect_cloud())


default_seed = 1234
default_perf = PerformanceSettings(batch_size=1)


def default_style(client: Client, sd_ver=SDVersion.sd15):
    version_checkpoints = [c for c in client.models.checkpoints if sd_ver.matches(c)]
    checkpoint = default_checkpoint[sd_ver]

    style = Style(Path("default.json"))
    style.sd_checkpoint = (
        checkpoint if checkpoint in version_checkpoints else version_checkpoints[0]
    )
    return style


def create(kind: WorkflowKind, client: Client, **kwargs):
    kwargs.setdefault("cond", ConditioningInput(""))
    kwargs.setdefault("style", default_style(client))
    kwargs.setdefault("seed", default_seed)
    kwargs.setdefault("perf", default_perf)
    return workflow.prepare(kind, models=client.models, **kwargs)


async def receive_images(client: Client, work: WorkflowInput):
    job_id = None
    async for msg in client.listen():
        if not job_id:
            job_id = await client.enqueue(work)
        if msg.event is ClientEvent.finished and msg.job_id == job_id:
            assert msg.images is not None
            return msg.images
        if msg.event is ClientEvent.error and msg.job_id == job_id:
            raise Exception(msg.error)
    assert False, "Connection closed without receiving images"


def run_and_save(
    qtapp,
    client: Client,
    work: WorkflowInput,
    filename: str,
    composition_image: Image | None = None,
    composition_mask: Mask | None = None,
    output_dir: Path = result_dir,
):
    dump_workflow(work, filename, client)

    async def runner():
        return await receive_images(client, work)

    results: ImageCollection = qtapp.run(runner())
    assert len(results) == 1
    client_name = "local" if isinstance(client, ComfyClient) else "cloud"
    if composition_image and composition_mask:
        composition_image.draw_image(results[0], composition_mask.bounds.offset)
        composition_image.save(output_dir / f"{filename}_{client_name}.png")
    else:
        results[0].save(output_dir / f"{filename}_{client_name}.png")
    return results[0]


def dump_workflow(work: WorkflowInput, filename: str, client: Client):
    flow = workflow.create(work, client.models)
    flow.dump((result_dir / "workflows" / filename).with_suffix(".json"))


def automatic_inpaint(
    image_extent: Extent,
    bounds: Bounds,
    sd_ver: SDVersion = SDVersion.sd15,
    prompt: str = "",
    control: list[ControlInput] = [],
):
    mode = workflow.detect_inpaint_mode(image_extent, bounds)
    return detect_inpaint(mode, bounds, sd_ver, prompt, control, strength=1.0)


def test_inpaint_params():
    bounds = Bounds(0, 0, 100, 100)

    a = detect_inpaint(InpaintMode.fill, bounds, SDVersion.sd15, "", [], 1.0)
    assert a.fill is FillMode.blur and a.use_inpaint_model == True and a.use_reference == True

    b = detect_inpaint(InpaintMode.add_object, bounds, SDVersion.sd15, "", [], 1.0)
    assert b.fill is FillMode.neutral and b.use_condition_mask == False

    c = detect_inpaint(InpaintMode.replace_background, bounds, SDVersion.sdxl, "", [], 1.0)
    assert c.fill is FillMode.replace and c.use_inpaint_model == True and c.use_reference == False

    d = detect_inpaint(InpaintMode.add_object, bounds, SDVersion.sd15, "prompt", [], 1.0)
    assert d.use_condition_mask == True

    control = [ControlInput(ControlMode.line_art, Image.create(Extent(4, 4)))]
    e = detect_inpaint(InpaintMode.add_object, bounds, SDVersion.sd15, "prompt", control, 1.0)
    assert e.use_condition_mask == False


def test_prepare_lora():
    models = ClientModels()
    models.checkpoints = {"CP": CheckpointInfo("CP", SDVersion.sd15)}
    models.loras = ["PINK_UNICORNS", "MOTHER_OF_PEARL"]
    style = Style(Path("default.json"))
    style.sd_checkpoint = "CP"
    style.loras.append(dict(name="MOTHER_OF_PEARL", strength=0.33))
    job = workflow.prepare(
        WorkflowKind.generate,
        canvas=Extent(512, 512),
        cond=ConditioningInput("test <lora:PINK_UNICORNS:0.77>"),
        style=style,
        seed=29,
        models=models,
        perf=default_perf,
    )
    assert job.conditioning and job.conditioning.positive == "test"
    assert (
        job.models
        and LoraInput("PINK_UNICORNS", 0.77) in job.models.loras
        and LoraInput("MOTHER_OF_PEARL", 0.33) in job.models.loras
    )


@pytest.mark.parametrize("extent", [Extent(256, 256), Extent(800, 800), Extent(512, 1024)])
def test_generate(qtapp, client, extent: Extent):
    prompt = ConditioningInput("ship")
    job = create(WorkflowKind.generate, client, canvas=extent, cond=prompt)
    result = run_and_save(qtapp, client, job, f"test_generate_{extent.width}x{extent.height}")
    assert result.extent == extent


def test_inpaint(qtapp, client):
    image = Image.load(image_dir / "beach_768x512.webp")
    mask = Mask.rectangle(Bounds(40, 120, 320, 200), feather=10)
    cond = ConditioningInput("beach, the sea, cliffs, palm trees")
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        style=default_style(client, SDVersion.sd15),
        cond=cond,
        perf=PerformanceSettings(batch_size=3),  # max 3 images@512x512 -> 2 images@768x512
        inpaint=detect_inpaint(
            InpaintMode.fill, mask.bounds, SDVersion.sd15, cond.positive, [], 1.0
        ),
    )

    async def main():
        results = await receive_images(client, job)
        assert len(results) == 2
        client_name = "local" if isinstance(client, ComfyClient) else "cloud"
        for i, result in enumerate(results):
            image.draw_image(result, mask.bounds.offset)
            image.save(result_dir / f"test_inpaint_{i}_{client_name}.png")
            assert result.extent == Extent(320, 200)

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_inpaint_upscale(qtapp, client, sdver):
    image = Image.load(image_dir / "beach_1536x1024.webp")
    mask = Mask.rectangle(Bounds(300, 200, 768, 512), feather=20)
    prompt = ConditioningInput("ship")
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        style=default_style(client, sdver),
        cond=prompt,
        perf=PerformanceSettings(batch_size=3),  # 2 images for 1.5, 1 image for XL
        inpaint=detect_inpaint(
            InpaintMode.add_object, mask.bounds, sdver, prompt.positive, [], 1.0
        ),
    )

    async def main():
        dump_workflow(job, f"test_inpaint_upscale_{sdver.name}.json", client)
        results = await receive_images(client, job)
        assert len(results) == 2 if sdver == SDVersion.sd15 else 1
        client_name = "local" if isinstance(client, ComfyClient) else "cloud"
        for i, result in enumerate(results):
            image.draw_image(result, mask.bounds.offset)
            image.save(result_dir / f"test_inpaint_upscale_{sdver.name}_{i}_{client_name}.png")
            assert result.extent == mask.bounds.extent

    qtapp.run(main())


def test_inpaint_odd_resolution(qtapp, client):
    image = Image.load(image_dir / "beach_768x512.webp")
    image = Image.scale(image, Extent(612, 513))
    mask = Mask.rectangle(Bounds(0, 0, 200, 513))
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        inpaint=automatic_inpaint(image.extent, mask.bounds),
    )
    result = run_and_save(qtapp, client, job, "test_inpaint_odd_resolution", image, mask)
    assert result.extent == mask.bounds.extent


def test_inpaint_area_conditioning(qtapp, client):
    image = Image.load(image_dir / "lake_1536x1024.webp")
    mask = Mask.load(image_dir / "lake_1536x1024_mask_bottom_right.png")
    prompt = ConditioningInput("(crocodile)")
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        cond=prompt,
        inpaint=detect_inpaint(
            InpaintMode.add_object, mask.bounds, SDVersion.sd15, prompt.positive, [], 1.0
        ),
    )
    run_and_save(qtapp, client, job, "test_inpaint_area_conditioning", image, mask)


def test_inpaint_remove_object(qtapp, client):
    image = Image.load(image_dir / "owls_inpaint.webp")
    mask = Mask.load(image_dir / "owls_mask_remove.webp")
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        cond=ConditioningInput("tree branch"),
        inpaint=detect_inpaint(
            InpaintMode.remove_object, mask.bounds, SDVersion.sd15, "tree", [], 1.0
        ),
    )
    run_and_save(qtapp, client, job, "test_inpaint_remove_object", image, mask)


@pytest.mark.parametrize("setup", ["sd15", "sdxl"])
def test_refine(qtapp, client, setup):
    sdver, extent, strength = {
        "sd15": (SDVersion.sd15, Extent(768, 508), 0.5),
        "sdxl": (SDVersion.sdxl, Extent(1111, 741), 0.65),
    }[setup]
    image = Image.load(image_dir / "beach_1536x1024.webp")
    image = Image.scale(image, extent)
    job = create(
        WorkflowKind.refine,
        client,
        canvas=image,
        style=default_style(client, sdver),
        cond=ConditioningInput("painting in the style of Vincent van Gogh"),
        strength=strength,
        perf=PerformanceSettings(batch_size=1, max_pixel_count=2),
    )
    result = run_and_save(qtapp, client, job, f"test_refine_{setup}")
    assert result.extent == extent


@pytest.mark.parametrize("setup", ["sd15_0.4", "sd15_0.6", "sdxl_0.7"])
def test_refine_region(qtapp, client, setup):
    sdver, strength = {
        "sd15_0.4": (SDVersion.sd15, 0.4),
        "sd15_0.6": (SDVersion.sd15, 0.6),
        "sdxl_0.7": (SDVersion.sdxl, 0.7),
    }[setup]
    image = Image.load(image_dir / "lake_region.webp")
    mask = Mask.load(image_dir / "lake_region_mask.png")
    prompt = ConditioningInput("waterfall")
    params = detect_inpaint(
        InpaintMode.fill, mask.bounds, SDVersion.sd15, prompt.positive, [], strength
    )
    job = create(
        WorkflowKind.refine_region,
        client,
        canvas=image,
        mask=mask,
        style=default_style(client, sdver),
        cond=prompt,
        strength=strength,
        inpaint=params,
    )
    result = run_and_save(qtapp, client, job, f"test_refine_region_{setup}", image, mask)
    assert result.extent == mask.bounds.extent


def test_differential_diffusion(qtapp, client):
    image = Image.scale(Image.load(image_dir / "beach_1536x1024.webp"), Extent(768, 512))
    mask = Mask.load(image_dir / "differential_diffusion_mask.webp")
    prompt = ConditioningInput("barren plain, volcanic wasteland, burned trees")
    params = detect_inpaint(InpaintMode.fill, mask.bounds, SDVersion.sd15, prompt.positive, [], 0.9)
    job = create(
        WorkflowKind.refine_region,
        client,
        canvas=image,
        mask=mask,
        style=default_style(client, SDVersion.sd15),
        cond=prompt,
        strength=0.9,
        inpaint=params,
    )
    run_and_save(qtapp, client, job, "test_differential_diffusion", image, mask)


@pytest.mark.parametrize(
    "op", ["generate", "inpaint", "refine", "refine_region", "inpaint_upscale"]
)
def test_control_scribble(qtapp, client, op):
    scribble_image = Image.load(image_dir / "owls_scribble.webp")
    inpaint_image = Image.load(image_dir / "owls_inpaint.webp")
    mask = Mask.load(image_dir / "owls_mask.png")
    mask.bounds = Bounds(256, 0, 256, 512)
    control = [ControlInput(ControlMode.scribble, scribble_image)]
    prompt = ConditioningInput("owls", control=control)

    args: dict[str, Any]
    if op == "generate":
        args = dict(kind=WorkflowKind.generate, canvas=Extent(512, 512))
    elif op == "inpaint":
        params = automatic_inpaint(inpaint_image.extent, mask.bounds)
        args = dict(kind=WorkflowKind.inpaint, canvas=inpaint_image, mask=mask, inpaint=params)
    elif op == "refine":
        args = dict(kind=WorkflowKind.refine, canvas=inpaint_image, strength=0.7)
    elif op == "refine_region":
        kind = WorkflowKind.refine_region
        crop_image = Image.crop(inpaint_image, mask.bounds)
        control[0].image = Image.crop(scribble_image, mask.bounds)
        crop_mask = Mask(Bounds(0, 0, 256, 512), mask.image)
        params = params = automatic_inpaint(crop_image.extent, crop_mask.bounds)
        args = dict(kind=kind, canvas=crop_image, mask=crop_mask, strength=0.7, inpaint=params)
    else:  # op == "inpaint_upscale":
        control[0].image = Image.scale(scribble_image, Extent(1024, 1024))
        inpaint_image = Image.scale(inpaint_image, Extent(1024, 1024))
        scaled_mask = Image.scale(Image(mask.image), Extent(512, 1024))
        mask = Mask(Bounds(512, 0, 512, 1024), scaled_mask._qimage)
        params = detect_inpaint(InpaintMode.fill, mask.bounds, SDVersion.sd15, "owls", control, 1.0)
        args = dict(kind=WorkflowKind.inpaint, canvas=inpaint_image, mask=mask, inpaint=params)

    job = create(client=client, cond=prompt, **args)
    if op in ["inpaint", "refine_region", "inpaint_upscale"]:
        run_and_save(qtapp, client, job, f"test_control_scribble_{op}", inpaint_image, mask)
    else:
        run_and_save(qtapp, client, job, f"test_control_scribble_{op}")


def test_control_canny_downscale(qtapp, client):
    canny_image = Image.load(image_dir / "shrine_canny.webp")
    control = [ControlInput(ControlMode.canny_edge, canny_image, 1.0)]
    prompt = ConditioningInput("shrine", control=control)
    job = create(WorkflowKind.generate, client, canvas=Extent(999, 999), cond=prompt)
    run_and_save(qtapp, client, job, "test_control_canny_downscale")


@pytest.mark.parametrize("mode", [m for m in ControlMode if m.has_preprocessor])
def test_create_control_image(qtapp, client: Client, mode):
    skip_cloud_modes = [ControlMode.normal, ControlMode.segmentation, ControlMode.hands]
    if isinstance(client, CloudClient) and mode in skip_cloud_modes:
        pytest.skip("Control image preproccessor not available")

    image_name = f"test_create_control_image_{mode.name}"
    image = Image.load(image_dir / "adobe_stock.jpg")
    job = workflow.prepare_create_control_image(image, mode, default_perf)

    result = run_and_save(qtapp, client, job, image_name)
    if isinstance(client, ComfyClient):
        reference = Image.load(reference_dir / image_name)
        threshold = 0.015 if mode is ControlMode.pose else 0.002
        assert Image.compare(result, reference) < threshold
        # cloud results are a bit different, maybe due to compression of input?


def test_create_open_pose_vector(qtapp, client: Client):
    image_name = f"test_create_open_pose_vector.svg"
    image = Image.load(image_dir / "adobe_stock.jpg")
    job = workflow.prepare_create_control_image(image, ControlMode.pose, default_perf)

    async def main():
        job_id = None
        async for msg in client.listen():
            if not job_id:
                job_id = await client.enqueue(job)
            if msg.event is ClientEvent.finished and msg.job_id == job_id:
                assert msg.result is not None
                result = Pose.from_open_pose_json(msg.result).to_svg()
                (result_dir / image_name).write_text(result)
                return
            if msg.event is ClientEvent.error and msg.job_id == job_id:
                raise Exception(msg.error)
        assert False, "Connection closed without receiving images"

    qtapp.run(main())


@pytest.mark.parametrize("setup", ["no_mask", "right_hand", "left_hand"])
def test_create_hand_refiner_image(qtapp, client: Client, setup):
    if isinstance(client, CloudClient):
        pytest.skip("Hand refiner is not available in the cloud")
    image_name = f"test_create_hand_refiner_image_{setup}"
    image = Image.load(image_dir / "character.webp")
    bounds = {
        "no_mask": None,
        "right_hand": Bounds(102, 398, 264, 240),
        "left_hand": Bounds(541, 642, 232, 248),
    }[setup]
    job = workflow.prepare_create_control_image(
        image, ControlMode.hands, default_perf, bounds, default_seed
    )
    result = run_and_save(qtapp, client, job, image_name)
    reference = Image.load(reference_dir / image_name)
    assert Image.compare(result, reference) < 0.002


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_ip_adapter(qtapp, client, sdver):
    image = Image.load(image_dir / "cat.webp")
    prompt = ConditioningInput("cat on a rooftop in paris")
    prompt.control = [ControlInput(ControlMode.reference, image, 0.6)]
    extent = Extent(512, 512) if sdver == SDVersion.sd15 else Extent(1024, 1024)
    style = default_style(client, sdver)
    job = create(WorkflowKind.generate, client, style=style, canvas=extent, cond=prompt)
    run_and_save(qtapp, client, job, f"test_ip_adapter_{sdver.name}")


def test_ip_adapter_region(qtapp, client):
    image = Image.load(image_dir / "flowers.webp")
    mask = Mask.load(image_dir / "flowers_mask.png")
    control_img = Image.load(image_dir / "pegonia.webp")
    prompt = ConditioningInput("potted flowers")
    prompt.control = [ControlInput(ControlMode.reference, control_img, 0.7)]
    inpaint = automatic_inpaint(image.extent, mask.bounds, SDVersion.sd15, prompt.positive)
    job = create(
        WorkflowKind.refine_region,
        client,
        canvas=image,
        mask=mask,
        inpaint=inpaint,
        cond=prompt,
        strength=0.6,
    )
    run_and_save(qtapp, client, job, "test_ip_adapter_region", image, mask)


def test_ip_adapter_batch(qtapp, client):
    image1 = Image.load(image_dir / "cat.webp")
    image2 = Image.load(image_dir / "pegonia.webp")
    control = [
        ControlInput(ControlMode.reference, image1, 1.0),
        ControlInput(ControlMode.reference, image2, 1.0),
    ]
    cond = ConditioningInput("", control=control)
    job = create(WorkflowKind.generate, client, canvas=Extent(512, 512), cond=cond)
    run_and_save(qtapp, client, job, "test_ip_adapter_batch")


def test_style_composition(qtapp, client):
    style_image = Image.load(image_dir / "watercolor.webp")
    composition_image = Image.load(image_dir / "flowers.webp")
    control = [
        ControlInput(ControlMode.style, style_image, 1.0),
        ControlInput(ControlMode.composition, composition_image, 0.8),
    ]
    job = create(
        WorkflowKind.generate,
        client,
        canvas=Extent(1024, 1024),
        cond=ConditioningInput("", control=control),
        style=default_style(client, SDVersion.sdxl),
    )
    run_and_save(qtapp, client, job, "test_style_composition")


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_ip_adapter_face(qtapp, client, sdver):
    if isinstance(client, CloudClient):
        pytest.skip("IP-adapter FaceID is not available in the cloud")
    extent = Extent(650, 650) if sdver == SDVersion.sd15 else Extent(1024, 1024)
    image = Image.load(image_dir / "face.webp")
    cond = ConditioningInput("portrait photo of a woman at a garden party")
    cond.control = [ControlInput(ControlMode.face, image, 0.9)]
    job = create(WorkflowKind.generate, client, canvas=extent, cond=cond)
    run_and_save(qtapp, client, job, f"test_ip_adapter_face_{sdver.name}")


def test_upscale_simple(qtapp, client: Client):
    image = Image.load(image_dir / "beach_768x512.webp")
    job = workflow.prepare_upscale_simple(image, client.models.default_upscaler, 2.0)
    run_and_save(qtapp, client, job, "test_upscale_simple")


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_upscale_tiled(qtapp, client: Client, sdver):
    image = Image.load(image_dir / "beach_768x512.webp")
    job = create(
        WorkflowKind.upscale_tiled,
        client,
        canvas=image,
        upscale_model=client.models.default_upscaler,
        upscale_factor=2.0,
        style=default_style(client, sdver),
        cond=ConditioningInput("4k uhd"),
        strength=0.5,
    )
    run_and_save(qtapp, client, job, f"test_upscale_tiled_{sdver.name}")


def test_generate_live(qtapp, client):
    scribble = Image.load(image_dir / "owls_scribble.webp")
    job = create(
        WorkflowKind.generate,
        client,
        canvas=Extent(512, 512),
        cond=ConditioningInput("owls", control=[ControlInput(ControlMode.scribble, scribble)]),
        is_live=True,
    )
    run_and_save(qtapp, client, job, "test_generate_live")


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_refine_live(qtapp, client, sdver):
    image = Image.load(image_dir / "pegonia.webp")
    if sdver is SDVersion.sdxl:
        image = Image.scale(image, Extent(1024, 1024))  # result will be a bit blurry
    job = create(
        WorkflowKind.refine,
        client,
        style=default_style(client, sdver),
        canvas=image,
        cond=ConditioningInput(""),
        strength=0.4,
        is_live=True,
    )
    run_and_save(qtapp, client, job, f"test_refine_live_{sdver.name}")


def test_refine_max_pixels(qtapp, client):
    perf_settings = PerformanceSettings(max_pixel_count=1)  # million pixels
    image = Image.load(image_dir / "lake_1536x1024.webp")
    cond = ConditioningInput("watercolor painting on structured paper, aquarelle, stylized")
    job = create(
        WorkflowKind.refine, client, canvas=image, cond=cond, strength=0.6, perf=perf_settings
    )
    run_and_save(qtapp, client, job, f"test_refine_max_pixels")


def test_outpaint_resolution_multiplier(qtapp, client):
    perf_settings = PerformanceSettings(batch_size=1, resolution_multiplier=0.8)
    image = Image.create(Extent(2048, 1024))
    beach = Image.load(image_dir / "beach_1536x1024.webp")
    image.draw_image(beach, (512, 0))
    mask = Mask.load(image_dir / "beach_outpaint_mask.png")
    prompt = ConditioningInput("photo of a beach and jungle, nature photography, tropical")
    params = automatic_inpaint(image.extent, mask.bounds, prompt=prompt.positive)
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        cond=prompt,
        inpaint=params,
        perf=perf_settings,
    )
    run_and_save(qtapp, client, job, f"test_outpaint_resolution_multiplier", image, mask)


def test_refine_tiled(qtapp, client: Client):
    image = Image.load(image_dir / "character.webp")
    job = create(
        WorkflowKind.refine_tiled,
        client,
        canvas=image,
        upscale_model=client.models.default_upscaler,
        style=default_style(client, SDVersion.sd15),
        cond=ConditioningInput("4k uhd"),
        upscale_factor=2.5,
        strength=0.5,
    )
    run_and_save(qtapp, client, job, "test_refine_tiled")


def test_refine_tiled_compare(qtapp, client: Client):
    image = Image.load(image_dir / "character.webp")
    job = create(
        WorkflowKind.upscale_tiled,
        client,
        canvas=image,
        upscale_model=client.models.default_upscaler,
        upscale_factor=2,
        style=default_style(client),
        cond=ConditioningInput("4k uhd"),
        strength=0.5,
    )
    run_and_save(qtapp, client, job, f"test_upscale_tiled_c")


inpaint_benchmark = {
    "tori": (InpaintMode.fill, "photo of tori, japanese garden", None),
    "bruges": (InpaintMode.fill, "photo of a canal in bruges, belgium", None),
    "apple-tree": (
        InpaintMode.expand,
        "children's illustration of kids next to an apple tree",
        Bounds(0, 640, 1024, 384),
    ),
    "girl-cornfield": (
        InpaintMode.expand,
        "anime artwork of girl in a cornfield",
        Bounds(0, 0, 773 - 261, 768),
    ),
    "cuban-guitar": (InpaintMode.replace_background, "photo of a beach bar", None),
    "jungle": (
        InpaintMode.fill,
        "concept artwork of a lake in a forest",
        Bounds(680, 640, 1480, 1280),
    ),
    "street": (InpaintMode.remove_object, "photo of a street in tokyo", None),
    "nature": (
        InpaintMode.add_object,
        "photo of a black bear standing in a stony river bed",
        Bounds(420, 200, 604, 718),
    ),
    "park": (
        InpaintMode.add_object,
        "photo of a lady sitting on a bench in a park",
        Bounds(310, 370, 940, 1110),
    ),
    "superman": (
        InpaintMode.expand,
        "superman giving a speech at a congress hall filled with people",
        None,
    ),
}


def run_inpaint_benchmark(
    qtapp, client, sdver: SDVersion, prompt_mode: str, scenario: str, seed: int, out_dir: Path
):
    mode, prompt, bounds = inpaint_benchmark[scenario]
    image = Image.load(image_dir / "inpaint" / f"{scenario}-image.webp")
    mask = Mask.load(image_dir / "inpaint" / f"{scenario}-mask.webp")
    if bounds:
        mask = Mask.crop(mask, bounds)
    text = ConditioningInput(prompt if prompt_mode == "prompt" else "")
    params = detect_inpaint(mode, mask.bounds, sdver, text.positive, [], 1.0)
    job = create(
        WorkflowKind.inpaint,
        client,
        style=default_style(client, sdver),
        canvas=image,
        mask=mask,
        cond=text,
        inpaint=params,
        seed=seed,
    )
    result_name = f"benchmark_inpaint_{scenario}_{sdver.name}_{prompt_mode}_{seed}"
    run_and_save(qtapp, client, job, result_name, image, mask, output_dir=out_dir)


def test_inpaint_benchmark(pytestconfig, qtapp, client):
    if not pytestconfig.getoption("--benchmark"):
        pytest.skip("Only runs with --benchmark")
    if isinstance(client, CloudClient):
        pytest.skip("Inpaint benchmark runs local")
    print()

    output_dir = config.benchmark_dir / datetime.now().strftime("%Y%m%d-%H%M")
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [4213, 897281]
    prompt_modes = ["prompt", "noprompt"]
    scenarios = inpaint_benchmark.keys()
    sdvers = [SDVersion.sd15, SDVersion.sdxl]
    runs = itertools.product(sdvers, scenarios, prompt_modes, seeds)

    for sdver, scenario, prompt_mode, seed in runs:
        mode, _, _ = inpaint_benchmark[scenario]
        prompt_required = mode in [InpaintMode.add_object, InpaintMode.replace_background]
        if prompt_required and prompt_mode == "noprompt":
            continue

        print("-", scenario, "|", sdver.name, "|", prompt_mode, "|", seed)
        run_inpaint_benchmark(qtapp, client, sdver, prompt_mode, scenario, seed, output_dir)
