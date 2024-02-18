import itertools
from typing import Any
import pytest
from datetime import datetime
from pathlib import Path

from ai_diffusion import workflow
from ai_diffusion.api import WorkflowKind, ControlInput, InpaintMode, FillMode
from ai_diffusion.api import TextInput
from ai_diffusion.comfyworkflow import ComfyWorkflow
from ai_diffusion.resources import ControlMode
from ai_diffusion.resolution import ScaledExtent
from ai_diffusion.image import Mask, Bounds, Extent, Image
from ai_diffusion.client import Client, ClientEvent
from ai_diffusion.style import SDVersion, Style
from ai_diffusion.pose import Pose
from ai_diffusion.workflow import Control, detect_inpaint
from ai_diffusion.util import ensure
from . import config
from .config import image_dir, result_dir, reference_dir, default_checkpoint


@pytest.fixture()
def comfy(pytestconfig, qtapp):
    if pytestconfig.getoption("--ci"):
        pytest.skip("Diffusion is disabled on CI")
    return qtapp.run(Client.connect())


default_seed = 1234


def default_style(comfy: Client, sd_ver=SDVersion.sd15):
    version_checkpoints = [c for c in comfy.models.checkpoints if sd_ver.matches(c)]
    checkpoint = default_checkpoint[sd_ver]

    style = Style(Path("default.json"))
    style.sd_checkpoint = (
        checkpoint if checkpoint in version_checkpoints else version_checkpoints[0]
    )
    return style


def create(kind: WorkflowKind, client: Client, **kwargs):
    kwargs.setdefault("text", TextInput(""))
    kwargs.setdefault("style", default_style(client))
    kwargs.setdefault("seed", default_seed)
    inputs = workflow.prepare(kind, models=client.models, **kwargs)
    return workflow.create(inputs, client.models)


async def receive_images(comfy: Client, workflow: ComfyWorkflow):
    job_id = None
    async for msg in comfy.listen():
        if not job_id:
            job_id = await comfy.enqueue(workflow)
        if msg.event is ClientEvent.finished and msg.job_id == job_id:
            assert msg.images is not None
            return msg.images
        if msg.event is ClientEvent.error and msg.job_id == job_id:
            raise Exception(msg.error)
    assert False, "Connection closed without receiving images"


def run_and_save(
    qtapp,
    comfy: Client,
    workflow: ComfyWorkflow,
    filename: str,
    composition_image: Image | None = None,
    composition_mask: Mask | None = None,
    output_dir: Path = result_dir,
):
    workflow.dump((output_dir / "workflows" / filename).with_suffix(".json"))

    async def runner():
        return await receive_images(comfy, workflow)

    results = qtapp.run(runner())
    assert len(results) == 1
    if composition_image and composition_mask:
        composition_image.draw_image(results[0], composition_mask.bounds.offset)
        composition_image.save(output_dir / filename)
    else:
        results[0].save(output_dir / filename)
    return results[0]


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


@pytest.mark.parametrize("extent", [Extent(256, 256), Extent(800, 800), Extent(512, 1024)])
def test_generate(qtapp, comfy, temp_settings, extent: Extent):
    temp_settings.batch_size = 1
    prompt = TextInput("ship")
    job = create(WorkflowKind.generate, comfy, canvas=extent, text=prompt)
    result = run_and_save(qtapp, comfy, job, f"test_generate_{extent.width}x{extent.height}.png")
    assert result.extent == extent


def test_inpaint(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 3  # max 3 images@512x512 -> 2 images@768x512
    image = Image.load(image_dir / "beach_768x512.webp")
    mask = Mask.rectangle(Bounds(40, 120, 320, 200), feather=10)
    cond = TextInput("beach, the sea, cliffs, palm trees")
    job = create(
        WorkflowKind.inpaint,
        comfy,
        canvas=image,
        mask=mask,
        style=default_style(comfy, SDVersion.sd15),
        text=cond,
        inpaint=detect_inpaint(
            InpaintMode.fill, mask.bounds, SDVersion.sd15, cond.positive, [], 1.0
        ),
    )

    async def main():
        results = await receive_images(comfy, job)
        assert len(results) == 2
        for i, result in enumerate(results):
            image.draw_image(result, mask.bounds.offset)
            image.save(result_dir / f"test_inpaint_{i}.png")
            assert result.extent == Extent(320, 200)

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_inpaint_upscale(qtapp, comfy, temp_settings, sdver):
    temp_settings.batch_size = 3  # 2 images for 1.5, 1 image for XL
    image = Image.load(image_dir / "beach_1536x1024.webp")
    mask = Mask.rectangle(Bounds(300, 200, 768, 512), feather=20)
    prompt = TextInput("ship")
    job = create(
        WorkflowKind.inpaint,
        comfy,
        canvas=image,
        mask=mask,
        text=prompt,
        inpaint=detect_inpaint(
            InpaintMode.add_object, mask.bounds, sdver, prompt.positive, [], 1.0
        ),
    )

    async def main():
        job.dump((result_dir / "workflows" / f"test_inpaint_upscale_{sdver.name}.json"))
        results = await receive_images(comfy, job)
        assert len(results) == 2 if sdver == SDVersion.sd15 else 1
        for i, result in enumerate(results):
            image.draw_image(result, mask.bounds.offset)
            image.save(result_dir / f"test_inpaint_upscale_{sdver.name}_{i}.png")
            assert result.extent == mask.bounds.extent

    qtapp.run(main())


def test_inpaint_odd_resolution(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "beach_768x512.webp")
    image = Image.scale(image, Extent(612, 513))
    mask = Mask.rectangle(Bounds(0, 0, 200, 513))
    job = create(
        WorkflowKind.inpaint,
        comfy,
        canvas=image,
        mask=mask,
        inpaint=automatic_inpaint(image.extent, mask.bounds),
    )
    result = run_and_save(qtapp, comfy, job, "test_inpaint_odd_resolution.png", image, mask)
    assert result.extent == mask.bounds.extent


def test_inpaint_area_conditioning(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "lake_1536x1024.webp")
    mask = Mask.load(image_dir / "lake_1536x1024_mask_bottom_right.png")
    prompt = TextInput("(crocodile)")
    job = create(
        WorkflowKind.inpaint,
        comfy,
        canvas=image,
        mask=mask,
        text=prompt,
        inpaint=detect_inpaint(
            InpaintMode.add_object, mask.bounds, SDVersion.sd15, prompt.positive, [], 1.0
        ),
    )
    run_and_save(qtapp, comfy, job, "test_inpaint_area_conditioning.png", image, mask)


@pytest.mark.parametrize("setup", ["sd15", "sdxl"])
def test_refine(qtapp, comfy, setup, temp_settings):
    temp_settings.batch_size = 1
    temp_settings.max_pixel_count = 2
    sdver, extent, strength = {
        "sd15": (SDVersion.sd15, Extent(768, 508), 0.5),
        "sdxl": (SDVersion.sdxl, Extent(1111, 741), 0.65),
    }[setup]
    image = Image.load(image_dir / "beach_1536x1024.webp")
    image = Image.scale(image, extent)
    job = create(
        WorkflowKind.refine,
        comfy,
        canvas=image,
        style=default_style(comfy, sdver),
        text=TextInput("painting in the style of Vincent van Gogh"),
        strength=strength,
    )
    result = run_and_save(qtapp, comfy, job, f"test_refine_{setup}.png")
    assert result.extent == extent


@pytest.mark.parametrize("setup", ["sd15_0.4", "sd15_0.6", "sdxl_0.7"])
def test_refine_region(qtapp, comfy, temp_settings, setup):
    temp_settings.batch_size = 1
    sdver, strength = {
        "sd15_0.4": (SDVersion.sd15, 0.4),
        "sd15_0.6": (SDVersion.sd15, 0.6),
        "sdxl_0.7": (SDVersion.sdxl, 0.7),
    }[setup]
    image = Image.load(image_dir / "lake_region.webp")
    mask = Mask.load(image_dir / "lake_region_mask.png")
    prompt = TextInput("waterfall")
    params = detect_inpaint(
        InpaintMode.fill, mask.bounds, SDVersion.sd15, prompt.positive, [], strength
    )
    job = create(
        WorkflowKind.refine_region,
        comfy,
        canvas=image,
        mask=mask,
        style=default_style(comfy, sdver),
        text=prompt,
        strength=strength,
        inpaint=params,
    )
    result = run_and_save(qtapp, comfy, job, f"test_refine_region_{setup}.png", image, mask)
    assert result.extent == mask.bounds.extent


@pytest.mark.parametrize(
    "op", ["generate", "inpaint", "refine", "refine_region", "inpaint_upscale"]
)
def test_control_scribble(qtapp, comfy, temp_settings, op):
    temp_settings.batch_size = 1
    scribble_image = Image.load(image_dir / "owls_scribble.webp")
    inpaint_image = Image.load(image_dir / "owls_inpaint.webp")
    mask = Mask.load(image_dir / "owls_mask.png")
    mask.bounds = Bounds(256, 0, 256, 512)
    prompt = TextInput("owls")
    control = [ControlInput(ControlMode.scribble, scribble_image)]

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

    job = create(client=comfy, text=prompt, control=control, **args)
    if op in ["inpaint", "refine_region", "inpaint_upscale"]:
        run_and_save(qtapp, comfy, job, f"test_control_scribble_{op}.png", inpaint_image, mask)
    else:
        run_and_save(qtapp, comfy, job, f"test_control_scribble_{op}.png")


def test_control_canny_downscale(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    canny_image = Image.load(image_dir / "shrine_canny.webp")
    prompt = TextInput("shrine")
    control = [ControlInput(ControlMode.canny_edge, canny_image, 1.0)]
    job = create(
        WorkflowKind.generate, comfy, canvas=Extent(999, 999), text=prompt, control=control
    )
    run_and_save(qtapp, comfy, job, "test_control_canny_downscale.png")


@pytest.mark.parametrize("mode", [m for m in ControlMode if m.has_preprocessor])
def test_create_control_image(qtapp, comfy: Client, mode):
    image_name = f"test_create_control_image_{mode.name}.png"
    image = Image.load(image_dir / "adobe_stock.jpg")
    extent = ScaledExtent.no_scaling(image.extent)
    models = comfy.models.for_version(SDVersion.sd15)
    job = workflow.create_control_image(models, image, mode, extent)

    result = run_and_save(qtapp, comfy, job, image_name)
    reference = Image.load(reference_dir / image_name)
    threshold = 0.015 if mode is ControlMode.pose else 0.002
    assert Image.compare(result, reference) < threshold


def test_create_open_pose_vector(qtapp, comfy: Client):
    image_name = f"test_create_open_pose_vector.svg"
    image = Image.load(image_dir / "adobe_stock.jpg")
    extent = ScaledExtent.no_scaling(image.extent)
    models = comfy.models.for_version(SDVersion.sd15)
    job = workflow.create_control_image(models, image, ControlMode.pose, extent)

    async def main():
        job_id = None
        async for msg in comfy.listen():
            if not job_id:
                job_id = await comfy.enqueue(job)
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
def test_create_hand_refiner_image(qtapp, comfy: Client, setup):
    image_name = f"test_create_hand_refiner_image_{setup}.png"
    image = Image.load(image_dir / "character.webp")
    extent = ScaledExtent.no_scaling(image.extent)
    bounds = {
        "no_mask": None,
        "right_hand": Bounds(102, 398, 264, 240),
        "left_hand": Bounds(541, 642, 232, 248),
    }[setup]
    models = comfy.models.for_version(SDVersion.sd15)
    job = workflow.create_control_image(
        models, image, ControlMode.hands, extent, bounds, default_seed
    )
    result = run_and_save(qtapp, comfy, job, image_name)
    reference = Image.load(reference_dir / image_name)
    assert Image.compare(result, reference) < 0.002


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_ip_adapter(qtapp, comfy, temp_settings, sdver):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "cat.webp")
    prompt = TextInput("cat on a rooftop in paris")
    control = [Control(ControlMode.reference, image, 0.6)]
    extent = Extent(512, 512) if sdver == SDVersion.sd15 else Extent(1024, 1024)
    style = default_style(comfy, sdver)
    job = create(
        WorkflowKind.generate, comfy, style=style, canvas=extent, text=prompt, control=control
    )
    run_and_save(qtapp, comfy, job, f"test_ip_adapter_{sdver.name}.png")


def test_ip_adapter_region(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "flowers.webp")
    mask = Mask.load(image_dir / "flowers_mask.png")
    control_img = Image.load(image_dir / "pegonia.webp")
    prompt = TextInput("potted flowers")
    control = [ControlInput(ControlMode.reference, control_img, 0.7)]
    inpaint = automatic_inpaint(image.extent, mask.bounds, SDVersion.sd15, prompt.positive, control)
    job = create(
        WorkflowKind.refine_region,
        comfy,
        canvas=image,
        mask=mask,
        inpaint=inpaint,
        text=prompt,
        control=control,
        strength=0.6,
    )
    run_and_save(qtapp, comfy, job, "test_ip_adapter_region.png", image, mask)


def test_ip_adapter_batch(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image1 = Image.load(image_dir / "cat.webp")
    image2 = Image.load(image_dir / "pegonia.webp")
    control = [
        ControlInput(ControlMode.reference, image1, 1.0),
        ControlInput(ControlMode.reference, image2, 1.0),
    ]
    job = create(WorkflowKind.generate, comfy, canvas=Extent(512, 512), control=control)
    run_and_save(qtapp, comfy, job, "test_ip_adapter_batch.png")


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_ip_adapter_face(qtapp, comfy, temp_settings, sdver):
    temp_settings.batch_size = 1
    extent = Extent(650, 650) if sdver == SDVersion.sd15 else Extent(1024, 1024)
    image = Image.load(image_dir / "face.webp")
    cond = TextInput("portrait photo of a woman at a garden party")
    control = [ControlInput(ControlMode.face, image, 0.9)]
    job = create(WorkflowKind.generate, comfy, canvas=extent, text=cond, control=control)
    run_and_save(qtapp, comfy, job, f"test_ip_adapter_face_{sdver.name}.png")


def test_upscale_simple(qtapp, comfy: Client):
    models = comfy.models.for_version(SDVersion.sd15)
    image = Image.load(image_dir / "beach_768x512.webp")
    job = workflow.upscale_simple(image, comfy.models.default_upscaler, 2.0, models)
    run_and_save(qtapp, comfy, job, "test_upscale_simple.png")


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_upscale_tiled(qtapp, comfy: Client, sdver):
    image = Image.load(image_dir / "beach_768x512.webp")
    job = create(
        WorkflowKind.upscale_tiled,
        comfy,
        canvas=image,
        upscale_model=comfy.models.default_upscaler,
        upscale_factor=2.0,
        style=default_style(comfy, sdver),
        text=TextInput("4k uhd"),
        strength=0.5,
    )
    run_and_save(qtapp, comfy, job, f"test_upscale_tiled_{sdver.name}.png")


def test_generate_live(qtapp, comfy):
    scribble = Image.load(image_dir / "owls_scribble.webp")
    job = create(
        WorkflowKind.generate,
        comfy,
        canvas=Extent(512, 512),
        text=TextInput("owls"),
        control=[ControlInput(ControlMode.scribble, scribble)],
        is_live=True,
    )
    run_and_save(qtapp, comfy, job, "test_generate_live.png")


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_refine_live(qtapp, comfy, sdver):
    image = Image.load(image_dir / "pegonia.webp")
    if sdver is SDVersion.sdxl:
        image = Image.scale(image, Extent(1024, 1024))  # result will be a bit blurry
    job = create(
        WorkflowKind.refine,
        comfy,
        style=default_style(comfy, sdver),
        canvas=image,
        text=TextInput(""),
        strength=0.4,
        is_live=True,
    )
    run_and_save(qtapp, comfy, job, f"test_refine_live_{sdver.name}.png")


def test_refine_max_pixels(qtapp, comfy, temp_settings):
    temp_settings.max_pixel_count = 1  # million pixels
    image = Image.load(image_dir / "lake_1536x1024.webp")
    cond = TextInput("watercolor painting on structured paper, aquarelle, stylized")
    job = create(WorkflowKind.refine, comfy, canvas=image, text=cond, strength=0.6)
    run_and_save(qtapp, comfy, job, f"test_refine_max_pixels.png")


def test_outpaint_resolution_multiplier(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    temp_settings.resolution_multiplier = 0.8
    image = Image.create(Extent(2048, 1024))
    beach = Image.load(image_dir / "beach_1536x1024.webp")
    image.draw_image(beach, (512, 0))
    mask = Mask.load(image_dir / "beach_outpaint_mask.png")
    prompt = TextInput("photo of a beach and jungle, nature photography, tropical")
    params = automatic_inpaint(image.extent, mask.bounds, prompt=prompt.positive)
    job = create(WorkflowKind.inpaint, comfy, canvas=image, mask=mask, text=prompt, inpaint=params)
    run_and_save(qtapp, comfy, job, f"test_outpaint_resolution_multiplier.png", image, mask)


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
    qtapp, comfy, sdver: SDVersion, prompt_mode: str, scenario: str, seed: int, out_dir: Path
):
    mode, prompt, bounds = inpaint_benchmark[scenario]
    image = Image.load(image_dir / "inpaint" / f"{scenario}-image.webp")
    mask = Mask.load(image_dir / "inpaint" / f"{scenario}-mask.webp")
    if bounds:
        mask = Mask.crop(mask, bounds)
    text = TextInput(prompt if prompt_mode == "prompt" else "")
    params = detect_inpaint(mode, mask.bounds, sdver, text.positive, [], 1.0)
    job = create(
        WorkflowKind.inpaint,
        comfy,
        style=default_style(comfy, sdver),
        canvas=image,
        text=text,
        inpaint=params,
        seed=seed,
    )
    result_name = f"benchmark_inpaint_{scenario}_{sdver.name}_{prompt_mode}_{seed}.webp"
    run_and_save(qtapp, comfy, job, result_name, image, mask, output_dir=out_dir)


def test_inpaint_benchmark(pytestconfig, qtapp, comfy, temp_settings):
    if not pytestconfig.getoption("--benchmark"):
        pytest.skip("Only runs with --benchmark")
    print()

    temp_settings.batch_size = 1
    output_dir = config.benchmark_dir / datetime.now().strftime("%Y%m%d-%H%M")
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
        run_inpaint_benchmark(qtapp, comfy, sdver, prompt_mode, scenario, seed, output_dir)
