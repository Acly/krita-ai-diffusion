import itertools
import shutil
import pytest
from datetime import datetime
from pathlib import Path

from ai_diffusion import comfyworkflow, workflow
from ai_diffusion.comfyworkflow import ComfyWorkflow
from ai_diffusion.resources import ControlMode
from ai_diffusion.image import Mask, Bounds, Extent, Image
from ai_diffusion.client import Client, ClientEvent
from ai_diffusion.style import SDVersion, Style
from ai_diffusion.pose import Pose
from ai_diffusion.workflow import (
    Conditioning,
    Control,
    CheckpointResolution,
    ScaledExtent,
    ScaleMode,
    InpaintMode,
    InpaintParams,
    FillMode,
)
from . import config
from .config import data_dir, image_dir, result_dir, reference_dir, default_checkpoint


@pytest.fixture(scope="session", autouse=True)
def clear_results():
    if result_dir.exists():
        for file in result_dir.iterdir():
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()
    result_dir.mkdir(exist_ok=True)


@pytest.fixture()
def comfy(pytestconfig, qtapp):
    if pytestconfig.getoption("--ci"):
        pytest.skip("Diffusion is disabled on CI")
    return qtapp.run(Client.connect())


default_seed = 1234
dummy_style = Style(Path("dummy.json"))


def default_style(comfy, sd_ver=SDVersion.sd15):
    version_checkpoints = [c for c in comfy.checkpoints if sd_ver.matches(c)]
    checkpoint = default_checkpoint[sd_ver]

    style = Style(Path("default.json"))
    style.sd_checkpoint = (
        checkpoint if checkpoint in version_checkpoints else version_checkpoints[0]
    )
    return style


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


async def run_and_save(
    comfy: Client,
    workflow: ComfyWorkflow,
    filename: str,
    composition_image: Image | None = None,
    composition_mask: Mask | None = None,
    output_dir: Path = result_dir,
):
    workflow.dump((output_dir / "workflows" / filename).with_suffix(".json"))
    results = await receive_images(comfy, workflow)
    assert len(results) == 1
    if composition_image and composition_mask:
        composition_image.draw_image(results[0], composition_mask.bounds.offset)
        composition_image.save(output_dir / filename)
    else:
        results[0].save(output_dir / filename)
    return results[0]


@pytest.mark.parametrize(
    "extent, min_size, max_batches, expected",
    [
        (Extent(512, 512), 512, 4, 4),
        (Extent(512, 512), 512, 6, 6),
        (Extent(1024, 512), 512, 8, 4),
        (Extent(1024, 1024), 512, 8, 2),
        (Extent(2048, 1024), 512, 6, 1),
        (Extent(256, 256), 512, 4, 4),
    ],
)
def test_compute_batch_size(extent, min_size, max_batches, expected):
    assert workflow.compute_batch_size(extent, min_size, max_batches) == expected


def test_scaled_extent_no_scaling():
    x = Extent(100, 100)
    e = ScaledExtent(x, x, x, x)
    assert e.initial_scaling is ScaleMode.none
    assert e.refinement_scaling is ScaleMode.none
    assert e.target_scaling is ScaleMode.none
    extent_names = ["input", "initial", "desired", "target"]
    assert all(
        e.convert(Extent(10, 10), a, b) == Extent(10, 10)
        for a in extent_names
        for b in extent_names
    )


def test_scaled_extent_upscale():
    e = ScaledExtent(Extent(100, 100), Extent(100, 100), Extent(400, 400), Extent(800, 800))
    assert e.initial_scaling is ScaleMode.none
    assert e.refinement_scaling is ScaleMode.upscale_quality
    assert e.target_scaling is ScaleMode.upscale_fast
    assert e.convert(Extent(10, 10), "initial", "desired") == Extent(40, 40)
    assert e.convert(Extent(10, 10), "initial", "target") == Extent(80, 80)
    assert e.convert(Extent(20, 20), "desired", "initial") == Extent(5, 5)


def test_scaled_extent_upscale_small():
    e = ScaledExtent(Extent(100, 50), Extent(100, 50), Extent(140, 70), Extent(144, 72))
    assert e.initial_scaling is ScaleMode.none
    assert e.refinement_scaling is ScaleMode.upscale_latent
    assert e.target_scaling is ScaleMode.resize
    assert e.convert(Extent(140, 70), "desired", "target") == Extent(144, 72)
    assert e.convert(Extent(140, 70), "desired", "initial") == Extent(100, 50)


def test_scaled_extent_downscale():
    e = ScaledExtent(Extent(100, 100), Extent(200, 200), Extent(200, 200), Extent(96, 96))
    assert e.initial_scaling is ScaleMode.resize
    assert e.refinement_scaling is ScaleMode.none
    assert e.target_scaling is ScaleMode.resize
    assert e.convert(Extent(10, 10), "initial", "desired") == Extent(10, 10)
    assert e.convert(Extent(10, 10), "desired", "target") == Extent(5, 5)
    assert e.convert(Extent(96, 96), "target", "initial") == Extent(200, 200)


def test_scaled_extent_multiple8():
    e = ScaledExtent(Extent(512, 513), Extent(512, 520), Extent(512, 520), Extent(512, 513))
    assert e.initial_scaling is ScaleMode.resize
    assert e.refinement_scaling is ScaleMode.none
    assert e.target_scaling is ScaleMode.resize
    assert e.convert(Extent(512, 513), "input", "initial") == Extent(512, 520)
    assert e.convert(Extent(512, 520), "desired", "target") == Extent(512, 513)


@pytest.mark.parametrize(
    "extent,preferred,expected",
    [
        (Extent(512, 512), 640, CheckpointResolution(512, 768, 5 / 4, 5 / 4)),
        (Extent(768, 768), 640, CheckpointResolution(512, 768, 5 / 6, 5 / 6)),
    ],
)
def test_compute_checkpoint_resolution(extent: Extent, preferred: int, expected):
    style = Style(Path("default.json"))
    style.preferred_resolution = preferred
    assert CheckpointResolution.compute(extent, SDVersion.sdxl, style) == expected


@pytest.mark.parametrize(
    "area,expected_extent,expected_crop",
    [
        (Bounds(0, 0, 128, 512), Extent(384, 512), (0, 256)),
        (Bounds(384, 0, 128, 512), Extent(384, 512), (383, 256)),
        (Bounds(0, 0, 512, 128), Extent(512, 384), (256, 0)),
        (Bounds(0, 384, 512, 128), Extent(512, 384), (256, 383)),
        (Bounds(0, 0, 512, 512), None, None),
        (Bounds(0, 0, 256, 256), None, None),
        (Bounds(256, 256, 256, 256), None, None),
    ],
    ids=["left", "right", "top", "bottom", "full", "small", "offset"],
)
def test_inpaint_context(area, expected_extent, expected_crop: tuple[int, int] | None):
    image = Image.load(data_dir / "outpaint_context.png")
    mode = workflow.detect_inpaint_mode(image.extent, area)
    result = workflow.get_inpaint_reference(image, area)
    if area.extent == image.extent:
        assert mode is InpaintMode.expand and result is None
    elif expected_crop:
        assert mode is InpaintMode.expand and isinstance(result, Image)
        assert result.extent == expected_extent
        assert result.pixel(*expected_crop) == (255, 255, 255, 255)
    else:
        assert mode is InpaintMode.fill and result is None


@pytest.mark.parametrize(
    "input,expected_initial,expected_desired",
    [
        (Extent(1536, 600), Extent(1008, 392), Extent(1536, 600)),
        (Extent(400, 1024), Extent(392, 1008), Extent(400, 1024)),
        (Extent(777, 999), Extent(560, 712), Extent(784, 1000)),
    ],
)
def test_prepare_highres(input, expected_initial, expected_desired):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    extent, image, mask_image, _ = workflow.prepare_masked(image, mask, SDVersion.sd15, dummy_style)
    assert (
        extent.input == image.extent
        and image.extent == expected_initial
        and mask_image.extent == expected_initial
        and extent.initial == expected_initial
        and extent.desired == expected_desired
        and extent.target == input
    )


@pytest.mark.parametrize(
    "input,expected",
    [
        (Extent(256, 256), Extent(512, 512)),
        (Extent(128, 450), Extent(280, 960)),
        (Extent(256, 333), Extent(456, 584)),  # multiple of 8
    ],
)
def test_prepare_lowres(input: Extent, expected: Extent):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    extent, image, mask_image, _ = workflow.prepare_masked(image, mask, SDVersion.sd15, dummy_style)
    assert (
        extent.input == input
        and image.extent == input
        and mask_image.extent == input
        and extent.target == input
        and extent.initial == expected
        and extent.desired == expected
    )


@pytest.mark.parametrize(
    "input",
    [Extent(512, 512), Extent(128, 600), Extent(768, 240)],
)
def test_prepare_passthrough(input: Extent):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    extent, image, mask_image, _ = workflow.prepare_masked(image, mask, SDVersion.sd15, dummy_style)
    assert (
        image == image
        and mask_image.extent == input
        and extent.input == input
        and extent.initial == input
        and extent.target == input
        and extent.desired == input
    )


@pytest.mark.parametrize(
    "input,expected", [(Extent(512, 513), Extent(512, 520)), (Extent(300, 1024), Extent(304, 1024))]
)
def test_prepare_multiple8(input: Extent, expected: Extent):
    result, _ = workflow.prepare_extent(input, SDVersion.sd15, dummy_style)
    assert (
        result.input == input
        and result.initial == expected
        and result.target == input
        and result.desired == input.multiple_of(8)
    )


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_prepare_extent(sdver: SDVersion):
    input = Extent(1024, 1536)
    result, _ = workflow.prepare_extent(input, sdver, dummy_style)
    expected = Extent(512, 768) if sdver == SDVersion.sd15 else Extent(840, 1256)
    assert result.initial == expected and result.desired == input and result.target == input


def test_prepare_no_mask():
    image = Image.create(Extent(256, 256))
    extent, result, _ = workflow.prepare_image(image, SDVersion.sd15, dummy_style)
    assert result == image and extent.initial == Extent(512, 512) and extent.target == image.extent


@pytest.mark.parametrize("input", [Extent(512, 512), Extent(1024, 1628), Extent(1536, 999)])
def test_prepare_no_downscale(input: Extent):
    image = Image.create(input)
    extent, result, _ = workflow.prepare_image(image, SDVersion.sd15, dummy_style, downscale=False)
    assert (
        result == image
        and extent.initial == input.multiple_of(8)
        and extent.desired == input.multiple_of(8)
        and extent.target == input
    )


@pytest.mark.parametrize(
    "sd_ver,input,expected_initial,expected_desired",
    [
        (SDVersion.sd15, Extent(2000, 2000), (632, 632), (1000, 1000)),
        (SDVersion.sd15, Extent(1000, 1000), (632, 632), (1000, 1000)),
        (SDVersion.sdxl, Extent(1024, 1024), (1024, 1024), (1024, 1024)),
        (SDVersion.sdxl, Extent(2000, 2000), (1000, 1000), (1000, 1000)),
        (SDVersion.sd15, Extent(801, 801), (632, 632), (808, 808)),
    ],
    ids=["sd15_large", "sd15_small", "sdxl_small", "sdxl_large", "sd15_odd"],
)
def test_prepare_max_pixel_count(input, sd_ver, expected_initial, expected_desired, temp_settings):
    temp_settings.max_pixel_count = 1  # million pixels
    result, _ = workflow.prepare_extent(input, sd_ver, dummy_style)
    assert (
        result.initial == expected_initial
        and result.desired == expected_desired
        and result.target == input
    )


@pytest.mark.parametrize(
    "input,multiplier,expected_initial,expected_desired",
    [
        (Extent(512, 512), 1.0, Extent(512, 512), Extent(512, 512)),
        (Extent(1024, 800), 0.5, Extent(512, 400), Extent(512, 400)),
        (Extent(2048, 1536), 0.5, Extent(728, 544), Extent(1024, 768)),
        (Extent(1024, 1024), 0.4, Extent(512, 512), Extent(512, 512)),
        (Extent(512, 768), 0.5, Extent(512, 768), Extent(512, 768)),
        (Extent(512, 512), 2.0, Extent(632, 632), Extent(1024, 1024)),
        (Extent(512, 512), 1.1, Extent(568, 568), Extent(568, 568)),
    ],
    ids=["1.0", "0.5", "0.5_large", "0.4", "0.5_tall", "2.0", "1.1"],
)
def test_prepare_resolution_multiplier(
    input, multiplier, expected_initial, expected_desired, temp_settings
):
    temp_settings.resolution_multiplier = multiplier
    result, _ = workflow.prepare_extent(input, SDVersion.sd15, dummy_style)
    assert (
        result.initial == expected_initial
        and result.desired == expected_desired
        and result.target == input
    )


def test_prepare_resolution_multiplier_inputs(temp_settings):
    temp_settings.resolution_multiplier = 0.5
    input = Extent(1024, 1024)
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    extent, image, mask_image, _ = workflow.prepare_masked(image, mask, SDVersion.sd15, dummy_style)
    assert (
        extent.input == Extent(512, 512)
        and image.extent == Extent(512, 512)
        and mask_image.extent == Extent(512, 512)
        and extent.initial == Extent(512, 512)
        and extent.desired == Extent(512, 512)
        and extent.target == Extent(1024, 1024)
    )


@pytest.mark.parametrize(
    "multiplier,expected",
    [(0.5, Extent(1024, 1024)), (2, Extent(1000, 1000)), (0.25, Extent(512, 512))],
)
def test_prepare_resolution_multiplier_max(multiplier, expected, temp_settings):
    temp_settings.resolution_multiplier = multiplier
    temp_settings.max_pixel_count = 1  # million pixels
    input = Extent(2048, 2048)
    result, _ = workflow.prepare_extent(input, SDVersion.sd15, dummy_style)
    assert result.initial.width <= 632 and result.desired == expected


def test_merge_prompt():
    assert workflow.merge_prompt("a", "b") == "a, b"
    assert workflow.merge_prompt("", "b") == "b"
    assert workflow.merge_prompt("a", "") == "a"
    assert workflow.merge_prompt("", "") == ""
    assert workflow.merge_prompt("a", "b {prompt} c") == "b a c"
    assert workflow.merge_prompt("", "b {prompt} c") == "b  c"


def test_extract_loras():
    loras = [
        "/path/to/Lora-One.safetensors",
        "Lora-two.safetensors",
    ]

    assert workflow.extract_loras("a ship", loras) == ("a ship", [])
    assert workflow.extract_loras("a ship <lora:lora-one>", loras) == (
        "a ship",
        [{"name": loras[0], "strength": 1.0}],
    )
    assert workflow.extract_loras("a ship <lora:LoRA-one>", loras) == (
        "a ship",
        [{"name": loras[0], "strength": 1.0}],
    )
    assert workflow.extract_loras("a ship <lora:lora-one:0.0>", loras) == (
        "a ship",
        [{"name": loras[0], "strength": 0.0}],
    )
    assert workflow.extract_loras("a ship <lora:lora-two:0.5>", loras) == (
        "a ship",
        [{"name": loras[1], "strength": 0.5}],
    )
    assert workflow.extract_loras("a ship <lora:lora-two:-1.0>", loras) == (
        "a ship",
        [{"name": loras[1], "strength": -1.0}],
    )

    try:
        workflow.extract_loras("a ship <lora:lora-three>", loras)
    except Exception as e:
        assert str(e).startswith("LoRA not found")

    try:
        workflow.extract_loras("a ship <lora:lora-one:test-invalid-str>", loras)
    except Exception as e:
        assert str(e).startswith("Invalid LoRA strength")


@pytest.mark.parametrize("ksampler_type", ["basic", "advanced"])
def test_increment_seed(ksampler_type):
    w = ComfyWorkflow()
    model, clip, vae = w.load_checkpoint(default_checkpoint[SDVersion.sd15])
    prompt = w.clip_text_encode(clip, "")
    o = w.empty_latent_image(Extent(512, 512))
    if ksampler_type == "basic":
        o = w.ksampler(model, prompt, prompt, o, seed=5)
    else:
        o = w.ksampler_advanced(model, prompt, prompt, o, seed=5)
    o = w.vae_decode(vae, o)
    w.save_image(o, "test_increment_seed.png")

    class_type, seed_name = {
        "basic": ("KSampler", "seed"),
        "advanced": ("KSamplerAdvanced", "noise_seed"),
    }[ksampler_type]
    ksampler = w.root["4"]
    assert ksampler["class_type"] == class_type
    assert ksampler["inputs"][seed_name] == 5
    assert w.seed == 5
    w.seed += 4
    assert ksampler["inputs"][seed_name] == 9


def test_inpaint_params():
    mask = Mask.rectangle(Bounds(0, 0, 100, 100))
    cond = Conditioning()

    a = InpaintParams.detect(mask, InpaintMode.fill, SDVersion.sd15, cond, 1.0)
    assert (
        a.fill is FillMode.blur
        and a.use_inpaint_control == True
        and a.use_inpaint_model == False
        and a.use_reference == True
    )
    b = InpaintParams.detect(mask, InpaintMode.add_object, SDVersion.sd15, cond, 1.0)
    assert (
        b.fill is FillMode.neutral
        and b.use_inpaint_control == True
        and b.use_condition_mask == False
    )
    c = InpaintParams.detect(mask, InpaintMode.replace_background, SDVersion.sdxl, cond, 1.0)
    assert (
        c.fill is FillMode.replace
        and c.use_inpaint_control == False
        and c.use_inpaint_model == True
        and c.use_reference == False
    )
    cond.prompt = "prompt"
    d = InpaintParams.detect(mask, InpaintMode.add_object, SDVersion.sd15, cond, 1.0)
    assert d.use_condition_mask == True

    cond.control.append(Control(ControlMode.line_art, Image.create(Extent(4, 4))))
    e = InpaintParams.detect(mask, InpaintMode.add_object, SDVersion.sd15, cond, 1.0)
    assert e.use_condition_mask == False


@pytest.mark.parametrize("extent", [Extent(256, 256), Extent(800, 800), Extent(512, 1024)])
def test_generate(qtapp, comfy, temp_settings, extent: Extent):
    temp_settings.batch_size = 1
    prompt = Conditioning("ship")

    async def main():
        job = workflow.generate(comfy, default_style(comfy), extent, prompt, default_seed)
        result = await run_and_save(comfy, job, f"test_generate_{extent.width}x{extent.height}.png")
        assert result.extent == extent

    qtapp.run(main())


def test_inpaint(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 3  # max 3 images@512x512 -> 2 images@768x512
    image = Image.load(image_dir / "beach_768x512.webp")
    mask = Mask.rectangle(Bounds(40, 120, 320, 200), feather=10)
    prompt = Conditioning("beach, the sea, cliffs, palm trees")
    params = InpaintParams.detect(mask, InpaintMode.fill, SDVersion.sd15, prompt, 1.0)
    style = default_style(comfy, SDVersion.sd15)
    job = workflow.inpaint(comfy, style, image, prompt, params, default_seed)

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
    prompt = Conditioning("ship")
    params = InpaintParams.detect(mask, InpaintMode.add_object, sdver, prompt, 1.0)
    style = default_style(comfy, sdver)
    job = workflow.inpaint(comfy, style, image, prompt, params, default_seed)

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
    prompt = Conditioning()
    params = InpaintParams.automatic(mask, SDVersion.sd15, prompt, image.extent)

    async def main():
        job = workflow.inpaint(comfy, default_style(comfy), image, prompt, params, default_seed)
        result = await run_and_save(comfy, job, "test_inpaint_odd_resolution.png", image, mask)
        assert result.extent == mask.bounds.extent

    qtapp.run(main())


def test_inpaint_area_conditioning(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "lake_1536x1024.webp")
    mask = Mask.load(image_dir / "lake_1536x1024_mask_bottom_right.png")
    prompt = Conditioning("(crocodile)")
    params = InpaintParams.detect(mask, InpaintMode.add_object, SDVersion.sd15, prompt, 1.0)
    job = workflow.inpaint(comfy, default_style(comfy), image, prompt, params, default_seed)

    async def main():
        await run_and_save(comfy, job, "test_inpaint_area_conditioning.png", image, mask)

    qtapp.run(main())


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
    prompt = Conditioning("painting in the style of Vincent van Gogh")

    async def main():
        job = workflow.refine(
            comfy, default_style(comfy, sdver), image, prompt, strength, default_seed
        )
        result = await run_and_save(comfy, job, f"test_refine_{setup}.png")
        assert result.extent == extent

    qtapp.run(main())


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
    prompt = Conditioning("waterfall")
    params = InpaintParams.detect(mask, InpaintMode.fill, SDVersion.sd15, prompt, strength)
    style = default_style(comfy, sdver)
    job = workflow.refine_region(comfy, style, image, params, prompt, strength, default_seed)

    async def main():
        result = await run_and_save(comfy, job, f"test_refine_region_{setup}.png", image, mask)
        assert result.extent == mask.bounds.extent

    qtapp.run(main())


@pytest.mark.parametrize(
    "op", ["generate", "inpaint", "refine", "refine_region", "inpaint_upscale"]
)
def test_control_scribble(qtapp, comfy, temp_settings, op):
    temp_settings.batch_size = 1
    style = default_style(comfy)
    scribble_image = Image.load(image_dir / "owls_scribble.webp")
    inpaint_image = Image.load(image_dir / "owls_inpaint.webp")
    mask = Mask.load(image_dir / "owls_mask.png")
    mask.bounds = Bounds(256, 0, 256, 512)
    control = Conditioning("owls", "", [Control(ControlMode.scribble, scribble_image)])

    if op == "generate":
        job = workflow.generate(comfy, style, Extent(512, 512), control, default_seed)
    elif op == "inpaint":
        params = InpaintParams.automatic(mask, SDVersion.sd15, control, inpaint_image.extent)
        job = workflow.inpaint(comfy, style, inpaint_image, control, params, default_seed)
    elif op == "refine":
        job = workflow.refine(comfy, style, inpaint_image, control, 0.7, default_seed)
    elif op == "refine_region":
        cropped_image = Image.crop(inpaint_image, mask.bounds)
        params = InpaintParams.automatic(mask, SDVersion.sd15, control, cropped_image.extent)
        job = workflow.refine_region(
            comfy, style, cropped_image, params, control, 0.7, default_seed
        )
    else:  # op == "inpaint_upscale":
        control.control[0].image = Image.scale(scribble_image, Extent(1024, 1024))
        inpaint_image = Image.scale(inpaint_image, Extent(1024, 1024))
        scaled_mask = Image.scale(Image(mask.image), Extent(512, 1024))
        mask = Mask(Bounds(512, 0, 512, 1024), scaled_mask._qimage)
        params = InpaintParams.detect(mask, InpaintMode.fill, SDVersion.sd15, control, 1.0)
        job = workflow.inpaint(comfy, style, inpaint_image, control, params, default_seed)

    async def main():
        if op in ["inpaint", "refine_region", "inpaint_upscale"]:
            await run_and_save(comfy, job, f"test_control_scribble_{op}.png", inpaint_image, mask)
        else:
            await run_and_save(comfy, job, f"test_control_scribble_{op}.png")

    qtapp.run(main())


def test_control_canny_downscale(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    style = default_style(comfy)
    canny_image = Image.load(image_dir / "shrine_canny.webp")
    control = Conditioning("shrine", "", [Control(ControlMode.canny_edge, canny_image, 1.0)])
    job = workflow.generate(comfy, style, Extent(999, 999), control, default_seed)

    async def main():
        await run_and_save(comfy, job, "test_control_canny_downscale.png")

    qtapp.run(main())


@pytest.mark.parametrize("mode", [m for m in ControlMode if m.has_preprocessor])
def test_create_control_image(qtapp, comfy, mode):
    image_name = f"test_create_control_image_{mode.name}.png"
    image = Image.load(image_dir / "adobe_stock.jpg")
    job = workflow.create_control_image(comfy, image, mode)

    async def main():
        result = await run_and_save(comfy, job, image_name)
        reference = Image.load(reference_dir / image_name)
        threshold = 0.015 if mode is ControlMode.pose else 0.002
        assert Image.compare(result, reference) < threshold

    qtapp.run(main())


def test_create_open_pose_vector(qtapp, comfy):
    image_name = f"test_create_open_pose_vector.svg"
    image = Image.load(image_dir / "adobe_stock.jpg")
    job = workflow.create_control_image(comfy, image, ControlMode.pose)

    async def main():
        job_id = None
        async for msg in comfy.listen():
            if not job_id:
                job_id = await comfy.enqueue(job)
            if msg.event is ClientEvent.finished and msg.job_id == job_id:
                result = Pose.from_open_pose_json(msg.result).to_svg()
                (result_dir / image_name).write_text(result)
                return
            if msg.event is ClientEvent.error and msg.job_id == job_id:
                raise Exception(msg.error)
        assert False, "Connection closed without receiving images"

    qtapp.run(main())


@pytest.mark.parametrize("setup", ["no_mask", "right_hand", "left_hand"])
def test_create_hand_refiner_image(qtapp, comfy, setup):
    image_name = f"test_create_hand_refiner_image_{setup}.png"
    image = Image.load(image_dir / "character.webp")
    bounds = {
        "no_mask": None,
        "right_hand": Bounds(102, 398, 264, 240),
        "left_hand": Bounds(541, 642, 232, 248),
    }[setup]
    job = workflow.create_control_image(comfy, image, ControlMode.hands, bounds, default_seed)

    async def main():
        result = await run_and_save(comfy, job, image_name)
        reference = Image.load(reference_dir / image_name)
        assert Image.compare(result, reference) < 0.002

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_ip_adapter(qtapp, comfy, temp_settings, sdver):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "cat.webp")
    control = Conditioning(
        "cat on a rooftop in paris", "", [Control(ControlMode.reference, image, 0.6)]
    )
    extent = Extent(512, 512) if sdver == SDVersion.sd15 else Extent(1024, 1024)
    job = workflow.generate(comfy, default_style(comfy, sdver), extent, control, default_seed)

    async def main():
        await run_and_save(comfy, job, f"test_ip_adapter_{sdver.name}.png")

    qtapp.run(main())


def test_ip_adapter_region(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "flowers.webp")
    mask = Mask.load(image_dir / "flowers_mask.png")
    control_img = Image.load(image_dir / "pegonia.webp")
    control = Conditioning("potted flowers", "", [Control(ControlMode.reference, control_img, 0.7)])
    inpaint = InpaintParams.automatic(mask, SDVersion.sd15, control, image.extent)
    job = workflow.refine_region(
        comfy, default_style(comfy), image, inpaint, control, 0.6, default_seed
    )

    async def main():
        await run_and_save(comfy, job, "test_ip_adapter_region.png", image, mask)

    qtapp.run(main())


def test_ip_adapter_batch(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image1 = Image.load(image_dir / "cat.webp")
    image2 = Image.load(image_dir / "pegonia.webp")
    control = Conditioning(
        "",
        "",
        [Control(ControlMode.reference, image1, 1.0), Control(ControlMode.reference, image2, 1.0)],
    )
    job = workflow.generate(comfy, default_style(comfy), Extent(512, 512), control, default_seed)

    async def main():
        await run_and_save(comfy, job, "test_ip_adapter_batch.png")

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_ip_adapter_face(qtapp, comfy, temp_settings, sdver):
    temp_settings.batch_size = 1
    extent = Extent(650, 650) if sdver == SDVersion.sd15 else Extent(1024, 1024)
    image = Image.load(image_dir / "face.webp")
    control = Conditioning(
        "portrait photo of a woman at a garden party", "", [Control(ControlMode.face, image, 0.9)]
    )
    job = workflow.generate(comfy, default_style(comfy), extent, control, default_seed)

    async def main():
        await run_and_save(comfy, job, f"test_ip_adapter_face_{sdver.name}.png")

    qtapp.run(main())


def test_upscale_simple(qtapp, comfy):
    image = Image.load(image_dir / "beach_768x512.webp")
    job = workflow.upscale_simple(comfy, image, comfy.default_upscaler, 2.0)

    async def main():
        await run_and_save(comfy, job, "test_upscale_simple.png")

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_upscale_tiled(qtapp, comfy, sdver):
    image = Image.load(image_dir / "beach_768x512.webp")
    job = workflow.upscale_tiled(
        comfy, image, comfy.default_upscaler, 2.0, default_style(comfy, sdver), 0.5, default_seed
    )

    async def main():
        await run_and_save(comfy, job, f"test_upscale_tiled_{sdver.name}.png")

    qtapp.run(main())


def test_generate_live(qtapp, comfy):
    scribble = Image.load(image_dir / "owls_scribble.webp")
    cond = Conditioning("owls", "", [Control(ControlMode.scribble, scribble)])
    job = workflow.generate(
        comfy, default_style(comfy), Extent(512, 512), cond, default_seed, is_live=True
    )

    async def main():
        await run_and_save(comfy, job, "test_generate_live.png")

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_refine_live(qtapp, comfy, sdver):
    image = Image.load(image_dir / "pegonia.webp")
    if sdver is SDVersion.sdxl:
        image = Image.scale(image, Extent(1024, 1024))  # result will be a bit blurry
    job = workflow.refine(
        comfy, default_style(comfy, sdver), image, Conditioning(), 0.4, default_seed, is_live=True
    )

    async def main():
        await run_and_save(comfy, job, f"test_refine_live_{sdver.name}.png")

    qtapp.run(main())


def test_refine_max_pixels(qtapp, comfy, temp_settings):
    temp_settings.max_pixel_count = 1  # million pixels
    image = Image.load(image_dir / "lake_1536x1024.webp")
    cond = Conditioning("watercolor painting on structured paper, aquarelle, stylized")
    job = workflow.refine(comfy, default_style(comfy), image, cond, 0.6, default_seed)

    async def main():
        await run_and_save(comfy, job, f"test_refine_max_pixels.png")

    qtapp.run(main())


def test_outpaint_resolution_multiplier(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    temp_settings.resolution_multiplier = 0.8
    image = Image.create(Extent(2048, 1024))
    beach = Image.load(image_dir / "beach_1536x1024.webp")
    image.draw_image(beach, (512, 0))
    mask = Mask.load(image_dir / "beach_outpaint_mask.png")
    cond = Conditioning("photo of a beach and jungle, nature photography, tropical")
    params = InpaintParams.automatic(mask, SDVersion.sd15, cond, image.extent)
    job = workflow.inpaint(comfy, default_style(comfy), image, cond, params, default_seed)

    async def main():
        await run_and_save(comfy, job, f"test_outpaint_resolution_multiplier.png", image, mask)

    qtapp.run(main())


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


async def run_inpaint_benchmark(
    comfy, sdver: SDVersion, prompt_mode: str, scenario: str, seed: int, out_dir: Path
):
    mode, prompt, bounds = inpaint_benchmark[scenario]
    image = Image.load(image_dir / "inpaint" / f"{scenario}-image.webp")
    mask = Mask.load(image_dir / "inpaint" / f"{scenario}-mask.webp")
    if bounds:
        mask = Mask.crop(mask, bounds)
    prompt_text = prompt if prompt_mode == "prompt" else ""
    cond = Conditioning(prompt_text)
    params = InpaintParams.detect(mask, mode, sdver, cond, 1.0)
    job = workflow.inpaint(comfy, default_style(comfy, sdver), image, cond, params, seed)
    result_name = f"benchmark_inpaint_{scenario}_{sdver.name}_{prompt_mode}_{seed}.webp"
    await run_and_save(comfy, job, result_name, image, mask, output_dir=out_dir)


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
        qtapp.run(run_inpaint_benchmark(comfy, sdver, prompt_mode, scenario, seed, output_dir))
