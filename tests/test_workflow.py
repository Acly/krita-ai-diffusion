import pytest
from ai_diffusion import comfyworkflow, workflow
from ai_diffusion.comfyworkflow import ComfyWorkflow
from ai_diffusion.resources import ControlMode
from ai_diffusion.image import Mask, Bounds, Extent, Image
from ai_diffusion.client import Client, ClientEvent
from ai_diffusion.style import SDVersion, Style
from ai_diffusion.pose import Pose
from ai_diffusion.workflow import LiveParams, Conditioning, Control
from pathlib import Path
from .config import data_dir, image_dir, result_dir, reference_dir, default_checkpoint


@pytest.fixture(scope="session", autouse=True)
def clear_results():
    if result_dir.exists():
        for file in result_dir.iterdir():
            file.unlink()
    result_dir.mkdir(exist_ok=True)


@pytest.fixture()
def comfy(pytestconfig, qtapp):
    if pytestconfig.getoption("--ci"):
        pytest.skip("Diffusion is disabled on CI")
    return qtapp.run(Client.connect())


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


async def run_and_save(comfy, workflow: ComfyWorkflow, filename: str):
    results = await receive_images(comfy, workflow)
    assert len(results) == 1
    results[0].save(result_dir / filename)
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
    default = comfyworkflow.Output(0, 0)
    result = workflow.create_inpaint_context(image, area, default)
    if expected_crop:
        assert isinstance(result, Image)
        assert result.extent == expected_extent
        assert result.pixel(*expected_crop) == (255, 255, 255, 255)
    else:
        assert result is default


@pytest.mark.parametrize(
    "input,expected_initial,expected_expanded,expected_scale",
    [
        (Extent(1536, 600), Extent(1008, 392), Extent(1536, 600), 0.6532),
        (Extent(400, 1024), Extent(392, 1008), Extent(400, 1024), 0.9798),
        (Extent(777, 999), Extent(560, 712), Extent(784, 1000), 0.7117),
    ],
)
def test_prepare_highres(input, expected_initial, expected_expanded, expected_scale):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    extent, image, mask_image, _ = workflow.prepare_masked(image, mask, SDVersion.sd15)
    assert (
        extent.requires_upscale
        and image.extent == expected_initial
        and mask_image.extent == expected_initial
        and extent.initial == expected_initial
        and extent.expanded == expected_expanded
        and extent.target == input
        and extent.scale == pytest.approx(expected_scale, abs=1e-3)
    )


@pytest.mark.parametrize(
    "input,expected",
    [
        (Extent(256, 256), Extent(512, 512)),
        (Extent(128, 450), Extent(280, 960)),
        (Extent(256, 333), Extent(456, 584)),  # multiple of 8
    ],
)
def test_prepare_lowres(input, expected):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    extent, image, mask_image, _ = workflow.prepare_masked(image, mask, SDVersion.sd15)
    assert (
        extent.requires_downscale
        and image.extent == input
        and mask_image.extent == input
        and extent.target == input
        and extent.initial == expected
        and extent.expanded == input.multiple_of(8)
        and extent.scale > 1
    )


@pytest.mark.parametrize(
    "input",
    [Extent(512, 512), Extent(128, 600), Extent(768, 240)],
)
def test_prepare_passthrough(input):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    extent, image, mask_image, _ = workflow.prepare_masked(image, mask, SDVersion.sd15)
    assert (
        image == image
        and mask_image.extent == input
        and extent.initial == input
        and extent.target == input
        and extent.expanded == input
        and extent.scale == 1
    )


@pytest.mark.parametrize(
    "input,expected", [(Extent(512, 513), Extent(512, 520)), (Extent(300, 1024), Extent(304, 1024))]
)
def test_prepare_multiple8(input, expected):
    result, _ = workflow.prepare_extent(input, SDVersion.sd15)
    assert (
        result.is_incompatible
        and result.initial == expected
        and result.target == input
        and result.expanded == input.multiple_of(8)
    )


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_prepare_extent(sdver: SDVersion):
    input = Extent(1024, 1536)
    result, _ = workflow.prepare_extent(input, sdver)
    expected = Extent(512, 768) if sdver == SDVersion.sd15 else Extent(840, 1256)
    assert result.initial == expected and result.target == input and result.scale < 1


def test_prepare_no_mask():
    image = Image.create(Extent(256, 256))
    extent, result, _ = workflow.prepare_image(image, SDVersion.sd15)
    assert (
        result == image
        and extent.initial == Extent(512, 512)
        and extent.target == image.extent
        and extent.scale == 2
    )


def test_prepare_no_downscale():
    image = Image.create(Extent(1536, 1536))
    extent, result, _ = workflow.prepare_image(image, SDVersion.sd15, downscale=False)
    assert (
        extent.requires_upscale is False
        and result == image
        and extent.initial == image.extent
        and extent.target == image.extent
        and extent.scale == 1
    )


def test_merge_prompt():
    assert workflow.merge_prompt("a", "b") == "a, b"
    assert workflow.merge_prompt("", "b") == "b"
    assert workflow.merge_prompt("a", "") == "a"
    assert workflow.merge_prompt("", "") == ""
    assert workflow.merge_prompt("a", "b {prompt} c") == "b a c"
    assert workflow.merge_prompt("", "b {prompt} c") == "b  c"


def test_parse_lora(comfy):
    client_loras = [
        "/path/to/Lora-One.safetensors",
        "Lora-two.safetensors",
    ]

    assert workflow._parse_loras(client_loras, "a ship") == []
    assert workflow._parse_loras(client_loras, "a ship <lora:lora-one>") == [
        {"name": client_loras[0], "strength": 1.0}
    ]
    assert workflow._parse_loras(client_loras, "a ship <lora:LoRA-one>") == [
        {"name": client_loras[0], "strength": 1.0}
    ]
    assert workflow._parse_loras(client_loras, "a ship <lora:lora-one:0.0>") == [
        {"name": client_loras[0], "strength": 0.0}
    ]
    assert workflow._parse_loras(client_loras, "a ship <lora:lora-two:0.5>") == [
        {"name": client_loras[1], "strength": 0.5}
    ]
    assert workflow._parse_loras(client_loras, "a ship <lora:lora-two:-1.0>") == [
        {"name": client_loras[1], "strength": -1.0}
    ]

    try:
        workflow._parse_loras(client_loras, "a ship <lora:lora-three>")
    except Exception as e:
        assert str(e).startswith("LoRA not found")

    try:
        workflow._parse_loras(client_loras, "a ship <lora:lora-one:test-invalid-str>")
    except Exception as e:
        assert str(e).startswith("Invalid LoRA strength")


@pytest.mark.parametrize("extent", [Extent(256, 256), Extent(800, 800), Extent(512, 1024)])
def test_generate(qtapp, comfy, temp_settings, extent):
    temp_settings.batch_size = 1
    prompt = Conditioning("ship")

    async def main():
        job = workflow.generate(comfy, default_style(comfy), extent, prompt)
        results = await receive_images(comfy, job)
        results[0].save(result_dir / f"test_generate_{extent.width}x{extent.height}.png")
        assert results[0].extent == extent

    qtapp.run(main())


def test_inpaint(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 3  # max 3 images@512x512 -> 2 images@768x512
    image = Image.load(image_dir / "beach_768x512.png")
    mask = Mask.rectangle(Bounds(50, 100, 320, 200), feather=10)
    prompt = Conditioning("ship")
    job = workflow.inpaint(comfy, default_style(comfy), image, mask, prompt)

    async def main():
        results = await receive_images(comfy, job)
        assert len(results) == 2
        for i, result in enumerate(results):
            result.save(result_dir / f"test_inpaint_{i}.png")
            assert result.extent == Extent(320, 200)

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_inpaint_upscale(qtapp, comfy, temp_settings, sdver):
    temp_settings.batch_size = 3  # 2 images for 1.5, 1 image for XL
    image = Image.load(image_dir / "beach_1536x1024.png")
    mask = Mask.rectangle(Bounds(600, 200, 768, 512), feather=10)
    prompt = Conditioning("ship")
    job = workflow.inpaint(comfy, default_style(comfy, sdver), image, mask, prompt)

    async def main():
        results = await receive_images(comfy, job)
        assert len(results) == 2 if sdver == SDVersion.sd15 else 1
        for i, result in enumerate(results):
            result.save(result_dir / f"test_inpaint_upscale_{sdver.name}_{i}.png")
            assert result.extent == mask.bounds.extent

    qtapp.run(main())


def test_inpaint_odd_resolution(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "beach_768x512.png")
    image = Image.scale(image, Extent(612, 513))
    mask = Mask.rectangle(Bounds(0, 0, 200, 513))
    prompt = Conditioning()

    async def main():
        job = workflow.inpaint(comfy, default_style(comfy), image, mask, prompt)
        results = await receive_images(comfy, job)
        results[0].save(result_dir / "test_inpaint_odd_resolution.png")
        assert results[0].extent == mask.bounds.extent

    qtapp.run(main())


def test_inpaint_area_conditioning(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "lake_1536x1024.png")
    mask = Mask.load(image_dir / "lake_1536x1024_mask_bottom_right.png")
    prompt = Conditioning("crocodile")
    job = workflow.inpaint(comfy, default_style(comfy), image, mask, prompt)

    async def main():
        await run_and_save(comfy, job, "test_inpaint_area_conditioning.png")

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_refine(qtapp, comfy, sdver, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "beach_768x512.png")
    prompt = Conditioning("painting in the style of Vincent van Gogh")
    strength = {SDVersion.sd15: 0.5, SDVersion.sdxl: 0.65}[sdver]

    async def main():
        job = workflow.refine(comfy, default_style(comfy, sdver), image, prompt, strength)
        results = await receive_images(comfy, job)
        results[0].save(result_dir / f"test_refine_{sdver.name}.png")
        assert results[0].extent == image.extent

    qtapp.run(main())


def test_refine_region(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "lake_region.png")
    mask = Mask.load(image_dir / "lake_region_mask.png")
    prompt = Conditioning("waterfall")

    async def main():
        job = workflow.refine_region(comfy, default_style(comfy), image, mask, prompt, 0.6)
        results = await receive_images(comfy, job)
        results[0].save(result_dir / "test_refine_region.png")
        assert results[0].extent == mask.bounds.extent

    qtapp.run(main())


@pytest.mark.parametrize(
    "op", ["generate", "inpaint", "refine", "refine_region", "inpaint_upscale"]
)
def test_control_scribble(qtapp, comfy, temp_settings, op):
    temp_settings.batch_size = 1
    style = default_style(comfy)
    scribble_image = Image.load(image_dir / "owls_scribble.png")
    inpaint_image = Image.load(image_dir / "owls_inpaint.png")
    mask = Mask.load(image_dir / "owls_mask.png")
    mask.bounds = Bounds(256, 0, 256, 512)
    control = Conditioning("owls", "", [Control(ControlMode.scribble, scribble_image)])

    if op == "generate":
        job = workflow.generate(comfy, style, Extent(512, 512), control)
    elif op == "inpaint":
        control.area = Bounds(322, 108, 144, 300)
        job = workflow.inpaint(comfy, style, inpaint_image, mask, control)
    elif op == "refine":
        job = workflow.refine(comfy, style, inpaint_image, control, 0.7)
    elif op == "refine_region":
        cropped_image = Image.crop(inpaint_image, mask.bounds)
        job = workflow.refine_region(comfy, style, cropped_image, mask, control, 0.7)
    else:  # op == "inpaint_upscale":
        control.control[0].image = Image.scale(scribble_image, Extent(1024, 1024))
        control.area = Bounds(322 * 2, 108 * 2, 144 * 2, 300 * 2)
        upscaled_image = Image.scale(inpaint_image, Extent(1024, 1024))
        upscaled_mask = Mask(
            Bounds(512, 0, 512, 1024), Image.scale(Image(mask.image), Extent(512, 1024))._qimage
        )
        job = workflow.inpaint(comfy, style, upscaled_image, upscaled_mask, control)

    async def main():
        await run_and_save(comfy, job, f"test_control_scribble_{op}.png")

    qtapp.run(main())


@pytest.mark.parametrize("mode", [m for m in ControlMode if m.has_preprocessor])
def test_create_control_image(qtapp, comfy, mode):
    image_name = f"test_create_control_image_{mode.name}.png"
    image = Image.load(image_dir / "adobe_stock.jpg")
    job = workflow.create_control_image(comfy, image, mode)

    async def main():
        result = await run_and_save(comfy, job, image_name)
        reference = Image.load(reference_dir / image_name)
        threshold = 0.005 if mode is ControlMode.pose else 0.002
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
                reference = (reference_dir / image_name).read_text()
                assert result == reference
                return
            if msg.event is ClientEvent.error and msg.job_id == job_id:
                raise Exception(msg.error)
        assert False, "Connection closed without receiving images"

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_ip_adapter(qtapp, comfy, temp_settings, sdver):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "cat.png")
    control = Conditioning(
        "cat on a rooftop in paris", "", [Control(ControlMode.image, image, 0.6)]
    )
    extent = Extent(512, 512) if sdver == SDVersion.sd15 else Extent(1024, 1024)
    job = workflow.generate(comfy, default_style(comfy, sdver), extent, control)

    async def main():
        await run_and_save(comfy, job, f"test_ip_adapter_{sdver.name}.png")

    qtapp.run(main())


def test_ip_adapter_region(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "flowers.png")
    mask = Mask.load(image_dir / "flowers_mask.png")
    control_img = Image.load(image_dir / "pegonia.png")
    control = Conditioning("potted flowers", "", [Control(ControlMode.image, control_img, 0.7)])
    job = workflow.refine_region(comfy, default_style(comfy), image, mask, control, 0.6)

    async def main():
        await run_and_save(comfy, job, "test_ip_adapter_region.png")

    qtapp.run(main())


def test_ip_adapter_batch(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image1 = Image.load(image_dir / "cat.png")
    image2 = Image.load(image_dir / "pegonia.png")
    control = Conditioning(
        "", "", [Control(ControlMode.image, image1, 1.0), Control(ControlMode.image, image2, 1.0)]
    )
    job = workflow.generate(comfy, default_style(comfy), Extent(512, 512), control)

    async def main():
        await run_and_save(comfy, job, "test_ip_adapter_batch.png")

    qtapp.run(main())


def test_upscale_simple(qtapp, comfy):
    image = Image.load(image_dir / "beach_768x512.png")
    job = workflow.upscale_simple(comfy, image, comfy.default_upscaler, 2.0)

    async def main():
        await run_and_save(comfy, job, "test_upscale_simple.png")

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_upscale_tiled(qtapp, comfy, sdver):
    image = Image.load(image_dir / "beach_768x512.png")
    job = workflow.upscale_tiled(
        comfy, image, comfy.default_upscaler, 2.0, default_style(comfy, sdver), 0.5
    )

    async def main():
        await run_and_save(comfy, job, f"test_upscale_tiled_{sdver.name}.png")

    qtapp.run(main())


def test_generate_live(qtapp, comfy):
    scribble = Image.load(image_dir / "owls_scribble.png")
    live = LiveParams(is_active=True, seed=1234)
    cond = Conditioning("owls", "", [Control(ControlMode.scribble, scribble)])
    job = workflow.generate(comfy, default_style(comfy), Extent(512, 512), cond, live)

    async def main():
        await run_and_save(comfy, job, "test_generate_live.png")

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd15, SDVersion.sdxl])
def test_refine_live(qtapp, comfy, sdver):
    image = Image.load(image_dir / "pegonia.png")
    if sdver is SDVersion.sdxl:
        image = Image.scale(image, Extent(1024, 1024))  # result will be a bit blurry
    live = LiveParams(is_active=True, seed=1234)
    job = workflow.refine(comfy, default_style(comfy, sdver), image, Conditioning(), 0.4, live)

    async def main():
        await run_and_save(comfy, job, f"test_refine_live_{sdver.name}.png")

    qtapp.run(main())
