from typing import List
import pytest
from ai_diffusion import (
    settings,
    workflow,
    ComfyWorkflow,
    Mask,
    Bounds,
    Extent,
    Image,
    Client,
    ClientEvent,
    SDVersion,
    Style,
)
from pathlib import Path

test_dir = Path(__file__).parent
image_dir = test_dir / "images"
result_dir = test_dir / ".results"
default_checkpoint = {
    SDVersion.sd1_5: "realisticVisionV51_v51VAE.safetensors",
    SDVersion.sdxl: "sdXL_v10VAEFix.safetensors",
}


@pytest.fixture(scope="session", autouse=True)
def clear_results():
    if result_dir.exists():
        for file in result_dir.iterdir():
            file.unlink()
    result_dir.mkdir(exist_ok=True)


@pytest.fixture()
def comfy(qtapp):
    return qtapp.run(Client.connect())


def default_style(comfy, sd_ver=SDVersion.sd1_5):
    version_checkpoints = [c for c in comfy.checkpoints if sd_ver.matches(c)]
    checkpoint = default_checkpoint[sd_ver]

    style = Style(Path("default.json"))
    style.sd_checkpoint = (
        checkpoint if checkpoint in version_checkpoints else version_checkpoints[0]
    )
    return style


async def receive_images(comfy, workflow: ComfyWorkflow):
    job_id = await comfy.enqueue(workflow)
    async for msg in comfy.listen():
        if msg.event is ClientEvent.finished and msg.job_id == job_id:
            return msg.images
    assert False, "Connection closed without receiving images"


@pytest.mark.parametrize(
    "input,expected,scale",
    [
        (Extent(1536, 600), Extent(1008, 392), 0.6532),
        (Extent(400, 1024), Extent(392, 1008), 0.9798),
    ],
)
def test_prepare_highres(input, expected, scale):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    result = workflow.prepare((image, mask), SDVersion.sd1_5)
    assert (
        result.image.extent == expected
        and result.mask_image.extent == expected
        and result.extent.initial == expected
        and result.extent.target == input
        and result.extent.scale == pytest.approx(scale, abs=1e-3)
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
    result = workflow.prepare((image, mask), SDVersion.sd1_5)
    assert (
        result.image.extent == input
        and result.mask_image.extent == input
        and result.extent.target == input
        and result.extent.initial == expected
        and result.extent.scale > 1
    )


@pytest.mark.parametrize(
    "input",
    [Extent(512, 512), Extent(128, 600), Extent(768, 240)],
)
def test_prepare_passthrough(input):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    result = workflow.prepare((image, mask), SDVersion.sd1_5)
    assert (
        result.image == image
        and result.mask_image.extent == input
        and result.extent.initial == input
        and result.extent.target == input
        and result.extent.scale == 1
    )


@pytest.mark.parametrize(
    "input,expected", [(Extent(512, 513), Extent(512, 520)), (Extent(300, 1024), Extent(304, 1024))]
)
def test_prepare_multiple8(input, expected):
    result = workflow.prepare(input, SDVersion.sd1_5)
    assert result.extent.initial == expected and result.extent.target == input


@pytest.mark.parametrize("sdver", [SDVersion.sd1_5, SDVersion.sdxl])
def test_prepare_extent(sdver: SDVersion):
    input = Extent(1024, 1536)
    result = workflow.prepare(input, sdver)
    expected = Extent(512, 768) if sdver == SDVersion.sd1_5 else Extent(840, 1256)
    assert (
        result.image is None
        and result.mask_image is None
        and result.extent.initial == expected
        and result.extent.target == input
        and result.extent.scale < 1
    )


def test_prepare_no_mask():
    image = Image.create(Extent(256, 256))
    result = workflow.prepare(image, SDVersion.sd1_5)
    assert (
        result.image == image
        and result.mask_image is None
        and result.extent.initial == Extent(512, 512)
        and result.extent.target == image.extent
        and result.extent.scale == 2
    )


def test_prepare_no_downscale():
    image = Image.create(Extent(1536, 1536))
    result = workflow.prepare(image, SDVersion.sd1_5, downscale=False)
    assert (
        result.image == image
        and result.mask_image is None
        and result.extent.initial == image.extent
        and result.extent.target == image.extent
        and result.extent.scale == 1
    )


@pytest.mark.parametrize("extent", [Extent(256, 256), Extent(512, 1024)])
def test_generate(qtapp, comfy, temp_settings, extent):
    temp_settings.batch_size = 1

    async def main():
        job = workflow.generate(comfy, default_style(comfy), extent, "ship")
        results = await receive_images(comfy, job)
        results[0].save(result_dir / f"test_generate_{extent.width}x{extent.height}.png")
        assert results[0].extent == extent

    qtapp.run(main())


def test_inpaint(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 3  # max 3 images@512x512 -> 2 images@768x512
    image = Image.load(image_dir / "beach_768x512.png")
    mask = Mask.rectangle(Bounds(50, 100, 320, 200), feather=10)

    async def main():
        job = workflow.inpaint(comfy, default_style(comfy), image, mask, "ship")
        results = await receive_images(comfy, job)
        assert len(results) == 2
        for i, result in enumerate(results):
            result.save(result_dir / f"test_inpaint_{i}.png")
            assert result.extent == Extent(320, 200)

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd1_5, SDVersion.sdxl])
def test_inpaint_upscale(qtapp, comfy, temp_settings, sdver):
    temp_settings.batch_size = 3  # 2 images for 1.5, 1 image for XL
    image = Image.load(image_dir / "beach_1536x1024.png")
    mask = Mask.rectangle(Bounds(600, 200, 768, 512), feather=10)

    async def main():
        job = workflow.inpaint(comfy, default_style(comfy, sdver), image, mask, "ship")
        results = await receive_images(comfy, job)
        assert len(results) == 2 if sdver == SDVersion.sd1_5 else 1
        for i, result in enumerate(results):
            result.save(result_dir / f"test_inpaint_upscale_{sdver.name}_{i}.png")
            assert result.extent == mask.bounds.extent

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [SDVersion.sd1_5, SDVersion.sdxl])
def test_refine(qtapp, comfy, sdver, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "beach_768x512.png")
    prompt = "painting in the style of Vincent van Gogh"
    strength = {SDVersion.sd1_5: 0.5, SDVersion.sdxl: 0.65}[sdver]

    async def main():
        job = workflow.refine(comfy, default_style(comfy, sdver), image, prompt, strength)
        results = await receive_images(comfy, job)
        results[0].save(result_dir / f"test_refine_{sdver.name}.png")
        assert results[0].extent == image.extent

    qtapp.run(main())


def test_refine_region(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "lake_1536x1024.png")
    mask = Mask.rectangle(Bounds(760, 240, 528, 376), feather=16)

    async def main():
        job = workflow.refine_region(comfy, default_style(comfy), image, mask, "waterfall", 0.5)
        results = await receive_images(comfy, job)
        results[0].save(result_dir / "test_refine_region.png")
        assert results[0].extent == mask.bounds.extent

    qtapp.run(main())
