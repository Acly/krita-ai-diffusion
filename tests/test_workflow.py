import pytest
from ai_tools import (
    settings,
    workflow,
    ComfyWorkflow,
    Mask,
    Bounds,
    Extent,
    Image,
    Client,
    ClientEvent,
)
from pathlib import Path

test_dir = Path(__file__).parent
image_dir = test_dir / "images"
result_dir = test_dir / ".results"
default_checkpoint = "realisticVisionV51_v51VAE.safetensors"


@pytest.fixture(scope="session", autouse=True)
def clear_results():
    if result_dir.exists():
        for file in result_dir.iterdir():
            file.unlink()
    result_dir.mkdir(exist_ok=True)


@pytest.fixture()
def comfy(qtapp):
    client = qtapp.run(Client.connect())
    if default_checkpoint in client.checkpoints:
        settings.sd_checkpoint = default_checkpoint
    return client


async def receive_images(comfy, workflow: ComfyWorkflow):
    job_id = await comfy.enqueue(workflow)
    async for msg in comfy.listen():
        if msg.event is ClientEvent.finished and msg.job_id == job_id:
            return msg.images
    assert False, "Connection closed without receiving images"


@pytest.mark.parametrize(
    "input,expected,scale",
    [(Extent(1536, 600), Extent(768, 304), 1 / 2), (Extent(300, 1024), Extent(232, 768), 3 / 4)],
)
def test_prepare_highres(input, expected, scale):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    result = workflow.prepare((image, mask))
    assert (
        result.image.extent == expected
        and result.mask_image.extent == expected
        and result.extent.initial == expected
        and result.extent.target == input
        and result.extent.scale == scale
    )


@pytest.mark.parametrize(
    "input,expected,scale",
    [
        (Extent(256, 256), Extent(512, 512), 2),
        (Extent(128, 450), Extent(512, 1800), 4),
        (Extent(256, 333), Extent(512, 672), 2),  # multiple of 8
    ],
)
def test_prepare_lowres(input, expected, scale):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    result = workflow.prepare((image, mask))
    assert (
        result.image.extent == input
        and result.mask_image.extent == input
        and result.extent.target == input
        and result.extent.initial == expected
        and result.extent.scale == scale
    )


@pytest.mark.parametrize(
    "input",
    [Extent(512, 512), Extent(128, 600), Extent(768, 240)],
)
def test_prepare_passthrough(input):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    result = workflow.prepare((image, mask))
    assert (
        result.image == image
        and result.mask_image.extent == input
        and result.extent.initial == input
        and result.extent.target == input
        and result.extent.scale == 1
    )


def test_prepare_multiple8():
    input = Extent(512, 513)
    result = workflow.prepare(input)
    assert result.extent.initial == Extent(512, 520) and result.extent.target == input


def test_prepare_extent():
    input = Extent(1024, 1536)
    result = workflow.prepare(input)
    assert (
        result.image is None
        and result.mask_image is None
        and result.extent.initial == Extent(512, 768)
        and result.extent.target == input
        and result.extent.scale == 1 / 2
    )


def test_prepare_no_mask():
    image = Image.create(Extent(256, 256))
    result = workflow.prepare(image)
    assert (
        result.image == image
        and result.mask_image is None
        and result.extent.initial == Extent(512, 512)
        and result.extent.target == image.extent
        and result.extent.scale == 2
    )


def test_prepare_no_downscale():
    image = Image.create(Extent(1536, 1536))
    result = workflow.prepare(image, downscale=False)
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
        job = workflow.generate(extent, "ship")
        results = await receive_images(comfy, job)
        results[0].save(result_dir / f"test_generate_{extent.width}x{extent.height}.png")
        assert results[0].extent == extent

    qtapp.run(main())


def test_inpaint(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 3  # max 3 images@512x512 -> 2 images@768x512
    image = Image.load(image_dir / "beach_768x512.png")
    mask = Mask.rectangle(Bounds(50, 100, 320, 200), feather=10)

    async def main():
        index = 0
        job = workflow.inpaint(comfy, image, mask, "ship")
        results = await receive_images(comfy, job)
        assert len(results) == 2
        for i, result in enumerate(results):
            result.save(result_dir / f"test_inpaint_{i}.png")
            assert result.extent == Extent(320, 200)

    qtapp.run(main())


def test_inpaint_upscale(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 3  # max 3 images@512x512 -> 2 images@768x512
    image = Image.load(image_dir / "beach_1536x1024.png")
    mask = Mask.rectangle(Bounds(600, 200, 768, 512), feather=10)

    async def main():
        job = workflow.inpaint(comfy, image, mask, "ship")
        results = await receive_images(comfy, job)
        assert len(results) == 2
        for i, result in enumerate(results):
            result.save(result_dir / f"test_inpaint_upscale_{i}.png")
            assert result.extent == mask.bounds.extent

    qtapp.run(main())


def test_refine(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "beach_768x512.png")
    prompt = "in the style of vermeer, van gogh"

    async def main():
        job = workflow.refine(image, prompt, 0.5)
        results = await receive_images(comfy, job)
        results[0].save(result_dir / "test_refine.png")
        assert results[0].extent == image.extent

    qtapp.run(main())


def test_refine_region(qtapp, comfy, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "lake_1536x1024.png")
    mask = Mask.rectangle(Bounds(760, 240, 528, 376), feather=16)

    async def main():
        job = workflow.refine_region(comfy, image, mask, "waterfall", 0.5)
        results = await receive_images(comfy, job)
        results[0].save(result_dir / "test_refine_region.png")
        assert results[0].extent == mask.bounds.extent

    qtapp.run(main())
