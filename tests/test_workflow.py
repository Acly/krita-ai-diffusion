import pytest
from ai_tools import (
    settings,
    workflow,
    Mask,
    Bounds,
    Extent,
    Image,
    Progress,
    ImageCollection,
    Auto1111,
)
from pathlib import Path

test_dir = Path(__file__).parent
image_dir = test_dir / "images"
result_dir = test_dir / ".results"


def check_progress(value):
    assert value >= 0 and value <= 1


async def expect_one(async_generator):
    results = [x async for x in async_generator]
    assert len(results) == 1
    return results[0]


@pytest.fixture()
def auto1111(qtapp):
    return qtapp.run(Auto1111.connect())


@pytest.mark.parametrize(
    "input,expected,scale",
    [(Extent(1536, 600), Extent(768, 300), 1 / 2), (Extent(300, 1024), Extent(225, 768), 3 / 4)],
)
def test_prepare_highres(input, expected, scale):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    progress = Progress(check_progress)
    result = workflow.prepare((image, mask), progress)
    assert (
        result.image.extent == expected
        and result.mask_image.extent == expected
        and result.extent.initial == expected
        and result.extent.target == input
        and result.extent.scale == scale
        and result.progress.scale == 0.5
    )


@pytest.mark.parametrize(
    "input,expected,scale",
    [(Extent(256, 256), Extent(512, 512), 2), (Extent(128, 450), Extent(512, 1800), 4)],
)
def test_prepare_lowres(input, expected, scale):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    progress = Progress(check_progress)
    result = workflow.prepare((image, mask), progress)
    assert (
        result.image.extent == input
        and result.mask_image.extent == input
        and result.extent.target == input
        and result.extent.initial == expected
        and result.extent.scale == scale
        and result.progress.scale == 1
    )


@pytest.mark.parametrize(
    "input",
    [Extent(512, 512), Extent(128, 600), Extent(768, 200)],
)
def test_prepare_passthrough(input):
    image = Image.create(input)
    mask = Mask.rectangle(Bounds(0, 0, input.width, input.height))
    progress = Progress(check_progress)
    result = workflow.prepare((image, mask), progress)
    assert (
        result.image == image
        and result.mask_image.extent == input
        and result.extent.initial == input
        and result.extent.target == input
        and result.extent.scale == 1
        and result.progress.scale == 1
    )


def test_prepare_extent():
    input = Extent(1024, 1536)
    progress = Progress(check_progress)
    result = workflow.prepare(input, progress)
    assert (
        result.image is None
        and result.mask_image is None
        and result.extent.initial == Extent(512, 768)
        and result.extent.target == input
        and result.extent.scale == 1 / 2
    )


def test_prepare_no_mask():
    image = Image.create(Extent(256, 256))
    progress = Progress(check_progress)
    result = workflow.prepare(image, progress)
    assert (
        result.image == image
        and result.mask_image is None
        and result.extent.initial == Extent(512, 512)
        and result.extent.target == image.extent
        and result.extent.scale == 2
    )


def test_prepare_no_downscale():
    image = Image.create(Extent(1536, 1536))
    progress = Progress(check_progress)
    result = workflow.prepare(image, progress, downscale=False)
    assert (
        result.image == image
        and result.mask_image is None
        and result.extent.initial == image.extent
        and result.extent.target == image.extent
        and result.extent.scale == 1
    )


@pytest.mark.parametrize(
    "target", [Extent(256, 256), Extent(256, 192), Extent(1024, 512), Extent(512, 256)]
)
def test_post(qtapp, auto1111, target):
    image = Image.create(Extent(512, 256))
    progress = Progress(check_progress)

    async def main():
        return await workflow.postprocess(auto1111, image, target, "", progress)

    result = qtapp.run(main())
    assert result.extent == target


@pytest.mark.parametrize("extent", [Extent(256, 256), Extent(512, 1024)])
def test_generate(qtapp, auto1111, temp_settings, extent):
    temp_settings.batch_size = 1

    async def main():
        result = await expect_one(
            workflow.generate(auto1111, extent, "ship", Progress(check_progress))
        )
        result.save(result_dir / f"test_generate_{extent.width}x{extent.height}.png")
        assert result.extent == extent

    qtapp.run(main())


def test_inpaint(qtapp, auto1111, temp_settings):
    temp_settings.batch_size = 3  # max 3 images@512x512 -> 2 images@768x512
    image = Image.load(image_dir / "beach_768x512.png")
    mask = Mask.rectangle(Bounds(50, 100, 320, 200), feather=10)

    async def main():
        index = 0
        results = workflow.inpaint(auto1111, image, mask, "ship", Progress(check_progress))
        async for result in results:
            result.save(result_dir / f"test_inpaint_{index}.png")
            assert result.extent == Extent(320, 200)
            index += 1
        assert index == 2

    qtapp.run(main())


def test_inpaint_upscale(qtapp, auto1111, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "beach_1536x1024.png")
    mask = Mask.rectangle(Bounds(600, 200, 768, 512), feather=10)

    async def main():
        result = await expect_one(
            workflow.inpaint(auto1111, image, mask, "ship", Progress(check_progress))
        )
        result.save(result_dir / "test_inpaint_upscale.png")
        assert result.extent == mask.bounds.extent

    qtapp.run(main())


def test_refine(qtapp, auto1111, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "beach_768x512.png")
    prompt = "in the style of vermeer, van gogh"

    async def main():
        result = await expect_one(
            workflow.refine(auto1111, image, prompt, 0.6, Progress(check_progress))
        )
        result.save(result_dir / "test_refine.png")
        assert result.extent == image.extent

    qtapp.run(main())


def test_refine_region(qtapp, auto1111, temp_settings):
    temp_settings.batch_size = 1
    image = Image.load(image_dir / "lake_1536x1024.png")
    mask = Mask.rectangle(Bounds(760, 240, 525, 375), feather=16)

    async def main():
        result = await expect_one(
            workflow.refine_region(
                auto1111, image, mask, "waterfall", 0.5, Progress(check_progress)
            )
        )
        result.save(result_dir / "test_refine_region.png")
        assert result.extent == mask.bounds.extent

    qtapp.run(main())
