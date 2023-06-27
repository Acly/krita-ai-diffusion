import pytest
from ai_tools import (
    Settings,
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
settings = Settings()


def check_progress(value):
    assert value >= 0 and value <= 1


@pytest.fixture()
def auto1111(qtapp):
    return qtapp.run(Auto1111.connect())


@pytest.mark.parametrize(
    "input,expected,scale",
    [(Extent(1536, 600), Extent(768, 300), 1 / 2), (Extent(300, 1024), Extent(225, 768), 3 / 4)],
)
def test_prepare_upscale(input, expected, scale):
    image = Image.create(input)
    mask = Image.create(input)
    progress = Progress(check_progress)
    result = workflow.prepare(image, mask, progress)
    assert (
        result.image.extent == expected
        and result.mask.extent == expected
        and result.target_extent == expected
        and result.scale == scale
        and result.progress.scale == 1 / (1 + settings.batch_size)
    )


@pytest.mark.parametrize(
    "input,expected,scale",
    [(Extent(256, 256), Extent(512, 512), 2), (Extent(128, 450), Extent(512, 1800), 4)],
)
def test_prepare_downscale(input, expected, scale):
    image = Image.create(input)
    mask = Image.create(input)
    progress = Progress(check_progress)
    result = workflow.prepare(image, mask, progress)
    assert (
        result.image.extent == input
        and result.mask.extent == input
        and result.target_extent == expected
        and result.scale == scale
        and result.progress.scale == 1
    )


@pytest.mark.parametrize(
    "input",
    [Extent(512, 512), Extent(128, 600), Extent(768, 200)],
)
def test_prepare_passthrough(input):
    image = Image.create(input)
    mask = Image.create(input)
    progress = Progress(check_progress)
    result = workflow.prepare(image, mask, progress)
    assert (
        result.image == image
        and result.mask == mask
        and result.target_extent == input
        and result.scale == 1
        and result.progress.scale == 1
    )


@pytest.mark.parametrize(
    "target", [Extent(256, 256), Extent(256, 192), Extent(1024, 512), Extent(512, 256)]
)
def test_post(qtapp, auto1111, target):
    images = ImageCollection([Image.create(Extent(512, 256))])
    progress = Progress(check_progress)

    async def main():
        return await workflow.postprocess(auto1111, images, target, "", progress)

    result = qtapp.run(main())
    assert result[0].extent == target


def test_generate(qtapp, auto1111):
    settings.batch_size = 2
    image = Image.load(image_dir / "beach_768x512.png")
    mask = Mask.rectangle(Bounds(50, 100, 320, 200), feather=10)

    async def main():
        result = await workflow.generate(auto1111, image, mask, "ship", Progress(check_progress))
        result.save(result_dir / "test_generate.png")

    qtapp.run(main())


def test_generate_upscale(qtapp, auto1111):
    settings.batch_size = 2
    image = Image.load(image_dir / "beach_1536x1024.png")
    mask = Mask.rectangle(Bounds(600, 200, 768, 512), feather=10)

    async def main():
        result = await workflow.generate(auto1111, image, mask, "ship", Progress(check_progress))
        result.save(result_dir / "test_generate_upscale.png")

    qtapp.run(main())


def test_refine(qtapp, auto1111):
    settings.batch_size = 2
    image = Image.load(image_dir / "lake_1536x1024.png")
    mask = Mask.rectangle(Bounds(760, 240, 525, 375), feather=16)

    async def main():
        result = await workflow.refine(
            auto1111, image, mask, "waterfall", 0.6, Progress(check_progress)
        )
        result.save(result_dir / "test_refine.png")

    qtapp.run(main())
