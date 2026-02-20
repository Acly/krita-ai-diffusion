from pathlib import Path

import pytest

from ai_diffusion import resolution, workflow
from ai_diffusion.api import InpaintMode
from ai_diffusion.image import Bounds, Extent, Image, Point
from ai_diffusion.resolution import CheckpointResolution, ScaledExtent, ScaleMode, TileLayout
from ai_diffusion.resources import Arch
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import Style

from .config import data_dir

dummy_style = Style(Path("dummy.json"))
perf = PerformanceSettings()


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
    assert resolution.compute_batch_size(extent, min_size, max_batches) == expected


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
    assert e.refinement_scaling is ScaleMode.upscale_small
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
    assert CheckpointResolution.compute(extent, Arch.sdxl, style) == expected


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
        assert result.to_numpy_format().pixel(*expected_crop) == (255, 255, 255, 255)
    else:
        assert mode is InpaintMode.fill and result is None


@pytest.mark.parametrize(
    "input,expected_initial,expected_desired",
    [
        (Extent(1536, 600), Extent(1008, 392), Extent(1536, 600)),
        (Extent(400, 1024), Extent(400, 1024), Extent(400, 1024)),
        (Extent(777, 999), Extent(560, 712), Extent(784, 1000)),
    ],
)
def test_prepare_highres(input, expected_initial, expected_desired):
    image = Image.create(input)
    r, _ = resolution.prepare_image(image, Arch.sd15, dummy_style, perf)
    assert (
        r.initial_image
        and r.extent.input == r.initial_image.extent
        and r.initial_image.extent == expected_initial
        and r.extent.initial == expected_initial
        and r.extent.desired == expected_desired
        and r.extent.target == input
    )


def test_prepare_hightres_inpaint():
    input = Extent(3000, 2000)
    image = Image.create(input)
    r, _ = resolution.prepare_image(image, Arch.flux, dummy_style, perf, inpaint=True)
    assert r.extent.initial == Extent(1256, 840)
    assert r.extent.desired == input


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
    r, _ = resolution.prepare_image(image, Arch.sd15, dummy_style, perf)
    assert (
        r.extent.input == input
        and image.extent == input
        and r.extent.target == input
        and r.extent.initial == expected
        and r.extent.desired == expected
    )


@pytest.mark.parametrize(
    "input",
    [Extent(512, 512), Extent(128, 600), Extent(768, 240)],
)
def test_prepare_passthrough(input: Extent):
    image = Image.create(input)
    r, _ = resolution.prepare_image(image, Arch.sd15, dummy_style, perf)
    assert (
        r.initial_image
        and r.initial_image == image
        and r.extent.input == input
        and r.extent.initial == input
        and r.extent.target == input
        and r.extent.desired == input
    )


@pytest.mark.parametrize(
    "input,expected", [(Extent(512, 513), Extent(512, 520)), (Extent(300, 1024), Extent(304, 1024))]
)
def test_prepare_multiple8(input: Extent, expected: Extent):
    r, _ = resolution.prepare_extent(input, Arch.sd15, dummy_style, perf)
    assert (
        r.extent.input == input
        and r.extent.initial == expected
        and r.extent.target == input
        and r.extent.desired == input.multiple_of(8)
    )


@pytest.mark.parametrize("sdver", [Arch.sd15, Arch.sdxl])
def test_prepare_extent(sdver: Arch):
    input = Extent(1024, 1536)
    r, _ = resolution.prepare_extent(input, sdver, dummy_style, perf)
    expected = Extent(512, 768) if sdver == Arch.sd15 else Extent(840, 1256)
    assert r.extent.initial == expected and r.extent.desired == input and r.extent.target == input


def test_prepare_no_mask():
    image = Image.create(Extent(256, 256))
    r, _ = resolution.prepare_image(image, Arch.sd15, dummy_style, perf)
    assert (
        r.initial_image
        and r.initial_image == image
        and r.extent.initial == Extent(512, 512)
        and r.extent.target == image.extent
    )


@pytest.mark.parametrize("input", [Extent(512, 512), Extent(1024, 1628), Extent(1536, 999)])
def test_prepare_no_downscale(input: Extent):
    image = Image.create(input)
    r, _ = resolution.prepare_image(image, Arch.sd15, dummy_style, perf, downscale=False)
    assert (
        r.initial_image
        and r.initial_image == image
        and r.extent.initial == input.multiple_of(8)
        and r.extent.desired == input.multiple_of(8)
        and r.extent.target == input
    )


@pytest.mark.parametrize(
    "sd_ver,input,expected_initial,expected_desired",
    [
        (Arch.sd15, Extent(2000, 2000), (632, 632), (1000, 1000)),
        (Arch.sd15, Extent(1000, 1000), (632, 632), (1000, 1000)),
        (Arch.sdxl, Extent(1024, 1024), (1024, 1024), (1024, 1024)),
        (Arch.sdxl, Extent(2000, 2000), (1000, 1000), (1000, 1000)),
        (Arch.sd15, Extent(801, 801), (632, 632), (808, 808)),
    ],
    ids=["sd15_large", "sd15_small", "sdxl_small", "sdxl_large", "sd15_odd"],
)
def test_prepare_max_pixel_count(input, sd_ver, expected_initial, expected_desired):
    perf_settings = PerformanceSettings(max_pixel_count=1)
    r, _ = resolution.prepare_extent(input, sd_ver, dummy_style, perf_settings)
    assert (
        r.extent.initial == expected_initial
        and r.extent.desired == expected_desired
        and r.extent.target == input
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
def test_prepare_resolution_multiplier(input, multiplier, expected_initial, expected_desired):
    perf_settings = PerformanceSettings(resolution_multiplier=multiplier)
    r, _ = resolution.prepare_extent(input, Arch.sd15, dummy_style, perf_settings)
    assert (
        r.extent.initial == expected_initial
        and r.extent.desired == expected_desired
        and r.extent.target == input
    )


@pytest.mark.parametrize("multiplier", [0.5, 0.2])
def test_prepare_resolution_multiplier_inputs(multiplier):
    perf_settings = PerformanceSettings(resolution_multiplier=multiplier)
    input = Extent(1024, 1024)
    image = Image.create(input)
    r, _ = resolution.prepare_image(image, Arch.sd15, dummy_style, perf_settings)
    assert (
        r.extent.input == Extent(512, 512)
        and r.initial_image
        and r.initial_image.extent == Extent(512, 512)
        and r.extent.initial == Extent(512, 512)
        and r.extent.desired == Extent(512, 512)
        and r.extent.target == Extent(1024, 1024)
    )


@pytest.mark.parametrize(
    "multiplier,expected",
    [(0.5, Extent(1024, 1024)), (2, Extent(1000, 1000)), (0.25, Extent(512, 512))],
)
def test_prepare_resolution_multiplier_max(multiplier, expected):
    perf_settings = PerformanceSettings(resolution_multiplier=multiplier, max_pixel_count=1)
    input = Extent(2048, 2048)
    r, _ = resolution.prepare_extent(input, Arch.sd15, dummy_style, perf_settings)
    assert r.extent.initial.width <= 632 and r.extent.desired == expected


tile_layouts = {
    "1024-512": {
        "extent": Extent(1024, 1024),
        "min_tile_size": 512,
        "padding": 32,
        "tile_count": (2, 2),
        "tile_size": (544, 544),
        "tiles": [
            {"start": Point(0, 0), "end": Point(544, 544)},
            {"start": Point(0, 480), "end": Point(544, 1024)},
            {"start": Point(480, 0), "end": Point(1024, 544)},
            {"start": Point(480, 480), "end": Point(1024, 1024)},
        ],
    },
    "2240-1024": {
        "extent": Extent(2880, 2240),
        "min_tile_size": 1024,
        "padding": 48,
        "tile_count": (3, 2),
        "tile_size": (1024, 1168),
        "tiles": [
            {"start": Point(0, 0), "end": Point(1024, 1168)},
            {"start": Point(0, 1072), "end": Point(1024, 2240)},
            {"start": Point(928, 0), "end": Point(1952, 1168)},
            {"start": Point(928, 1072), "end": Point(1952, 2240)},
            {"start": Point(1856, 0), "end": Point(2880, 1168)},
            {"start": Point(1856, 1072), "end": Point(2880, 2240)},
        ],
    },
    "single-tile": {
        "extent": Extent(800, 800),
        "min_tile_size": 896,
        "padding": 32,
        "tile_count": (1, 1),
        "tile_size": (800, 800),
        "tiles": [{"start": Point(0, 0), "end": Point(800, 800)}],
    },
    "no-overlap": {
        "extent": Extent(1024, 1024),
        "min_tile_size": 512,
        "padding": 0,
        "tile_count": (2, 2),
        "tile_size": (512, 512),
        "tiles": [
            {"start": Point(0, 0), "end": Point(512, 512)},
            {"start": Point(0, 512), "end": Point(512, 1024)},
            {"start": Point(512, 0), "end": Point(1024, 512)},
            {"start": Point(512, 512), "end": Point(1024, 1024)},
        ],
    },
    "multiple-16": {
        "extent": Extent(1472, 512),
        "min_tile_size": 512,
        "padding": 32,
        "multiple": 16,
        "tile_count": (3, 1),
        "tile_size": (544, 512),
        "tiles": [
            {"start": Point(0, 0), "end": Point(544, 512)},
            {"start": Point(480, 0), "end": Point(1024, 512)},
            {"start": Point(960, 0), "end": Point(1472, 512)},
        ],
    },
}


@pytest.mark.parametrize("test_set", tile_layouts.keys())
def test_tile_layout(test_set):
    params = tile_layouts[test_set]
    layout = TileLayout(
        params["extent"], params["min_tile_size"], params["padding"], params.get("multiple", 8)
    )
    assert layout.tile_count == params["tile_count"]
    assert layout.tile_extent == params["tile_size"]
    assert layout.total_tiles == len(params["tiles"])
    for i in range(layout.total_tiles):
        expected = params["tiles"][i]
        coord = layout.coord(i)
        assert layout.start(coord) == expected["start"]
        assert layout.end(coord) == expected["end"]
