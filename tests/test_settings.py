from ai_tools import Settings
from tempfile import TemporaryDirectory
from pathlib import Path


def test_get_set():
    s = Settings()
    assert (
        s.min_image_size == Settings._min_image_size.default
        and s.max_image_size == Settings._max_image_size.default
    )
    s.min_image_size = 5
    s.max_image_size = 99
    assert s.min_image_size == 5 and s.max_image_size == 99


def test_upscaler_index():
    s = Settings()
    assert s.upscaler == Settings._upscaler.default
    s.upscalers = ["A", Settings._upscaler.default, "B", "C"]
    assert s.upscaler_index == 1
    s.upscaler = "B"
    assert s.upscaler_index == 2


def test_restore():
    s = Settings()
    s.min_image_size = 5
    s.max_image_size = 99
    s.restore()
    assert (
        s.min_image_size == Settings._min_image_size.default
        and s.max_image_size == Settings._max_image_size.default
    )


def test_save():
    original = Settings()
    original.min_image_size = 5
    original.max_image_size = 99
    result = Settings()
    with TemporaryDirectory(dir=Path(__file__).parent) as dir:
        filepath = Path(dir) / "test_settings.json"
        original.save(filepath)
        result.load(filepath)
    assert result.min_image_size == 5 and result.max_image_size == 99
