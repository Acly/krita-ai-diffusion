import json
from tempfile import TemporaryDirectory
from pathlib import Path

from ai_diffusion.settings import PerformancePreset, Settings, Setting, ServerMode
from ai_diffusion.style import Style, Styles, StyleSettings


def test_get_set():
    s = Settings()
    assert (
        s.history_size == Settings._history_size.default
        and s.server_mode == Settings._server_mode.default
    )
    s.history_size = 5
    s.server_mode = ServerMode.external
    assert s.history_size == 5 and s.server_mode == ServerMode.external


def test_restore():
    s = Settings()
    s.history_size = 5
    s.server_mode = ServerMode.external
    s.restore()
    assert (
        s.history_size == Settings._history_size.default
        and s.server_mode == Settings._server_mode.default
    )


def test_save():
    original = Settings()
    original.history_size = 5
    original.server_mode = ServerMode.external
    original.performance_preset = PerformancePreset.low
    result = Settings()
    with TemporaryDirectory(dir=Path(__file__).parent) as dir:
        filepath = Path(dir) / "test_settings.json"
        original.save(filepath)
        result.load(filepath)
    assert (
        result.history_size == 5
        and result.server_mode == ServerMode.external
        and result.performance_preset == PerformancePreset.low
    )


def test_performance_preset():
    s = Settings()
    s.performance_preset = PerformancePreset.low
    assert s.batch_size == 2 and s.diffusion_tile_size == 1024


def style_is_default(style):
    return all([
        getattr(style, name) == s.default
        for name, s in StyleSettings.__dict__.items()
        if isinstance(s, Setting) and name != "name"
    ])


def test_styles():
    with TemporaryDirectory(dir=Path(__file__).parent) as dir:
        style = Style(Path(dir) / "test_style.json")
        style.name = "Test Style"
        style.save()

        styles = Styles(Path(dir))
        assert len(styles) == 1
        loaded_style = styles[0]
        assert loaded_style.filename == style.filename
        assert loaded_style.name == "Test Style"
        assert styles.find(style.filename) == (loaded_style, 0)
        assert styles.find("nonexistent.json") == (None, -1)
        assert style_is_default(loaded_style)


def test_bad_style_file():
    with TemporaryDirectory(dir=Path(__file__).parent) as dir:
        path = Path(dir) / "test_style.json"
        path.write_text("bad json")
        styles = Styles(Path(dir))
        assert len(styles) == 1  # no error, default style inserted
        assert style_is_default(styles[0])


def test_bad_style_type():
    with TemporaryDirectory(dir=Path(__file__).parent) as dir:
        path = Path(dir) / "test_style.json"
        path.write_text(json.dumps({"cfg_scale": "bad", "sampler": "bad", "style_prompt": -1}))
        style = Style.load(path)
        assert (
            style is not None
            and style.cfg_scale == StyleSettings.cfg_scale.default
            and style.sampler == StyleSettings.sampler.default
            and style.style_prompt == StyleSettings.style_prompt.default
        )


def test_default_style():
    with TemporaryDirectory(dir=Path(__file__).parent) as dir:
        styles = Styles(Path(dir))
        style = styles.default
        assert style_is_default(style)
