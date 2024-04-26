import json
from tempfile import TemporaryDirectory
from pathlib import Path

from ai_diffusion.settings import PerformancePreset, Settings, Setting, ServerMode
from ai_diffusion.style import Style, Styles, StyleSettings, SamplerPreset, SamplerPresets
from ai_diffusion import style


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
    assert s.server_mode == Settings._server_mode.default

    s.history_size = 5
    s.server_mode = ServerMode.external
    s.restore()
    assert s.history_size == Settings._history_size.default and s.server_mode is ServerMode.managed


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
    assert s.batch_size == 2 and s.max_pixel_count == 2 and s.resolution_multiplier == 1.0


def style_is_default(style):
    return all(
        [
            getattr(style, name) == s.default
            for name, s in StyleSettings.__dict__.items()
            if isinstance(s, Setting) and name != "name"
        ]
    )


def test_styles(tmp_path_factory):
    builtin_dir = tmp_path_factory.mktemp("builtin")
    user_dir = tmp_path_factory.mktemp("user")

    style = Style(user_dir / "test_style.json")
    style.name = "Test Style"
    style.save()

    styles = Styles(builtin_dir, user_dir)
    assert len(styles) == 1
    loaded_style = styles[0]
    assert loaded_style.filename == style.filename
    assert loaded_style.name == "Test Style"
    assert styles.find(style.filename) == loaded_style
    assert styles.find("nonexistent.json") is None
    assert style_is_default(loaded_style)


def test_style_folders(tmp_path_factory):
    builtin_dir = tmp_path_factory.mktemp("builtin")
    user_dir = tmp_path_factory.mktemp("user")

    builtin = Style(builtin_dir / "test_style.json")
    builtin.name = "Built-in Style"
    builtin.save()

    user = Style(user_dir / "test_style.json")
    user.name = "User Style"
    user.save()

    styles = Styles(builtin_dir, user_dir)
    assert len(styles) == 2
    for style in styles:
        if style.filepath == builtin.filepath:
            assert style.name == "Built-in Style"
        elif style.filepath == user.filepath:
            assert style.name == "User Style"
        else:
            assert False

    only_user = styles.filtered(show_builtin=False)
    assert len(only_user) == 1
    assert only_user[0].name == "User Style"


def test_bad_style_file(tmp_path_factory):
    builtin_dir = tmp_path_factory.mktemp("builtin")
    user_dir = tmp_path_factory.mktemp("user")

    path = user_dir / "test_style.json"
    path.write_text("bad json")
    styles = Styles(builtin_dir, user_dir)
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


def test_default_style(tmp_path_factory):
    styles = Styles(tmp_path_factory.mktemp("builtin"), tmp_path_factory.mktemp("user"))
    style = styles.default
    assert style_is_default(style)


def test_sampler_presets(tmp_path_factory):
    dir = tmp_path_factory.mktemp("presets")

    builtin_file = dir / "builtin.json"
    builtin_file.write_text(
        json.dumps(
            {
                "Builtin": {"sampler": "dpmpp_2m", "scheduler": "normal", "steps": 42, "cfg": 7.0},
            }
        )
    )

    user_file = dir / "user.json"
    user_file.write_text(
        json.dumps(
            {
                "User": {"sampler": "user_sampler", "scheduler": "normal", "steps": 13, "cfg": 1.0},
            }
        )
    )

    presets = SamplerPresets(builtin_file, user_file)
    assert len(presets) == 2

    builtin = presets["Builtin"]
    assert builtin == SamplerPreset("dpmpp_2m", "normal", 42, 7.0)

    user = presets["User"]
    assert user == SamplerPreset("user_sampler", "normal", 13, 1.0)

    presets.add_missing("DDIM", 99, 2.3)
    assert len(presets) == 3
    assert presets["DDIM"] == SamplerPreset("ddim", "ddim_uniform", 99, 2.3)


def test_sampler_preset_conversion():
    presets = SamplerPresets()
    for old, new in style.legacy_map.items():
        assert presets[old] == presets[new]
