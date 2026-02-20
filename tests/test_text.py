from ai_diffusion.api import LoraInput
from ai_diffusion.files import File, FileCollection
from ai_diffusion.image import Bounds
from ai_diffusion.jobs import JobParams
from ai_diffusion.text import (
    char16_index_to_str_index,
    char16_len,
    create_img_metadata,
    edit_attention,
    eval_wildcards,
    extract_layers,
    extract_loras,
    merge_prompt,
    select_on_cursor_pos,
    str_index_to_char16_index,
    strip_prompt_comments,
)


def test_char16_len():
    assert char16_len("hello") == 5
    assert char16_len("a") == 1
    assert char16_len("") == 0
    assert char16_len("😀") == 2
    assert char16_len("hello😀") == 7
    assert char16_len("😀😀") == 4
    assert char16_len("a😀b") == 4
    assert char16_len("café") == 4


def test_char16_index_to_str_index():
    text = "a😀b"
    assert char16_index_to_str_index(text, 0) == 0
    assert char16_index_to_str_index(text, 1) == 1
    assert char16_index_to_str_index(text, 3) == 2
    assert char16_index_to_str_index(text, 4) == 3


def test_str_index_to_char16_index():
    text = "a😀b"
    assert str_index_to_char16_index(text, 0) == 0
    assert str_index_to_char16_index(text, 1) == 1
    assert str_index_to_char16_index(text, 2) == 3
    assert str_index_to_char16_index(text, 3) == 4


def test_roundtrip_conversion():
    text = "hello😀world"
    for i in range(len(text)):
        c16_index = str_index_to_char16_index(text, i)
        str_index = char16_index_to_str_index(text, c16_index)
        assert str_index == i


def test_char16_index_to_str_index_cjk():
    text = "你好"
    assert char16_index_to_str_index(text, 0) == 0
    assert char16_index_to_str_index(text, 1) == 1
    assert char16_index_to_str_index(text, 2) == 2


def test_str_index_to_char16_index_cjk():
    text = "你好"
    assert str_index_to_char16_index(text, 0) == 0
    assert str_index_to_char16_index(text, 1) == 1
    assert str_index_to_char16_index(text, 2) == 2


def test_strip_prompt_comment():
    assert strip_prompt_comments("Hello # this is a comment") == "Hello"
    assert strip_prompt_comments("Hello \\# this is no comment") == "Hello # this is no comment"
    assert strip_prompt_comments("Hello # this is a comment \\# and this is not") == "Hello"


def test_strip_prompt_comments_multiline():
    prompt = "Hello # comment\nWorld # another comment"
    expected = "Hello\nWorld"
    assert strip_prompt_comments(prompt) == expected

    prompt = "Line1 # comment\nLine2 \\# not a comment # comment\n# Line3"
    expected = "Line1\nLine2 # not a comment"
    assert strip_prompt_comments(prompt) == expected


def test_merge_prompt():
    assert merge_prompt("a", "b") == "a, b"
    assert merge_prompt("", "b") == "b"
    assert merge_prompt("a", "") == "a"
    assert merge_prompt("", "") == ""
    assert merge_prompt("a", "b {prompt} c") == "b a c"
    assert merge_prompt("", "b {prompt} c") == "b  c"


def test_language_directives():
    assert merge_prompt("a", "b", "zh") == "lang:zh a lang:en , b"
    assert merge_prompt("", "b", "zh") == "b"
    assert merge_prompt("a", "", "zh") == "lang:zh a lang:en "
    assert merge_prompt("", "", "zh") == ""
    assert merge_prompt("a", "b {prompt} c", "zh") == "b lang:zh a lang:en  c"
    assert merge_prompt("", "b {prompt} c", "zh") == "b  c"
    assert (
        merge_prompt("lang:de x lang:en y", "a {prompt} b", "zh")
        == "a lang:zh lang:de x lang:en y lang:en  b"
    )
    assert (
        merge_prompt("x lang:en y", "lang:de a {prompt} lang:en b", "zh")
        == "lang:de a lang:zh x lang:en y lang:en  lang:en b"
    )


def test_extract_loras():
    loras = FileCollection()
    loras.add(File.remote("/path/to/Lora-One.safetensors"))
    loras.add(File.remote("Lora-two.safetensors"))
    loras.add(File.remote("folder\\lora-three.safetensors"))

    assert extract_loras("a ship", loras) == ("a ship", [])
    assert extract_loras("a ship <lora:lora-one>", loras) == (
        "a ship",
        [LoraInput(loras[0].id, 1.0)],
    )
    assert extract_loras("a ship <lora:LoRA-one>", loras) == (
        "a ship",
        [LoraInput(loras[0].id, 1.0)],
    )
    assert extract_loras("a ship <lora:lora-one:0.0>", loras) == (
        "a ship",
        [LoraInput(loras[0].id, 0.0)],
    )
    assert extract_loras("a ship <lora:lora-two:0.5>", loras) == (
        "a ship",
        [LoraInput(loras[1].id, 0.5)],
    )
    assert extract_loras("a ship <lora:lora-two:-1.0>", loras) == (
        "a ship",
        [LoraInput(loras[1].id, -1.0)],
    )
    assert extract_loras("banana <lora:folder/lora-three:0.5>", loras) == (
        "banana",
        [LoraInput(loras[2].id, 0.5)],
    )

    try:
        extract_loras("a ship <lora:lora-three>", loras)
    except Exception as e:
        assert str(e).startswith("LoRA not found")

    try:
        extract_loras("a ship <lora:lora-one:test-invalid-str>", loras)
    except Exception as e:
        assert str(e).startswith("Invalid LoRA strength")


def test_extract_loras_meta():
    loras = FileCollection()
    lora = loras.add(File.remote("zap.safetensors"))
    loras.set_meta(lora, "lora_strength", 0.5)
    loras.set_meta(lora, "lora_triggers", "zippity")

    assert extract_loras("a ship <lora:zap> zap", loras) == (
        "a ship  zap",  # triggers are inserted on auto-complete, not at extraction
        [LoraInput(lora.id, 0.5)],
    )


def test_extract_layers():
    prompt = "A beautiful scenery <layer:Background> and a cat <layer:Foreground>"
    modified_prompt, layers = extract_layers(prompt, replacement="Picture {}", start_index=1)
    assert modified_prompt == "A beautiful scenery Picture 1 and a cat Picture 2"
    assert layers == ["Background", "Foreground"]

    prompt = "<layer:Sky> above the mountains"
    modified_prompt, layers = extract_layers(prompt, replacement="Picture {}", start_index=3)
    assert modified_prompt == "Picture 3 above the mountains"
    assert layers == ["Sky"]

    prompt = "No layers here"
    modified_prompt, layers = extract_layers(prompt, replacement="Picture {}", start_index=1)
    assert modified_prompt == "No layers here"
    assert layers == []

    prompt = "<layer:layer (merged)> and <layer:ba<yer:k>"
    modified_prompt, layers = extract_layers(prompt, replacement="{}")
    assert modified_prompt == "1 and 2"
    assert layers == ["layer (merged)", "ba<yer:k"]


def test_wildcards():
    prompt = "beg {a1(/#|b} mid {1|2|3} end"
    evaluated = eval_wildcards(prompt, seed=42)
    assert evaluated == "beg a1(/# mid 1 end"

    assert eval_wildcards("no {wild|card", seed=42) == "no {wild|card"
    assert eval_wildcards("no {wildcard}", seed=42) == "no {wildcard}"
    assert eval_wildcards("no wild|card}", seed=42) == "no wild|card}"

    assert eval_wildcards("{ bla| piong }", seed=5) == "piong"
    assert eval_wildcards("{ bla| piong }", seed=2) == "bla"

    assert eval_wildcards("start {ab|{12|34}|cd} end", seed=4) == "start 12 end"
    assert eval_wildcards("start {ab|{12|34}|cd} end", seed=3) == "start cd end"


def test_wildcard_distribution():
    prompt = "beg {a|b|c} mid {1|2|3} end"
    results: dict[str, int] = {}
    for seed in range(1000):
        evaluated = eval_wildcards(prompt, seed)
        results[evaluated] = results.get(evaluated, 0) + 1

    assert len(results) == 9
    assert all(count > 50 for count in results.values())
    assert all(count < 150 for count in results.values())


def test_create_img_metadata_basic():
    bounds = Bounds(0, 0, 512, 768)
    metadata = {
        "prompt": "A cat",
        "negative_prompt": "dog",
        "sampler": "Euler - euler_a",
        "steps": 20,
        "guidance": 7.5,
        "checkpoint": "model.ckpt",
        "strength": 0.8,
        "loras": [],
    }
    job_params = JobParams(
        bounds=bounds,
        name="test",
        metadata=metadata,
        seed=12345,
    )

    result = create_img_metadata(job_params)
    assert "A cat" in result
    assert "Negative prompt: dog" in result
    assert (
        "Steps: 20, Sampler: Euler - euler_a, CFG scale: 7.5, Seed: 12345, Size: 512x768, Model hash: unknown, Model: model.ckpt, Denoising strength: 0.8"
        in result
    )


def test_create_img_metadata_loras_dict_and_tuple():
    bounds = Bounds(0, 0, 128, 128)

    metadata = {
        "prompt": "Prompt",
        "negative_prompt": "",
        "sampler": "Euler - euler_a",
        "steps": 20,
        "guidance": 7.0,
        "checkpoint": "loramodel.ckpt",
        "loras": [{"name": "lora1", "weight": 0.7}, ("lora2", 0.5), ["lora3", 0.9]],
    }

    job_params = JobParams(
        bounds=bounds,
        name="test",
        metadata=metadata,
        seed=0,
    )
    result = create_img_metadata(job_params)
    assert "<lora:lora1:0.7>" in result
    assert "<lora:lora2:0.5>" in result
    assert "<lora:lora3:0.9>" in result


def test_create_img_metadata_strength_none_and_one():
    bounds = Bounds(0, 0, 64, 64)

    job_params_none = JobParams(
        bounds=bounds,
        name="test",
        metadata={
            "prompt": "Prompt",
            "negative_prompt": "",
            "sampler": "Euler - euler_a",
            "steps": 10,
            "guidance": 2.0,
            "checkpoint": "model.ckpt",
            "strength": None,
            "loras": [],
        },
        seed=12345,
    )

    job_params_one = JobParams(
        bounds=bounds,
        name="test",
        metadata={
            "prompt": "Prompt",
            "negative_prompt": "",
            "sampler": "Euler - euler_a",
            "steps": 10,
            "guidance": 2.0,
            "checkpoint": "model.ckpt",
            "strength": 1.0,
            "loras": [],
        },
        seed=12345,
    )

    result_none = create_img_metadata(job_params_none)
    result_one = create_img_metadata(job_params_one)
    assert "Denoising strength" not in result_none
    assert "Denoising strength" not in result_one


def test_create_img_metadata_missing_metadata_fields():
    jp = JobParams(
        bounds=Bounds(0, 0, 100, 200),
        name="test",
        metadata={},
        seed=999,
    )

    result = create_img_metadata(jp)
    assert "" in result
    assert "Negative prompt: " in result
    assert "Steps: 0" in result
    assert "Sampler: " in result
    assert "CFG scale: 0.0" in result
    assert "Seed: 999" in result
    assert "Size: 100x200" in result
    assert "Model: Unknown" in result


class TestEditAttention:
    def test_empty_selection(self):
        assert edit_attention("", positive=True) == ""

    def test_adjust_from_1(self):
        assert edit_attention("bar", positive=True) == "(bar:1.1)"

    def test_adjust_to_1(self):
        assert edit_attention("(bar:1.1)", positive=False) == "bar"

    def test_upper_bound(self):
        assert edit_attention("(bar:2.0)", positive=True) == "(bar:2.0)"
        assert edit_attention("(bar:1.95)", positive=True) == "(bar:2.0)"

    def test_lower_bound(self):
        assert edit_attention("(bar:-1.95)", positive=False) == "(bar:-2.0)"
        assert edit_attention("(bar:-2.0)", positive=False) == "(bar:-2.0)"

    def test_single_digit(self):
        assert edit_attention("(bar:0)", positive=True) == "(bar:0.1)"
        assert edit_attention("(bar:.1)", positive=True) == "(bar:0.2)"

    def test_nested(self):
        assert edit_attention("(foo:1.0), bar", positive=True) == "((foo:1.0), bar:1.1)"
        assert (
            edit_attention("(1girl:1.1), foo, (bar:1.3)", positive=True)
            == "((1girl:1.1), foo, (bar:1.3):1.1)"
        )
        assert (
            edit_attention("((foo:1.5), (bar:0.9), baz:1.4)", positive=True)
            == "((foo:1.5), (bar:0.9), baz:1.5)"
        )
        assert edit_attention("(:):1.0)", positive=True) == "(:):1.1)"

    def test_invalid_weight(self):
        assert edit_attention("(foo:bar)", positive=True) == "((foo:bar):1.1)"

    def test_no_weight(self):
        assert edit_attention("(foo)", positive=True) == "((foo):1.1)"

    def test_angle_bracket(self):
        assert edit_attention("<bar:1.0>", positive=True) == "<bar:1.1>"
        assert edit_attention("<foo:bar:1.0>", positive=True) == "<foo:bar:1.1>"
        assert edit_attention("<foo:bar:1.1>", positive=False) == "<foo:bar:1.0>"
        assert edit_attention("<foo:bar:0.0>", positive=False) == "<foo:bar:-0.1>"

    def test_newline(self):
        assert edit_attention("foo\nbar", positive=True) == "(foo\nbar:1.1)"
        assert edit_attention("(foo\nbar:1.1)", positive=True) == "(foo\nbar:1.2)"


class TestSelectOnCursorPos:
    def test_word_selection(self):
        assert select_on_cursor_pos("(foo:1.3), bar, baz", 12) == (11, 14)
        assert select_on_cursor_pos("((foo):1.3), bar, baz", 14) == (13, 16)
        assert select_on_cursor_pos("foo, bar, baz", 5) == (5, 8)
        assert select_on_cursor_pos("foo, bar, baz", 5) == (5, 8)

    def test_range_selection(self):
        assert select_on_cursor_pos("(foo:1.3), bar, baz", 1) == (0, 9)
        assert select_on_cursor_pos("foo, (bar:1.1), baz", 6) == (5, 14)
        assert select_on_cursor_pos("foo, (bar:1.1) <bar:baz:1.0>", 16) == (15, 28)
